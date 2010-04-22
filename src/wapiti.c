/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2010  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the CNRS, nor the names of its contributors may be
 *       used to endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/times.h>

#include <pthread.h>

#define unused(v) ((void)(v))
#define none ((size_t)-1)

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) < (b) ? (b) : (a))

typedef struct tms tms_t;

/*******************************************************************************
 * Error handling and memory managment
 *
 *   Wapiti use a very simple system for error handling: violently fail. Errors
 *   can occurs in two cases, when user feed Wapiti with bad datas or when there
 *   is a problem on the system side. In both cases, there is nothing we can do,
 *   so the best thing is to exit with a meaning full error message.
 *
 *   Memory allocation is one of the possible point of failure and its painfull
 *   to always remeber to check return value of malloc so we provide wrapper
 *   around it and realloc who check and fail in case of error.
 ******************************************************************************/

/* fatal:
 *   This is the main error function, it will print the given message with same
 *   formating than the printf family and exit program with an error. We let the
 *   OS care about freeing ressources.
 */
static void fatal(const char *msg, ...) {
	va_list args;
	fprintf(stderr, "error: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n");
	exit(EXIT_FAILURE);
}

/* pfatal:
 *   This one is very similar to the fatal function but print an additional
 *   system error message depending on the errno. This can be used when a
 *   function who set the errno fail to print more detailed informations. You
 *   must be carefull to not call other functino that might reset it before
 *   calling pfatal.
 */
static void pfatal(const char *msg, ...) {
	const char *err = strerror(errno);
	va_list args;
	fprintf(stderr, "error: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n\t<%s>\n", err);
	exit(EXIT_FAILURE);
}

/* warning:
 *   This one is less violent as it just print a warning on stderr, but doesn't
 *   exit the program. It is intended to inform the user that something strange
 *   have happen and the result might be not what it have expected.
 */
static void warning(const char *msg, ...) {
	va_list args;
	fprintf(stderr, "warning: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n");
}

/* info:
 *   Function used for all progress reports. This is where an eventual verbose
 *   level can be implemented later or redirection to a logfile. For now, it is
 *   just a wrapper for printf to stderr. Note that unlike the previous one,
 *   this function doesn't automatically append a new line character.
 */
static void info(const char *msg, ...) {
	va_list args;
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
}

/* xmalloc:
 *   A simple wrapper around malloc who violently fail if memory cannot be
 *   allocated, so it will never return NULL.
 */
static void *xmalloc(size_t size) {
	void *ptr = malloc(size);
	if (ptr == NULL)
		fatal("out of memory");
	return ptr;
}

/* xrealloc:
 *   As xmalloc, this is a simple wrapper around realloc who fail on memory
 *   error and so never return NULL.
 */
static void *xrealloc(void *ptr, size_t size) {
	void *new = realloc(ptr, size);
	if (new == NULL)
		fatal("out of memory");
	return new;
}

/* xstrdup:
 *   As the previous one, this is a safe version of xstrdup who fail on
 *   allocation error.
 */
static char *xstrdup(const char *str) {
	const int len = strlen(str) + 1;
	char *res = xmalloc(sizeof(char) * len);
	memcpy(res, str, len);
	return res;
}

/******************************************************************************
 *                        Command line parsing
 *
 *  This module handle command line parsing and put all things defined by the
 *  user in a special structure in order to make them accessible to the
 *  remaining of the program.
 *  
 ******************************************************************************/

/* opt_t:
 *   This structure hold all user configurable parameter for Wapiti and is
 *   filled with parameters from command line.
 */
typedef struct opt_s opt_t;
struct opt_s {
	int    nthread;
	bool   sparse;
	double rho1, rho2;
};

/******************************************************************************
 * Multi-threading code
 *
 *   This module handle the thread managment code using POSIX pthreads, on
 *   non-POSIX systems you will have to rewrite this using your systems threads.
 *   all code who depend on threads is located here so this process must not be
 *   too difficult.
 *
 *   This code is also used to launch a single thread but with a controled stack
 *   size, so ensure that the stack size code is well handled by your system
 *   when you port it.
 ******************************************************************************/
typedef void (func_t)(int id, int cnt, void *ud);

typedef struct mth_s mth_t;
struct mth_s {
	int     id;
	int     cnt;
	func_t *f;
	void   *ud;
};

static void *mth_stub(void *ud) {
	mth_t *mth = (mth_t *)ud;
	mth->f(mth->id, mth->cnt, mth->ud);
	return NULL;
}

/* mth_spawn:
 *   This function spawn W threads for calling the 'f' function. It ensure that
 *   there is at least 'stacksz' byte of stack space available in each thread.
 *   The function will get a unique identifier between 0 and W-1 and a user data
 *   from the 'ud' array.
 */
static void mth_spawn(func_t *f, int W, size_t stacksz, void *ud[W]) {
	// We first adjust the requested stack size to be sure it is a round
	// number of system page size as requested by pthread. As there is no
	// portable to get the system page size and all systems, at my
	// knowledge, use power of two lower or equal than 4096, using this
	// value is safe.
	if (stacksz % 4096 != 0)
		stacksz += 4096 - (stacksz % 4096);
	// Next we prepare the pthreads attributes. We check that the default
	// stack size is too low before attempting to raise it as if the system
	// provide a big stack we don't want to insist on getting a smaller one.
	size_t stackdef = 0;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_attr_getstacksize(&attr, &stackdef);
	if (stackdef < stacksz)
		if (pthread_attr_setstacksize(&attr, stacksz) != 0)
			fatal("cannot change thread stack size");
	// We prepare the parameters structures that will be send to the threads
	// with informations for calling the user function.
	mth_t p[W];
	for (int w = 0; w < W; w++) {
		p[w].id  = w;
		p[w].cnt = W;
		p[w].f   = f;
		p[w].ud  = ud[w];
	}
	// We are now ready to spawn the threads and wait for them to finish
	// their jobs. So we just create all the thread and try to join them
	// waiting for there return.
	pthread_t th[W];
	for (int w = 0; w < W; w++)
		if (pthread_create(&th[w], &attr, &mth_stub, &p[w]) != 0)
			fatal("failed to create thread");
	for (int w = 0; w < W; w++)
		if (pthread_join(th[w], NULL) != 0)
			fatal("failed to join thread");
	pthread_attr_destroy(&attr);
}

/******************************************************************************
 * eXtended Vector Maths
 *
 *   These functions are vector operations that you can found in almost all
 *   linear agebra library like BLAS. As compielrs optimize them quite well,
 *   there is just plain C99 version of them. If you have an optimized BLAS
 *   library for your system, fell free to call it here as you will probably
 *   gain a little performance improvment but don't expect to much.
 *
 *   The main thing is that we work on very huge vectors that do not fit in
 *   cache so the bottleneck is in memory access not in the computations
 *   themselves.
 *
 *   There is one exception, the component exponential minus a constant. The
 *   exponential function is quite slow and can take a huge amount of total time
 *   so we provide an SSE2 optimized version which can be up to four time
 *   quicker than the system one, depending on your processor.
 *
 *   As the code is pretty straight forward, there is no need for comments here.
 *
 ******************************************************************************/
static double xvm_norm(const double x[], size_t N) {
	double res = 0.0;
	for (size_t n = 0; n < N; n++)
		res += x[n] * x[n];
	return sqrt(res);
}

static double xvm_unit(double r[], const double x[], size_t N) {
	double sum = 0.0;
	for (size_t n = 0; n < N; n++)
		sum += x[n];
	const double scale = 1.0 / sum;
	for (size_t n = 0; n < N; n++)
		r[n] = x[n] * scale;
	return scale;
}

static double xvm_dot(const double x[], const double y[], size_t N) {
	double res = 0.0;
	for (size_t n = 0; n < N; n++)
		res += x[n] * y[n];
	return res;
}

static void xvm_axpy(double r[], double a, const double x[], const double y[],
                     size_t N) {
	for (size_t n = 0; n < N; n++)
		r[n] = a * x[n] + y[n];
}

/* vms_expma:
 *   Compute the component-wise exponential minus <a>:
 *       r[i] <-- e^x[i] - a
 *
 *   The following comments apply to the SSE2 version of this code:
 *
 *   Computation is done four doubles as a time by doing computation in paralell
 *   on two vectors of two doubles using SSE2 intrisics.  If size is not a
 *   multiple of 4, the remaining elements are computed using the stdlib exp().
 *
 *   The computation is done by first doing a range reduction of the argument of
 *   the type e^x = 2^k * e^f choosing k and f so that f is in [-0.5, 0.5].
 *   Then 2^k can be computed exactly using bit operations to build the double
 *   result and e^f can be efficiently computed with enough precision using a
 *   polynomial approximation.
 *
 *   The polynomial approximation is done with 11th order polynomial computed by
 *   Remez algorithm with the Solya suite, instead of the more classical Pade
 *   polynomial form cause it is better suited to parallel execution. In order
 *   to achieve the same precision, a Pade form seems to require three less
 *   multiplications but need a very costly division, so it will be less
 *   efficient.
 *
 *   The maximum error is less than 1lsb and special cases are correctly
 *   handled:
 *     +inf or +oor  -->   return +inf
 *     -inf or -oor  -->   return  0.0
 *     qNaN or sNaN  -->   return qNaN
 *
 *   This code is copyright 2004-2010 Thomas Lavergne and licenced under the
 *   BSD licence like the remaining of Wapiti.
 */
#ifndef __SSE2__
#define xvm_align
static void xvm_expma(double r[], const double x[], double a, size_t N) {
	for (size_t n = 0; n < N; n++)
		r[n] = exp(x[n]) - a;
}
#else
#include <emmintrin.h>
#define xvm_align __attribute__((aligned(16)))
#define xvm_vconst(v) (_mm_castsi128_pd(_mm_set1_epi64x((v))))
static void xvm_expma(double r[], const double x[], double a, size_t N) {
	assert(r != NULL && ((size_t)r % 16) == 0);
	assert(x != NULL && ((size_t)x % 16) == 0);
	const __m128i vl  = _mm_set1_epi64x(0x3ff0000000000000ULL);
	const __m128d ehi = xvm_vconst(0x4086232bdd7abcd2ULL);
	const __m128d elo = xvm_vconst(0xc086232bdd7abcd2ULL);
	const __m128d l2e = xvm_vconst(0x3ff71547652b82feULL);
	const __m128d hal = xvm_vconst(0x3fe0000000000000ULL);
	const __m128d nan = xvm_vconst(0xfff8000000000000ULL);
	const __m128d inf = xvm_vconst(0x7ff0000000000000ULL);
	const __m128d c1  = xvm_vconst(0x3fe62e4000000000ULL);
	const __m128d c2  = xvm_vconst(0x3eb7f7d1cf79abcaULL);
	const __m128d p0  = xvm_vconst(0x3feffffffffffffeULL);
	const __m128d p1  = xvm_vconst(0x3ff000000000000bULL);
	const __m128d p2  = xvm_vconst(0x3fe0000000000256ULL);
	const __m128d p3  = xvm_vconst(0x3fc5555555553a2aULL);
	const __m128d p4  = xvm_vconst(0x3fa55555554e57d3ULL);
	const __m128d p5  = xvm_vconst(0x3f81111111362f4fULL);
	const __m128d p6  = xvm_vconst(0x3f56c16c25f3bae1ULL);
	const __m128d p7  = xvm_vconst(0x3f2a019fc9310c33ULL);
	const __m128d p8  = xvm_vconst(0x3efa01825f3cb28bULL);
	const __m128d p9  = xvm_vconst(0x3ec71e2bd880fdd8ULL);
	const __m128d p10 = xvm_vconst(0x3e9299068168ac8fULL);
	const __m128d p11 = xvm_vconst(0x3e5ac52350b60b19ULL);
	const __m128d va  = _mm_set1_pd(a);
	size_t n, d = N % 4;
	for (n = 0; n < N - d; n += 4) {
		__m128d mn1, mn2, mi1, mi2;
		__m128d t1,  t2,  d1,  d2;
		__m128d v1,  v2,  w1,  w2;
		__m128i k1,  k2;
		__m128d f1,  f2;
		// Load the next four values
		__m128d x1 = _mm_load_pd(x + n    );
		__m128d x2 = _mm_load_pd(x + n + 2);
		// Check for out of ranges, infinites and NaN
		mn1 = _mm_cmpneq_pd(x1, x1);	mn2 = _mm_cmpneq_pd(x2, x2);
		mi1 = _mm_cmpgt_pd(x1, ehi);	mi2 = _mm_cmpgt_pd(x2, ehi);
		x1  = _mm_max_pd(x1, elo);	x2  = _mm_max_pd(x2, elo);
		// Range reduction: we search k and f such that e^x = 2^k * e^f
		// with f in [-0.5, 0.5]
		t1  = _mm_mul_pd(x1, l2e);	t2  = _mm_mul_pd(x2, l2e);
		t1  = _mm_add_pd(t1, hal);	t2  = _mm_add_pd(t2, hal);
		k1  = _mm_cvttpd_epi32(t1);	k2  = _mm_cvttpd_epi32(t2);
		d1  = _mm_cvtepi32_pd(k1);	d2  = _mm_cvtepi32_pd(k2);
		t1  = _mm_mul_pd(d1, c1);	t2  = _mm_mul_pd(d2, c1);
		f1  = _mm_sub_pd(x1, t1);	f2  = _mm_sub_pd(x2, t2);
		t1  = _mm_mul_pd(d1, c2);	t2  = _mm_mul_pd(d2, c2);
		f1  = _mm_sub_pd(f1, t1);	f2  = _mm_sub_pd(f2, t2);
		// Evaluation of e^f using a 11th order polynom in Horner form
		v1  = _mm_mul_pd(f1, p11);	v2  = _mm_mul_pd(f2, p11);
		v1  = _mm_add_pd(v1, p10);	v2  = _mm_add_pd(v2, p10);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p9);	v2  = _mm_add_pd(v2, p9);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p8);	v2  = _mm_add_pd(v2, p8);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p7);	v2  = _mm_add_pd(v2, p7);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p6);	v2  = _mm_add_pd(v2, p6);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p5);	v2  = _mm_add_pd(v2, p5);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p4);	v2  = _mm_add_pd(v2, p4);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p3);	v2  = _mm_add_pd(v2, p3);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p2);	v2  = _mm_add_pd(v2, p2);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p1);	v2  = _mm_add_pd(v2, p1);
		v1  = _mm_mul_pd(v1, f1);	v2  = _mm_mul_pd(v2, f2);
		v1  = _mm_add_pd(v1, p0);	v2  = _mm_add_pd(v2, p0);
		// Evaluation of 2^k using bitops to achieve exact computation
		k1  = _mm_slli_epi32(k1, 20);	k2  = _mm_slli_epi32(k2, 20);
		k1  = _mm_shuffle_epi32(k1, 0x72);
		k2  = _mm_shuffle_epi32(k2, 0x72);
		k1  = _mm_add_epi32(k1, vl);	k2  = _mm_add_epi32(k2, vl);
		w1  = _mm_castsi128_pd(k1);	w2  = _mm_castsi128_pd(k2);
		// Return to full range to substract <a>
	        v1  = _mm_mul_pd(v1, w1);	v2  = _mm_mul_pd(v2, w2);
		v1  = _mm_sub_pd(v1, va);	v2  = _mm_sub_pd(v2, va);
		// Finally apply infinite and NaN where needed
		v1  = _mm_or_pd(_mm_and_pd(mi1, inf), _mm_andnot_pd(mi1, v1));
		v2  = _mm_or_pd(_mm_and_pd(mi2, inf), _mm_andnot_pd(mi2, v2));
		v1  = _mm_or_pd(_mm_and_pd(mn1, nan), _mm_andnot_pd(mn1, v1));
		v2  = _mm_or_pd(_mm_and_pd(mn2, nan), _mm_andnot_pd(mn2, v2));
		// Store the results
		_mm_store_pd(r + n,     v1);
		_mm_store_pd(r + n + 2, v2);
	}
	// Handle the lasts elements
	for ( ; n < N; n++)
		r[n] = exp(x[n]) - a;
}
#endif

/******************************************************************************
 * Quark database
 *
 *   Implement quark database: mapping between strings and identifiers in both
 *   directions.
 *
 *   The mapping between strings to identifiers is done with splay tree, a kind
 *   of self balancing search tree. They are simpler to implement than a lot of
 *   other data-structures and very cache friendly. For most use they are at
 *   least as efficient as red-black tree or B-tree. See [1] for more
 *   informations. The strings are interned directly in the node for easier
 *   memory managment.
 *
 *   The identifier to string mapping is done with a simple vector with pointer
 *   to the key interned in the tree node, so pointer returned are always
 *   constant.
 *
 *   [1] Sleator, Daniel D. and Tarjan, Robert E. ; Self-adjusting binary search
 *   trees, Journal of the ACM 32 (3): pp. 652--686, 1985. DOI:10.1145/3828.3835
 *
 *   This code is copyright 2002-2010 Thomas Lavergne and licenced under the BSD
 *   Licence like the remaining of Wapiti.
 *
 ******************************************************************************/

/* qrk_node_t:
 *   Node of the splay tree whoe hold a (key, value) pair. The left and right
 *   childs are stored in an array so code for each side can be factorized.
 */
typedef struct qrk_node_s qrk_node_t;
struct qrk_node_s {
	qrk_node_t *child[2]; // Left and right childs of the node
	size_t      value;    // Value stored in the node
	char        key[];    // The key directly stored in the node
};

/* qrk_t:
 *   The quark database with his <tree> and <vector>. The dabase hold <count>
 *   (key, value) pairs, but the vector is of size <size>, it will grow as
 *   needed. If <lock> is true, new key will not be added to the quark and
 *   none will be returned as an identifier.
 */
typedef struct qrk_s qrk_t;
struct qrk_s {
	qrk_node_t  *tree;    //       The tree for direct mapping
	char       **vector;  // [N']  The array for the reverse mapping
	size_t       count;   //  N    The number of items in the database
	size_t       size;    //  N'   The real size of <vector>
	bool         lock;    //       Are new keys added to the dictionnary ?
};

/* qrk_newnode:
 *   Create a new qrk_node_t object with given key, value and no childs. The key
 *   is interned in the node so no reference is kept to the given string. The
 *   object must be freed with qrk_freenode when not needed anymore.
 */
static qrk_node_t *qrk_newnode(const char *key, size_t value) {
	const int len = strlen(key) + 1;
	qrk_node_t *nd = xmalloc(sizeof(qrk_node_t) + len);
	memcpy(nd->key, key, len);
	nd->value = value;
	nd->child[0] = NULL;
	nd->child[1] = NULL;
	return nd;
}

/* qrk_freenode:
 *   Free a qrk_node_t object and all his childs recursively.
 */
static void qrk_freenode(qrk_node_t *nd) {
	if (nd->child[0] != NULL)
		qrk_freenode(nd->child[0]);
	if (nd->child[1] != NULL)
		qrk_freenode(nd->child[1]);
	free(nd);
}

/* qrk_new:
 *   Create a new qrk_t object ready for doing mappings. This object must be
 *   freed with qrk_free when not used anymore.
 */
static qrk_t *qrk_new(void) {
	qrk_t *qrk = xmalloc(sizeof(qrk_t));
	qrk->tree   = NULL;
	qrk->vector = NULL;
	qrk->count  = 0;
	qrk->size   = 0;
	qrk->lock   = false;
	return qrk;
}

/* qrk_free:
 *   Free all memory used by the given quark. All strings returned by qrk_unmap
 *   become invalid and must not be used anymore.
 */
static void qrk_free(qrk_t *qrk) {
	if (qrk->count != 0) {
		qrk_freenode(qrk->tree);
		free(qrk->vector);
	} else if (qrk->vector != NULL) {
		free(qrk->vector);
	}
	free(qrk);
}

/* qrk_splay:
 *   Do a splay operation on the tree part of the quark with the given key.
 *   Return -1 if the key is in the quark, so if it has been move at the top,
 *   else return the side wether the key should go.
 */
static int qrk_splay(qrk_t *qrk, const char *key) {
	qrk_node_t  nil = {{NULL, NULL}, 0};
	qrk_node_t *root[2] = {&nil, &nil};
	qrk_node_t *nd = qrk->tree;
	int side;
	while (true) {
		side = strcmp(key, nd->key);
		side = (side == 0) ? -1 : (side < 0) ? 0 : 1;
		if (side == -1 || nd->child[side] == NULL)
			break;
		const int tst = (side == 0)
			? strcmp(key, nd->child[side]->key) < 0
			: strcmp(key, nd->child[side]->key) > 0;
		if (tst) {
			qrk_node_t *tmp = nd->child[side];
			nd->child[side] = tmp->child[1 - side];
			tmp->child[1 - side] = nd;
			nd = tmp;
			if (nd->child[side] == NULL)
				break;
		}
		root[1 - side]->child[side] = nd;
		root[1 - side] = nd;
		nd = nd->child[side];
	}
	root[0]->child[1] = nd->child[0];
	root[1]->child[0] = nd->child[1];
	nd->child[0] = nil.child[1];
	nd->child[1] = nil.child[0];
	qrk->tree = nd;
	return side;
}

/* qrk_id2str:
 *   Return the key associated with the given identifier. The key must not be
 *   modified nor freed by the caller and remain valid for the lifetime of the
 *   quark object.
 *   Raise a fatal error if the indentifier is invalid.
 */
static const char *qrk_id2str(const qrk_t *qrk, size_t id) {
	if (id >= qrk->count)
		fatal("invalid identifier");
	return qrk->vector[id];
}

/* qrk_str2id:
 *   Return the identifier corresponding to the given key in the quark object.
 *   If the key is not already present, it is inserted unless the quark is
 *   locked, in which case none is returned.
 */
static size_t qrk_str2id(qrk_t *qrk, const char *key) {
	// if tree is empty, directly add a root
	if (qrk->count == 0) {
		if (qrk->size == 0) {
			const size_t size = 128;
			qrk->vector = xmalloc(sizeof(char *) * size);
			qrk->size = size;
		}
		qrk_node_t *nd = qrk_newnode(key, 0);
		qrk->tree = nd;
		qrk->count = 1;
		qrk->vector[0] = nd->key;
		return 0;
	}
	// else if key is already there, return his value
	int side = qrk_splay(qrk, key);
	if (side == -1)
		return qrk->tree->value;
	if (qrk->lock == true)
		return none;
	// else, add the key to the quark
	if (qrk->count == qrk->size) {
		qrk->size *= 1.4;
		const size_t size = sizeof(char **) * qrk->size;
		qrk->vector = xrealloc(qrk->vector, size);
	}
	size_t id = qrk->count;
	qrk_node_t *nd = qrk_newnode(key, id);
	nd->child[    side] = qrk->tree->child[side];
	nd->child[1 - side] = qrk->tree;
	qrk->tree->child[side] = NULL;
	qrk->tree = nd;
	qrk->vector[id] = nd->key;
	qrk->count++;
	return id;
}

/*******************************************************************************
 * Sequences and Dataset objects
 *
 *   Sequences represent the input data feeded by the user in Wapiti either for
 *   training or labelling. The internal form used here is very different from
 *   the data read from files and the convertion process is done in three steps
 *   illustrated here:
 *         +------+     +-------+     +-------+     +-------+
 *         | FILE | --> | raw_t | --> | tok_t | --> | seq_t |
 *         +------+     +-------+     +-------+     +-------+
 *   First the sequence is read as a set of lines from the input file, this
 *   give a raw_t object. Next this set of lines is split in tokens and
 *   eventually the last one is separated as it will become a label, this result
 *   in a tok_t object.
 *   The last step consist in applying all the patterns givens by the user to
 *   extract from these tokens the observations made on the sequence in order to
 *   build the seq_t object which can be used by the trainer and tagger.
 *
 *   A dataset object is just a container for a list of sequences in internal
 *   form used to store either training or development set.
 *
 *   All the convertion process is driven by the reader object and, as it is
 *   responsible for creating the objects with a quite special allocation
 *   scheme, we just have to implement function for freeing these objects here.
 ******************************************************************************/

/* raw_t:
 *   Data-structure representing a raw sequence as a set of lines read from the
 *   input file. This is the result of the first step of the interning process.
 *   We keep this form separate from the tokenized one as we want to be able to
 *   output the sequence as it was read in the labelling mode.
 *
 *   This represent a sequence of lengths <len> and for each position 't' you
 *   find the corresponding line at <lines>[t].
 *
 *   The <lines> array is allocated with data structure, and the different lines
 *   are allocated separatly.
 */
typedef struct raw_s {
	int   len;      //   T     Sequence length
	char *lines[];  //  [T]    Raw lines directly from file
} raw_t;

/* tok_t:
 *   Data-structure representing a tokenized sequence. This is the result of the
 *   second step of the interning process after the raw sequence have been split
 *   in tokens and eventual labels separated from the observations.
 *
 *   For each position 't' in the sequence of length <len>, you find at <lbl>[t]
 *   the eventual label provided in input file, and at <toks>[t] a list of
 *   string tokens of length <cnts>[t].
 *
 *   Memory allocation here is a bit special as the first token at each position
 *   point to a memory block who hold a copy of the raw line. Each other tokens
 *   and the label are pointer in this block. This reduce memory fragmentation.
 */
typedef struct tok_s {
	int    len;     //   T     Sequence length
	char **lbl;     //  [T]    List of labels strings
	int   *cnts;    //  [T]    Length of tokens lists
	char **toks[];  //  [T][]  Tokens lists
} tok_t;

/* seq_t:
 *   Data-structure representing a sequence of length <len> in the internal form
 *   used by the trainers and the tagger. For each position 't' in the sequence
 *   (0 <= t < <len>) there is some observations made on the data and an
 *   eventual label if provided in the input file.
 *
 *   There is two kind of features: unigrams and bigrams one, build by combining
 *   one observation and one or two labels. At position 't', the unigrams
 *   features are build using the list of observations from <uobs>[t] which
 *   contains <ucnt>[t] items, and the observation at <lbl>[t]. The bigrams
 *   features are obtained in the same way using <bobs> and <bcnt>, and have to
 *   be combined also with <lbl>[t-1].
 *
 *   If the sequence is read from a file without label, as it is the case in
 *   labelling mode, the <lbl> field will be NULL and so, the sequence cannot be
 *   used for training.
 *
 *   The raw field is private and used internaly for efficient memory
 *   allocation. This allow to allocate <lbl>, <*cnt>, and all the list in
 *   <*obs> with the datastructure itself.
 */
typedef struct pos_s pos_t;
typedef struct seq_s seq_t;
struct seq_s {
	int     len;
	size_t *raw;
	struct pos_s {
		size_t  lbl;
		size_t  ucnt,  bcnt;
		size_t *uobs, *bobs;
	} pos[];
};

/* dat_t:
 *   Data-structure representing a full dataset: a collection of sequences ready
 *   to be used for training or to be labelled. It keep tracks of the maximum
 *   sequence length as the trainer need this for memory allocation. The dataset
 *   contains <nseq> sequence stored in <seq>. These sequences are labeled only
 *   if <lbl> is true.
 */
typedef struct dat_s {
	bool    lbl;   //         True iff sequences are labelled
	int     mlen;  //         Length of the longest sequence in the set
	int     nseq;  //   S     Number of sequences in the set
	seq_t **seq;   //  [S]    List of sequences
} dat_t;

/******************************************************************************
 * A simple regular expression matcher
 *
 *   This module implement a simple regular expression matcher, it implement
 *   just a subset of the classical regexp simple to implement but sufficient
 *   for most usages and avoid to add a dependency to a full regexp library.
 *
 *   The recognized subset is quite simple. First for matching characters :
 *       .  -> match any characters
 *       \x -> match a character class (in uppercase, match the complement)
 *               \d : digit       \a : alpha      \w : alpha + digit
 *               \l : lowercase   \u : uppercase  \p : punctuation
 *             or escape a character
 *       x  -> any other character match itself
 *   And the constructs :
 *       ^  -> at the begining of the regexp, anchor it at start of string
 *       $  -> at the end of regexp, anchor it at end of string
 *       *  -> match any number of repetition of the previous character
 *       ?  -> optionally match the previous character
 *
 *   This subset is implemented quite efficiently using recursion. All recursive
 *   calls are tail-call so they should be optimized by the compiler. As we do
 *   direct interpretation, we have to backtrack so performance can be very poor
 *   on specialy designed regexp. This is not a problem as the regexp as well as
 *   the string is expected to be very simple here. If this is not the case, you
 *   better have to prepare your data better.
 ******************************************************************************/

/* rex_matchit:
 *   Match a single caracter at the start fo the string. The character might be
 *   a plain char, a dot or char class.
 */
static bool rex_matchit(const char *ch, const char *str) {
	if (str[0] == '\0')
		return false;
	if (ch[0] == '.')
		return true;
	if (ch[0] == '\\') {
		switch (ch[1]) {
			case 'a': return  isalpha(str[0]);
			case 'd': return  isdigit(str[0]);
			case 'l': return  islower(str[0]);
			case 'p': return  ispunct(str[0]);
			case 'u': return  isupper(str[0]);
			case 'w': return  isalnum(str[0]);
			case 'A': return !isalpha(str[0]);
			case 'D': return !isdigit(str[0]);
			case 'L': return !islower(str[0]);
			case 'P': return !ispunct(str[0]);
			case 'U': return !isupper(str[0]);
			case 'W': return !isalnum(str[0]);
		}
		return ch[1] == str[0];
	}
	return ch[0] == str[0];
}

/* rex_matchme:
 *   Match a regular expresion at the start of the string. If a match is found,
 *   is length is returned in len. The mathing is done through tail-recursion
 *   for good performances.
 */
static bool rex_matchme(const char *re, const char *str, int *len) {
	// Special check for end of regexp
	if (re[0] == '\0')
		return true;
	if (re[0] == '$' && re[1] == '\0')
		return (str[0] == '\0');
	// Get first char of regexp
	const char *ch  = re;
	const char *nxt = re + 1 + (ch[0] == '\\');
	// Special check for the following construct "x**" where the first star
	// is consumed normally but lead the second (which is wrong) to be
	// interpreted as a char to mach as if it was escaped (and same for the
	// optional construct)
	if (*ch == '*' || *ch == '?')
		fatal("unescaped * or ? in regexp: %s", re);
	// Handle star repetition
	if (nxt[0] == '*') {
		nxt++;
		do {
			const int save = *len;
			if (rex_matchme(nxt, str, len))
				return true;
			*len = save + 1;
		} while (rex_matchit(ch, str++));
		return false;
	}
	// Handle optional
	if (nxt[0] == '?') {
		nxt++;
		if (rex_matchit(ch, str)) {
			(*len)++;
			if (rex_matchme(nxt, str + 1, len))
				return true;
			(*len)--;
		}
		return rex_matchme(nxt, str, len);
	}
	// Classical char matching
	(*len)++;
	if (rex_matchit(ch, str))
		return rex_matchme(nxt, str + 1, len);
	return false;
}

/* rex_match:
 *   Match a regular expresion in the given string. If a match is found, the
 *   position of the start of the match is returned and is len is returned in
 *   len, else -1 is returned.
 */
static int rex_match(const char *re, const char *str, int *len) {
	// Special case for anchor at start
	if (*re == '^') {
		*len = 0;
		if (rex_matchme(re + 1, str, len))
			return 0;
		return -1;
	}
	// And general case for any position
	int pos = 0;
	do {
		*len = 0;
		if (rex_matchme(re, str + pos, len))
			return pos;
	} while (str[pos++] != '\0');
	// Matching failed
	return -1;
}

/*******************************************************************************
 * Pattern handling
 *
 *   Patterns are the heart the data input process, they provide a way to tell
 *   Wapiti how the interesting information can be extracted from the input
 *   data. A pattern is simply a string who embed special commands about tokens
 *   to extract from the input sequence. They are compiled to a special form
 *   used during data loading.
 *   For training, each position of a sequence hold a list of observation made
 *   at this position, pattern give a way to specify these observations.
 *
 *   During sequence loading, all patterns are applied at each position to
 *   produce a list of string representing the observations which will be in
 *   turn transformed to numerical identifiers. This module take care of
 *   building the string representation.
 *
 *   As said, a patern is a string with specific commands in the forms %c[...]
 *   where 'c' is the command with arguments between the bracket. All commands
 *   take at least to numerical arguments which define a token in the input
 *   sequence. The first one is an offset from the current position and the
 *   second one is a column number. With these two parameters, we get a string
 *   in the input sequence on which we apply the command.
 *
 *   All command are specified with a character and result in a string which
 *   will replace the command in the pattern string. If the command character is
 *   lower case, the result is copied verbatim, if it is uppercase, the result
 *   is copied with casing removed. The following commands are available:
 *     'x' -- result is the token itself
 *     't' -- test if a regular expression match the token. Result will be
 *            either "true" or "false"
 *     'm' -- match a regular expression on the token. Result is the first
 *            substring matched.
 ******************************************************************************/

typedef struct pat_s pat_t;
typedef struct pat_item_s pat_item_t;
struct pat_s {
	char *src;
	int   ntoks;
	int   nitems;
	struct pat_item_s {
		char  type;
		bool  caps;
		char *value;
		int   offset;
		int   column;
	} items[];
};

/* pat_comp:
 *   Compile the pattern to a form more suitable to easily apply it on tokens
 *   list during data reading. The given pattern string is interned in the
 *   compiled pattern and will be freed with it, so you don't have to take care
 *   of it and must not modify it after the compilation.
 */
static pat_t *pat_comp(char *p) {
	pat_t *pat = NULL;
	// Allocate memory for the compiled pattern, the allocation is based
	// on an over-estimation of the number of required item. As compiled
	// pattern take a neglectible amount of memory, this waste is not
	// important.
	int mitems = 0;
	for (int pos = 0; p[pos] != '\0'; pos++)
		if (p[pos] == '%')
			mitems++;
	mitems = mitems * 2 + 1;
	pat = xmalloc(sizeof(pat_t) + sizeof(pat->items[0]) * mitems);
	pat->src = p;
	// Next, we go through the pattern compiling the items as they are
	// found. Commands are parsed and put in a corresponding item, and
	// segment of char not in a command are put in a 's' item.
	int nitems = 0;
	int ntoks = 0;
	int pos = 0;
	while (p[pos] != '\0') {
		pat_item_t *item = &(pat->items[nitems++]);
		item->value = NULL;
		if (p[pos] == '%') {
			// This is a command, so first parse its type and check
			// its a valid one. Next prepare the item.
			const char type = tolower(p[pos + 1]);
			if (type != 'x' && type != 't' && type != 'm')
				fatal("unknown command type: '%c'", type);
			item->type = type;
			item->caps = (p[pos + 1] != type);
			pos += 2;
			// Next we parse the offset and column and store them in
			// the item.
			int off, col, nch;
			if (sscanf(p + pos, "[%d,%d%n", &off, &col, &nch) != 2)
				fatal("invalid pattern: %s", p);
			if (col < 0)
				fatal("invalid column number: %d", col);
			item->offset = off;
			item->column = col;
			ntoks = max(ntoks, col);
			pos += nch;
			// And parse the end of the argument list, for 'x' there
			// is nothing to read but for 't' and 'm' we have to get
			// read the regexp.
			if (type == 't' || type == 'm') {
				if (p[pos] != ',' && p[pos + 1] != '"')
					fatal("missing arg in pattern: %s", p);
				const int start = (pos += 2);
				while (p[pos] != '\0') {
					if (p[pos] == '"')
						break;
					if (p[pos] == '\\' && p[pos+1] != '\0')
						pos++;
					pos++;
				}
				if (p[pos] != '"')
					fatal("unended argument: %s", p);
				const int len = pos - start;
				item->value = xmalloc(sizeof(char) * (len + 1));
				memcpy(item->value, p + start, len);
				item->value[len] = '\0';
				pos++;
			}
			// Just check the end of the arg list and loop.
			if (p[pos] != ']')
				fatal("missing end of pattern: %s", p);
			pos++;
		} else {
			// No command here, so build an 's' item with the chars
			// until end of pattern or next command and put it in
			// the list.
			const int start = pos;
			while (p[pos] != '\0' && p[pos] != '%')
				pos++;
			const int len = pos - start;
			item->type  = 's';
			item->caps  = false;
			item->value = xmalloc(sizeof(char) * (len + 1));
			memcpy(item->value, p + start, len);
			item->value[len] = '\0';
		}
	}
	pat->ntoks = ntoks;
	pat->nitems = nitems;
	return pat;
}

/* pat_exec:
 *   Execute a compiled pattern at position 'at' in the given tokens sequences
 *   in order to produce an observation string. The string is returned as a
 *   newly allocated memory block and the caller is responsible to free it when
 *   not needed anymore.
 */
static char *pat_exec(pat_t *pat, tok_t *tok, int at) {
	static char *bval[] = {"_x-1", "_x-2", "_x-3", "_x-4", "_x-#"};
	static char *eval[] = {"_x+1", "_x+2", "_x+3", "_x+4", "_x+#"};
	const int T = tok->len;
	// Prepare the buffer who will hold the result
	int size = 16, pos = 0;
	char *buffer = xmalloc(sizeof(char) * size);
	// And loop over the compiled items
	for (int it = 0; it < pat->nitems; it++) {
		const pat_item_t *item = &(pat->items[it]);
		char *value = NULL;
		int len = 0;
		// First, if needed, we retrieve the token at the referenced
		// position in the sequence. We store it in value and let the
		// command handler do what it need with it.
		if (item->type != 's') {
			int pos = at + item->offset;
			int col = item->column;
			if (pos < 0)
				value = bval[min(-pos - 1, 4)];
			else if (pos >= T)
				value = eval[min( pos - T, 4)];
			else if (col >= tok->cnts[pos])
				fatal("missing tokens, cannot apply pattern");
			else
				value = tok->toks[pos][col];
		}
		// Next, we handle the command, 's' and 'x' are very simple but
		// 't' and 'm' require us to call the regexp matcher.
		if (item->type == 's') {
			value = item->value;
			len = strlen(value);
		} else if (item->type == 'x') {
			len = strlen(value);
		} else if (item->type == 't') {
			if (rex_match(item->value, value, &len) == -1)
				value = "false";
			else
				value = "true";
			len = strlen(value);
		} else if (item->type == 'm') {
			int pos = rex_match(item->value, value, &len);
			if (pos == -1)
				len = 0;
			value += pos;
		}
		// And we add it to the buffer, growing it if needed. If the
		// user requested it, we also remove caps from the string.
		if (pos + len >= size - 1) {
			while (pos + len >= size - 1)
				size = size * 1.4;
			buffer = xrealloc(buffer, sizeof(char) * size);
		}
		memcpy(buffer + pos, value, len);
		if (item->caps)
			for (int i = pos; i < pos + len; i++)
				buffer[i] = tolower(buffer[i]);
		pos += len;
	}
	// Adjust the result and return it.
	buffer[pos++] = '\0';
	buffer = xrealloc(buffer, sizeof(char) * pos);
	return buffer;
}

/* pat_free:
 *   Free all memory used by a compiled pattern object. Note that this will free
 *   the pointer to the source string given to pat_comp so you must be sure to
 *   not use this pointer again.
 */
static void pat_free(pat_t *pat) {
	for (int it = 0; it < pat->nitems; it++)
		free(pat->items[it].value);
	free(pat->src);
	free(pat);
}

/*******************************************************************************
 * Datafile reader
 *
 *   And now come the data file reader which use the previous module to parse
 *   the input data in order to produce seq_t objects representing interned
 *   sequences.
 *
 *   This is where the sequence will go through the tree steps to build seq_t
 *   objects used internally. There is two way do do this. First the simpler is
 *   to use the rdr_readseq function which directly read a sequence from a file
 *   and convert it to a seq_t object transparently. This is how the training
 *   and development data are loaded.
 *   The second way consist of read a raw sequence with rdr_readraw and next
 *   converting it to a seq_t object with rdr_raw2seq. This allow the caller to
 *   keep the raw sequence and is used by the tagger to produce a clean output.
 *
 *   There is no public interface to the tok_t object as it is intended only for
 *   internal use in the reader as an intermediate step to apply patterns.
 ******************************************************************************/

/* rdr_t:
 *   The reader object who hold all informations needed to parse the input file:
 *   the patterns and quark for labels and observations. We keep separate count
 *   for unigrams and bigrams pattern for simpler allocation of sequences. We
 *   also store the expected number of column in the input data to check that
 *   pattern are appliables.
 */
typedef struct rdr_s rdr_t;
struct rdr_s {
	int     npats;      //  P   Total number of patterns
	int     nuni, nbi;  //      Number of unigram and bigram patterns
	int     ntoks;      //      Expected number of tokens in input
	pat_t **pats;       // [P]  List of precompiled patterns
	qrk_t  *lbl;        //      Labels database
	qrk_t  *obs;        //      Observation database
};

/* rdr_new:
 *   Create a new empty reader object. You mut load patterns in it or a
 *   previously saved reader if you want to use it for reading sequences.
 */
static rdr_t *rdr_new(void) {
	rdr_t *rdr = xmalloc(sizeof(rdr_t));
	rdr->npats = rdr->nuni = rdr->nbi = 0;
	rdr->ntoks = 0;
	rdr->pats = NULL;
	rdr->lbl = qrk_new();
	rdr->obs = qrk_new();
	return rdr;
}

/* rdr_free:
 *   Free all memory used by a reader object including the quark database, so
 *   any string returned by them must not be used after this call.
 */
static void rdr_free(rdr_t *rdr) {
	for (int i = 0; i < rdr->npats; i++)
		pat_free(rdr->pats[i]);
	free(rdr->pats);
	qrk_free(rdr->lbl);
	qrk_free(rdr->obs);
	free(rdr);
}

/* rdr_freeraw:
 *   Free all memory used by a raw_t object.
 */
static void rdr_freeraw(raw_t *raw) {
	for (int t = 0; t < raw->len; t++)
		free(raw->lines[t]);
	free(raw);
}

/* rdr_freeseq:
 *   Free all memory used by a seq_t object.
 */
static void rdr_freeseq(seq_t *seq) {
	free(seq->raw);
	free(seq);
}

/* rdr_freedat:
 *   Free all memory used by a dat_t object.
 */
static void rdr_freedat(dat_t *dat) {
	for (int i = 0; i < dat->nseq; i++)
		free(dat->seq[i]);
	free(dat->seq);
	free(dat);
}

/* rdr_readline:
 *   Read an input line from <file>. The line can be of any size limited only by
 *   available memory, a buffer large enough is allocated and returned. The
 *   caller is responsible to free it. On end-of-file, NULL is returned.
 */
static char *rdr_readline(FILE *file) {
	if (feof(file))
		return NULL;
	// Initialize the buffer
	int len = 0, size = 16;
	char *buffer = xmalloc(size);
	// We read the line chunk by chunk until end of line, file or error
	while (!feof(file)) {
		if (fgets(buffer + len, size - len, file) == NULL) {
			// On NULL return there is two possible cases, either an
			// error or the end of file
			if (ferror(file))
				pfatal("cannot read from file");
			// On end of file, we must check if we have already read
			// some data or not
			if (len == 0) {
				free(buffer);
				return NULL;
			}
			break;
		}
		// Check for end of line, if this is not the case enlarge the
		// buffer and go read more data
		len += strlen(buffer + len);
		if (len == size - 1 && buffer[len - 1] != '\n') {
			size = size * 1.4;
			buffer = xrealloc(buffer, size);
			continue;
		}
		break;
	}
	// At this point empty line should have already catched so we just
	// remove the end of line if present and resize the buffer to fit the
	// data
	if (buffer[len - 1] == '\n')
		buffer[--len] = '\0';
	return xrealloc(buffer, len + 1);
}

/* rdr_loadpat:
 *   Load and compile patterns from given file and store them in the reader. As
 *   we compile patterns, syntax errors in them will be raised at this time.
 */
static void rdr_loadpat(rdr_t *rdr, FILE *file) {
	while (!feof(file)) {
		// Read raw input line
		char *line = rdr_readline(file);
		if (line == NULL)
			break;
		// Remove comments and trailing spaces
		int end = strcspn(line, "#");
		while (end != 0 && isspace(line[end - 1]))
			end--;
		if (end == 0) {
			free(line);
			continue;
		}
		line[end] = '\0';
		line[0] = tolower(line[0]);
		// Compile pattern and add it to the list
		pat_t *pat = pat_comp(line);
		rdr->npats++;
		switch (line[0]) {
			case 'u': rdr->nuni++; break;
			case 'b': rdr->nbi++; break;
			case '*': rdr->nuni++;
			          rdr->nbi++; break;
			default:
				fatal("unknown pattern type '%c'", line[0]);
		}
		rdr->pats = xrealloc(rdr->pats, sizeof(char *) * rdr->npats);
		rdr->pats[rdr->npats - 1] = pat;
		rdr->ntoks = max(rdr->ntoks, pat->ntoks);
	}
}

/* rdr_readraw:
 *   Read a raw sequence from given file: a set of lines terminated by end of
 *   file or by an empty line. Return NULL if file end was reached before any
 *   sequence was read.
 *   The reader object is not used in this function but is specified as a
 *   parameter to be more coherent.
 */
static raw_t *rdr_readraw(rdr_t *rdr, FILE *file) {
	unused(rdr);
	if (feof(file))
		return NULL;
	// Prepare the raw sequence object
	int size = 32, cnt = 0;
	raw_t *raw = xmalloc(sizeof(raw_t) + sizeof(char *) * size);
	// And read the next sequence in the file, this will skip any blank line
	// before reading the sequence stoping at end of file or on a new blank
	// line.
	while (!feof(file)) {
		char *line = rdr_readline(file);
		if (line == NULL)
			break;
		// Check for empty line marking the end of the current sequence
		int len = strlen(line);
		while (len != 0 && isspace(line[len - 1]))
			len--;
		if (len == 0) {
			free(line);
			// Special case when no line was already read, we try
			// again. This allow multiple blank lines beetwen
			// sequences.
			if (cnt == 0)
				continue;
			break;
		}
		// Next, grow the buffer if needed and add the new line in it
		if (size == cnt) {
			size *= 1.4;
			raw = xrealloc(raw, sizeof(raw_t)
			                + sizeof(char *) * size);
		}
		raw->lines[cnt++] = line;
	}
	// If no lines was read, we just free allocated memory and return NULL
	// to signal the end of file to the caller. Else, we adjust the object
	// size and return it.
	if (cnt == 0) {
		free(raw);
		return NULL;
	}
	raw = xrealloc(raw, sizeof(raw_t) + sizeof(char *) * cnt);
	raw->len = cnt;
	return raw;
}

/* rdr_raw2seq:
 *   Convert a raw sequence to a seq_t object suitable for training or
 *   labelling. If lbl is true, the last column is assumed to be a label and
 *   interned also.
 */
static seq_t *rdr_raw2seq(rdr_t *rdr, const raw_t *raw, bool lbl) {
	const int T = raw->len;
	// Allocate the tok_t object, the label array is allocated only if they
	// are requested by the user.
	tok_t *tok = xmalloc(sizeof(tok_t) + T * sizeof(char **));
	tok->cnts = xmalloc(sizeof(size_t) * T);
	tok->lbl = NULL;
	if (lbl == true)
		tok->lbl = xmalloc(sizeof(char *) * T);
	// We now take the raw sequence line by line and split them in list of
	// tokens. To reduce memory fragmentation, the raw line is copied and
	// his reference is kept by the first tokens, next tokens are pointer to
	// this copy.
	for (int t = 0; t < T; t++) {
		// Get a copy of the raw line skiping leading space characters
		const char *src = raw->lines[t];
		while (isspace(*src))
			src++;
		char *line = xstrdup(src);
		// Split it in tokens
		const int len = strlen(line);
		char *toks[len / 2];
		int cnt = 0;
		while (*line != '\0') {
			toks[cnt++] = line;
			while (*line != '\0' && !isspace(*line))
				line++;
			if (*line == '\0')
				break;
			*line++ = '\0';
			while (*line != '\0' && isspace(*line))
				line++;
		}
		// If user specified that data are labelled, move the last token
		// to the label array.
		if (lbl == true) {
			tok->lbl[t] = toks[cnt - 1];
			cnt--;
		}
		// And put the remaining tokens in the tok_t object
		tok->cnts[t] = cnt;
		tok->toks[t] = xmalloc(sizeof(char *) * cnt);
		memcpy(tok->toks[t], toks, sizeof(char *) * cnt);
	}
	tok->len = T;
	// So now the tok object is ready, we can start building the seq_t
	// object by appling patterns. First we allocate the seq_t object. The
	// sequence itself as well as the sub array are allocated in one time.
	seq_t *seq = xmalloc(sizeof(seq_t) + sizeof(pos_t) * T);
	seq->raw = xmalloc(sizeof(size_t) * (rdr->nuni + rdr->nbi) * T);
	seq->len = T;
	size_t *tmp = seq->raw;
	for (int t = 0; t < T; t++) {
		seq->pos[t].lbl  = none;
		seq->pos[t].uobs = tmp; tmp += rdr->nuni;
		seq->pos[t].bobs = tmp; tmp += rdr->nbi;
	}
	// Next, we can build the observations list by applying the patterns on
	// the tok_t sequence.
	for (int t = 0; t < T; t++) {
		pos_t *pos = &seq->pos[t];
		pos->ucnt = 0;
		pos->bcnt = 0;
		for (int x = 0; x < rdr->npats; x++) {
			// Get the observation and map it to an identifier
			const char *obs = pat_exec(rdr->pats[x], tok, t);
			size_t id = qrk_str2id(rdr->obs, obs);
			if (id == none)
				continue;
			// If the observation is ok, add it to the lists
			int kind = 0;
			switch (obs[0]) {
				case 'u': kind = 1; break;
				case 'b': kind = 2; break;
				case '*': kind = 3; break;
			}
			if (kind & 1)
				pos->uobs[pos->ucnt++] = id;
			if (kind & 2)
				pos->bobs[pos->bcnt++] = id;
		}
	}
	// And finally, if the user specified it, populate the labels
	if (lbl == true) {
		for (int t = 0; t < T; t++) {
			const char *lbl = tok->lbl[t];
			size_t id = qrk_str2id(rdr->lbl, lbl);
			seq->pos[t].lbl = id;
		}
	}
	// Before returning the sequence, we have to free the tok_t
	for (int t = 0; t < T; t++) {
		if (tok->cnts[t] == 0)
			continue;
		free(tok->toks[t][0]);
		free(tok->toks[t]);
	}
	free(tok->cnts);
	if (lbl == true)
		free(tok->lbl);
	free(tok);
	return seq;
}

/* rdr_readseq:
 *   Simple wrapper around rdr_readraw and rdr_raw2seq to directly read a
 *   sequence as a seq_t object from file. This take care of all the process
 *   and correctly free temporary data. If lbl is true the sequence is assumed
 *   to be labeled.
 *   Return NULL if end of file occure before anything as been read.
 */
static seq_t *rdr_readseq(rdr_t *rdr, FILE *file, bool lbl) {
	raw_t *raw = rdr_readraw(rdr, file);
	if (raw == NULL)
		return NULL;
	seq_t *seq = rdr_raw2seq(rdr, raw, lbl);
	rdr_freeraw(raw);
	return seq;
}

/* rdr_readdat:
 *   Read a full dataset at once and return it as a dat_t object. This function
 *   take and interpret his parameters like the single sequence reading
 *   function.
 */
static dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl) {
	// Prepare dataset
	int size = 1000;
	dat_t *dat = xmalloc(sizeof(dat_t));
	dat->nseq = 0;
	dat->mlen = 0;
	dat->lbl = lbl;
	dat->seq = xmalloc(sizeof(seq_t *) * size);
	// Load sequences
	while (!feof(file)) {
		// Read the next sequence
		seq_t *seq = rdr_readseq(rdr, file, lbl);
		if (seq == NULL)
			break;
		// Grow the buffer if needed
		if (dat->nseq == size) {
			size *= 1.4;
			dat->seq = xrealloc(dat->seq, sizeof(seq_t *) * size);
		}
		// And store the sequence
		dat->seq[dat->nseq++] = seq;
		dat->mlen = max(dat->mlen, seq->len);
		if (dat->nseq % 1000 == 0)
			info("%7d sequences loaded\n", dat->nseq);
	}
	// If no sequence readed, cleanup and repport
	if (dat->nseq == 0) {
		free(dat->seq);
		free(dat);
		return NULL;
	}
	// Adjust the dataset size and return
	if (size > dat->nseq)
		dat->seq = xrealloc(dat->seq, sizeof(seq_t *) * dat->nseq);
	return dat;
}

/*******************************************************************************
 * Linear chain CRF model
 *
 *   There is three concept that must be well understand here, the labels,
 *   observations, and features. The labels are the values predicted by the
 *   model at each point of the sequence and denoted by Y. The observations are
 *   the values, at each point of the sequence, given to the model in order to
 *   predict the label and denoted by O. A feature is a test on both labels and
 *   observations, denoted by F. In linear chain CRF there is two kinds of
 *   features :
 *     - unigram feature who represent a test on the observations at the current
 *       point and the label at current point.
 *     - bigram feature who represent a test on the observation at the current
 *       point and two labels : the current one and the previous one.
 *   So for each observation, there Y possible unigram features and Y*Y possible
 *   bigram features. The kind of features used by the model for a given
 *   observation depend on the pattern who generated it.
 ******************************************************************************/
typedef struct mdl_s mdl_t;
struct mdl_s {
	opt_t   *opt;     //       options for training

	// Size of various model parameters
	size_t   nlbl;    //   Y   number of labels
	size_t   nobs;    //   O   number of observations
	size_t   nftr;    //   F   number of features

	// Informations about observations
	char    *kind;    //  [O]  observations type
	size_t  *uoff;    //  [O]  unigram weights offset
	size_t  *boff;    //  [O]  bigram weights offset

	// The model itself
	double  *theta;   //  [F]  features weights

	// Datasets
	dat_t   *train;   //       training dataset
	dat_t   *devel;   //       development dataset
	rdr_t   *reader;

	// Timing
	tms_t    timer;   //       start time of last iter
	double   total;   //       total training time
};

/* mdl_new:
 *   Allocate a new empty model object linked with the given reader. The model
 *   have to be synchronized before starting training or labelling. If you not
 *   provide a reader (as it will loaded from file for example) you must be sure
 *   to set one in the model before any attempts to synchronize it.
 */
static mdl_t *mdl_new(rdr_t *rdr) {
	mdl_t *mdl = xmalloc(sizeof(mdl_t));
	mdl->nlbl   = mdl->nobs  = mdl->nftr = 0;
	mdl->kind   = NULL;
	mdl->uoff   = mdl->boff  = NULL;
	mdl->theta  = NULL;
	mdl->train  = mdl->devel = NULL;
	mdl->reader = rdr;
	mdl->total  = 0.0;
	return mdl;
}

/* mdl_free:
 *   Free all memory used by a model object inculding the reader and datasets
 *   loaded in the model.
 */
static void mdl_free(mdl_t *mdl) {
	free(mdl->kind);
	free(mdl->uoff);
	free(mdl->boff);
	free(mdl->theta);
	if (mdl->train != NULL)
		rdr_freedat(mdl->train);
	if (mdl->devel != NULL)
		rdr_freedat(mdl->devel);
	if (mdl->reader != NULL)
		rdr_free(mdl->reader);
}

/* mdl_sync:
 *   Synchronize the model with its reader. As the model is just a placeholder
 *   for features weights and interned sequences, it know very few about the
 *   labels and observations, all the informations are kept in the reader. A
 *   sync will get the labels and observations count as well as the observation
 *   kind from the reader and build internal structures representing the model.
 *
 *   If the model was already synchronized before, there is an existing model
 *   incompatible with the new one to be created. In this case there is two
 *   possibility :
 *     - If only new observations was added, the weights of the old ones remain
 *       valid and are kept as they form a probably good starting point for
 *       training the new model, the new observation get a 0 weight ;
 *     - If new labels was added, the old model are trully meaningless so we
 *       have to fully discard them and build a new empty model.
 *   In any case, you must never change existing labels or observations, if this
 *   happen, you need to create a new model and destroy this one.
 *
 *   After synchronization, the labels and observations databases are locked to
 *   prevent new one to be created. You must unlock them explicitly if needed.
 *   This reduce the risk of mistakes.
 */
static void mdl_sync(mdl_t *mdl) {
	const size_t Y = mdl->reader->lbl->count;
	const size_t O = mdl->reader->obs->count;
	// If model is already synchronized, do nothing and just return
	if (mdl->nlbl == Y && mdl->nobs == O)
		return;
	if (Y == 0 || O == 0)
		fatal("cannot synchronize an empty model");
	// If new labels was added, we have to discard all the model. In this
	// case we also display a warning as this is probably not expected by
	// the user. If only new observations was added, we will try to expand
	// the model.
	size_t oldF = mdl->nftr;
	size_t oldO = mdl->nobs;
	if (mdl->nlbl != Y && mdl->nlbl != 0) {
		warning("labels count changed, discarding the model");
		free(mdl->kind);  mdl->kind  = NULL;
		free(mdl->uoff);  mdl->uoff  = NULL;
		free(mdl->boff);  mdl->boff  = NULL;
		free(mdl->theta); mdl->theta = NULL;
		oldF = oldO = 0;
		mdl->nlbl = Y;
	}
	mdl->nobs = O;
	// Allocate the observations datastructure. If the model is empty or
	// discarded, a new one iscreated, else the old one is expanded.
	char   *kind = xrealloc(mdl->kind, sizeof(char  ) * O);
	size_t *uoff = xrealloc(mdl->uoff, sizeof(char  ) * O);
	size_t *boff = xrealloc(mdl->boff, sizeof(char  ) * O);
	mdl->kind = kind;
	mdl->uoff = uoff;
	mdl->boff = boff;
	// Now, we can setup the features. For each new observations we fill the
	// kind and offsets arrays and count total number of features as well.
	size_t F = oldF;
	for (size_t o = oldO; o < O; o++) {
		const char *obs = qrk_id2str(mdl->reader->obs, o);
		switch (obs[0]) {
			case 'u': kind[o] = 1; break;
			case 'b': kind[o] = 2; break;
			case '*': kind[o] = 3; break;
		}
		if (kind[o] & 1)
			uoff[o] = F, F += Y;
		if (kind[o] & 2)
			boff[o] = F, F += Y * Y;
	}
	mdl->nftr = F;
	// We can finally grow the features weights vector itself. We set all
	// the new features to 0.0 but don't touch the old ones.
	mdl->theta = xrealloc(mdl->theta, sizeof(double) * F);
	for (size_t f = oldF; f < F; f++)
		mdl->theta[f] = 0.0;
	// And lock the databases
	mdl->reader->lbl->lock = true;
	mdl->reader->obs->lock = true;
}

/******************************************************************************
 * Sequence tagging
 *
 *   This module implement sequence tagging using a trained model and model
 *   evaluation on devlopment set.
 *
 *   The viterbi can be quite intensive on the stack if you push in it long
 *   sequence and use large labels set. It's less a problem than in gradient
 *   computations but it can show up in particular cases. The fix is to call it
 *   through the mth_spawn function and request enough stack space, this will be
 *   fixed in next version.
 ******************************************************************************/

/* viterbi:
 *   This function implement the Viterbi algorithm in order to decode the most
 *   probable sequence of labels according to the model. Some part of this code
 *   is very similar to the computation of the gradient as expected.
 *
 *   And like for the gradient, the caller is responsible to ensure there is
 *   enough stack space.
 */
static void tag_viterbi(const mdl_t *mdl, const seq_t *seq, size_t out[]) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	// Like for the gradient, we rely on stack storage and let the caller
	// ensure there is enough free space there. This function will need
	//   8 * T * (2 + Y * (1 + Y))
	// bytes of stack plus a bit more for variables.
	double psi [T][Y][Y];
	double back[T][Y];
	double cur [Y];
	double old [Y];
	// We first have to compute the Ψ_t(y',y,x_t) weights defined as
	//   Ψ_t(y',y,x_t) = \exp( ∑_k θ_k f_k(y',y,x_t) )
	// So at position 't' in the sequence, for each couple (y',y) we have
	// to sum weights of all features.
	// This is the same than what we do for computing the gradient but, as
	// the viterbi algorithm also work in the logarithmic space, we can
	// remove the exponential.
	//
	// Only the observations present at this position will have a non-nul
	// weight so we can sum only on thoses.
	//
	// As we use only two kind of features: unigram and bigram, we can
	// rewrite this as
	//   ∑_k μ_k(y, x_t) f_k(y, x_t) + ∑_k λ_k(y', y, x_t) f_k(y', y, x_t)
	// Where the first sum is over the unigrams features and the second is
	// over bigrams ones.
	//
	// This allow us to compute Ψ efficiently in two steps
	//   1/ we sum the unigrams features weights by looping over actives
	//        unigrams observations. (we compute this sum once and use it
	//        for each value of y')
	//   2/ we add the bigrams features weights by looping over actives
	//        bigrams observations (we don't have to do this for t=0 since
	//        there is no bigrams here)
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				sum += x[mdl->uoff[o] + y];
			}
			for (size_t yp = 0; yp < Y; yp++)
				psi[t][yp][y] = sum;
		}
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double sum = 0.0;
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					sum += x[mdl->boff[o] + d];
				}
				psi[t][yp][y] += sum;
			}
		}
	}
	// Now we can do the Viterbi algorithm. This is very similar to the
	// forward pass
	//   | α_1(y) = Ψ_1(y,x_1)
	//   | α_t(y) = max_{y'} α_{t-1}(y') + Ψ_t(y',y,x_t)
	// We just replace the sum by a max and as we do the computation in the
	// logarithmic space the product become a sum. (this also mean that we
	// don't have to worry about numerical problems)
	//
	// Next we have to walk backward over the α in order to find the best
	// path. In order to do this efficiently, we keep in the 'back' array
	// the indice of the y value selected by the max. This also mean that
	// we only need the current and previous value of the α vectors, not
	// the full matrix.
	for (size_t y = 0; y < Y; y++)
		cur[y] = psi[0][0][y];
	for (int t = 1; t < T; t++) {
		for (size_t y = 0; y < Y; y++)
			old[y] = cur[y];
		for (size_t y = 0; y < Y; y++) {
			double bst = -1.0;
			int    idx = 0;
			for (size_t yp = 0; yp < Y; yp++) {
				double val = psi[t][yp][y] + old[yp];
				if (val > bst) {
					bst = val;
					idx = yp;
				}
			}
			back[t][y] = idx;
			cur[y]     = bst;
		}
	}
	// We can now build the sequence of labels predicted by the model. For
	// this we search in the last α vector the best value. Using this index
	// as a starting point in the back-pointer array we finally can decode
	// the best sequence.
	int bst = 0;
	for (size_t y = 1; y < Y; y++)
		if (cur[y] > cur[bst])
			bst = y;
	for (int t = T; t > 0; t--) {
		out[t - 1] = bst;
		bst = back[t - 1][bst];
	}
}

/* tag_label:
 *   Label a data file using the current model. This output an almost exact copy
 *   of the input file with an additional column with the predicted label. If
 *   the check option is specified, the input file must be labelled and the
 *   predicted labels will be checked against the provided ones. This will
 *   output error rates during the labelling and detailed statistics per label
 *   at the end.
 */
static void tag_label(const mdl_t *mdl, FILE *fin, FILE *fout, bool check) {
	qrk_t *lbls = mdl->reader->lbl;
	const size_t Y = mdl->nlbl;
	// We start by preparing the statistic collection to be ready if check
	// option is used. The stat array hold the following for each label
	//   [0] # of reference with this label
	//   [1] # of token we have taged with this label
	//   [2] # of match of the two preceding
	int tcnt = 0, terr = 0;
	int scnt = 0, serr = 0;
	int stat[3][Y];
	for (size_t y = 0; y < Y; y++)
		stat[0][y] = stat[1][y] = stat[2][y] = 0;
	// Next read the input file sequence by sequence and label them, we have
	// to take care of not discarding the raw input as we want to send it
	// back to the output with the additional predicted labels.
	while (!feof(fin)) {
		// So, first read an input sequence keeping the raw_t object
		// available, and label it with Viterbi.
		raw_t *raw = rdr_readraw(mdl->reader, fin);
		if (raw == NULL)
			break;
		seq_t *seq = rdr_raw2seq(mdl->reader, raw, check);
		size_t out[seq->len];
		tag_viterbi(mdl, seq, out);
		// Next we output the raw sequence with an aditional column for
		// the predicted labels
		for (int t = 0; t < seq->len; t++) {
			fprintf(fout, "%s ", raw->lines[t]);
			fprintf(fout, "\t%s\n", qrk_id2str(lbls, out[t]));
		}
		fprintf(fout, "\n");
		// If user provided reference labels, use them to collect
		// statistics about how well we have performed here.
		if (check) {
			bool err = false;
			for (int t = 0; t < seq->len; t++) {
				stat[0][seq->pos[t].lbl]++;
				stat[1][out[t]]++;
				if (seq->pos[t].lbl != out[t])
					terr++, err = true;
				else
					stat[2][out[t]]++;
			}
			tcnt += seq->len;
			serr += err;
		}
		// Cleanup memory used for this sequence
		rdr_freeseq(seq);
		rdr_freeraw(raw);
		// And report our progress, at regular interval we display how
		// much sequence are labelled and if possible the current tokens
		// and sequence error rates.
		if (++scnt % 1000 == 0) {
			info("%10zu sequences labeled", scnt);
			if (check) {
				const double te = (double)terr  / tcnt * 100.0;
				const double se = (double)serr  / scnt * 100.0;
				info("\t%5.2f%%/%5.2f%%", te, se);
			}
			info("\n");
		}
	}
	// If user have provided reference labels, we have collected a lot of
	// statistics and we can repport global token and sequence error rate as
	// well as precision recall and f-measure for each labels.
	if (check) {
		const double te = (double)terr  / tcnt * 100.0;
		const double se = (double)serr  / scnt * 100.0;
		info("    Nb sequences  : %zu\n", scnt);
		info("    Token error   : %5.2f%%\n", te);
		info("    Sequence error: %5.2f%%\n", se);
		info("* Per label statistics\n");
		for (size_t y = 0; y < Y; y++) {
			const char *lbl = qrk_id2str(lbls, y);
			const double Rc = (double)stat[2][y] / stat[0][y];
			const double Pr = (double)stat[2][y] / stat[1][y];
			const double F1 = 2.0 * (Pr * Rc) / (Pr + Rc);
			info("    %-6s", lbl);
			info("  Pr=%.2f", Pr);
			info("  Rc=%.2f", Rc);
			info("  F1=%.2f\n", F1);
		}
	}
}

/*******************************************************************************
 * User interaction during training
 *
 *   Handle progress reporting during training and clean early stoping. Trainers
 *   have to call uit_progress at the end of each iterations, this will display
 *   various informations for the user.
 *   Timing is also done here, an iteration is assumed to take all the time
 *   between to call to the progress function and evualtion on the devel data
 *   are included.
 *
 *   This module setup a signal handler for SIGINT. If this signal is catched,
 *   the uit_stop global variable to inform the trainer that it have to stop as
 *   early as possible, discarding the recent computations if they cannot be
 *   integrated very quickly. They must leave the model in a clean state. Any
 *   further signal will terminate the program. So it's simple :
 *     - 1 signal mean "I can wait a little so try to stop as soon as possible
 *         but leave me a working model"
 *     - 2 signal mean "Stop immediatly what you are doing, I can't wait and
 *         don't care about getting a working model"
 ******************************************************************************/

/* uit_stop:
 *   This value is set to true when the user request the trainer to stop. In
 *   this case, the trainer have to stop as soon as possible in a clean state,
 *   discarding the lasts computations if it cannot integrate them quickly.
 */
static bool uit_stop = false;

/* uit_signal:
 *   Signal handler to catch interupt signal. When a signal is received, the
 *   trainer is aksed to stop as soon as possible leaving the model in a clean
 *   state. We don't reinstall the handler so if user send a second interupt
 *   signal, the program will stop imediatly. (to cope with BSD system, we even
 *   reinstall explicitly the default handler)
 */
static void uit_signal(int sig) {
	signal(sig, SIG_DFL);
	uit_stop = true;
}

/* uit_setup:
 *   Install the signal handler for clean early stop from the user if possible
 *   and start the timer.
 */
static void uit_setup(mdl_t *mdl) {
	uit_stop = false;
	if (signal(SIGINT, uit_signal) == SIG_ERR)
		warning("failed to set signal handler, no clean early stop");
	times(&mdl->timer);
}

/* uit_cleanup:
 *   Remove the signal handler restoring the defaul behavior in case of
 *   interrupt.
 */
static void uit_cleanup(mdl_t *mdl) {
	unused(mdl);
	signal(SIGINT, SIG_DFL);
}

/* uit_progress:
 *   Display a progress repport to the user consisting of some informations
 *   provided by the trainer: iteration count and objective function value, and
 *   some informations computed here on the current model performances.
 *   This function return true if the trainer have to keep training the model
 *   and false if he must stop, so this is were we will implement the trainer
 *   independant stoping criterion.
 */
static bool uit_progress(mdl_t *mdl, int it, double obj) {
	// We first evaluate the current model performances on the devel dataset
	// if available, else on the training dataset. We compute tokens and
	// sequence error rate.
	dat_t *dat = (mdl->devel == NULL) ? mdl->train : mdl->devel;
	int tcnt = 0, terr = 0;
	int scnt = 0, serr = 0;
	for (int s = 0; s < dat->nseq; s++) {
		// Tag the sequence with the viterbi
		const seq_t *seq = dat->seq[s];
		const int    T   = seq->len;
		size_t out[T];
		tag_viterbi(mdl, seq, out);
		// And check for eventual (probable ?) errors
		bool err = false;
		for (int t = 0; t < T; t++)
			if (seq->pos[t].lbl != out[t])
				terr++, err = true;
		tcnt += T, scnt += 1;
		serr += err;
	}
	const double te = (double)terr / tcnt * 100.0;
	const double se = (double)serr / scnt * 100.0;
	// Next, we compute the number of active features
	size_t act = 0;
	for (size_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			act++;
	// Compute timings. As some training algorithms are multi-threaded, we
	// cannot use ansi/c function and must rely on posix one to sum time
	// spent in main thread and in child ones.
	tms_t now; times(&now);
	double tm = (now.tms_utime  - mdl->timer.tms_utime )
		  + (now.tms_cutime - mdl->timer.tms_cutime);
	tm /= sysconf(_SC_CLK_TCK);
	mdl->total += tm;
	mdl->timer  = now;
	// And display progress report
	info("  [%4d]", it);
	info(obj >= 0.0 ? " obj=%-10.2f" : " obj=NA", obj);
	info(" act=%-8zu", act);
	info(" err=%5.2f%%/%5.2f%%", te, se);
	info(" time=%.2fs/%.2fs", tm, mdl->total);
	info("\n");
	// And return
	if (uit_stop)
		return false;
	return true;
}

/******************************************************************************
 *                    Single sequence gradient computation
 *
 * This section is responsible for computing the gradient of the log-likelihood
 * function to optimize over a single sequence.
 *
 * There is two version of this code, one using dense matrix and one with sparse
 * matrix. The sparse version use the fact that for L1 regularized trainers, the
 * bigrams scores will be very sparse so there is a way to reduce the amount of
 * computation needed in the forward backward at the price of a more complex
 * implementation. Due to the fact that using a sparse matrix have a cost, this
 * implementation is slower on L2 regularized models and on lighty
 * L1-regularized models, this is why there is also a classical dense version of
 * the algorithm used for example by the L-BFGS trainer.
 *
 * The sparse matrix implementation is a bit tricky because we need to store all
 * values in sequences in order to use the vector exponential who gives also a
 * lot of performance improvement on vector able machine.
 * We need four arrays noted <val>, <off>, <idx>, and <yp>. For each positions
 * t, <off>[t] value indicate where the non-zero values for t starts in <val>.
 * The other arrays gives the y and yp indices of these values. The easier one
 * to retrieve is yp, the yp indice for value at <val>[<off>[t] + n] is stored
 * at the same position in <yp>.
 * The y are more difficult: the indice y are stored with n between <idx>[y-1]
 * and <idx>[y]. It may seems inefective but the matrix is indexed in the other
 * way, we go through the idx array, and for each y we get the yp and values, so
 * in practice it's very efficient.
 *
 * This can seem too complex but we have to keep in mind that Y are generally
 * very low and any sparse-matrix have overhead so we have to reduce it to the
 * minimum in order to get a real improvment. Dedicated library are optimized
 * for bigger matrix where the overhead is not a so important problem.
 * Another problem here is cache size. The optimization process will last most
 * of his time in this function so it have to be well optimized and we already
 * need a lot of memory for other data so we have to be carefull here if we
 * don't want to flush the cache all the time. Sparse matrix require less memory
 * than dense one only if we now in advance the number of non-zero entries,
 * which is not the case here, so we have to use a scheme which in the worst
 * case use as less as possible memory.
 *
 ******************************************************************************/

/* grd_fldoseq:
 *   This function compute the gradient and value of the negative log-likelihood
 *   of the model over a single training sequence.
 *   The computation is organised to make the best compromise between efficiency
 *   and readability. Some things can be optimised a bit but the gain is not
 *   really noticeable so I prefer keep thing like this.
 *
 *   This function will not clear the gradient before computation, but instead
 *   just accumulate the values for the given sequence in it. This allow to
 *   easily compute the gradient over a set of sequences.
 *
 *   Beware that this function is quite stack intensive and it is the caller
 *   responsibility to ensure that there will be no problems. See the note below
 *   for more details.
 */
static double grd_fldoseq(const mdl_t *mdl, const seq_t *seq, double *g) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	// In order to simplify the code as well as the memory managment we
	// allocate all the temporary array on the stack instead of the heap.
	// This imply that for long sequence or huge labels set we will need a
	// large amount of stack memory. We let the caller ensure there is
	// enough free space there. This function will need
	//   8 * T * (3 + Y * (2 + Y))
	// bytes of stack (assuming that sizeof(double) is eight bytes) for the
	// arrays and a bit more for the other variables and temporary created
	// by the compiler.
	double psi  [T][Y][Y] xvm_align;
	double alpha[T][Y];
	double beta [T][Y];
	double scale[T];
	double uz   [T];
	double bz   [T];
	// We first have to compute the Ψ_t(y',y,x) weights defined as
	//   Ψ_t(y',y,x) = \exp( ∑_k θ_k f_k(y',y,x_t) )
	// So at position 't' in the sequence, for each couple (y',y) we have
	// to sum weights of all features.
	//
	// Only the observations present at this position will have a non-nul
	// weight so we can sum only on thoses.
	//
	// As we use only two kind of features: unigram and bigram, we can
	// rewrite this as
	//   \exp (  ∑_k μ_k(y, x_t)     f_k(y, x_t)
	//         + ∑_k λ_k(y', y, x_t) f_k(y', y, x_t) )
	// Where the first sum is over the unigrams features and the second is
	// over bigrams ones.
	//
	// This allow us to compute Ψ efficiently in three steps
	//   1/ we sum the unigrams features weights by looping over actives
	//        unigrams observations. (we compute this sum once and use it
	//        for each value of y')
	//   2/ we add the bigrams features weights by looping over actives
	//        bigrams observations (we don't have to do this for t=0 since
	//        there is no bigrams here)
	//   3/ we take the component-wise exponential of the resulting matrix
	//        (this can be done efficiently with vector maths)
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				sum += x[mdl->uoff[o] + y];
			}
			for (size_t yp = 0; yp < Y; yp++)
				psi[t][yp][y] = sum;
		}
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double sum = 0.0;
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					sum += x[mdl->boff[o] + d];
				}
				psi[t][yp][y] += sum;
			}
		}
	}
	xvm_expma((double *)psi, (double *)psi, 0.0, (size_t)T * Y * Y);
	// Now, we go to the forward-backward algorithm. As this part of the
	// code rely on a lot of recursive sums and products of exponentials,
	// we have to take care of numerical problems.
	//
	// First the forward recursion
	//   | α_1(y) = Ψ_1(y,x)
	//   | α_t(y) = ∑_{y'} α_{t-1}(y') * Ψ_t(y',y,x)
	//
	// The numerical problems can appear here. To solve them we will scale
	// the α_t vectors so they sum to 1 but we have to keep the scaling
	// coeficient as we will need them later.
	for (size_t y = 0; y < Y; y++)
		alpha[0][y] = psi[0][0][y];
	scale[0] = xvm_unit(alpha[0], alpha[0], Y);
	for (int t = 1; t < T; t++) {
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t yp = 0; yp < Y; yp++)
				sum += alpha[t - 1][yp] * psi[t][yp][y];
			alpha[t][y] = sum;
		}
		scale[t] = xvm_unit(alpha[t], alpha[t], Y);
	}
	// Next come the backward recursion which is very similar
	//   | β_T(y') = 1
	//   | β_t(y') = ∑_y β_{t+1}(y) * Ψ_{t+1}(y',y,x)
	//
	// We also do scaling here as the same problems can happen be we don't
	// have to keep the scaling factor as we will see later.
	for (size_t yp = 0; yp < Y; yp++)
		beta[T - 1][yp] = 1.0 / Y;
	for (int t = T - 1; t > 0; t--) {
		for (size_t yp = 0; yp < Y; yp++) {
			double sum = 0.0;
			for (size_t y = 0; y < Y; y++)
				sum += beta[t][y] * psi[t][yp][y];
			beta[t - 1][yp] = sum;
		}
		xvm_unit(beta[t - 1], beta[t - 1], Y);
	}
	// Now, we have to compute the nomalization factor. But, due to the
	// scaling performed during the forward-backward recursions, we have to
	// compute it at each positions and separately for unigrams and bigrams
	// using
	//   for unigrams: Z_θ(t) = ∑_y α_t(y) β_t(y)
	//   for bigrams:  Z_θ(t) = ∑_y α_t(y) β_t(y) / α-scale_t
	// with α-scale_t the scaling factor used for the α vector at position t
	// in the forward recursion.
	//
	// In order to speedup a bit the computations, we directly compute the
	// inverse of these values.
	for (int t = 0; t < T; t++) {
		double z = 0.0;
		for (size_t y = 0; y < Y; y++)
			z += alpha[t][y] * beta[t][y];
		uz[t] = 1.0 / z;
		bz[t] = scale[t] / z;
	}
	// Now, we have all we need to compute the gradient of the negative log-
	// likelihood
	//  ∂-L(θ)
	//  ------ =    ∑_t ∑_{(y',y)} f_k(y',y,x_t) p_θ(y_{t-1}=y',y_t=y|x)
	//   ∂θ_k     - ∑_t f_k(y_{t-1},y_t,x_t)
	//
	// The first term is the expectation of f_k under the model distribution
	// and the second one is the expectation of f_k under the empirical
	// distribution.
	//
	// The second is very simple to compute as we just have to sum over the
	// actives observations in the sequence.
	// The first one is more tricky as it involve computing the probability
	// p_θ. This is where we use all the previous computations. Again we
	// separate the computations for unigrams and bigrams here.
	//
	// These probabilities are given by
	//   p_θ(y_t=y|x)            = α_t(y)β_t(y) / Z_θ
	//   p_θ(y_{t-1}=y',y_t=y|x) = α_{t-1}(y') Ψ_t(y',y,x) β_t(y) / Z_θ
	// but we have to remember that, since we have scaled the α and β, we
	// have to use the local normalization constants.
	//
	// We must also take care of not clearing previous value of the gradient
	// vector but just adding the contribution of this sequence. This allow
	// to compute it easily the gradient over more than one sequence.
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// Add the expectation over the model distribution
		for (size_t y = 0; y < Y; y++) {
			const double e = alpha[t][y] * beta[t][y] * uz[t];
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				g[mdl->uoff[o] + y] += e;
			}
		}
		// And substract the expectation over the empirical one.
		const size_t y = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			g[mdl->uoff[pos->uobs[n]] + y] -= 1.0;
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// Add the expectation over the model distribution
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				const double e = alpha[t - 1][yp] * beta[t][y]
				               * psi[t][yp][y] * bz[t];
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					g[mdl->boff[o] + d] += e;
				}
			}
		}
		// And substract the expectation over the empirical one.
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		const size_t d  = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			g[mdl->boff[pos->bobs[n]] + d] -= 1.0;
	}
	// And the final touch, the computation of the negative log-likelihood
	//   -L(θ) = log(Z_θ) - ∑_t ∑_k θ_k f_k(y_{t-1}, y_t, x_t)
	//
	// The numerical problems show again here as we cannot compute the Z_θ
	// directly for the same reason we have done scaling. Fortunately, there
	// is a way to directly compute his logarithm
	//   log(Z_θ) = log( ∑_y α_t(y) β_t(y) )
	//            - ∑_{i=1..t} log(α-scale_i)
	//            - ∑_{i=t..T} log(β-scale_i)
	// for any value of t.
	//
	// So we can compute it at any position in the sequence but the last one
	// is easier as the value of β_T(y) and β-scale_T are constant and
	// cancel out. This is why we have just keep the α-scale_t values.
	double logz = 0.0;
	for (size_t y = 0; y < Y; y++)
		logz += alpha[T - 1][y];
	logz = log(logz);
	for (int t = 0; t < T; t++)
		logz -= log(scale[t]);
	// Now, we have the first term of -L(θ). We have now to substract the
	// second one. As we have done for the computation of Ψ, we separate the
	// sum over K in two sums, one for unigrams and one for bigrams. And, as
	// here also the weights will be non-nul only for observations present
	// in the sequence, we sum only over these ones.
	double lloss = logz;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t y   = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			lloss -= x[mdl->uoff[pos->uobs[n]] + y];
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t yp  = seq->pos[t - 1].lbl;
		const size_t y   = seq->pos[t    ].lbl;
		const size_t d   = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			lloss -= x[mdl->boff[pos->bobs[n]] + d];
	}
	return lloss;
}

/* grd_spdoseq:
 *   This function compute the gradient and value of the negative log-likelihood
 *   of the model over a single training sequence as grd_fldoseq but using
 *   sparse-matrix computation. This can speed-up training on sparse models but
 *   come at the price of stack memory usage.
 *
 *   This function is, as expected, extremely similar to the dense version but
 *   the difference are big enough that making a single one will extremly
 *   complex. I prefer to keep some code duplicated here than making them
 *   unreadable, but keep in mind that bugfix to one of these will probably
 *   apply to the other.
 */
static double grd_spdoseq(const mdl_t *mdl, const seq_t *seq, double *g) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	// In order to simplify the code as well as the memory managment we
	// allocate all the temporary array on the stack instead of the heap.
	// This imply that for long sequence or huge labels set we will need a
	// large amount of stack memory. We let the caller ensure there is
	// enough free space there. This function will need
	//   8 * T * (4 + Y * (4 + 2 * Y)) + Y * Y
	// bytes of stack (assuming that sizeof(double) is eight bytes) for the
	// arrays and a bit more for the other variables and temporary created
	// by the compiler.
	double psiuni[T][Y]      xvm_align;
	double psival[T * Y * Y] xvm_align;
	size_t psiyp [T * Y * Y];
	size_t psiidx[T][Y];
	size_t psioff[T];
	double alpha [T][Y];
	double beta  [T][Y];
	double scale [T];
	double uz    [T];
	double bz    [T];
	// We first have to compute the Ψ_t(y',y,x) weights defined as
	//   Ψ_t(y',y,x) = \exp( ∑_k θ_k f_k(y',y,x_t) )
	// So at position 't' in the sequence, for each couple (y',y) we have
	// to sum weights of all features.
	//
	// Only the observations present at this position will have a non-nul
	// weight so we can sum only on thoses.
	//
	// As we use only two kind of features: unigram and bigram, we can
	// rewrite this as
	//   \exp (  ∑_k μ_k(y, x_t)     f_k(y, x_t)
	//         + ∑_k λ_k(y', y, x_t) f_k(y', y, x_t) )
	// Where the first sum is over the unigrams features and the second is
	// over bigrams ones.
	//
	// For the sparse version, we keep the two sum separate so we will have
	// separate Ψ_t(y,x) and Ψ_t(y',y,x). The first one define a vector for
	// unigram at each position, and the second one a matrix for bigrams.
	// This is where the trick is as we will store Ψ_t(y',y,x) - 1. If the
	// sum is nul, his exponential will be 1.0 and so we have to store 0.0.
	// As most of the sum are expected to be nul the resulting matrix will
	// be very sparse and we will save computation in the forward-backward.
	//
	// So we compute Ψ differently here
	//   1/ we sum the unigrams features weights by looping over actives
	//        unigrams observations and store them in |psiuni|.
	//   2/ we sum the bigrams features weights by looping over actives
	//        bigrams observations (we don't have to do this for t=0 since
	//        there is no bigrams here) and we store the non-nul one in the
	//        sparse matrix.
	//   3/ we take the component-wise exponential of the unigrams vectors,
	//        and the component-wise exponential of the sparse matrix minus
	//        one. (here also this can be done efficiently with vector
	//        maths)
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				sum += x[mdl->uoff[o] + y];
			}
			psiuni[t][y] = sum;
		}
	}
	size_t off = 0;
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		psioff[t] = off;
		for (size_t y = 0, nnz = 0; y < Y; y++) {
			for (size_t yp = 0; yp < Y; yp++) {
				double sum = 0.0;
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					sum += x[mdl->boff[o] + yp * Y + y];
				}
				if (sum == 0.0)
					continue;
				psiyp [off] = yp;
				psival[off] = sum;
				nnz++, off++;
			}
			psiidx[t][y] = nnz;
		}
	}
	xvm_expma((double *)psiuni, (double *)psiuni, 0.0, (size_t)T * Y);
	xvm_expma((double *)psival, (double *)psival, 1.0, off);
	// Now, we go to the forward-backward algorithm. The numerical problems
	// are solved with scaling as explained in the dense gradient
	// computation.
	//
	// First the forward recursion
	//   | α_1(y) = Ψ_1(y,x)
	//   | α_t(y) = Ψ_t(y,x) * ∑_{y'} α_{t-1}(y') * Ψ_t(y',y,x)
	//
	// To work with the sparse matrix, this can be rewriten as
	//   | α_1(y) = Ψ_1(y,x)
	//   | α_t(y) = Ψ_t(y,x) * (   ∑_{y'} α_{t-1}(y')
	//                           + ∑_{y'} α_{t-1}(y') * (Ψ_t(y',y,x) - 1) )
	// The inner part contains two sums, the first one will be 1.0 as we
	// scale the α vectors, and the second is a sparse matrix multiplication
	// who need less than |Y|x|Y| multiplication if the matrix is really
	// sparse, so we will gain here.
	for (size_t y = 0; y < Y; y++)
		alpha[0][y] = psiuni[0][y];
	scale[0] = xvm_unit(alpha[0], alpha[0], Y);
	for (int t = 1; t < T; t++) {
		for (size_t y = 0; y < Y; y++)
			alpha[t][y] = 1.0;
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < psiidx[t][Y - 1]; ) {
			while (n >= psiidx[t][y])
				y++;
			while (n < psiidx[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				alpha[t][y] += alpha[t - 1][yp] * v;
				n++;
			}
		}
		for (size_t y = 0; y < Y; y++)
			alpha[t][y] *= psiuni[t][y];
		scale[t] = xvm_unit(alpha[t], alpha[t], Y);
	}
	// Next come the backward recursion which is very similar
	//   | β_T(y') = 1
	//   | β_t(y') = ∑_y β_{t+1}(y) * Ψ_{t+1}(y,x) * Ψ_{t+1}(y',y,x)
	//
	// Here also we have to rework a bit the formula to use the sparse
	// matrix
	//   | β_T(y') = 1
	//   | β_t(y') = ∑_y v_{t+1}(y) + ∑_y v_{t+1}(y) * (Ψ_{t+1}(y',y,x) - 1)
	// with
	//   v_{t+1}(y) = β_{t+1}(y) * Ψ_{t+1}(y,x)
	//
	// And here also we reduce the number of multiplication if the matrix is
	// really sparse.
	for (size_t yp = 0; yp < Y; yp++)
		beta[T - 1][yp] = 1.0 / Y;
	for (int t = T - 1; t > 0; t--) {
		double sum = 0.0, tmp[Y];
		for (size_t y = 0; y < Y; y++) {
			tmp[y] = beta[t][y] * psiuni[t][y];
			sum += tmp[y];
		}
		for (size_t y = 0; y < Y; y++)
			beta[t - 1][y] = sum;
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < psiidx[t][Y - 1]; ) {
			while (n >= psiidx[t][y])
				y++;
			while (n < psiidx[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				beta[t - 1][yp] += v * tmp[y];
				n++;
			}
		}
		xvm_unit(beta[t - 1], beta[t - 1], Y);
	}
	// Now, we have to compute the nomalization factor. But, due to the
	// scaling performed during the forward-backward recursions, we have to
	// compute it at each positions and separately for unigrams and bigrams
	// using
	//   for unigrams: Z_θ(t) = ∑_y α_t(y) β_t(y)
	//   for bigrams:  Z_θ(t) = ∑_y α_t(y) β_t(y) / α-scale_t
	// with α-scale_t the scaling factor used for the α vector at position t
	// in the forward recursion.
	//
	// In order to speedup a bit the computations, we directly compute the
	// inverse of these values.
	for (int t = 0; t < T; t++) {
		double z = 0.0;
		for (size_t y = 0; y < Y; y++)
			z += alpha[t][y] * beta[t][y];
		uz[t] = 1.0 / z;
		bz[t] = scale[t] / z;
	}
	// Now, we have all we need to compute the gradient of the negative log-
	// likelihood
	//  ∂-L(θ)
	//  ------ =    ∑_t ∑_{(y',y)} f_k(y',y,x_t) p_θ(y_{t-1}=y',y_t=y|x)
	//   ∂θ_k     - ∑_t f_k(y_{t-1},y_t,x_t)
	//
	// The first term is the expectation of f_k under the model distribution
	// and the second one is the expectation of f_k under the empirical
	// distribution.
	//
	// The second is very simple to compute as we just have to sum over the
	// actives observations in the sequence.
	// The first one is more tricky as it involve computing the probability
	// p_θ. This is where we use all the previous computations. Again we
	// separate the computations for unigrams and bigrams here.
	//
	// These probabilities are given by
	//   p_θ(y_t=y|x)            = α_t(y)β_t(y) / Z_θ
	//   p_θ(y_{t-1}=y',y_t=y|x) = α_{t-1}(y') Ψ_t(y',y,x_t) β_t(y) / Z_θ
	// but we have to remember that, since we have scaled the α and β, we
	// have to use the local normalization constants.
	//
	// The sparse matrix make things a bit more complicated here as we
	// cannot directly multiply with the original Ψ_t(y',y,x) because we
	// have split it two components and the second one is sparse, so we
	// have to make a quite complex workaround to fix that. We have to
	// explicitly build the expectation matrix. We first fill it with the
	// unigram component and next multiply it with the bigram one.
	//
	// We must also take care of not clearing previous value of the gradient
	// vector but just adding the contribution of this sequence. This allow
	// to compute it easily the gradient over more than one sequence.
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// Add the expectation over the model distribution
		for (size_t y = 0; y < Y; y++) {
			const double e = alpha[t][y] * beta[t][y] * uz[t];
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				g[mdl->uoff[o] + y] += e;
			}
		}
		// And substract the expectation over the empirical one.
		const size_t y = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			g[mdl->uoff[pos->uobs[n]] + y] -= 1.0;
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// We build the expectation matrix
		double e[Y][Y];
		for (size_t yp = 0; yp < Y; yp++)
			for (size_t y = 0; y < Y; y++)
				e[yp][y] = alpha[t - 1][yp] * beta[t][y]
				         * psiuni[t][y] * bz[t];
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < psiidx[t][Y - 1]; ) {
			while (n >= psiidx[t][y])
				y++;
			while (n < psiidx[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				e[yp][y] += e[yp][y] * v;
				n++;
			}
		}
		// Add the expectation over the model distribution
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					g[mdl->boff[o] + d] += e[yp][y];
				}
			}
		}
		// And substract the expectation over the empirical one.
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		const size_t d  = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			g[mdl->boff[pos->bobs[n]] + d] -= 1.0;
	}
	// And the final touch, the computation of the negative log-likelihood
	//   -L(θ) = log(Z_θ) - ∑_t ∑_k θ_k f_k(y_{t-1}, y_t, x_t)
	//
	// The numerical problems show again here as we cannot compute the Z_θ
	// directly for the same reason we have done scaling. Fortunately, there
	// is a way to directly compute his logarithm
	//   log(Z_θ) = log( ∑_y α_t(y) β_t(y) )
	//            - ∑_{i=1..t} log(α-scale_i)
	//            - ∑_{i=t..T} log(β-scale_i)
	// for any value of t.
	//
	// So we can compute it at any position in the sequence but the last one
	// is easier as the value of β_T(y) and β-scale_T are constant and
	// cancel out. This is why we have just keep the α-scale_t values.
	double logz = 0.0;
	for (size_t y = 0; y < Y; y++)
		logz += alpha[T - 1][y];
	logz = log(logz);
	for (int t = 0; t < T; t++)
		logz -= log(scale[t]);
	// Now, we have the first term of -L(θ). We have now to substract the
	// second one. As we have done for the computation of Ψ, we separate the
	// sum over K in two sums, one for unigrams and one for bigrams. And, as
	// here also the weights will be non-nul only for observations present
	// in the sequence, we sum only over these ones.
	double lloss = logz;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t y   = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			lloss -= x[mdl->uoff[pos->uobs[n]] + y];
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t yp  = seq->pos[t - 1].lbl;
		const size_t y   = seq->pos[t    ].lbl;
		const size_t d   = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			lloss -= x[mdl->boff[pos->bobs[n]] + d];
	}
	return lloss;
}

/* grd_doseq:
 *   This function is just a wrapper arround the two previous ones, selecting
 *   which one to call depending of the user setting.
 */
static double grd_doseq(const mdl_t *mdl, const seq_t *seq, double g[]) {
	if (!mdl->opt->sparse)
		return grd_fldoseq(mdl, seq, g);
	else
		return grd_spdoseq(mdl, seq, g);
}

/*******************************************************************************
 *
 ******************************************************************************/
int main(void) {
	return EXIT_SUCCESS;
}

