/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2012  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
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
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "wapiti.h"
#include "tools.h"
#include "vmath.h"

#if defined(__SSE2__) && !defined(XVM_ANSI)
#include <emmintrin.h>
#endif

/* xvm_mode:
 *   Return a string describing the SSE level used in the optimized code paths.
 */
const char *xvm_mode(void) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
	return "sse2";
#else
	return "no-sse";
#endif
}

/* xvm_new:
 *   Allocate a new vector suitable to be used in the SSE code paths. This
 *   ensure that the vector size contains the need padding. You must only use
 *   vector allocated by this function if you use the optimized code paths.
 */
double *xvm_new(uint64_t N) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
	if (N % 4 != 0)
		N += 4 - N % 4;
	void *ptr = _mm_malloc(sizeof(double) * N, 16);
	if (ptr == NULL)
		fatal("out of memory");
	return ptr;
#else
	return xmalloc(sizeof(double) * N);
#endif
}

/* xvm_free:
 *   Free a vector allocated by xvm_new.
 */
void xvm_free(double x[]) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
	_mm_free(x);
#else
	free(x);
#endif
}

/* xvm_neg:
 *   Return the component-wise negation of the given vector:
 *       r = -x
 */
void xvm_neg(double r[], const double x[], uint64_t N) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
	assert(r != NULL && ((uintptr_t)r % 16) == 0);
	assert(x != NULL && ((uintptr_t)x % 16) == 0);
	const __m128d vz = _mm_setzero_pd();
	for (uint64_t n = 0; n < N; n += 4) {
		const __m128d x0 = _mm_load_pd(x + n    );
		const __m128d x1 = _mm_load_pd(x + n + 2);
		const __m128d r0 = _mm_sub_pd(vz, x0);
		const __m128d r1 = _mm_sub_pd(vz, x1);
		_mm_store_pd(r + n,     r0);
		_mm_store_pd(r + n + 2, r1);
	}
#else
	for (uint64_t n = 0; n < N; n++)
		r[n] = -x[n];
#endif
}

/* xvm_sub:
 *   Return the difference of the two given vector:
 *       r = x .- y
 */
void xvm_sub(double r[], const double x[], const double y[], uint64_t N) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
	assert(r != NULL && ((uintptr_t)r % 16) == 0);
	assert(x != NULL && ((uintptr_t)x % 16) == 0);
	assert(y != NULL && ((uintptr_t)y % 16) == 0);
	for (uint64_t n = 0; n < N; n += 4) {
		const __m128d x0 = _mm_load_pd(x + n    );
		const __m128d x1 = _mm_load_pd(x + n + 2);
		const __m128d y0 = _mm_load_pd(y + n    );
		const __m128d y1 = _mm_load_pd(y + n + 2);
		const __m128d r0 = _mm_sub_pd(x0, y0);
		const __m128d r1 = _mm_sub_pd(x1, y1);
		_mm_store_pd(r + n,     r0);
		_mm_store_pd(r + n + 2, r1);
	}
#else
	for (uint64_t n = 0; n < N; n++)
		r[n] = x[n] - y[n];
#endif
}

/* xvm_scale:
 *   Return the given vector scaled by a constant:
 *     r = a * x
 */
void xvm_scale(double r[], const double x[], double a, uint64_t N) {
	for (uint64_t n = 0; n < N; n++)
		r[n] = x[n] * a;
}

/* xvm_norm:
 *   Store a normalized copy of the given vector in r and return the
 *   normalization factor.
 */
double xvm_unit(double r[], const double x[], uint64_t N) {
	double sum = 0.0;
	for (uint64_t n = 0; n < N; n++)
		sum += x[n];
	const double scale = 1.0 / sum;
	xvm_scale(r, x, scale, N);
	return scale;
}

/* xvm_norm:
 *   Return the euclidian norm of the given vector.
 */
double xvm_norm(const double x[], uint64_t N) {
	double r = 0.0;
#if defined(__SSE2__) && !defined(XVM_ANSI)
	assert(x != NULL && ((uintptr_t)x % 16) == 0);
	uint64_t n, d = N % 4;
	__m128d s0 = _mm_setzero_pd();
	__m128d s1 = _mm_setzero_pd();
	for (n = 0; n < N - d; n += 4) {
		const __m128d x0 = _mm_load_pd(x + n    );
		const __m128d x1 = _mm_load_pd(x + n + 2);
		const __m128d r0 = _mm_mul_pd(x0, x0);
		const __m128d r1 = _mm_mul_pd(x1, x1);
		s0 = _mm_add_pd(s0, r0);
		s1 = _mm_add_pd(s1, r1);
	}
	s0 = _mm_add_pd(s0, s1);
	s1 = _mm_shuffle_pd(s0, s0, _MM_SHUFFLE2(1, 1));
	s0 = _mm_add_pd(s0, s1);
	_mm_store_sd(&r, s0);
	for ( ; n < N; n++)
		r += x[n] * x[n];
#else
	for (uint64_t n = 0; n < N; n++)
		r += x[n] * x[n];
#endif
	return sqrt(r);
}

/* xvm_dot:
 *   Return the dot product of the two given vectors.
 */
double xvm_dot(const double x[], const double y[], uint64_t N) {
	double r = 0.0;
#if defined(__SSE2__) && !defined(XVM_ANSI)
	assert(x != NULL && ((uintptr_t)x % 16) == 0);
	assert(y != NULL && ((uintptr_t)y % 16) == 0);
	uint64_t n, d = N % 4;
	__m128d s0 = _mm_setzero_pd();
	__m128d s1 = _mm_setzero_pd();
	for (n = 0; n < N - d; n += 4) {
		const __m128d x0 = _mm_load_pd(x + n    );
		const __m128d x1 = _mm_load_pd(x + n + 2);
		const __m128d y0 = _mm_load_pd(y + n    );
		const __m128d y1 = _mm_load_pd(y + n + 2);
		const __m128d r0 = _mm_mul_pd(x0, y0);
		const __m128d r1 = _mm_mul_pd(x1, y1);
		s0 = _mm_add_pd(s0, r0);
		s1 = _mm_add_pd(s1, r1);
	}
	s0 = _mm_add_pd(s0, s1);
	s1 = _mm_shuffle_pd(s0, s0, _MM_SHUFFLE2(1, 1));
	s0 = _mm_add_pd(s0, s1);
	_mm_store_sd(&r, s0);
	for ( ; n < N; n++)
		r += x[n] * y[n];
#else
	for (uint64_t n = 0; n < N; n++)
		r += x[n] * y[n];
#endif
	return r;
}

/* xvm_axpy:
 *   Return the sum of x scaled by a and y:
 *       r = a * x + y
 */
void xvm_axpy(double r[], double a, const double x[], const double y[],
		uint64_t N) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
	assert(r != NULL && ((uintptr_t)r % 16) == 0);
	assert(x != NULL && ((uintptr_t)x % 16) == 0);
	assert(y != NULL && ((uintptr_t)y % 16) == 0);
	const __m128d va = _mm_set1_pd(a);
	for (uint64_t n = 0; n < N; n += 4) {
		const __m128d x0 = _mm_load_pd(x + n    );
		const __m128d x1 = _mm_load_pd(x + n + 2);
		const __m128d y0 = _mm_load_pd(y + n    );
		const __m128d y1 = _mm_load_pd(y + n + 2);
		const __m128d t0 = _mm_mul_pd(x0, va);
		const __m128d t1 = _mm_mul_pd(x1, va);
		const __m128d r0 = _mm_add_pd(t0, y0);
		const __m128d r1 = _mm_add_pd(t1, y1);
		_mm_store_pd(r + n,     r0);
		_mm_store_pd(r + n + 2, r1);
	}
#else
	for (uint64_t n = 0; n < N; n++)
		r[n] = a * x[n] + y[n];
#endif
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
 *   This code is copyright 2004-2012 Thomas Lavergne and licenced under the
 *   BSD licence like the remaining of Wapiti.
 */
void xvm_expma(double r[], const double x[], double a, uint64_t N) {
#if defined(__SSE2__) && !defined(XVM_ANSI)
  #define xvm_vconst(v) (_mm_castsi128_pd(_mm_set1_epi64x((v))))
	assert(r != NULL && ((uintptr_t)r % 16) == 0);
	assert(x != NULL && ((uintptr_t)x % 16) == 0);
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
	for (uint64_t n = 0; n < N; n += 4) {
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
#else
	for (uint64_t n = 0; n < N; n++)
		r[n] = exp(x[n]) - a;
#endif
}

