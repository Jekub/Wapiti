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
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/times.h>
#include <sys/resource.h>

#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"
#include "thread.h"
#include "viterbi.h"
#include "vmath.h"
#include "wapiti.h"

/******************************************************************************
 * Quasi-Newton optimizer
 *
 *   This section implement the quasi-Newton optimizer. We use the L-BFGS
 *   algorithm described by Liu and Nocedal in [1] and [2]. If an l1-norm must
 *   be applyed we fallback on the OWL-QN variant described in [3] by Galen and
 *   Jianfeng which allow to use L-BFGS for function not differentiable in 0.0.
 *
 *   [1] Updating quasi-Newton matrices with limited storage, Jorge Nocedal, in
 *       Mathematics of Computation, vol. 35(151) 773-782, July 1980.
 *   [2] On the limited memory BFGS method for large scale optimization, Dong C.
 *       Liu and Jorge Nocedal, in Mathematical Programming, vol. 45(1) 503-528,
 *       January 1989.
 *   [3] Scalable Training of L1-Regularized Log-Linear Models, Andrew Galen and
 *       Gao Jianfeng, in Proceedings of the 24th International Conference on
 *       Machine Learning (ICML), Corvallis, OR, 2007.
 ******************************************************************************/

static void trn_lbfgs(mdl_t *mdl) {
	const size_t F  = mdl->nftr;
	const int    K  = mdl->opt->maxiter;
	const int    M  = mdl->opt->lbfgs.histsz;
	const size_t W  = mdl->opt->nthread;
	const bool   l1 = mdl->opt->rho1 != 0.0;
	double *x, *xp; // Current and previous value of the variables
	double *g, *gp; // Current and previous value of the gradient
	double *pg;     // The pseudo-gradient (only for owl-qn)
	double *d;      // The search direction
	double *s[M];   // History value s_k = Δ(x,px)
	double *y[M];   // History value y_k = Δ(g,pg)
	double  p[M];   // ρ_k
	grd_t  *grds[W];
	// Initialization: Here, we have to allocate memory on the heap as we
	// cannot request so much memory on the stack as this will have a too
	// big impact on performance and will be refused by the system on non-
	// trivial models.
	// To make things simpler, we allocate all the memory in one call to
	// malloc and dispatch memory in the various arrays. The main pointer
	// will remain in the raw variable to be freed at the end.
	double *raw = xmalloc(sizeof(double) * F * (4 + M * 2 + l1 + W));
	double *tmp = raw;
	x  = mdl->theta;
	xp = tmp; tmp += F; g = tmp; tmp += F;
	gp = tmp; tmp += F; d = tmp; tmp += F;
	for (int m = 0; m < M; m++) {
		s[m] = tmp; tmp += F;
		y[m] = tmp; tmp += F;
	}
	pg = NULL;
	if (l1 == true)
		pg = tmp, tmp += F;
	grds[0] = grd_new(mdl, g);
	for (size_t w = 1; w < W; w++)
		grds[w] = grd_new(mdl, tmp), tmp += F;
	// Minimization: This is the heart of the function. (a big heart...) We
	// will perform iterations until one these conditions is reached
	//   - the maximum iteration count is reached
	//   - we have converged (upto numerical precision)
	//   - the report function return false
	//   - an error happen somewhere
	double fx = grd_gradient(mdl, g, pg, grds);
	for (int k = 0; !uit_stop && k < K; k++) {
		// 1st step: We compute the search direction. We search in the
		// direction who minimize the second order approximation given
		// by the Taylor series which give
		//   d_k = - H_k^{-1} g_k
		// But computing the inverse of the hessian is intractable so
		// the l-bfgs only approximate it's diagonal. The exact
		// computation is well described in [1, pp 779].
		// The only special thing for owl-qn here is to use the pseudo
		// gradient instead of the true one.
		for (size_t f = 0; f < F; f++)
			d[f] = l1 ? -pg[f] : -g[f];
		if (k != 0) {
			const int km = k % M;
			const int bnd = (k <= M) ? k : M;
			double alpha[M], beta;
			// α_i = ρ_j s_j^T q_{i+1}
			// q_i = q_{i+1} - α_i y_i
			for (int i = bnd; i > 0; i--) {
				const int j = (k - i + M) % M;
				alpha[i - 1] = p[j] * xvm_dot(s[j], d, F);
				xvm_axpy(d, -alpha[i - 1], y[j], d, F);
			}
			// r_0 = H_0 q_0
			//     Scaling is described in [2, pp 515]
			//     for k = 0: H_0 = I
			//     for k > 0: H_0 = I * y_k^T s_k / ||y_k||²
			//                    = I * 1 / ρ_k ||y_k||²
			const double y2 = xvm_dot(y[km], y[km], F);
			const double v = 1.0 / (p[km] * y2);
			for (size_t f = 0; f < F; f++)
				d[f] *= v;
			// β_j     = ρ_j y_j^T r_i
			// r_{i+1} = r_i + s_j (α_i - β_i)
			for (int i = 0; i < bnd; i++) {
				const int j = (k - i - 1 + M) % M;
				beta = p[j] * xvm_dot(y[j], d, F);
				xvm_axpy(d, alpha[i] - beta, s[j], d, F);
			}
		}
		// For owl-qn, we must remain in the same orthant than the
		// pseudo-gradient, so we have to constrain the search
		// direction as described in [3, pp 35(3)]
		//   d^k = π(d^k ; v^k)
		//       = π(d^k ; -◇f(x^k))
		if (l1)
			for (size_t f = 0; f < F; f++)
				if (d[f] * pg[f] >= 0.0)
					d[f] = 0.0;
		// 2nd step: we perform a linesearch in the computed direction,
		// we search a step value that satisfy the constrains using a
		// backtracking algorithm. Much elaborated algorithm can perform
		// better in the general case, but for CRF training, bactracking
		// is very efficient and simple to implement.
		// For quasi-Newton, the natural step is 1.0 so we start with
		// this one and reduce it only if it fail with an exception for
		// the first step where a better guess can be done.
		// We have to keep track of the current point and gradient as we
		// will need to compute the delta between those and the found
		// point, and perhaps need to restore them if linesearch fail.
		memcpy(xp, x, sizeof(double) * F);
		memcpy(gp, g, sizeof(double) * F);
		double gd = l1 ? 0.0 : xvm_dot(g, d, F); // gd = g_k^T d_k
		double stp = 1.0, fi = fx;
		if (k == 0)
			stp = 1.0 / xvm_norm(d, F);
		double sc = 0.5;
		bool err = false;
		for (int ls = 1; !uit_stop; ls++, stp *= sc) {
			// We compute the new point using the current step and
			// search direction
			xvm_axpy(x, stp, d, xp, F);
			// For owl-qn, we have to project back the point in the
			// current orthant [3, pp 35]
			//   x^{k+1} = π(x^k + αp^k ; ξ)
			if (l1) {
				for (size_t f = 0; f < F; f++) {
					double or = xp[f];
					if (or == 0.0)
						or = -pg[f];
					if (x[f] * or <= 0.0)
						x[f] = 0.0;
				}
			}
			// And we ask for the value of the objective function
			// and its gradient and pseudo gradient.
			fx = grd_gradient(mdl, g, pg, grds);
			// Now we check if the step satisfy the conditions. For
			// l-bfgs, we check the classical decrease and curvature
			// known as the Wolfe conditions [2, pp 506]
			//   f(x_k + α_k d_k) ≤ f(x_k) + β' α_k g_k^T d_k
			//   g(x_k + α_k d_k)^T d_k ≥ β g_k^T d_k
			//
			// And for owl-qn we check a variant of the Armijo rule
			// described in [3, pp 36]
			//   f(π(x^k+αp^k;ξ)) ≤ f(x^k) - γv^T[π(x^k+αp^k;ξ)-x^k]
			if (!l1) {
				if (fx > fi + stp * gd * 1e-4)
					sc = 0.5;
				else if (xvm_dot(g, d, F) < gd * 0.9)
					sc = 2.1;
				else
					break;
			} else {
				double vp = 0.0;
				for (size_t f = 0; f < F; f++)
					vp += (x[f] - xp[f]) * d[f];
				if (fx < fi + vp * 1e-4)
					break;
			}
			// If we reach the maximum number of linesearsh steps
			// without finding a good one, we just fail.
			if (ls == mdl->opt->lbfgs.maxls) {
				warning("maximum linesearch reached");
				err = true;
				break;
			}
		}
		// If linesearch failed or user interupted training, we return
		// to the last valid point and stop the training. The model is
		// probably not fully optimized but we let the user decide what
		// to do with it.
		if (err || uit_stop) {
			memcpy(x, xp, sizeof(double) * F);
			break;
		}
		if (uit_progress(mdl, k + 1, fx) == false)
			break;
		// 3rd step: We check for convergence and if not, we update the
		// history to prepare the next iteration. The convergence check
		// is quite simple [2, pp 508]
		//   ||g|| / max(1, ||x||) ≤ ε
		// with ε small enough so we stop when numerical precision is
		// reached. For owl-qn we just have to check against the pseudo-
		// gradient instead of the true one.
		const double xn = xvm_norm(x, F);
		const double gn = xvm_norm(l1 ? pg : g, F);
		if (gn / max(xn, 1.0) <= 1e-5)
			break;
		if (k + 1 == K)
			break;
		// So, last we update the history used for approximating the
		// inverse of the diagonal of the hessian
		//   s_k = x_{k+1} - x_k
		//   y_k = g_{k+1} - g_k
		//   ρ_k = 1 / y_k^T s_k
		const int kn = (k + 1) % M;
		for (size_t f = 0; f < F; f++) {
			s[kn][f] = x[f] - xp[f];
			y[kn][f] = g[f] - gp[f];
		}
		p[kn] = 1.0 / xvm_dot(y[kn], s[kn], F);
	}
	// Cleanup: This is very simple as we have carefully allocated memory in
	// a sigle block, we must not forget to free it.
	free(raw);
	for (size_t w = 0; w < W; w++)
		grd_free(grds[w]);
}

/******************************************************************************
 * The SGD-L1 trainer
 *
 *   Implementation of the stochatic gradient descend with L1 penalty described
 *   in [1] by Tsurukoa et al. This allow to build really sparse models with the
 *   SGD method.
 *
 *   [1] Stochastic gradient descent training for L1-regularized log-linear
 *       models with cumulative penalty, Yoshimasa Tsuruoka and Jun'ichi Tsuji
 *       and Sophia Ananiadou, in Proceedings of the ACL and the 4th IJCNLP of
 *       the AFNLP, pages 477-485, August 2009
 ******************************************************************************/
typedef struct sgd_idx_s {
	size_t *uobs;
	size_t *bobs;
} sgd_idx_t;

/* applypenalty:
 *   This macro is quite ugly as it make a lot of things and use local variables
 *   of the function below. I'm sorry for this but this is allow to not
 *   duplicate the code below. Due to the way unigrams and bigrams observation
 *   are stored we must use this two times. As this macro is dangerous when
 *   called outsize of sgd-l1 we undef it just after.
 *   This function match exactly the APPLYPENALTY function defined in [1] pp 481
 *   and the formula on the middle of the page 480.
 */
#define applypenalty(f) do {                               \
	const double z = w[f];                             \
	if      (z > 0.0) w[f] = max(0.0, z - (u + q[f])); \
	else if (z < 0.0) w[f] = min(0.0, z + (u - q[f])); \
	q[f] += w[f] - z;                                  \
} while (false)

/* trn_sgdl1:
 *   Train the model with the SGD-l1 algorithm described by tsurukoa et al.
 */
static void trn_sgdl1(mdl_t *mdl) {
	const size_t  Y = mdl->nlbl;
	const size_t  O = mdl->nobs;
	const size_t  F = mdl->nftr;
	const int     S = mdl->train->nseq;
	const int     K = mdl->opt->maxiter;
	      double *w = mdl->theta;
	// First we have to build and index who hold, for each sequences, the
	// list of actives observations.
	// The index is a simple table indexed by sequences number. Each entry
	// point to two lists of observations terminated by <none>, one for
	// unigrams obss and one for bigrams obss.
	info("    - Build the index\n");
	sgd_idx_t *idx  = xmalloc(sizeof(sgd_idx_t) * S);
	int       *mark = xmalloc(sizeof(int) * O);
	for (size_t o = 0; o < O; o++)
		mark[o] = -1;
	for (int s = 0; s < S; s++) {
		const seq_t *seq = mdl->train->seq[s];
		// Listing active observations in sequence is easy, we scan
		// unigrams and bigrams observations list and mark the actives
		// one in the <mark> array with the sequence number. Next we
		// can scan this array to search the marked obss.
		for (int t = 0; t < seq->len; t++) {
			const pos_t *pos = &seq->pos[t];
			for (size_t p = 0; p < pos->ucnt; p++)
				mark[pos->uobs[p]] = s;
			for (size_t p = 0; p < pos->bcnt; p++)
				mark[pos->bobs[p]] = s;
		}
		// We scan the <mark> array a first time to count the number of
		// active sequences and allocate memory.
		size_t ucnt = 1, bcnt = 1;
		for (size_t o = 0; o < O; o++) {
			ucnt += (mark[o] == s) && (mdl->kind[o] & 1);
			bcnt += (mark[o] == s) && (mdl->kind[o] & 2);
		}
		idx[s].uobs = xmalloc(sizeof(size_t) * ucnt);
		idx[s].bobs = xmalloc(sizeof(size_t) * bcnt);
		// And a second time to fill the allocated array without
		// forgetting to set the end marker.
		size_t upos = 0, bpos = 0;
		for (size_t o = 0; o < O; o++) {
			if ((mark[o] == s) && (mdl->kind[o] & 1))
				idx[s].uobs[upos++] = o;
			if ((mark[o] == s) && (mdl->kind[o] & 2))
				idx[s].bobs[bpos++] = o;
		}
		idx[s].uobs[upos] = none;
		idx[s].bobs[bpos] = none;
	}
	free(mark);
	info("      Done\n");
	// We will process sequences in random order in each iteration, so we
	// will have to permute them. The current permutation is stored in a
	// vector called <perm> shuffled at the start of each iteration. We
	// just initialize it with the identity permutation.
	// As we use the same gradient function than the other trainers, we need
	// an array to store it. These functions accumulate the gradient so we
	// need to clear it at start and before each new computation. As we now
	// which features are active and so which gradient cell are updated, we
	// can clear them selectively instead of fully clear the gradient each
	// time.
	// We also need an aditional vector named <q> who hold the penalty
	// already applied to each features.
	int *perm = xmalloc(sizeof(int) * S);
	for (int s = 0; s < S; s++)
		perm[s] = s;
	double *g = xmalloc(sizeof(double) * F);
	double *q = xmalloc(sizeof(double) * F);
	for (size_t f = 0; f < F; f++)
		g[f] = q[f] = 0.0;
	// We can now start training the model, we perform the requested number
	// of iteration, each of these going through all the sequences. For
	// computing the decay, we will need to keep track of the number of
	// already processed sequences, this is tracked by the <i> variable.
	double u = 0.0;
	grd_t *grd = grd_new(mdl, g);
	for (int k = 0, i = 0; k < K && !uit_stop; k++) {
		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (int s = 0; s < S; s++) {
			const int a = rand() % S;
			const int b = rand() % S;
			const int t = perm[a];
			perm[a] = perm[b];
			perm[b] = t;
		}
		// And so, we can process sequence in a random order
		for (int sp = 0; sp < S && !uit_stop; sp++, i++) {
			const int s = perm[sp];
			const seq_t *seq = mdl->train->seq[s];
			grd_doseq(grd, seq);
			// Before applying the gradient, we have to compute the
			// learning rate to apply to this sequence. For this we
			// use an exponential decay [1, pp 481(5)]
			//   η_i = η_0 * α^{i/S}
			// And at the same time, we update the total penalty
			// that must have been applied to each features.
			//   u <- u + η * rho1 / S
			const double n0    = mdl->opt->sgdl1.eta0;
			const double alpha = mdl->opt->sgdl1.alpha;
			const double nk = n0 * pow(alpha, (double)i / S);
			u = u + nk * mdl->opt->rho1 / S;
			// Now we apply the update to all unigrams and bigrams
			// observations actives in the current sequence. We must
			// not forget to clear the gradient for the next
			// sequence.
			for (size_t n = 0; idx[s].uobs[n] != none; n++) {
				size_t f = mdl->uoff[idx[s].uobs[n]];
				for (size_t y = 0; y < Y; y++, f++) {
					w[f] -= nk * g[f];
					applypenalty(f);
					g[f] = 0.0;
				}
			}
			for (size_t n = 0; idx[s].bobs[n] != none; n++) {
				size_t f = mdl->boff[idx[s].bobs[n]];
				for (size_t d = 0; d < Y * Y; d++, f++) {
					w[f] -= nk * g[f];
					applypenalty(f);
					g[f] = 0.0;
				}
			}
		}
		if (uit_stop)
			break;
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	grd_free(grd);
	// Cleanup allocated memory before returning
	for (int s = 0; s < S; s++) {
		free(idx[s].uobs);
		free(idx[s].bobs);
	}
	free(perm);
	free(g);
	free(q);
}
#undef applypenalty

/******************************************************************************
 * Blockwise Coordinates descent trainer
 *   The gradient and hessian computation used for the BCD is very similar to
 *   the generic one define below but there is some important differences:
 *     - The forward and backward recursions doesn't have to be performed fully
 *       but just in the range of activity of the considered block. So if the
 *       block is active only at position t, the alpha recusion is done from 1
 *       to t and the beta one from T to t, dividing the amount of computations
 *       by 2.
 *     - Samely the update of the gradient and hessian have to be done only at
 *       position where the block is active, so in the common case where the
 *       block is active only once in the sequence, the improvement can be huge.
 *     - And finally, there is no need to compute the logloss, which can take a
 *       long time due to the computation of the log()s.
 ******************************************************************************/
typedef struct bcd_s bcd_t;
struct bcd_s {
	double *ugrd;    //  [Y]
	double *uhes;    //  [Y]
	double *bgrd;    //  [Y][Y]
	double *bhes;    //  [Y][Y]
	size_t *actpos;  //  [T]
	size_t  actcnt;
	grd_t  *grd;
};

/* bcd_soft:
 *   The softmax function.
 */
static double bcd_soft(double z, double r) {
	if (z >  r) return z - r;
	if (z < -r) return z + r;
	return 0.0;
}

/* bcd_actpos:
 *   List position where the given block is active in the sequence and setup the
 *   limits for the fwd/bwd.
 */
static void bcd_actpos(mdl_t *mdl, bcd_t *bcd, const seq_t *seq, size_t o) {
	const int T = seq->len;
	size_t *actpos = bcd->actpos;
	size_t  actcnt = 0;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		bool ok = false;
		if (mdl->kind[o] & 1)
			for (size_t n = 0; !ok && n < pos->ucnt; n++)
				if (pos->uobs[n] == o)
					ok = true;
		if (mdl->kind[o] & 2)
			for (size_t n = 0; !ok && n < pos->bcnt; n++)
				if (pos->bobs[n] == o)
					ok = true;
		if (!ok)
			continue;
		actpos[actcnt++] = t;
	}
	assert(actcnt != 0);
	bcd->actcnt = actcnt;
	bcd->grd->first = actpos[0];
	bcd->grd->last  = actpos[actcnt - 1];
}

/* bct_flgradhes:
 *   Update the gradient and hessian for <blk> on sequence <seq>. This one is
 *   very similar than the trn_spupgrad function but does the computation only
 *   at active pos and approximate also the hessian.
 */
static void bcd_flgradhes(mdl_t *mdl, bcd_t *bcd, const seq_t *seq, size_t o) {
	const grd_t *grd = bcd->grd;
	const size_t Y = mdl->nlbl;
	const size_t T = seq->len;
	const double (*psi  )[T][Y][Y] = (void *)grd->psi;
	const double (*alpha)[T][Y]    = (void *)grd->alpha;
	const double (*beta )[T][Y]    = (void *)grd->beta;
	const double  *unorm           =         grd->unorm;
	const double  *bnorm           =         grd->bnorm;
	const size_t  *actpos          =         bcd->actpos;
	const size_t   actcnt          =         bcd->actcnt;
	double *ugrd = bcd->ugrd;
	double *uhes = bcd->uhes;
	double *bgrd = bcd->bgrd;
	double *bhes = bcd->bhes;
	// Update the gradient and the hessian but here we sum only on the
	// positions where the block is active for unigrams features
	if (mdl->kind[o] & 1) {
		for (size_t n = 0; n < actcnt; n++) {
			const size_t t = actpos[n];
			for (size_t y = 0; y < Y; y++) {
				const double e = (*alpha)[t][y] * (*beta)[t][y]
				               * unorm[t];
				ugrd[y] += e;
				uhes[y] += e * (1.0 - e);
			}
			const size_t y = seq->pos[t].lbl;
			ugrd[y] -= 1.0;
		}
	}
	if ((mdl->kind[o] & 2) == 0)
		return;
	// for bigrams features
	for (size_t n = 0; n < actcnt; n++) {
		const size_t t = actpos[n];
		if (t == 0)
			continue;
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double e = (*alpha)[t - 1][yp] * (*beta)[t][y]
				         * (*psi)[t][yp][y] * bnorm[t];
				bgrd[d] += e;
				bhes[d] += e * (1.0 - e);
			}
		}
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		bgrd[yp * Y + y] -= 1.0;
	}
}

/* bct_spgradhes:
 *   Update the gradient and hessian for <blk> on sequence <seq>. This one is
 *   very similar than the trn_spupgrad function but does the computation only
 *   at active pos and approximate also the hessian.
 */
static void bcd_spgradhes(mdl_t *mdl, bcd_t *bcd, const seq_t *seq, size_t o) {
	const grd_t *grd = bcd->grd;
	const size_t Y = mdl->nlbl;
	const size_t T = seq->len;
	const double (*psiuni)[T][Y] = (void *)grd->psiuni;
	const double  *psival        =         grd->psi;
	const size_t  *psiyp         =         grd->psiyp;
	const size_t (*psiidx)[T][Y] = (void *)grd->psiidx;
	const size_t  *psioff        =         grd->psioff;
	const double (*alpha)[T][Y]  = (void *)grd->alpha;
	const double (*beta )[T][Y]  = (void *)grd->beta;
	const double  *unorm         =         grd->unorm;
	const double  *bnorm         =         grd->bnorm;
	const size_t  *actpos        =         bcd->actpos;
	const size_t   actcnt        =         bcd->actcnt;
	double *ugrd = bcd->ugrd;
	double *uhes = bcd->uhes;
	double *bgrd = bcd->bgrd;
	double *bhes = bcd->bhes;
	// Update the gradient and the hessian but here we sum only on the
	// positions where the block is active for unigrams features
	if (mdl->kind[o] & 1) {
		for (size_t n = 0; n < actcnt; n++) {
			const size_t t = actpos[n];
			for (size_t y = 0; y < Y; y++) {
				const double e = (*alpha)[t][y] * (*beta)[t][y]
				               * unorm[t];
				ugrd[y] += e;
				uhes[y] += e * (1.0 - e);
			}
			const size_t y = seq->pos[t].lbl;
			ugrd[y] -= 1.0;
		}
	}
	if ((mdl->kind[o] & 2) == 0)
		return;
	// for bigrams features
	for (size_t n = 0; n < actcnt; n++) {
		const size_t t = actpos[n];
		if (t == 0)
			continue;
		// We build the expectation matrix
		double e[Y][Y];
		for (size_t yp = 0; yp < Y; yp++)
			for (size_t y = 0; y < Y; y++)
				e[yp][y] = (*alpha)[t - 1][yp] * (*beta)[t][y]
				         * (*psiuni)[t][y] * bnorm[t];
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < (*psiidx)[t][Y - 1]; ) {
			while (n >= (*psiidx)[t][y])
				y++;
			while (n < (*psiidx)[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				e[yp][y] += e[yp][y] * v;
				n++;
			}
		}
		// And use it
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				bgrd[d] += e[yp][y];
				bhes[d] += e[yp][y] * (1.0 - e[yp][y]);
			}
		}
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		bgrd[yp * Y + y] -= 1.0;
	}
}

/* bct_update:
 *   Update the model with the computed gradient and hessian.
 */
static void bcd_update(mdl_t *mdl, bcd_t *bcd, size_t o) {
	const double  rho1  = mdl->opt->rho1;
	const double  rho2  = mdl->opt->rho2;
	const double  kappa = mdl->opt->bcd.kappa;
	const size_t  Y     = mdl->nlbl;
	const double *ugrd  = bcd->ugrd;
	const double *bgrd  = bcd->bgrd;
	      double *uhes  = bcd->uhes;
	      double *bhes  = bcd->bhes;
	if (mdl->kind[o] & 1) {
		// Adjust the hessian
		double a = 1.0;
		for (size_t y = 0; y < Y; y++)
			a = max(a, fabs(ugrd[y] / uhes[y]));
		xvm_scale(uhes, uhes, a * kappa, Y);
		// Update the model
		double *w = mdl->theta + mdl->uoff[o];
		for (size_t y = 0; y < Y; y++) {
			double z = uhes[y] * w[y] - ugrd[y];
			double d = uhes[y] + rho2;
			w[y] = bcd_soft(z, rho1) / d;
		}
	}
	if (mdl->kind[o] & 2) {
		// Adjust the hessian
		double a = 1.0;
		for (size_t i = 0; i < Y * Y; i++)
			a = max(a, fabs(bgrd[i] / bhes[i]));
		xvm_scale(bhes, bhes, a * kappa, Y * Y);
		// Update the model
		double *bw = mdl->theta + mdl->boff[o];
		for (size_t i = 0; i < Y * Y; i++) {
			double z = bhes[i] * bw[i] - bgrd[i];
			double d = bhes[i] + rho2;
			bw[i] = bcd_soft(z, rho1) / d;
		}
	}
}

/* trn_bcd
 *   Train the model using the blockwise coordinates descend method.
 */
static void trn_bcd(mdl_t *mdl) {
	const size_t Y = mdl->nlbl;
	const size_t O = mdl->nobs;
	const size_t T = mdl->train->mlen;
	const size_t S = mdl->train->nseq;
	const int    K = mdl->opt->maxiter;
	// Build the index:
	//   Count active sequences per blocks
	info("    - Build the index\n");
	info("        1/2 -- scan the sequences\n");
	size_t tot = 0, cnt[O], lcl[O];
	for (size_t o = 0; o < O; o++)
		cnt[o] = 0, lcl[o] = none;
	for (size_t s = 0; s < S; s++) {
		// List actives blocks
		const seq_t *seq = mdl->train->seq[s];
		for (int t = 0; t < seq->len; t++) {
			for (size_t b = 0; b < seq->pos[t].ucnt; b++)
				lcl[seq->pos[t].uobs[b]] = s;
			for (size_t b = 0; b < seq->pos[t].bcnt; b++)
				lcl[seq->pos[t].bobs[b]] = s;
		}
		// Updates blocks count
		for (size_t o = 0; o < O; o++)
			cnt[o] += (lcl[o] == s);
	}
	for (size_t o = 0; o < O; o++)
		tot += cnt[o];
	// Allocate memory
	size_t  *idx_cnt = xmalloc(sizeof(size_t  ) * O);
	size_t **idx_lst = xmalloc(sizeof(size_t *) * O);
	for (size_t o = 0; o < O; o++) {
		idx_cnt[o] = cnt[o];
		idx_lst[o] = xmalloc(sizeof(size_t) * cnt[o]);
	}
	// Populate the index
	info("        2/2 -- Populate the index\n");
	for (size_t o = 0; o < O; o++)
		cnt[o] = 0, lcl[o] = none;
	for (size_t s = 0; s < S; s++) {
		// List actives blocks
		const seq_t *seq = mdl->train->seq[s];
		for (int t = 0; t < seq->len; t++) {
			for (size_t b = 0; b < seq->pos[t].ucnt; b++)
				lcl[seq->pos[t].uobs[b]] = s;
			for (size_t b = 0; b < seq->pos[t].bcnt; b++)
				lcl[seq->pos[t].bobs[b]] = s;
		}
		// Build index
		for (size_t o = 0; o < O; o++)
			if (lcl[o] == s)
				idx_lst[o][cnt[o]++] = s;
	}
	info("      Done\n");
	// Allocate the specific trainer of BCD
	bcd_t *bcd = xmalloc(sizeof(bcd_t));
	bcd->ugrd   = xmalloc(sizeof(double) * Y);
	bcd->uhes   = xmalloc(sizeof(double) * Y);
	bcd->bgrd   = xmalloc(sizeof(double) * Y * Y);
	bcd->bhes   = xmalloc(sizeof(double) * Y * Y);
	bcd->actpos = xmalloc(sizeof(size_t) * T);
	bcd->grd    = grd_new(mdl, NULL);
	// And train the model
	for (int i = 0; i < K; i++) {
		for (size_t o = 0; o < O; o++) {
			// Clear the gradient and the hessian
			for (size_t y = 0, d = 0; y < Y; y++) {
				bcd->ugrd[y] = 0.0;
				bcd->uhes[y] = 0.0;
				for (size_t yp = 0; yp < Y; yp++, d++) {
					bcd->bgrd[d] = 0.0;
					bcd->bhes[d] = 0.0;
				}
			}
			// Process active sequences
			for (size_t s = 0; s < idx_cnt[o]; s++) {
				const size_t id = idx_lst[o][s];
				const seq_t *seq = mdl->train->seq[id];
				bcd_actpos(mdl, bcd, seq, o);
				if (mdl->opt->sparse) {
					grd_spdopsi(bcd->grd, seq);
					grd_spfwdbwd(bcd->grd, seq);
					bcd_spgradhes(mdl, bcd, seq, o);
				} else {
					grd_fldopsi(bcd->grd, seq);
					grd_flfwdbwd(bcd->grd, seq);
					bcd_flgradhes(mdl, bcd, seq, o);
				}
			}
			// And update the model
			bcd_update(mdl, bcd, o);
		}
		if (!uit_progress(mdl, i + 1, -1.0))
			break;
	}
	// Cleanup memory
	grd_free(bcd->grd);
	free(bcd->ugrd); free(bcd->uhes);
	free(bcd->bgrd); free(bcd->bhes);
	free(bcd->actpos);
	free(bcd);
	for (size_t o = 0; o < O; o++)
		free(idx_lst[o]);
	free(idx_lst);
	free(idx_cnt);
}

/*******************************************************************************
 * Training
 ******************************************************************************/
static const struct {
	char *name;
	void (* train)(mdl_t *mdl);
} trn_lst[] = {
	{"l-bfgs", trn_lbfgs},
	{"sgd-l1", trn_sgdl1},
	{"bcd",    trn_bcd  }
};
static const int trn_cnt = sizeof(trn_lst) / sizeof(trn_lst[0]);

static void dotrain(mdl_t *mdl) {
	// Check if the user requested the trainer list. If this is not the
	// case, search the trainer.
	if (!strcmp(mdl->opt->algo, "list")) {
		info("Available training algorithms:\n");
		for (int i = 0; i < trn_cnt; i++)
			info("\t%s\n", trn_lst[i].name);
		exit(EXIT_SUCCESS);
	}
	int trn;
	for (trn = 0; trn < trn_cnt; trn++)
		if (!strcmp(mdl->opt->algo, trn_lst[trn].name))
			break;
	if (trn == trn_cnt)
		fatal("unknown algorithm '%s'", mdl->opt->algo);
	// Load a previous model to train again if specified by the user.
	if (mdl->opt->pattern == NULL && mdl->opt->model == NULL)
		fatal("you must specify either a pattern or a model");
	if (mdl->opt->model != NULL) {
		info("* Load previous model\n");
		FILE *file = fopen(mdl->opt->model, "r");
		if (file == NULL)
			pfatal("cannot open input model file");
		mdl_load(mdl, file);
	}
	// Load the pattern file. This is mandatory if no models was loaded as
	// we need some patterns to load data files. This will unlock the
	// database if previously locked by loading a model.
	if (mdl->opt->pattern != NULL) {
		info("* Load patterns\n");
		FILE *file = fopen(mdl->opt->pattern, "r");
		if (file == NULL)
			pfatal("cannot open pattern file");
		rdr_loadpat(mdl->reader, file);
		fclose(file);
		qrk_lock(mdl->reader->obs, false);
	}
	if (mdl->reader->npats == 0)
		fatal("no patterns, cannot load input data");
	// Load the training data. When this is done we lock the quarks as we
	// don't want to put in the model, informations present only in the
	// devlopment set.
	info("* Load training data\n");
	FILE *file = stdin;
	if (mdl->opt->input != NULL) {
		file = fopen(mdl->opt->input, "r");
		if (file == NULL)
			pfatal("cannot open input data file");
	}
	mdl->train = rdr_readdat(mdl->reader, file, true);
	if (mdl->opt->input != NULL)
		fclose(file);
	qrk_lock(mdl->reader->lbl, true);
	qrk_lock(mdl->reader->obs, true);
	if (mdl->train == NULL || mdl->train->nseq == 0)
		fatal("no train data loaded");
	// If present, load the development set in the model. If not specified,
	// the training dataset will be used instead.
	if (mdl->opt->devel != NULL) {
		info("* Load development data\n");
		FILE *file = fopen(mdl->opt->devel, "r");
		if (file == NULL)
			pfatal("cannot open development file");
		mdl->devel = rdr_readdat(mdl->reader, file, true);
		fclose(file);
	}
	// Initialize the model. If a previous model was loaded, this will be
	// just a resync, else the model structure will be created.
	if (mdl->theta == NULL)
		info("* Initialize the model\n");
	else
		info("* Resync the model\n");
	mdl_sync(mdl);
	// Display some statistics as we all love this.
	info("* Summary\n");
	info("    nb train:    %d\n", mdl->train->nseq);
	if (mdl->devel != NULL)
		info("    nb devel:    %d\n", mdl->devel->nseq);
	info("    nb labels:   %zu\n", mdl->nlbl);
	info("    nb blocks:   %zu\n", mdl->nobs);
	info("    nb features: %zu\n", mdl->nftr);
	// And train the model...
	info("* Train the model with %s\n", mdl->opt->algo);
	uit_setup(mdl);
	trn_lst[trn].train(mdl);
	uit_cleanup(mdl);
	// If requested compact the model.
	if (mdl->opt->compact) {
		const size_t O = mdl->nobs;
		const size_t F = mdl->nftr;
		info("* Compacting the model\n");
		mdl_compact(mdl);
		info("    %8zu observations removed\n", O - mdl->nobs);
		info("    %8zu features removed\n", F - mdl->nftr);
	}
	// And save the trained model
	info("* Save the model\n");
	file = stdout;
	if (mdl->opt->output != NULL) {
		file = fopen(mdl->opt->output, "w");
		if (file == NULL)
			pfatal("cannot open output model");
	}
	mdl_save(mdl, file);
	if (mdl->opt->output != NULL)
		fclose(file);
	info("* Done\n");
}

/*******************************************************************************
 * Labeling
 ******************************************************************************/
static void dolabel(mdl_t *mdl) {
	// First, load the model provided by the user. This is mandatory to
	// label new datas ;-)
	if (mdl->opt->model == NULL)
		fatal("you must specify a model");
	info("* Load model\n");
	FILE *file = fopen(mdl->opt->model, "r");
	if (file == NULL)
		pfatal("cannot open input model file");
	mdl_load(mdl, file);
	// Open input and output files
	FILE *fin = stdin, *fout = stdout;
	if (mdl->opt->input != NULL) {
		fin = fopen(mdl->opt->input, "r");
		if (fin == NULL)
			pfatal("cannot open input data file");
	}
	if (mdl->opt->output != NULL) {
		fout = fopen(mdl->opt->output, "w");
		if (fout == NULL)
			pfatal("cannot open output data file");
	}
	// Do the labelling
	info("* Label sequences\n");
	tag_label(mdl, fin, fout);
	info("* Done\n");
	// And close files
	if (mdl->opt->input != NULL)
		fclose(fin);
	if (mdl->opt->output != NULL)
		fclose(fout);
}

/*******************************************************************************
 * Dumping
 ******************************************************************************/
static void dodump(mdl_t *mdl) {
	// Load input model file
	info("* Load model\n");
	FILE *fin = stdin;
	if (mdl->opt->input != NULL) {
		fin = fopen(mdl->opt->input, "r");
		if (fin == NULL)
			pfatal("cannot open input data file");
	}
	mdl_load(mdl, fin);
	if (mdl->opt->input != NULL)
		fclose(fin);
	// Open output file
	FILE *fout = stdout;
	if (mdl->opt->output != NULL) {
		fout = fopen(mdl->opt->output, "w");
		if (fout == NULL)
			pfatal("cannot open output data file");
	}
	// Dump model
	info("* Dump model\n"); 
	const size_t Y = mdl->nlbl;
	const size_t O = mdl->nobs;
	const qrk_t *Qlbl = mdl->reader->lbl;
	const qrk_t *Qobs = mdl->reader->obs;
	for (size_t o = 0; o < O; o++) {
		const char *obs = qrk_id2str(Qobs, o);
		bool empty = true;
		if (mdl->kind[o] & 1) {
			const double *w = mdl->theta + mdl->uoff[o];
			for (size_t y = 0; y < Y; y++) {
				if (w[y] == 0.0)
					continue;
				const char *ly = qrk_id2str(Qlbl, y);
				fprintf(fout, "%s\t#\t%s\t%f\n", obs, ly, w[y]);
				empty = false;
			}
		}
		if (mdl->kind[o] & 2) {
			const double *w = mdl->theta + mdl->boff[o];
			for (size_t d = 0; d < Y * Y; d++) {
				if (w[d] == 0.0)
					continue;
				const char *ly  = qrk_id2str(Qlbl, d % Y);
				const char *lyp = qrk_id2str(Qlbl, d / Y);
				fprintf(fout, "%s\t%s\t%s\t%f\n", obs, lyp, ly,
				       w[d]);
				empty = false;
			}
		}
		if (!empty)
			fprintf(fout, "\n");
	}
	if (mdl->opt->output != NULL)
		fclose(fout);
}

/*******************************************************************************
 * Entry point
 ******************************************************************************/
int main(int argc, char *argv[argc]) {
	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Wapiti stack usage is quite intensive. This simplify a lot the memory
	// mangement and code but, if you have long sequences or huge label set,
	// the stack can overflow. In effect, some OS provide only very small
	// stacks by default.
	// For the L-BFGS trainer, this is not a problem as the computations are
	// done in independant threads and we can adjust their stack easily, but
	// for SGD-L1 and the tagger, this is not the case.
	// I don't known a really portable way to increase the main stack so I
	// will have to move these in workers threads also but this need some
	// thinking.
	// As a quick hack this small code will work on all unix of my knowledge
	// but is not really POSIX compliant and I don't known if it work with
	// cygwin on windows. This is truly a hack as it just raise the soft
	// stack limit to match the hard stack limit without any checking than
	// this will be enough.
	struct rlimit rlp;
	if (getrlimit(RLIMIT_STACK, &rlp) != 0)
		pfatal("cannot get stack size");
	rlp.rlim_cur = rlp.rlim_max;
	if (setrlimit(RLIMIT_STACK, &rlp) != 0)
		pfatal("cannot set stack size");
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	// We first parse command line switchs
	opt_t opt = opt_defaults;
	opt_parse(argc, argv, &opt);
	// Next we prepare the model
	mdl_t *mdl = mdl_new(rdr_new());
	mdl->opt = &opt;
	// And switch to requested mode
	switch (opt.mode) {
		case 0: dotrain(mdl); break;
		case 1: dolabel(mdl); break;
		case 2: dodump(mdl); break;
	}
	// And cleanup
	mdl_free(mdl);
	return EXIT_SUCCESS;
}

