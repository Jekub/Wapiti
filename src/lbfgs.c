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
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"

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

void trn_lbfgs(mdl_t *mdl) {
	const uint64_t F  = mdl->nftr;
	const uint32_t K  = mdl->opt->maxiter;
	const uint32_t C  = mdl->opt->objwin;
	const uint32_t M  = mdl->opt->lbfgs.histsz;
	const bool     l1 = mdl->opt->rho1 != 0.0;
	double *x, *xp; // Current and previous value of the variables
	double *g, *gp; // Current and previous value of the gradient
	double *pg;     // The pseudo-gradient (only for owl-qn)
	double *d;      // The search direction
	double *s[M];   // History value s_k = Δ(x,px)
	double *y[M];   // History value y_k = Δ(g,pg)
	double  p[M];   // ρ_k
	double  fh[C];  // f(x) history
	// Initialization: Here, we have to allocate memory on the heap as we
	// cannot request so much memory on the stack as this will have a too
	// big impact on performance and will be refused by the system on non-
	// trivial models.
	x  = mdl->theta;
	xp = xvm_new(F); g = xvm_new(F);
	gp = xvm_new(F); d = xvm_new(F);
	for (uint32_t m = 0; m < M; m++) {
		s[m] = xvm_new(F);
		y[m] = xvm_new(F);
	}
	pg = l1 ? xvm_new(F) : NULL;
	grd_t *grd = grd_new(mdl, g);
	// Restore a saved state if user specified one.
	if (mdl->opt->rstate != NULL) {
		const char *err = "invalid state file";
		FILE *file = fopen(mdl->opt->rstate, "r");
		if (file == NULL)
			fatal("failed to open input state file");
		int type, histsz;
		uint64_t nftr;
		if (fscanf(file, "#state#%d#%d#%"SCNu64"\n", &type, &histsz,
				&nftr) != 3)
			fatal("0 %s", err);
		if (type != 0 || histsz != (int)M)
			fatal("state is not compatible");
		for (uint64_t i = 0; i < nftr; i++) {
			uint64_t f;
			if (fscanf(file, "%"PRIu64, &f) != 1)
				fatal("1 %s", err);
			if (fscanf(file, "%la %la", &xp[f], &gp[f]) != 2)
				fatal("2 %s", err);
			for (uint32_t m = 0; m < M; m++) {
				if (fscanf(file, "%la", &s[m][f]) != 1)
					fatal("3 %s", err);
				if (fscanf(file, "%la", &y[m][f]) != 1)
					fatal("4 %s", err);
			}
		}
		for (uint32_t m = 0; m < M; m++)
			p[m] = 1.0 / xvm_dot(y[m], s[m], F);
		fclose(file);
	}
	// Minimization: This is the heart of the function. (a big heart...) We
	// will perform iterations until one these conditions is reached
	//   - the maximum iteration count is reached
	//   - we have converged (upto numerical precision)
	//   - the report function return false
	//   - an error happen somewhere
	double fx = grd_gradient(grd);
	for (uint32_t k = 0; !uit_stop && k < K; k++) {
		// We first compute the pseudo-gradient of f for owl-qn. It is
		// defined in [3, pp 335(4)]
		//              | ∂_i^- f(x)   if ∂_i^- f(x) > 0
		//   ◇_i f(x) = | ∂_i^+ f(x)   if ∂_i^+ f(x) < 0
		//              | 0            otherwise
		// with
		//   ∂_i^± f(x) = ∂/∂x_i l(x) + | Cσ(x_i) if x_i ≠ 0
		//                              | ±C      if x_i = 0
		if (l1) {
			const double rho1 = mdl->opt->rho1;
			for (uint64_t f = 0; f < F; f++) {
				if (x[f] < 0.0)
					pg[f] = g[f] - rho1;
				else if (x[f] > 0.0)
					pg[f] = g[f] + rho1;
				else if (g[f] < -rho1)
					pg[f] = g[f] + rho1;
				else if (g[f] > rho1)
					pg[f] = g[f] - rho1;
				else
					pg[f] = 0.0;
			}
		}
		// 1st step: We compute the search direction. We search in the
		// direction who minimize the second order approximation given
		// by the Taylor series which give
		//   d_k = - H_k^{-1} g_k
		// But computing the inverse of the hessian is intractable so
		// the l-bfgs only approximate it's diagonal. The exact
		// computation is well described in [1, pp 779].
		// The only special thing for owl-qn here is to use the pseudo
		// gradient instead of the true one.
		xvm_neg(d, l1 ? pg : g, F);
		if (k != 0) {
			const uint32_t km = k % M;
			const uint32_t bnd = (k <= M) ? k : M;
			double alpha[M], beta;
			// α_i = ρ_j s_j^T q_{i+1}
			// q_i = q_{i+1} - α_i y_i
			for (uint32_t i = bnd; i > 0; i--) {
				const uint32_t j = (M + 1 + k - i) % M;
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
			for (uint64_t f = 0; f < F; f++)
				d[f] *= v;
			// β_j     = ρ_j y_j^T r_i
			// r_{i+1} = r_i + s_j (α_i - β_i)
			for (uint32_t i = 0; i < bnd; i++) {
				const uint32_t j = (M + k - i) % M;
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
			for (uint64_t f = 0; f < F; f++)
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
		double sc  = (k == 0) ? 0.1 : 0.5;
		double stp = (k == 0) ? 1.0 / xvm_norm(d, F) : 1.0;
		double gd  = l1 ? 0.0 : xvm_dot(g, d, F); // gd = g_k^T d_k
		double fi  = fx;
		bool err = false;
		for (uint32_t ls = 1; !uit_stop; ls++, stp *= sc) {
			// We compute the new point using the current step and
			// search direction
			xvm_axpy(x, stp, d, xp, F);
			// For owl-qn, we have to project back the point in the
			// current orthant [3, pp 35]
			//   x^{k+1} = π(x^k + αp^k ; ξ)
			if (l1) {
				for (uint64_t f = 0; f < F; f++) {
					double or = xp[f];
					if (or == 0.0)
						or = -pg[f];
					if (x[f] * or <= 0.0)
						x[f] = 0.0;
				}
			}
			// And we ask for the value of the objective function
			// and its gradient.
			fx = grd_gradient(grd);
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
				for (uint64_t f = 0; f < F; f++)
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
		// 3rd step: we update the history used for approximating the
		// inverse of the diagonal of the hessian
		//   s_k = x_{k+1} - x_k
		//   y_k = g_{k+1} - g_k
		//   ρ_k = 1 / y_k^T s_k
		const uint32_t kn = (k + 1) % M;
		xvm_sub(s[kn], x, xp, F);
		xvm_sub(y[kn], g, gp, F);
		p[kn] = 1.0 / xvm_dot(y[kn], s[kn], F);
		// And last, we check for convergence. The convergence check is
		// quite simple [2, pp 508]
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
		// Second stoping criterion tested is a check for improvement of
		// the function value over the past W iteration. When this come
		// under an epsilon, we also stop the minimization.
		fh[k % C] = fx;
		double dlt = 1.0;
		if (k >= C) {
			const double of = fh[(k + 1) % C];
			dlt = fabs(of - fx) / of;
			if (dlt < mdl->opt->stopeps)
				break;
		}
	}
	// Save the optimizer state if requested by the user
	if (mdl->opt->sstate != NULL) {
		FILE *file = fopen(mdl->opt->sstate, "w");
		if (file == NULL)
			fatal("failed to open output state file");
		fprintf(file, "#state#0#%"PRIu32"#%"PRIu64"\n", M, F);
		for (uint64_t f = 0; f < F; f++) {
			fprintf(file, "%"PRIu64, f);
			fprintf(file, " %la %la", xp[f], gp[f]);
			for (uint32_t m = 0; m < M; m++)
				fprintf(file, " %la %la", s[m][f], y[m][f]);
			fprintf(file, "\n");
		}
		fclose(file);
	}
	// Cleanup: We free all the vectors we have allocated.
	xvm_free(xp); xvm_free(g);
	xvm_free(gp); xvm_free(d);
	for (uint32_t m = 0; m < M; m++) {
		xvm_free(s[m]);
		xvm_free(y[m]);
	}
	if (l1)
		xvm_free(pg);
	grd_free(grd);
}

