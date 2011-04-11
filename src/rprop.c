/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2011  CNRS
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
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
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

#define sign(v) ((v) < 0.0 ? -1.0 : ((v) > 0.0 ? 1.0 : 0.0))
#define sqr(v)  ((v) * (v))

/******************************************************************************
 * Resilient propagation optimizer
 *
 *   This is an implementation of the RPROP algorithm (resilient propagation)
 *   described by Riedmiller and Braun in [1] with an adaptation to be useable
 *   with l1 regularization.
 *   The adaptation consist of using a pseudo-gradient similar to the one used
 *   in OWL-QN to choose an orthant at iterations steps and projecting the step
 *   in this orthant before the weight update.
 *
 *   [1] A direct adaptive method for faster backpropagation learning: The RPROP
 *       algorithm, Martin Riedmiller and Heinrich Braun, IEEE International
 *       Conference on Neural Networks, San Francisco, USA, 586-591, March 1993.
 ******************************************************************************/
typedef struct rprop_s rprop_t;
struct rprop_s {
	mdl_t *mdl;
	double *xp;
	double *stp;
	double *g;
	double *gp;
};

/* trn_rpropsub:
 *   Partial update of the weight vector including partial gradient in case of
 *   l1 regularisation. The sub vector updated depend on the id and cnt
 *   parameter given, the job scheduling system is not used here as we can
 *   easily split processing in equals parts.
 */
static void trn_rpropsub(job_t *job, int id, int cnt, rprop_t *st) {
	unused(job);
	mdl_t *mdl = st->mdl;
	const size_t F = mdl->nftr;
	const double stpmin = mdl->opt->rprop.stpmin;
	const double stpmax = mdl->opt->rprop.stpmax;
	const double stpinc = mdl->opt->rprop.stpinc;
	const double stpdec = mdl->opt->rprop.stpdec;
	const bool   wbt    = strcmp(mdl->opt->algo, "rprop-");
	const double rho1   = mdl->opt->rho1;
	const bool   l1     = rho1 != 0.0;
	double *x = mdl->theta;
	double *xp  = st->xp,   *stp = st->stp;
	double *g   = st->g,    *gp  = st->gp;
	const size_t from = F * id / cnt;
	const size_t to   = F * (id + 1) / cnt;
	for (size_t f = from; f < to; f++) {
		// If there is a l1 component in the regularization
		// component, we check for cutdown.
		if (l1 && sqr(g[f] + rho1 * sign(x[f])) < sqr(rho1)) {
			if (x[f] == 0.0 || (   gp[f] * g[f] < 0.0
			                    && xp[f] * x[f] < 0.0)) {
				xp[f] = x[f];
				x[f]  = 0.0;
				gp[f] = g[f];
				continue;
			}
		}
		// Next we adjust the step depending of the new and
		// previous gradient values and update the weight.
		if (gp[f] * g[f] > 0.0)
			stp[f] = min(stp[f] * stpinc, stpmax);
		else if (gp[f] * g[f] < 0.0)
			stp[f] = max(stp[f] * stpdec, stpmin);
		if (!wbt || gp[f] * g[f] > 0.0) {
			xp[f]  = x[f];
			x[f]  += stp[f] * -sign(g[f]);
		} else if (gp[f] * g[f] < 0.0) {
			const double tmp = xp[f];
			xp[f] = x[f];
			x[f]  = tmp;
			g[f]  = 0.0;
		} else {
			xp[f]  = x[f];
			x[f]  += stp[f] * -sign(g[f]);
		}
		gp[f] = g[f];
	}
}

void trn_rprop(mdl_t *mdl) {
	const size_t F = mdl->nftr;
	const int    K = mdl->opt->maxiter;
	const size_t W = mdl->opt->nthread;
	// Allocate state memory and initialize it
	double *xp  = xvm_new(F), *stp = xvm_new(F);
	double *g   = xvm_new(F), *gp  = xvm_new(F);
	for (unsigned f = 0; f < F; f++) {
		xp[f]  = 0.0;
		gp[f]  = 0.0;
		stp[f] = 0.1;
	}
	// Prepare the rprop state used to send information to the rprop worker
	// about updating weight using the gradient.
	rprop_t *st = xmalloc(sizeof(rprop_t));
	st->mdl = mdl;
	st->xp  = xp;  st->stp = stp;
	st->g   = g;   st->gp  = gp;
	rprop_t *rprop[W];
	for (size_t w = 0; w < W; w++)
		rprop[w] = st;
	// Prepare the gradient state for the distributed gradient computation.
	grd_t *grds[W];
	grds[0] = grd_new(mdl, g);
	for (size_t w = 1; w < W; w++)
		grds[w] = grd_new(mdl, xvm_new(F));
	// And iterate the gradient computation / weight update process until
	// convergence or stop request
	for (int k = 0; !uit_stop && k < K; k++) {
		double fx = grd_gradient(mdl, g, grds);
		if (uit_stop)
			break;
		mth_spawn((func_t *)trn_rpropsub, W, (void **)rprop, 0, 0);
		if (uit_progress(mdl, k + 1, fx) == false)
			break;
	}
	// Free all allocated memory
	xvm_free(xp);
	xvm_free(g);
	xvm_free(gp);
	for (size_t w = 1; w < W; w++)
		xvm_free(grds[w]->g);
	for (size_t w = 0; w < W; w++)
		grd_free(grds[w]);
	free(st);
}

