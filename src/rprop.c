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
#include <float.h>
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

#define EPSILON (DBL_EPSILON * 64.0)

#define sign(v) ((v) < -EPSILON ? -1.0 : ((v) > EPSILON ? 1.0 : 0.0))
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
	mdl_t  *mdl;
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
static void trn_rpropsub(job_t *job, uint32_t id, uint32_t cnt, rprop_t *st) {
	unused(job);
	mdl_t *mdl = st->mdl;
	const uint64_t F = mdl->nftr;
	const double   stpmin = mdl->opt->rprop.stpmin;
	const double   stpmax = mdl->opt->rprop.stpmax;
	const double   stpinc = mdl->opt->rprop.stpinc;
	const double   stpdec = mdl->opt->rprop.stpdec;
	const bool     wbt    = strcmp(mdl->opt->algo, "rprop-");
	const double   rho1   = mdl->opt->rho1;
	const int      l1     = (rho1 != 0.0) ? mdl->opt->rprop.cutoff + 1: 0;
	double *x = mdl->theta;
	double *xp  = st->xp,   *stp = st->stp;
	double *g   = st->g,    *gp  = st->gp;
	const uint64_t from = F * id / cnt;
	const uint64_t to   = F * (id + 1) / cnt;
	for (uint64_t f = from; f < to; f++) {
		double pg = g[f];
		// If there is a l1 component in the regularization component,
		// we either project the gradient in the current orthant or
		// check for cutdown depending on the projection scheme wanted.
		if (l1 == 1) {
			     if (x[f] < -EPSILON) pg -= rho1;
			else if (x[f] >  EPSILON) pg += rho1;
			else if (g[f] < -rho1)    pg += rho1;
			else if (g[f] >  rho1)    pg -= rho1;
			else                      pg  = 0.0;
		} else if (l1 && sqr(g[f] + rho1 * sign(x[f])) < sqr(rho1)) {
			if (x[f] == 0.0 || (   gp[f] * g[f] < 0.0
			                    && xp[f] * x[f] < 0.0)) {
				if (wbt)
					xp[f] = x[f];
				x[f]  = 0.0;
				gp[f] = g[f];
				continue;
			}
		}
		const double sgp = sign(gp[f]);
		const double spg = sign(pg);
		// Next we adjust the step depending of the new and
		// previous gradient values.
		if (sgp * spg > 0.0)
			stp[f] = min(stp[f] * stpinc, stpmax);
		else if (sgp * spg < 0.0)
			stp[f] = max(stp[f] * stpdec, stpmin);
		// Finally update the weight. if there is l1 penalty
		// and the pseudo gradient projection is used, we have to
		// project back the update in the choosen orthant.
		if (!wbt || sgp * spg > 0.0) {
			double dlt = stp[f] * -sign(g[f]);
			if (l1 == 1 && dlt * spg >= 0.0)
				dlt = 0.0;
			if (wbt)
				xp[f] = x[f];
			x[f] += dlt;
		} else if (sgp * spg < -0.0) {
			x[f] = xp[f];
			g[f] = 0.0;
		} else {
			xp[f] = x[f];
			if (l1 != 1)
				x[f] += stp[f] * -spg;
		}
		gp[f] = g[f];
	}
}

void trn_rprop(mdl_t *mdl) {
	const uint64_t F   = mdl->nftr;
	const uint32_t K   = mdl->opt->maxiter;
	const uint32_t W   = mdl->opt->nthread;
	const bool     wbt = strcmp(mdl->opt->algo, "rprop-");
	const int      cut = mdl->opt->rprop.cutoff;
	// Allocate state memory and initialize it
	double *xp  = NULL,       *stp = xvm_new(F);
	double *g   = xvm_new(F), *gp  = xvm_new(F);
	if (wbt && !cut)
		xp = xvm_new(F);
	for (uint64_t f = 0; f < F; f++) {
		if (wbt && !cut)
			xp[f]  = 0.0;
		gp[f]  = 0.0;
		stp[f] = 0.1;
	}
	// Restore a saved state if given by the user
	if (mdl->opt->rstate != NULL) {
		const char *err = "invalid state file";
		FILE *file = fopen(mdl->opt->rstate, "r");
		if (file == NULL)
			fatal("failed to open input state file");
		int type;
		uint64_t nftr;
		if (fscanf(file, "#state#%d#%"SCNu64"\n", &type, &nftr) != 2)
			fatal(err);
		if (type != 3)
			fatal("state is not for rprop model");
		for (uint64_t i = 0; i < nftr; i++) {
			uint64_t f;
			double vxp, vstp, vgp;
			if (fscanf(file, "%"PRIu64" %la %la %la\n", &f, &vxp,
					&vstp, &vgp) != 4)
				fatal(err);
			if (wbt && !cut) xp[f] = vxp;
			gp[f] = vgp;
			stp[f] = vstp;
		}
		fclose(file);
	}
	// Prepare the rprop state used to send information to the rprop worker
	// about updating weight using the gradient.
	rprop_t *st = xmalloc(sizeof(rprop_t));
	st->mdl = mdl;
	st->xp  = xp;  st->stp = stp;
	st->g   = g;   st->gp  = gp;
	rprop_t *rprop[W];
	for (uint32_t w = 0; w < W; w++)
		rprop[w] = st;
	// Prepare the gradient state for the distributed gradient computation.
	grd_t *grd = grd_new(mdl, g);
	// And iterate the gradient computation / weight update process until
	// convergence or stop request
	for (uint32_t k = 0; !uit_stop && k < K; k++) {
		double fx = grd_gradient(grd);
		if (uit_stop)
			break;
		mth_spawn((func_t *)trn_rpropsub, W, (void **)rprop, 0, 0);
		if (uit_progress(mdl, k + 1, fx) == false)
			break;
	}
	// Save state if user requested it
	if (mdl->opt->sstate != NULL) {
		FILE *file = fopen(mdl->opt->sstate, "w");
		if (file == NULL)
			fatal("failed to open output state file");
		fprintf(file, "#state#3#%"PRIu64"\n", F);
		for (uint64_t f = 0; f < F; f++) {
			double vxp = xp != NULL ? xp[f] : 0.0;
			double vstp = stp[f], vgp = gp[f];
			fprintf(file, "%"PRIu64" ", f);
			fprintf(file, "%la %la %la\n", vxp, vstp, vgp);
		}
		fclose(file);
	}
	// Free all allocated memory
	if (wbt && !cut)
		xvm_free(xp);
	xvm_free(g);
	xvm_free(gp);
	xvm_free(stp);
	grd_free(grd);
	free(st);
}

