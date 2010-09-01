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

/******************************************************************************
 * Resilient propagation optimizer
 *
 *   This is an implementation of the RPROP algorithm (resilient propagation)
 *   described by Riedmiller and Braun in [1] with an adaptation to be useable
 *   with l1 regularization.
 *   The adaptation consiste of using a pseudo-gradient similar to the one used
 *   in OWL-QN to choose an orthant at iterations steps and projecting the step
 *   in this orthant before the weight update.
 *
 *   [1] A direct adaptive method for faster backpropagation learning: The RPROP
 *       algorithm, Martin Riedmiller and Heinrich Braun, IEEE International
 *       Conference on Neural Networks, San Francisco, USA, 586-591, March 1993.
 ******************************************************************************/
void trn_rprop(mdl_t *mdl) {
	const size_t F = mdl->nftr;
	const int    K = mdl->opt->maxiter;
	const size_t W = mdl->opt->nthread;
	const double stpmin = mdl->opt->rprop.stpmin;
	const double stpmax = mdl->opt->rprop.stpmax;
	const double stpinc = mdl->opt->rprop.stpinc;
	const double stpdec = mdl->opt->rprop.stpdec;
	const double rho1   = mdl->opt->rho1;
	const bool   l1     = rho1 != 0.0;
	double *x   = mdl->theta;
	double *g   = xvm_new(F), *gp  = xvm_new(F);
	double *stp = xvm_new(F), *dlt = xvm_new(F);
	for (unsigned f = 0; f < F; f++) {
		gp[f]  = 0.0;
		stp[f] = 0.1;
	}
	grd_t *grds[W];
	grds[0] = grd_new(mdl, g);
	for (size_t w = 1; w < W; w++)
		grds[w] = grd_new(mdl, xvm_new(F));
	for (int k = 0; !uit_stop && k < K; k++) {
		double fx = grd_gradient(mdl, g, grds);
		for (unsigned f = 0; f < F; f++) {
			// If there is a l1 component in the regularization
			// component, we project the gradient in the current
			// orthant.
			if (l1) {
				if (x[f] < 0.0)        g[f] -= rho1;
				else if (x[f] > 0.0)   g[f] += rho1;
				else if (g[f] < -rho1) g[f] += rho1;
				else if (g[f] > rho1)  g[f] -= rho1;
				else                   g[f]  = 0.0;
			}
			// Next we adjust the step depending of the new and
			// previous gradient values and update the weight. if
			// there is l1 penalty, we have to project back the
			// update in the choosen orthant.
			if (gp[f] * g[f] > 0.0) {
				stp[f] = min(stp[f] * stpinc, stpmax);
				dlt[f] = stp[f] * -sign(g[f]);
				if (l1 && dlt[f] * g[f] >= 0.0)
					dlt[f] = 0.0;
				x[f] += dlt[f];
			} else if (gp[f] * g[f] < 0.0) {
				stp[f] = max(stp[f] * stpdec, stpmin);
				x[f]   = x[f] - dlt[f];
				g[f]   = 0.0;
			} else {
				dlt[f] = stp[f] * -sign(g[f]);
				if (l1 && dlt[f] * g[f] >= 0.0)
					dlt[f] = 0.0;
				x[f] += dlt[f];
			}
			gp[f] = g[f];
		}
		if (uit_progress(mdl, k + 1, fx) == false)
			break;
	}
	xvm_free(g);   xvm_free(gp);
	xvm_free(stp); xvm_free(dlt);
	for (size_t w = 1; w < W; w++)
		xvm_free(grds[w]->g);
	for (size_t w = 0; w < W; w++)
		grd_free(grds[w]);
}

