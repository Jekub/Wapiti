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

#ifndef gradient_h
#define gradient_h

#include "model.h"
#include "sequence.h"

typedef struct grd_s grd_t;
struct grd_s {
	mdl_t  *mdl;
	double *g;
	double  lloss;
	double *psi;
	double *psiuni;
	size_t *psiyp;
	size_t *psiidx;
	size_t *psioff;
	double *alpha;
	double *beta;
	double *scale;
	double *unorm;
	double *bnorm;
	int     first;
	int     last;
};

grd_t *grd_new(mdl_t *mdl, double *g);
void grd_free(grd_t *grd);

void grd_fldopsi(grd_t *grd, const seq_t *seq);
void grd_flfwdbwd(grd_t *grd, const seq_t *seq);
void grd_flupgrad(grd_t *grd, const seq_t *seq);

void grd_spdopsi(grd_t *grd, const seq_t *seq);
void grd_spfwdbwd(grd_t *grd, const seq_t *seq);
void grd_spupgrad(grd_t *grd, const seq_t *seq);

void grd_logloss(grd_t *grd, const seq_t *seq);

void grd_doseq(grd_t *grd, const seq_t *seq);
double grd_gradient(mdl_t *mdl, double *g, double *pg, grd_t *grds[]);

#endif

