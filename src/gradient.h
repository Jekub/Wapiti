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

#ifndef gradient_h
#define gradient_h

#include "wapiti.h"
#include "model.h"
#include "sequence.h"

/* grd_st_t:
 *   State tracker for the gradient computation. To compute the gradient we need
 *   to perform several steps and communicate between them a lot of intermediate
 *   values, all these temporary are stored in this object.
 *   A tracker can be used to compute sequence of length <len> at most, before
 *   using it you must call grd_stcheck to ensure that the tracker is big enough
 *   for your sequence.
 *   This tracker is used to perform single sample gradient computations or
 *   partial gradient computation in online algorithms and for decoding with
 *   posteriors.
 */
typedef struct grd_st_s grd_st_t;
struct grd_st_s {
	mdl_t    *mdl;
	uint32_t len;     // =T        max length of sequence
	double   *g;       // [F]       vector where to put gradient updates
	double    lloss;   //           loss value for the sequence
	double   *psi;     // [T][Y][Y] the transitions scores
	double   *psiuni;  // [T][Y]    | Same as psi in sparse format
	uint32_t *psiyp;   // [T][Y][Y] |
	uint32_t *psiidx;  // [T][Y]    |
	uint32_t *psioff;  // [T]
	double   *alpha;   // [T][Y]    forward scores
	double   *beta;    // [T][Y]    backward scores
	double   *scale;   // [T]       scaling factors of forward scores
	double   *unorm;   // [T]       normalization factors for unigrams
	double   *bnorm;   // [T]       normalization factors for bigrams
	uint32_t  first;   //           first position where gradient is needed
	uint32_t  last;    //           last position where gradient is needed
};

grd_st_t *grd_stnew(mdl_t *mdl, double *g);
void grd_stfree(grd_st_t *grd_st);
void grd_stcheck(grd_st_t *grd_st, uint32_t len);

void grd_fldopsi(grd_st_t *grd_st, const seq_t *seq);
void grd_flfwdbwd(grd_st_t *grd_st, const seq_t *seq);
void grd_flupgrad(grd_st_t *grd_st, const seq_t *seq);

void grd_spdopsi(grd_st_t *grd_st, const seq_t *seq);
void grd_spfwdbwd(grd_st_t *grd_st, const seq_t *seq);
void grd_spupgrad(grd_st_t *grd_st, const seq_t *seq);

void grd_logloss(grd_st_t *grd_st, const seq_t *seq);

void grd_dospl(grd_st_t *grd_st, const seq_t *seq);

/* grd_t:
 *   Multi-threaded full dataset gradient computer. This is used to compute the
 *   gradient by algorithm working on the full dataset at each iterations. It
 *   efficiently compute it using the fact it is additive to use as many threads
 *   as allowed.
 */
typedef struct grd_s grd_t;
struct grd_s {
	mdl_t     *mdl;
	grd_st_t **grd_st;
};

grd_t *grd_new(mdl_t *mdl, double *g);
void   grd_free(grd_t *grd);
double grd_gradient(grd_t *grd);

#endif

