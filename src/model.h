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

#ifndef model_h
#define model_h

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>

#include "wapiti.h"
#include "options.h"
#include "sequence.h"
#include "reader.h"

typedef struct timeval tms_t;

/* mdl_t:
 *   Represent a linear-chain CRF model. The model contain both unigram and
 *   bigram features. It is caracterized by <nlbl> the number of labels, <nobs>
 *   the number of observations, and <nftr> the number of features.
 *
 *   Each observations have a corresponding entry in <kind> whose first bit is
 *   set if the observation is unigram and second one if it is bigram. Note that
 *   an observation can be both. An unigram observation produce Y features and a
 *   bigram one produce Y * Y features.
 *   The <theta> array keep all features weights. The <*off> array give for each
 *   observations the offset in the <theta> array where the features of the
 *   observation are stored.
 *
 *   The <*off> and <theta> array are initialized only when the model is
 *   synchronized. As you can add new labels and observations after a sync, we
 *   keep track of the old counts in <olbl> and <oblk> to detect inconsistency
 *   and resynchronize the model if needed. In this case, if the number of
 *   labels have not changed, the previously trained weights are kept, else they
 *   are now meaningless so discarded.
 */
typedef struct mdl_s mdl_t;
struct mdl_s {
	opt_t    *opt;     //       options for training
	int       type;    //       model type

	// Size of various model parameters
	uint32_t  nlbl;    //   Y   number of labels
	uint64_t  nobs;    //   O   number of observations
	uint64_t  nftr;    //   F   number of features

	// Informations about observations
	char     *kind;    //  [O]  observations type
	uint64_t *uoff;    //  [O]  unigram weights offset
	uint64_t *boff;    //  [O]  bigram weights offset

	// The model itself
	double   *theta;   //  [F]  features weights

	// Datasets
	dat_t    *train;   //       training dataset
	dat_t    *devel;   //       development dataset
	rdr_t    *reader;

	// Stoping criterion
	double   *werr;    //       Window of error rate of last iters
	uint32_t  wcnt;    //       Number of iters in the window
	uint32_t  wpos;    //       Position for the next iter

	// Timing
	tms_t     timer;   //       start time of last iter
	double    total;   //       total training time
};

mdl_t *mdl_new(rdr_t *rdr);
void mdl_free(mdl_t *mdl);
void mdl_sync(mdl_t *mdl);
void mdl_compact(mdl_t *mdl);
void mdl_save(mdl_t *mdl, FILE *file);
void mdl_load(mdl_t *mdl, FILE *file);

#endif
