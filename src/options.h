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
#ifndef options_h
#define options_h

#include <stdint.h>
#include <stdbool.h>

#include "wapiti.h"

/* opt_t:
 *   This structure hold all user configurable parameter for Wapiti and is
 *   filled with parameters from command line.
 */
typedef struct opt_s opt_t;
struct opt_s {
	int       mode;
	char     *input,  *output;
	bool      maxent;
	// Options for training
	char     *type;
	char     *algo,   *pattern;
	char     *model,  *devel;
	char     *rstate, *sstate;
	bool      compact, sparse;
	uint32_t  nthread;
	uint32_t  jobsize;
	uint32_t  maxiter;
	double    rho1,    rho2;
	// Window size criterion
	uint32_t  objwin;
	uint32_t  stopwin;
	double    stopeps;
	// Options specific to L-BFGS
	struct {
		bool     clip;
		uint32_t histsz;
		uint32_t maxls;
	} lbfgs;
	// Options specific to SGD-L1
	struct {
		double   eta0;
		double   alpha;
	} sgdl1;
	// Options specific to BCD
	struct {
		double   kappa;
	} bcd;
	// Options specific to RPROP
	struct {
		double   stpmin;
		double   stpmax;
		double   stpinc;
		double   stpdec;
		bool     cutoff;
	} rprop;
	// Options for labelling
	bool      label;
	bool      check;
	bool      outsc;
	bool      lblpost;
	uint32_t  nbest;
	bool      force;
};

extern const opt_t opt_defaults;

void opt_parse(int argc, char *argv[argc], opt_t *opt);

#endif

