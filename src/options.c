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

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "wapiti.h"
#include "tools.h"
#include "options.h"
#include "vmath.h"

/******************************************************************************
 * Command line parsing
 *
 *   This module handle command line parsing and put all things defined by the
 *   user in a special structure in order to make them accessible to the
 *   remaining of the program.
 ******************************************************************************/

/* opt_help:
 *   Just display the help message describing modes and switch.
 */
static void opt_help(const char *pname) {
	static const char msg[] =
		"Global switchs:\n"
		"\t-h | --help      display this help message\n"
		"\t   | --version   display version information\n"
		"\n"
		"Training mode:\n"
		"    %1$s train [options] [input data] [model file]\n"
		"\t-a | --algo     STRING  training algorithm to use\n"
		"\t-p | --pattern  FILE    patterns for extracting features\n"
		"\t-m | --model    FILE    model file to preload\n"
		"\t-d | --devel    FILE    development dataset\n"
		"\t-c | --compact          compact model after training\n"
		"\t-t | --nthread  INT     number of worker threads\n"
		"\t-s | --sparse           enable sparse forward/backward\n"
		"\t-i | --maxiter  INT     maximum number of iterations\n"
		"\t-1 | --rho1     FLOAT   l1 penalty parameter\n"
		"\t-2 | --rho2     FLOAT   l2 penalty parameter\n"
		"\t-o | --objwin   INT     convergence window size\n"
		"\t-w | --stopwin  INT     stop window size\n"
		"\t-e | --stopeps  FLOAT   stop epsilon value\n"
		"\t   | --clip             (l-bfgs) clip gradient\n"
		"\t   | --histsz   INT     (l-bfgs) history size\n"
		"\t   | --maxls    INT     (l-bfgs) max linesearch iters\n"
		"\t   | --eta0     FLOAT   (sgd-l1) learning rate\n"
		"\t   | --alpha    FLOAT   (sgd-l1) exp decay parameter\n"
		"\t   | --kappa    FLOAT   (bcd)    stability parameter\n"
		"\n"
		"Labelling mode:\n"
		"    %1$s label [options] [input data] [output data]\n"
		"\t-m | --model    FILE    model file to load\n"
		"\t-l | --label            output only labels\n"
		"\t-c | --check            input is already labeled\n"
		"\t-s | --score            add scores to output\n"
		"\t-n | --nbest    INT     output n-best list\n"
		"\n"
		"Dumping mode\n"
		"    %1$s dump [input model] [output text]\n";
	fprintf(stderr, msg, pname);
}

/* opt_defaults:
 *   Default values for all parameters of the model.
 */
const opt_t opt_defaults = {
	.mode    = -1,
	.input   = NULL,     .output  = NULL,
	.algo    = "l-bfgs", .pattern = NULL,  .model   = NULL, .devel   = NULL,
	.compact = false,    .sparse  = false, .nthread = 1,    .maxiter = 0,
	.rho1    = 0.5,      .rho2    = 0.0001,
	.objwin  = 5,        .stopwin = 5,     .stopeps = 0.02,
	.lbfgs = {.clip  = false, .histsz = 5, .maxls = 40},
	.sgdl1 = {.eta0  = 0.8,   .alpha  = 0.85},
	.bcd   = {.kappa = 1.5},
	.label   = false,    .check   = false, .outsc = false,  .nbest = 1
};

/* opt_switch:
 *   Define available switchs for the different modes in a readable way for the
 *   command line argument parser.
 */
struct {
	int     mode;
	char   *dshort;
	char   *dlong;
	char    kind;
	size_t  offset;
} opt_switch[] = {
	{0, "-a", "--algo",    'S', offsetof(opt_t, algo        )},
	{0, "-p", "--pattern", 'S', offsetof(opt_t, pattern     )},
	{0, "-m", "--model",   'S', offsetof(opt_t, model       )},
	{0, "-d", "--devel",   'S', offsetof(opt_t, devel       )},
	{0, "-c", "--compact", 'B', offsetof(opt_t, compact     )},
	{0, "-s", "--sparse",  'B', offsetof(opt_t, sparse      )},
	{0, "-t", "--nthread", 'I', offsetof(opt_t, nthread     )},
	{0, "-i", "--maxiter", 'I', offsetof(opt_t, maxiter     )},
	{0, "-1", "--rho1",    'F', offsetof(opt_t, rho1        )},
	{0, "-2", "--rho2",    'F', offsetof(opt_t, rho2        )},
	{0, "-o", "--objsz",   'I', offsetof(opt_t, objwin      )},
	{0, "-w", "--stopwin", 'I', offsetof(opt_t, stopwin     )},
	{0, "-e", "--stopeps", 'F', offsetof(opt_t, stopeps     )},
	{0, "##", "--clip",    'B', offsetof(opt_t, lbfgs.clip  )},
	{0, "##", "--histsz",  'I', offsetof(opt_t, lbfgs.histsz)},
	{0, "##", "--maxls",   'I', offsetof(opt_t, lbfgs.maxls )},
	{0, "##", "--eta0",    'F', offsetof(opt_t, sgdl1.eta0  )},
	{0," ##", "--alpha",   'F', offsetof(opt_t, sgdl1.alpha )},
	{0, "##", "--kappa",   'F', offsetof(opt_t, bcd.kappa   )},
	{1, "-m", "--model",   'S', offsetof(opt_t, model       )},
	{1, "-l", "--label",   'B', offsetof(opt_t, label       )},
	{1, "-c", "--check",   'B', offsetof(opt_t, check       )},
	{1, "-s", "--score",   'B', offsetof(opt_t, outsc       )},
	{1, "-n", "--nbest",   'I', offsetof(opt_t, nbest       )},
	{-1, NULL, NULL, '\0', 0}
};

/* argparse:
 *   This is the main function for command line parsing. It use the previous
 *   table to known how to interpret the switchs and store values in the opt_t
 *   structure.
 */
void opt_parse(int argc, char *argv[argc], opt_t *opt) {
	static const char *err_badval = "invalid value for switch '%s'";
	const char *pname = argv[0];
	argc--, argv++;
	if (argc == 0)
		fatal("no mode specified");
	// First special handling for help and version
	if (!strcmp(argv[0], "-h") || !strcmp(argv[0], "--help")) {
		opt_help(pname);
		exit(EXIT_FAILURE);
	} else if (!strcmp(argv[0], "--version")) {
		fprintf(stderr, "Wapiti v" VERSION "\n");
		fprintf(stderr, "  Optimization mode:   %s\n", xvm_mode());
		exit(EXIT_SUCCESS);
	}
	// Get the mode to use
	if (!strcmp(argv[0], "t") || !strcmp(argv[0], "train")) {
		opt->mode = 0;
	} else if (!strcmp(argv[0], "l") || !strcmp(argv[0], "label")) {
		opt->mode = 1;
	} else if (!strcmp(argv[0], "d") || !strcmp(argv[0], "dump")) {
		opt->mode = 2;
	} else {
		fatal("unknown mode <%s>", argv[0]);
	}
	argc--, argv++;
	// Parse remaining arguments
	while (argc > 0 && argv[0][0] == '-') {
		const char *arg = argv[0];
		int idx;
		// Search the current switch in the table or fail if it cannot
		// be found.
		for (idx = 0; opt_switch[idx].mode != -1; idx++) {
			if (opt_switch[idx].mode != opt->mode)
				continue;
			if (!strcmp(arg, opt_switch[idx].dshort))
				break;
			if (!strcmp(arg, opt_switch[idx].dlong))
				break;
		}
		if (opt_switch[idx].mode == -1)
			fatal("unknown option '%s'", arg);
		// Decode the argument and store it in the structure
		if (opt_switch[idx].kind != 'B' && argc < 2)
			fatal("missing argument for switch '%s'", arg);
		void *ptr = (void *)((char *)opt + opt_switch[idx].offset);
		switch (opt_switch[idx].kind) {
			case 'S':
				*((char **)ptr) = argv[1];
				argc -= 2, argv += 2;
				break;
			case 'I':
				if (sscanf(argv[1], "%d", (int *)ptr) != 1)
					fatal(err_badval, arg);
				argc -= 2, argv += 2;
				break;
			case 'F': {
				double tmp;
				if (sscanf(argv[1], "%lf", &tmp) != 1)
					fatal(err_badval, arg);
				*((double *)ptr) = tmp;
				argc -= 2, argv += 2;
				break; }
			case 'B':
				*((bool *)ptr) = true;
				argc--, argv++;
				break;
		}
	}
	// Remaining arguments are input and output files
	if (argc > 0)
		opt->input = argv[0];
	if (argc > 1)
		opt->output = argv[1];
	// Small trick for the maxiter switch
	if (opt->maxiter == 0)
		opt->maxiter = INT_MAX;
	// Check that all options are valid
	#define argchecksub(name, test)                      \
		if (!(test))                                 \
			fatal("invalid value for <"name">");
	argchecksub("--thread", opt->nthread      >  0  );
	argchecksub("--rho1",   opt->rho1         >= 0.0);
	argchecksub("--rho2",   opt->rho2         >= 0.0);
	argchecksub("--histsz", opt->lbfgs.histsz >  0  );
	argchecksub("--maxls",  opt->lbfgs.maxls  >  0  );
	argchecksub("--eta0",   opt->sgdl1.eta0   >  0.0);
	argchecksub("--alpha",  opt->sgdl1.alpha  >  0.0);
	argchecksub("--nbest",  opt->nbest        >  0  );
	#undef argchecksub
}

