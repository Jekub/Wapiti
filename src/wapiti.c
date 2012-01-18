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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "decoder.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"
#include "trainers.h"
#include "wapiti.h"

/*******************************************************************************
 * Training
 ******************************************************************************/
static const char *typ_lst[] = {
	"maxent",
	"memm",
	"crf"
};
static const uint32_t typ_cnt = sizeof(typ_lst) / sizeof(typ_lst[0]);

static const struct {
	char *name;
	void (* train)(mdl_t *mdl);
} trn_lst[] = {
	{"l-bfgs", trn_lbfgs},
	{"sgd-l1", trn_sgdl1},
	{"bcd",    trn_bcd  },
	{"rprop",  trn_rprop},
	{"rprop+", trn_rprop},
	{"rprop-", trn_rprop},
};
static const uint32_t trn_cnt = sizeof(trn_lst) / sizeof(trn_lst[0]);

static void dotrain(mdl_t *mdl) {
	// Check if the user requested the type or trainer list. If this is not
	// the case, search them in the lists.
	if (!strcmp(mdl->opt->type, "list")) {
		info("Available types of models:\n");
		for (uint32_t i = 0; i < typ_cnt; i++)
			info("\t%s\n", typ_lst[i]);
		exit(EXIT_SUCCESS);
	}
	if (!strcmp(mdl->opt->algo, "list")) {
		info("Available training algorithms:\n");
		for (uint32_t i = 0; i < trn_cnt; i++)
			info("\t%s\n", trn_lst[i].name);
		exit(EXIT_SUCCESS);
	}
	uint32_t typ, trn;
	for (typ = 0; typ < typ_cnt; typ++)
		if (!strcmp(mdl->opt->type, typ_lst[typ]))
			break;
	if (typ == typ_cnt)
		fatal("unknown model type '%s'", mdl->opt->type);
	mdl->type = typ;
	for (trn = 0; trn < trn_cnt; trn++)
		if (!strcmp(mdl->opt->algo, trn_lst[trn].name))
			break;
	if (trn == trn_cnt)
		fatal("unknown algorithm '%s'", mdl->opt->algo);
	// Load a previous model to train again if specified by the user.
	if (mdl->opt->model != NULL) {
		info("* Load previous model\n");
		FILE *file = fopen(mdl->opt->model, "r");
		if (file == NULL)
			pfatal("cannot open input model file");
		mdl_load(mdl, file);
	}
	// Load the pattern file. This will unlock the database if previously
	// locked by loading a model.
	if (mdl->opt->pattern != NULL) {
		info("* Load patterns\n");
		FILE *file = fopen(mdl->opt->pattern, "r");
		if (file == NULL)
			pfatal("cannot open pattern file");
		rdr_loadpat(mdl->reader, file);
		fclose(file);
		qrk_lock(mdl->reader->obs, false);
	}
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
	info("    nb train:    %"PRIu32"\n", mdl->train->nseq);
	if (mdl->devel != NULL)
		info("    nb devel:    %"PRIu32"\n", mdl->devel->nseq);
	info("    nb labels:   %"PRIu32"\n", mdl->nlbl);
	info("    nb blocks:   %"PRIu64"\n", mdl->nobs);
	info("    nb features: %"PRIu64"\n", mdl->nftr);
	// And train the model...
	info("* Train the model with %s\n", mdl->opt->algo);
	uit_setup(mdl);
	trn_lst[trn].train(mdl);
	uit_cleanup(mdl);
	// If requested compact the model.
	if (mdl->opt->compact) {
		const uint64_t O = mdl->nobs;
		const uint64_t F = mdl->nftr;
		info("* Compacting the model\n");
		mdl_compact(mdl);
		info("    %8"PRIu64" observations removed\n", O - mdl->nobs);
		info("    %8"PRIu64" features removed\n", F - mdl->nftr);
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
	const uint32_t Y = mdl->nlbl;
	const uint64_t O = mdl->nobs;
	const qrk_t *Qlbl = mdl->reader->lbl;
	const qrk_t *Qobs = mdl->reader->obs;
	for (uint64_t o = 0; o < O; o++) {
		const char *obs = qrk_id2str(Qobs, o);
		bool empty = true;
		if (mdl->kind[o] & 1) {
			const double *w = mdl->theta + mdl->uoff[o];
			for (uint32_t y = 0; y < Y; y++) {
				if (w[y] == 0.0)
					continue;
				const char *ly = qrk_id2str(Qlbl, y);
				fprintf(fout, "%s\t#\t%s\t%f\n", obs, ly, w[y]);
				empty = false;
			}
		}
		if (mdl->kind[o] & 2) {
			const double *w = mdl->theta + mdl->boff[o];
			for (uint32_t d = 0; d < Y * Y; d++) {
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
	// We first parse command line switchs
	opt_t opt = opt_defaults;
	opt_parse(argc, argv, &opt);
	// Next we prepare the model
	mdl_t *mdl = mdl_new(rdr_new(opt.maxent));
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

