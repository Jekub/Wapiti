/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2013  CNRS
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
#include <ctype.h>
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

static void dotrain(mdl_t *mdl, iol_t *iol) {
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
		mdl_load(mdl);
	}
	// Load the pattern file. This will unlock the database if previously
	// locked by loading a model.
	if (mdl->opt->pattern != NULL) {
		info("* Load patterns\n");
		FILE *file = fopen(mdl->opt->pattern, "r");
		if (file == NULL)
			pfatal("cannot open pattern file");
		iol_t *iol = iol_new(file, NULL);
		rdr_loadpat(mdl->reader, iol);
		fclose(file);
		iol_free(iol);
		qrk_lock(mdl->reader->obs, false);
	}
	// Load the training data. When this is done we lock the quarks as we
	// don't want to put in the model, informations present only in the
	// devlopment set.
	info("* Load training data\n");
	mdl->train = rdr_readdat(mdl->reader, iol, true);
	qrk_lock(mdl->reader->lbl, true);
	qrk_lock(mdl->reader->obs, true);
	if (mdl->train == NULL || mdl->train->nseq == 0)
		fatal("no train data loaded");
	// If present, load the development set in the model. If not specified,
	// the training dataset will be used instead.
	if (mdl->opt->devel != NULL) {
		info("* Load development data\n");
		FILE *file = fopen(mdl->opt->devel, "r");
		iol_t *iol = iol_new(file, NULL);
		if (file == NULL)
			pfatal("cannot open development file");
		mdl->devel = rdr_readdat(mdl->reader, iol, true);
		iol_free(iol);
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
	mdl_save(mdl, iol);
	info("* Done\n");
}

/*******************************************************************************
 * Labeling
 ******************************************************************************/
static void dolabel(mdl_t *mdl, iol_t *iol) {
	// First, load the model provided by the user. This is mandatory to
	// label new datas ;-)
	info("* Load model\n");
	mdl_load(mdl);

	// Do the labelling
	info("* Label sequences\n");
	tag_label(mdl, iol);
	info("* Done\n");
}

/*******************************************************************************
 * Dumping
 ******************************************************************************/
static void dodump(mdl_t *mdl, iol_t *iol) {
	// Load input model file
	info("* Load model\n");
	mdl_load(mdl);
	// Dump model
	info("* Dump model\n");
	const uint32_t Y = mdl->nlbl;
	const uint64_t O = mdl->nobs;
	const qrk_t *Qlbl = mdl->reader->lbl;
	const qrk_t *Qobs = mdl->reader->obs;
	char fmt[16];
	sprintf(fmt, "%%.%df\n", mdl->opt->prec);
	for (uint64_t o = 0; o < O; o++) {
		const char *obs = qrk_id2str(Qobs, o);
		bool empty = true;
		if (mdl->kind[o] & 1) {
			const double *w = mdl->theta + mdl->uoff[o];
			for (uint32_t y = 0; y < Y; y++) {
				if (!mdl->opt->all && w[y] == 0.0)
					continue;
				const char *ly = qrk_id2str(Qlbl, y);
				iol->print_cb(iol->out, "%s\t#\t%s\t", obs, ly);
				iol->print_cb(iol->out, fmt, w[y]);
				empty = false;
			}
		}
		if (mdl->kind[o] & 2) {
			const double *w = mdl->theta + mdl->boff[o];
			for (uint32_t d = 0; d < Y * Y; d++) {
				if (!mdl->opt->all && w[d] == 0.0)
					continue;
				const char *ly  = qrk_id2str(Qlbl, d % Y);
				const char *lyp = qrk_id2str(Qlbl, d / Y);
				iol->print_cb(iol->out, "%s\t%s\t%s\t", obs, lyp, ly);
				iol->print_cb(iol->out, fmt, w[d]);
				empty = false;
			}
		}
		if (!empty)
			iol->print_cb(iol->out, "\n");
	}
}


/*******************************************************************************
 * Updating
 ******************************************************************************/
static void doupdt(mdl_t *mdl, iol_t *iol) {
	// Load input model file
	info("* Load model\n");
	mdl_load(mdl);

	// Open patch file
	info("* Update model\n");
	int nline = 0;
	while (true) {
		char *line = iol->gets_cb(iol->in);
		if (line == NULL)
			break;
		nline++;
		// First we split the line in space separated tokens. We expect
		// four of them and skip empty lines.
		char *toks[4];
		int ntoks = 0;
		while (ntoks < 4) {
			while (isspace(*line))
				line++;
			if (*line == '\0')
				break;
			toks[ntoks++] = line;
			while (*line != '\0' && !isspace(*line))
				line++;
			if (*line == '\0')
				break;
			*line++ = '\0';
		}
		if (ntoks == 0) {
			free(line);
			continue;
		} else if (ntoks != 4) {
			fatal("invalid line at %d", nline);
		}
		// Parse the tokens, the first three should be string maping to
		// observations and labels and the last should be the weight.
		uint64_t obs = none, yp = none, y = none;
		obs = qrk_str2id(mdl->reader->obs, toks[0]);
		if (obs == none)
			fatal("bad on observation on line %d", nline);
		if (strcmp(toks[1], "#")) {
			yp = qrk_str2id(mdl->reader->lbl, toks[1]);
			if (yp == none)
				fatal("bad label <%s> line %d", toks[1], nline);
		}
		y = qrk_str2id(mdl->reader->lbl, toks[2]);
		if (y == none)
			fatal("bad label <%s> line %d", toks[2], nline);
		double wgh = 0.0;
		if (sscanf(toks[3], "%lf", &wgh) != 1)
			fatal("bad weight on line %d", nline);

		const uint32_t Y = mdl->nlbl;
		if (yp == none) {
			double *w = mdl->theta + mdl->uoff[obs];
			w[y] = wgh;
		} else {
			double *w = mdl->theta + mdl->boff[obs];
			w[yp * Y + y] = wgh;
		}
		free(line);
	}
	// If requested compact the model.
	if (mdl->opt->compact) {
		const uint64_t O = mdl->nobs;
		const uint64_t F = mdl->nftr;
		info("* Compacting the model\n");
		mdl_compact(mdl);
		info("    %8"PRIu64" observations removed\n", O - mdl->nobs);
		info("    %8"PRIu64" features removed\n", F - mdl->nftr);
	}
	// And save the updated model
	info("* Save the model\n");
	mdl_save(mdl, iol);
	info("* Done\n");
}

static iol_t *create_iol(opt_t *opt) {
	FILE *fin = stdin, *fout = stdout;
	if (opt->input != NULL) {
		fin = fopen(opt->input, "r");
		if (fin == NULL)
			pfatal("cannot open input data file");
	}
	if (opt->output != NULL) {
		fout = fopen(opt->output, "w");
		if (fout == NULL)
			pfatal("cannot open output data file");
	}
 
        return iol_new(fin, fout);
}

static void close_iol(iol_t *iol) {
	if (iol->in != NULL)
		fclose(iol->in);
	if (iol->out != NULL)
		fclose(iol->out);
}

static iol_t *create_model_iol(opt_t *opt) {
        if (opt->model == NULL) 
            fatal("you must specify a model");

	FILE *fin = fopen(opt->model, "r");
	if (fin == NULL)
            pfatal("cannot open model file %s", opt->model);

        iol_t *iol = iol_new(fin, NULL);
        return iol;
}



/*******************************************************************************
 * Entry point
 ******************************************************************************/
int main(int argc, char *argv[argc]) {
	// We first parse command line switchs
	opt_t opt = opt_defaults;
	opt_parse(argc, argv, &opt);
	// Next we prepare the model
        iol_t *io_iol = create_iol(&opt);
        iol_t *model_iol;
	switch (opt.mode) {
	        case 2:  model_iol = io_iol; break;
                default: model_iol = create_model_iol(&opt); break;
	}
	mdl_t *mdl = mdl_new(rdr_new(model_iol, opt.maxent));
	mdl->opt = &opt;
	// And switch to requested mode
	switch (opt.mode) {
                case 0: dotrain(mdl, io_iol); break;
                case 1: dolabel(mdl, io_iol); break;
	        case 2: dodump(mdl, io_iol);  break;
	        case 3: doupdt(mdl, io_iol);  break;
	}
	// And cleanup
	close_iol(io_iol);
	mdl_free(mdl);
	return EXIT_SUCCESS;
}

