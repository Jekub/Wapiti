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
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/times.h>
#include <sys/resource.h>

#include "model.h"
#include "options.h"
#include "quark.h"
#include "reader.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"
#include "wapiti.h"

typedef struct tms tms_t;

/*******************************************************************************
 * Linear chain CRF model
 *
 *   There is three concept that must be well understand here, the labels,
 *   observations, and features. The labels are the values predicted by the
 *   model at each point of the sequence and denoted by Y. The observations are
 *   the values, at each point of the sequence, given to the model in order to
 *   predict the label and denoted by O. A feature is a test on both labels and
 *   observations, denoted by F. In linear chain CRF there is two kinds of
 *   features :
 *     - unigram feature who represent a test on the observations at the current
 *       point and the label at current point.
 *     - bigram feature who represent a test on the observation at the current
 *       point and two labels : the current one and the previous one.
 *   So for each observation, there Y possible unigram features and Y*Y possible
 *   bigram features. The kind of features used by the model for a given
 *   observation depend on the pattern who generated it.
 ******************************************************************************/

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
	opt_t   *opt;     //       options for training

	// Size of various model parameters
	size_t   nlbl;    //   Y   number of labels
	size_t   nobs;    //   O   number of observations
	size_t   nftr;    //   F   number of features

	// Informations about observations
	char    *kind;    //  [O]  observations type
	size_t  *uoff;    //  [O]  unigram weights offset
	size_t  *boff;    //  [O]  bigram weights offset

	// The model itself
	double  *theta;   //  [F]  features weights

	// Datasets
	dat_t   *train;   //       training dataset
	dat_t   *devel;   //       development dataset
	rdr_t   *reader;

	// Stoping criterion
	double  *werr;    //       Window of error rate of last iters
	int      wcnt;    //       Number of iters in the window
	int      wpos;    //       Position for the next iter

	// Timing
	tms_t    timer;   //       start time of last iter
	double   total;   //       total training time
};

/* mdl_new:
 *   Allocate a new empty model object linked with the given reader. The model
 *   have to be synchronized before starting training or labelling. If you not
 *   provide a reader (as it will loaded from file for example) you must be sure
 *   to set one in the model before any attempts to synchronize it.
 */
static mdl_t *mdl_new(rdr_t *rdr) {
	mdl_t *mdl = xmalloc(sizeof(mdl_t));
	mdl->nlbl   = mdl->nobs  = mdl->nftr = 0;
	mdl->kind   = NULL;
	mdl->uoff   = mdl->boff  = NULL;
	mdl->theta  = NULL;
	mdl->train  = mdl->devel = NULL;
	mdl->reader = rdr;
	mdl->werr   = NULL;
	mdl->total  = 0.0;
	return mdl;
}

/* mdl_free:
 *   Free all memory used by a model object inculding the reader and datasets
 *   loaded in the model.
 */
static void mdl_free(mdl_t *mdl) {
	free(mdl->kind);
	free(mdl->uoff);
	free(mdl->boff);
	free(mdl->theta);
	if (mdl->train != NULL)
		rdr_freedat(mdl->train);
	if (mdl->devel != NULL)
		rdr_freedat(mdl->devel);
	if (mdl->reader != NULL)
		rdr_free(mdl->reader);
	if (mdl->werr != NULL)
		free(mdl->werr);
}

/* mdl_sync:
 *   Synchronize the model with its reader. As the model is just a placeholder
 *   for features weights and interned sequences, it know very few about the
 *   labels and observations, all the informations are kept in the reader. A
 *   sync will get the labels and observations count as well as the observation
 *   kind from the reader and build internal structures representing the model.
 *
 *   If the model was already synchronized before, there is an existing model
 *   incompatible with the new one to be created. In this case there is two
 *   possibility :
 *     - If only new observations was added, the weights of the old ones remain
 *       valid and are kept as they form a probably good starting point for
 *       training the new model, the new observation get a 0 weight ;
 *     - If new labels was added, the old model are trully meaningless so we
 *       have to fully discard them and build a new empty model.
 *   In any case, you must never change existing labels or observations, if this
 *   happen, you need to create a new model and destroy this one.
 *
 *   After synchronization, the labels and observations databases are locked to
 *   prevent new one to be created. You must unlock them explicitly if needed.
 *   This reduce the risk of mistakes.
 */
static void mdl_sync(mdl_t *mdl) {
	const size_t Y = qrk_count(mdl->reader->lbl);
	const size_t O = qrk_count(mdl->reader->obs);
	// If model is already synchronized, do nothing and just return
	if (mdl->nlbl == Y && mdl->nobs == O)
		return;
	if (Y == 0 || O == 0)
		fatal("cannot synchronize an empty model");
	// If new labels was added, we have to discard all the model. In this
	// case we also display a warning as this is probably not expected by
	// the user. If only new observations was added, we will try to expand
	// the model.
	size_t oldF = mdl->nftr;
	size_t oldO = mdl->nobs;
	if (mdl->nlbl != Y && mdl->nlbl != 0) {
		warning("labels count changed, discarding the model");
		free(mdl->kind);  mdl->kind  = NULL;
		free(mdl->uoff);  mdl->uoff  = NULL;
		free(mdl->boff);  mdl->boff  = NULL;
		free(mdl->theta); mdl->theta = NULL;
		oldF = oldO = 0;
	}
	mdl->nlbl = Y;
	mdl->nobs = O;
	// Allocate the observations datastructure. If the model is empty or
	// discarded, a new one iscreated, else the old one is expanded.
	char   *kind = xrealloc(mdl->kind, sizeof(char  ) * O);
	size_t *uoff = xrealloc(mdl->uoff, sizeof(size_t) * O);
	size_t *boff = xrealloc(mdl->boff, sizeof(size_t) * O);
	mdl->kind = kind;
	mdl->uoff = uoff;
	mdl->boff = boff;
	// Now, we can setup the features. For each new observations we fill the
	// kind and offsets arrays and count total number of features as well.
	size_t F = oldF;
	for (size_t o = oldO; o < O; o++) {
		const char *obs = qrk_id2str(mdl->reader->obs, o);
		switch (obs[0]) {
			case 'u': kind[o] = 1; break;
			case 'b': kind[o] = 2; break;
			case '*': kind[o] = 3; break;
		}
		if (kind[o] & 1)
			uoff[o] = F, F += Y;
		if (kind[o] & 2)
			boff[o] = F, F += Y * Y;
	}
	mdl->nftr = F;
	// We can finally grow the features weights vector itself. We set all
	// the new features to 0.0 but don't touch the old ones.
	mdl->theta = xrealloc(mdl->theta, sizeof(double) * F);
	for (size_t f = oldF; f < F; f++)
		mdl->theta[f] = 0.0;
	// And lock the databases
	qrk_lock(mdl->reader->lbl, true);
	qrk_lock(mdl->reader->obs, true);
}

/* mdl_compact:
 *   Comapct the given model by removing from it all observation who lead to
 *   zero actives features. On model trained with l1 regularization this can
 *   lead to a drastic model size reduction and so to faster loading, training
 *   and labeling.
 */
static void mdl_compact(mdl_t *mdl) {
	const size_t Y = mdl->nlbl;
	// We first build the new observation list with only observations which
	// lead to at least one active feature. At the same time we build the
	// translation table which map the new observations index to the old
	// ones.
	info("    - Scan the model\n");
	qrk_t *old_obs = mdl->reader->obs;
	qrk_t *new_obs = qrk_new();
	size_t *trans = xmalloc(sizeof(size_t) * mdl->nobs);
	for (size_t oldo = 0; oldo < mdl->nobs; oldo++) {
		bool active = false;
		if (mdl->kind[oldo] & 1)
			for (size_t y = 0; y < Y; y++)
				if (mdl->theta[mdl->uoff[oldo] + y] != 0.0)
					active = true;
		if (mdl->kind[oldo] & 2)
			for (size_t d = 0; d < Y * Y; d++)
				if (mdl->theta[mdl->boff[oldo] + d] != 0.0)
					active = true;
		if (!active)
			continue;
		const char   *str  = qrk_id2str(old_obs, oldo);
		const size_t  newo = qrk_str2id(new_obs, str);
		trans[newo] = oldo;
	}
	mdl->reader->obs = new_obs;
	// Now we save the old model features informations and build a new one
	// corresponding to the compacted model.
	size_t *old_uoff  = mdl->uoff;  mdl->uoff  = NULL;
	size_t *old_boff  = mdl->boff;  mdl->boff  = NULL;
	double *old_theta = mdl->theta; mdl->theta = NULL;
	free(mdl->kind);
	mdl->kind = NULL;
	mdl->nlbl = mdl->nobs = mdl->nftr = 0;
	mdl_sync(mdl);
	// The model is now ready, so we copy in it the features weights from
	// the old model for observations we have kept.
	info("    - Compact it\n");
	for (size_t newo = 0; newo < mdl->nobs; newo++) {
		const size_t oldo = trans[newo];
		if (mdl->kind[newo] & 1) {
			double *src = old_theta  + old_uoff[oldo];
			double *dst = mdl->theta + mdl->uoff[newo];
			for (size_t y = 0; y < Y; y++)
				dst[y] = src[y];
		}
		if (mdl->kind[newo] & 2) {
			double *src = old_theta  + old_boff[oldo];
			double *dst = mdl->theta + mdl->boff[newo];
			for (size_t d = 0; d < Y * Y; d++)
				dst[d] = src[d];
		}
	}
	// And cleanup
	free(trans);
	qrk_free(old_obs);
	free(old_uoff);
	free(old_boff);
	free(old_theta);
}

/* mdl_save:
 *   Save a model to be restored later in a platform independant way.
 */
static void mdl_save(mdl_t *mdl, FILE *file) {
	size_t nact = 0;
	for (size_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			nact++;
	fprintf(file, "#mdl#%zu\n", nact);
	rdr_save(mdl->reader, file);
	for (size_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			fprintf(file, "%zu=%la\n", f, mdl->theta[f]);
}

/* mdl_load:
 *   Read back a previously saved model to continue training or start labeling.
 *   The returned model is synced and the quarks are locked. You must give to
 *   this function an empty model fresh from mdl_new.
 */
static void mdl_load(mdl_t *mdl, FILE *file) {
	const char *err = "invalid model format";
	size_t nact = 0;
	if (fscanf(file, "#mdl#%zu\n", &nact) != 1)
		fatal(err);
	rdr_load(mdl->reader, file);
	mdl_sync(mdl);
	for (size_t i = 0; i < nact; i++) {
		size_t f;
		double v;
		if (fscanf(file, "%zu=%la\n", &f, &v) != 2)
			fatal(err);
		mdl->theta[f] = v;
	}
}

/******************************************************************************
 * Sequence tagging
 *
 *   This module implement sequence tagging using a trained model and model
 *   evaluation on devlopment set.
 *
 *   The viterbi can be quite intensive on the stack if you push in it long
 *   sequence and use large labels set. It's less a problem than in gradient
 *   computations but it can show up in particular cases. The fix is to call it
 *   through the mth_spawn function and request enough stack space, this will be
 *   fixed in next version.
 ******************************************************************************/

/* tag_viterbi:
 *   This function implement the Viterbi algorithm in order to decode the most
 *   probable sequence of labels according to the model. Some part of this code
 *   is very similar to the computation of the gradient as expected.
 *
 *   And like for the gradient, the caller is responsible to ensure there is
 *   enough stack space.
 */
static void tag_viterbi(const mdl_t *mdl, const seq_t *seq, size_t out[]) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	// Like for the gradient, we rely on stack storage and let the caller
	// ensure there is enough free space there. This function will need
	//   8 * T * (2 + Y * (1 + Y))
	// bytes of stack plus a bit more for variables.
	double psi [T][Y][Y];
	double back[T][Y];
	double cur [Y];
	double old [Y];
	// We first have to compute the Ψ_t(y',y,x_t) weights defined as
	//   Ψ_t(y',y,x_t) = \exp( ∑_k θ_k f_k(y',y,x_t) )
	// So at position 't' in the sequence, for each couple (y',y) we have
	// to sum weights of all features.
	// This is the same than what we do for computing the gradient but, as
	// the viterbi algorithm also work in the logarithmic space, we can
	// remove the exponential.
	//
	// Only the observations present at this position will have a non-nul
	// weight so we can sum only on thoses.
	//
	// As we use only two kind of features: unigram and bigram, we can
	// rewrite this as
	//   ∑_k μ_k(y, x_t) f_k(y, x_t) + ∑_k λ_k(y', y, x_t) f_k(y', y, x_t)
	// Where the first sum is over the unigrams features and the second is
	// over bigrams ones.
	//
	// This allow us to compute Ψ efficiently in two steps
	//   1/ we sum the unigrams features weights by looping over actives
	//        unigrams observations. (we compute this sum once and use it
	//        for each value of y')
	//   2/ we add the bigrams features weights by looping over actives
	//        bigrams observations (we don't have to do this for t=0 since
	//        there is no bigrams here)
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				sum += x[mdl->uoff[o] + y];
			}
			for (size_t yp = 0; yp < Y; yp++)
				psi[t][yp][y] = sum;
		}
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double sum = 0.0;
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					sum += x[mdl->boff[o] + d];
				}
				psi[t][yp][y] += sum;
			}
		}
	}
	// Now we can do the Viterbi algorithm. This is very similar to the
	// forward pass
	//   | α_1(y) = Ψ_1(y,x_1)
	//   | α_t(y) = max_{y'} α_{t-1}(y') + Ψ_t(y',y,x_t)
	// We just replace the sum by a max and as we do the computation in the
	// logarithmic space the product become a sum. (this also mean that we
	// don't have to worry about numerical problems)
	//
	// Next we have to walk backward over the α in order to find the best
	// path. In order to do this efficiently, we keep in the 'back' array
	// the indice of the y value selected by the max. This also mean that
	// we only need the current and previous value of the α vectors, not
	// the full matrix.
	for (size_t y = 0; y < Y; y++)
		cur[y] = psi[0][0][y];
	for (int t = 1; t < T; t++) {
		for (size_t y = 0; y < Y; y++)
			old[y] = cur[y];
		for (size_t y = 0; y < Y; y++) {
			double bst = -1.0;
			int    idx = 0;
			for (size_t yp = 0; yp < Y; yp++) {
				double val = psi[t][yp][y] + old[yp];
				if (val > bst) {
					bst = val;
					idx = yp;
				}
			}
			back[t][y] = idx;
			cur[y]     = bst;
		}
	}
	// We can now build the sequence of labels predicted by the model. For
	// this we search in the last α vector the best value. Using this index
	// as a starting point in the back-pointer array we finally can decode
	// the best sequence.
	int bst = 0;
	for (size_t y = 1; y < Y; y++)
		if (cur[y] > cur[bst])
			bst = y;
	for (int t = T; t > 0; t--) {
		out[t - 1] = bst;
		bst = back[t - 1][bst];
	}
}

/* tag_label:
 *   Label a data file using the current model. This output an almost exact copy
 *   of the input file with an additional column with the predicted label. If
 *   the check option is specified, the input file must be labelled and the
 *   predicted labels will be checked against the provided ones. This will
 *   output error rates during the labelling and detailed statistics per label
 *   at the end.
 */
static void tag_label(const mdl_t *mdl, FILE *fin, FILE *fout) {
	qrk_t *lbls = mdl->reader->lbl;
	const size_t Y = mdl->nlbl;
	// We start by preparing the statistic collection to be ready if check
	// option is used. The stat array hold the following for each label
	//   [0] # of reference with this label
	//   [1] # of token we have taged with this label
	//   [2] # of match of the two preceding
	int tcnt = 0, terr = 0;
	int scnt = 0, serr = 0;
	int stat[3][Y];
	for (size_t y = 0; y < Y; y++)
		stat[0][y] = stat[1][y] = stat[2][y] = 0;
	// Next read the input file sequence by sequence and label them, we have
	// to take care of not discarding the raw input as we want to send it
	// back to the output with the additional predicted labels.
	while (!feof(fin)) {
		// So, first read an input sequence keeping the raw_t object
		// available, and label it with Viterbi.
		raw_t *raw = rdr_readraw(mdl->reader, fin);
		if (raw == NULL)
			break;
		seq_t *seq = rdr_raw2seq(mdl->reader, raw, mdl->opt->check);
		size_t out[seq->len];
		tag_viterbi(mdl, seq, out);
		// Next we output the raw sequence with an aditional column for
		// the predicted labels
		for (int t = 0; t < seq->len; t++) {
			if (!mdl->opt->label)
				fprintf(fout, "%s\t", raw->lines[t]);
			fprintf(fout, "%s\n", qrk_id2str(lbls, out[t]));
		}
		fprintf(fout, "\n");
		// If user provided reference labels, use them to collect
		// statistics about how well we have performed here.
		if (mdl->opt->check) {
			bool err = false;
			for (int t = 0; t < seq->len; t++) {
				stat[0][seq->pos[t].lbl]++;
				stat[1][out[t]]++;
				if (seq->pos[t].lbl != out[t])
					terr++, err = true;
				else
					stat[2][out[t]]++;
			}
			tcnt += seq->len;
			serr += err;
		}
		// Cleanup memory used for this sequence
		rdr_freeseq(seq);
		rdr_freeraw(raw);
		// And report our progress, at regular interval we display how
		// much sequence are labelled and if possible the current tokens
		// and sequence error rates.
		if (++scnt % 1000 == 0) {
			info("%10zu sequences labeled", scnt);
			if (mdl->opt->check) {
				const double te = (double)terr  / tcnt * 100.0;
				const double se = (double)serr  / scnt * 100.0;
				info("\t%5.2f%%/%5.2f%%", te, se);
			}
			info("\n");
		}
	}
	// If user have provided reference labels, we have collected a lot of
	// statistics and we can repport global token and sequence error rate as
	// well as precision recall and f-measure for each labels.
	if (mdl->opt->check) {
		const double te = (double)terr  / tcnt * 100.0;
		const double se = (double)serr  / scnt * 100.0;
		info("    Nb sequences  : %zu\n", scnt);
		info("    Token error   : %5.2f%%\n", te);
		info("    Sequence error: %5.2f%%\n", se);
		info("* Per label statistics\n");
		for (size_t y = 0; y < Y; y++) {
			const char *lbl = qrk_id2str(lbls, y);
			const double Rc = (double)stat[2][y] / stat[0][y];
			const double Pr = (double)stat[2][y] / stat[1][y];
			const double F1 = 2.0 * (Pr * Rc) / (Pr + Rc);
			info("    %-6s", lbl);
			info("  Pr=%.2f", Pr);
			info("  Rc=%.2f", Rc);
			info("  F1=%.2f\n", F1);
		}
	}
}

/*******************************************************************************
 * User interaction during training
 *
 *   Handle progress reporting during training and clean early stoping. Trainers
 *   have to call uit_progress at the end of each iterations, this will display
 *   various informations for the user.
 *   Timing is also done here, an iteration is assumed to take all the time
 *   between to call to the progress function and evualtion on the devel data
 *   are included.
 *
 *   This module setup a signal handler for SIGINT. If this signal is catched,
 *   the uit_stop global variable to inform the trainer that it have to stop as
 *   early as possible, discarding the recent computations if they cannot be
 *   integrated very quickly. They must leave the model in a clean state. Any
 *   further signal will terminate the program. So it's simple :
 *     - 1 signal mean "I can wait a little so try to stop as soon as possible
 *         but leave me a working model"
 *     - 2 signal mean "Stop immediatly what you are doing, I can't wait and
 *         don't care about getting a working model"
 ******************************************************************************/

/* uit_stop:
 *   This value is set to true when the user request the trainer to stop. In
 *   this case, the trainer have to stop as soon as possible in a clean state,
 *   discarding the lasts computations if it cannot integrate them quickly.
 */
static bool uit_stop = false;

/* uit_signal:
 *   Signal handler to catch interupt signal. When a signal is received, the
 *   trainer is aksed to stop as soon as possible leaving the model in a clean
 *   state. We don't reinstall the handler so if user send a second interupt
 *   signal, the program will stop imediatly. (to cope with BSD system, we even
 *   reinstall explicitly the default handler)
 */
static void uit_signal(int sig) {
	signal(sig, SIG_DFL);
	uit_stop = true;
}

/* uit_setup:
 *   Install the signal handler for clean early stop from the user if possible
 *   and start the timer.
 */
static void uit_setup(mdl_t *mdl) {
	uit_stop = false;
	if (signal(SIGINT, uit_signal) == SIG_ERR)
		warning("failed to set signal handler, no clean early stop");
	times(&mdl->timer);
	if (mdl->opt->stopwin != 0)
		mdl->werr = xmalloc(sizeof(double) * mdl->opt->stopwin);
	mdl->wcnt = mdl->wpos = 0;
}

/* uit_cleanup:
 *   Remove the signal handler restoring the defaul behavior in case of
 *   interrupt.
 */
static void uit_cleanup(mdl_t *mdl) {
	unused(mdl);
	if (mdl->opt->stopwin != 0) {
		free(mdl->werr);
		mdl->werr = NULL;
	}
	signal(SIGINT, SIG_DFL);
}

/* uit_progress:
 *   Display a progress repport to the user consisting of some informations
 *   provided by the trainer: iteration count and objective function value, and
 *   some informations computed here on the current model performances.
 *   This function return true if the trainer have to keep training the model
 *   and false if he must stop, so this is were we will implement the trainer
 *   independant stoping criterion.
 */
static bool uit_progress(mdl_t *mdl, int it, double obj) {
	// We first evaluate the current model performances on the devel dataset
	// if available, else on the training dataset. We compute tokens and
	// sequence error rate.
	dat_t *dat = (mdl->devel == NULL) ? mdl->train : mdl->devel;
	int tcnt = 0, terr = 0;
	int scnt = 0, serr = 0;
	for (int s = 0; s < dat->nseq; s++) {
		// Tag the sequence with the viterbi
		const seq_t *seq = dat->seq[s];
		const int    T   = seq->len;
		size_t out[T];
		tag_viterbi(mdl, seq, out);
		// And check for eventual (probable ?) errors
		bool err = false;
		for (int t = 0; t < T; t++)
			if (seq->pos[t].lbl != out[t])
				terr++, err = true;
		tcnt += T, scnt += 1;
		serr += err;
	}
	const double te = (double)terr / tcnt * 100.0;
	const double se = (double)serr / scnt * 100.0;
	// Next, we compute the number of active features
	size_t act = 0;
	for (size_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			act++;
	// Compute timings. As some training algorithms are multi-threaded, we
	// cannot use ansi/c function and must rely on posix one to sum time
	// spent in main thread and in child ones.
	tms_t now; times(&now);
	double tm = (now.tms_utime  - mdl->timer.tms_utime )
		  + (now.tms_cutime - mdl->timer.tms_cutime);
	tm /= sysconf(_SC_CLK_TCK);
	mdl->total += tm;
	mdl->timer  = now;
	// And display progress report
	info("  [%4d]", it);
	info(obj >= 0.0 ? " obj=%-10.2f" : " obj=NA", obj);
	info(" act=%-8zu", act);
	info(" err=%5.2f%%/%5.2f%%", te, se);
	info(" time=%.2fs/%.2fs", tm, mdl->total);
	info("\n");
	// If requested, check the error rate stoping criterion. We check if the
	// error rate is stable enought over a few iterations.
	bool res = true;
	if (mdl->opt->stopwin != 0) {
		mdl->werr[mdl->wpos] = te;
		mdl->wpos = (mdl->wpos + 1) % mdl->opt->stopwin;
		mdl->wcnt++;
		if (mdl->wcnt >= mdl->opt->stopwin) {
			double emin = 200.0, emax = -100.0;
			for (int i = 0; i < mdl->opt->stopwin; i++) {
				emin = min(emin, mdl->werr[i]);
				emax = max(emax, mdl->werr[i]);
			}
			if (emax - emin < mdl->opt->stopeps)
				res = false;
		}
	}
	// And return
	if (uit_stop)
		return false;
	return res;
}

/******************************************************************************
 * Single sequence gradient computation
 *
 *   This section is responsible for computing the gradient of the
 *   log-likelihood function to optimize over a single sequence.
 *
 *   There is two version of this code, one using dense matrix and one with
 *   sparse matrix. The sparse version use the fact that for L1 regularized
 *   trainers, the bigrams scores will be very sparse so there is a way to
 *   reduce the amount of computation needed in the forward backward at the
 *   price of a more complex implementation. Due to the fact that using a sparse
 *   matrix have a cost, this implementation is slower on L2 regularized models
 *   and on lighty L1-regularized models, this is why there is also a classical
 *   dense version of the algorithm used for example by the L-BFGS trainer.
 *
 *   The sparse matrix implementation is a bit tricky because we need to store
 *   all values in sequences in order to use the vector exponential who gives
 *   also a lot of performance improvement on vector able machine.
 *   We need four arrays noted <val>, <off>, <idx>, and <yp>. For each positions
 *   t, <off>[t] value indicate where the non-zero values for t starts in <val>.
 *   The other arrays gives the y and yp indices of these values. The easier one
 *   to retrieve is yp, the yp indice for value at <val>[<off>[t] + n] is stored
 *   at the same position in <yp>.
 *   The y are more difficult: the indice y are stored with n between <idx>[y-1]
 *   and <idx>[y]. It may seems inefective but the matrix is indexed in the
 *   other way, we go through the idx array, and for each y we get the yp and
 *   values, so in practice it's very efficient.
 *
 *   This can seem too complex but we have to keep in mind that Y are generally
 *   very low and any sparse-matrix have overhead so we have to reduce it to the
 *   minimum in order to get a real improvment. Dedicated library are optimized
 *   for bigger matrix where the overhead is not a so important problem.
 *   Another problem here is cache size. The optimization process will last most
 *   of his time in this function so it have to be well optimized and we already
 *   need a lot of memory for other data so we have to be carefull here if we
 *   don't want to flush the cache all the time. Sparse matrix require less
 *   memory than dense one only if we now in advance the number of non-zero
 *   entries, which is not the case here, so we have to use a scheme which in
 *   the worst case use as less as possible memory.
 ******************************************************************************/

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

/* grd_new:
 *   Allocation memory for gradient computation state. This allocate memory for
 *   the longest sequence present in the data set.
 */
static grd_t *grd_new(mdl_t *mdl, double *g) {
	const size_t Y = mdl->nlbl;
	const int    T = mdl->train->mlen;
	grd_t *grd = xmalloc(sizeof(grd_t));
	grd->mdl   = mdl;
	grd->g     = g;
	grd->psi   = xvm_alloc(sizeof(double) * T * Y * Y);
	grd->alpha = xmalloc(sizeof(double) * T * Y);
	grd->beta  = xmalloc(sizeof(double) * T * Y);
	grd->scale = xmalloc(sizeof(double) * T);
	grd->unorm = xmalloc(sizeof(double) * T);
	grd->bnorm = xmalloc(sizeof(double) * T);
	if (mdl->opt->sparse) {
		grd->psiuni = xvm_alloc(sizeof(double) * T * Y);
		grd->psiyp  = xmalloc(sizeof(double) * T * Y * Y);
		grd->psiidx = xmalloc(sizeof(double) * T * Y);
		grd->psioff = xmalloc(sizeof(double) * T);
	}
	return grd;
}

/* grd_free:
 *   Free all memory used by gradient computation.
 */
static void grd_free(grd_t *grd) {
	if (grd->mdl->opt->sparse) {
		xvm_free(grd->psiuni);
		free(grd->psiyp);
		free(grd->psiidx);
		free(grd->psioff);
	}
	xvm_free(grd->psi);
	free(grd->bnorm);
	free(grd->unorm);
	free(grd->scale);
	free(grd->beta);
	free(grd->alpha);
	free(grd);
}

/* grd_fldopsi:
 *   We first have to compute the Ψ_t(y',y,x) weights defined as
 *       Ψ_t(y',y,x) = \exp( ∑_k θ_k f_k(y',y,x_t) )
 *   So at position 't' in the sequence, for each couple (y',y) we have to sum
 *   weights of all features. Only the observations present at this position
 *   will have a non-nul weight so we can sum only on thoses. As we use only two
 *   kind of features: unigram and bigram, we can rewrite this as
 *       \exp (  ∑_k μ_k(y, x_t)     f_k(y, x_t)
 *             + ∑_k λ_k(y', y, x_t) f_k(y', y, x_t) )
 *   Where the first sum is over the unigrams features and the second is over
 *   bigrams ones.
 *   This allow us to compute Ψ efficiently in three steps
 *     1/ we sum the unigrams features weights by looping over actives
 *          unigrams observations. (we compute this sum once and use it
 *          for each value of y')
 *     2/ we add the bigrams features weights by looping over actives
 *          bigrams observations (we don't have to do this for t=0 since
 *          there is no bigrams here)
 *     3/ we take the component-wise exponential of the resulting matrix
 *          (this can be done efficiently with vector maths)
 */
static void grd_fldopsi(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	double (*psi)[T][Y][Y] = (void *)grd->psi;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				sum += x[mdl->uoff[o] + y];
			}
			for (size_t yp = 0; yp < Y; yp++)
				(*psi)[t][yp][y] = sum;
		}
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double sum = 0.0;
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					sum += x[mdl->boff[o] + d];
				}
				(*psi)[t][yp][y] += sum;
			}
		}
	}
	xvm_expma((double *)psi, (double *)psi, 0.0, (size_t)T * Y * Y);
}

/* grd_spdopsi:
 *   For the sparse version, we keep the two sum separate so we will have
 *   separate Ψ_t(y,x) and Ψ_t(y',y,x). The first one define a vector for
 *   unigram at each position, and the second one a matrix for bigrams.  This is
 *   where the trick is as we will store Ψ_t(y',y,x) - 1. If the sum is nul, his
 *   exponential will be 1.0 and so we have to store 0.0.  As most of the sum
 *   are expected to be nul the resulting matrix will be very sparse and we will
 *   save computation in the forward-backward.
 *
 *   So we compute Ψ differently here
 *     1/ we sum the unigrams features weights by looping over actives
 *          unigrams observations and store them in |psiuni|.
 *     2/ we sum the bigrams features weights by looping over actives
 *          bigrams observations (we don't have to do this for t=0 since
 *          there is no bigrams here) and we store the non-nul one in the
 *          sparse matrix.
 *     3/ we take the component-wise exponential of the unigrams vectors,
 *          and the component-wise exponential of the sparse matrix minus
 *          one. (here also this can be done efficiently with vector
 *          maths)
 */
static void grd_spdopsi(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	double (*psiuni)[T][Y] = (void *)grd->psiuni;
	double  *psival        =         grd->psi;
	size_t  *psiyp         =         grd->psiyp;
	size_t (*psiidx)[T][Y] = (void *)grd->psiidx;
	size_t  *psioff        =         grd->psioff;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				sum += x[mdl->uoff[o] + y];
			}
			(*psiuni)[t][y] = sum;
		}
	}
	size_t off = 0;
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		psioff[t] = off;
		for (size_t y = 0, nnz = 0; y < Y; y++) {
			for (size_t yp = 0; yp < Y; yp++) {
				double sum = 0.0;
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					sum += x[mdl->boff[o] + yp * Y + y];
				}
				if (sum == 0.0)
					continue;
				psiyp [off] = yp;
				psival[off] = sum;
				nnz++, off++;
			}
			(*psiidx)[t][y] = nnz;
		}
	}
	xvm_expma((double *)psiuni, (double *)psiuni, 0.0, (size_t)T * Y);
	xvm_expma((double *)psival, (double *)psival, 1.0, off);
}

/* grd_flfwdbwd:
 *   Now, we go to the forward-backward algorithm. As this part of the code rely
 *   on a lot of recursive sums and products of exponentials, we have to take
 *   care of numerical problems.
 *   First the forward recursion
 *       | α_1(y) = Ψ_1(y,x)
 *       | α_t(y) = ∑_{y'} α_{t-1}(y') * Ψ_t(y',y,x)
 *   Next come the backward recursion which is very similar
 *       | β_T(y') = 1
 *       | β_t(y') = ∑_y β_{t+1}(y) * Ψ_{t+1}(y',y,x)
 *   The numerical problems can appear here. To solve them we will scale the α_t
 *   and β_t vectors so they sum to 1 but we have to keep the scaling coeficient
 *   as we will need them later.
 *   Now, we have to compute the nomalization factor. But, due to the scaling
 *   performed during the forward-backward recursions, we have to compute it at
 *   each positions and separately for unigrams and bigrams using
 *       for unigrams: Z_θ(t) = ∑_y α_t(y) β_t(y)
 *       for bigrams:  Z_θ(t) = ∑_y α_t(y) β_t(y) / α-scale_t
 *   with α-scale_t the scaling factor used for the α vector at position t
 *   in the forward recursion.
 */
static void grd_flfwdbwd(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const size_t Y = mdl->nlbl;
	const int    T = seq->len;
	const double (*psi)[T][Y][Y] = (void *)grd->psi;
	double (*alpha)[T][Y] = (void *)grd->alpha;
	double (*beta )[T][Y] = (void *)grd->beta;
	double  *scale        =         grd->scale;
	double  *unorm        =         grd->unorm;
	double  *bnorm        =         grd->bnorm;
	for (size_t y = 0; y < Y; y++)
		(*alpha)[0][y] = (*psi)[0][0][y];
	scale[0] = xvm_unit((*alpha)[0], (*alpha)[0], Y);
	for (int t = 1; t < grd->last + 1; t++) {
		for (size_t y = 0; y < Y; y++) {
			double sum = 0.0;
			for (size_t yp = 0; yp < Y; yp++)
				sum += (*alpha)[t - 1][yp] * (*psi)[t][yp][y];
			(*alpha)[t][y] = sum;
		}
		scale[t] = xvm_unit((*alpha)[t], (*alpha)[t], Y);
	}
	for (size_t yp = 0; yp < Y; yp++)
		(*beta)[T - 1][yp] = 1.0 / Y;
	for (int t = T - 1; t > grd->first; t--) {
		for (size_t yp = 0; yp < Y; yp++) {
			double sum = 0.0;
			for (size_t y = 0; y < Y; y++)
				sum += (*beta)[t][y] * (*psi)[t][yp][y];
			(*beta)[t - 1][yp] = sum;
		}
		xvm_unit((*beta)[t - 1], (*beta)[t - 1], Y);
	}
	for (int t = 0; t < T; t++) {
		double z = 0.0;
		for (size_t y = 0; y < Y; y++)
			z += (*alpha)[t][y] * (*beta)[t][y];
		unorm[t] = 1.0 / z;
		bnorm[t] = scale[t] / z;
	}
}

/* grd_spfwdbwd:
 *   And the sparse version which is a bit more cmoplicated but follow the same
 *   general path. First the forward recursion
 *       | α_1(y) = Ψ_1(y,x)
 *       | α_t(y) = Ψ_t(y,x) * (   ∑_{y'} α_{t-1}(y')
 *                               + ∑_{y'} α_{t-1}(y') * (Ψ_t(y',y,x) - 1) )
 *   The inner part contains two sums, the first one will be 1.0 as we scale the
 *   α vectors, and the second is a sparse matrix multiplication who need less
 *   than |Y|x|Y| multiplication if the matrix is really sparse, so we will gain
 *   here.
 *   Next come the backward recursion which is very similar
 *       | β_T(y') = 1
 *       | β_t(y') = ∑_y v_{t+1}(y) + ∑_y v_{t+1}(y) * (Ψ_{t+1}(y',y,x) - 1)
 *   with
 *       v_{t+1}(y) = β_{t+1}(y) * Ψ_{t+1}(y,x)
 *   And here also we reduce the number of multiplication if the matrix is
 *   really sparse.
 */
static void grd_spfwdbwd(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const size_t Y = mdl->nlbl;
	const int    T = seq->len;
	const double (*psiuni)[T][Y] = (void *)grd->psiuni;
	const double  *psival        =         grd->psi;
	const size_t  *psiyp         =         grd->psiyp;
	const size_t (*psiidx)[T][Y] = (void *)grd->psiidx;
	const size_t  *psioff        =         grd->psioff;
	double (*alpha)[T][Y] = (void *)grd->alpha;
	double (*beta )[T][Y] = (void *)grd->beta;
	double  *scale        =         grd->scale;
	double  *unorm        =         grd->unorm;
	double  *bnorm        =         grd->bnorm;
	for (size_t y = 0; y < Y; y++)
		(*alpha)[0][y] = (*psiuni)[0][y];
	scale[0] = xvm_unit((*alpha)[0], (*alpha)[0], Y);
	for (int t = 1; t < grd->last + 1; t++) {
		for (size_t y = 0; y < Y; y++)
			(*alpha)[t][y] = 1.0;
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < (*psiidx)[t][Y - 1]; ) {
			while (n >= (*psiidx)[t][y])
				y++;
			while (n < (*psiidx)[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				(*alpha)[t][y] += (*alpha)[t - 1][yp] * v;
				n++;
			}
		}
		for (size_t y = 0; y < Y; y++)
			(*alpha)[t][y] *= (*psiuni)[t][y];
		scale[t] = xvm_unit((*alpha)[t], (*alpha)[t], Y);
	}
	for (size_t yp = 0; yp < Y; yp++)
		(*beta)[T - 1][yp] = 1.0 / Y;
	for (int t = T - 1; t > grd->first; t--) {
		double sum = 0.0, tmp[Y];
		for (size_t y = 0; y < Y; y++) {
			tmp[y] = (*beta)[t][y] * (*psiuni)[t][y];
			sum += tmp[y];
		}
		for (size_t y = 0; y < Y; y++)
			(*beta)[t - 1][y] = sum;
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < (*psiidx)[t][Y - 1]; ) {
			while (n >= (*psiidx)[t][y])
				y++;
			while (n < (*psiidx)[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				(*beta)[t - 1][yp] += v * tmp[y];
				n++;
			}
		}
		xvm_unit((*beta)[t - 1], (*beta)[t - 1], Y);
	}
	for (int t = 0; t < T; t++) {
		double z = 0.0;
		for (size_t y = 0; y < Y; y++)
			z += (*alpha)[t][y] * (*beta)[t][y];
		unorm[t] = 1.0 / z;
		bnorm[t] = scale[t] / z;
	}
}

/* grd_flupgrad:
 *   Now, we have all we need to compute the gradient of the negative log-
 *   likelihood
 *       ∂-L(θ)
 *       ------ =    ∑_t ∑_{(y',y)} f_k(y',y,x_t) p_θ(y_{t-1}=y',y_t=y|x)
 *        ∂θ_k     - ∑_t f_k(y_{t-1},y_t,x_t)
 *
 *   The first term is the expectation of f_k under the model distribution and
 *   the second one is the expectation of f_k under the empirical distribution.
 *
 *   The second is very simple to compute as we just have to sum over the
 *   actives observations in the sequence. The first one is more tricky as it
 *   involve computing the probability p_θ. This is where we use all the
 *   previous computations. Again we separate the computations for unigrams and
 *   bigrams here.
 *
 *   These probabilities are given by
 *       p_θ(y_t=y|x)            = α_t(y)β_t(y) / Z_θ
 *       p_θ(y_{t-1}=y',y_t=y|x) = α_{t-1}(y') Ψ_t(y',y,x) β_t(y) / Z_θ
 *   but we have to remember that, since we have scaled the α and β, we have to
 *   use the local normalization constants.
 *
 *   We must also take care of not clearing previous value of the gradient
 *   vector but just adding the contribution of this sequence. This allow to
 *   compute it easily the gradient over more than one sequence.
 */
static void grd_flupgrad(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const size_t Y = mdl->nlbl;
	const int    T = seq->len;
	const double (*psi  )[T][Y][Y] = (void *)grd->psi;
	const double (*alpha)[T][Y]    = (void *)grd->alpha;
	const double (*beta )[T][Y]    = (void *)grd->beta;
	const double  *unorm           =         grd->unorm;
	const double  *bnorm           =         grd->bnorm;
	double *g = grd->g;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// Add the expectation over the model distribution
		for (size_t y = 0; y < Y; y++) {
			double e = (*alpha)[t][y] * (*beta)[t][y] * unorm[t];
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				g[mdl->uoff[o] + y] += e;
			}
		}
		// And substract the expectation over the empirical one.
		const size_t y = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			g[mdl->uoff[pos->uobs[n]] + y] -= 1.0;
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// Add the expectation over the model distribution
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double e = (*alpha)[t - 1][yp] * (*beta)[t][y]
				         * (*psi)[t][yp][y] * bnorm[t];
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					g[mdl->boff[o] + d] += e;
				}
			}
		}
		// And substract the expectation over the empirical one.
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		const size_t d  = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			g[mdl->boff[pos->bobs[n]] + d] -= 1.0;
	}
}

/* grd_flupgrad:
 *   The sparse matrix make things a bit more complicated here as we cannot
 *   directly multiply with the original Ψ_t(y',y,x) because we have split it
 *   two components and the second one is sparse, so we have to make a quite
 *   complex workaround to fix that. We have to explicitly build the expectation
 *   matrix. We first fill it with the unigram component and next multiply it
 *   with the bigram one.
 */
static void grd_spupgrad(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	const double (*psiuni)[T][Y] = (void *)grd->psiuni;
	const double  *psival        =         grd->psi;
	const size_t  *psiyp         =         grd->psiyp;
	const size_t (*psiidx)[T][Y] = (void *)grd->psiidx;
	const size_t  *psioff        =         grd->psioff;
	const double (*alpha)[T][Y]  = (void *)grd->alpha;
	const double (*beta )[T][Y]  = (void *)grd->beta;
	const double  *unorm         =         grd->unorm;
	const double  *bnorm         =         grd->bnorm;
	double *g = grd->g;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// Add the expectation over the model distribution
		for (size_t y = 0; y < Y; y++) {
			double e = (*alpha)[t][y] * (*beta)[t][y] * unorm[t];
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				g[mdl->uoff[o] + y] += e;
			}
		}
		// And substract the expectation over the empirical one.
		const size_t y = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			g[mdl->uoff[pos->uobs[n]] + y] -= 1.0;
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// We build the expectation matrix
		double e[Y][Y];
		for (size_t yp = 0; yp < Y; yp++)
			for (size_t y = 0; y < Y; y++)
				e[yp][y] = (*alpha)[t - 1][yp] * (*beta)[t][y]
				         * (*psiuni)[t][y] * bnorm[t];
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < (*psiidx)[t][Y - 1]; ) {
			while (n >= (*psiidx)[t][y])
				y++;
			while (n < (*psiidx)[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				e[yp][y] += e[yp][y] * v;
				n++;
			}
		}
		// Add the expectation over the model distribution
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				for (size_t n = 0; n < pos->bcnt; n++) {
					const size_t o = pos->bobs[n];
					g[mdl->boff[o] + d] += e[yp][y];
				}
			}
		}
		// And substract the expectation over the empirical one.
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		const size_t d  = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			g[mdl->boff[pos->bobs[n]] + d] -= 1.0;
	}
}

/* grd_logloss:
 *   And the final touch, the computation of the negative log-likelihood
 *       -L(θ) = log(Z_θ) - ∑_t ∑_k θ_k f_k(y_{t-1}, y_t, x_t)
 *
 *   The numerical problems show again here as we cannot compute the Z_θ
 *   directly for the same reason we have done scaling. Fortunately, there is a
 *   way to directly compute his logarithm
 *       log(Z_θ) = log( ∑_y α_t(y) β_t(y) )
 *                - ∑_{i=1..t} log(α-scale_i)
 *                - ∑_{i=t..T} log(β-scale_i)
 *   for any value of t.
 * 
 *   So we can compute it at any position in the sequence but the last one is
 *   easier as the value of β_T(y) and β-scale_T are constant and cancel out.
 *   This is why we have just keep the α-scale_t values.
 *
 *   Now, we have the first term of -L(θ). We have now to substract the second
 *   one. As we have done for the computation of Ψ, we separate the sum over K
 *   in two sums, one for unigrams and one for bigrams. And, as here also the
 *   weights will be non-nul only for observations present in the sequence, we
 *   sum only over these ones.
 */
static void grd_logloss(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	const double (*alpha)[T][Y] = (void *)grd->alpha;
	const double  *scale        =         grd->scale;
	double logz = 0.0;
	for (size_t y = 0; y < Y; y++)
		logz += (*alpha)[T - 1][y];
	logz = log(logz);
	for (int t = 0; t < T; t++)
		logz -= log(scale[t]);
	double lloss = logz;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t y   = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			lloss -= x[mdl->uoff[pos->uobs[n]] + y];
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t yp  = seq->pos[t - 1].lbl;
		const size_t y   = seq->pos[t    ].lbl;
		const size_t d   = yp * Y + y;
		for (size_t n = 0; n < pos->bcnt; n++)
			lloss -= x[mdl->boff[pos->bobs[n]] + d];
	}
	grd->lloss += lloss;
}

/* grd_doseq:
 *   This function compute the gradient and value of the negative log-likelihood
 *   of the model over a single training sequence.
 *
 *   This function will not clear the gradient before computation, but instead
 *   just accumulate the values for the given sequence in it. This allow to
 *   easily compute the gradient over a set of sequences.
 */
static void grd_doseq(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	grd->first = 0;
	grd->last  = seq->len - 1;
	if (!mdl->opt->sparse) {
		grd_fldopsi(grd, seq);
		grd_flfwdbwd(grd, seq);
		grd_flupgrad(grd, seq);
	} else {
		grd_spdopsi(grd, seq);
		grd_spfwdbwd(grd, seq);
		grd_spupgrad(grd, seq);
	}
	grd_logloss(grd, seq);
}

/******************************************************************************
 * Dataset gradient computation
 *
 *   This section is responsible for computing the gradient of the
 *   log-likelihood function to optimize over the full training set.
 *
 *   The gradient computation is multi-threaded, you first have to call the
 *   function 'grd_setup' to prepare the workers pool, and next you can use
 *   'grd_gradient' to ask for the full gradient as many time as you want. Each
 *   time the gradient is computed over the full training set, using the curent
 *   value of the parameters and applying the regularization. If need the
 *   pseudo-gradient can also be computed. When you have done, you have to call
 *   'grd_cleanup' to free the allocated memory.
 *
 *   This require an additional vector of size <nftr> per thread after the
 *   first, so it can take a lot of memory to compute big models on a lot of
 *   threads. It is strongly discouraged to ask for more threads than you have
 *   cores, or to more thread than you have memory to hold vectors.
 ******************************************************************************/

/* grd_worker:
 *   This is a simple function who compute the gradient over a subset of the
 *   training set. It is mean to be called by the thread spawner in order to
 *   compute the gradient over the full training set.
 */
static void grd_worker(int id, int cnt, grd_t *grd) {
	mdl_t *mdl = grd->mdl;
	const dat_t *dat = mdl->train;
	const size_t F = mdl->nftr;
	// We first cleanup the gradient and value as our parent don't do it (it
	// is better to do this also in parallel)
	grd->lloss = 0.0;
	for (size_t f = 0; f < F; f++)
		grd->g[f] = 0.0;
	// Now all is ready, we can process our sequences and accumulate the
	// gradient and inverse log-likelihood
	for (int s = id; !uit_stop && s < dat->nseq; s += cnt)
		grd_doseq(grd, dat->seq[s]);
}

/* grd_gradient:
 *   Compute the gradient and value of the negative log-likelihood of the model
 *   at current point. It will also compute the pseudo gradient for owl-qn if
 *   the 'pg' vector is not NULL.
 *   The computation is done in parallel taking profit of the fact that the
 *   gradient over the full training set is just the sum of the gradient of
 *   each sequence.
 */
static double grd_gradient(mdl_t *mdl, double *g, double *pg, grd_t *grds[]) {
	const double *x = mdl->theta;
	const size_t  F = mdl->nftr;
	const size_t  W = mdl->opt->nthread;
	// All is ready to compute the gradient, we spawn the threads of
	// workers, each one working on a part of the data. As the gradient and
	// log-likelihood are additive, computing the final values will be
	// trivial.
	if (W == 1)
		grd_worker(1, 1, grds[0]);
	else
		mth_spawn((func_t *)grd_worker, W, (void **)grds);
	if (uit_stop)
		return -1.0;
	// All computations are done, it just remain to add all the gradients
	// and inverse log-likelihood from all the workers.
	double fx = grds[0]->lloss;
	for (size_t w = 1; w < W; w++) {
		for (size_t f = 0; f < F; f++)
			g[f] += grds[w]->g[f];
		fx += grds[w]->lloss;
	}
	// If needed we clip the gradient: setting to 0.0 all coordinate where
	// the function is 0.0.
	if (mdl->opt->lbfgs.clip == true)
		for (size_t f = 0; f < F; f++)
			if (x[f] == 0.0)
				g[f] = 0.0;
	// Now we can apply the elastic-net penalty. Depending of the values of
	// rho1 and rho2, this can in fact be a classical L1 or L2 penalty.
	const double rho1 = mdl->opt->rho1;
	const double rho2 = mdl->opt->rho2;
	double nl1 = 0.0, nl2 = 0.0;
	for (size_t f = 0; f < F; f++) {
		const double v = x[f];
		g[f] += rho2 * v;
		nl1  += fabs(v);
		nl2  += v * v;
	}
	fx += nl1 * rho1 + nl2 * rho2 / 2.0;
	// And the last step is to compute the pseudo gradient for owl-qn if
	// requested by the caller. It is define in [3, pp 35(4)]
	//              | ∂_i^- f(x) if ∂_i^- f(x) > 0
	//   ◇_i f(x) = | ∂_i^+ f(x) if ∂_i^+ f(x) < 0
	//              | 0          otherwise
	// with
	//   ∂_i^± f(x) = ∂/∂x_i l(x) + | Cσ(x_i) if x_i ≠ 0
	//                              | ±C      if x_i = 0
	if (pg != NULL) {
		for (size_t f = 0; f < F; f++) {
			if (x[f] < 0.0)
				pg[f] = g[f] - rho1;
			else if (x[f] > 0.0)
				pg[f] = g[f] + rho1;
			else if (g[f] < -rho1)
				pg[f] = g[f] + rho1;
			else if (g[f] > rho1)
				pg[f] = g[f] - rho1;
			else
				pg[f] = 0.0;
		}
	}
	return fx;
}

/******************************************************************************
 * Quasi-Newton optimizer
 *
 *   This section implement the quasi-Newton optimizer. We use the L-BFGS
 *   algorithm described by Liu and Nocedal in [1] and [2]. If an l1-norm must
 *   be applyed we fallback on the OWL-QN variant described in [3] by Galen and
 *   Jianfeng which allow to use L-BFGS for function not differentiable in 0.0.
 *
 *   [1] Updating quasi-Newton matrices with limited storage, Jorge Nocedal, in
 *       Mathematics of Computation, vol. 35(151) 773-782, July 1980.
 *   [2] On the limited memory BFGS method for large scale optimization, Dong C.
 *       Liu and Jorge Nocedal, in Mathematical Programming, vol. 45(1) 503-528,
 *       January 1989.
 *   [3] Scalable Training of L1-Regularized Log-Linear Models, Andrew Galen and
 *       Gao Jianfeng, in Proceedings of the 24th International Conference on
 *       Machine Learning (ICML), Corvallis, OR, 2007.
 ******************************************************************************/

static void trn_lbfgs(mdl_t *mdl) {
	const size_t F  = mdl->nftr;
	const int    K  = mdl->opt->maxiter;
	const int    M  = mdl->opt->lbfgs.histsz;
	const size_t W  = mdl->opt->nthread;
	const bool   l1 = mdl->opt->rho1 != 0.0;
	double *x, *xp; // Current and previous value of the variables
	double *g, *gp; // Current and previous value of the gradient
	double *pg;     // The pseudo-gradient (only for owl-qn)
	double *d;      // The search direction
	double *s[M];   // History value s_k = Δ(x,px)
	double *y[M];   // History value y_k = Δ(g,pg)
	double  p[M];   // ρ_k
	grd_t  *grds[W];
	// Initialization: Here, we have to allocate memory on the heap as we
	// cannot request so much memory on the stack as this will have a too
	// big impact on performance and will be refused by the system on non-
	// trivial models.
	// To make things simpler, we allocate all the memory in one call to
	// malloc and dispatch memory in the various arrays. The main pointer
	// will remain in the raw variable to be freed at the end.
	double *raw = xmalloc(sizeof(double) * F * (4 + M * 2 + l1 + W));
	double *tmp = raw;
	x  = mdl->theta;
	xp = tmp; tmp += F; g = tmp; tmp += F;
	gp = tmp; tmp += F; d = tmp; tmp += F;
	for (int m = 0; m < M; m++) {
		s[m] = tmp; tmp += F;
		y[m] = tmp; tmp += F;
	}
	pg = NULL;
	if (l1 == true)
		pg = tmp, tmp += F;
	grds[0] = grd_new(mdl, g);
	for (size_t w = 1; w < W; w++)
		grds[w] = grd_new(mdl, tmp), tmp += F;
	// Minimization: This is the heart of the function. (a big heart...) We
	// will perform iterations until one these conditions is reached
	//   - the maximum iteration count is reached
	//   - we have converged (upto numerical precision)
	//   - the report function return false
	//   - an error happen somewhere
	double fx = grd_gradient(mdl, g, pg, grds);
	for (int k = 0; !uit_stop && k < K; k++) {
		// 1st step: We compute the search direction. We search in the
		// direction who minimize the second order approximation given
		// by the Taylor series which give
		//   d_k = - H_k^{-1} g_k
		// But computing the inverse of the hessian is intractable so
		// the l-bfgs only approximate it's diagonal. The exact
		// computation is well described in [1, pp 779].
		// The only special thing for owl-qn here is to use the pseudo
		// gradient instead of the true one.
		for (size_t f = 0; f < F; f++)
			d[f] = l1 ? -pg[f] : -g[f];
		if (k != 0) {
			const int km = k % M;
			const int bnd = (k <= M) ? k : M;
			double alpha[M], beta;
			// α_i = ρ_j s_j^T q_{i+1}
			// q_i = q_{i+1} - α_i y_i
			for (int i = bnd; i > 0; i--) {
				const int j = (k - i + M) % M;
				alpha[i - 1] = p[j] * xvm_dot(s[j], d, F);
				xvm_axpy(d, -alpha[i - 1], y[j], d, F);
			}
			// r_0 = H_0 q_0
			//     Scaling is described in [2, pp 515]
			//     for k = 0: H_0 = I
			//     for k > 0: H_0 = I * y_k^T s_k / ||y_k||²
			//                    = I * 1 / ρ_k ||y_k||²
			const double y2 = xvm_dot(y[km], y[km], F);
			const double v = 1.0 / (p[km] * y2);
			for (size_t f = 0; f < F; f++)
				d[f] *= v;
			// β_j     = ρ_j y_j^T r_i
			// r_{i+1} = r_i + s_j (α_i - β_i)
			for (int i = 0; i < bnd; i++) {
				const int j = (k - i - 1 + M) % M;
				beta = p[j] * xvm_dot(y[j], d, F);
				xvm_axpy(d, alpha[i] - beta, s[j], d, F);
			}
		}
		// For owl-qn, we must remain in the same orthant than the
		// pseudo-gradient, so we have to constrain the search
		// direction as described in [3, pp 35(3)]
		//   d^k = π(d^k ; v^k)
		//       = π(d^k ; -◇f(x^k))
		if (l1)
			for (size_t f = 0; f < F; f++)
				if (d[f] * pg[f] >= 0.0)
					d[f] = 0.0;
		// 2nd step: we perform a linesearch in the computed direction,
		// we search a step value that satisfy the constrains using a
		// backtracking algorithm. Much elaborated algorithm can perform
		// better in the general case, but for CRF training, bactracking
		// is very efficient and simple to implement.
		// For quasi-Newton, the natural step is 1.0 so we start with
		// this one and reduce it only if it fail with an exception for
		// the first step where a better guess can be done.
		// We have to keep track of the current point and gradient as we
		// will need to compute the delta between those and the found
		// point, and perhaps need to restore them if linesearch fail.
		memcpy(xp, x, sizeof(double) * F);
		memcpy(gp, g, sizeof(double) * F);
		double gd = l1 ? 0.0 : xvm_dot(g, d, F); // gd = g_k^T d_k
		double stp = 1.0, fi = fx;
		if (k == 0)
			stp = 1.0 / xvm_norm(d, F);
		double sc = 0.5;
		bool err = false;
		for (int ls = 1; !uit_stop; ls++, stp *= sc) {
			// We compute the new point using the current step and
			// search direction
			xvm_axpy(x, stp, d, xp, F);
			// For owl-qn, we have to project back the point in the
			// current orthant [3, pp 35]
			//   x^{k+1} = π(x^k + αp^k ; ξ)
			if (l1) {
				for (size_t f = 0; f < F; f++) {
					double or = xp[f];
					if (or == 0.0)
						or = -pg[f];
					if (x[f] * or <= 0.0)
						x[f] = 0.0;
				}
			}
			// And we ask for the value of the objective function
			// and its gradient and pseudo gradient.
			fx = grd_gradient(mdl, g, pg, grds);
			// Now we check if the step satisfy the conditions. For
			// l-bfgs, we check the classical decrease and curvature
			// known as the Wolfe conditions [2, pp 506]
			//   f(x_k + α_k d_k) ≤ f(x_k) + β' α_k g_k^T d_k
			//   g(x_k + α_k d_k)^T d_k ≥ β g_k^T d_k
			//
			// And for owl-qn we check a variant of the Armijo rule
			// described in [3, pp 36]
			//   f(π(x^k+αp^k;ξ)) ≤ f(x^k) - γv^T[π(x^k+αp^k;ξ)-x^k]
			if (!l1) {
				if (fx > fi + stp * gd * 1e-4)
					sc = 0.5;
				else if (xvm_dot(g, d, F) < gd * 0.9)
					sc = 2.1;
				else
					break;
			} else {
				double vp = 0.0;
				for (size_t f = 0; f < F; f++)
					vp += (x[f] - xp[f]) * d[f];
				if (fx < fi + vp * 1e-4)
					break;
			}
			// If we reach the maximum number of linesearsh steps
			// without finding a good one, we just fail.
			if (ls == mdl->opt->lbfgs.maxls) {
				warning("maximum linesearch reached");
				err = true;
				break;
			}
		}
		// If linesearch failed or user interupted training, we return
		// to the last valid point and stop the training. The model is
		// probably not fully optimized but we let the user decide what
		// to do with it.
		if (err || uit_stop) {
			memcpy(x, xp, sizeof(double) * F);
			break;
		}
		if (uit_progress(mdl, k + 1, fx) == false)
			break;
		// 3rd step: We check for convergence and if not, we update the
		// history to prepare the next iteration. The convergence check
		// is quite simple [2, pp 508]
		//   ||g|| / max(1, ||x||) ≤ ε
		// with ε small enough so we stop when numerical precision is
		// reached. For owl-qn we just have to check against the pseudo-
		// gradient instead of the true one.
		const double xn = xvm_norm(x, F);
		const double gn = xvm_norm(l1 ? pg : g, F);
		if (gn / max(xn, 1.0) <= 1e-5)
			break;
		if (k + 1 == K)
			break;
		// So, last we update the history used for approximating the
		// inverse of the diagonal of the hessian
		//   s_k = x_{k+1} - x_k
		//   y_k = g_{k+1} - g_k
		//   ρ_k = 1 / y_k^T s_k
		const int kn = (k + 1) % M;
		for (size_t f = 0; f < F; f++) {
			s[kn][f] = x[f] - xp[f];
			y[kn][f] = g[f] - gp[f];
		}
		p[kn] = 1.0 / xvm_dot(y[kn], s[kn], F);
	}
	// Cleanup: This is very simple as we have carefully allocated memory in
	// a sigle block, we must not forget to free it.
	free(raw);
	for (size_t w = 0; w < W; w++)
		grd_free(grds[w]);
}

/******************************************************************************
 * The SGD-L1 trainer
 *
 *   Implementation of the stochatic gradient descend with L1 penalty described
 *   in [1] by Tsurukoa et al. This allow to build really sparse models with the
 *   SGD method.
 *
 *   [1] Stochastic gradient descent training for L1-regularized log-linear
 *       models with cumulative penalty, Yoshimasa Tsuruoka and Jun'ichi Tsuji
 *       and Sophia Ananiadou, in Proceedings of the ACL and the 4th IJCNLP of
 *       the AFNLP, pages 477-485, August 2009
 ******************************************************************************/
typedef struct sgd_idx_s {
	size_t *uobs;
	size_t *bobs;
} sgd_idx_t;

/* applypenalty:
 *   This macro is quite ugly as it make a lot of things and use local variables
 *   of the function below. I'm sorry for this but this is allow to not
 *   duplicate the code below. Due to the way unigrams and bigrams observation
 *   are stored we must use this two times. As this macro is dangerous when
 *   called outsize of sgd-l1 we undef it just after.
 *   This function match exactly the APPLYPENALTY function defined in [1] pp 481
 *   and the formula on the middle of the page 480.
 */
#define applypenalty(f) do {                               \
	const double z = w[f];                             \
	if      (z > 0.0) w[f] = max(0.0, z - (u + q[f])); \
	else if (z < 0.0) w[f] = min(0.0, z + (u - q[f])); \
	q[f] += w[f] - z;                                  \
} while (false)

/* trn_sgdl1:
 *   Train the model with the SGD-l1 algorithm described by tsurukoa et al.
 */
static void trn_sgdl1(mdl_t *mdl) {
	const size_t  Y = mdl->nlbl;
	const size_t  O = mdl->nobs;
	const size_t  F = mdl->nftr;
	const int     S = mdl->train->nseq;
	const int     K = mdl->opt->maxiter;
	      double *w = mdl->theta;
	// First we have to build and index who hold, for each sequences, the
	// list of actives observations.
	// The index is a simple table indexed by sequences number. Each entry
	// point to two lists of observations terminated by <none>, one for
	// unigrams obss and one for bigrams obss.
	info("    - Build the index\n");
	sgd_idx_t *idx  = xmalloc(sizeof(sgd_idx_t) * S);
	int       *mark = xmalloc(sizeof(int) * O);
	for (size_t o = 0; o < O; o++)
		mark[o] = -1;
	for (int s = 0; s < S; s++) {
		const seq_t *seq = mdl->train->seq[s];
		// Listing active observations in sequence is easy, we scan
		// unigrams and bigrams observations list and mark the actives
		// one in the <mark> array with the sequence number. Next we
		// can scan this array to search the marked obss.
		for (int t = 0; t < seq->len; t++) {
			const pos_t *pos = &seq->pos[t];
			for (size_t p = 0; p < pos->ucnt; p++)
				mark[pos->uobs[p]] = s;
			for (size_t p = 0; p < pos->bcnt; p++)
				mark[pos->bobs[p]] = s;
		}
		// We scan the <mark> array a first time to count the number of
		// active sequences and allocate memory.
		size_t ucnt = 1, bcnt = 1;
		for (size_t o = 0; o < O; o++) {
			ucnt += (mark[o] == s) && (mdl->kind[o] & 1);
			bcnt += (mark[o] == s) && (mdl->kind[o] & 2);
		}
		idx[s].uobs = xmalloc(sizeof(size_t) * ucnt);
		idx[s].bobs = xmalloc(sizeof(size_t) * bcnt);
		// And a second time to fill the allocated array without
		// forgetting to set the end marker.
		size_t upos = 0, bpos = 0;
		for (size_t o = 0; o < O; o++) {
			if ((mark[o] == s) && (mdl->kind[o] & 1))
				idx[s].uobs[upos++] = o;
			if ((mark[o] == s) && (mdl->kind[o] & 2))
				idx[s].bobs[bpos++] = o;
		}
		idx[s].uobs[upos] = none;
		idx[s].bobs[bpos] = none;
	}
	free(mark);
	info("      Done\n");
	// We will process sequences in random order in each iteration, so we
	// will have to permute them. The current permutation is stored in a
	// vector called <perm> shuffled at the start of each iteration. We
	// just initialize it with the identity permutation.
	// As we use the same gradient function than the other trainers, we need
	// an array to store it. These functions accumulate the gradient so we
	// need to clear it at start and before each new computation. As we now
	// which features are active and so which gradient cell are updated, we
	// can clear them selectively instead of fully clear the gradient each
	// time.
	// We also need an aditional vector named <q> who hold the penalty
	// already applied to each features.
	int *perm = xmalloc(sizeof(int) * S);
	for (int s = 0; s < S; s++)
		perm[s] = s;
	double *g = xmalloc(sizeof(double) * F);
	double *q = xmalloc(sizeof(double) * F);
	for (size_t f = 0; f < F; f++)
		g[f] = q[f] = 0.0;
	// We can now start training the model, we perform the requested number
	// of iteration, each of these going through all the sequences. For
	// computing the decay, we will need to keep track of the number of
	// already processed sequences, this is tracked by the <i> variable.
	double u = 0.0;
	grd_t *grd = grd_new(mdl, g);
	for (int k = 0, i = 0; k < K && !uit_stop; k++) {
		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (int s = 0; s < S; s++) {
			const int a = rand() % S;
			const int b = rand() % S;
			const int t = perm[a];
			perm[a] = perm[b];
			perm[b] = t;
		}
		// And so, we can process sequence in a random order
		for (int sp = 0; sp < S && !uit_stop; sp++, i++) {
			const int s = perm[sp];
			const seq_t *seq = mdl->train->seq[s];
			grd_doseq(grd, seq);
			// Before applying the gradient, we have to compute the
			// learning rate to apply to this sequence. For this we
			// use an exponential decay [1, pp 481(5)]
			//   η_i = η_0 * α^{i/S}
			// And at the same time, we update the total penalty
			// that must have been applied to each features.
			//   u <- u + η * rho1 / S
			const double n0    = mdl->opt->sgdl1.eta0;
			const double alpha = mdl->opt->sgdl1.alpha;
			const double nk = n0 * pow(alpha, (double)i / S);
			u = u + nk * mdl->opt->rho1 / S;
			// Now we apply the update to all unigrams and bigrams
			// observations actives in the current sequence. We must
			// not forget to clear the gradient for the next
			// sequence.
			for (size_t n = 0; idx[s].uobs[n] != none; n++) {
				size_t f = mdl->uoff[idx[s].uobs[n]];
				for (size_t y = 0; y < Y; y++, f++) {
					w[f] -= nk * g[f];
					applypenalty(f);
					g[f] = 0.0;
				}
			}
			for (size_t n = 0; idx[s].bobs[n] != none; n++) {
				size_t f = mdl->boff[idx[s].bobs[n]];
				for (size_t d = 0; d < Y * Y; d++, f++) {
					w[f] -= nk * g[f];
					applypenalty(f);
					g[f] = 0.0;
				}
			}
		}
		if (uit_stop)
			break;
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	grd_free(grd);
	// Cleanup allocated memory before returning
	for (int s = 0; s < S; s++) {
		free(idx[s].uobs);
		free(idx[s].bobs);
	}
	free(perm);
	free(g);
	free(q);
}
#undef applypenalty

/******************************************************************************
 * Blockwise Coordinates descent trainer
 *   The gradient and hessian computation used for the BCD is very similar to
 *   the generic one define below but there is some important differences:
 *     - The forward and backward recursions doesn't have to be performed fully
 *       but just in the range of activity of the considered block. So if the
 *       block is active only at position t, the alpha recusion is done from 1
 *       to t and the beta one from T to t, dividing the amount of computations
 *       by 2.
 *     - Samely the update of the gradient and hessian have to be done only at
 *       position where the block is active, so in the common case where the
 *       block is active only once in the sequence, the improvement can be huge.
 *     - And finally, there is no need to compute the logloss, which can take a
 *       long time due to the computation of the log()s.
 ******************************************************************************/
typedef struct bcd_s bcd_t;
struct bcd_s {
	double *ugrd;    //  [Y]
	double *uhes;    //  [Y]
	double *bgrd;    //  [Y][Y]
	double *bhes;    //  [Y][Y]
	size_t *actpos;  //  [T]
	size_t  actcnt;
	grd_t  *grd;
};

/* bcd_soft:
 *   The softmax function.
 */
static double bcd_soft(double z, double r) {
	if (z >  r) return z - r;
	if (z < -r) return z + r;
	return 0.0;
}

/* bcd_actpos:
 *   List position where the given block is active in the sequence and setup the
 *   limits for the fwd/bwd.
 */
static void bcd_actpos(mdl_t *mdl, bcd_t *bcd, const seq_t *seq, size_t o) {
	const int T = seq->len;
	size_t *actpos = bcd->actpos;
	size_t  actcnt = 0;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		bool ok = false;
		if (mdl->kind[o] & 1)
			for (size_t n = 0; !ok && n < pos->ucnt; n++)
				if (pos->uobs[n] == o)
					ok = true;
		if (mdl->kind[o] & 2)
			for (size_t n = 0; !ok && n < pos->bcnt; n++)
				if (pos->bobs[n] == o)
					ok = true;
		if (!ok)
			continue;
		actpos[actcnt++] = t;
	}
	assert(actcnt != 0);
	bcd->actcnt = actcnt;
	bcd->grd->first = actpos[0];
	bcd->grd->last  = actpos[actcnt - 1];
}

/* bct_flgradhes:
 *   Update the gradient and hessian for <blk> on sequence <seq>. This one is
 *   very similar than the trn_spupgrad function but does the computation only
 *   at active pos and approximate also the hessian.
 */
static void bcd_flgradhes(mdl_t *mdl, bcd_t *bcd, const seq_t *seq, size_t o) {
	const grd_t *grd = bcd->grd;
	const size_t Y = mdl->nlbl;
	const size_t T = seq->len;
	const double (*psi  )[T][Y][Y] = (void *)grd->psi;
	const double (*alpha)[T][Y]    = (void *)grd->alpha;
	const double (*beta )[T][Y]    = (void *)grd->beta;
	const double  *unorm           =         grd->unorm;
	const double  *bnorm           =         grd->bnorm;
	const size_t  *actpos          =         bcd->actpos;
	const size_t   actcnt          =         bcd->actcnt;
	double *ugrd = bcd->ugrd;
	double *uhes = bcd->uhes;
	double *bgrd = bcd->bgrd;
	double *bhes = bcd->bhes;
	// Update the gradient and the hessian but here we sum only on the
	// positions where the block is active for unigrams features
	if (mdl->kind[o] & 1) {
		for (size_t n = 0; n < actcnt; n++) {
			const size_t t = actpos[n];
			for (size_t y = 0; y < Y; y++) {
				const double e = (*alpha)[t][y] * (*beta)[t][y]
				               * unorm[t];
				ugrd[y] += e;
				uhes[y] += e * (1.0 - e);
			}
			const size_t y = seq->pos[t].lbl;
			ugrd[y] -= 1.0;
		}
	}
	if ((mdl->kind[o] & 2) == 0)
		return;
	// for bigrams features
	for (size_t n = 0; n < actcnt; n++) {
		const size_t t = actpos[n];
		if (t == 0)
			continue;
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				double e = (*alpha)[t - 1][yp] * (*beta)[t][y]
				         * (*psi)[t][yp][y] * bnorm[t];
				bgrd[d] += e;
				bhes[d] += e * (1.0 - e);
			}
		}
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		bgrd[yp * Y + y] -= 1.0;
	}
}

/* bct_spgradhes:
 *   Update the gradient and hessian for <blk> on sequence <seq>. This one is
 *   very similar than the trn_spupgrad function but does the computation only
 *   at active pos and approximate also the hessian.
 */
static void bcd_spgradhes(mdl_t *mdl, bcd_t *bcd, const seq_t *seq, size_t o) {
	const grd_t *grd = bcd->grd;
	const size_t Y = mdl->nlbl;
	const size_t T = seq->len;
	const double (*psiuni)[T][Y] = (void *)grd->psiuni;
	const double  *psival        =         grd->psi;
	const size_t  *psiyp         =         grd->psiyp;
	const size_t (*psiidx)[T][Y] = (void *)grd->psiidx;
	const size_t  *psioff        =         grd->psioff;
	const double (*alpha)[T][Y]  = (void *)grd->alpha;
	const double (*beta )[T][Y]  = (void *)grd->beta;
	const double  *unorm         =         grd->unorm;
	const double  *bnorm         =         grd->bnorm;
	const size_t  *actpos        =         bcd->actpos;
	const size_t   actcnt        =         bcd->actcnt;
	double *ugrd = bcd->ugrd;
	double *uhes = bcd->uhes;
	double *bgrd = bcd->bgrd;
	double *bhes = bcd->bhes;
	// Update the gradient and the hessian but here we sum only on the
	// positions where the block is active for unigrams features
	if (mdl->kind[o] & 1) {
		for (size_t n = 0; n < actcnt; n++) {
			const size_t t = actpos[n];
			for (size_t y = 0; y < Y; y++) {
				const double e = (*alpha)[t][y] * (*beta)[t][y]
				               * unorm[t];
				ugrd[y] += e;
				uhes[y] += e * (1.0 - e);
			}
			const size_t y = seq->pos[t].lbl;
			ugrd[y] -= 1.0;
		}
	}
	if ((mdl->kind[o] & 2) == 0)
		return;
	// for bigrams features
	for (size_t n = 0; n < actcnt; n++) {
		const size_t t = actpos[n];
		if (t == 0)
			continue;
		// We build the expectation matrix
		double e[Y][Y];
		for (size_t yp = 0; yp < Y; yp++)
			for (size_t y = 0; y < Y; y++)
				e[yp][y] = (*alpha)[t - 1][yp] * (*beta)[t][y]
				         * (*psiuni)[t][y] * bnorm[t];
		const size_t off = psioff[t];
		for (size_t n = 0, y = 0; n < (*psiidx)[t][Y - 1]; ) {
			while (n >= (*psiidx)[t][y])
				y++;
			while (n < (*psiidx)[t][y]) {
				const size_t yp = psiyp [off + n];
				const double v  = psival[off + n];
				e[yp][y] += e[yp][y] * v;
				n++;
			}
		}
		// And use it
		for (size_t yp = 0, d = 0; yp < Y; yp++) {
			for (size_t y = 0; y < Y; y++, d++) {
				bgrd[d] += e[yp][y];
				bhes[d] += e[yp][y] * (1.0 - e[yp][y]);
			}
		}
		const size_t yp = seq->pos[t - 1].lbl;
		const size_t y  = seq->pos[t    ].lbl;
		bgrd[yp * Y + y] -= 1.0;
	}
}

/* bct_update:
 *   Update the model with the computed gradient and hessian.
 */
static void bcd_update(mdl_t *mdl, bcd_t *bcd, size_t o) {
	const double  rho1  = mdl->opt->rho1;
	const double  rho2  = mdl->opt->rho2;
	const double  kappa = mdl->opt->bcd.kappa;
	const size_t  Y     = mdl->nlbl;
	const double *ugrd  = bcd->ugrd;
	const double *bgrd  = bcd->bgrd;
	      double *uhes  = bcd->uhes;
	      double *bhes  = bcd->bhes;
	if (mdl->kind[o] & 1) {
		// Adjust the hessian
		double a = 1.0;
		for (size_t y = 0; y < Y; y++)
			a = max(a, fabs(ugrd[y] / uhes[y]));
		xvm_scale(uhes, uhes, a * kappa, Y);
		// Update the model
		double *w = mdl->theta + mdl->uoff[o];
		for (size_t y = 0; y < Y; y++) {
			double z = uhes[y] * w[y] - ugrd[y];
			double d = uhes[y] + rho2;
			w[y] = bcd_soft(z, rho1) / d;
		}
	}
	if (mdl->kind[o] & 2) {
		// Adjust the hessian
		double a = 1.0;
		for (size_t i = 0; i < Y * Y; i++)
			a = max(a, fabs(bgrd[i] / bhes[i]));
		xvm_scale(bhes, bhes, a * kappa, Y * Y);
		// Update the model
		double *bw = mdl->theta + mdl->boff[o];
		for (size_t i = 0; i < Y * Y; i++) {
			double z = bhes[i] * bw[i] - bgrd[i];
			double d = bhes[i] + rho2;
			bw[i] = bcd_soft(z, rho1) / d;
		}
	}
}

/* trn_bcd
 *   Train the model using the blockwise coordinates descend method.
 */
static void trn_bcd(mdl_t *mdl) {
	const size_t Y = mdl->nlbl;
	const size_t O = mdl->nobs;
	const size_t T = mdl->train->mlen;
	const size_t S = mdl->train->nseq;
	const int    K = mdl->opt->maxiter;
	// Build the index:
	//   Count active sequences per blocks
	info("    - Build the index\n");
	info("        1/2 -- scan the sequences\n");
	size_t tot = 0, cnt[O], lcl[O];
	for (size_t o = 0; o < O; o++)
		cnt[o] = 0, lcl[o] = none;
	for (size_t s = 0; s < S; s++) {
		// List actives blocks
		const seq_t *seq = mdl->train->seq[s];
		for (int t = 0; t < seq->len; t++) {
			for (size_t b = 0; b < seq->pos[t].ucnt; b++)
				lcl[seq->pos[t].uobs[b]] = s;
			for (size_t b = 0; b < seq->pos[t].bcnt; b++)
				lcl[seq->pos[t].bobs[b]] = s;
		}
		// Updates blocks count
		for (size_t o = 0; o < O; o++)
			cnt[o] += (lcl[o] == s);
	}
	for (size_t o = 0; o < O; o++)
		tot += cnt[o];
	// Allocate memory
	size_t  *idx_cnt = xmalloc(sizeof(size_t  ) * O);
	size_t **idx_lst = xmalloc(sizeof(size_t *) * O);
	for (size_t o = 0; o < O; o++) {
		idx_cnt[o] = cnt[o];
		idx_lst[o] = xmalloc(sizeof(size_t) * cnt[o]);
	}
	// Populate the index
	info("        2/2 -- Populate the index\n");
	for (size_t o = 0; o < O; o++)
		cnt[o] = 0, lcl[o] = none;
	for (size_t s = 0; s < S; s++) {
		// List actives blocks
		const seq_t *seq = mdl->train->seq[s];
		for (int t = 0; t < seq->len; t++) {
			for (size_t b = 0; b < seq->pos[t].ucnt; b++)
				lcl[seq->pos[t].uobs[b]] = s;
			for (size_t b = 0; b < seq->pos[t].bcnt; b++)
				lcl[seq->pos[t].bobs[b]] = s;
		}
		// Build index
		for (size_t o = 0; o < O; o++)
			if (lcl[o] == s)
				idx_lst[o][cnt[o]++] = s;
	}
	info("      Done\n");
	// Allocate the specific trainer of BCD
	bcd_t *bcd = xmalloc(sizeof(bcd_t));
	bcd->ugrd   = xmalloc(sizeof(double) * Y);
	bcd->uhes   = xmalloc(sizeof(double) * Y);
	bcd->bgrd   = xmalloc(sizeof(double) * Y * Y);
	bcd->bhes   = xmalloc(sizeof(double) * Y * Y);
	bcd->actpos = xmalloc(sizeof(size_t) * T);
	bcd->grd    = grd_new(mdl, NULL);
	// And train the model
	for (int i = 0; i < K; i++) {
		for (size_t o = 0; o < O; o++) {
			// Clear the gradient and the hessian
			for (size_t y = 0, d = 0; y < Y; y++) {
				bcd->ugrd[y] = 0.0;
				bcd->uhes[y] = 0.0;
				for (size_t yp = 0; yp < Y; yp++, d++) {
					bcd->bgrd[d] = 0.0;
					bcd->bhes[d] = 0.0;
				}
			}
			// Process active sequences
			for (size_t s = 0; s < idx_cnt[o]; s++) {
				const size_t id = idx_lst[o][s];
				const seq_t *seq = mdl->train->seq[id];
				bcd_actpos(mdl, bcd, seq, o);
				if (mdl->opt->sparse) {
					grd_spdopsi(bcd->grd, seq);
					grd_spfwdbwd(bcd->grd, seq);
					bcd_spgradhes(mdl, bcd, seq, o);
				} else {
					grd_fldopsi(bcd->grd, seq);
					grd_flfwdbwd(bcd->grd, seq);
					bcd_flgradhes(mdl, bcd, seq, o);
				}
			}
			// And update the model
			bcd_update(mdl, bcd, o);
		}
		if (!uit_progress(mdl, i + 1, -1.0))
			break;
	}
	// Cleanup memory
	grd_free(bcd->grd);
	free(bcd->ugrd); free(bcd->uhes);
	free(bcd->bgrd); free(bcd->bhes);
	free(bcd->actpos);
	free(bcd);
	for (size_t o = 0; o < O; o++)
		free(idx_lst[o]);
	free(idx_lst);
	free(idx_cnt);
}

/*******************************************************************************
 * Training
 ******************************************************************************/
static const struct {
	char *name;
	void (* train)(mdl_t *mdl);
} trn_lst[] = {
	{"l-bfgs", trn_lbfgs},
	{"sgd-l1", trn_sgdl1},
	{"bcd",    trn_bcd  }
};
static const int trn_cnt = sizeof(trn_lst) / sizeof(trn_lst[0]);

static void dotrain(mdl_t *mdl) {
	// Check if the user requested the trainer list. If this is not the
	// case, search the trainer.
	if (!strcmp(mdl->opt->algo, "list")) {
		info("Available training algorithms:\n");
		for (int i = 0; i < trn_cnt; i++)
			info("\t%s\n", trn_lst[i].name);
		exit(EXIT_SUCCESS);
	}
	int trn;
	for (trn = 0; trn < trn_cnt; trn++)
		if (!strcmp(mdl->opt->algo, trn_lst[trn].name))
			break;
	if (trn == trn_cnt)
		fatal("unknown algorithm '%s'", mdl->opt->algo);
	// Load a previous model to train again if specified by the user.
	if (mdl->opt->pattern == NULL && mdl->opt->model == NULL)
		fatal("you must specify either a pattern or a model");
	if (mdl->opt->model != NULL) {
		info("* Load previous model\n");
		FILE *file = fopen(mdl->opt->model, "r");
		if (file == NULL)
			pfatal("cannot open input model file");
		mdl_load(mdl, file);
	}
	// Load the pattern file. This is mandatory if no models was loaded as
	// we need some patterns to load data files. This will unlock the
	// database if previously locked by loading a model.
	if (mdl->opt->pattern != NULL) {
		info("* Load patterns\n");
		FILE *file = fopen(mdl->opt->pattern, "r");
		if (file == NULL)
			pfatal("cannot open pattern file");
		rdr_loadpat(mdl->reader, file);
		fclose(file);
		qrk_lock(mdl->reader->obs, false);
	}
	if (mdl->reader->npats == 0)
		fatal("no patterns, cannot load input data");
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
	info("    nb train:    %d\n", mdl->train->nseq);
	if (mdl->devel != NULL)
		info("    nb devel:    %d\n", mdl->devel->nseq);
	info("    nb labels:   %zu\n", mdl->nlbl);
	info("    nb blocks:   %zu\n", mdl->nobs);
	info("    nb features: %zu\n", mdl->nftr);
	// And train the model...
	info("* Train the model with %s\n", mdl->opt->algo);
	uit_setup(mdl);
	trn_lst[trn].train(mdl);
	uit_cleanup(mdl);
	// If requested compact the model.
	if (mdl->opt->compact) {
		const size_t O = mdl->nobs;
		const size_t F = mdl->nftr;
		info("* Compacting the model\n");
		mdl_compact(mdl);
		info("    %8zu observations removed\n", O - mdl->nobs);
		info("    %8zu features removed\n", F - mdl->nftr);
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
	const size_t Y = mdl->nlbl;
	const size_t O = mdl->nobs;
	const qrk_t *Qlbl = mdl->reader->lbl;
	const qrk_t *Qobs = mdl->reader->obs;
	for (size_t o = 0; o < O; o++) {
		const char *obs = qrk_id2str(Qobs, o);
		bool empty = true;
		if (mdl->kind[o] & 1) {
			const double *w = mdl->theta + mdl->uoff[o];
			for (size_t y = 0; y < Y; y++) {
				if (w[y] == 0.0)
					continue;
				const char *ly = qrk_id2str(Qlbl, y);
				fprintf(fout, "%s\t#\t%s\t%f\n", obs, ly, w[y]);
				empty = false;
			}
		}
		if (mdl->kind[o] & 2) {
			const double *w = mdl->theta + mdl->boff[o];
			for (size_t d = 0; d < Y * Y; d++) {
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
	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	// Wapiti stack usage is quite intensive. This simplify a lot the memory
	// mangement and code but, if you have long sequences or huge label set,
	// the stack can overflow. In effect, some OS provide only very small
	// stacks by default.
	// For the L-BFGS trainer, this is not a problem as the computations are
	// done in independant threads and we can adjust their stack easily, but
	// for SGD-L1 and the tagger, this is not the case.
	// I don't known a really portable way to increase the main stack so I
	// will have to move these in workers threads also but this need some
	// thinking.
	// As a quick hack this small code will work on all unix of my knowledge
	// but is not really POSIX compliant and I don't known if it work with
	// cygwin on windows. This is truly a hack as it just raise the soft
	// stack limit to match the hard stack limit without any checking than
	// this will be enough.
	struct rlimit rlp;
	if (getrlimit(RLIMIT_STACK, &rlp) != 0)
		pfatal("cannot get stack size");
	rlp.rlim_cur = rlp.rlim_max;
	if (setrlimit(RLIMIT_STACK, &rlp) != 0)
		pfatal("cannot set stack size");
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	// We first parse command line switchs
	opt_t opt = opt_defaults;
	opt_parse(argc, argv, &opt);
	// Next we prepare the model
	mdl_t *mdl = mdl_new(rdr_new());
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

