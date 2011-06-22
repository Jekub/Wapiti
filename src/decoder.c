/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2011  CNRS
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

#include <float.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "thread.h"
#include "tools.h"
#include "decoder.h"

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

/* tag_expsc:
 *   Compute the score lattice for classical Viterbi decoding. This is the same
 *   as for the first step of the gradient computation with the exception that
 *   we don't need to take the exponential of the scores as the Viterbi decoding
 *   works in log-space.
 */
static int tag_expsc(mdl_t *mdl, const seq_t *seq, double *vpsi) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	double (*psi)[T][Y][Y] = (void *)vpsi;
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
	return 0;
}

/* tag_postsc:
 *   This function compute score lattice with posteriors. This generally result
 *   in a slightly best labelling and allow to output normalized score for the
 *   sequence and for each labels but this is more costly as we have to perform
 *   a full forward backward instead of just the forward pass.
 */
static int tag_postsc(mdl_t *mdl, const seq_t *seq, double *vpsi) {
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	double (*psi)[T][Y][Y] = (void *)vpsi;
	grd_t *grd = grd_new(mdl, NULL);
	grd->first = 0;
	grd->last  = T - 1;
	grd_check(grd, seq->len);
	if (mdl->opt->sparse) {
		grd_spdopsi(grd, seq);
		grd_spfwdbwd(grd, seq);
	} else {
		grd_fldopsi(grd, seq);
		grd_flfwdbwd(grd, seq);
	}
	double (*alpha)[T][Y] = (void *)grd->alpha;
	double (*beta )[T][Y] = (void *)grd->beta;
	double  *unorm        =         grd->unorm;
	for (int t = 0; t < T; t++) {
		for (size_t y = 0; y < Y; y++) {
			double e = (*alpha)[t][y] * (*beta)[t][y] * unorm[t];
			for (size_t yp = 0; yp < Y; yp++)
				(*psi)[t][yp][y] = e;
		}
	}
	grd_free(grd);
	return 1;
}

/* tag_viterbi:
 *   This function implement the Viterbi algorithm in order to decode the most
 *   probable sequence of labels according to the model. Some part of this code
 *   is very similar to the computation of the gradient as expected.
 *
 *   And like for the gradient, the caller is responsible to ensure there is
 *   enough stack space.
 */
void tag_viterbi(mdl_t *mdl, const seq_t *seq,
	         size_t out[], double *sc, double psc[]) {
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	double  *vpsi  = xmalloc(sizeof(double) * T * Y * Y);
	size_t  *vback = xmalloc(sizeof(size_t) * T * Y);
	double (*psi) [T][Y][Y] = (void *)vpsi;
	size_t (*back)[T][Y]    = (void *)vback;
	double  *cur = xmalloc(sizeof(double) * Y);
	double  *old = xmalloc(sizeof(double) * Y);
	// We first compute the scores for each transitions in the lattice of
	// labels.
	int op;
	if (mdl->opt->lblpost)
		op = tag_postsc(mdl, seq, vpsi);
	else
		op = tag_expsc(mdl, seq, vpsi);
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
		cur[y] = (*psi)[0][0][y];
	for (int t = 1; t < T; t++) {
		for (size_t y = 0; y < Y; y++)
			old[y] = cur[y];
		for (size_t y = 0; y < Y; y++) {
			double bst = -1.0;
			int    idx = 0;
			for (size_t yp = 0; yp < Y; yp++) {
				double val = old[yp];
				if (op)
					val *= (*psi)[t][yp][y];
				else
					val += (*psi)[t][yp][y];
				if (val > bst) {
					bst = val;
					idx = yp;
				}
			}
			(*back)[t][y] = idx;
			cur[y]        = bst;
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
	if (sc != NULL)
		*sc = cur[bst];
	for (int t = T; t > 0; t--) {
		const size_t yp = (t != 1) ? (*back)[t - 1][bst] : 0;
		const size_t y  = bst;
		out[t - 1] = y;
		if (psc != NULL)
			psc[t - 1] = (*psi)[t - 1][yp][y];
		bst = yp;
	}
	free(old);
	free(cur);
	free(vback);
	free(vpsi);
}

/* tag_nbviterbi:
 *   This function implement the Viterbi algorithm in order to decode the N-most
 *   probable sequences of labels according to the model. It can be used to
 *   compute only the best one and will return the same sequence than the
 *   previous function but will be slower to do it.
 */
void tag_nbviterbi(mdl_t *mdl, const seq_t *seq, size_t N,
	           size_t out[][N], double sc[], double psc[][N]) {
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	double  *vpsi  = xmalloc(sizeof(double) * T * Y * Y);
	size_t  *vback = xmalloc(sizeof(size_t) * T * Y * N);
	double (*psi) [T][Y    ][Y] = (void *)vpsi;
	size_t (*back)[T][Y * N]    = (void *)vback;
	double  *cur = xmalloc(sizeof(double) * Y * N);
	double  *old = xmalloc(sizeof(double) * Y * N);
	// We first compute the scores for each transitions in the lattice of
	// labels.
	int op;
	if (mdl->opt->lblpost)
		op = tag_postsc(mdl, seq, (double *)psi);
	else
		op = tag_expsc(mdl, seq, (double *)psi);
	// Here also, it's classical but we have to keep the N best paths
	// leading to each nodes of the lattice instead of only the best one.
	// This mean that code is less trivial and the current implementation is
	// not the most efficient way to do this but it works well and is good
	// enough for the moment.
	// We first build the list of all incoming arcs from all paths from all
	// N-best nodes and next select the N-best one. There is a lot of room
	// here for later optimisations if needed.
	for (size_t y = 0, d = 0; y < Y; y++) {
		cur[d++] = (*psi)[0][0][y];
		for (size_t n = 1; n < N; n++)
			cur[d++] = -DBL_MAX;
	}
	for (int t = 1; t < T; t++) {
		for (size_t d = 0; d < Y * N; d++)
			old[d] = cur[d];
		for (size_t y = 0; y < Y; y++) {
			// 1st, build the list of all incoming
			double lst[Y * N];
			for (size_t yp = 0, d = 0; yp < Y; yp++) {
				for (size_t n = 0; n < N; n++, d++) {
					lst[d] = old[d];
					if (op)
						lst[d] *= (*psi)[t][yp][y];
					else
						lst[d] += (*psi)[t][yp][y];
				}
			}
			// 2nd, init the back with the N first
			size_t *bk = &(*back)[t][y * N];
			for (size_t n = 0; n < N; n++)
				bk[n] = n;
			// 3rd, search the N highest values
			for (size_t i = N; i < N * Y; i++) {
				// Search the smallest current value
				size_t idx = 0;
				for (size_t n = 1; n < N; n++)
					if (lst[bk[n]] < lst[bk[idx]])
						idx = n;
				// And replace it if needed
				if (lst[i] > lst[bk[idx]])
					bk[idx] = i;
			}
			// 4th, get the new scores
			for (size_t n = 0; n < N; n++)
				cur[y * N + n] = lst[bk[n]];
		}
	}
	// Retrieving the best paths is similar to classical Viterbi except that
	// we have to search for the N bet ones and there is N time more
	// possibles starts.
	for (size_t n = 0; n < N; n++) {
		int bst = 0;
		for (size_t d = 1; d < Y * N; d++)
			if (cur[d] > cur[bst])
				bst = d;
		if (sc != NULL)
			sc[n] = cur[bst];
		cur[bst] = -DBL_MAX;
		for (int t = T; t > 0; t--) {
			const size_t yp = (t != 1) ? (*back)[t - 1][bst] / N: 0;
			const size_t y  = bst / N;
			out[t - 1][n] = y;
			if (psc != NULL)
				psc[t - 1][n] = (*psi)[t - 1][yp][y];
			bst = (*back)[t - 1][bst];
		}
	}
	free(old);
	free(cur);
	free(vback);
	free(vpsi);
}

/* tag_label:
 *   Label a data file using the current model. This output an almost exact copy
 *   of the input file with an additional column with the predicted label. If
 *   the check option is specified, the input file must be labelled and the
 *   predicted labels will be checked against the provided ones. This will
 *   output error rates during the labelling and detailed statistics per label
 *   at the end.
 */
void tag_label(mdl_t *mdl, FILE *fin, FILE *fout) {
	qrk_t *lbls = mdl->reader->lbl;
	const size_t Y = mdl->nlbl;
	const size_t N = mdl->opt->nbest;
	// We start by preparing the statistic collection to be ready if check
	// option is used. The stat array hold the following for each label
	//   [0] # of reference with this label
	//   [1] # of token we have taged with this label
	//   [2] # of match of the two preceding
	size_t tcnt = 0, terr = 0;
	size_t scnt = 0, serr = 0;
	size_t stat[3][Y];
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
		const int T = seq->len;
		size_t *out = xmalloc(sizeof(size_t) * T * N);
		double *psc = xmalloc(sizeof(double) * T * N);
		double *scs = xmalloc(sizeof(double) * N);
		if (N == 1)
			tag_viterbi(mdl, seq, (size_t*)out, scs, (double*)psc);
		else
			tag_nbviterbi(mdl, seq, N, (void*)out, scs, (void*)psc);
		// Next we output the raw sequence with an aditional column for
		// the predicted labels
		for (size_t n = 0; n < N; n++) {
			if (mdl->opt->outsc)
				fprintf(fout, "# %d %f\n", (int)n, scs[n]);
			for (int t = 0; t < T; t++) {
				if (!mdl->opt->label)
					fprintf(fout, "%s\t", raw->lines[t]);
				size_t lbl = out[t * N + n];
				const char *lblstr = qrk_id2str(lbls, lbl);
				fprintf(fout, "%s", lblstr);
				if (mdl->opt->outsc) {
					fprintf(fout, "\t%s", lblstr);
					fprintf(fout, "/%f", psc[t * N + n]);
				}
				fprintf(fout, "\n");
			}
			fprintf(fout, "\n");
		}
		fflush(fout);
		// If user provided reference labels, use them to collect
		// statistics about how well we have performed here.
		if (mdl->opt->check) {
			bool err = false;
			for (int t = 0; t < T; t++) {
				stat[0][seq->pos[t].lbl]++;
				stat[1][out[t * N]]++;
				if (seq->pos[t].lbl != out[t * N])
					terr++, err = true;
				else
					stat[2][out[t * N]]++;
			}
			tcnt += (size_t)T;
			serr += err;
		}
		// Cleanup memory used for this sequence
		free(scs);
		free(psc);
		free(out);
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
			const char   *lbl = qrk_id2str(lbls, y);
			const double  Rc  = (double)stat[2][y] / stat[0][y];
			const double  Pr  = (double)stat[2][y] / stat[1][y];
			const double  F1  = 2.0 * (Pr * Rc) / (Pr + Rc);
			info("    %-6s", lbl);
			info("  Pr=%.2f", Pr);
			info("  Rc=%.2f", Rc);
			info("  F1=%.2f\n", F1);
		}
	}
}

/* eval_t:
 *   This a state tracker used to communicate between the main eval function and
 *   its workers threads, the <mdl> and <dat> fields are used to transmit to the
 *   workers informations needed to make the computation, the other fields are
 *   for returning the partial results.
 */
typedef struct eval_s eval_t;
struct eval_s {
	mdl_t  *mdl;
	dat_t  *dat;
	size_t  tcnt;  // Processed tokens count
	size_t  terr;  // Tokens error found
	size_t  scnt;  // Processes sequences count
	size_t  serr;  // Sequence error found
};

/* tag_evalsub:
 *   This is where the real evaluation is done by the workers, we process data
 *   by batch and for each batch do a simple Viterbi and scan the result to find
 *   errors.
 */
static void tag_evalsub(job_t *job, int id, int cnt, eval_t *eval) {
	unused(id && cnt);
	mdl_t *mdl = eval->mdl;
	dat_t *dat = eval->dat;
	eval->tcnt = 0;
	eval->terr = 0;
	eval->scnt = 0;
	eval->serr = 0;
	// We just get a job a process all the squence in it.
	size_t count, pos;
	while (mth_getjob(job, &count, &pos)) {
		for (size_t s = pos; s < pos + count; s++) {
			// Tag the sequence with the viterbi
			const seq_t *seq = dat->seq[s];
			const int    T   = seq->len;
			size_t out[T];
			tag_viterbi(mdl, seq, out, NULL, NULL);
			// And check for eventual (probable ?) errors
			bool err = false;
			for (int t = 0; t < T; t++)
				if (seq->pos[t].lbl != out[t])
					eval->terr++, err = true;
			eval->tcnt += (size_t)T;
			eval->scnt += 1;
			eval->serr += err;
		}
	}
}

/* tag_eval:
 *   Compute the token error rate and sequence error rate over the devel set (or
 *   taining set if not available).
 */
void tag_eval(mdl_t *mdl, double *te, double *se) {
	const size_t W = mdl->opt->nthread;
	dat_t *dat = (mdl->devel == NULL) ? mdl->train : mdl->devel;
	// First we prepare the eval state for all the workers threads, we just
	// have to give them the model and dataset to use. This state will be
	// used to retrieve partial result they computed.
	eval_t *eval[W];
	for (size_t w = 0; w < W; w++) {
		eval[w] = xmalloc(sizeof(eval_t));
		eval[w]->mdl = mdl;
		eval[w]->dat = dat;
	}
	// And next, we call the workers to do the job and reduce the partial
	// result by summing them and computing the final error rates.
	mth_spawn((func_t *)tag_evalsub, W, (void *)eval, dat->nseq,
		mdl->opt->jobsize);
	size_t tcnt = 0, terr = 0;
	size_t scnt = 0, serr = 0;
	for (size_t w = 0; w < W; w++) {
		tcnt += eval[w]->tcnt;
		terr += eval[w]->terr;
		scnt += eval[w]->scnt;
		serr += eval[w]->serr;
		free(eval[w]);
	}
	*te = (double)terr / tcnt * 100.0;
	*se = (double)serr / scnt * 100.0;
}

