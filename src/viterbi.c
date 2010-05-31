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

#include <float.h>
#include <stddef.h>
#include <stdio.h>

#include "model.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"
#include "viterbi.h"

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
void tag_viterbi(const mdl_t *mdl, const seq_t *seq, size_t out[], double *sc) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	// Like for the gradient, we rely on stack storage and let the caller
	// ensure there is enough free space there. This function will need
	//   8 * ((T * Y * (1 + Y)) + 2 * Y)
	// bytes of stack plus a bit more for variables.
	double psi [T][Y][Y];
	size_t back[T][Y];
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
	if (sc != NULL)
		*sc = cur[bst];
	for (int t = T; t > 0; t--) {
		out[t - 1] = bst;
		bst = back[t - 1][bst];
	}
}

/* tag_nbviterbi:
 *   This function implement the Viterbi algorithm in order to decode the N-most
 *   probable sequences of labels according to the model. It can be used to
 *   compute only the best one and will return the same sequence than the
 *   previous function but will be slower to do it.
 */
void tag_nbviterbi(const mdl_t *mdl, const seq_t *seq, size_t out[],
                   double scs[], size_t N) {
	const double *x = mdl->theta;
	const size_t  Y = mdl->nlbl;
	const int     T = seq->len;
	// Like for the gradient, we rely on stack storage and let the caller
	// ensure there is enough free space there. This function will need
	//   8 * (Y * N * (T + 2) + Y * Y * T)
	// bytes of stack plus a bit more for variables.
	double psi [T][Y    ][Y];
	size_t back[T][Y * N];
	double cur    [Y * N];
	double old    [Y * N];
	// We first have to compute the Ψ_t(y',y,x_t) weights defined as
	//   Ψ_t(y',y,x_t) = \exp( ∑_k θ_k f_k(y',y,x_t) )
	// This is exactly the same as standard Viterbi so see comment above.
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
	// Here also, it's classical but we have to keep the N best paths
	// leading to each nodes of the lattice instead of only the best one.
	// This mean that code is less trivial and the current implementation is
	// not the most efficient way to do this but it works well and is good
	// enough for the moment.
	// We first build the list of all incoming arcs from all paths from all
	// N-best nodes and next select the N-best one. There is a lot of room
	// here for later optimisations if needed.
	for (size_t y = 0, d = 0; y < Y; y++) {
		cur[d++] = psi[0][0][y];
		for (size_t n = 1; n < N; n++)
			cur[d++] = -DBL_MAX;
	}
	for (int t = 1; t < T; t++) {
		for (size_t d = 0; d < Y * N; d++)
			old[d] = cur[d];
		for (size_t y = 0; y < Y; y++) {
			// 1st, build the list of all incoming
			double lst[Y * N];
			for (size_t yp = 0, d = 0; yp < Y; yp++)
				for (size_t n = 0; n < N; n++, d++)
					lst[d] = psi[t][yp][y] + old[d];
			// 2nd, init the back with the N first
			size_t *bk = &back[t][y * N];
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
		size_t *o = &out[n * T];
		int bst = 0;
		for (size_t d = 1; d < Y * N; d++)
			if (cur[d] > cur[bst])
				bst = d;
		if (scs != NULL)
			scs[n] = cur[bst];
		cur[bst] = -DBL_MAX;
		for (int t = T; t > 0; t--) {
			o[t - 1] = bst / N;
			bst = back[t - 1][bst];
		}
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
void tag_label(const mdl_t *mdl, FILE *fin, FILE *fout) {
	qrk_t *lbls = mdl->reader->lbl;
	const size_t Y = mdl->nlbl;
	const size_t N = mdl->opt->nbest;
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
		size_t out[seq->len * N];
		double scs[N];
		if (N == 1)
			tag_viterbi(mdl, seq, out, scs);
		else
			tag_nbviterbi(mdl, seq, out, scs, N);
		// If requested, output the scores
		if (mdl->opt->outsc) {
			fprintf(fout, "#");
			for (size_t n = 0; n < N; n++)
				fprintf(fout, " %f", scs[n]);
			fprintf(fout, "\n");
		}
		// Next we output the raw sequence with an aditional column for
		// the predicted labels
		for (int t = 0; t < seq->len; t++) {
			if (!mdl->opt->label)
				fprintf(fout, "%s\t", raw->lines[t]);
			for (size_t n = 0; n < N; n++) {
				size_t lbl = out[n * seq->len + t];
				fprintf(fout, "%s", qrk_id2str(lbls, lbl));
				fprintf(fout, "%c", n == N - 1 ? '\n' : '\t');
			}
		}
		fprintf(fout, "\n");
		// If user provided reference labels, use them to collect
		// statistics about how well we have performed here.
		if (mdl->opt->check) {
			bool err = false;
			for (int t = 0; t < seq->len; t++) {
				stat[0][seq->pos[t].lbl]++;
				stat[1][out[t * N]]++;
				if (seq->pos[t].lbl != out[t])
					terr++, err = true;
				else
					stat[2][out[t * N]]++;
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

