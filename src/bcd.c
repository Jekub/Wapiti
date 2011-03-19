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

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "vmath.h"

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
void trn_bcd(mdl_t *mdl) {
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
	bcd->ugrd   = xvm_new(Y);
	bcd->uhes   = xvm_new(Y);
	bcd->bgrd   = xvm_new(Y * Y);
	bcd->bhes   = xvm_new(Y * Y);
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
				grd_check(bcd->grd, seq->len);
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
	xvm_free(bcd->ugrd); xvm_free(bcd->uhes);
	xvm_free(bcd->bgrd); xvm_free(bcd->bhes);
	free(bcd->actpos);
	free(bcd);
	for (size_t o = 0; o < O; o++)
		free(idx_lst[o]);
	free(idx_lst);
	free(idx_cnt);
}

