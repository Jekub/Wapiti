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
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"

/******************************************************************************
 * Maxent optimized gradient computation
 *
 *   Maxent or maximum entropy models are a specific case of CRF where the
 *   output graph is reduced to a single node. In this specific case, the
 *   computation of the gradient can be simplified a lot as it is done in this
 *   part of the code.
 *
 *   This code will be used to compute gradient for sequences of length one and
 *   without actives bigrams features. All other case are handled by the next
 *   section.
 ******************************************************************************/
void grd_dosingle(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const double *x = mdl->theta;
	const int     T = seq->len;
	const size_t  Y = mdl->nlbl;
	double *psi = grd->psi;
	double *g   = grd->g;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		// We first compute for each Y the sum of weights of all
		// features actives in the sample:
		//     Ψ(y,x^i) = \exp( ∑_k θ_k f_k(y,x^i) )
		//     Z_θ(x^i) = ∑_y Ψ(y,x^i)
		double Z = 0.0;
		for (size_t y = 0; y < Y; y++)
			psi[y] = 0.0;
		for (size_t n = 0; n < pos->ucnt; n++) {
			const double *wgh = x + mdl->uoff[pos->uobs[n]];
			for (size_t y = 0; y < Y; y++)
				psi[y] += wgh[y];
		}
		double lloss = psi[pos->lbl];
		for (size_t y = 0; y < Y; y++) {
			psi[y] = (psi[y] == 0.0) ? 1.0 : exp(psi[y]);
			Z += psi[y];
		}
		// Now, we can compute the gradient update, for each active
		// feature in the sample the update is the expectation over the
		// current model minus the expectation over the observed
		// distribution:
		//     E_{q_θ}(x,y) - E_{p}(x,y)
		// and we can compute the expectation over the model with:
		//     E_{q_θ}(x,y) = f_k(y,x^i) * ψ(y,x) / Z_θ(x)
		for (size_t y = 0; y < Y; y++)
			psi[y] /= Z;
		for (size_t n = 0; n < pos->ucnt; n++) {
			double *grd = g + mdl->uoff[pos->uobs[n]];
			for (size_t y = 0; y < Y; y++)
				grd[y] += psi[y];
			grd[pos->lbl] -= 1.0;
		}
		// And finally the log-likelihood with:
		//     L_θ(x^i,y^i) = log(Z_θ(x^i)) - log(ψ(y^i,x^i))
		grd->lloss += log(Z) - lloss;
	}
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

/* grd_check:
 *   Check that enough memory is allocated in the gradient object so that the
 *   linear-chain codepath can be computed for a sequence of the given length.
 */
void grd_check(grd_t *grd, int len) {
	// Check if user ask for clearing the state tracker or if he requested a
	// bigger tracker. In this case we have to free the previous allocated
	// memory.
	if (len == 0 || (len > grd->len && grd->len != 0)) {
		if (grd->mdl->opt->sparse) {
			xvm_free(grd->psiuni); grd->psiuni = NULL;
			free(grd->psiyp);      grd->psiyp  = NULL;
			free(grd->psiidx);     grd->psiidx = NULL;
			free(grd->psioff);     grd->psioff = NULL;
		}
		xvm_free(grd->psi);   grd->psi   = NULL;
		xvm_free(grd->alpha); grd->alpha = NULL;
		xvm_free(grd->beta);  grd->beta  = NULL;
		xvm_free(grd->unorm); grd->unorm = NULL;
		xvm_free(grd->bnorm); grd->bnorm = NULL;
		xvm_free(grd->scale); grd->scale = NULL;
		grd->len = 0;
	}
	if (len == 0 || len <= grd->len)
		return;
	// If we are here, we have to allocate a new state. This is simple, we
	// just have to take care of the special case for sparse mode.
	const size_t Y = grd->mdl->nlbl;
	const int    T = len;
	grd->psi   = xvm_new(T * Y * Y);
	grd->alpha = xvm_new(T * Y);
	grd->beta  = xvm_new(T * Y);
	grd->scale = xvm_new(T);
	grd->unorm = xvm_new(T);
	grd->bnorm = xvm_new(T);
	if (grd->mdl->opt->sparse) {
		grd->psiuni = xvm_new(T * Y);
		grd->psiyp  = xmalloc(sizeof(size_t) * T * Y * Y);
		grd->psiidx = xmalloc(sizeof(size_t) * T * Y);
		grd->psioff = xmalloc(sizeof(size_t) * T);
	}
	grd->len = len;
}

/* grd_new:
 *   Allocation memory for gradient computation state. This allocate memory for
 *   the longest sequence present in the data set.
 */
grd_t *grd_new(mdl_t *mdl, double *g) {
	grd_t *grd  = xmalloc(sizeof(grd_t));
	grd->mdl    = mdl;
	grd->len    = 0;
	grd->g      = g;
	grd->psi    = NULL;
	grd->psiuni = NULL;
	grd->psiyp  = NULL;
	grd->psiidx = NULL;
	grd->psioff = NULL;
	grd->alpha  = NULL;
	grd->beta   = NULL;
	grd->unorm  = NULL;
	grd->bnorm  = NULL;
	grd->scale  = NULL;
	return grd;
}

/* grd_free:
 *   Free all memory used by gradient computation.
 */
void grd_free(grd_t *grd) {
	grd_check(grd, 0);
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
void grd_fldopsi(grd_t *grd, const seq_t *seq) {
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
void grd_spdopsi(grd_t *grd, const seq_t *seq) {
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
void grd_flfwdbwd(grd_t *grd, const seq_t *seq) {
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
void grd_spfwdbwd(grd_t *grd, const seq_t *seq) {
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
 *   actives observations in the sequence and will be done by the grd_subemp.
 *   The first one is more tricky as it involve computing the probability p_θ.
 *   This is where we use all the previous computations. Again we separate the
 *   computations for unigrams and bigrams here.
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
void grd_flupgrad(grd_t *grd, const seq_t *seq) {
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
		for (size_t y = 0; y < Y; y++) {
			double e = (*alpha)[t][y] * (*beta)[t][y] * unorm[t];
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				g[mdl->uoff[o] + y] += e;
			}
		}
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
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
	}
}

/* grd_spupgrad:
 *   The sparse matrix make things a bit more complicated here as we cannot
 *   directly multiply with the original Ψ_t(y',y,x) because we have split it
 *   two components and the second one is sparse, so we have to make a quite
 *   complex workaround to fix that. We have to explicitly build the expectation
 *   matrix. We first fill it with the unigram component and next multiply it
 *   with the bigram one.
 */
void grd_spupgrad(grd_t *grd, const seq_t *seq) {
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
		for (size_t y = 0; y < Y; y++) {
			double e = (*alpha)[t][y] * (*beta)[t][y] * unorm[t];
			for (size_t n = 0; n < pos->ucnt; n++) {
				const size_t o = pos->uobs[n];
				g[mdl->uoff[o] + y] += e;
			}
		}
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
	}
}

/* grd_subemp:
 *   Substract from the gradient, the expectation over the empirical
 *   distribution. This is the second step of the gradient computation shared
 *   by the non-sparse and sparse version.
 */
void grd_subemp(grd_t *grd, const seq_t *seq) {
	const mdl_t *mdl = grd->mdl;
	const size_t Y = mdl->nlbl;
	const int    T = seq->len;
	double *g = grd->g;
	for (int t = 0; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
		const size_t y = seq->pos[t].lbl;
		for (size_t n = 0; n < pos->ucnt; n++)
			g[mdl->uoff[pos->uobs[n]] + y] -= 1.0;
	}
	for (int t = 1; t < T; t++) {
		const pos_t *pos = &(seq->pos[t]);
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
void grd_logloss(grd_t *grd, const seq_t *seq) {
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
void grd_doseq(grd_t *grd, const seq_t *seq) {
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
	grd_subemp(grd, seq);
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

/* grd_dospl:
 *   Compute the gradient of a single sample choosing between the maxent
 *   optimised codepath and classical one depending of the sample.
 */
void grd_dospl(grd_t *grd, const seq_t *seq) {
	grd_check(grd, seq->len);
	if (seq->len == 1 || grd->mdl->reader->nbi == 0)
		grd_dosingle(grd, seq);
	else
		grd_doseq(grd, seq);
}

/* grd_worker:
 *   This is a simple function who compute the gradient over a subset of the
 *   training set. It is mean to be called by the thread spawner in order to
 *   compute the gradient over the full training set.
 */
static void grd_worker(job_t *job, int id, int cnt, grd_t *grd) {
	unused(id && cnt);
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
	size_t count, pos;
	while (mth_getjob(job, &count, &pos)) {
		for (size_t s = pos; !uit_stop && s < pos + count; s++)
			grd_dospl(grd, dat->seq[s]);
		if (uit_stop)
			break;
	}
}

/* grd_gradient:
 *   Compute the gradient and value of the negative log-likelihood of the model
 *   at current point. The computation is done in parallel taking profit of
 *   the fact that the gradient over the full training set is just the sum of
 *   the gradient of each sequence.
 */
double grd_gradient(mdl_t *mdl, double *g, grd_t *grds[]) {
	const double *x = mdl->theta;
	const size_t  F = mdl->nftr;
	const size_t  W = mdl->opt->nthread;
	// All is ready to compute the gradient, we spawn the threads of
	// workers, each one working on a part of the data. As the gradient and
	// log-likelihood are additive, computing the final values will be
	// trivial.
	mth_spawn((func_t *)grd_worker, W, (void **)grds, mdl->train->nseq,
		mdl->opt->jobsize);
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
	// If needed we clip the gradient: setting to 0.0 all coordinates where
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
	return fx;
}

