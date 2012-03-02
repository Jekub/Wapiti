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

#include "wapiti.h"
#include "model.h"
#include "options.h"
#include "quark.h"
#include "reader.h"
#include "tools.h"
#include "vmath.h"

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

/* mdl_new:
 *   Allocate a new empty model object linked with the given reader. The model
 *   have to be synchronized before starting training or labelling. If you not
 *   provide a reader (as it will loaded from file for example) you must be sure
 *   to set one in the model before any attempts to synchronize it.
 */
mdl_t *mdl_new(rdr_t *rdr) {
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
void mdl_free(mdl_t *mdl) {
	free(mdl->kind);
	free(mdl->uoff);
	free(mdl->boff);
	if (mdl->theta != NULL)
		xvm_free(mdl->theta);
	if (mdl->train != NULL)
		rdr_freedat(mdl->train);
	if (mdl->devel != NULL)
		rdr_freedat(mdl->devel);
	if (mdl->reader != NULL)
		rdr_free(mdl->reader);
	if (mdl->werr != NULL)
		free(mdl->werr);
	free(mdl);
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
void mdl_sync(mdl_t *mdl) {
	const uint32_t Y = qrk_count(mdl->reader->lbl);
	const uint64_t O = qrk_count(mdl->reader->obs);
	// If model is already synchronized, do nothing and just return
	if (mdl->nlbl == Y && mdl->nobs == O)
		return;
	if (Y == 0 || O == 0)
		fatal("cannot synchronize an empty model");
	// If new labels was added, we have to discard all the model. In this
	// case we also display a warning as this is probably not expected by
	// the user. If only new observations was added, we will try to expand
	// the model.
	uint64_t oldF = mdl->nftr;
	uint64_t oldO = mdl->nobs;
	if (mdl->nlbl != Y && mdl->nlbl != 0) {
		warning("labels count changed, discarding the model");
		free(mdl->kind);  mdl->kind  = NULL;
		free(mdl->uoff);  mdl->uoff  = NULL;
		free(mdl->boff);  mdl->boff  = NULL;
		if (mdl->theta != NULL) {
			xvm_free(mdl->theta);
			mdl->theta = NULL;
		}
		oldF = oldO = 0;
	}
	mdl->nlbl = Y;
	mdl->nobs = O;
	// Allocate the observations datastructure. If the model is empty or
	// discarded, a new one iscreated, else the old one is expanded.
	char     *kind = xrealloc(mdl->kind, sizeof(char    ) * O);
	uint64_t *uoff = xrealloc(mdl->uoff, sizeof(uint64_t) * O);
	uint64_t *boff = xrealloc(mdl->boff, sizeof(uint64_t) * O);
	mdl->kind = kind;
	mdl->uoff = uoff;
	mdl->boff = boff;
	// Now, we can setup the features. For each new observations we fill the
	// kind and offsets arrays and count total number of features as well.
	uint64_t F = oldF;
	for (uint64_t o = oldO; o < O; o++) {
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
	// This is a bit tricky as aligned malloc cannot be simply grown so we
	// have to allocate a new vector and copy old values ourself.
	if (oldF != 0) {
		double *new = xvm_new(F);
		for (uint64_t f = 0; f < oldF; f++)
			new[f] = mdl->theta[f];
		xvm_free(mdl->theta);
		mdl->theta = new;
	} else {
		mdl->theta = xvm_new(F);
	}
	for (uint64_t f = oldF; f < F; f++)
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
void mdl_compact(mdl_t *mdl) {
	const uint32_t Y = mdl->nlbl;
	// We first build the new observation list with only observations which
	// lead to at least one active feature. At the same time we build the
	// translation table which map the new observations index to the old
	// ones.
	info("    - Scan the model\n");
	qrk_t *old_obs = mdl->reader->obs;
	qrk_t *new_obs = qrk_new();
	uint64_t *trans = xmalloc(sizeof(uint64_t) * mdl->nobs);
	for (uint64_t oldo = 0; oldo < mdl->nobs; oldo++) {
		bool active = false;
		if (mdl->kind[oldo] & 1)
			for (uint32_t y = 0; y < Y; y++)
				if (mdl->theta[mdl->uoff[oldo] + y] != 0.0)
					active = true;
		if (mdl->kind[oldo] & 2)
			for (uint32_t d = 0; d < Y * Y; d++)
				if (mdl->theta[mdl->boff[oldo] + d] != 0.0)
					active = true;
		if (!active)
			continue;
		const char     *str  = qrk_id2str(old_obs, oldo);
		const uint64_t  newo = qrk_str2id(new_obs, str);
		trans[newo] = oldo;
	}
	mdl->reader->obs = new_obs;
	// Now we save the old model features informations and build a new one
	// corresponding to the compacted model.
	uint64_t *old_uoff  = mdl->uoff;  mdl->uoff  = NULL;
	uint64_t *old_boff  = mdl->boff;  mdl->boff  = NULL;
	double   *old_theta = mdl->theta; mdl->theta = NULL;
	free(mdl->kind);
	mdl->kind = NULL;
	mdl->nlbl = mdl->nobs = mdl->nftr = 0;
	mdl_sync(mdl);
	// The model is now ready, so we copy in it the features weights from
	// the old model for observations we have kept.
	info("    - Compact it\n");
	for (uint64_t newo = 0; newo < mdl->nobs; newo++) {
		const uint64_t oldo = trans[newo];
		if (mdl->kind[newo] & 1) {
			double *src = old_theta  + old_uoff[oldo];
			double *dst = mdl->theta + mdl->uoff[newo];
			for (uint32_t y = 0; y < Y; y++)
				dst[y] = src[y];
		}
		if (mdl->kind[newo] & 2) {
			double *src = old_theta  + old_boff[oldo];
			double *dst = mdl->theta + mdl->boff[newo];
			for (uint32_t d = 0; d < Y * Y; d++)
				dst[d] = src[d];
		}
	}
	// And cleanup
	free(trans);
	qrk_free(old_obs);
	free(old_uoff);
	free(old_boff);
	xvm_free(old_theta);
}

/* mdl_save:
 *   Save a model to be restored later in a platform independant way.
 */
void mdl_save(mdl_t *mdl, FILE *file) {
	uint64_t nact = 0;
	for (uint64_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			nact++;
	fprintf(file, "#mdl#%d#%"PRIu64"\n", mdl->type, nact);
	rdr_save(mdl->reader, file);
	for (uint64_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			fprintf(file, "%"PRIu64"=%la\n", f, mdl->theta[f]);
}

/* mdl_load:
 *   Read back a previously saved model to continue training or start labeling.
 *   The returned model is synced and the quarks are locked. You must give to
 *   this function an empty model fresh from mdl_new.
 */
void mdl_load(mdl_t *mdl, FILE *file) {
	const char *err = "invalid model format";
	uint64_t nact = 0;
	int type;
	if (fscanf(file, "#mdl#%d#%"SCNu64"\n", &type, &nact) == 2)
		mdl->type = type;
	else if (fscanf(file, "#mdl#%"SCNu64"\n", &nact) == 1)
		mdl->type = 0;
	else
		fatal(err);
	rdr_load(mdl->reader, file);
	mdl_sync(mdl);
	for (uint64_t i = 0; i < nact; i++) {
		uint64_t f;
		double v;
		if (fscanf(file, "%"SCNu64"=%la\n", &f, &v) != 2)
			fatal(err);
		mdl->theta[f] = v;
	}
}

