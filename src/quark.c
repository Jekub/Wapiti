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
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "quark.h"
#include "tools.h"

/******************************************************************************
 * Map object
 *
 *   Implement quark object for mapping strings to identifiers through crit-bit
 *   tree (also known as PATRICIA tries). In fact it only store a compressed
 *   version of the trie to reduce memory footprint. The special trick of using
 *   the last bit of the reference to differenciate between nodes and leafs come
 *   from Daniel J. Bernstein implementation of crit-bit tree that can be found
 *   on his web site.
 *   [1] Morrison, Donald R. ; PATRICIA-Practical Algorithm To Retrieve
 *   Information Coded in Alphanumeric, Journal of the ACM 15 (4): pp. 514--534,
 *   1968. DOI:10.1145/321479.321481
 *
 *   This code is copyright 2002-2012 Thomas Lavergne and licenced under the BSD
 *   Licence like the remaining of Wapiti.
 ******************************************************************************/

typedef struct node_s node_t;
typedef struct leaf_s leaf_t;
struct qrk_s {
	struct node_s {
		node_t   *child[2];
		uint32_t  pos;
		uint8_t   byte;
	} *root;
	struct leaf_s {
		uint64_t  id;
		char      key[];
	} **leafs;
	bool     lock;
	uint64_t count;
	uint64_t size;
};

#define qrk_lf2nd(lf)  ((node_t *)((intptr_t)(lf) |  1))
#define qrk_nd2lf(nd)  ((leaf_t *)((intptr_t)(nd) & ~1))
#define qrk_isleaf(nd) ((intptr_t)(nd) & 1)

/* qrk_new:
 *   This initialize the object for holding a new empty trie, with some pre-
 *   allocations. The returned object must be freed with a call to qrk_free when
 *   not needed anymore.
 */
qrk_t *qrk_new(void) {
	const uint64_t size = 128;
	qrk_t *qrk = xmalloc(sizeof(qrk_t));
	qrk->root  = NULL;
	qrk->count = 0;
	qrk->lock  = false;
	qrk->size  = size;
	qrk->leafs = xmalloc(sizeof(leaf_t) * size);
	return qrk;
}

/* qrk_free:
 *   Release all the memory used by a qrk_t object allocated with qrk_new. This
 *   will release all key string stored internally so all key returned by
 *   qrk_unmap become invalid and must not be used anymore.
 */
void qrk_free(qrk_t *qrk) {
	const uint32_t stkmax = 1024;
	if (qrk->count != 0) {
		node_t *stk[stkmax];
		uint32_t cnt = 0;
		stk[cnt++] = qrk->root;
		while (cnt != 0) {
			node_t *nd = stk[--cnt];
			if (qrk_isleaf(nd)) {
				free(qrk_nd2lf(nd));
				continue;
			}
			stk[cnt++] = nd->child[0];
			stk[cnt++] = nd->child[1];
			free(nd);
		}
	}
	free(qrk->leafs);
	free(qrk);
}

/* qrk_insert:
 *   Map a key to a uniq identifier. If the key already exist in the map, return
 *   its identifier, else allocate a new identifier and insert the new (key,id)
 *   pair inside the quark. This function is not thread safe and should not be
 *   called on the same map from different thread without locking.
 */
uint64_t qrk_str2id(qrk_t *qrk, const char *key) {
	const uint8_t *raw = (void *)key;
	const size_t   len = strlen(key);
	// We first take care of the empty trie case so later we can safely
	// assume that the trie is well formed and so there is no NULL pointers
	// in it.
	if (qrk->count == 0) {
		if (qrk->lock == true)
			return none;
		const size_t size = sizeof(char) * (len + 1);
		leaf_t *lf = xmalloc(sizeof(leaf_t) + size);
		memcpy(lf->key, key, size);
		lf->id = 0;
		qrk->root = qrk_lf2nd(lf);
		qrk->leafs[0] = lf;
		qrk->count = 1;
		return 0;
	}
	// If the trie is not empty, we first go down the trie to the leaf like
	// if we are searching for the key. When at leaf there is two case,
	// either we have found our key or we have found another key with all
	// its critical bit identical to our one. So we search for the first
	// differing bit between them to know where we have to add the new node.
	const node_t *nd = qrk->root;
	while (!qrk_isleaf(nd)) {
		const uint8_t chr = nd->pos < len ? raw[nd->pos] : 0;
		const int side = ((chr | nd->byte) + 1) >> 8;
		nd = nd->child[side];
	}
	const char *bst = qrk_nd2lf(nd)->key;
	size_t pos;
	for (pos = 0; pos < len; pos++)
		if (key[pos] != bst[pos])
			break;
	uint8_t byte;
	if (pos != len)
		byte = key[pos] ^ bst[pos];
	else if (bst[pos] != '\0')
		byte = bst[pos];
	else
		return qrk_nd2lf(nd)->id;
	if (qrk->lock == true)
		return none;
	// Now we known the two key are different and we know in which byte. It
	// remain to build the mask for the new critical bit and build the new
	// internal node and leaf.
	while (byte & (byte - 1))
		byte &= byte - 1;
	byte ^= 255;
	const uint8_t chr = bst[pos];
	const int side = ((chr | byte) + 1) >> 8;
	const size_t size = sizeof(char) * (len + 1);
	node_t *nx = xmalloc(sizeof(node_t));
	leaf_t *lf = xmalloc(sizeof(leaf_t) + size);
	memcpy(lf->key, key, size);
	lf->id   = qrk->count++;
	nx->pos  = pos;
	nx->byte = byte;
	nx->child[1 - side] = qrk_lf2nd(lf);
	if (lf->id == qrk->size) {
		qrk->size *= 1.4;
		const size_t size = sizeof(leaf_t *) * qrk->size;
		qrk->leafs = xrealloc(qrk->leafs, size);
	}
	qrk->leafs[lf->id] = lf;
	// And last thing to do: inserting the new node in the trie. We have to
	// walk down the trie again as we have to keep the ordering of nodes. So
	// we search for the good position to insert it.
	node_t **trg = &qrk->root;
	while (true) {
		node_t *nd = *trg;
		if (qrk_isleaf(nd) || nd->pos > pos)
			break;
		if (nd->pos == pos && nd->byte > byte)
			break;
		const uint8_t chr = nd->pos < len ? raw[nd->pos] : 0;
		const int side = ((chr | nd->byte) + 1) >> 8;
		trg = &nd->child[side];
	}
	nx->child[side] = *trg;
	*trg = nx;
	return lf->id;
}

/* qrk_id2str:
 *    Retrieve the key associated to an identifier. The key is returned as a
 *    constant string that should not be modified or freed by the caller, it is
 *    a pointer to the internal copy of the key kept by the map object and
 *    remain valid only for the life time of the quark, a call to qrk_free will
 *    make this pointer invalid.
 */
const char *qrk_id2str(const qrk_t *qrk, uint64_t id) {
	if (id >= qrk->count)
		fatal("invalid identifier");
	return qrk->leafs[id]->key;
}

/* qrk_save:
 *   Save list of keys present in the map object in the id order to the given
 *   file. We put one key per line so, if no key contains a new line, the line
 *   number correspond to the id.
 */
void qrk_save(const qrk_t *qrk, FILE *file) {
	if (fprintf(file, "#qrk#%"PRIu64"\n", qrk->count) < 0)
		pfatal("cannot write to file");
	if (qrk->count == 0)
		return;
	for (uint64_t n = 0; n < qrk->count; n++)
		ns_writestr(file, qrk->leafs[n]->key);
}

/* qrk_load:
 *   Load a list of key from the given file and add them to the map. Each lines
 *   of the file is taken as a single key and mapped to the next available id if
 *   not already present. If all keys are single lines and the given map is
 *   initilay empty, this will load a map exactly as saved by qrk_save.
 */
void qrk_load(qrk_t *qrk, FILE *file) {
	uint64_t cnt = 0;
	if (fscanf(file, "#qrk#%"SCNu64"\n", &cnt) != 1) {
		if (ferror(file) != 0)
			pfatal("cannot read from file");
		pfatal("invalid format");
	}
	for (uint64_t n = 0; n < cnt; ++n) {
		char *str = ns_readstr(file);
		qrk_str2id(qrk, str);
		free(str);
	}
}

/* qrk_count:
 *   Return the number of mappings stored in the quark.
 */
uint64_t qrk_count(const qrk_t *qrk) {
	return qrk->count;
}

/* qrk_lock:
 *   Set the lock value of the quark and return the old one.
 */
bool qrk_lock(qrk_t *qrk, bool lock) {
	bool old = qrk->lock;
	qrk->lock = lock;
	return old;
}

