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
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "quark.h"
#include "tools.h"

/******************************************************************************
 * Quark database
 *
 *   Implement quark database: mapping between strings and identifiers in both
 *   directions.
 *
 *   The mapping between strings to identifiers is done with splay tree, a kind
 *   of self balancing search tree. They are simpler to implement than a lot of
 *   other data-structures and very cache friendly. For most use they are at
 *   least as efficient as red-black tree or B-tree. See [1] for more
 *   informations. The strings are interned directly in the node for easier
 *   memory managment.
 *
 *   The identifier to string mapping is done with a simple vector with pointer
 *   to the key interned in the tree node, so pointer returned are always
 *   constant.
 *
 *   [1] Sleator, Daniel D. and Tarjan, Robert E. ; Self-adjusting binary search
 *   trees, Journal of the ACM 32 (3): pp. 652--686, 1985. DOI:10.1145/3828.3835
 *
 *   This code is copyright 2002-2010 Thomas Lavergne and licenced under the BSD
 *   Licence like the remaining of Wapiti.
 ******************************************************************************/

/* qrk_node_t:
 *   Node of the splay tree whoe hold a (key, value) pair. The left and right
 *   childs are stored in an array so code for each side can be factorized.
 */
typedef struct qrk_node_s qrk_node_t;
struct qrk_node_s {
	qrk_node_t *child[2]; // Left and right childs of the node
	size_t      value;    // Value stored in the node
	char        key[];    // The key directly stored in the node
};

/* qrk_t:
 *   The quark database with his <tree> and <vector>. The database hold <count>
 *   (key, value) pairs, but the vector is of size <size>, it will grow as
 *   needed. If <lock> is true, new key will not be added to the quark and
 *   none will be returned as an identifier.
 */
struct qrk_s {
	qrk_node_t  *tree;    //       The tree for direct mapping
	qrk_node_t **vector;  // [N']  The array for the reverse mapping
	size_t       count;   //  N    The number of items in the database
	size_t       size;    //  N'   The real size of <vector>
	bool         lock;    //       Are new keys added to the dictionnary ?
};

/* qrk_newnode:
 *   Create a new qrk_node_t object with given key, value and no childs. The key
 *   is interned in the node so no reference is kept to the given string. The
 *   object must be freed with qrk_freenode when not needed anymore.
 */
static qrk_node_t *qrk_newnode(const char *key, size_t value) {
	const int len = strlen(key) + 1;
	qrk_node_t *nd = xmalloc(sizeof(qrk_node_t) + len);
	memcpy(nd->key, key, len);
	nd->value = value;
	nd->child[0] = NULL;
	nd->child[1] = NULL;
	return nd;
}

/* qrk_freenode:
 *   Free a qrk_node_t object and all his childs recursively.
 */
static void qrk_freenode(qrk_node_t *nd) {
	if (nd->child[0] != NULL)
		qrk_freenode(nd->child[0]);
	if (nd->child[1] != NULL)
		qrk_freenode(nd->child[1]);
	free(nd);
}

/* qrk_new:
 *   Create a new qrk_t object ready for doing mappings. This object must be
 *   freed with qrk_free when not used anymore.
 */
qrk_t *qrk_new(void) {
	qrk_t *qrk = xmalloc(sizeof(qrk_t));
	qrk->tree   = NULL;
	qrk->vector = NULL;
	qrk->count  = 0;
	qrk->size   = 0;
	qrk->lock   = false;
	return qrk;
}

/* qrk_free:
 *   Free all memory used by the given quark. All strings returned by qrk_unmap
 *   become invalid and must not be used anymore.
 */
void qrk_free(qrk_t *qrk) {
	for (size_t n = 0; n < qrk->count; n++)
		free(qrk->vector[n]);
	free(qrk->vector);
	free(qrk);
}

/* qrk_count:
 *   Return the number of mappings stored in the quark.
 */
size_t qrk_count(const qrk_t *qrk) {
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

/* qrk_splay:
 *   Do a splay operation on the tree part of the quark with the given key.
 *   Return -1 if the key is in the quark, so if it has been move at the top,
 *   else return the side wether the key should go.
 */
static int qrk_splay(qrk_t *qrk, const char *key) {
	qrk_node_t  nil = {{NULL, NULL}, 0};
	qrk_node_t *root[2] = {&nil, &nil};
	qrk_node_t *nd = qrk->tree;
	int side;
	while (true) {
		side = strcmp(key, nd->key);
		side = (side == 0) ? -1 : (side < 0) ? 0 : 1;
		if (side == -1 || nd->child[side] == NULL)
			break;
		const int tst = (side == 0)
			? strcmp(key, nd->child[side]->key) < 0
			: strcmp(key, nd->child[side]->key) > 0;
		if (tst) {
			qrk_node_t *tmp = nd->child[side];
			nd->child[side] = tmp->child[1 - side];
			tmp->child[1 - side] = nd;
			nd = tmp;
			if (nd->child[side] == NULL)
				break;
		}
		root[1 - side]->child[side] = nd;
		root[1 - side] = nd;
		nd = nd->child[side];
	}
	root[0]->child[1] = nd->child[0];
	root[1]->child[0] = nd->child[1];
	nd->child[0] = nil.child[1];
	nd->child[1] = nil.child[0];
	qrk->tree = nd;
	return side;
}

/* qrk_id2str:
 *   Return the key associated with the given identifier. The key must not be
 *   modified nor freed by the caller and remain valid for the lifetime of the
 *   quark object.
 *   Raise a fatal error if the indentifier is invalid.
 */
const char *qrk_id2str(const qrk_t *qrk, size_t id) {
	if (id >= qrk->count)
		fatal("invalid identifier");
	return qrk->vector[id]->key;
}

/* qrk_str2id:
 *   Return the identifier corresponding to the given key in the quark object.
 *   If the key is not already present, it is inserted unless the quark is
 *   locked, in which case none is returned.
 */
size_t qrk_str2id(qrk_t *qrk, const char *key) {
	// if tree is empty, directly add a root
	if (qrk->count == 0) {
		if (qrk->lock == true)
			return none;
		if (qrk->size == 0) {
			const size_t size = 128;
			qrk->vector = xmalloc(sizeof(char *) * size);
			qrk->size = size;
		}
		qrk_node_t *nd = qrk_newnode(key, 0);
		qrk->tree = nd;
		qrk->count = 1;
		qrk->vector[0] = nd;
		return 0;
	}
	// else if key is already there, return his value
	int side = qrk_splay(qrk, key);
	if (side == -1)
		return qrk->tree->value;
	if (qrk->lock == true)
		return none;
	// else, add the key to the quark
	if (qrk->count == qrk->size) {
		qrk->size *= 1.4;
		const size_t size = sizeof(char **) * qrk->size;
		qrk->vector = xrealloc(qrk->vector, size);
	}
	size_t id = qrk->count;
	qrk_node_t *nd = qrk_newnode(key, id);
	nd->child[    side] = qrk->tree->child[side];
	nd->child[1 - side] = qrk->tree;
	qrk->tree->child[side] = NULL;
	qrk->tree = nd;
	qrk->vector[id] = nd;
	qrk->count++;
	return id;
}

/* qrk_load:
 *   Load a quark object preivously saved with a call to qrk_load. The given
 *   quark must be empty.
 */
void qrk_load(qrk_t *qrk, FILE *file) {
	size_t cnt = 0;
	if (fscanf(file, "#qrk#%zu\n", &cnt) != 1) {
		if (ferror(file) != 0)
			pfatal("cannot read from file");
		pfatal("invalid format");
	}
	for (size_t n = 0; n < cnt; ++n) {
		char *str = ns_readstr(file);
		qrk_str2id(qrk, str);
		free(str);
	}
}

/* qrk_save:
 *   Save all the content of a quark object in the given file. The format is
 *   plain text and portable across platforms.
 */
void qrk_save(const qrk_t *qrk, FILE *file) {
	if (fprintf(file, "#qrk#%zu\n", qrk->count) < 0)
		pfatal("cannot write to file");
	if (qrk->count == 0)
		return;
	for (size_t n = 0; n < qrk->count; ++n)
		ns_writestr(file, qrk->vector[n]->key);
}

