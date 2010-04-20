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
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define unused(v) ((void)(v))

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) < (b) ? (b) : (a))

#define none ((size_t)-1)

/*******************************************************************************
 * Error handling and memory managment
 *
 *   Wapiti use a very simple system for error handling: violently fail. Errors
 *   can occurs in two cases, when user feed Wapiti with bad datas or when there
 *   is a problem on the system side. In both cases, there is nothing we can do,
 *   so the best thing is to exit with a meaning full error message.
 *
 *   Memory allocation is one of the possible point of failure and its painfull
 *   to always remeber to check return value of malloc so we provide wrapper
 *   around it and realloc who check and fail in case of error.
 ******************************************************************************/

/* fatal:
 *   This is the main error function, it will print the given message with same
 *   formating than the printf family and exit program with an error. We let the
 *   OS care about freeing ressources.
 */
static void fatal(const char *msg, ...) {
	va_list args;
	fprintf(stderr, "error: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n");
	exit(EXIT_FAILURE);
}

/* pfatal:
 *   This one is very similar to the fatal function but print an additional
 *   system error message depending on the errno. This can be used when a
 *   function who set the errno fail to print more detailed informations. You
 *   must be carefull to not call other functino that might reset it before
 *   calling pfatal.
 */
static void pfatal(const char *msg, ...) {
	const char *err = strerror(errno);
	va_list args;
	fprintf(stderr, "error: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n\t<%s>\n", err);
	exit(EXIT_FAILURE);
}

/* warning:
 *   This one is less violent as it just print a warning on stderr, but doesn't
 *   exit the program. It is intended to inform the user that something strange
 *   have happen and the result might be not what it have expected.
 */
static void warning(const char *msg, ...) {
	va_list args;
	fprintf(stderr, "warning: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n");
}

/* xmalloc:
 *   A simple wrapper around malloc who violently fail if memory cannot be
 *   allocated, so it will never return NULL.
 */
static void *xmalloc(size_t size) {
	void *ptr = malloc(size);
	if (ptr == NULL)
		fatal("out of memory");
	return ptr;
}

/* xrealloc:
 *   As xmalloc, this is a simple wrapper around realloc who fail on memory
 *   error and so never return NULL.
 */
static void *xrealloc(void *ptr, size_t size) {
	void *new = realloc(ptr, size);
	if (new == NULL)
		fatal("out of memory");
	return new;
}

/* xstrdup:
 *   As the previous one, this is a safe version of xstrdup who fail on
 *   allocation error.
 */
static char *xstrdup(const char *str) {
	const int len = strlen(str) + 1;
	char *res = xmalloc(sizeof(char) * len);
	memcpy(res, str, len);
	return res;
}

/*******************************************************************************
 * Sequences and Dataset objects
 *
 *   Sequences represent the input data feeded by the user in Wapiti either for
 *   training or labelling. The internal form used here is very different from
 *   the data read from files and the convertion process is done in three steps
 *   illustrated here:
 *         +------+     +-------+     +-------+     +-------+
 *         | FILE | --> | raw_t | --> | tok_t | --> | seq_t |
 *         +------+     +-------+     +-------+     +-------+
 *   First the sequence is read as a set of lines from the input file, this
 *   give a raw_t object. Next this set of lines is split in tokens and
 *   eventually the last one is separated as it will become a label, this result
 *   in a tok_t object.
 *   The last step consist in applying all the patterns givens by the user to
 *   extract from these tokens the observations made on the sequence in order to
 *   build the seq_t object which can be used by the trainer and tagger.
 *
 *   A dataset object is just a container for a list of sequences in internal
 *   form used to store either training or development set.
 *
 *   All the convertion process is driven by the reader object and, as it is
 *   responsible for creating the objects with a quite special allocation
 *   scheme, we just have to implement function for freeing these objects here.
 ******************************************************************************/

/* raw_t:
 *   Data-structure representing a raw sequence as a set of lines read from the
 *   input file. This is the result of the first step of the interning process.
 *   We keep this form separate from the tokenized one as we want to be able to
 *   output the sequence as it was read in the labelling mode.
 *
 *   This represent a sequence of lengths <len> and for each position 't' you
 *   find the corresponding line at <lines>[t].
 *
 *   The <lines> array is allocated with data structure, and the different lines
 *   are allocated separatly.
 */
typedef struct raw_s {
	int   len;      //   T     Sequence length
	char *lines[];  //  [T]    Raw lines directly from file
} raw_t;

/* tok_t:
 *   Data-structure representing a tokenized sequence. This is the result of the
 *   second step of the interning process after the raw sequence have been split
 *   in tokens and eventual labels separated from the observations.
 *
 *   For each position 't' in the sequence of length <len>, you find at <lbl>[t]
 *   the eventual label provided in input file, and at <toks>[t] a list of
 *   string tokens of length <cnts>[t].
 *
 *   Memory allocation here is a bit special as the first token at each position
 *   point to a memory block who hold a copy of the raw line. Each other tokens
 *   and the label are pointer in this block. This reduce memory fragmentation.
 */
typedef struct tok_s {
	int    len;     //   T     Sequence length
	char **lbl;     //  [T]    List of labels strings
	int   *cnts;    //  [T]    Length of tokens lists
	char **toks[];  //  [T][]  Tokens lists
} tok_t;

/* seq_t:
 *   Data-structure representing a sequence of length <len> in the internal form
 *   used by the trainers and the tagger. For each position 't' in the sequence
 *   (0 <= t < <len>) there is some observations made on the data and an
 *   eventual label if provided in the input file.
 *
 *   There is two kind of features: unigrams and bigrams one, build by combining
 *   one observation and one or two labels. At position 't', the unigrams
 *   features are build using the list of observations from <uobs>[t] which
 *   contains <ucnt>[t] items, and the observation at <lbl>[t]. The bigrams
 *   features are obtained in the same way using <bobs> and <bcnt>, and have to
 *   be combined also with <lbl>[t-1].
 *
 *   If the sequence is read from a file without label, as it is the case in
 *   labelling mode, the <lbl> field will be NULL and so, the sequence cannot be
 *   used for training.
 *
 *   The raw field is private and used internaly for efficient memory
 *   allocation. This allow to allocate <lbl>, <*cnt>, and all the list in
 *   <*obs> with the datastructure itself.
 */
typedef struct pos_s pos_t;
typedef struct seq_s seq_t;
struct seq_s {
	int     len;
	size_t *raw;
	struct pos_s {
		size_t  lbl;
		size_t  ucnt,  bcnt;
		size_t *uobs, *bobs;
	} pos[];
};

/* dat_t:
 *   Data-structure representing a full dataset: a collection of sequences ready
 *   to be used for training or to be labelled. It keep tracks of the maximum
 *   sequence length as the trainer need this for memory allocation. The dataset
 *   contains <nseq> sequence stored in <seq>. These sequences are labeled only
 *   if <lbl> is true.
 */
typedef struct dat_s {
	bool    lbl;   //         True iff sequences are labelled
	int     mlen;  //         Length of the longest sequence in the set
	int     nseq;  //   S     Number of sequences in the set
	seq_t **seq;   //  [S]    List of sequences
} dat_t;

/******************************************************************************
 *                               Quark database
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
 *
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
 *   The quark database with his <tree> and <vector>. The dabase hold <count>
 *   (key, value) pairs, but the vector is of size <size>, it will grow as
 *   needed. If <lock> is true, new key will not be added to the quark and
 *   none will be returned as an identifier.
 */
typedef struct qrk_s qrk_t;
struct qrk_s {
	qrk_node_t  *tree;    //       The tree for direct mapping
	char       **vector;  // [N']  The array for the reverse mapping
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
static qrk_t *qrk_new(void) {
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
static void qrk_free(qrk_t *qrk) {
	if (qrk->count != 0) {
		qrk_freenode(qrk->tree);
		free(qrk->vector);
	} else if (qrk->vector != NULL) {
		free(qrk->vector);
	}
	free(qrk);
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
static const char *qrk_id2str(const qrk_t *qrk, size_t id) {
	if (id >= qrk->count)
		fatal("invalid identifier");
	return qrk->vector[id];
}

/* qrk_str2id:
 *   Return the identifier corresponding to the given key in the quark object.
 *   If the key is not already present, it is inserted unless the quark is
 *   locked, in which case none is returned.
 */
static size_t qrk_str2id(qrk_t *qrk, const char *key) {
	// if tree is empty, directly add a root
	if (qrk->count == 0) {
		if (qrk->size == 0) {
			const size_t size = 128;
			qrk->vector = xmalloc(sizeof(char *) * size);
			qrk->size = size;
		}
		qrk_node_t *nd = qrk_newnode(key, 0);
		qrk->tree = nd;
		qrk->count = 1;
		qrk->vector[0] = nd->key;
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
	qrk->vector[id] = nd->key;
	qrk->count++;
	return id;
}

/******************************************************************************
 * A simple regular expression matcher
 *
 *   This module implement a simple regular expression matcher, it implement
 *   just a subset of the classical regexp simple to implement but sufficient
 *   for most usages and avoid to add a dependency to a full regexp library.
 *
 *   The recognized subset is quite simple. First for matching characters :
 *       .  -> match any characters
 *       \x -> match a character class (in uppercase, match the complement)
 *               \d : digit       \a : alpha      \w : alpha + digit
 *               \l : lowercase   \u : uppercase  \p : punctuation
 *             or escape a character
 *       x  -> any other character match itself
 *   And the constructs :
 *       ^  -> at the begining of the regexp, anchor it at start of string
 *       $  -> at the end of regexp, anchor it at end of string
 *       *  -> match any number of repetition of the previous character
 *       ?  -> optionally match the previous character
 *
 *   This subset is implemented quite efficiently using recursion. All recursive
 *   calls are tail-call so they should be optimized by the compiler. As we do
 *   direct interpretation, we have to backtrack so performance can be very poor
 *   on specialy designed regexp. This is not a problem as the regexp as well as
 *   the string is expected to be very simple here. If this is not the case, you
 *   better have to prepare your data better.
 ******************************************************************************/

/* rex_matchit:
 *   Match a single caracter at the start fo the string. The character might be
 *   a plain char, a dot or char class.
 */
static bool rex_matchit(const char *ch, const char *str) {
	if (str[0] == '\0')
		return false;
	if (ch[0] == '.')
		return true;
	if (ch[0] == '\\') {
		switch (ch[1]) {
			case 'a': return  isalpha(str[0]);
			case 'd': return  isdigit(str[0]);
			case 'l': return  islower(str[0]);
			case 'p': return  ispunct(str[0]);
			case 'u': return  isupper(str[0]);
			case 'w': return  isalnum(str[0]);
			case 'A': return !isalpha(str[0]);
			case 'D': return !isdigit(str[0]);
			case 'L': return !islower(str[0]);
			case 'P': return !ispunct(str[0]);
			case 'U': return !isupper(str[0]);
			case 'W': return !isalnum(str[0]);
		}
		return ch[1] == str[0];
	}
	return ch[0] == str[0];
}

/* rex_matchme:
 *   Match a regular expresion at the start of the string. If a match is found,
 *   is length is returned in len. The mathing is done through tail-recursion
 *   for good performances.
 */
static bool rex_matchme(const char *re, const char *str, int *len) {
	// Special check for end of regexp
	if (re[0] == '\0')
		return true;
	if (re[0] == '$' && re[1] == '\0')
		return (str[0] == '\0');
	// Get first char of regexp
	const char *ch  = re;
	const char *nxt = re + 1 + (ch[0] == '\\');
	// Special check for the following construct "x**" where the first star
	// is consumed normally but lead the second (which is wrong) to be
	// interpreted as a char to mach as if it was escaped (and same for the
	// optional construct)
	if (*ch == '*' || *ch == '?')
		fatal("unescaped * or ? in regexp: %s", re);
	// Handle star repetition
	if (nxt[0] == '*') {
		nxt++;
		do {
			const int save = *len;
			if (rex_matchme(nxt, str, len))
				return true;
			*len = save + 1;
		} while (rex_matchit(ch, str++));
		return false;
	}
	// Handle optional
	if (nxt[0] == '?') {
		nxt++;
		if (rex_matchit(ch, str)) {
			(*len)++;
			if (rex_matchme(nxt, str + 1, len))
				return true;
			(*len)--;
		}
		return rex_matchme(nxt, str, len);
	}
	// Classical char matching
	(*len)++;
	if (rex_matchit(ch, str))
		return rex_matchme(nxt, str + 1, len);
	return false;
}

/* rex_match:
 *   Match a regular expresion in the given string. If a match is found, the
 *   position of the start of the match is returned and is len is returned in
 *   len, else -1 is returned.
 */
static int rex_match(const char *re, const char *str, int *len) {
	// Special case for anchor at start
	if (*re == '^') {
		*len = 0;
		if (rex_matchme(re + 1, str, len))
			return 0;
		return -1;
	}
	// And general case for any position
	int pos = 0;
	do {
		*len = 0;
		if (rex_matchme(re, str + pos, len))
			return pos;
	} while (str[pos++] != '\0');
	// Matching failed
	return -1;
}

/*******************************************************************************
 * Pattern handling
 *
 *   Patterns are the heart the data input process, they provide a way to tell
 *   Wapiti how the interesting information can be extracted from the input
 *   data. A pattern is simply a string who embed special commands about tokens
 *   to extract from the input sequence. They are compiled to a special form
 *   used during data loading.
 *   For training, each position of a sequence hold a list of observation made
 *   at this position, pattern give a way to specify these observations.
 *
 *   During sequence loading, all patterns are applied at each position to
 *   produce a list of string representing the observations which will be in
 *   turn transformed to numerical identifiers. This module take care of
 *   building the string representation.
 *
 *   As said, a patern is a string with specific commands in the forms %c[...]
 *   where 'c' is the command with arguments between the bracket. All commands
 *   take at least to numerical arguments which define a token in the input
 *   sequence. The first one is an offset from the current position and the
 *   second one is a column number. With these two parameters, we get a string
 *   in the input sequence on which we apply the command.
 *
 *   All command are specified with a character and result in a string which
 *   will replace the command in the pattern string. If the command character is
 *   lower case, the result is copied verbatim, if it is uppercase, the result
 *   is copied with casing removed. The following commands are available:
 *     'x' -- result is the token itself
 *     't' -- test if a regular expression match the token. Result will be
 *            either "true" or "false"
 *     'm' -- match a regular expression on the token. Result is the first
 *            substring matched.
 ******************************************************************************/

typedef struct pat_s pat_t;
typedef struct pat_item_s pat_item_t;
struct pat_s {
	char *src;
	int   ntoks;
	int   nitems;
	struct pat_item_s {
		char  type;
		bool  caps;
		char *value;
		int   offset;
		int   column;
	} items[];
};

/* pat_comp:
 *   Compile the pattern to a form more suitable to easily apply it on tokens
 *   list during data reading. The given pattern string is interned in the
 *   compiled pattern and will be freed with it, so you don't have to take care
 *   of it and must not modify it after the compilation.
 */
static pat_t *pat_comp(char *p) {
	pat_t *pat = NULL;
	// Allocate memory for the compiled pattern, the allocation is based
	// on an over-estimation of the number of required item. As compiled
	// pattern take a neglectible amount of memory, this waste is not
	// important.
	int mitems = 0;
	for (int pos = 0; p[pos] != '\0'; pos++)
		if (p[pos] == '%')
			mitems++;
	mitems = mitems * 2 + 1;
	pat = xmalloc(sizeof(pat_t) + sizeof(pat->items[0]) * mitems);
	pat->src = p;
	// Next, we go through the pattern compiling the items as they are
	// found. Commands are parsed and put in a corresponding item, and
	// segment of char not in a command are put in a 's' item.
	int nitems = 0;
	int ntoks = 0;
	int pos = 0;
	while (p[pos] != '\0') {
		pat_item_t *item = &(pat->items[nitems++]);
		item->value = NULL;
		if (p[pos] == '%') {
			// This is a command, so first parse its type and check
			// its a valid one. Next prepare the item.
			const char type = tolower(p[pos + 1]);
			if (type != 'x' && type != 't' && type != 'm')
				fatal("unknown command type: '%c'", type);
			item->type = type;
			item->caps = (p[pos + 1] != type);
			pos += 2;
			// Next we parse the offset and column and store them in
			// the item.
			int off, col, nch;
			if (sscanf(p + pos, "[%d,%d%n", &off, &col, &nch) != 2)
				fatal("invalid pattern: %s", p);
			if (col < 0)
				fatal("invalid column number: %d", col);
			item->offset = off;
			item->column = col;
			ntoks = max(ntoks, col);
			pos += nch;
			// And parse the end of the argument list, for 'x' there
			// is nothing to read but for 't' and 'm' we have to get
			// read the regexp.
			if (type == 't' || type == 'm') {
				if (p[pos] != ',' && p[pos + 1] != '"')
					fatal("missing arg in pattern: %s", p);
				const int start = (pos += 2);
				while (p[pos] != '\0') {
					if (p[pos] == '"')
						break;
					if (p[pos] == '\\' && p[pos+1] != '\0')
						pos++;
					pos++;
				}
				if (p[pos] != '"')
					fatal("unended argument: %s", p);
				const int len = pos - start;
				item->value = xmalloc(sizeof(char) * (len + 1));
				memcpy(item->value, p + start, len);
				item->value[len] = '\0';
				pos++;
			}
			// Just check the end of the arg list and loop.
			if (p[pos] != ']')
				fatal("missing end of pattern: %s", p);
			pos++;
		} else {
			// No command here, so build an 's' item with the chars
			// until end of pattern or next command and put it in
			// the list.
			const int start = pos;
			while (p[pos] != '\0' && p[pos] != '%')
				pos++;
			const int len = pos - start;
			item->type  = 's';
			item->caps  = false;
			item->value = xmalloc(sizeof(char) * (len + 1));
			memcpy(item->value, p + start, len);
			item->value[len] = '\0';
		}
	}
	pat->ntoks = ntoks;
	pat->nitems = nitems;
	return pat;
}

/* pat_exec:
 *   Execute a compiled pattern at position 'at' in the given tokens sequences
 *   in order to produce an observation string. The string is returned as a
 *   newly allocated memory block and the caller is responsible to free it when
 *   not needed anymore.
 */
static char *pat_exec(pat_t *pat, tok_t *tok, int at) {
	static char *bval[] = {"_x-1", "_x-2", "_x-3", "_x-4", "_x-#"};
	static char *eval[] = {"_x+1", "_x+2", "_x+3", "_x+4", "_x+#"};
	const int T = tok->len;
	// Prepare the buffer who will hold the result
	int size = 16, pos = 0;
	char *buffer = xmalloc(sizeof(char) * size);
	// And loop over the compiled items
	for (int it = 0; it < pat->nitems; it++) {
		const pat_item_t *item = &(pat->items[it]);
		char *value = NULL;
		int len = 0;
		// First, if needed, we retrieve the token at the referenced
		// position in the sequence. We store it in value and let the
		// command handler do what it need with it.
		if (item->type != 's') {
			int pos = at + item->offset;
			int col = item->column;
			if (pos < 0)
				value = bval[min(-pos - 1, 4)];
			else if (pos >= T)
				value = eval[min( pos - T, 4)];
			else if (col >= tok->cnts[pos])
				fatal("missing tokens, cannot apply pattern");
			else
				value = tok->toks[pos][col];
		}
		// Next, we handle the command, 's' and 'x' are very simple but
		// 't' and 'm' require us to call the regexp matcher.
		if (item->type == 's') {
			value = item->value;
			len = strlen(value);
		} else if (item->type == 'x') {
			len = strlen(value);
		} else if (item->type == 't') {
			if (rex_match(item->value, value, &len) == -1)
				value = "false";
			else
				value = "true";
			len = strlen(value);
		} else if (item->type == 'm') {
			int pos = rex_match(item->value, value, &len);
			if (pos == -1)
				len = 0;
			value += pos;
		}
		// And we add it to the buffer, growing it if needed. If the
		// user requested it, we also remove caps from the string.
		if (pos + len >= size - 1) {
			while (pos + len >= size - 1)
				size = size * 1.4;
			buffer = xrealloc(buffer, sizeof(char) * size);
		}
		memcpy(buffer + pos, value, len);
		if (item->caps)
			for (int i = pos; i < pos + len; i++)
				buffer[i] = tolower(buffer[i]);
		pos += len;
	}
	// Adjust the result and return it.
	buffer[pos++] = '\0';
	buffer = xrealloc(buffer, sizeof(char) * pos);
	return buffer;
}

/* pat_free:
 *   Free all memory used by a compiled pattern object. Note that this will free
 *   the pointer to the source string given to pat_comp so you must be sure to
 *   not use this pointer again.
 */
static void pat_free(pat_t *pat) {
	for (int it = 0; it < pat->nitems; it++)
		free(pat->items[it].value);
	free(pat->src);
	free(pat);
}

/*******************************************************************************
 * Datafile reader
 *
 *   And now come the data file reader which use the previous module to parse
 *   the input data in order to produce seq_t objects representing interned
 *   sequences.
 *
 *   This is where the sequence will go through the tree steps to build seq_t
 *   objects used internally. There is two way do do this. First the simpler is
 *   to use the rdr_readseq function which directly read a sequence from a file
 *   and convert it to a seq_t object transparently. This is how the training
 *   and development data are loaded.
 *   The second way consist of read a raw sequence with rdr_readraw and next
 *   converting it to a seq_t object with rdr_raw2seq. This allow the caller to
 *   keep the raw sequence and is used by the tagger to produce a clean output.
 *
 *   There is no public interface to the tok_t object as it is intended only for
 *   internal use in the reader as an intermediate step to apply patterns.
 ******************************************************************************/

/* rdr_t:
 *   The reader object who hold all informations needed to parse the input file:
 *   the patterns and quark for labels and observations. We keep separate count
 *   for unigrams and bigrams pattern for simpler allocation of sequences. We
 *   also store the expected number of column in the input data to check that
 *   pattern are appliables.
 */
typedef struct rdr_s rdr_t;
struct rdr_s {
	int     npats;      //  P   Total number of patterns
	int     nuni, nbi;  //      Number of unigram and bigram patterns
	int     ntoks;      //      Expected number of tokens in input
	pat_t **pats;       // [P]  List of precompiled patterns
	qrk_t  *lbl;        //      Labels database
	qrk_t  *obs;        //      Observation database
};

/* rdr_new:
 *   Create a new empty reader object. You mut load patterns in it or a
 *   previously saved reader if you want to use it for reading sequences.
 */
static rdr_t *rdr_new(void) {
	rdr_t *rdr = xmalloc(sizeof(rdr_t));
	rdr->npats = rdr->nuni = rdr->nbi = 0;
	rdr->ntoks = 0;
	rdr->pats = NULL;
	rdr->lbl = qrk_new();
	rdr->obs = qrk_new();
	return rdr;
}

/* rdr_free:
 *   Free all memory used by a reader object including the quark database, so
 *   any string returned by them must not be used after this call.
 */
static void rdr_free(rdr_t *rdr) {
	for (int i = 0; i < rdr->npats; i++)
		pat_free(rdr->pats[i]);
	free(rdr->pats);
	qrk_free(rdr->lbl);
	qrk_free(rdr->obs);
	free(rdr);
}

/* rdr_freeraw:
 *   Free all memory used by a raw_t object.
 */
static void rdr_freeraw(raw_t *raw) {
	for (int t = 0; t < raw->len; t++)
		free(raw->lines[t]);
	free(raw);
}

/* rdr_freeseq:
 *   Free all memory used by a seq_t object.
 */
static void rdr_freeseq(seq_t *seq) {
	free(seq->raw);
	free(seq);
}

/* rdr_readline:
 *   Read an input line from <file>. The line can be of any size limited only by
 *   available memory, a buffer large enough is allocated and returned. The
 *   caller is responsible to free it. On end-of-file, NULL is returned.
 */
static char *rdr_readline(FILE *file) {
	if (feof(file))
		return NULL;
	// Initialize the buffer
	int len = 0, size = 16;
	char *buffer = xmalloc(size);
	// We read the line chunk by chunk until end of line, file or error
	while (!feof(file)) {
		if (fgets(buffer + len, size - len, file) == NULL) {
			// On NULL return there is two possible cases, either an
			// error or the end of file
			if (ferror(file))
				pfatal("cannot read from file");
			// On end of file, we must check if we have already read
			// some data or not
			if (len == 0) {
				free(buffer);
				return NULL;
			}
			break;
		}
		// Check for end of line, if this is not the case enlarge the
		// buffer and go read more data
		len += strlen(buffer + len);
		if (len == size - 1 && buffer[len - 1] != '\n') {
			size = size * 1.4;
			buffer = xrealloc(buffer, size);
			continue;
		}
		break;
	}
	// At this point empty line should have already catched so we just
	// remove the end of line if present and resize the buffer to fit the
	// data
	if (buffer[len - 1] == '\n')
		buffer[--len] = '\0';
	return xrealloc(buffer, len + 1);
}

/* rdr_loadpat:
 *   Load and compile patterns from given file and store them in the reader. As
 *   we compile patterns, syntax errors in them will be raised at this time.
 */
static void rdr_loadpat(rdr_t *rdr, FILE *file) {
	while (!feof(file)) {
		// Read raw input line
		char *line = rdr_readline(file);
		if (line == NULL)
			break;
		// Remove comments and trailing spaces
		int end = strcspn(line, "#");
		while (end != 0 && isspace(line[end - 1]))
			end--;
		if (end == 0) {
			free(line);
			continue;
		}
		line[end] = '\0';
		line[0] = tolower(line[0]);
		// Compile pattern and add it to the list
		pat_t *pat = pat_comp(line);
		rdr->npats++;
		switch (line[0]) {
			case 'u': rdr->nuni++; break;
			case 'b': rdr->nbi++; break;
			case '*': rdr->nuni++;
			          rdr->nbi++; break;
			default:
				fatal("unknown pattern type '%c'", line[0]);
		}
		rdr->pats = xrealloc(rdr->pats, sizeof(char *) * rdr->npats);
		rdr->pats[rdr->npats - 1] = pat;
		rdr->ntoks = max(rdr->ntoks, pat->ntoks);
	}
}

/* rdr_readraw:
 *   Read a raw sequence from given file: a set of lines terminated by end of
 *   file or by an empty line. Return NULL if file end was reached before any
 *   sequence was read.
 *   The reader object is not used in this function but is specified as a
 *   parameter to be more coherent.
 */
static raw_t *rdr_readraw(rdr_t *rdr, FILE *file) {
	unused(rdr);
	if (feof(file))
		return NULL;
	// Prepare the raw sequence object
	int size = 32, cnt = 0;
	raw_t *raw = xmalloc(sizeof(raw_t) + sizeof(char *) * size);
	// And read the next sequence in the file, this will skip any blank line
	// before reading the sequence stoping at end of file or on a new blank
	// line.
	while (!feof(file)) {
		char *line = rdr_readline(file);
		if (line == NULL)
			break;
		// Check for empty line marking the end of the current sequence
		int len = strlen(line);
		while (len != 0 && isspace(line[len - 1]))
			len--;
		if (len == 0) {
			free(line);
			// Special case when no line was already read, we try
			// again. This allow multiple blank lines beetwen
			// sequences.
			if (cnt == 0)
				continue;
			break;
		}
		// Next, grow the buffer if needed and add the new line in it
		if (size == cnt) {
			size *= 1.4;
			raw = xrealloc(raw, sizeof(raw_t)
			                + sizeof(char *) * size);
		}
		raw->lines[cnt++] = line;
	}
	// If no lines was read, we just free allocated memory and return NULL
	// to signal the end of file to the caller. Else, we adjust the object
	// size and return it.
	if (cnt == 0) {
		free(raw);
		return NULL;
	}
	raw = xrealloc(raw, sizeof(raw_t) + sizeof(char *) * cnt);
	raw->len = cnt;
	return raw;
}

/* rdr_raw2seq:
 *   Convert a raw sequence to a seq_t object suitable for training or
 *   labelling. If lbl is true, the last column is assumed to be a label and
 *   interned also.
 */
static seq_t *rdr_raw2seq(rdr_t *rdr, const raw_t *raw, bool lbl) {
	const int T = raw->len;
	// Allocate the tok_t object, the label array is allocated only if they
	// are requested by the user.
	tok_t *tok = xmalloc(sizeof(tok_t) + T * sizeof(char **));
	tok->cnts = xmalloc(sizeof(size_t) * T);
	tok->lbl = NULL;
	if (lbl == true)
		tok->lbl = xmalloc(sizeof(char *) * T);
	// We now take the raw sequence line by line and split them in list of
	// tokens. To reduce memory fragmentation, the raw line is copied and
	// his reference is kept by the first tokens, next tokens are pointer to
	// this copy.
	for (int t = 0; t < T; t++) {
		// Get a copy of the raw line skiping leading space characters
		const char *src = raw->lines[t];
		while (isspace(*src))
			src++;
		char *line = xstrdup(src);
		// Split it in tokens
		const int len = strlen(line);
		char *toks[len / 2];
		int cnt = 0;
		while (*line != '\0') {
			toks[cnt++] = line;
			while (*line != '\0' && !isspace(*line))
				line++;
			if (*line == '\0')
				break;
			*line++ = '\0';
			while (*line != '\0' && isspace(*line))
				line++;
		}
		// If user specified that data are labelled, move the last token
		// to the label array.
		if (lbl == true) {
			tok->lbl[t] = toks[cnt - 1];
			cnt--;
		}
		// And put the remaining tokens in the tok_t object
		tok->cnts[t] = cnt;
		tok->toks[t] = xmalloc(sizeof(char *) * cnt);
		memcpy(tok->toks[t], toks, sizeof(char *) * cnt);
	}
	tok->len = T;
	// So now the tok object is ready, we can start building the seq_t
	// object by appling patterns. First we allocate the seq_t object. The
	// sequence itself as well as the sub array are allocated in one time.
	seq_t *seq = xmalloc(sizeof(seq_t) + sizeof(pos_t) * T);
	seq->raw = xmalloc(sizeof(size_t) * (rdr->nuni + rdr->nbi) * T);
	seq->len = T;
	size_t *tmp = seq->raw;
	for (int t = 0; t < T; t++) {
		seq->pos[t].lbl  = none;
		seq->pos[t].uobs = tmp; tmp += rdr->nuni;
		seq->pos[t].bobs = tmp; tmp += rdr->nbi;
	}
	// Next, we can build the observations list by applying the patterns on
	// the tok_t sequence.
	for (int t = 0; t < T; t++) {
		pos_t *pos = &seq->pos[t];
		pos->ucnt = 0;
		pos->bcnt = 0;
		for (int x = 0; x < rdr->npats; x++) {
			// Get the observation and map it to an identifier
			const char *obs = pat_exec(rdr->pats[x], tok, t);
			size_t id = qrk_str2id(rdr->obs, obs);
			if (id == none)
				continue;
			// If the observation is ok, add it to the lists
			int kind = 0;
			switch (obs[0]) {
				case 'u': kind = 1; break;
				case 'b': kind = 2; break;
				case '*': kind = 3; break;
			}
			if (kind & 1)
				pos->uobs[pos->ucnt++] = id;
			if (kind & 2)
				pos->bobs[pos->bcnt++] = id;
		}
	}
	// And finally, if the user specified it, populate the labels
	if (lbl == true) {
		for (int t = 0; t < T; t++) {
			const char *lbl = tok->lbl[t];
			size_t id = qrk_str2id(rdr->lbl, lbl);
			seq->pos[t].lbl = id;
		}
	}
	// Before returning the sequence, we have to free the tok_t
	for (int t = 0; t < T; t++) {
		if (tok->cnts[t] == 0)
			continue;
		free(tok->toks[t][0]);
		free(tok->toks[t]);
	}
	free(tok->cnts);
	if (lbl == true)
		free(tok->lbl);
	free(tok);
	return seq;
}

/*******************************************************************************
 *
 ******************************************************************************/
int main(void) {
	return EXIT_SUCCESS;
}

