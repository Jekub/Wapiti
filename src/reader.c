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
#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "wapiti.h"
#include "pattern.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"

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

/* rdr_new:
 *   Create a new empty reader object. If no patterns are loaded before you
 *   start using the reader the input data are assumed to be already prepared
 *   list of features. They must either start with a prefix 'u', 'b', or '*', or
 *   you must set autouni to true in order to automatically add a 'u' prefix.
 */
rdr_t *rdr_new(bool autouni) {
	rdr_t *rdr = xmalloc(sizeof(rdr_t));
	rdr->autouni = autouni;
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
void rdr_free(rdr_t *rdr) {
	for (uint32_t i = 0; i < rdr->npats; i++)
		pat_free(rdr->pats[i]);
	free(rdr->pats);
	qrk_free(rdr->lbl);
	qrk_free(rdr->obs);
	free(rdr);
}

/* rdr_freeraw:
 *   Free all memory used by a raw_t object.
 */
void rdr_freeraw(raw_t *raw) {
	for (uint32_t t = 0; t < raw->len; t++)
		free(raw->lines[t]);
	free(raw);
}

/* rdr_freeseq:
 *   Free all memory used by a seq_t object.
 */
void rdr_freeseq(seq_t *seq) {
	free(seq->raw);
	free(seq);
}

/* rdr_freedat:
 *   Free all memory used by a dat_t object.
 */
void rdr_freedat(dat_t *dat) {
	for (uint32_t i = 0; i < dat->nseq; i++)
		rdr_freeseq(dat->seq[i]);
	free(dat->seq);
	free(dat);
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
	uint32_t len = 0, size = 16;
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
void rdr_loadpat(rdr_t *rdr, FILE *file) {
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
 */
raw_t *rdr_readraw(rdr_t *rdr, FILE *file) {
	if (feof(file))
		return NULL;
	// Prepare the raw sequence object
	uint32_t size = 32, cnt = 0;
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
		// In autouni mode, there will be only unigram features so we
		// can use small sequences to improve multi-theading.
		if (rdr->autouni)
			break;
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

/* rdr_mapobs:
 *   Map an observation to its identifier, automatically adding a 'u' prefix in
 *   'autouni' mode.
 */
static uint64_t rdr_mapobs(rdr_t *rdr, const char *str) {
	if (!rdr->autouni)
		return qrk_str2id(rdr->obs, str);
	char tmp[strlen(str) + 2];
	tmp[0] = 'u';
	strcpy(tmp + 1, str);
	return qrk_str2id(rdr->obs, tmp);
}

/* rdr_rawtok2seq:
 *   Convert a tok_t to a seq_t object taking each tokens as a feature without
 *   applying patterns.
 */
static seq_t *rdr_rawtok2seq(rdr_t *rdr, const tok_t *tok) {
	const uint32_t T = tok->len;
	uint32_t size = 0;
	if (rdr->autouni) {
		size = tok->cnts[0];
	} else {
		for (uint32_t t = 0; t < T; t++) {
			for (uint32_t n = 0; n < tok->cnts[t]; n++) {
				const char *o = tok->toks[t][n];
				switch (o[0]) {
					case 'u': size += 1; break;
					case 'b': size += 1; break;
					case '*': size += 2; break;
					default:
						fatal("invalid feature: %s", o);
				}
			}
		}
	}
	seq_t *seq = xmalloc(sizeof(seq_t) + sizeof(pos_t) * T);
	seq->raw = xmalloc(sizeof(uint64_t) * size);
	seq->len = T;
	uint64_t *raw = seq->raw;
	for (uint32_t t = 0; t < T; t++) {
		seq->pos[t].lbl = (uint32_t)-1;
		seq->pos[t].ucnt = 0;
		seq->pos[t].uobs = raw;
		for (uint32_t n = 0; n < tok->cnts[t]; n++) {
			if (!rdr->autouni && tok->toks[t][n][0] == 'b')
				continue;
			uint64_t id = rdr_mapobs(rdr, tok->toks[t][n]);
			if (id != none) {
				(*raw++) = id;
				seq->pos[t].ucnt++;
			}
		}
		seq->pos[t].bcnt = 0;
		if (rdr->autouni)
			continue;
		seq->pos[t].bobs = raw;
		for (uint32_t n = 0; n < tok->cnts[t]; n++) {
			if (tok->toks[t][n][0] == 'u')
				continue;
			uint64_t id = rdr_mapobs(rdr, tok->toks[t][n]);
			if (id != none) {
				(*raw++) = id;
				seq->pos[t].bcnt++;
			}
		}
	}
	// And finally, if the user specified it, populate the labels
	if (tok->lbl != NULL) {
		for (uint32_t t = 0; t < T; t++) {
			const char *lbl = tok->lbl[t];
			uint64_t id = qrk_str2id(rdr->lbl, lbl);
			seq->pos[t].lbl = id;
		}
	}
	return seq;
}

/* rdr_pattok2seq:
 *   Convert a tok_t to a seq_t object by applying the patterns of the reader.
 */
static seq_t *rdr_pattok2seq(rdr_t *rdr, const tok_t *tok) {
	const uint32_t T = tok->len;
	// So now the tok object is ready, we can start building the seq_t
	// object by appling patterns. First we allocate the seq_t object. The
	// sequence itself as well as the sub array are allocated in one time.
	seq_t *seq = xmalloc(sizeof(seq_t) + sizeof(pos_t) * T);
	seq->raw = xmalloc(sizeof(uint64_t) * (rdr->nuni + rdr->nbi) * T);
	seq->len = T;
	uint64_t *tmp = seq->raw;
	for (uint32_t t = 0; t < T; t++) {
		seq->pos[t].lbl  = (uint32_t)-1;
		seq->pos[t].uobs = tmp; tmp += rdr->nuni;
		seq->pos[t].bobs = tmp; tmp += rdr->nbi;
	}
	// Next, we can build the observations list by applying the patterns on
	// the tok_t sequence.
	for (uint32_t t = 0; t < T; t++) {
		pos_t *pos = &seq->pos[t];
		pos->ucnt = 0;
		pos->bcnt = 0;
		for (uint32_t x = 0; x < rdr->npats; x++) {
			// Get the observation and map it to an identifier
			char *obs = pat_exec(rdr->pats[x], tok, t);
			uint64_t id = rdr_mapobs(rdr, obs);
			if (id == none) {
				free(obs);
				continue;
			}
			// If the observation is ok, add it to the lists
			char kind = 0;
			switch (obs[0]) {
				case 'u': kind = 1; break;
				case 'b': kind = 2; break;
				case '*': kind = 3; break;
			}
			if (kind & 1)
				pos->uobs[pos->ucnt++] = id;
			if (kind & 2)
				pos->bobs[pos->bcnt++] = id;
			free(obs);
		}
	}
	// And finally, if the user specified it, populate the labels
	if (tok->lbl != NULL) {
		for (uint32_t t = 0; t < T; t++) {
			const char *lbl = tok->lbl[t];
			uint64_t id = qrk_str2id(rdr->lbl, lbl);
			seq->pos[t].lbl = id;
		}
	}
	return seq;
}

/* rdr_raw2seq:
 *   Convert a raw sequence to a seq_t object suitable for training or
 *   labelling. If lbl is true, the last column is assumed to be a label and
 *   interned also.
 */
seq_t *rdr_raw2seq(rdr_t *rdr, const raw_t *raw, bool lbl) {
	const uint32_t T = raw->len;
	// Allocate the tok_t object, the label array is allocated only if they
	// are requested by the user.
	tok_t *tok = xmalloc(sizeof(tok_t) + T * sizeof(char **));
	tok->cnts = xmalloc(sizeof(uint32_t) * T);
	tok->lbl = NULL;
	if (lbl == true)
		tok->lbl = xmalloc(sizeof(char *) * T);
	// We now take the raw sequence line by line and split them in list of
	// tokens. To reduce memory fragmentation, the raw line is copied and
	// his reference is kept by the first tokens, next tokens are pointer to
	// this copy.
	for (uint32_t t = 0; t < T; t++) {
		// Get a copy of the raw line skiping leading space characters
		const char *src = raw->lines[t];
		while (isspace(*src))
			src++;
		char *line = xstrdup(src);
		// Split it in tokens
		char *toks[strlen(line) / 2 + 1];
		uint32_t cnt = 0;
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
	// Convert the tok_t to a seq_t
	seq_t *seq = NULL;
	if (rdr->npats == 0)
		seq = rdr_rawtok2seq(rdr, tok);
	else
		seq = rdr_pattok2seq(rdr, tok);
	// Before returning the sequence, we have to free the tok_t
	for (uint32_t t = 0; t < T; t++) {
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

/* rdr_readseq:
 *   Simple wrapper around rdr_readraw and rdr_raw2seq to directly read a
 *   sequence as a seq_t object from file. This take care of all the process
 *   and correctly free temporary data. If lbl is true the sequence is assumed
 *   to be labeled.
 *   Return NULL if end of file occure before anything as been read.
 */
seq_t *rdr_readseq(rdr_t *rdr, FILE *file, bool lbl) {
	raw_t *raw = rdr_readraw(rdr, file);
	if (raw == NULL)
		return NULL;
	seq_t *seq = rdr_raw2seq(rdr, raw, lbl);
	rdr_freeraw(raw);
	return seq;
}

/* rdr_readdat:
 *   Read a full dataset at once and return it as a dat_t object. This function
 *   take and interpret his parameters like the single sequence reading
 *   function.
 */
dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl) {
	// Prepare dataset
	uint32_t size = 1000;
	dat_t *dat = xmalloc(sizeof(dat_t));
	dat->nseq = 0;
	dat->mlen = 0;
	dat->lbl = lbl;
	dat->seq = xmalloc(sizeof(seq_t *) * size);
	// Load sequences
	while (!feof(file)) {
		// Read the next sequence
		seq_t *seq = rdr_readseq(rdr, file, lbl);
		if (seq == NULL)
			break;
		// Grow the buffer if needed
		if (dat->nseq == size) {
			size *= 1.4;
			dat->seq = xrealloc(dat->seq, sizeof(seq_t *) * size);
		}
		// And store the sequence
		dat->seq[dat->nseq++] = seq;
		dat->mlen = max(dat->mlen, seq->len);
		if (dat->nseq % 1000 == 0)
			info("%7"PRIu32" sequences loaded\n", dat->nseq);
	}
	// If no sequence readed, cleanup and repport
	if (dat->nseq == 0) {
		free(dat->seq);
		free(dat);
		return NULL;
	}
	// Adjust the dataset size and return
	if (size > dat->nseq)
		dat->seq = xrealloc(dat->seq, sizeof(seq_t *) * dat->nseq);
	return dat;
}

/* rdr_load:
 *   Read from the given file a reader saved previously with rdr_save. The given
 *   reader must be empty, comming fresh from rdr_new. Be carefull that this
 *   function performs almost no checks on the input data, so if you modify the
 *   reader and make a mistake, it will probably result in a crash.
 */
void rdr_load(rdr_t *rdr, FILE *file) {
	const char *err = "broken file, invalid reader format";
	int autouni = rdr->autouni;
	if (fscanf(file, "#rdr#%"PRIu32"/%"PRIu32"/%d\n",
			&rdr->npats, &rdr->ntoks, &autouni) != 3) {
		// This for compatibility with previous file format
		if (fscanf(file, "#rdr#%"PRIu32"/%"PRIu32"\n",
				&rdr->npats, &rdr->ntoks) != 2)
			fatal(err);
	}
	rdr->autouni = autouni;
	rdr->nuni = rdr->nbi = 0;
	if (rdr->npats != 0) {
		rdr->pats = xmalloc(sizeof(pat_t *) * rdr->npats);
		for (uint32_t p = 0; p < rdr->npats; p++) {
			char *pat = ns_readstr(file);
			rdr->pats[p] = pat_comp(pat);
			switch (tolower(pat[0])) {
				case 'u': rdr->nuni++; break;
				case 'b': rdr->nbi++;  break;
				case '*': rdr->nuni++;
				          rdr->nbi++;  break;
			}
		}
	}
	qrk_load(rdr->lbl, file);
	qrk_load(rdr->obs, file);
}

/* rdr_save:
 *   Save the reader to the given file so it can be loaded back. The save format
 *   is plain text and portable accros computers.
 */
void rdr_save(const rdr_t *rdr, FILE *file) {
	if (fprintf(file, "#rdr#%"PRIu32"/%"PRIu32"/%d\n",
			rdr->npats, rdr->ntoks, rdr->autouni) < 0)
		pfatal("cannot write to file");
	for (uint32_t p = 0; p < rdr->npats; p++)
		ns_writestr(file, rdr->pats[p]->src);
	qrk_save(rdr->lbl, file);
	qrk_save(rdr->obs, file);
}

