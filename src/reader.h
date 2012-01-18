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

#ifndef reader_h
#define reader_h

#include <stdbool.h>
#include <stdio.h>

#include "wapiti.h"
#include "pattern.h"
#include "quark.h"
#include "sequence.h"

/* rdr_t:
 *   The reader object who hold all informations needed to parse the input file:
 *   the patterns and quark for labels and observations. We keep separate count
 *   for unigrams and bigrams pattern for simpler allocation of sequences. We
 *   also store the expected number of column in the input data to check that
 *   pattern are appliables.
 */
typedef struct rdr_s rdr_t;
struct rdr_s {
	bool       autouni;    //      Automatically add 'u' prefix
	uint32_t   npats;      //  P   Total number of patterns
	uint32_t   nuni, nbi;  //      Number of unigram and bigram patterns
	uint32_t   ntoks;      //      Expected number of tokens in input
	pat_t    **pats;       // [P]  List of precompiled patterns
	qrk_t     *lbl;        //      Labels database
	qrk_t     *obs;        //      Observation database
};

rdr_t *rdr_new(bool autouni);
void rdr_free(rdr_t *rdr);
void rdr_freeraw(raw_t *raw);
void rdr_freeseq(seq_t *seq);
void rdr_freedat(dat_t *dat);

void rdr_loadpat(rdr_t *rdr, FILE *file);
raw_t *rdr_readraw(rdr_t *rdr, FILE *file);
seq_t *rdr_raw2seq(rdr_t *rdr, const raw_t *raw, bool lbl);
seq_t *rdr_readseq(rdr_t *rdr, FILE *file, bool lbl);
dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl);

void rdr_load(rdr_t *rdr, FILE *file);
void rdr_save(const rdr_t *rdr, FILE *file);

#endif

