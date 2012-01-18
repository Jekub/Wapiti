/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2012  CNRS
 * All rights reserved.
i *
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pattern.h"
#include "sequence.h"
#include "tools.h"

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
 *               \s : space
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
			case 's': return  isspace(str[0]);
			case 'u': return  isupper(str[0]);
			case 'w': return  isalnum(str[0]);
			case 'A': return !isalpha(str[0]);
			case 'D': return !isdigit(str[0]);
			case 'L': return !islower(str[0]);
			case 'P': return !ispunct(str[0]);
			case 'S': return !isspace(str[0]);
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
static bool rex_matchme(const char *re, const char *str, uint32_t *len) {
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
			const uint32_t save = *len;
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
static int32_t rex_match(const char *re, const char *str, uint32_t *len) {
	// Special case for anchor at start
	if (*re == '^') {
		*len = 0;
		if (rex_matchme(re + 1, str, len))
			return 0;
		return -1;
	}
	// And general case for any position
	int32_t pos = 0;
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

/* pat_comp:
 *   Compile the pattern to a form more suitable to easily apply it on tokens
 *   list during data reading. The given pattern string is interned in the
 *   compiled pattern and will be freed with it, so you don't have to take care
 *   of it and must not modify it after the compilation.
 */
pat_t *pat_comp(char *p) {
	pat_t *pat = NULL;
	// Allocate memory for the compiled pattern, the allocation is based
	// on an over-estimation of the number of required item. As compiled
	// pattern take a neglectible amount of memory, this waste is not
	// important.
	uint32_t mitems = 0;
	for (uint32_t pos = 0; p[pos] != '\0'; pos++)
		if (p[pos] == '%')
			mitems++;
	mitems = mitems * 2 + 1;
	pat = xmalloc(sizeof(pat_t) + sizeof(pat->items[0]) * mitems);
	pat->src = p;
	// Next, we go through the pattern compiling the items as they are
	// found. Commands are parsed and put in a corresponding item, and
	// segment of char not in a command are put in a 's' item.
	uint32_t nitems = 0;
	uint32_t ntoks = 0;
	uint32_t pos = 0;
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
			const char *at = p + pos;
			uint32_t col;
			int32_t off;
			int nch;
			item->absolute = false;
			if (sscanf(at, "[@%"SCNi32",%"SCNu32"%n", &off, &col, &nch) == 2)
				item->absolute = true;
			else if (sscanf(at, "[%"SCNi32",%"SCNu32"%n", &off, &col, &nch) != 2)
				fatal("invalid pattern: %s", p);
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
				const int32_t start = (pos += 2);
				while (p[pos] != '\0') {
					if (p[pos] == '"')
						break;
					if (p[pos] == '\\' && p[pos+1] != '\0')
						pos++;
					pos++;
				}
				if (p[pos] != '"')
					fatal("unended argument: %s", p);
				const int32_t len = pos - start;
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
			const int32_t start = pos;
			while (p[pos] != '\0' && p[pos] != '%')
				pos++;
			const int32_t len = pos - start;
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
char *pat_exec(const pat_t *pat, const tok_t *tok, uint32_t at) {
	static char *bval[] = {"_x-1", "_x-2", "_x-3", "_x-4", "_x-#"};
	static char *eval[] = {"_x+1", "_x+2", "_x+3", "_x+4", "_x+#"};
	const uint32_t T = tok->len;
	// Prepare the buffer who will hold the result
	uint32_t size = 16, pos = 0;
	char *buffer = xmalloc(sizeof(char) * size);
	// And loop over the compiled items
	for (uint32_t it = 0; it < pat->nitems; it++) {
		const pat_item_t *item = &(pat->items[it]);
		char *value = NULL;
		uint32_t len = 0;
		// First, if needed, we retrieve the token at the referenced
		// position in the sequence. We store it in value and let the
		// command handler do what it need with it.
		if (item->type != 's') {
			int pos = item->offset;
			if (item->absolute) {
				if (item->offset < 0)
					pos += T;
				else
					pos--;
			} else {
				pos += at;
			}
			uint32_t col = item->column;
			if (pos < 0)
				value = bval[min(-pos - 1, 4)];
			else if (pos >= (int32_t)T)
				value = eval[min( pos - (int32_t)T, 4)];
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
			int32_t pos = rex_match(item->value, value, &len);
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
			for (uint32_t i = pos; i < pos + len; i++)
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
void pat_free(pat_t *pat) {
	for (uint32_t it = 0; it < pat->nitems; it++)
		free(pat->items[it].value);
	free(pat->src);
	free(pat);
}


