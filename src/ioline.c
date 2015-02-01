/*
 *      Wapiti - A linear-chain CRF tool
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ioline.h"
#include "tools.h"

static char *iol_gets(void *in);
static int   iol_puts(void *out, char *msg, ...);
static int   iol_print(void *out, char *msg, ...);

iol_t *iol_new(FILE *in, FILE *out) {
    iol_t *iol = xmalloc(sizeof(iol_t));
    iol->gets_cb  = iol_gets,
    iol->in       = in;
    iol->puts_cb  = iol_puts;
    iol->print_cb = iol_print;
    iol->out      = out;
    return iol;
}

iol_t *iol_new2(gets_cb_t gets_cb, void *in, print_cb_t print_cb, void *out) {
    iol_t *iol = xmalloc(sizeof(iol_t));
    iol->gets_cb  = gets_cb;
    iol->in       = in;
    iol->print_cb = print_cb;
    iol->out      = out;
    return iol;
}

void iol_free(iol_t *iol) {
    free(iol);
}


/* iol_gets:
 *   Read an input line from <in>. The line can be of any size limited only by
 *   available memory, a buffer large enough is allocated and returned. The
 *   caller is responsible to free it. If the input is exhausted, NULL is returned.
 */
static char *iol_gets(void *in) {
        FILE *file = (FILE*)in;
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

/* iol_puts:
 *   Puts a line to <out>.  A new line character is appended to the end of
 *   the line.
 */
static int iol_puts(void *out, char *msg, ...) {
        FILE *file = (FILE*)out;
	va_list args;
	va_start(args, msg);
	int rc = vfprintf(file, msg, args);
        fprintf(file, "\n");
	va_end(args);
        return rc;
}

/* iol_puts:
 *   Print a line to <out>.
 */
static int iol_print(void *out, char *msg, ...) {
        FILE *file = (FILE*)out;
	va_list args;
	va_start(args, msg);
	int rc = vfprintf(file, msg, args);
	va_end(args);
        return rc;
}

