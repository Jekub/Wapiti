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

#ifndef ioline_h
#define ioline_h

#include <stdio.h>

typedef char *(*gets_cb_t)(void *);
typedef int   (*print_cb_t)(void *, char *, ...);

/* iol_t:
 *   Represents a class to do IO in a line by line basis.
 */
typedef struct iol_s iol_t;
struct iol_s {
    gets_cb_t gets_cb;   // callback to get a line from in
    print_cb_t print_cb; // callback to print a line to out

    void *in;          // state passed to the gets callback
    void *out;         // state passed to the puts callback
};    

iol_t *iol_new(FILE *in, FILE *out);
iol_t *iol_new2(gets_cb_t gets_cb, void *in, print_cb_t print_cb, void *out);
void iol_free(iol_t *iol);

#endif
