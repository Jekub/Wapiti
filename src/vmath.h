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

#ifndef vmath_h
#define vmath_h

#include <stdint.h>

const char *xvm_mode(void);

double *xvm_new(uint64_t N);
void    xvm_free(double x[]);

void xvm_neg(double r[], const double x[], uint64_t N);
void xvm_sub(double r[], const double x[], const double y[], uint64_t N);
void xvm_scale(double r[], const double x[], double a, uint64_t N);
double xvm_unit(double r[], const double x[], uint64_t N);

double xvm_norm(const double x[], uint64_t N);
double xvm_dot(const double x[], const double y[], uint64_t N);

void xvm_axpy(double r[], double a, const double x[], const double y[],
		uint64_t N);

void xvm_expma(double r[], const double x[], double a, uint64_t N);

#endif

