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
#ifndef wapiti_h
#define wapiti_h

#define VERSION "0.9.18"

/* The following two defined allow you to control the precision used for
 * floating point computation and the use of vector operations. Both of them
 * will have an impact on the numerical precision of the computations.
 *
 * On x86 platform, by default, computation are done using double precision and
 * use SSE. This mean that the real precision of computation is 64bit.
 *
 * If you switch off SSE  optimisation and force your compiler to use the FP
 * stack, the precision will go up to 80bit for internal computation. If you
 * switch to single precision without SSE, you will get 32bit precision with
 * internal computation done in 80bit. And with single precision with SSE, you
 * will get only 32bit precision with only the IEEE garantee.
 */
//#define WAP_PREC_SGL
#define WAP_USE_SSE

#ifndef WAP_PREC_SGL
  #define WAP_PREC_DBL
#endif

#ifdef WAP_PREC_SGL
typedef float real;
#else
typedef double real;
#endif

#endif

