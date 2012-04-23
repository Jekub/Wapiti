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
#ifndef wapiti_h
#define wapiti_h

#define VERSION "1.4.0"

/* XVM_ANSI:
 *   By uncomenting the following define, you can force wapiti to not use SSE2
 *   even if available.
 */
//#define XVM_ANSI

/* MTH_ANSI:
 *   By uncomenting the following define, you can disable the use of POSIX
 *   threads in the multi-threading part of Wapiti, for non-POSIX systems.
 */
//#define MTH_ANSI

/* ATM_ANSI:
 *   By uncomenting the following define, you can disable the use of atomic
 *   operation to update the gradient. This imply that multi-threaded gradient
 *   computation will require more memory but is more portable.
 */
//#define ATM_ANSI

/* Without multi-threading we disable atomic updates as they are not needed and
 * can only decrease performances in this case.
 */
#ifdef MTH_ANSI
#define ATM_ANSI
#endif

#endif

