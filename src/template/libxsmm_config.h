/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_CONFIG_H
#define LIBXSMM_CONFIG_H

/** Parameters the library and static kernels were built for. */
#define LIBXSMM_ALIGNMENT $ALIGNMENT
#define LIBXSMM_PREFETCH $PREFETCH
#define LIBXSMM_MAX_MNK $MAX_MNK
#define LIBXSMM_MAX_M $MAX_M
#define LIBXSMM_MAX_N $MAX_N
#define LIBXSMM_MAX_K $MAX_K
#define LIBXSMM_AVG_M $AVG_M
#define LIBXSMM_AVG_N $AVG_N
#define LIBXSMM_AVG_K $AVG_K
#define LIBXSMM_FLAGS $FLAGS
#define LIBXSMM_ILP64 $ILP64
#define LIBXSMM_ALPHA $ALPHA
#define LIBXSMM_BETA $BETA
#define LIBXSMM_WRAP $WRAP
#define LIBXSMM_SYNC $SYNC
#define LIBXSMM_JIT $JIT
#define LIBXSMM_BIG $BIG
$LIBXSMM_OFFLOAD_BUILD

#endif /*LIBXSMM_CONFIG_H*/

