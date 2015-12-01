/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#ifndef LIBXSMM_TYPEDEFS_H
#define LIBXSMM_TYPEDEFS_H

#include "libxsmm_macros.h"


/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_gemm_flags {
  /** Transpose matrix A. */
  LIBXSMM_GEMM_FLAG_TRANS_A = 1,
  /** Transpose matrix B. */
  LIBXSMM_GEMM_FLAG_TRANS_B = 2,
  /** Generate aligned load instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_A = 4,
  /** Aligned load/store instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_C = 8
} libxsmm_gemm_flags;

/** Enumeration of the available prefetch strategies. */
typedef enum libxsmm_prefetch_type {
  /** No prefetching and no prefetch fn. signature. */
  LIBXSMM_PREFETCH_NONE               = 0,
  /** Only function prefetch signature. */
  LIBXSMM_PREFETCH_SIGNATURE          = 1,
  /** Prefetch PA using accesses to A. */
  LIBXSMM_PREFETCH_AL2                = 2,
  /** Prefetch PA (aggressive). */
  LIBXSMM_PREFETCH_AL2_JPST           = 4,
  /** Prefetch PB using accesses to C. */
  LIBXSMM_PREFETCH_BL2_VIA_C          = 8,
  /** Prefetch A ahead. */
  LIBXSMM_PREFETCH_AL2_AHEAD          = 16,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C       = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST  = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_JPST,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_AHEAD
} libxsmm_prefetch_type;

#endif /*LIBXSMM_TYPEDEFS_H*/

