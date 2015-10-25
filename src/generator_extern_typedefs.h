/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_EXTERN_TYPEDEFS_H
#define GENERATOR_EXTERN_TYPEDEFS_H

#define LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, PRECISION, PREFETCH, ALIGN_LD, ALIGN_ST, TRANSA, TRANSB, ALPHA, BETA, M, N, K, LDA, LDB, LDC) \
  (DESCRIPTOR); (DESCRIPTOR).single_precision = (PRECISION); (DESCRIPTOR).prefetch = (PREFETCH); \
  (DESCRIPTOR).trans_a = (TRANSA); (DESCRIPTOR).trans_b = (TRANSB); \
  (DESCRIPTOR).alpha = (ALPHA); (DESCRIPTOR).beta = (BETA); \
  (DESCRIPTOR).m = (M); (DESCRIPTOR).n = (N); (DESCRIPTOR).k = (K); \
  (DESCRIPTOR).lda = (LDA); (DESCRIPTOR).ldb = (LDB); (DESCRIPTOR).ldc = (LDC); \
  (DESCRIPTOR).aligned_a = (ALIGN_LD); (DESCRIPTOR).aligned_c = (ALIGN_ST)

#define LIBXSMM_XGEMM_DESCRIPTOR_SIZE (12 * sizeof(int) + 2)

/* Enumerate the available prefetch schemes. */
typedef enum libxsmm_prefetch_type {
  /* No prefetching and no prefetch fn. signature. */
  LIBXSMM_PREFETCH_NONE               = 0,
  /* Only function prefetch signature. */
  LIBXSMM_PREFETCH_SIGNATURE          = 1,
  /* Prefetch PA using accesses to A. */
  LIBXSMM_PREFETCH_AL2                = 2,
  /* Prefetch PA (aggressive). */
  LIBXSMM_PREFETCH_AL2_JPST           = 4,
  /* Prefetch PB using accesses to C. */
  LIBXSMM_PREFETCH_BL2_VIA_C          = 8,
  /* Prefetch A ahead. */
  LIBXSMM_PREFETCH_AL2_AHEAD          = 16,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C       = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST  = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_JPST,
  LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD = LIBXSMM_PREFETCH_BL2_VIA_C | LIBXSMM_PREFETCH_AL2_AHEAD
} libxsmm_prefetch_type;

/* Structure storing the xgemm argument description.
   The binary data layout must be fixed across
   translation units regardless of the
   alignment and the padding. */
typedef struct libxsmm_xgemm_descriptor_struct {
  unsigned int m, n, k, lda, ldb, ldc;
  unsigned int aligned_a, aligned_c;
  unsigned int single_precision;
  unsigned int prefetch;
  int alpha, beta;
  char trans_a;
  char trans_b;
} libxsmm_xgemm_descriptor;

/* struct for storing the generated code
   and some information attached to it */
typedef struct libxsmm_generated_code_struct {
  void* generated_code;              /* pointer to memory which can contain strings or binary code */
  unsigned int buffer_size;          /* total size if the buffer generated_code */
  unsigned int code_size;            /* size of bytes used in generated_code */
  unsigned int code_type;            /*   0: generated code contains inline assembly in a C
                                             function which can be dumped into into a *.c/cc/cpp file
                                          1: generated code contains assembly which can be
                                             dumped into a *.s file
                                         >1: generated code contains a function in binary code which can be
                                             called, when the buffer is copied to executable memory */
  unsigned int last_error;           /* 0    no error occured
                                        > 0  the occured error code */
} libxsmm_generated_code;

/* function to translate LIBXSMM Generator error codes
   to error messages */
const char* libxsmm_strerror( const unsigned int      i_error_code );

#endif /* GENERATOR_EXTERN_TYPEDEFS_H */

