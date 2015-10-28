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

#define LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) \
  (DESCRIPTOR).flags = (unsigned short)(FLAGS); (DESCRIPTOR).m = (unsigned char)(M); (DESCRIPTOR).n = (unsigned char)(N); (DESCRIPTOR).k = (unsigned char)(K); \
  (DESCRIPTOR).lda   = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).alpha = (signed char)(0 != (FALPHA) ? (0 == ((FLAGS) & LIBXSMM_XGEMM_FLAG_ALPHA_F) ? (FALPHA) : 0) : 0); \
  (DESCRIPTOR).beta  = (signed char)(0 != (FBETA)  ? (0 == ((FLAGS) & LIBXSMM_XGEMM_FLAG_BETA_F)  ? (FBETA)  : 0) : 0)

#define LIBXSMM_XGEMM_DESCRIPTOR_TYPE(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) \
  libxsmm_xgemm_descriptor DESCRIPTOR; LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, \
    M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA)

#define LIBXSMM_XGEMM_DESCRIPTOR_SIZE ( 3 * sizeof(int) + 3/*mnk*/ + 2/*ab*/ + 1/*p*/ \
                                      + 1 * sizeof(short)/*f*/)

/**
 * Structure storing the xgemm argument description.
 * The binary data layout must be fixed across
 * translation units regardless of the
 * alignment and the padding.
 */
typedef struct libxsmm_xgemm_descriptor {
  /** Leading dimensions are general offsets. */
  unsigned int lda, ldb, ldc;
  /** Matrix extents are limited (8 bit). */
  unsigned char m, n, k;
  /** Integer unless FLAG_*_F is raised. */
  signed char alpha, beta;
  /** Prefetch strategy enumeration (8 bit). */
  unsigned char prefetch;
  /** Collection of various flags (8 bit). */
  unsigned short flags;
} libxsmm_xgemm_descriptor;

/**
 * Flag enumeration which can be binary ORed
 * into libxsmm_xgemm_descriptor::flags.
 */
typedef enum libxsmm_xgemm_flags {
  LIBXSMM_XGEMM_FLAG_DEFAULT  = 0,
  LIBXSMM_XGEMM_FLAG_F32PREC  = 1,
  LIBXSMM_XGEMM_FLAG_TRANS_A  = 2,
  LIBXSMM_XGEMM_FLAG_TRANS_B  = 4,
  LIBXSMM_XGEMM_FLAG_ALIGN_A  = 8,
  LIBXSMM_XGEMM_FLAG_ALIGN_C  = 16,
  /** Fractional: 1/alpha, or General: 1/0. */
  LIBXSMM_XGEMM_FLAG_ALPHA_F  = 32,
  /** Fractional: 1/beta, or General: 1/0. */
  LIBXSMM_XGEMM_FLAG_BETA_F   = 64
} libxsmm_xgemm_flags;

/**
 * Structure referring to the generated code
 * with some attached information.
 */
typedef struct libxsmm_generated_code_struct {
  void* generated_code;       /** pointer to memory which can contain strings or binary code */
  unsigned int buffer_size;   /** total size if the buffer generated_code */
  unsigned int code_size;     /** size of bytes used in generated_code */
  unsigned int code_type;     /**
                               *  0: generated code contains inline assembly in a C function
                               *    which can be dumped into into a *.c/cc/cpp file
                               *  1: generated code contains assembly which can be
                               *     dumped into an *.s file
                               * >1: generated code contains a function in binary code which can be
                               *     called, when the code is copied into executable memory
                               */
  unsigned int last_error;    /**
                               *  0: no error occured
                               * >0: the occured error code
                               */
} libxsmm_generated_code;

/**
 * Enumeration of the available prefetch schemes.
 */
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

/** function to translate LIBXSMM Generator error codes to error messages */
const char* libxsmm_strerror(unsigned int i_error_code);

#endif /* GENERATOR_EXTERN_TYPEDEFS_H */

