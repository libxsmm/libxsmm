/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#ifndef LIBXSMM_GENERATOR_H
#define LIBXSMM_GENERATOR_H

#include "libxsmm_typedefs.h"
#include "libxsmm_macros.h"

/**
 * Defining LIBXSMM_GENERATOR_AUTOALIGN and enabling emitting aligned loads/stores (instead of unaligned loads/stores)
 * does not provide any benefit on modern architectures (if the addresses are actually aligned).
 */
#if defined(LIBXSMM_GENERATOR_AUTOALIGN)
# define LIBXSMM_GEMM_DESCRIPTOR_AUTOALIGN(VECTOR_WIDTH, FLAGS, LDA, LDC) \
    ((~(0 != LIBXSMM_MOD2((LDA) * (0 == (LIBXSMM_GEMM_FLAG_F32PREC & (FLAGS)) ? sizeof(double) : sizeof(float)), VECTOR_WIDTH) \
      ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0)) && \
     (~(0 != LIBXSMM_MOD2((LDC) * (0 == (LIBXSMM_GEMM_FLAG_F32PREC & (FLAGS)) ? sizeof(double) : sizeof(float)), VECTOR_WIDTH) \
      ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0)) && \
     (FLAGS))
#else
# define LIBXSMM_GEMM_DESCRIPTOR_AUTOALIGN(VECTOR_WIDTH, FLAGS, LDA, LDC) (FLAGS)
#endif

#define LIBXSMM_GEMM_DESCRIPTOR_FIXUP(DESCRIPTOR, FLAGS, ALPHA, BETA) \
  if (0 != ((FLAGS) & (LIBXSMM_GEMM_FLAG_ALPHA_F | LIBXSMM_GEMM_FLAG_BETA_F))) { \
    if (0 != ((FLAGS) & LIBXSMM_GEMM_FLAG_ALPHA_F)) { \
      if (0 < (ALPHA) || 0 > (ALPHA)) { \
        const double dalpha = ALPHA; \
        (DESCRIPTOR).alpha = (signed char)(1.0 / dalpha); \
      } \
    } \
    if (0 != ((FLAGS) & LIBXSMM_GEMM_FLAG_BETA_F)) { \
      if (0 < (BETA) || 0 > (BETA)) { \
        const double dbeta = BETA; \
        (DESCRIPTOR).beta = (signed char)(1.0 / dbeta); \
      } \
    } \
  }

#if defined(LIBXSMM_GENERATOR_BIGDESC)
/* TODO: support libxsmm_blasint in the backend, or make sure to fallback earlier */
# define LIBXSMM_GENERATOR_SIZE_TYPE unsigned int
# define LIBXSMM_GEMM_DESCRIPTOR_SIZE 28 /* LDA,LDB,LDC: 3 * sizeof(LIBXSMM_GENERATOR_BIGDESC)
                                          * M,N,K:       3 * sizeof(LIBXSMM_GENERATOR_BIGDESC)
                                          * flags:       1 * sizeof(unsigned char)
                                          * alpha,beta:  2 * sizeof(signed char)
                                          * prefetch:    1 * sizeof(unsigned char)
                                          */
#else
/* TODO: make sure to fallback earlier if index space is exhaused */
# define LIBXSMM_GENERATOR_SIZE_TYPE unsigned short
# define LIBXSMM_GEMM_DESCRIPTOR_SIZE 16 /* LDA,LDB,LDC: 3 * sizeof(LIBXSMM_GENERATOR_BIGDESC)
                                          * M,N,K:       3 * sizeof(LIBXSMM_GENERATOR_BIGDESC)
                                          * flags:       1 * sizeof(unsigned char)
                                          * alpha,beta:  2 * sizeof(signed char)
                                          * prefetch:    1 * sizeof(unsigned char)
                                          */
#endif

/**
 * Construct a GEMM descriptor after it has been declared. The descriptor flags will sanitized to remove any
 * alignment request which cannot be satisfied (avoids to build an unnecessary code version).
 */
#define LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, VECTOR_WIDTH, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  (DESCRIPTOR).lda = (LIBXSMM_GENERATOR_SIZE_TYPE)(LDA); (DESCRIPTOR).ldb = (LIBXSMM_GENERATOR_SIZE_TYPE)(LDB); \
  (DESCRIPTOR).ldc = (LIBXSMM_GENERATOR_SIZE_TYPE)(LDC); (DESCRIPTOR).m = (LIBXSMM_GENERATOR_SIZE_TYPE)(M); \
  (DESCRIPTOR).n = (LIBXSMM_GENERATOR_SIZE_TYPE)(N); (DESCRIPTOR).k = (LIBXSMM_GENERATOR_SIZE_TYPE)(K); \
  (DESCRIPTOR).flags = (unsigned char)LIBXSMM_GEMM_DESCRIPTOR_AUTOALIGN(VECTOR_WIDTH, FLAGS, LDA, LDC); \
  (DESCRIPTOR).alpha = (signed char)(ALPHA); (DESCRIPTOR).beta = (signed char)(BETA); \
  (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  LIBXSMM_GEMM_DESCRIPTOR_FIXUP(DESCRIPTOR, FLAGS, ALPHA, BETA)

/** Declare and construct a GEMM descriptor. */
#define LIBXSMM_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, VECTOR_WIDTH, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  libxsmm_gemm_descriptor DESCRIPTOR; LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, VECTOR_WIDTH, \
    FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/**
 * Structure storing the GEMM argument description. The binary data layout must be fixed across translation units
 * regardless of the alignment and the padding. This structure must be ordered by the size of the members (packed).
 */
typedef struct libxsmm_gemm_descriptor {
  /** Leading dimensions are general offsets. */
  LIBXSMM_GENERATOR_SIZE_TYPE lda, ldb, ldc;
  /** Extents of the matrix. */
  LIBXSMM_GENERATOR_SIZE_TYPE m, n, k;
  /** Collection of various flags. */
  unsigned char flags;
  /** Integer unless FLAG_*_F is raised. */
  signed char alpha, beta;
  /** Prefetch strategy enumeration. */
  unsigned char prefetch;
} libxsmm_gemm_descriptor;

/** Extended flag set complementing libxsmm_gemm_flags. */
typedef enum libxsmm_gemm_xflags {
  LIBXSMM_GEMM_FLAG_F32PREC = 16,
  /** Fractional: 1/alpha, or General: 1/0. */
  LIBXSMM_GEMM_FLAG_ALPHA_F = 32,
  /** Fractional: 1/beta, or General: 1/0. */
  LIBXSMM_GEMM_FLAG_BETA_F  = 64
} libxsmm_gemm_xflags;

/** Structure referring to the generated code with some attached information. */
typedef struct libxsmm_generated_code {
  void* generated_code;       /** pointer to memory which can contain strings or binary code */
  unsigned int buffer_size;   /** total size if the buffer generated_code */
  unsigned int code_size;     /** size of bytes used in generated_code */
  unsigned int code_type;     /**
                               *  0: generated code contains inline assembly in a C function
                               *     which can be dumped into a *.c/cc/cpp file
                               *  1: generated code contains assembly which can be
                               *     dumped into an *.s file
                               * >1: generated code contains a function in binary code which can be
                               *     called, when the code is copied into executable memory
                               */
  unsigned int last_error;    /**
                               *  0: no error occurred
                               * >0: error code
                               */
} libxsmm_generated_code;

/** function to translate LIBXSMM Generator error codes to error messages */
LIBXSMM_INTERNAL_API
const char* libxsmm_strerror(unsigned int i_error_code);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_inlineasm(const char*                     i_file_out,
                                      const char*                     i_routine_name,
                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                      const char*                     i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_directasm(const char*                     i_file_out,
                                      const char*                     i_routine_name,
                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                      const char*                     i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_kernel( libxsmm_generated_code*         io_generated_code,
                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                    const char*                     i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm( const char*                    i_file_out,
                               const char*                    i_routine_name,
                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                               const char*                    i_arch,
                               const char*                    i_file_in,
                               const int                      i_is_csr );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csc_kernel( libxsmm_generated_code*         io_generated_code,
                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                          const char*                     i_arch,
                                          const unsigned int*             i_row_idx,
                                          const unsigned int*             i_column_idx,
                                          const double*                   i_values );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_kernel( libxsmm_generated_code*         io_generated_code,
                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                          const char*                     i_arch,
                                          const unsigned int*             i_row_idx,
                                          const unsigned int*             i_column_idx,
                                          const double*                   i_values );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_soa_kernel( libxsmm_generated_code*         io_generated_code,
                                              const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                              const char*                     i_arch,
                                              const unsigned int*             i_row_idx,
                                              const unsigned int*             i_column_idx,
                                              const double*                   i_values );

#endif /*LIBXSMM_GENERATOR_H*/

