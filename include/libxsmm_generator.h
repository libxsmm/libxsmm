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
#ifndef LIBXSMM_GENERATOR_H
#define LIBXSMM_GENERATOR_H

#include "libxsmm_typedefs.h"

#define LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, VECTOR_WIDTH, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) { \
  (DESCRIPTOR).m = (unsigned int)(M); (DESCRIPTOR).n = (unsigned int)(N); (DESCRIPTOR).k = (unsigned int)(K); \
  (DESCRIPTOR).lda = (unsigned int)(0 == (LIBXSMM_GEMM_FLAG_ALIGN_A & (FLAGS)) \
    ? (0 == (LDA) ? (DESCRIPTOR).m : LIBXSMM_MAX((unsigned int)(LDA), (DESCRIPTOR).m)) \
    : LIBXSMM_ALIGN_VALUE(0 == (LDA) ? (DESCRIPTOR).m : LIBXSMM_MAX((unsigned int)(LDA), (DESCRIPTOR).m), \
       0 == (LIBXSMM_GEMM_FLAG_F32PREC & (FLAGS)) ? sizeof(double) : sizeof(float), VECTOR_WIDTH)); \
  (DESCRIPTOR).ldb = (unsigned int)LIBXSMM_MAX((unsigned int)(LDB), (unsigned int)(K)); \
  (DESCRIPTOR).ldc = (unsigned int)(0 == (LIBXSMM_GEMM_FLAG_ALIGN_C & (FLAGS)) \
    ? (0 == (LDC) ? (DESCRIPTOR).m : LIBXSMM_MAX((unsigned int)(LDC), (DESCRIPTOR).m)) \
    : LIBXSMM_ALIGN_VALUE(0 == (LDC) ? (DESCRIPTOR).m : LIBXSMM_MAX((unsigned int)(LDC), (DESCRIPTOR).m), \
       0 == (LIBXSMM_GEMM_FLAG_F32PREC & (FLAGS)) ? sizeof(double) : sizeof(float), VECTOR_WIDTH)); \
  (DESCRIPTOR).flags = (unsigned char)(FLAGS); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).alpha = (signed char)((0 < (ALPHA) || 0 > (ALPHA)) ? (0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_ALPHA_F) ? (ALPHA) : 0) : 0); \
  (DESCRIPTOR).beta  = (signed char)((0 < (BETA)  || 0 > (BETA))  ? (0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_BETA_F)  ? (BETA)  : 0) : 0); \
}

#define LIBXSMM_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, VECTOR_WIDTH, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  libxsmm_gemm_descriptor DESCRIPTOR; LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, VECTOR_WIDTH, \
    FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/** The libxsmm_gemm_descriptor structure must be ordered by the size of the members (packed). */
#define LIBXSMM_GEMM_DESCRIPTOR_SIZE (3 * sizeof(unsigned int)  /*LDA,LDB,LDC*/ \
                                    + 3 * sizeof(unsigned int)  /*M,N,K*/       \
                                    + 1 * sizeof(unsigned char) /*flags*/       \
                                    + 2 * sizeof(signed char)   /*alpha,beta*/  \
                                    + 1 * sizeof(unsigned char) /*prefetch*/)

/**
 * Structure storing the gemm argument description.
 * The binary data layout must be fixed across
 * translation units regardless of the
 * alignment and the padding.
 */
typedef struct libxsmm_gemm_descriptor {
  /** Leading dimensions are general offsets. */
  unsigned int lda, ldb, ldc;
  /** Extents of the matrix. */
  unsigned int m, n, k;
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

/** function to translate LIBXSMM Generator error codes to error messages */
const char* libxsmm_strerror(unsigned int i_error_code);

/* @TODO change int based architecture value */
void libxsmm_generator_dense_inlineasm(const char*                     i_file_out,
                                       const char*                     i_routine_name,
                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch );

/* @TODO change int based architecture value */
void libxsmm_generator_dense_directasm(const char*                     i_file_out,
                                       const char*                     i_routine_name,
                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch );

/* @TODO change int based architecture value */
void libxsmm_generator_dense_kernel( libxsmm_generated_code*         io_generated_code,
                                     const libxsmm_gemm_descriptor* i_xgemm_desc,
                                     const char*                     i_arch );

/* @TODO change int based architecture value */
void libxsmm_generator_sparse( const char*                     i_file_out,
                               const char*                     i_routine_name,
                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                               const char*                     i_arch,
                               const char*                     i_csc_file_in );

/* @TODO change int based architecture value */
void libxsmm_generator_sparse_kernel( libxsmm_generated_code*         io_generated_code,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                     i_arch,
                                      const unsigned int*             i_row_idx,
                                      const unsigned int*             i_column_idx,
                                      const double*                   i_values );

#endif /*LIBXSMM_GENERATOR_H*/

