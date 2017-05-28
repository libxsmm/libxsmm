/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_GENERATOR_H
#define LIBXSMM_GENERATOR_H

#include "libxsmm_typedefs.h"
#include "libxsmm_macros.h"

/** Check if M, N, K, or LDx fits into the descriptor. */
#if (0 != LIBXSMM_ILP64)
# define LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) (((unsigned int)(-1)) >= (M) && ((unsigned int)(-1)) >= (N) && ((unsigned int)(-1)) >= (K))
#else /* always fits */
# define LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXSMM_FRONTEND_H) /* assert available */
# define LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) assert(LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

/**
 * Construct a GEMM descriptor after it has been declared. The descriptor flags will sanitized to remove any
 * alignment request which cannot be satisfied (avoids to build an unnecessary code version).
 */
#define LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); LIBXSMM_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).flags = (unsigned short)(FLAGS); (DESCRIPTOR).prefetch = (unsigned short)(PREFETCH); \
  (DESCRIPTOR).alpha = (signed char)(ALPHA); (DESCRIPTOR).beta = (signed char)(BETA); \
  (DESCRIPTOR).datatype = (unsigned char)(DATA_TYPE); (DESCRIPTOR).iflags = 0

/** Declare and construct a GEMM descriptor. */
#define LIBXSMM_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  libxsmm_gemm_descriptor DESCRIPTOR = { 0 }; LIBXSMM_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, \
    FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

#define LIBXSMM_GEMM_DESCRIPTOR_SIZE 32

/**
 * Structure, which stores the argument description of GEMM routines.
 * This structure must be ordered by the size of the members (packed).
 * The size of the structure matches LIBXSMM_GEMM_DESCRIPTOR_SIZE.
 */
typedef struct libxsmm_gemm_descriptor {
  /** Leading dimensions are general offsets. */
  unsigned int lda, ldb, ldc;
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Flag set. */
  unsigned short flags;
  /** Prefetch strategy enumeration. */
  unsigned short prefetch;
  /** Integer value. */
  signed char alpha, beta;
  /** Denotes the data-type*/
  unsigned char datatype;
  /** INTERNAL (last member!) */
  unsigned char iflags;
} libxsmm_gemm_descriptor;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_matcopy_flags {
  /** If set, then use zero matrix as source */
  LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE = 1
} libxsmm_matcopy_flags;

/** Structure storing the matcopy argument description. */
typedef struct libxsmm_matcopy_descriptor { /* 20 Byte */
  /** M, N, and LDx I/O */
  unsigned int m, n, ldi, ldo;
  /** Size of an individual data element */
  unsigned char typesize;
  /** Defines the level of unrolling in the copy */
  unsigned char unroll_level;
  /** @TODO fix this, non-zero for prefetch */
  unsigned char prefetch;
  /** Collection of various flags. */
  unsigned char flags;
} libxsmm_matcopy_descriptor;

/** Structure storing the transpose argument description. */
typedef struct libxsmm_transpose_descriptor { /* 13 Byte */
  /** M, N, and LDO */
  unsigned int m, n, ldo;
  /** Size of an individual data element */
  unsigned char typesize;
} libxsmm_transpose_descriptor;

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
void libxsmm_generator_gemm_inlineasm(const char*                    i_file_out,
                                      const char*                    i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_directasm(const char*                    i_file_out,
                                      const char*                    i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_gemm_kernel(libxsmm_generated_code*        io_generated_code,
                                   const libxsmm_gemm_descriptor* i_xgemm_desc,
                                   const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm(const char*                    i_file_out,
                              const char*                    i_routine_name,
                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                              const char*                    i_arch,
                              const char*                    i_file_in,
                              const int                      i_is_csr);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csc_kernel(libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                         const char*                    i_arch,
                                         const unsigned int*            i_row_idx,
                                         const unsigned int*            i_column_idx,
                                         const double*                  i_values);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_kernel(libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                         const char*                    i_arch,
                                         const unsigned int*            i_row_idx,
                                         const unsigned int*            i_column_idx,
                                         const double*                  i_values);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_reg_kernel(libxsmm_generated_code*        io_generated_code,
                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
                                             const char*                    i_arch,
                                             const unsigned int*            i_row_idx,
                                             const unsigned int*            i_column_idx,
                                             const double*                  i_values);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_spgemm_csr_soa_kernel(libxsmm_generated_code*        io_generated_code,
                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
                                             const char*                    i_arch,
                                             const unsigned int*            i_row_idx,
                                             const unsigned int*            i_column_idx,
                                             const void*                    i_values);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_matcopy_kernel( libxsmm_generated_code*                      io_generated_code,
                                       const libxsmm_matcopy_descriptor*            i_matcopy_desc,
                                       const char*                                  i_arch );

LIBXSMM_INTERNAL_API
void libxsmm_generator_transpose_kernel( libxsmm_generated_code*                        io_generated_code,
                                         const libxsmm_transpose_descriptor*            i_trans_desc,
                                         const char*                                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_forward_inlineasm(const char*                       i_file_out,
                                                     const char*                       i_routine_name,
                                                     const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                     const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_forward_directasm(const char*                       i_file_out,
                                                     const char*                       i_routine_name,
                                                     const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                     const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_forward_kernel(libxsmm_generated_code*           io_generated_code,
                                                  const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                  const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_backward_kernel(libxsmm_generated_code*           io_generated_code,
                                                   const libxsmm_convolution_backward_descriptor* i_conv_desc,
                                                   const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_weight_update_kernel(libxsmm_generated_code*           io_generated_code,
                                                        const libxsmm_convolution_weight_update_descriptor* i_conv_desc,
                                                        const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_winograd_weight_update_kernel(libxsmm_generated_code*                        io_generated_code,
                                                                 const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                 const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_winograd_weight_update_inlineasm(const char*                                    i_file_out,
                                                                    const char*                                    i_routine_name,
                                                                    const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                    const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_winograd_weight_update_directasm(const char*                                    i_file_out,
                                                                    const char*                                    i_routine_name,
                                                                    const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                    const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_winograd_forward_kernel(libxsmm_generated_code*                        io_generated_code,
                                                           const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                           const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_winograd_forward_inlineasm(const char*                                    i_file_out,
                                                              const char*                                    i_routine_name,
                                                              const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                              const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_INTERNAL_API
void libxsmm_generator_convolution_winograd_forward_directasm(const char*                                    i_file_out,
                                                              const char*                                    i_routine_name,
                                                              const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                              const char*                                    i_arch);

#endif /*LIBXSMM_GENERATOR_H*/

