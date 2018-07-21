/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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

#define LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) ( \
  0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) && \
        (LIBXSMM_FEQ(1, ALPHA) /*|| LIBXSMM_FEQ(-1, ALPHA)*/) && \
        (LIBXSMM_FEQ(1, BETA) || LIBXSMM_FEQ(0, BETA)))

#define LIBXSMM_TRANS_NO_BYPASS(M, N) ( \
  (((unsigned int)(M)) * (N)) <= ((unsigned int)(LIBXSMM_AVG_M) * (LIBXSMM_AVG_N)))


/** Initialize GEMM descriptor as used by low-level routines (type-specific). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_dgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_sgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_wigemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_wsgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, libxsmm_gemm_prefetch_type prefetch);

/** Initialize GEMM descriptor (generic: double-precision alpha/beta). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_dinit(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, double alpha, double beta,
  int flags, libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_dinit2(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, libxsmm_gemm_prefetch_type prefetch);

/** Initialize GEMM descriptor as used by low-level routines (generic). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init2(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, libxsmm_gemm_prefetch_type prefetch);
/** Similar to libxsmm_gemm_descriptor_init2 with optional type-converted alpha/beta (dalpha/dbeta). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init3(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, libxsmm_gemm_prefetch_type prefetch,
  double* dalpha, double* dbeta);

/** Initialize transpose descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_trans_descriptor* libxsmm_trans_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, unsigned int m, unsigned int n, unsigned int ldo);

/** Initialize transpose descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_mcopy_descriptor* libxsmm_mcopy_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, unsigned int m, unsigned int n, unsigned int ldo,
  unsigned int ldi, int flags, int prefetch, const int* unroll);

/** Initialize transpose descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_trsm_descriptor* libxsmm_trsm_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint lda, libxsmm_blasint ldb,
  const void* alpha, char transa, char diag, char side, char uplo, int layout);

/** Structure referring to the generated code with some attached information. */
LIBXSMM_EXTERN_C typedef struct libxsmm_generated_code {
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
LIBXSMM_API
const char* libxsmm_strerror(unsigned int i_error_code);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_gemm_inlineasm(const char*                    i_file_out,
                                      const char*                    i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_gemm_directasm(const char*                    i_file_out,
                                      const char*                    i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_gemm_kernel(libxsmm_generated_code*        io_generated_code,
                                   const libxsmm_gemm_descriptor* i_xgemm_desc,
                                   const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm(const char*                    i_file_out,
                              const char*                    i_routine_name,
                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                              const char*                    i_arch,
                              const char*                    i_file_in,
                              const int                      i_is_csr);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csc_kernel(libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                         const char*                    i_arch,
                                         const unsigned int*            i_row_idx,
                                         const unsigned int*            i_column_idx,
                                         const double*                  i_values);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csr_kernel(libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                         const char*                    i_arch,
                                         const unsigned int*            i_row_idx,
                                         const unsigned int*            i_column_idx,
                                         const double*                  i_values);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csr_reg_kernel(libxsmm_generated_code*        io_generated_code,
                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
                                             const char*                    i_arch,
                                             const unsigned int*            i_row_idx,
                                             const unsigned int*            i_column_idx,
                                             const double*                  i_values);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csr_soa_kernel(libxsmm_generated_code*        io_generated_code,
                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
                                             const char*                    i_arch,
                                             const unsigned int*            i_row_idx,
                                             const unsigned int*            i_column_idx,
                                             const void*                    i_values);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csc_soa_kernel( libxsmm_generated_code*        io_generated_code,
                                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                                              const char*                    i_arch,
                                              const unsigned int*            i_row_idx,
                                              const unsigned int*            i_column_idx,
                                              const void*                    i_values );

/* @TODO change int based architecture value */
LIBXSMM_API void libxsmm_generator_gemm_rm_ac_soa( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const char*                     i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API void libxsmm_generator_gemm_rm_bc_soa( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const char*                     i_arch );

LIBXSMM_API
void libxsmm_generator_trsm_kernel ( libxsmm_generated_code*        io_generated_code,
                                     const libxsmm_trsm_descriptor* i_packed_trsm_desc,
                                     const char*                    i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_matcopy_kernel( libxsmm_generated_code*            io_generated_code,
                                       const libxsmm_mcopy_descriptor*    i_matcopy_desc,
                                       const char*                        i_arch );

LIBXSMM_API
void libxsmm_generator_transpose_kernel( libxsmm_generated_code*          io_generated_code,
                                         const libxsmm_trans_descriptor*  i_trans_desc,
                                         const char*                      i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_forward_inlineasm(const char*                       i_file_out,
                                                     const char*                       i_routine_name,
                                                     const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                     const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_forward_directasm(const char*                       i_file_out,
                                                     const char*                       i_routine_name,
                                                     const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                     const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_forward_kernel(libxsmm_generated_code*           io_generated_code,
                                                  const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                  const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_backward_kernel(libxsmm_generated_code*           io_generated_code,
                                                   const libxsmm_convolution_backward_descriptor* i_conv_desc,
                                                   const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_weight_update_kernel(libxsmm_generated_code*           io_generated_code,
                                                        const libxsmm_convolution_weight_update_descriptor* i_conv_desc,
                                                        const char*                       i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_winograd_weight_update_kernel(libxsmm_generated_code*                        io_generated_code,
                                                                 const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                 const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_winograd_weight_update_inlineasm(const char*                                    i_file_out,
                                                                    const char*                                    i_routine_name,
                                                                    const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                    const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_winograd_weight_update_directasm(const char*                                    i_file_out,
                                                                    const char*                                    i_routine_name,
                                                                    const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                    const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_winograd_forward_kernel(libxsmm_generated_code*                        io_generated_code,
                                                           const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                           const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_winograd_forward_inlineasm(const char*                                    i_file_out,
                                                              const char*                                    i_routine_name,
                                                              const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                              const char*                                    i_arch);

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_convolution_winograd_forward_directasm(const char*                                    i_file_out,
                                                              const char*                                    i_routine_name,
                                                              const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                              const char*                                    i_arch);

/** Verbosity level (0: quiet, 1: errors, 2: warnings, 3: info, neg.: all/dump). */
LIBXSMM_APIVAR(int libxsmm_verbosity);

#endif /*LIBXSMM_GENERATOR_H*/

