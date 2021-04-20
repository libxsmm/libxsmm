/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_GENERATOR_H
#define LIBXSMM_GENERATOR_H

#include "libxsmm_typedefs.h"

#define LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) ( \
  0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A)) && \
        (LIBXSMM_FEQ(1, ALPHA) /*|| LIBXSMM_FEQ(-1, ALPHA)*/) && \
        (LIBXSMM_FEQ(1, BETA) || LIBXSMM_FEQ(0, BETA)))


/** Initialize GEMM descriptor as used by low-level routines (type-specific). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_dgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_sgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_wigemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_bigemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_bbgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_bsgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_bgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, int prefetch);

/** Initialize GEMM descriptor (generic: double-precision alpha/beta). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_dinit(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, double alpha, double beta,
  int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_dinit2(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, int prefetch);

/** Initialize GEMM descriptor as used by low-level routines (generic). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, int prefetch);
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init2(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, int prefetch);
/** Similar to libxsmm_gemm_descriptor_init2 with optional type-converted alpha/beta (dalpha/dbeta). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init3(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, int prefetch, double* dalpha, double* dbeta);

/** Initialize transpose descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_meltw_descriptor* libxsmm_meltw_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype in_type, libxsmm_datatype out_type,
  libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldo, libxsmm_blasint ldi,
  unsigned short flags, unsigned char param, unsigned char operation);
LIBXSMM_API libxsmm_meltw_descriptor* libxsmm_meltw_descriptor_init2(libxsmm_descriptor_blob* blob,
  libxsmm_datatype in_type, libxsmm_datatype in2_type, libxsmm_datatype out_type, libxsmm_datatype out2_type,
  libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldo, libxsmm_blasint ldi, libxsmm_blasint ldi2, libxsmm_blasint ldi3,
  unsigned short flags, unsigned char param, unsigned char operation);

/** Initialize matrix equation as used by low-level routines */
LIBXSMM_API libxsmm_meqn_descriptor* libxsmm_meqn_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype type, libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldo, unsigned int eqn_idx);


/** Initialize packed trsm descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_trsm_descriptor* libxsmm_trsm_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint lda, libxsmm_blasint ldb,
  const void* alpha, char transa, char diag, char side, char uplo, int layout);

/** Initialize packed trmm descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_trmm_descriptor* libxsmm_trmm_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint lda, libxsmm_blasint ldb,
  const void* alpha, char transa, char diag, char side, char uplo, int layout);

/** Initialize packed getrf descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_getrf_descriptor* libxsmm_getrf_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint lda, int layout);

/** Initialize packed pgemm descriptor as used by low-level routines. */
LIBXSMM_API libxsmm_pgemm_descriptor* libxsmm_pgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const void* alpha, char transa, char transb, int layout);

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
  unsigned int arch;          /* target arch for the current code generation task */
  unsigned int sf_size;       /* offset of RSP to the beginning of the stack frame
                               * we track this value to have RBP availbale for general compute
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

LIBXSMM_API
void libxsmm_generator_gemm_kernel(libxsmm_generated_code*        io_generated_code,
                                   const libxsmm_gemm_descriptor* i_xgemm_desc );

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

LIBXSMM_API
void libxsmm_generator_packed_spgemm_csr_kernel( libxsmm_generated_code*        io_generated_code,
                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                 const unsigned int*            i_row_idx,
                                                 const unsigned int*            i_column_idx,
                                                 const void*                    i_values,
                                                 const unsigned int             i_packed_width );

LIBXSMM_API
void libxsmm_generator_packed_spgemm_csc_kernel( libxsmm_generated_code*        io_generated_code,
                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                 const unsigned int*            i_row_idx,
                                                 const unsigned int*            i_column_idx,
                                                 const void*                    i_values,
                                                 const unsigned int             i_packed_width );

LIBXSMM_API
void libxsmm_generator_packed_gemm_ac_rm( libxsmm_generated_code*         io_generated_code,
                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                          const unsigned int              i_packed_width );

LIBXSMM_API
void libxsmm_generator_packed_gemm_bc_rm( libxsmm_generated_code*         io_generated_code,
                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                          const unsigned int              i_packed_width );

LIBXSMM_API
void libxsmm_generator_pgemm_kernel( libxsmm_generated_code*          io_generated_code,
                                     const libxsmm_pgemm_descriptor*  i_packed_pgemm_desc,
                                     int                              i_arch, ... );

LIBXSMM_API
void libxsmm_generator_getrf_kernel( libxsmm_generated_code*          io_generated_code,
                                     const libxsmm_getrf_descriptor*  i_packed_pgemm_desc,
                                     int                              i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_trmm_kernel( libxsmm_generated_code*         io_generated_code,
                                    const libxsmm_trmm_descriptor*  i_packed_trmm_desc,
                                    const char*                     i_arch );

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_trsm_kernel( libxsmm_generated_code*         io_generated_code,
                                    const libxsmm_trsm_descriptor*  i_packed_trsm_desc,
                                    const char*                     i_arch );

LIBXSMM_API
void libxsmm_generator_mateltwise_kernel( libxsmm_generated_code*            io_generated_code,
                                          const libxsmm_meltw_descriptor*    i_mateltw_desc );

LIBXSMM_API
void libxsmm_generator_matequation_kernel( libxsmm_generated_code*        io_generated_code,
                                           const libxsmm_meqn_descriptor* i_mateqn_desc );

/** Initialization counter that can be used to check whether the library is initialized (!=0) or not (==0). */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_ninit);
/** Target architecture (libxsmm_get_target_archid, libxsmm_set_target_archid). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_target_archid);
/** Verbosity level (0: quiet, 1: errors, 2: warnings, 3: info, neg.: all/dump). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_verbosity);
/** Security-enhanced environment. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_se);

#endif /*LIBXSMM_GENERATOR_H*/

