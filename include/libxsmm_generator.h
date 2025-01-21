/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_GENERATOR_H
#define LIBXSMM_GENERATOR_H

#include "libxsmm_typedefs.h"

#define LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) ( \
  (LIBXSMM_FEQ(1, ALPHA) /*|| LIBXSMM_FEQ(-1, ALPHA)*/) && \
  (LIBXSMM_FEQ(1, BETA) || LIBXSMM_FEQ(0, BETA)))

LIBXSMM_API libxsmm_gemm_shape libxsmm_create_gemm_shape( const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k,
                                                          const libxsmm_blasint lda, const libxsmm_blasint ldb, const libxsmm_blasint ldc,
                                                          const libxsmm_datatype a_in_type, const libxsmm_datatype b_in_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );
LIBXSMM_API libxsmm_gemm_batch_reduce_config libxsmm_create_gemm_batch_reduce_config( const libxsmm_gemm_batch_reduce_type br_type,
                                                                                      const libxsmm_blasint br_stride_a_hint, const libxsmm_blasint br_stride_b_hint,
                                                                                      const unsigned char br_unroll_hint );
LIBXSMM_API libxsmm_gemm_ext_unary_argops libxsmm_create_gemm_ext_unary_argops( const libxsmm_blasint ldap, const libxsmm_meltw_unary_type ap_unary_type, const libxsmm_bitfield ap_unary_flags, const libxsmm_blasint store_ap,
                                                                                const libxsmm_blasint ldbp, const libxsmm_meltw_unary_type bp_unary_type, const libxsmm_bitfield bp_unary_flags, const libxsmm_blasint store_bp,
                                                                                const libxsmm_blasint ldcp, const libxsmm_meltw_unary_type cp_unary_type, const libxsmm_bitfield cp_unary_flags, const libxsmm_blasint store_cp );
LIBXSMM_API libxsmm_gemm_ext_binary_postops libxsmm_create_gemm_ext_binary_postops( const libxsmm_blasint ldd, const libxsmm_datatype d_in_type, const libxsmm_meltw_binary_type d_binary_type, const libxsmm_bitfield d_binary_flags );

LIBXSMM_API libxsmm_meltw_unary_shape libxsmm_create_meltw_unary_shape( const libxsmm_blasint m, const libxsmm_blasint n,
                                                                        const libxsmm_blasint ldi, const libxsmm_blasint ldo,
                                                                        const libxsmm_datatype in0_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );
LIBXSMM_API libxsmm_meltw_binary_shape libxsmm_create_meltw_binary_shape( const libxsmm_blasint m, const libxsmm_blasint n,
                                                                          const libxsmm_blasint ldi, const libxsmm_blasint ldi2, const libxsmm_blasint ldo,
                                                                          const libxsmm_datatype in0_type, const libxsmm_datatype in1_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );
LIBXSMM_API libxsmm_meltw_ternary_shape libxsmm_create_meltw_ternary_shape( const libxsmm_blasint m, const libxsmm_blasint n,
                                                                            const libxsmm_blasint ldi, const libxsmm_blasint ldi2, const libxsmm_blasint ldi3, const libxsmm_blasint ldo,
                                                                            const libxsmm_datatype in0_type, const libxsmm_datatype in1_type, const libxsmm_datatype in2_type, const libxsmm_datatype out_type, const libxsmm_datatype comp_type );

/** Initialize GEMM descriptor (generic). */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype a_type, libxsmm_datatype b_type, libxsmm_datatype comp_type, libxsmm_datatype c_type,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, int flags, int prefetch);

/** Initialize mateltwise descriptor */
LIBXSMM_API libxsmm_meltw_descriptor* libxsmm_meltw_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype in_type, libxsmm_datatype out_type,
  libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldo, libxsmm_blasint ldi,
  unsigned short flags, unsigned short param, unsigned char operation);
LIBXSMM_API libxsmm_meltw_descriptor* libxsmm_meltw_descriptor_init2(libxsmm_descriptor_blob* blob,
  libxsmm_datatype in0_type, libxsmm_datatype in1_type, libxsmm_datatype in2_type, libxsmm_datatype comp_type, libxsmm_datatype out_type,
  libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldi2, libxsmm_blasint ldi3,
  unsigned short flags, unsigned short param, unsigned char operation);

/** Initialize matrix equation as used by low-level routines */
LIBXSMM_API libxsmm_meqn_descriptor* libxsmm_meqn_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype type, libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldo, unsigned int eqn_idx);

/** routines for setting up desciptors using similar API as the JITers */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init_gemm( libxsmm_descriptor_blob* blob, const libxsmm_gemm_shape gemm_shape,
                                                                        const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags );
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init_brgemm( libxsmm_descriptor_blob* blob, const libxsmm_gemm_shape gemm_shape,
                                                                          const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags,
                                                                          const libxsmm_gemm_batch_reduce_config brgemm_config );
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init_brgemm_ext( libxsmm_descriptor_blob* blob, const libxsmm_gemm_shape gemm_shape,
                                                                              const libxsmm_bitfield gemm_flags, const libxsmm_bitfield prefetch_flags,
                                                                              const libxsmm_gemm_batch_reduce_config brgemm_config,
                                                                              const libxsmm_gemm_ext_unary_argops unary_argops,
                                                                              const libxsmm_gemm_ext_binary_postops binary_postops );

/** Structure referring to the generated code with some attached information. */
LIBXSMM_EXTERN_C typedef struct libxsmm_generated_code {
  void* generated_code;       /** pointer to memory which can contain strings or binary code */
  unsigned int buffer_size;   /** total size of the buffer generated_code */
  unsigned int code_size;     /** size in bytes used for generated_code (without constant data) */
  unsigned int code_type;     /**
                               *  0: generated code contains inline assembly in a C function
                               *     which can be dumped into a *.c/cc/cpp file
                               *  1: generated code contains assembly which can be
                               *     dumped into an *.s file
                               * >1: generated code contains a function in binary code which can
                               *     be called, when the code is copied into executable memory
                               */
  unsigned int data_size;     /**
                               * amount of constant data located after the generated code
                               * data_size size is separate/excluded from code_size
                               */
  unsigned int last_error;    /**
                               *  0: no error occurred
                               * >0: error code
                               */
  unsigned int arch;          /* target arch for the current code generation task */
} libxsmm_generated_code;

/** Translate LIBXSMM generator error-codes to error messages */
LIBXSMM_API
const char* libxsmm_strerror(unsigned int i_error_code);

/* TODO: change int based architecture value */
LIBXSMM_API
void libxsmm_generator_gemm_inlineasm(const char*                    i_file_out,
                                      const char*                    i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                    i_arch );

/* TODO: change int based architecture value */
LIBXSMM_API
void libxsmm_generator_gemm_directasm(const char*                    i_file_out,
                                      const char*                    i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                    i_arch );

LIBXSMM_API
void libxsmm_generator_gemm_kernel(libxsmm_generated_code*        io_generated_code,
                                   const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API
void libxsmm_generator_gemm_reference_kernel(libxsmm_generated_code*        io_generated_code,
                                             const libxsmm_gemm_descriptor* i_xgemm_desc );

/* TODO: change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm(const char*                    i_file_out,
                              const char*                    i_routine_name,
                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                              const char*                    i_arch,
                              const char*                    i_file_in,
                              const int                      i_is_csr);

/* TODO: change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csc_kernel(libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                         const char*                    i_arch,
                                         const unsigned int*            i_row_idx,
                                         const unsigned int*            i_column_idx,
                                         const double*                  i_values);

/* TODO: change int based architecture value */
LIBXSMM_API
void libxsmm_generator_spgemm_csr_kernel(libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                         const char*                    i_arch,
                                         const unsigned int*            i_row_idx,
                                         const unsigned int*            i_column_idx,
                                         const double*                  i_values);

LIBXSMM_API
void libxsmm_generator_spgemm_csr_reg_kernel(libxsmm_generated_code*        io_generated_code,
                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
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
void libxsmm_generator_packed_spgemm_bcsc_kernel( libxsmm_generated_code*        io_generated_code,
                                                  const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                  const unsigned int             i_packed_width,
                                                  const unsigned int             i_bk,
                                                  const unsigned int             i_bn );
LIBXSMM_API
void libxsmm_generator_packed_gemm_ac_rm( libxsmm_generated_code*         io_generated_code,
                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                          const unsigned int              i_packed_width );

LIBXSMM_API
void libxsmm_generator_packed_gemm_bc_rm( libxsmm_generated_code*         io_generated_code,
                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                          const unsigned int              i_packed_width );

LIBXSMM_API
void libxsmm_generator_packed_gemm( libxsmm_generated_code*         io_generated_code,
                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                    const unsigned int              i_packed_width );

LIBXSMM_API
void libxsmm_generator_mateltwise_kernel( libxsmm_generated_code*            io_generated_code,
                                          const libxsmm_meltw_descriptor*    i_mateltw_desc );

LIBXSMM_API
void libxsmm_generator_mateltwise_reference_kernel( libxsmm_generated_code*            io_generated_code,
                                          const libxsmm_meltw_descriptor*    i_mateltw_desc );

LIBXSMM_API
void libxsmm_generator_matequation_kernel( libxsmm_generated_code*        io_generated_code,
                                           const libxsmm_meqn_descriptor* i_mateqn_desc );

LIBXSMM_API
void libxsmm_generator_matequation_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                           const libxsmm_meqn_descriptor* i_mateqn_desc );

/** Used for system/user specific locking (I/O). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_stdio_handle);
/** Initialization counter that can be used to check whether the library is initialized (!=0) or not (==0). */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_ninit);
/** Target architecture (libxsmm_get_target_archid, libxsmm_set_target_archid). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_target_archid);
/** Verbosity level (0: quiet, 1: errors, 2: warnings, 3: info, neg.: all/dump). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_verbosity);
/** Security-enhanced environment. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_se);

#endif /*LIBXSMM_GENERATOR_H*/
