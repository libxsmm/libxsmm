/******************************************************************************
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_PPC64LE_VSX_MICROKERNEL_H
#define GENERATOR_PPC64LE_VSX_MICROKERNEL_H

#include "generator_ppc64le_instructions.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load_trans( libxsmm_generated_code * io_generated_code,
                                               libxsmm_datatype const   i_datatype,
                                               libxsmm_datatype const   i_comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * io_reg_tracker,
                                               unsigned int           * i_loaded_regs,
                                               unsigned int const       i_ptr_gpr,
                                               unsigned int const       i_n_rows,
                                               unsigned int const       i_n_cols,
                                               unsigned int const       i_stride );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load( libxsmm_generated_code * io_generated_code,
                                         libxsmm_datatype const   i_datatype,
                                         libxsmm_datatype const   i_comptype, /* currently unsuded */
                                         libxsmm_ppc64le_reg    * io_reg_tracker,
                                         unsigned int           * i_loaded_regs,
                                         unsigned int const       i_ptr_gpr,
                                         unsigned int const       i_n_rows,
                                         unsigned int const       i_n_cols,
                                         unsigned int const       i_stride );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_store( libxsmm_generated_code * io_generated_code,
                                          libxsmm_datatype const   i_datatype,
                                          libxsmm_datatype const   i_comptype, /* currently unsuded */
                                          libxsmm_ppc64le_reg    * io_reg_tracker,
                                          unsigned int           * i_loaded_regs,
                                          unsigned int const       i_ptr_gpr,
                                          unsigned int const       i_n_rows,
                                          unsigned int const       i_n_cols,
                                          unsigned int const       i_stride );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load_bcast( libxsmm_generated_code * io_generated_code,
                                               libxsmm_datatype const   i_datatype,
                                               libxsmm_datatype const   i_comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * io_reg_tracker,
                                               unsigned int           * i_loaded_regs,
                                               unsigned int const       i_ptr_gpr,
                                               unsigned int const       i_n_rows,
                                               unsigned int const       i_n_cols,
                                               unsigned int const       i_stride );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_bcast( libxsmm_generated_code * io_generated_code,
                                              libxsmm_datatype const   i_datatype,
                                              unsigned int const       i_m,
                                              unsigned int const       i_n,
                                              unsigned int const       i_k,
                                              unsigned int           * i_a,
                                              unsigned int const       i_lda,
                                              unsigned int           * i_b,
                                              unsigned int const       i_ldb,
                                              unsigned int const       i_beta,
                                              unsigned int           * io_c,
                                              unsigned int const       i_ldc );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code        * io_generated_code,
                                        libxsmm_gemm_descriptor const * i_xgemm_desc,
                                        libxsmm_ppc64le_reg           * io_reg_tracker,
                                        libxsmm_loop_label_tracker    * io_loop_labels,
                                        unsigned int                  * i_blocking,
                                        unsigned char const             i_a_ptr_gpr,
                                        unsigned char const             i_b_ptr_gpr,
                                        unsigned char const             i_c_ptr_gpr );

#endif
