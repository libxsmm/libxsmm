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
void libxsmm_generator_gemm_vsx_block_load_vsr( libxsmm_generated_code *io_generated_code,
                                                libxsmm_ppc64le_reg    *io_reg_tracker,
                                                libxsmm_datatype const  i_datatype,
                                                libxsmm_datatype const  i_comptype, /* currently unsuded */
                                                unsigned int            i_a,
                                                unsigned int            i_m,
                                                unsigned int            i_n,
                                                unsigned int            i_lda,
                                                unsigned int           *io_t,
                                                unsigned int            i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_block_store_vsr( libxsmm_generated_code *io_generated_code,
                                                 libxsmm_ppc64le_reg    *io_reg_tracker,
                                                 libxsmm_datatype const  i_datatype,
                                                 libxsmm_datatype const  i_comptype, /* currently unsuded */
                                                 unsigned int            i_a,
                                                 unsigned int            i_m,
                                                 unsigned int            i_n,
                                                 unsigned int            i_lda,
                                                 unsigned int           *io_t,
                                                 unsigned int            i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_micro_load_vsr_splat( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      libxsmm_datatype const  i_datatype,
                                                      libxsmm_datatype const  i_comptype, /* currently unsuded */
                                                      unsigned int            i_a,
                                                      unsigned int            i_m,
                                                      unsigned int            i_n,
                                                      unsigned int            i_lda,
                                                      unsigned int           *io_t,
                                                      unsigned int            i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_block_load_vsr_splat( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      libxsmm_datatype const  i_datatype,
                                                      libxsmm_datatype const  i_comptype, /* currently unsuded */
                                                      unsigned int            i_a,
                                                      unsigned int            i_m,
                                                      unsigned int            i_n,
                                                      unsigned int            i_lda,
                                                      unsigned int           *io_t,
                                                      unsigned int            i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_alu( libxsmm_generated_code *io_generated_code,
                                libxsmm_datatype const  i_datatype,
                                unsigned int            i_a,
                                unsigned int            i_b,
                                unsigned int            i_c,
                                int                     i_alpha,
                                int                     i_beta );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_splat( libxsmm_generated_code *io_generated_code,
                                              libxsmm_datatype const  i_datatype,
                                              unsigned int            i_m,
                                              unsigned int            i_n,
                                              unsigned int            i_k,
                                              unsigned int           *i_a,
                                              unsigned int            i_lda,
                                              unsigned int           *i_b,
                                              unsigned int            i_ldb,
                                              unsigned int            i_beta,
                                              unsigned int           *io_c,
                                              unsigned int            i_ldc );


LIBXSMM_API_INTERN
void libxsmm_generator_br_vsx_microkernel( libxsmm_generated_code        *io_generated_code,
                                           libxsmm_gemm_descriptor const *i_xgemm_desc,
                                           libxsmm_ppc64le_blocking      *i_blocking,
                                           libxsmm_ppc64le_reg           *io_reg_tracker,
                                           libxsmm_loop_label_tracker    *io_loop_labels,
                                           unsigned int                   i_a,
                                           unsigned int                   i_b,
                                           unsigned int                   i_c,
                                           unsigned int                  *i_c_reg );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code        *io_generated_code,
                                        libxsmm_gemm_descriptor const *i_xgemm_desc,
                                        libxsmm_ppc64le_blocking      *i_blocking,
                                        libxsmm_ppc64le_reg           *io_reg_tracker,
                                        libxsmm_loop_label_tracker    *io_loop_labels,
                                        unsigned int                   i_a,
                                        unsigned int                   i_b,
                                        unsigned int                   i_c );

#endif
