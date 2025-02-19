/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_VXRS_MICROKERNEL_H
#define GENERATOR_GEMM_VXRS_MICROKERNEL_H

#include "generator_s390x_instructions.h"

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_microkernel( libxsmm_generated_code        *io_generated_code,
                                         const libxsmm_gemm_descriptor *i_xgemm_desc,
                                         libxsmm_s390x_reg             *io_reg_tracker,
                                         libxsmm_loop_label_tracker    *io_loop_labels,
                                         const libxsmm_s390x_blocking  *i_blocking,
                                         unsigned int                   i_a,
                                         unsigned int                   i_b,
                                         unsigned int                   i_c );

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_fma_b_splat( libxsmm_generated_code *io_generated_code,
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
void libxsmm_generator_vxrs_block_load_mult( libxsmm_generated_code        *io_generated_code,
                                             libxsmm_s390x_reg             *io_reg_tracker,
                                             const libxsmm_datatype         i_datatype,
                                             const libxsmm_datatype         i_comptype,
                                             unsigned int                   i_a,
                                             unsigned int                   i_m,
                                             unsigned int                   i_n,
                                             unsigned int                   i_lda,
                                             unsigned int                  *io_t,
                                             unsigned int                   i_ldt );

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_load_bcast( libxsmm_generated_code        *io_generated_code,
                                              libxsmm_s390x_reg             *io_reg_tracker,
                                              const libxsmm_datatype         i_datatype,
                                              const libxsmm_datatype         i_comptype,
                                              unsigned int                   i_a,
                                              unsigned int                   i_m,
                                              unsigned int                   i_n,
                                              unsigned int                   i_lda,
                                              unsigned int                  *io_t,
                                              unsigned int                   i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_store_mult( libxsmm_generated_code        *io_generated_code,
                                              libxsmm_s390x_reg             *io_reg_tracker,
                                              const libxsmm_datatype         i_datatype,
                                              const libxsmm_datatype         i_comptype,
                                              unsigned int                   i_a,
                                              unsigned int                   i_m,
                                              unsigned int                   i_n,
                                              unsigned int                   i_lda,
                                              unsigned int                  *io_t,
                                              unsigned int                   i_ldt );

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_prefetch( libxsmm_generated_code     *io_generated_code,
                                            libxsmm_s390x_reg          *io_reg_tracker,
                                            const libxsmm_datatype      i_datatype,
                                            libxsmm_s390x_prefetch_type i_type,
                                            unsigned int                i_a,
                                            unsigned int                i_idx,
                                            unsigned int                i_m,
                                            unsigned int                i_n,
                                            unsigned int                i_lda );

#endif
