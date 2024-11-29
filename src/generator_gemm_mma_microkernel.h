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

#ifndef GENERATOR_PPC64LE_MMA_MICROKERNEL_H
#define GENERATOR_PPC64LE_MMA_MICROKERNEL_H

#include "generator_ppc64le_instructions.h"
#include "generator_gemm_vsx_microkernel.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_f64_b_reg_alloc( libxsmm_generated_code *io_generated_code,
                                                    libxsmm_ppc64le_reg    *io_reg_tracker,
                                                    unsigned int const      i_n_rows,
                                                    unsigned int const      i_n_cols,
                                                    unsigned int           *o_reg,
                                                    unsigned int const      i_ld );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_micro_load_trans( libxsmm_generated_code *io_generated_code,
                                                  libxsmm_ppc64le_reg    *io_reg_tracker,
                                                  libxsmm_datatype const  i_datatype,
                                                  libxsmm_datatype const  i_comptype,
                                                  unsigned int const      i_a,
                                                  unsigned int const      i_m,
                                                  unsigned int const      i_n,
                                                  unsigned int const      i_lda,
                                                  unsigned int           *io_t,
                                                  unsigned int const      i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_trans( libxsmm_generated_code *io_generated_code,
                                                  libxsmm_ppc64le_reg    *io_reg_tracker,
                                                  libxsmm_datatype const  i_datatype,
                                                  libxsmm_datatype const  i_comptype,
                                                  unsigned int const      i_a,
                                                  unsigned int const      i_m,
                                                  unsigned int const      i_n,
                                                  unsigned int const      i_lda,
                                                  unsigned int           *io_t,
                                                  unsigned int const      i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_acc( libxsmm_generated_code *io_generated_code,
                                                libxsmm_ppc64le_reg    *io_reg_tracker,
                                                libxsmm_datatype const  i_datatype,
                                                libxsmm_datatype const  i_comptype,
                                                unsigned int const      i_a,
                                                unsigned int const      i_m,
                                                unsigned int const      i_n,
                                                unsigned int const      i_lda,
                                                unsigned int           *io_t,
                                                unsigned int const      i_ldt );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_store_acc( libxsmm_generated_code *io_generated_code,
                                                 libxsmm_ppc64le_reg    *io_reg_tracker,
                                                 libxsmm_datatype const  i_datatype,
                                                 libxsmm_datatype const  i_comptype,
                                                 unsigned int const      i_a,
                                                 unsigned int const      i_m,
                                                 unsigned int const      i_n,
                                                 unsigned int const      i_lda,
                                                 unsigned int           *io_t,
                                                 unsigned int const      i_ldt );




LIBXSMM_API_INTERN
void libxsmm_generator_mma_micro_ger( libxsmm_generated_code *io_generated_code,
                                      libxsmm_datatype const  i_comptype,
                                      unsigned int const      i_m,
                                      unsigned int const      i_n,
                                      unsigned int const      i_k,
                                      unsigned int           *i_a,
                                      unsigned int const      i_lda,
                                      unsigned int           *i_b,
                                      unsigned int const      io_c );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_micro_ger_f32( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      io_c );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_micro_ger_f64( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      io_c );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger( libxsmm_generated_code *io_generated_code,
                                      libxsmm_datatype const  i_comptype,
                                      unsigned int const      i_m,
                                      unsigned int const      i_n,
                                      unsigned int const      i_k,
                                      unsigned int           *i_a,
                                      unsigned int const      i_lda,
                                      unsigned int           *i_b,
                                      unsigned int const      i_ldb,
                                      unsigned int const      i_beta,
                                      unsigned int           *io_c,
                                      unsigned int const      i_ldc );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f32( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      i_ldb,
                                          unsigned int const      i_beta,
                                          unsigned int           *io_c,
                                          unsigned int const      i_ldc );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f64( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      i_ldb,
                                          unsigned int const      i_beta,
                                          unsigned int           *io_c,
                                          unsigned int const      i_ldc );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_microkernel( libxsmm_generated_code        *io_generated_code,
                                        libxsmm_gemm_descriptor const *i_xgemm_desc,
                                        libxsmm_ppc64le_blocking      *i_blocking,
                                        libxsmm_ppc64le_reg           *io_reg_tracker,
                                        libxsmm_loop_label_tracker    *io_loop_labels,
                                        unsigned int                  *i_acc,
                                        unsigned char const            i_a,
                                        unsigned char const            i_b,
                                        unsigned char const            i_c );


LIBXSMM_API_INTERN
void libxsmm_generator_mma_microkernel_sched( libxsmm_generated_code         *io_generated_code,
                                              libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                              libxsmm_ppc64le_blocking       *i_blocking,
                                              libxsmm_ppc64le_reg            *io_reg_tracker,
                                              libxsmm_loop_label_tracker     *io_loop_labels,
                                              libxsmm_ppc64le_node          **i_schedule,
                                              unsigned int                   *i_acc,
                                              unsigned char const             i_a,
                                              unsigned char const             i_b,
                                              unsigned char const             i_c );

#endif
