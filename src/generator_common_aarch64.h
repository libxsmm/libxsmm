/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_COMMON_AARCH64_H
#define GENERATOR_COMMON_AARCH64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_trips );

LIBXSMM_API_INTERN
void libxsmm_generator_loop_footer_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_loop_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_aarch64( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_gp_reg_addr,
                                                const unsigned int      i_gp_reg_scratch_a,
                                                const unsigned int      i_gp_reg_scratch_b,
                                                const unsigned int      i_vec_length,
                                                const unsigned int      i_vec_reg_count,
                                                const unsigned int      i_m_blocking,
                                                const unsigned int      i_n_blocking,
                                                const unsigned int      i_ld,
                                                const unsigned int      i_zero );

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_aarch64( libxsmm_generated_code* io_generated_code,
                                                 const unsigned int      i_gp_reg_addr,
                                                 const unsigned int      i_gp_reg_scratch_a,
                                                 const unsigned int      i_gp_reg_scratch_b,
                                                 const unsigned int      i_vec_length,
                                                 const unsigned int      i_vec_reg_count,
                                                 const unsigned int      i_m_blocking,
                                                 const unsigned int      i_n_blocking,
                                                 const unsigned int      i_ld );

#if 0
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                       libxsmm_micro_kernel_config*       i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config );
#endif

#endif /* GENERATOR_COMMON_AARCH64_H */

