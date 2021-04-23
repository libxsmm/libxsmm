/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evanelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_MATELTWISE_SSE_AVX_AVX512_H
#define GENERATOR_MATELTWISE_SSE_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
int libxsmm_generator_meltw_get_rbp_relative_offset( libxsmm_meltw_stack_var stack_var );

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_getval_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meltw_stack_var            stack_var,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setval_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meltw_stack_var            stack_var,
                                                unsigned int                        i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                              const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                              libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                              libxsmm_mateltwise_kernel_config*    i_micro_kernel_config) ;

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
    const libxsmm_meltw_descriptor*     i_mateltwise_desc,
    const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config );


LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_m_loop( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*      i_kernel_config,
                                              const unsigned int                        i_gp_reg_m_loop );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_m_loop( libxsmm_generated_code*                       io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*          i_kernel_config,
                                              const unsigned int                            i_gp_reg_m_loop,
                                              const unsigned int                            i_m );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_n_loop( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*      i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_loop( libxsmm_generated_code*                       io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*          i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_n );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_n_dyn_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop,
                                              int                                       skip_init );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_dyn_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_gp_reg_n_bound );
LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_initialize_avx_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                       i_mask_reg,
    const unsigned int                       i_mask_count );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                       i_gp_reg_tmp,
    const unsigned int                       i_mask_reg,
    const unsigned int                       i_mask_count,
    const unsigned int                       i_precision);

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_update_micro_kernel_config_vectorlength( libxsmm_generated_code*           io_generated_code,
                                                                           libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*   i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( libxsmm_generated_code*           io_generated_code,
                                                                       libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*   i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_sse_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meltw_descriptor* i_mateltw_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_create_reduce_desc_from_unary_desc(libxsmm_descriptor_blob *blob, const libxsmm_meltw_descriptor *in_desc, libxsmm_meltw_descriptor **out_desc);

#endif /* GENERATOR_MATELTWISE_SSE_AVX_AVX512_H */

