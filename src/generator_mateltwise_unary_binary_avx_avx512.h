/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evanelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_UNARY_BINARY_AVX_AVX512_H
#define GENERATOR_MATELTWISE_UNARY_BINARY_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_configure_unary_kernel_vregs_masks( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 libxsmm_datatype                        i_compute_dtype,
                                                 unsigned int                            op,
                                                 unsigned int                            flags,
                                                 unsigned int                            i_gp_reg_tmp,
                                                 const unsigned int                      i_gp_reg_aux0,
                                                 const unsigned int                      i_gp_reg_aux1);

LIBXSMM_API_INTERN
void libxsmm_finalize_unary_kernel_vregs_masks( libxsmm_generated_code*                 io_generated_code,
                                                libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                unsigned int                            op,
                                                unsigned int                            flags,
                                                unsigned int                            i_gp_reg_tmp,
                                                const unsigned int                      i_gp_reg_aux0,
                                                const unsigned int                      i_gp_reg_aux1);

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg,
                                                 unsigned int                            i_adjust_instr,
                                                 unsigned int                            m_microkernel,
                                                 unsigned int                            n_microkernel,
                                                 unsigned int                            i_loop_type );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg,
                                                 unsigned int                            i_adjust_instr,
                                                 unsigned int                            i_adjust_param,
                                                 unsigned int                            i_loop_type );

LIBXSMM_API_INTERN
void libxsmm_generator_configure_avx512_vlens(const libxsmm_meltw_descriptor* i_mateltwise_desc, libxsmm_mateltwise_kernel_config* i_micro_kernel_config);

LIBXSMM_API_INTERN
void libxsmm_generator_configure_M_N_blocking(unsigned int m, unsigned int n, unsigned int vlen, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int available_vregs);

LIBXSMM_API_INTERN
void libxsmm_generator_configure_loop_order(const libxsmm_meltw_descriptor* i_mateltwise_desc, unsigned int *loop_order, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int *out_blocking, unsigned int *inner_blocking, unsigned int *out_bound, unsigned int *inner_bound);

LIBXSMM_API_INTERN
void libxsmm_load_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_store_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_op( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_relu( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_relu_inv( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_dropout( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_dropout_inv( libxsmm_generated_code*                 io_generated_code,
                                                     libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                     unsigned int                            i_vlen,
                                                     unsigned int                            i_start_vreg,
                                                     unsigned int                            i_m_blocking,
                                                     unsigned int                            i_n_blocking,
                                                     unsigned int                            i_mask_last_m_chunk,
                                                     unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_binary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_ternary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_binary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_setup_input_output_masks( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_tmp_reg,
                                                 unsigned int                            i_m,
                                                 unsigned int*                           i_use_m_input_masking,
                                                 unsigned int*                           i_mask_reg_in,
                                                 unsigned int*                           i_use_m_output_masking,
                                                 unsigned int*                           i_mask_reg_out);

LIBXSMM_API_INTERN
void libxsmm_configure_microkernel_loops( libxsmm_generated_code*                        io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            i_use_m_input_masking,
                                                 unsigned int*                           i_m_trips,
                                                 unsigned int*                           i_n_trips,
                                                 unsigned int*                           i_m_unroll_factor,
                                                 unsigned int*                           i_n_unroll_factor,
                                                 unsigned int*                           i_m_assm_trips,
                                                 unsigned int*                           i_n_assm_trips,
                                                 unsigned int*                           i_out_loop_trips,
                                                 unsigned int*                           i_inner_loop_trips,
                                                 unsigned int*                           i_out_loop_bound,
                                                 unsigned int*                           i_inner_loop_bound,
                                                 unsigned int*                           i_out_loop_reg,
                                                 unsigned int*                           i_inner_loop_reg,
                                                 unsigned int*                           i_out_unroll_factor,
                                                 unsigned int*                           i_inner_unroll_factor);

LIBXSMM_API_INTERN
void libxsmm_configure_kernel_vregs_masks( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg_tmp,
                                                 const unsigned int                      i_gp_reg_aux0,
                                                 const unsigned int                      i_gp_reg_aux1);

LIBXSMM_API_INTERN
void libxsmm_finalize_kernel_vregs_masks( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*        i_mateltwise_desc,
                                                 unsigned int                           i_gp_reg_tmp,
                                                 const unsigned int                     i_gp_reg_aux0,
                                                 const unsigned int                     i_gp_reg_aux1);

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_2d_microkernel( libxsmm_generated_code*                     io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n);

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_avx512_microkernel( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_UNARY_AVX_AVX512_H */

