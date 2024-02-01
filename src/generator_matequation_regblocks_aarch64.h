/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas  (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATEQUATION_REGBLOCKS_AARCH64_H
#define GENERATOR_MATEQUATION_REGBLOCKS_AARCH64_H

#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_copy_opargs_aarch64(libxsmm_generated_code*        io_generated_code,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    libxsmm_meqn_elem             *cur_node,
    unsigned int                        *oparg_id,
    libxsmm_meqn_tmp_info         *oparg_info,
    unsigned int                        input_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_copy_input_args_aarch64(libxsmm_generated_code*        io_generated_code,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    libxsmm_meqn_elem             *cur_node,
    unsigned int                        *arg_id,
    libxsmm_meqn_arg           *arg_info,
    unsigned int                        input_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(libxsmm_generated_code*        io_generated_code,
    const libxsmm_meqn_descriptor       *i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    unsigned int                        i_adjust_instr,
    unsigned int                        i_adjust_amount,
    unsigned int                        i_adjust_type,
    libxsmm_meqn_tmp_info         *oparg_info);

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_adjust_args_addr_aarch64(libxsmm_generated_code*        io_generated_code,
    const libxsmm_meqn_descriptor       *i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    unsigned int                        i_adjust_instr,
    unsigned int                        i_adjust_amount,
    unsigned int                        i_adjust_type,
    libxsmm_meqn_arg           *arg_info);

LIBXSMM_API_INTERN
void libxsmm_configure_mateqn_microkernel_loops_aarch64( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_matequation_kernel_config       *i_micro_kernel_config,
                                                 libxsmm_matrix_eqn                      *i_eqn,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            i_use_m_input_masking,
                                                 unsigned int*                           i_m_trips,
                                                 unsigned int*                           i_n_trips,
                                                 unsigned int*                           i_m_unroll_factor,
                                                 unsigned int*                           i_n_unroll_factor,
                                                 unsigned int*                           i_m_assm_trips,
                                                 unsigned int*                           i_n_assm_trips);

LIBXSMM_API_INTERN
void libxsmm_meqn_setup_input_output_masks_aarch64( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_tmp_reg,
                                                 unsigned int                            i_m,
                                                 unsigned int*                           i_use_m_input_masking,
                                                 unsigned int*                           i_mask_reg_in,
                                                 unsigned int*                           i_use_m_output_masking,
                                                 unsigned int*                           i_mask_reg_out);

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_arg_id,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg,
                                                 unsigned int                            i_skip_dtype_cvt );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_dump_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_ld,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg,
                                                 libxsmm_datatype                        i_regblock_dtype,
                                                 unsigned int                            i_gp_reg_out );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_unpackstore_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_reduce_to_scalar_output_aarch64( libxsmm_generated_code*          io_generated_code,
                                                             libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                             libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                             const libxsmm_meqn_descriptor*          i_meqn_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_unary_op_2d_reg_block_aarch64( libxsmm_generated_code*     io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_unary_type                i_op_type,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 libxsmm_datatype                        i_dtype );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_binary_op_2d_reg_block_aarch64( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_binary_type               i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 libxsmm_datatype                        i_dtype );

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_ternary_op_2d_reg_block_aarch64( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_ternary_type              i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 libxsmm_datatype                        i_dtype );


LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_2d_microkernel_aarch64( libxsmm_generated_code*            io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 libxsmm_matrix_eqn                      *i_eqn,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            skip_n_loop_reg_cleanup );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_configure_M_N_blocking_aarch64( libxsmm_generated_code*                    io_generated_code,
    libxsmm_matequation_kernel_config* i_micro_kernel_config,
    libxsmm_matrix_eqn *i_eqn,
    unsigned int m,
    unsigned int n,
    unsigned int vlen,
    unsigned int *m_blocking,
    unsigned int *n_blocking);


LIBXSMM_API_INTERN
void libxsmm_generator_configure_equation_aarch64_vlens( libxsmm_generated_code*                    io_generated_code,
    libxsmm_matequation_kernel_config* i_micro_kernel_config,
    libxsmm_matrix_eqn *eqn);


LIBXSMM_API_INTERN
unsigned int libxsmm_generator_matequation_regblocks_unary_op_req_zmms_aarch64(libxsmm_generated_code*                    io_generated_code,  libxsmm_meltw_unary_type u_type);


LIBXSMM_API_INTERN
unsigned int libxsmm_generator_matequation_regblocks_binary_op_req_zmms_aarch64(libxsmm_generated_code*                    io_generated_code, libxsmm_meltw_binary_type b_type);

LIBXSMM_API_INTERN
void libxsmm_adjust_required_zmms_aarch64( libxsmm_generated_code*                    io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config,
    libxsmm_meltw_unary_type u_type,
    libxsmm_meltw_binary_type b_type,
    unsigned int pool_id );

LIBXSMM_API_INTERN
void libxsmm_mark_reserved_zmms_aarch64( libxsmm_generated_code*                    io_generated_code,  libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_meqn_elem *cur_node );

LIBXSMM_API_INTERN
void libxsmm_configure_reserved_zmms_and_masks_aarch64(libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_matrix_eqn                      *eqn );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_register_block_aarch64_kernel( libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_loop_label_tracker*             io_loop_label_tracker,
    libxsmm_matrix_eqn*                     eqn );

#endif /*GENERATOR_MATEQUATION_REGBLOCKS_AARCH64_H*/
