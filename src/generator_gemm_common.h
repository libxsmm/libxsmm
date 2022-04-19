/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_COMMON_H
#define GENERATOR_GEMM_COMMON_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vnni_store_C_from_scratch( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_relu_to_vreg( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 zero_vreg,
    const unsigned int                 inout_vreg,
    const unsigned int                 store_bitmask,
    const unsigned int                 gpr_bitmask,
    const unsigned int                 store_bitmask_offset,
    const unsigned int                 is_32_bit_relu,
    const unsigned int                 aux_gpr,
    const unsigned int                 aux_vreg);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_sigmoid_to_vreg_from_scratch( libxsmm_generated_code*             io_generated_code,
    libxsmm_micro_kernel_config*       i_micro_kernel_config_mod,
    const unsigned int                 scratch_gpr,
    const unsigned int                 in_vreg,
    const unsigned int                 out_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_restore_2D_regblock_from_scratch( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 scratch_gpr,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_2D_regblock_to_scratch( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 scratch_gpr,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_dump_2D_block_and_prepare_sigmoid_fusion( libxsmm_generated_code*             io_generated_code,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking,
    const unsigned int                 scratch_gpr,
    const unsigned int                 aux_gpr);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_prepare_relu_fusion( libxsmm_generated_code*             io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 zero_vreg,
    const unsigned int                 store_bitmask,
    const unsigned int                 bitmask_gpr,
    const unsigned int                 aux_gpr);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_cleanup_relu_fusion( libxsmm_generated_code*             io_generated_code,
    const unsigned int                 store_bitmask,
    const unsigned int                 bitmask_gpr,
    const unsigned int                 aux_gpr);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_cleanup_sigmoid_fusion( libxsmm_generated_code*             io_generated_code,
    const unsigned int                 scratch_gpr,
    const unsigned int                 aux_gpr );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_colbias_to_2D_block( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    libxsmm_datatype                   colbias_precision,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_add_colbias_to_2D_block( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    libxsmm_datatype                   colbias_precision,
    const unsigned int                 l_vec_reg_acc_start,
    const unsigned int                 l_m_blocking,
    const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_prepare_coeffs_sigmoid_ps_rational_78_avx_avx512( libxsmm_generated_code*                        io_generated_code,
    libxsmm_micro_kernel_config*        i_micro_kernel_config,
    unsigned int                        reserved_zmms,
    unsigned int                        reserved_mask_regs,
    unsigned int                        temp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_fill_stack_vars_v2( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*        i_micro_kernel_config,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_fill_stack_vars(libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*        i_micro_kernel_config);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_allocate_scratch( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*        i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                  const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                  libxsmm_micro_kernel_config*        i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                  const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                  const libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_fusion_microkernel_properties_v2(const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                libxsmm_micro_kernel_config*        i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_fusion_microkernel_properties(const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                libxsmm_micro_kernel_config*        i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_fullvector( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                 const unsigned int             i_arch,
                                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                 const unsigned int             i_use_masking_a_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_halfvector( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                 const unsigned int             i_arch,
                                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                 const unsigned int             i_use_masking_a_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_scalar( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                             const unsigned int             i_arch,
                                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                             const unsigned int             i_use_masking_a_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_add_flop_counter( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_kloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_k_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_kloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_max_blocked_k,
                                           const unsigned int                 i_kloop_complete );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_reduceloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_nloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_n_init,
                                           const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_nloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_n_blocking,
                                           const unsigned int                 i_n_done );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_mloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_m_init,
                                           const unsigned int                 i_m_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_mloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_m_done );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C( libxsmm_generated_code*             io_generated_code,
                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                     const unsigned int                 i_m_blocking,
                                     const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C( libxsmm_generated_code*             io_generated_code,
                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                      const unsigned int                 i_m_blocking,
                                      const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
                                                    const unsigned int                 i_gp_reg_tmp,
                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                    const unsigned int                 i_mask_count );

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_get_rbp_relative_offset( libxsmm_gemm_stack_var stack_var );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_getval_stack_var( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              libxsmm_gemm_stack_var              stack_var,
                                              unsigned int                        i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setval_stack_var( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              libxsmm_gemm_stack_var              stack_var,
                                              unsigned int                        i_gp_reg );

#endif /* GENERATOR_GEMM_COMMON_H */
