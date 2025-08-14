/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_TRANSFORM_RV64_SVE_H
#define GENERATOR_MATELTWISE_TRANSFORM_RV64_SVE_H

#include "generator_rv64_instructions.h"
#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_rv64_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const unsigned int                      i_gp_reg_scratch,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_load_regblock_8x8_rv64( libxsmm_generated_code*  io_generated_code,
                                                         const unsigned int       i_gp_reg_addr,
                                                         const unsigned int       i_gp_reg_dst,
                                                         const unsigned int       i_gp_reg_scratch,
                                                         const unsigned int       i_valid_e_regs,
                                                         const unsigned int       i_valid_o_regs,
                                                         const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                         const libxsmm_meltw_descriptor* i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_store_regblock_8x8_rv64( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_addr,
                                                          const unsigned int      i_gp_reg_dst,
                                                          const unsigned int      i_gp_reg_scratch,
                                                          const unsigned int      i_valid_e_regs,
                                                          const unsigned int      i_valid_o_regs,
                                                          const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                          const libxsmm_meltw_descriptor* i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_store_regblock_4x4_rv64( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_addr,
                                                          const unsigned int      i_gp_reg_dst,
                                                          const unsigned int      i_gp_reg_scratch,
                                                          const unsigned int      i_valid_e_regs,
                                                          const unsigned int      i_valid_o_regs,
                                                          const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                          const libxsmm_meltw_descriptor* i_mateltwise_desc );

LIBXSMM_API_INTERN                                                                                                   void libxsmm_generator_transform_norm_to_normt_shuffle_regblock_32bit_8x8_rvv( libxsmm_generated_code* io_generated_code,
                                                                               const unsigned int      i_gp_reg_dst_e,
                                                                               const unsigned int      i_gp_reg_dst_o,
                                                                               const unsigned int      i_gp_reg_scratch,
                                                                               const unsigned int      i_mask_e,
                                                                               const unsigned int      i_mask_o,
                                                                               const unsigned int      i_shuffle_stride );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_shuffle_regblock_64bit_4x4_rvv( libxsmm_generated_code* io_generated_code,
                                                                               const unsigned int      i_gp_reg_dst_e,
                                                                               const unsigned int      i_gp_reg_dst_o,
                                                                               const unsigned int      i_gp_reg_scratch,
                                                                               const unsigned int      i_mask_e,
                                                                               const unsigned int      i_mask_o,
                                                                               const unsigned int      i_shuffle_stride );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_rvv( libxsmm_generated_code*     io_generated_code,
                                                                             libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                                             const unsigned int          i_gp_reg_in,                                                                             const unsigned int          i_gp_reg_out,                                                                                                                                                                                                 const unsigned int          i_gp_reg_scratch,
                                                                             const unsigned int          i_m,
                                                                             const unsigned int          i_n,
                                                                             const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor* i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_rvv( libxsmm_generated_code*     io_generated_code,
                                                                             libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                                             const unsigned int          i_gp_reg_in,                                                                             const unsigned int          i_gp_reg_out,
                                                                             const unsigned int          i_gp_reg_scratch,
                                                                             const unsigned int          i_m,
                                                                             const unsigned int          i_n,
                                                                             const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor* i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_rvv_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_rvv_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_rv64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                   libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                   libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                   const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                   const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_TRANSFORM_RV64_SVE_H */
