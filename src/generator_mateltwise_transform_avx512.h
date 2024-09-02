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
#ifndef GENERATOR_MATELTWISE_TRANSFORM_AVX_AVX512_H
#define GENERATOR_MATELTWISE_TRANSFORM_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_vl512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_vl256_microkernel( libxsmm_generated_code*             io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const unsigned int                      i_gp_reg_mask,
                                                                        const unsigned int                      i_mask_reg_0,
                                                                        const unsigned int                      i_mask_reg_1,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                        const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_vnni_lo_reg,
                                                                                 const unsigned int                      i_vnni_hi_reg,
                                                                                 const unsigned int                      i_vnni_lo_reg_2,
                                                                                 const unsigned int                      i_vnni_hi_reg_2,
                                                                                 const unsigned int                      i_m_step,
                                                                                 const unsigned int                      i_n_step,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx512_vl512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_gp_reg_m_loop,
                                                                                 const unsigned int                      i_gp_reg_n_loop,
                                                                                 const unsigned int                      i_gp_reg_mask,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx512_vl256_microkernel( libxsmm_generated_code*             io_generated_code,
                                                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_gp_reg_m_loop,
                                                                                 const unsigned int                      i_gp_reg_n_loop,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_mask_reg_1,
                                                                                  const unsigned int                      i_vnni_lo_reg,
                                                                                  const unsigned int                      i_vnni_hi_reg,
                                                                                  const unsigned int                      i_vnni_lo_reg_2,
                                                                                  const unsigned int                      i_vnni_hi_reg_2,
                                                                                  const unsigned int                      i_vnni_lo_reg_4,
                                                                                  const unsigned int                      i_vnni_hi_reg_4,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_shuffle_network_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned char     i_in_idx[16],
                                                              const unsigned char     i_shuf_imm[16],
                                                              const unsigned int      i_vec_reg_src_start,
                                                              const unsigned int      i_vec_reg_dst_start,
                                                              const unsigned int      i_out_offset,
                                                              const unsigned int      i_shuffle_instr,
                                                              const unsigned int      i_ways );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_byteshuffle_network_avx512( libxsmm_generated_code* io_generated_code,
                                                                  const char              i_vector_name,
                                                                  const unsigned char     i_in_idx[16],
                                                                  const unsigned int      i_vec_reg_suffle_cntl,
                                                                  const unsigned int      i_vec_reg_src_start,
                                                                  const unsigned int      i_vec_reg_dst_start,
                                                                  const unsigned int      i_shuffle_instr,
                                                                  const unsigned int      i_ways );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_16way_permute_network_avx512( libxsmm_generated_code* io_generated_code,
                                                               const char              i_vector_name,
                                                               const unsigned char     i_perm_mask[2],
                                                               const unsigned char     i_perm_imm[2],
                                                               const unsigned int      i_vec_reg_srcdst_start,
                                                               const unsigned int      i_perm_instr );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_permutevar1_network_avx512( libxsmm_generated_code* io_generated_code,
                                                                  const char              i_vector_name,
                                                                  const unsigned int      i_vec_reg_perm_idx,
                                                                  const unsigned int      i_vec_reg_srcdst_start,
                                                                  const unsigned int      i_perm_instr,
                                                                  const unsigned int      i_ways );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_08way_permutevar_network_avx512( libxsmm_generated_code* io_generated_code,
                                                                  const char              i_vector_name,
                                                                  const unsigned int      i_vec_reg_perm_idx_lo,
                                                                  const unsigned int      i_vec_reg_perm_idx_hi,
                                                                  const unsigned int      i_vec_reg_srcdst_start,
                                                                  const unsigned int      i_perm_instr );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_permute_network_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned char     i_perm_mask[2],
                                                              const unsigned char     i_perm_imm[2],
                                                              const unsigned int      i_vec_reg_srcdst_start,
                                                              const unsigned int      i_perm_instr,
                                                              const unsigned int      i_ways );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_half_load_blend_avx512( libxsmm_generated_code* io_generated_code,
                                                              const char              i_vector_name,
                                                              const unsigned int      i_gp_reg_in,
                                                              const unsigned int      i_vec_reg_dst_start,
                                                              const unsigned int      i_ld,
                                                              const unsigned int*     i_ld_idx,
                                                              const unsigned int      i_blend_mult,
                                                              const unsigned int      i_ld_instr,
                                                              const unsigned int      i_ways,
                                                              const unsigned int      i_mask_reg[2],
                                                              const unsigned int      i_m );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_Xway_quarter_load_blend_avx512( libxsmm_generated_code* io_generated_code,
                                                                 const char              i_vector_name,
                                                                 const unsigned int      i_gp_reg_in,
                                                                 const unsigned int      i_vec_reg_dst_start,
                                                                 const unsigned int      i_ld,
                                                                 const unsigned int      i_ld_instr,
                                                                 const unsigned int      i_ways,
                                                                 const unsigned int      i_mask_reg[4],
                                                                 const unsigned int      i_n,
                                                                 const unsigned int      is_non32bit_ld );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_32way_half_store_avx512( libxsmm_generated_code* io_generated_code,
                                                          const char              i_vector_name,
                                                          const unsigned int      i_gp_reg_out,
                                                          const unsigned int      i_vec_reg_src_start,
                                                          const unsigned int      i_ld,
                                                          const unsigned int      i_st_instr );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_4x4_64bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                     const char              i_vector_name,
                                                                     const unsigned int      i_vec_reg_src_start,
                                                                     const unsigned int      i_vec_reg_dst_start,
                                                                     const unsigned int      i_mask_reg_1,
                                                                     const unsigned int      i_mask_reg_2 );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_128bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                          libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                          const unsigned int                      i_gp_reg_in,
                                                                          const unsigned int                      i_gp_reg_out,
                                                                          const unsigned int                      i_gp_reg_m_loop,
                                                                          const unsigned int                      i_gp_reg_n_loop,
                                                                          const unsigned int                      i_gp_reg_mask,
                                                                          const unsigned int                      i_mask_reg_0,
                                                                          const unsigned int                      i_mask_reg_1,
                                                                          const unsigned int                      i_mask_reg_2,
                                                                          const unsigned int                      i_mask_reg_3,
                                                                          const unsigned int                      i_mask_reg_4,
                                                                          const unsigned int                      i_mask_reg_5,
                                                                          const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                          const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_8x8_32bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                     const char              i_vector_name,
                                                                     const unsigned int      i_vec_reg_srcdst_start,
                                                                     const unsigned int      i_vec_reg_tmp_start,
                                                                     const unsigned int      i_mask_reg_1,
                                                                     const unsigned int      i_mask_reg_2 );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_4x4_32bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                      const char              i_vector_name,
                                                                      const unsigned int      i_vec_reg_srcdst_start,
                                                                      const unsigned int      i_vec_reg_tmp_start );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_copy_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                          libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                          const unsigned int                      i_gp_reg_in,
                                                          const unsigned int                      i_gp_reg_out,
                                                          const unsigned int                      i_gp_reg_m_loop,
                                                          const unsigned int                      i_gp_reg_n_loop,
                                                          const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                          const unsigned int                      i_ldi,
                                                          const unsigned int                      i_ldo,
                                                          const unsigned int                      i_m,
                                                          const unsigned int                      i_n,
                                                          const unsigned int                      i_bsize );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx512_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_mask,
                                                                             const unsigned int                      i_gp_reg_mask_2,
                                                                             const unsigned int                      i_mask_reg_0,
                                                                             const unsigned int                      i_mask_reg_1,
                                                                             const unsigned int                      i_mask_reg_2,
                                                                             const unsigned int                      i_mask_reg_3,
                                                                             const unsigned int                      i_mask_reg_4,
                                                                             const unsigned int                      i_mask_reg_5,
                                                                             const unsigned int                      i_mask_reg_6,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx512_pre_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                 const unsigned int                      i_gp_reg_in,
                                                                                 const unsigned int                      i_gp_reg_out,
                                                                                 const unsigned int                      i_gp_reg_m_loop,
                                                                                 const unsigned int                      i_gp_reg_n_loop,
                                                                                 const unsigned int                      i_gp_reg_mask,
                                                                                 const unsigned int                      i_mask_reg_0,
                                                                                 const unsigned int                      i_mask_reg_1,
                                                                                 const unsigned int                      i_mask_reg_2,
                                                                                 const unsigned int                      i_mask_reg_3,
                                                                                 const unsigned int                      i_mask_reg_4,
                                                                                 const unsigned int                      i_mask_reg_5,
                                                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_8x8_16bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                      const char              i_vector_name,
                                                                      const unsigned int      i_vec_reg_src_start,
                                                                      const unsigned int      i_vec_reg_dst_start );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_16x16_08bit_norm_to_normt_avx512( libxsmm_generated_code* io_generated_code,
                                                                        const char              i_vector_name,
                                                                        const unsigned int      i_vec_reg_srcdst_start,
                                                                        const unsigned int      i_vec_reg_tmp_start );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_gp_reg_mask_2,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const unsigned int                      i_mask_reg_2,
                                                                         const unsigned int                      i_mask_reg_3,
                                                                         const unsigned int                      i_mask_reg_4,
                                                                         const unsigned int                      i_mask_reg_5,
                                                                         const unsigned int                      i_mask_reg_6,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_8x8_08bit_vnni4_to_vnni4t_avx512( libxsmm_generated_code* io_generated_code,
                                                                       const char              i_vector_name,
                                                                       const unsigned int      i_vec_reg_srcdst_start,
                                                                       const unsigned int      i_shuffle_op,
                                                                       const unsigned int      i_mask_reg_1,
                                                                       const unsigned int      i_mask_reg_2 );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_two_8x8_16bit_vnni2_to_vnni2t_avx512( libxsmm_generated_code* io_generated_code,
                                                                       const char              i_vector_name,
                                                                       const unsigned int      i_vec_reg_srcdst_start,
                                                                       const unsigned int      i_vec_reg_tmp_start,
                                                                       const unsigned int      i_shuffle_op,
                                                                       const unsigned int      i_mask_reg_1,
                                                                       const unsigned int      i_mask_reg_2 );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_four_4x4_16bit_vnni2_to_vnni2t_avx512( libxsmm_generated_code* io_generated_code,
                                                                        const char              i_vector_name,
                                                                        const unsigned int      i_vec_reg_srcdst_start,
                                                                        const unsigned int      i_vec_reg_tmp_start,
                                                                        const unsigned int      i_shuffle_op );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                               libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                               const unsigned int                      i_gp_reg_in,
                                                                               const unsigned int                      i_gp_reg_out,
                                                                               const unsigned int                      i_gp_reg_m_loop,
                                                                               const unsigned int                      i_gp_reg_n_loop,
                                                                               const unsigned int                      i_gp_reg_mask,
                                                                               const unsigned int                      i_gp_reg_mask_2,
                                                                               const unsigned int                      i_mask_reg_0,
                                                                               const unsigned int                      i_mask_reg_1,
                                                                               const unsigned int                      i_mask_reg_2,
                                                                               const unsigned int                      i_mask_reg_3,
                                                                               const unsigned int                      i_mask_reg_4,
                                                                               const unsigned int                      i_mask_reg_5,
                                                                               const unsigned int                      i_mask_reg_6,
                                                                               const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_pre_spr_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                   libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                   const unsigned int                      i_gp_reg_in,
                                                                                   const unsigned int                      i_gp_reg_out,
                                                                                   const unsigned int                      i_gp_reg_m_loop,
                                                                                   const unsigned int                      i_gp_reg_n_loop,
                                                                                   const unsigned int                      i_gp_reg_mask,
                                                                                   const unsigned int                      i_mask_reg_0,
                                                                                   const unsigned int                      i_mask_reg_1,
                                                                                   const unsigned int                      i_mask_reg_2,
                                                                                   const unsigned int                      i_mask_reg_3,
                                                                                   const unsigned int                      i_mask_reg_4,
                                                                                   const unsigned int                      i_mask_reg_5,
                                                                                   const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                   const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_gp_reg_mask_2,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const unsigned int                      i_mask_reg_2,
                                                                           const unsigned int                      i_mask_reg_3,
                                                                           const unsigned int                      i_mask_reg_4,
                                                                           const unsigned int                      i_mask_reg_5,
                                                                           const unsigned int                      i_mask_reg_6,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_perm_2nd_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_norm_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_perm_2nd_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_norm_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_mask_reg_1,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                   const unsigned int                      i_gp_reg_in,
                                                                                   const unsigned int                      i_gp_reg_out,
                                                                                   const unsigned int                      i_mask_reg_0,
                                                                                   const unsigned int                      i_mask_reg_1,
                                                                                   const unsigned int                      i_vnni_lo_reg,
                                                                                   const unsigned int                      i_vnni_hi_reg,
                                                                                   const unsigned int                      i_m_step,
                                                                                   const unsigned int                      i_n_step,
                                                                                   const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                   const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const unsigned int                      i_mask_reg_2,
                                                                           const unsigned int                      i_mask_reg_3,
                                                                           const unsigned int                      i_mask_reg_4,
                                                                           const unsigned int                      i_mask_reg_5,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const unsigned int                      i_mask_reg_2,
                                                                           const unsigned int                      i_mask_reg_3,
                                                                           const unsigned int                      i_mask_reg_4,
                                                                           const unsigned int                      i_mask_reg_5,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_08bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_mask_reg_0,
                                                                                  const unsigned int                      i_mask_reg_1,
                                                                                  const unsigned int                      i_perm_1st_stage_reg,
                                                                                  const unsigned int                      i_m_step,
                                                                                  const unsigned int                      i_n_step,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                         libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                         const unsigned int                      i_gp_reg_in,
                                                                         const unsigned int                      i_gp_reg_out,
                                                                         const unsigned int                      i_gp_reg_m_loop,
                                                                         const unsigned int                      i_gp_reg_n_loop,
                                                                         const unsigned int                      i_gp_reg_mask,
                                                                         const unsigned int                      i_mask_reg_0,
                                                                         const unsigned int                      i_mask_reg_1,
                                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                         const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_mnblock_micro_kernel( libxsmm_generated_code*                 io_generated_code,
                                                                                    const unsigned int                      i_gp_reg_in,
                                                                                    const unsigned int                      i_gp_reg_out,
                                                                                    const unsigned int                      i_mask_reg_0,
                                                                                    const unsigned int                      i_mask_reg_1,
                                                                                    const unsigned int                      i_m_step_in,
                                                                                    const unsigned int                      i_m_step_out,
                                                                                    const unsigned int                      i_n_step,
                                                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod4_08bit_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                           libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                           const unsigned int                      i_gp_reg_in,
                                                                           const unsigned int                      i_gp_reg_out,
                                                                           const unsigned int                      i_gp_reg_m_loop,
                                                                           const unsigned int                      i_gp_reg_n_loop,
                                                                           const unsigned int                      i_gp_reg_mask,
                                                                           const unsigned int                      i_mask_reg_0,
                                                                           const unsigned int                      i_mask_reg_1,
                                                                           const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                     libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                     libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                     const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                     const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_TRANSFORM_AVX_AVX512_H */

