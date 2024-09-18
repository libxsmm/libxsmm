/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer, Antonio Noack (FSU Jena), Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_COMMON_AARCH64_H
#define GENERATOR_COMMON_AARCH64_H

#include "generator_common.h"
#include "generator_aarch64_instructions.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_vreg_asimd_aarch64( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_addr,
                                                          const unsigned int      i_gp_reg_scratch0,
                                                          const unsigned int      i_gp_reg_scratch1,
                                                          const unsigned int      i_idx_vec_reg,
                                                          const unsigned int      i_idx_datatype_size,
                                                          const unsigned int      i_srcdst_vec_reg,
                                                          const unsigned int      i_datatype_size,
                                                          const unsigned int      i_masked_elems,
                                                          const unsigned int      i_is_gather );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_bf16f32_aarch64_sve( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32bf16_aarch64_sve( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32i8_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_scf_vec_reg,
    const unsigned int i_pred_reg,
    const unsigned int i_skip_scaling,
    const unsigned int i_sign_sat );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_i8f32_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_scf_vec_reg,
    const unsigned int i_pred_reg,
    unsigned int       i_skip_scaling );

/**
 * Sets the given predicate register.
 *
 * @param io_generated_code will be updated with respective instructions.
 * @param i_pred_reg id of the predicate register which is set.
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_bf16f32_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32bf16_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_set_p_register_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                   unsigned int            i_p_reg,
                                                            int            i_n_bits,
                                                   unsigned int            i_gp_reg_scratch );

LIBXSMM_API_INTERN
void libxsmm_generator_mov_aarch64( libxsmm_generated_code* io_generated_code,
                                    unsigned int            i_src_reg,
                                    unsigned int            i_dst_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_aarch64( libxsmm_generated_code* io_generated_code,
                                         unsigned int            instr,
                                         const unsigned int      i_vec_inout,
                                         libxsmm_aarch64_asimd_tupletype  i_tuple_type    );

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_trips );

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_gp_reg_bound_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_gp_reg_bound );
LIBXSMM_API_INTERN
void libxsmm_generator_loop_footer_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_loop_blocking );

/*
 * loads i_masked_elems elements from i_gp_reg_addr
 * if i_masked_elems is 0, it loads as many as fit into a vector register
 * if the architecture is SVE, the predicate registers are assumed to be set already: 0 must be ptrue, 1 must have length i_masked_elems
 */
LIBXSMM_API_INTERN
void libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_gp_reg_addr,
                                                             const unsigned int      i_gp_reg_scratch,
                                                             const unsigned int      i_vec_reg,
                                                             const unsigned int      i_datatype_size,
                                                             const unsigned int      i_masked_elems,
                                                             const unsigned int      i_adv_gpr,
                                                             const unsigned int      i_is_store );

/*
 * loads i_masked_elems elements from i_gp_reg_addr
 * if i_masked_elems is 0, it loads as many as fit into a vector register
 * if the architecture is SVE, the predicate register i_mask_reg is assumed to be set already
 */
LIBXSMM_API_INTERN
void libxsmm_generator_vloadstore_masked_vreg_aarch64( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_addr,
                                                       const unsigned int      i_gp_reg_scratch,
                                                       const unsigned int      i_vec_reg,
                                                       const unsigned int      i_datatype_size,
                                                       const unsigned int      i_masked_elems,
                                                       const unsigned int      i_adv_gpr,
                                                       const unsigned int      i_is_store,
                                                       const unsigned char     i_mask_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                            const unsigned int      i_gp_reg_addr,
                                                            const unsigned int      i_gp_reg_scratch,
                                                            const unsigned int      i_vec_reg,
                                                            const unsigned int      i_datatype_size,
                                                            const unsigned int      i_masked_elems,
                                                            const unsigned int      i_adv_gpr );

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_addr,
                                                      const unsigned int      i_gp_reg_scratch_a,
                                                      const unsigned int      i_vec_length,
                                                      const unsigned int      i_vec_reg_count,
                                                      const unsigned int      i_m_blocking,
                                                      const unsigned int      i_n_blocking,
                                                      const unsigned int      i_ld,
                                                      const unsigned int      i_zero );

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_mmla_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int      i_gp_reg_addr,
                                                           const unsigned int      i_gp_reg_scratch,
                                                           const unsigned int      i_vec_length,
                                                           const unsigned int      i_vec_reg_count,
                                                           const unsigned int      i_m_blocking,
                                                           const unsigned int      i_n_blocking,
                                                           const unsigned int      i_ld,
                                                           const unsigned int      i_zero,
                                                           const unsigned char     i_zip_row_major );

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                    const libxsmm_datatype  i_datatype,
                                                    const unsigned int      i_gp_reg_addr,
                                                    const unsigned int      i_gp_reg_scratch,
                                                    const unsigned int      i_vec_length,
                                                    const unsigned int      i_vec_reg_count,
                                                    const unsigned int      i_m_blocking,
                                                    const unsigned int      i_n_blocking,
                                                    const unsigned int      i_ld,
                                                    const unsigned int      i_zero );

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_mmla_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const unsigned int      i_gp_reg_addr,
                                                         const unsigned int      i_gp_reg_scratch,
                                                         const unsigned int      i_gp_reg_scratch1,
                                                         const unsigned int      i_vec_length,
                                                         const unsigned int      i_vec_reg_count,
                                                         const unsigned int      i_m_blocking,
                                                         const unsigned int      i_n_blocking,
                                                         const unsigned int      i_ld,
                                                         const unsigned int      i_data_size,
                                                         const unsigned int      i_zero,
                                                         const unsigned char     i_zip_row_major );

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_addr,
                                                       const unsigned int      i_gp_reg_scratch_a,
                                                       const unsigned int      i_vec_length,
                                                       const unsigned int      i_vec_reg_count,
                                                       const unsigned int      i_m_blocking,
                                                       const unsigned int      i_n_blocking,
                                                       const unsigned int      i_ld );

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_mmla_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                            const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                            const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                            const unsigned int      i_gp_reg_addr,
                                                            const unsigned int      i_gp_reg_scratch,
                                                            const unsigned int      i_vec_length,
                                                            const unsigned int      i_vec_reg_count,
                                                            const unsigned int      i_m_blocking,
                                                            const unsigned int      i_n_blocking,
                                                            const unsigned int      i_ld,
                                                            const unsigned char     i_zip_row_major );

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_mmla_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                          const unsigned int      i_gp_reg_addr,
                                                          const unsigned int      i_gp_reg_scratch,
                                                          const unsigned int      i_gp_reg_scratch1,
                                                          const unsigned int      i_gp_reg_scratch2,
                                                          const unsigned int      i_gp_reg_scratch3,
                                                          const unsigned int      i_vec_length,
                                                          const unsigned int      i_vec_reg_count,
                                                          const unsigned int      i_m_blocking,
                                                          const unsigned int      i_n_blocking,
                                                          const unsigned int      i_ld,
                                                          const unsigned int      i_data_size,
                                                          const unsigned int      i_zero,
                                                          const unsigned char     i_zip_row_major );

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                     const libxsmm_datatype  i_datatype,
                                                     const unsigned int      i_gp_reg_addr,
                                                     const unsigned int      i_gp_reg_scratch,
                                                     const unsigned int      i_vec_length,
                                                     const unsigned int      i_vec_reg_count,
                                                     const unsigned int      i_m_blocking,
                                                     const unsigned int      i_n_blocking,
                                                     const unsigned int      i_ld,
                                                     const libxsmm_datatype  i_inp_datatype,
                                                     const unsigned int      i_aux_gp_reg,
                                                     const unsigned int      i_reduce_on_output  );

LIBXSMM_API_INTERN
void libxsmm_generator_load_prng_state_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_prng_state_ptr,
                                                      const unsigned int      prng_state0_vreg,
                                                      const unsigned int      prng_state1_vreg,
                                                      const unsigned int      prng_state2_vreg,
                                                      const unsigned int      prng_state3_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_store_prng_state_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_prng_state_ptr,
                                                       const unsigned int      prng_state0_vreg,
                                                       const unsigned int      prng_state1_vreg,
                                                       const unsigned int      prng_state2_vreg,
                                                       const unsigned int      prng_state3_vreg );


LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_tmp,
                                                      const unsigned int      i_gp_reg_prob_ptr,
                                                      const unsigned int      dropout_vreg_one,
                                                      const unsigned int      dropout_prob_vreg,
                                                      const unsigned int      dropout_invprob_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_inv_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_tmp,
                                                          const unsigned int      i_gp_reg_prob_ptr,
                                                          const unsigned int      dropout_vreg_one,
                                                          const unsigned int      dropout_vreg_zero,
                                                          const unsigned int      dropout_prob_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128p_f32_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_vec_reg_rng_state_0,
                                                      const unsigned int      i_vec_reg_rng_state_1,
                                                      const unsigned int      i_vec_reg_rng_state_2,
                                                      const unsigned int      i_vec_reg_rng_state_3,
                                                      const unsigned int      i_vec_reg_rng_tmp_0,
                                                      const unsigned int      i_vec_reg_rng_tmp_1,
                                                      const unsigned int      i_vec_reg_rng_one,
                                                      const unsigned int      o_vec_reg_rng );

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128p_f32_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_reg_rng_state_0,
                                                    const unsigned int      i_vec_reg_rng_state_1,
                                                    const unsigned int      i_vec_reg_rng_state_2,
                                                    const unsigned int      i_vec_reg_rng_state_3,
                                                    const unsigned int      i_vec_reg_rng_tmp_0,
                                                    const unsigned int      i_vec_reg_rng_tmp_1,
                                                    const unsigned int      i_vec_reg_rng_one,
                                                    const unsigned int      o_vec_reg_rng,
                                                    const libxsmm_aarch64_sve_type i_sve_type,
                                                    const unsigned char     i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128pp_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_reg_rng_state_0,
                                                   const unsigned int      i_vec_reg_rng_state_1,
                                                   const unsigned int      i_vec_reg_rng_state_2,
                                                   const unsigned int      i_vec_reg_rng_state_3,
                                                   const unsigned int      i_vec_reg_rng_tmp_0,
                                                   const unsigned int      i_vec_reg_rng_tmp_1,
                                                   const unsigned int      o_vec_reg_rng );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                                      const unsigned int                             i_vec_thres,
                                                                      const unsigned int                             i_vec_absmask,
                                                                      const unsigned int                             i_vec_scale,
                                                                      const unsigned int                             i_vec_shifter,
                                                                      const unsigned int                             i_vec_half,
                                                                      const unsigned int                             i_vec_c0,
                                                                      const unsigned int                             i_vec_c01,
                                                                      const unsigned int                             i_vec_c02,
                                                                      const unsigned int                             i_vec_c03,
                                                                      const unsigned int                             i_vec_c1,
                                                                      const unsigned int                             i_vec_c11,
                                                                      const unsigned int                             i_vec_c12,
                                                                      const unsigned int                             i_vec_c13,
                                                                      const unsigned int                             i_vec_c2,
                                                                      const unsigned int                             i_vec_c21,
                                                                      const unsigned int                             i_vec_c22,
                                                                      const unsigned int                             i_vec_c23,
                                                                      const unsigned int                             i_vec_tmp,
                                                                      const unsigned int                             i_vec_tmp1,
                                                                      const unsigned int                             i_gp_reg_tmp,
                                                                      const unsigned int                             i_gp_reg_tmp1,
                                                                      const libxsmm_aarch64_asimd_tupletype          i_tupletype );
/**
 * prepares registers for gelu call
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_thres threshold sve register which is set to 0x40879fff for each elemenent
 * @param i_vec_absmask mask sve register which is set to 0x7fffffff for each elemenent
 * @param i_vec_scale scale sve register which is set to 0x406a0ea1 for each elemenent
 * @param i_vec_shifter shift sve register which is set to 0x4b400000 for each elemenent
 * @param i_vec_half half sve register which is set to 0x3f000000 for each elemenent
 * @param i_vec_c0 sve register which is set to number of the length 16 array for approxiation
 * @param i_vec_c01 sve register which is set to number of the length 16 array C1 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c1 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c11 sve register which is set to number of the length 16 array C2 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c2 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c21 sve register which is set to number of the length 16 array C3 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_exp_mask exponent mask sve register which is set to 0x000000ff for each elemenent
 * @param i_gp_reg_tmp first temp register
 * @param i_gp_reg_tmp1 second temp register
 * @param i_sve_type type of sve (B, H, S, D)
 * @param i_pred_reg predicate register
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64_sve( libxsmm_generated_code*        io_generated_code,
                                                                    const unsigned int             i_vec_thres,
                                                                    const unsigned int             i_vec_absmask,
                                                                    const unsigned int             i_vec_scale,
                                                                    const unsigned int             i_vec_shifter,
                                                                    const unsigned int             i_vec_half,
                                                                    const unsigned int             i_vec_c0,
                                                                    const unsigned int             i_vec_c01,
                                                                    const unsigned int             i_vec_c02,
                                                                    const unsigned int             i_vec_c03,
                                                                    const unsigned int             i_vec_c1,
                                                                    const unsigned int             i_vec_c11,
                                                                    const unsigned int             i_vec_c12,
                                                                    const unsigned int             i_vec_c13,
                                                                    const unsigned int             i_vec_c2,
                                                                    const unsigned int             i_vec_c21,
                                                                    const unsigned int             i_vec_c22,
                                                                    const unsigned int             i_vec_c23,
                                                                    const unsigned int             i_vec_exp_mask,
                                                                    const unsigned int             i_gp_reg_tmp,
                                                                    const unsigned int             i_gp_reg_tmp1,
                                                                    const libxsmm_aarch64_sve_type i_sve_type,
                                                                    const unsigned char            i_pred_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                       const unsigned int                             i_vec_x,
                                                       const unsigned int                             i_vec_xr,
                                                       const unsigned int                             i_vec_xa,
                                                       const unsigned int                             i_vec_index,
                                                       const unsigned int                             i_vec_C0,
                                                       const unsigned int                             i_vec_C1,
                                                       const unsigned int                             i_vec_C2,
                                                       const unsigned int                             i_vec_thres,
                                                       const unsigned int                             i_vec_absmask,
                                                       const unsigned int                             i_vec_scale,
                                                       const unsigned int                             i_vec_shifter,
                                                       const unsigned int                             i_vec_half,
                                                       const unsigned int                             i_vec_c0,
                                                       const unsigned int                             i_vec_c1,
                                                       const unsigned int                             i_vec_c2,
                                                       const unsigned int                             i_vec_tmp,
                                                       const unsigned int                             i_vec_tmp1,
                                                       const libxsmm_aarch64_asimd_tupletype          i_tupletype );

/**
 * prepares registers for gelu call
 * @param i_vec_x sve register
 * @param i_vec_xr sve register
 * @param i_vec_xa sve register
 * @param i_vec_index sve register with the index for the TBL
 * @param i_vec_C0 sve register for the result of the TBL on array C0
 * @param i_vec_C1 sve register for the result of the TBL on array C1
 * @param i_vec_C2 sve register for the result of the TBL on array C2
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_thres threshold sve register which is set to 0x40879fff for each elemenent
 * @param i_vec_absmask mask sve register which is set to 0x7fffffff for each elemenent
 * @param i_vec_scale scale sve register which is set to 0x406a0ea1 for each elemenent
 * @param i_vec_shifter shift sve register which is set to 0x4b400000 for each elemenent
 * @param i_vec_half half sve register which is set to 0x3f000000 for each elemenent
 * @param i_vec_c0 sve register which is set to number of the length 16 array for approxiation
 * @param i_vec_c01 sve register which is set to number of the length 16 array C1 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c1 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c11 sve register which is set to number of the length 16 array C2 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c2 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c21 sve register which is set to number of the length 16 array C3 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_exp_mask exponent mask sve register which is set to 0x000000ff for each elemenent
 * @param i_vec_temp temp sve register
 * @param i_sve_type type of sve (B, H, S, D)
 * @param i_pred_reg predicate register
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_aarch64_sve( libxsmm_generated_code*        io_generated_code,
                                                     const unsigned int             i_vec_x,
                                                     const unsigned int             i_vec_xr,
                                                     const unsigned int             i_vec_xa,
                                                     const unsigned int             i_vec_index,
                                                     const unsigned int             i_vec_C0,
                                                     const unsigned int             i_vec_C1,
                                                     const unsigned int             i_vec_C2,
                                                     const unsigned int             i_vec_thres,
                                                     const unsigned int             i_vec_absmask,
                                                     const unsigned int             i_vec_scale,
                                                     const unsigned int             i_vec_shifter,
                                                     const unsigned int             i_vec_half,
                                                     const unsigned int             i_vec_c0,
                                                     const unsigned int             i_vec_c01,
                                                     const unsigned int             i_vec_c02,
                                                     const unsigned int             i_vec_c03,
                                                     const unsigned int             i_vec_c1,
                                                     const unsigned int             i_vec_c11,
                                                     const unsigned int             i_vec_c12,
                                                     const unsigned int             i_vec_c13,
                                                     const unsigned int             i_vec_c2,
                                                     const unsigned int             i_vec_c21,
                                                     const unsigned int             i_vec_c22,
                                                     const unsigned int             i_vec_c23,
                                                     const unsigned int             i_vec_exp_mask,
                                                     const unsigned int             i_vec_temp,
                                                     const libxsmm_aarch64_sve_type i_sve_type,
                                                     const unsigned char            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                           const unsigned int                             i_vec_x,
                                                           const unsigned int                             i_vec_xr,
                                                           const unsigned int                             i_vec_xa,
                                                           const unsigned int                             i_vec_index,
                                                           const unsigned int                             i_vec_C0,
                                                           const unsigned int                             i_vec_C1,
                                                           const unsigned int                             i_vec_C2,
                                                           const unsigned int                             i_vec_thres,
                                                           const unsigned int                             i_vec_absmask,
                                                           const unsigned int                             i_vec_scale,
                                                           const unsigned int                             i_vec_shifter,
                                                           const unsigned int                             i_vec_half,
                                                           const unsigned int                             i_vec_c0,
                                                           const unsigned int                             i_vec_c1,
                                                           const unsigned int                             i_vec_c2,
                                                           const unsigned int                             i_vec_tmp,
                                                           const unsigned int                             i_vec_tmp1,
                                                           const libxsmm_aarch64_asimd_tupletype          i_tupletype );

/**
 * prepares registers for gelu call
 * @param i_vec_x sve register
 * @param i_vec_xr sve register
 * @param i_vec_xa sve register
 * @param i_vec_index sve register with the index for the TBL
 * @param i_vec_C0 sve register for the result of the TBL on array C0
 * @param i_vec_C1 sve register for the result of the TBL on array C1
 * @param i_vec_C2 sve register for the result of the TBL on array C2
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_thres threshold sve register which is set to 0x408f5fff for each elemenent
 * @param i_vec_absmask mask sve register which is set to 0x7fffffff for each elemenent
 * @param i_vec_scale scale sve register which is set to 0x405d67c9 for each elemenent
 * @param i_vec_shifter shift sve register which is set to 0x4b400000 for each elemenent
 * @param i_vec_half half sve register which is set to 0x3f000000 for each elemenent
 * @param i_vec_c0 sve register which is set to number of the length 16 array for approxiation
 * @param i_vec_c01 sve register which is set to number of the length 16 array C1 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c1 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c11 sve register which is set to number of the length 16 array C2 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c2 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c21 sve register which is set to number of the length 16 array C3 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_exp_mask exponent mask sve register which is set to 0x000000ff for each elemenent
 * @param i_vec_temp temp sve register
 * @param i_sve_type type of sve (B, H, S, D)
 * @param i_pred_reg predicate register
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_aarch64_sve( libxsmm_generated_code*        io_generated_code,
                                                         const unsigned int             i_vec_x,
                                                         const unsigned int             i_vec_xr,
                                                         const unsigned int             i_vec_xa,
                                                         const unsigned int             i_vec_index,
                                                         const unsigned int             i_vec_C0,
                                                         const unsigned int             i_vec_C1,
                                                         const unsigned int             i_vec_C2,
                                                         const unsigned int             i_vec_thres,
                                                         const unsigned int             i_vec_absmask,
                                                         const unsigned int             i_vec_scale,
                                                         const unsigned int             i_vec_shifter,
                                                         const unsigned int             i_vec_half,
                                                         const unsigned int             i_vec_c0,
                                                         const unsigned int             i_vec_c01,
                                                         const unsigned int             i_vec_c02,
                                                         const unsigned int             i_vec_c03,
                                                         const unsigned int             i_vec_c1,
                                                         const unsigned int             i_vec_c11,
                                                         const unsigned int             i_vec_c12,
                                                         const unsigned int             i_vec_c13,
                                                         const unsigned int             i_vec_c2,
                                                         const unsigned int             i_vec_c21,
                                                         const unsigned int             i_vec_c22,
                                                         const unsigned int             i_vec_c23,
                                                         const unsigned int             i_vec_exp_mask,
                                                         const unsigned int             i_vec_temp,
                                                         const libxsmm_aarch64_sve_type i_sve_type,
                                                         const unsigned char            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                                          const unsigned int                             i_vec_thres,
                                                                          const unsigned int                             i_vec_absmask,
                                                                          const unsigned int                             i_vec_scale,
                                                                          const unsigned int                             i_vec_shifter,
                                                                          const unsigned int                             i_vec_half,
                                                                          const unsigned int                             i_vec_c0,
                                                                          const unsigned int                             i_vec_c01,
                                                                          const unsigned int                             i_vec_c02,
                                                                          const unsigned int                             i_vec_c03,
                                                                          const unsigned int                             i_vec_c1,
                                                                          const unsigned int                             i_vec_c11,
                                                                          const unsigned int                             i_vec_c12,
                                                                          const unsigned int                             i_vec_c13,
                                                                          const unsigned int                             i_vec_c2,
                                                                          const unsigned int                             i_vec_c21,
                                                                          const unsigned int                             i_vec_c22,
                                                                          const unsigned int                             i_vec_c23,
                                                                          const unsigned int                             i_vec_tmp,
                                                                          const unsigned int                             i_vec_tmp1,
                                                                          const unsigned int                             i_gp_reg_tmp,
                                                                          const unsigned int                             i_gp_reg_tmp1,
                                                                          const libxsmm_aarch64_asimd_tupletype          i_tupletype );

/**
 * prepares registers for gelu call
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_thres threshold sve register which is set to 0x408f5fff for each elemenent
 * @param i_vec_absmask mask sve register which is set to 0x7fffffff for each elemenent
 * @param i_vec_scale scale sve register which is set to 0x405d67c9 for each elemenent
 * @param i_vec_shifter shift sve register which is set to 0x4b400000 for each elemenent
 * @param i_vec_half half sve register which is set to 0x3f000000 for each elemenent
 * @param i_vec_c0 sve register which is set to number of the length 16 array for approxiation
 * @param i_vec_c01 sve register which is set to number of the length 16 array C1 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c1 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c11 sve register which is set to number of the length 16 array C2 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_c2 sve register which is set to number of the length 16 array for approxiation.
 * @param i_vec_c21 sve register which is set to number of the length 16 array C3 for approxiation. Only required if vectorlength < 512, else ignored.
 * @param i_vec_exp_mask exponent mask sve register which is set to 0x000000ff for each elemenent
 * @param i_gp_reg_tmp first temp register
 * @param i_gp_reg_tmp1 second temp register
 * @param i_sve_type type of sve (B, H, S, D)
 * @param i_pred_reg predicate register
 **/
LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64_sve( libxsmm_generated_code*        io_generated_code,
                                                                        const unsigned int             i_vec_thres,
                                                                        const unsigned int             i_vec_absmask,
                                                                        const unsigned int             i_vec_scale,
                                                                        const unsigned int             i_vec_shifter,
                                                                        const unsigned int             i_vec_half,
                                                                        const unsigned int             i_vec_c0,
                                                                        const unsigned int             i_vec_c01,
                                                                        const unsigned int             i_vec_c02,
                                                                        const unsigned int             i_vec_c03,
                                                                        const unsigned int             i_vec_c1,
                                                                        const unsigned int             i_vec_c11,
                                                                        const unsigned int             i_vec_c12,
                                                                        const unsigned int             i_vec_c13,
                                                                        const unsigned int             i_vec_c2,
                                                                        const unsigned int             i_vec_c21,
                                                                        const unsigned int             i_vec_c22,
                                                                        const unsigned int             i_vec_c23,
                                                                        const unsigned int             i_vec_exp_mask,
                                                                        const unsigned int             i_gp_reg_tmp,
                                                                        const unsigned int             i_gp_reg_tmp1,
                                                                        const libxsmm_aarch64_sve_type i_sve_type,
                                                                        const unsigned char            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                                 const unsigned int                             i_vec_c0,
                                                                 const unsigned int                             i_vec_c1,
                                                                 const unsigned int                             i_vec_c2,
                                                                 const unsigned int                             i_vec_c3,
                                                                 const unsigned int                             i_vec_halves,
                                                                 const unsigned int                             i_vec_log2e,
                                                                 const unsigned int                             i_vec_expmask,
                                                                 const unsigned int                             i_vec_hi_bound,
                                                                 const unsigned int                             i_vec_lo_bound,
                                                                 const unsigned int                             i_gp_reg_tmp,
                                                                 const libxsmm_aarch64_asimd_tupletype          i_tupletype );
LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                                               const unsigned int                             i_vec_c0,
                                                               const unsigned int                             i_vec_c1,
                                                               const unsigned int                             i_vec_c2,
                                                               const unsigned int                             i_vec_c3,
                                                               const unsigned int                             i_vec_halves,
                                                               const unsigned int                             i_vec_log2e,
                                                               const unsigned int                             i_vec_expmask,
                                                               const unsigned int                             i_vec_hi_bound,
                                                               const unsigned int                             i_vec_lo_bound,
                                                               const unsigned int                             i_gp_reg_tmp,
                                                               const libxsmm_aarch64_sve_type                 i_sve_type,
                                                               const unsigned char                            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                  const unsigned int                             i_vec_x,
                                                  const unsigned int                             i_vec_y,
                                                  const unsigned int                             i_vec_z,
                                                  const unsigned int                             i_vec_c0,
                                                  const unsigned int                             i_vec_c1,
                                                  const unsigned int                             i_vec_c2,
                                                  const unsigned int                             i_vec_c3,
                                                  const unsigned int                             i_vec_halves,
                                                  const unsigned int                             i_vec_log2e,
                                                  const unsigned int                             i_vec_expmask,
                                                  const unsigned int                             i_vec_hi_bound,
                                                  const unsigned int                             i_vec_lo_bound,
                                                  const libxsmm_aarch64_asimd_tupletype          i_tupletype );
LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                                const unsigned int                             i_vec_x,
                                                const unsigned int                             i_vec_y,
                                                const unsigned int                             i_vec_z,
                                                const unsigned int                             i_vec_c0,
                                                const unsigned int                             i_vec_c1,
                                                const unsigned int                             i_vec_c2,
                                                const unsigned int                             i_vec_c3,
                                                const unsigned int                             i_vec_halves,
                                                const unsigned int                             i_vec_log2e,
                                                const unsigned int                             i_vec_expmask,
                                                const unsigned int                             i_vec_hi_bound,
                                                const unsigned int                             i_vec_lo_bound,
                                                const libxsmm_aarch64_sve_type                 i_sve_type,
                                                const unsigned char                            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                               const unsigned int                             i_vec_x,
                                               const unsigned int                             i_vec_y,
                                               const unsigned int                             i_vec_z,
                                               const unsigned int                             i_vec_expmask,
                                               const libxsmm_aarch64_asimd_tupletype          i_tupletype );

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                             const unsigned int                             i_vec_x,
                                             const unsigned int                             i_vec_y,
                                             const unsigned int                             i_vec_z,
                                             const unsigned int                             i_vec_expmask,
                                             const libxsmm_aarch64_sve_type                 i_sve_type,
                                             const unsigned char                            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                                         const unsigned int                             i_vec_c0,
                                                                         const unsigned int                             i_vec_c1,
                                                                         const unsigned int                             i_vec_c2,
                                                                         const unsigned int                             i_vec_c3,
                                                                         const unsigned int                             i_vec_c1_d,
                                                                         const unsigned int                             i_vec_c2_d,
                                                                         const unsigned int                             i_vec_c3_d,
                                                                         const unsigned int                             i_vec_hi_bound,
                                                                         const unsigned int                             i_vec_lo_bound,
                                                                         const unsigned int                             i_vec_ones,
                                                                         const unsigned int                             i_vec_neg_ones,
                                                                         const unsigned int                             i_gp_reg_tmp,
                                                                         const libxsmm_aarch64_asimd_tupletype          i_tupletype );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                                                       const unsigned int                             i_vec_c0,
                                                                       const unsigned int                             i_vec_c1,
                                                                       const unsigned int                             i_vec_c2,
                                                                       const unsigned int                             i_vec_c3,
                                                                       const unsigned int                             i_vec_c1_d,
                                                                       const unsigned int                             i_vec_c2_d,
                                                                       const unsigned int                             i_vec_c3_d,
                                                                       const unsigned int                             i_vec_hi_bound,
                                                                       const unsigned int                             i_vec_lo_bound,
                                                                       const unsigned int                             i_vec_ones,
                                                                       const unsigned int                             i_vec_neg_ones,
                                                                       const unsigned int                             i_gp_reg_tmp,
                                                                       const libxsmm_aarch64_sve_type                 i_sve_type,
                                                                       const unsigned char                            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                                            const unsigned int                             i_vec_c0,
                                                                            const unsigned int                             i_vec_c1,
                                                                            const unsigned int                             i_vec_c2,
                                                                            const unsigned int                             i_vec_c3,
                                                                            const unsigned int                             i_vec_c1_d,
                                                                            const unsigned int                             i_vec_c2_d,
                                                                            const unsigned int                             i_vec_c3_d,
                                                                            const unsigned int                             i_vec_hi_bound,
                                                                            const unsigned int                             i_vec_lo_bound,
                                                                            const unsigned int                             i_vec_ones,
                                                                            const unsigned int                             i_vec_neg_ones,
                                                                            const unsigned int                             i_vec_halves,
                                                                            const unsigned int                             i_gp_reg_tmp,
                                                                            const libxsmm_aarch64_asimd_tupletype          i_tupletype );
LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                                                          const unsigned int                             i_vec_c0,
                                                                          const unsigned int                             i_vec_c1,
                                                                          const unsigned int                             i_vec_c2,
                                                                          const unsigned int                             i_vec_c3,
                                                                          const unsigned int                             i_vec_c1_d,
                                                                          const unsigned int                             i_vec_c2_d,
                                                                          const unsigned int                             i_vec_c3_d,
                                                                          const unsigned int                             i_vec_hi_bound,
                                                                          const unsigned int                             i_vec_lo_bound,
                                                                          const unsigned int                             i_vec_ones,
                                                                          const unsigned int                             i_vec_neg_ones,
                                                                          const unsigned int                             i_vec_halves,
                                                                          const unsigned int                             i_gp_reg_tmp,
                                                                          const libxsmm_aarch64_sve_type                 i_sve_type,
                                                                          const unsigned char                            i_pred_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_tanh_ps_rational_78_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                          const unsigned int                             i_vec_x,
                                                          const unsigned int                             i_vec_x2,
                                                          const unsigned int                             i_vec_nom,
                                                          const unsigned int                             i_vec_denom,
                                                          const unsigned int                             i_mask_hi,
                                                          const unsigned int                             i_mask_lo,
                                                          const unsigned int                             i_vec_c0,
                                                          const unsigned int                             i_vec_c1,
                                                          const unsigned int                             i_vec_c2,
                                                          const unsigned int                             i_vec_c3,
                                                          const unsigned int                             i_vec_c1_d,
                                                          const unsigned int                             i_vec_c2_d,
                                                          const unsigned int                             i_vec_c3_d,
                                                          const unsigned int                             i_vec_hi_bound,
                                                          const unsigned int                             i_vec_lo_bound,
                                                          const unsigned int                             i_vec_ones,
                                                          const unsigned int                             i_vec_neg_ones,
                                                          const unsigned int                             i_vec_tmp,
                                                          const libxsmm_aarch64_asimd_tupletype          i_tupletype );
LIBXSMM_API_INTERN
void libxsmm_generator_tanh_ps_rational_78_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                                        const unsigned int                             i_vec_x,
                                                        const unsigned int                             i_vec_x2,
                                                        const unsigned int                             i_vec_nom,
                                                        const unsigned int                             i_vec_denom,
                                                        const unsigned int                             i_mask_hi,
                                                        const unsigned int                             i_mask_lo,
                                                        const unsigned int                             i_vec_c0,
                                                        const unsigned int                             i_vec_c1,
                                                        const unsigned int                             i_vec_c2,
                                                        const unsigned int                             i_vec_c3,
                                                        const unsigned int                             i_vec_c1_d,
                                                        const unsigned int                             i_vec_c2_d,
                                                        const unsigned int                             i_vec_c3_d,
                                                        const unsigned int                             i_vec_hi_bound,
                                                        const unsigned int                             i_vec_lo_bound,
                                                        const unsigned int                             i_vec_ones,
                                                        const unsigned int                             i_vec_neg_ones,
                                                        const unsigned int                             i_vec_tmp,
                                                        const libxsmm_aarch64_sve_type                 i_sve_type,
                                                        const unsigned char                            i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_sigmoid_ps_rational_78_aarch64_asimd( libxsmm_generated_code*                        io_generated_code,
                                                             const unsigned int                             i_vec_x,
                                                             const unsigned int                             i_vec_x2,
                                                             const unsigned int                             i_vec_nom,
                                                             const unsigned int                             i_vec_denom,
                                                             const unsigned int                             i_mask_hi,
                                                             const unsigned int                             i_mask_lo,
                                                             const unsigned int                             i_vec_c0,
                                                             const unsigned int                             i_vec_c1,
                                                             const unsigned int                             i_vec_c2,
                                                             const unsigned int                             i_vec_c3,
                                                             const unsigned int                             i_vec_c1_d,
                                                             const unsigned int                             i_vec_c2_d,
                                                             const unsigned int                             i_vec_c3_d,
                                                             const unsigned int                             i_vec_hi_bound,
                                                             const unsigned int                             i_vec_lo_bound,
                                                             const unsigned int                             i_vec_ones,
                                                             const unsigned int                             i_vec_neg_ones,
                                                             const unsigned int                             i_vec_halves,
                                                             const unsigned int                             i_vec_tmp,
                                                             const libxsmm_aarch64_asimd_tupletype          i_tupletype );
LIBXSMM_API_INTERN
void libxsmm_generator_sigmoid_ps_rational_78_aarch64_sve( libxsmm_generated_code*                        io_generated_code,
                                                           const unsigned int                             i_vec_x,
                                                           const unsigned int                             i_vec_x2,
                                                           const unsigned int                             i_vec_nom,
                                                           const unsigned int                             i_vec_denom,
                                                           const unsigned int                             i_mask_hi,
                                                           const unsigned int                             i_mask_lo,
                                                           const unsigned int                             i_vec_c0,
                                                           const unsigned int                             i_vec_c1,
                                                           const unsigned int                             i_vec_c2,
                                                           const unsigned int                             i_vec_c3,
                                                           const unsigned int                             i_vec_c1_d,
                                                           const unsigned int                             i_vec_c2_d,
                                                           const unsigned int                             i_vec_c3_d,
                                                           const unsigned int                             i_vec_hi_bound,
                                                           const unsigned int                             i_vec_lo_bound,
                                                           const unsigned int                             i_vec_ones,
                                                           const unsigned int                             i_vec_neg_ones,
                                                           const unsigned int                             i_vec_halves,
                                                           const unsigned int                             i_vec_tmp,
                                                           const libxsmm_aarch64_sve_type                 i_sve_type,
                                                           const unsigned char                            i_pred_reg );

/**
 * Load constants in the ASIMD register
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_reg ASIMD register
 * @param i_gp_reg_tmp the gp register which contains the comperitor
 * @param i_tupletype input datatype
 * @param imm64 constant value to land in ASIMD reg
*/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( libxsmm_generated_code*               io_generated_code,
                                                                 const unsigned char                   i_vec_reg,
                                                                 const unsigned int                    i_gp_reg_tmp,
                                                                 const libxsmm_aarch64_asimd_tupletype i_tupletype,
                                                                 unsigned long long                    imm64 );
/**
 * Load constants in the SVE register
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_reg SVE register
 * @param i_gp_reg_tmp the gp register which contains the comperitor
 * @param i_sve_type input datatype
 * @param imm64 constant value to land in SVE reg
*/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( libxsmm_generated_code*               io_generated_code,
                                                               const unsigned char                   i_vec_reg,
                                                               const unsigned int                    i_gp_reg_tmp,
                                                               const libxsmm_aarch64_sve_type        i_sve_type,
                                                               const unsigned char                   i_pred_reg,
                                                               unsigned long long                    imm64 );

/**
 * Load constants in the ASIMD register
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_reg ASIMD register
 * @param i_gp_reg_tmp0 the gp register which contains the comperitor
*/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_load16bytes_const_to_vec( libxsmm_generated_code* io_generated_code,
                                                           const unsigned char     i_vec_reg,
                                                           const unsigned int      i_gp_reg_tmp0,
                                                           const unsigned int      i_gp_reg_tmp1,
                                                           void*                   imm64_array,
                                                           const unsigned int      i_start_index);

/**
 * Load constants in an SVE register with 128-bit replication
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_reg SVE register to fill
 * @param i_gp_reg_tmp0 first temp register
 * @param i_gp_reg_tmp1 second temp register
 * @param i_pred_tmp predicate temp register
 * @param imm64_array array containing data to load
 * @param i_start_index starting offset in the array
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_rep16bytes_const_to_vec( libxsmm_generated_code* io_generated_code,
                                                              const unsigned int      i_vec_reg,
                                                              const unsigned int      i_gp_reg_tmp0,
                                                              const unsigned int      i_gp_reg_tmp1,
                                                              const unsigned int      i_pred_tmp,
                                                              void*                   imm64_array,
                                                              const unsigned int      i_start_index);

/**
 * Load constants in a SVE register
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_reg SVE register to fill
 * @param i_gp_reg_tmp0 first temp register
 * @param i_gp_reg_tmp1 second temp register
 * @param i_pred_tmp predicate temp register
 * @param imm64_array array containing data to load
 * @param i_bytes number of bytes to load to the vector register, must be a multiple of 16
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( libxsmm_generated_code* io_generated_code,
                                                             const unsigned char     i_vec_reg,
                                                             const unsigned int      i_gp_reg_tmp0,
                                                             const unsigned int      i_gp_reg_tmp1,
                                                             const unsigned int      i_pred_tmp,
                                                             void*                   imm64_array,
                                                             const unsigned int      i_bytes);


/**
 * Copies data from one pointer to another.
 * Resets both source and destination pointer after the copy has completed.
 *
 * @param io_generated_code current program
 * @param i_gp_reg_src source pointer register
 * @param i_gp_reg_dst destination pointer register
 * @param i_gp_reg_tmp a temporary general-purpose register
 * @param i_vec_reg_tmp a temporary vector register
 * @param i_pred_reg_tmp a temporary predicate register
 * @param i_element_count how many elements are to be copied
 * @param i_sve_type the type of element to be copied (defines amount of bytes/element)
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_memcpy( libxsmm_generated_code*        io_generated_code,
                                             const unsigned char            i_gp_reg_src,
                                             const unsigned char            i_gp_reg_dst,
                                             const unsigned char            i_gp_reg_tmp,
                                             const unsigned char            i_vec_reg_tmp,
                                             const unsigned char            i_pred_reg_tmp,
                                             const unsigned int             i_element_count,
                                             const libxsmm_aarch64_sve_type i_sve_type );

#endif /* GENERATOR_COMMON_AARCH64_H */
