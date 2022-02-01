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

#ifndef GENERATOR_COMMON_X86_H
#define GENERATOR_COMMON_X86_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_header_no_idx_inc( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_init_val);

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_footer_with_idx_inc( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_step,
    const unsigned int                 i_loop_bound);

LIBXSMM_API_INTERN
void libxsmm_generator_x86_save_gpr_regs(libxsmm_generated_code*   io_generated_code,
    const unsigned short    i_save_bitmask);

LIBXSMM_API_INTERN
void libxsmm_generator_x86_restore_gpr_regs(libxsmm_generated_code*   io_generated_code,
    const unsigned short    i_restore_bitmask);

LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_avx( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   instr,
    const unsigned int                             i_vec_inout,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2);

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vpermd_16way_avx2( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_index,
    const unsigned int                             i_vec_c_lo,
    const unsigned int                             i_vec_c_hi,
    const unsigned int                             i_vec_result,
    const unsigned int                             i_vec_tmp0,
    const unsigned int                             i_vec_tmp1 );

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    const unsigned int                             i_vec_tmp0,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2,
    const unsigned int                             i_vec_tmp3,
    const unsigned int                             i_vec_tmp4,
    const unsigned int                             i_vec_tmp5,
    const unsigned int                             i_vec_tmp6,
    const unsigned int                             i_vec_tmp7,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   i_gp_reg_tmp,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half );

LIBXSMM_API_INTERN
void libxsmm_generator_sigmoid_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_x2,
    const unsigned int                             i_vec_nom,
    const unsigned int                             i_vec_denom,
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
    const unsigned int                             i_vec_neg_ones);

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones);

LIBXSMM_API_INTERN
void libxsmm_generator_tanh_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_x2,
    const unsigned int                             i_vec_nom,
    const unsigned int                             i_vec_denom,
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
    const unsigned int                             i_vec_neg_ones );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones );

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_y,
    const unsigned int                             i_vec_z,
    const unsigned int                             i_vec_expmask );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e,
    const unsigned int                             i_vec_expmask,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound );

LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_avx( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_lo_bound );

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_header( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_init_val,
    const unsigned int                 i_loop_step);

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_footer( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_bound);

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e );

LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_y,
    const unsigned int                             i_vec_z,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2 );


LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   i_gp_reg_tmp,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half );

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_c2 );

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    const unsigned int                             i_vec_tmp0,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2,
    const unsigned int                             i_vec_tmp3,
    const unsigned int                             i_vec_tmp4,
    const unsigned int                             i_vec_tmp5,
    const unsigned int                             i_vec_tmp6,
    const unsigned int                             i_vec_tmp7,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2 );

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_c2 );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones );

LIBXSMM_API_INTERN
void libxsmm_generator_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones);

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_halves );

LIBXSMM_API_INTERN
void libxsmm_generator_load_prng_state_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                   const unsigned char     i_vname,
                                                   const unsigned int      i_gp_reg_prng_state_ptr,
                                                   const unsigned int      prng_state0_vreg,
                                                   const unsigned int      prng_state1_vreg,
                                                   const unsigned int      prng_state2_vreg,
                                                   const unsigned int      prng_state3_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_store_prng_state_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                    const unsigned char     i_vname,
                                                    const unsigned int      i_gp_reg_prng_state_ptr,
                                                    const unsigned int      prng_state0_vreg,
                                                    const unsigned int      prng_state1_vreg,
                                                    const unsigned int      prng_state2_vreg,
                                                    const unsigned int      prng_state3_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_load_vreg_minus_infinity(libxsmm_generated_code* io_generated_code,
                                                   const unsigned char     i_vname,
                                                   const unsigned int      i_gp_reg_tmp,
                                                   const unsigned int      i_vreg_minus_infinity);

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                   const unsigned char     i_vname,
                                                   const unsigned int      i_gp_reg_tmp,
                                                   const unsigned int      i_gp_reg_prob_ptr,
                                                   const unsigned int      dropout_vreg_one,
                                                   const unsigned int      dropout_prob_vreg,
                                                   const unsigned int      dropout_invprob_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_inv_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                       const unsigned char     i_vname,
                                                       const unsigned int      i_gp_reg_tmp,
                                                       const unsigned int      i_gp_reg_prob_ptr,
                                                       const unsigned int      dropout_vreg_one,
                                                       const unsigned int      dropout_vreg_zero,
                                                       const unsigned int      dropout_prob_vreg );

LIBXSMM_API_INTERN
void libxsmm_generator_sigmoid_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_halves );

LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_avx512( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   instr,
    const unsigned int                             i_vec_inout,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2);

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128p_f32_avx2_avx512( libxsmm_generated_code* io_generated_code,
                                                    const unsigned char     i_vname,
                                                    const unsigned int      i_vec_reg_rng_state_0,
                                                    const unsigned int      i_vec_reg_rng_state_1,
                                                    const unsigned int      i_vec_reg_rng_state_2,
                                                    const unsigned int      i_vec_reg_rng_state_3,
                                                    const unsigned int      i_vec_reg_rng_tmp_0,
                                                    const unsigned int      i_vec_reg_rng_tmp_1,
                                                    const unsigned int      i_vec_reg_rng_one,
                                                    const unsigned int      o_vec_reg_rng );

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128pp_axv2_avx512( libxsmm_generated_code* io_generated_code,
                                                 const unsigned char     i_vname,
                                                 const unsigned int      i_vec_reg_rng_state_0,
                                                 const unsigned int      i_vec_reg_rng_state_1,
                                                 const unsigned int      i_vec_reg_rng_state_2,
                                                 const unsigned int      i_vec_reg_rng_state_3,
                                                 const unsigned int      i_vec_reg_rng_tmp_0,
                                                 const unsigned int      i_vec_reg_rng_tmp_1,
                                                 const unsigned int      o_vec_reg_rng );

LIBXSMM_API_INTERN
void libxsmm_generator_cvtbf16ps_avx512( libxsmm_generated_code* io_generated_code,
                                         const char              i_vname,
                                         const unsigned int      i_vec_reg,
                                         const unsigned int      o_vec_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      io_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( libxsmm_generated_code* io_generated_code,
                                                         const unsigned int      io_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( libxsmm_generated_code* io_generated_code,
                                                          const char              i_vname,
                                                          const unsigned int      i_vec_reg,
                                                          const unsigned int      o_vec_teg,
                                                          const unsigned int      io_vec_tmp_0,
                                                          const unsigned int      io_vec_tmp_1,
                                                          const unsigned int      io_mask_0,
                                                          const unsigned int      io_mask_1 );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_preppedstack_nocompact( libxsmm_generated_code* io_generated_code,
                                                          const char              i_vname,
                                                          const unsigned int      i_vec_reg,
                                                          const unsigned int      o_vec_teg,
                                                          const unsigned int      io_vec_tmp_0,
                                                          const unsigned int      io_vec_tmp_1,
                                                          const unsigned int      io_mask_0,
                                                          const unsigned int      io_mask_1 );
LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512( libxsmm_generated_code* io_generated_code,
                                             const char              i_vname,
                                             const unsigned int      i_vec_reg,
                                             const unsigned int      o_vec_teg,
                                             const unsigned int      io_gp_reg,
                                             const unsigned int      io_vec_tmp_0,
                                             const unsigned int      io_vec_tmp_1,
                                             const unsigned int      io_mask_0,
                                             const unsigned int      io_mask_1 );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2int_avx512( libxsmm_generated_code* io_generated_code,
                                            const libxsmm_datatype  i_datatype,
                                            const unsigned int      io_vec_reg,
                                            const unsigned int      i_scf_vec_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtint2ps_avx512( libxsmm_generated_code* io_generated_code,
                                          const libxsmm_datatype  i_datatype,
                                          const unsigned int      io_vec_reg,
                                          const unsigned int      i_scf_vec_reg );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_unified_vec_move( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_vmove_instr,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const char              i_vector_name,
                                                const unsigned int      i_vec_reg_number_0,
                                                const unsigned int      i_use_masking,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      i_is_store );

#endif /* GENERATOR_COMMON_X86_H */

