/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer, Antonio Noack (FSU Jena), Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_gemm_common_aarch64.h"


LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_bf16f32_aarch64_sve( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg) {
  LIBXSMM_UNUSED( i_pred_reg );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(4) );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V, i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 16, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(4) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32bf16_aarch64_sve( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg) {
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_BFCVT_V_P, i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout, i_pred_reg, libxsmm_generator_aarch64_get_sve_type(4) );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, i_vec_inout, i_vec_inout, 0, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(2) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_bf16f32_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg) {
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    /* TODO */
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, i_vec_inout, i_pred_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32bf16_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_pred_reg) {
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    /* TODO */
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, i_vec_inout, i_pred_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32i8_aarch64_sve( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_scf_vec_reg,
    const unsigned int i_pred_reg,
    const unsigned int i_skip_scaling,
    const unsigned int i_sign_sat ) {
  if (i_skip_scaling == 0) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                             i_vec_inout, i_scf_vec_reg, 0, i_vec_inout, i_pred_reg, libxsmm_generator_aarch64_get_sve_type(4) );
  }
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRINTI_V_P,
                                           i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout, i_pred_reg, libxsmm_generator_aarch64_get_sve_type(4) );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCVTZS_V_P_SS,
                                           i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout,
                                           i_pred_reg, libxsmm_generator_aarch64_get_sve_type(4) );
  if (i_sign_sat > 0) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SMAX_V_I,
                                            i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, -128, i_vec_inout,
                                            0, libxsmm_generator_aarch64_get_sve_type(4) );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SMIN_V_I,
                                            i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 127, i_vec_inout,
                                            0, libxsmm_generator_aarch64_get_sve_type(4) );
  }
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, i_vec_inout, i_vec_inout, 0, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(2) );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, i_vec_inout, i_vec_inout, 0, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(1) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_i8f32_aarch64_sve( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_scf_vec_reg,
    const unsigned int i_pred_reg,
    unsigned int       i_skip_scaling ) {
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(2) );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout, 0, libxsmm_generator_aarch64_get_sve_type(4) );
  /* Convert to ps  */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
                                           i_vec_inout, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_inout,
                                           i_pred_reg, libxsmm_generator_aarch64_get_sve_type(4) );
  /* Scale  */
  if (i_skip_scaling == 0) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                             i_vec_inout, i_scf_vec_reg, 0, i_vec_inout, i_pred_reg, libxsmm_generator_aarch64_get_sve_type(4) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_f32i8_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_scf_vec_reg,
    const unsigned int i_pred_reg,
    const unsigned int i_skip_scaling,
    const unsigned int i_sign_sat ) {
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    /* TODO  */
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_vcvt_f32i8_aarch64_sve( io_generated_code, i_vec_inout, i_scf_vec_reg, i_pred_reg, i_skip_scaling, i_sign_sat );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvt_i8f32_aarch64( libxsmm_generated_code* io_generated_code,
    const unsigned int i_vec_inout,
    const unsigned int i_scf_vec_reg,
    const unsigned int i_pred_reg,
    unsigned int       i_skip_scaling ) {
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 || io_generated_code->arch == LIBXSMM_AARCH64_V82 || io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    /* TODO  */
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_vcvt_i8f32_aarch64_sve( io_generated_code, i_vec_inout, i_scf_vec_reg, i_pred_reg, i_skip_scaling );
  }
}

/** Executes the asimd compute instruction <instr> twice. The register i_vec_inout is set as src0,src1 and dst; the element size is 4S */
LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_aarch64( libxsmm_generated_code* io_generated_code,
    unsigned int                                   instr,
    const unsigned int                             i_vec_inout,
    libxsmm_aarch64_asimd_tupletype                i_tuple_type) {
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, instr, i_vec_inout, i_vec_inout, 0, i_vec_inout, i_tuple_type );
  if (i_tuple_type == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S) {
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, instr, i_vec_inout, i_vec_inout, 0, i_vec_inout, i_tuple_type );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_set_p_register_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                   unsigned int            i_p_reg,
                                                   int                     i_n_bits,
                                                   unsigned int            i_gp_reg_scratch ) {
  if ( i_n_bits < 0 ) {
    libxsmm_aarch64_instruction_sve_pcompute( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_PTRUE,
                                              i_p_reg,
                                              LIBXSMM_AARCH64_ASIMD_REG_UNDEF,
                                              LIBXSMM_AARCH64_GP_WIDTH_W,
                                              LIBXSMM_AARCH64_ASIMD_REG_UNDEF,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                              LIBXSMM_AARCH64_SVE_TYPE_B );
  }
  else {
    /* store number of bits in gp register */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                               i_gp_reg_scratch,
                                               i_n_bits );

    libxsmm_aarch64_instruction_sve_pcompute( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_WHILELT,
                                              i_p_reg,
                                              LIBXSMM_AARCH64_GP_REG_XZR,
                                              LIBXSMM_AARCH64_GP_WIDTH_X,
                                              i_gp_reg_scratch,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                              LIBXSMM_AARCH64_SVE_TYPE_B );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mov_aarch64( libxsmm_generated_code* io_generated_code,
                                    unsigned int            i_src_reg,
                                    unsigned int            i_dst_reg )
{
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ORR_SR,
                                                       LIBXSMM_AARCH64_GP_REG_XZR,
                                                       i_src_reg, i_dst_reg,
                                                       0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
}

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_trips ) {
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_loop_cnt, i_trips );
  libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_loop_footer_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_loop_blocking ) {
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 i_gp_reg_loop_cnt, i_gp_reg_loop_cnt, i_loop_blocking, 0 );
  libxsmm_aarch64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBNZ,
                                                       i_gp_reg_loop_cnt, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_gp_reg_bound_aarch64( libxsmm_generated_code*     io_generated_code,
                                                         libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                         const unsigned int          i_gp_reg_loop_cnt,
                                                         const unsigned int          i_gp_reg_bound ) {
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, i_gp_reg_bound, i_gp_reg_bound, i_gp_reg_loop_cnt, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_gp_reg_addr,
                                                             const unsigned int      i_gp_reg_scratch,
                                                             const unsigned int      i_vec_reg,
                                                             const unsigned int      i_datatype_size,
                                                             const unsigned int      i_masked_elems,
                                                             const unsigned int      i_adv_gpr,
                                                             const unsigned int      i_is_store ) {
  libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_addr, i_gp_reg_scratch,
                                                    i_vec_reg, i_datatype_size, i_masked_elems, i_adv_gpr, i_is_store,
                                                    i_masked_elems == 0 ? 0 : 1 );
}

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
                                                          const unsigned int      i_is_gather ) {
  unsigned int i = 0;
  unsigned int l_load_move_instr = (i_datatype_size == 4) ? LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF : LIBXSMM_AARCH64_INSTR_GP_LDRH_I_OFF;
  unsigned int l_store_move_instr = (i_datatype_size == 4) ? LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF : LIBXSMM_AARCH64_INSTR_GP_STRH_I_OFF;
  unsigned int l_shift_amount = (i_datatype_size == 4) ? 2 : 1;
  libxsmm_aarch64_asimd_width l_idx_move_asimd_width = (i_idx_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D;
  libxsmm_aarch64_asimd_width l_data_move_asimd_width = (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_H;
  unsigned int l_active_elements = (i_masked_elems == 0) ? libxsmm_cpuid_vlen(io_generated_code->arch)/i_idx_datatype_size : i_masked_elems;

  for (i = 0; i < l_active_elements; i++) {
    /* Move index i from index asimd reg to GPR0 */
    libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G, i_gp_reg_scratch0, i_idx_vec_reg, LIBXSMM_CAST_USHORT(i), l_idx_move_asimd_width);

    /* Add (index * elem_size) to base_addr -> GPR0 */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, i_gp_reg_addr, i_gp_reg_scratch0, i_gp_reg_scratch0, l_shift_amount, LIBXSMM_AARCH64_SHIFTMODE_LSL );

    if (i_is_gather > 0) {
      /* Load element to GPR1 from GPR0 */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, l_load_move_instr, i_gp_reg_scratch0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0x1f & i_gp_reg_scratch1 );

      /* Move GPR1 to position i of dest asimd reg */
      libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V, i_gp_reg_scratch1, i_srcdst_vec_reg, LIBXSMM_CAST_USHORT(i), l_data_move_asimd_width);
    } else {
      /* Move position i of src asimd reg to GPR1 */
      libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G, i_gp_reg_scratch1, i_srcdst_vec_reg, LIBXSMM_CAST_USHORT(i), l_data_move_asimd_width);

      /* Store element from GPR1 to GPR0 */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, l_store_move_instr, i_gp_reg_scratch0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, 0x1f & i_gp_reg_scratch1 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vloadstore_masked_vreg_aarch64( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_addr,
                                                       const unsigned int      i_gp_reg_scratch,
                                                       const unsigned int      i_vec_reg,
                                                       const unsigned int      i_datatype_size,
                                                       const unsigned int      i_masked_elems,
                                                       const unsigned int      i_adv_gpr,
                                                       const unsigned int      i_is_store,
                                                       const unsigned char     i_mask_reg ) {

  /* i_masked_elems: 0 = load as many as you can; else: load <i_masked_elems> elements */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
      (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT)  ) {
    if ( i_masked_elems == 0 ) {/* just load/store without predicate */
      unsigned int l_instr = i_is_store ? LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF;
      libxsmm_aarch64_instruction_sve_move( io_generated_code, l_instr, i_gp_reg_addr, 0, 0, i_vec_reg, i_mask_reg );
    } else {
      unsigned int l_store_instr = (i_datatype_size == 4) ? LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF : ((i_datatype_size == 8) ? LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF : ((i_datatype_size == 2) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_ST1B_I_OFF));
      unsigned int l_load_instr  = (i_datatype_size == 4) ? LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF : ((i_datatype_size == 8) ? LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF : ((i_datatype_size == 2) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF));
      unsigned int l_instr = i_is_store ? l_store_instr : l_load_instr;
      libxsmm_aarch64_instruction_sve_move( io_generated_code, l_instr, i_gp_reg_addr, 0, 0, i_vec_reg, i_mask_reg );
    }
    /* increment register if needed; add via immediate */
    if ( i_adv_gpr ) {
      unsigned int l_sve_length = libxsmm_cpuid_vlen( io_generated_code->arch ); /* 512 bits = 64 bytes for the A64FX */
      unsigned int l_offset = i_datatype_size * (i_masked_elems ? i_masked_elems : l_sve_length / i_datatype_size);
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, i_gp_reg_addr, i_gp_reg_addr, l_offset, 0 );
    }
    return;
  } else {/* ASIMD with 1-4 fp32 elements */
    unsigned int l_gpload_instr  = (  i_adv_gpr == 0 ) ? LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF : LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST;
    unsigned int l_gpstore_instr = (  i_adv_gpr == 0 ) ? LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF : LIBXSMM_AARCH64_INSTR_GP_STR_I_POST;
    unsigned int l_gploadh_instr = (  i_adv_gpr == 0 ) ? LIBXSMM_AARCH64_INSTR_GP_LDRH_I_OFF : LIBXSMM_AARCH64_INSTR_GP_LDRH_I_POST;
    unsigned int l_gpstoreh_instr= (  i_adv_gpr == 0 ) ? LIBXSMM_AARCH64_INSTR_GP_STRH_I_OFF : LIBXSMM_AARCH64_INSTR_GP_STRH_I_POST;
    unsigned int l_vload_instr   = (  i_adv_gpr == 0 ) ? LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF : LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST;
    unsigned int l_vstore_instr  = (  i_adv_gpr == 0 ) ? LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF : LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST;
    unsigned int l_vmove_instr   = ( i_is_store == 0 ) ? l_vload_instr : l_vstore_instr;

    switch(i_masked_elems){
      case 1: {
        const unsigned char l_offset = LIBXSMM_CAST_UCHAR(i_adv_gpr == 0 ? 0 : i_datatype_size);
        libxsmm_aarch64_asimd_width l_move_asimd_width = (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S : ( (i_datatype_size == 2) ? LIBXSMM_AARCH64_ASIMD_WIDTH_H: LIBXSMM_AARCH64_ASIMD_WIDTH_D);

        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        break;
      }
      case 2: {
        const unsigned char l_offset = ( i_adv_gpr == 0 ? 0 : LIBXSMM_CAST_UCHAR(i_datatype_size * 2) );
        libxsmm_aarch64_asimd_width l_move_asimd_width = (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : ( (i_datatype_size == 2) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S: LIBXSMM_AARCH64_ASIMD_WIDTH_Q);
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        break;
      }
      case 3: {
        const unsigned char l_offset  = ( i_adv_gpr == 0 ? 0 : LIBXSMM_CAST_UCHAR(i_datatype_size * 2) );
        const unsigned char l_offset2 = ( i_adv_gpr == 0 ? LIBXSMM_CAST_UCHAR(i_datatype_size * 2) : LIBXSMM_CAST_UCHAR(i_datatype_size) );
        libxsmm_aarch64_asimd_width l_move_asimd_width = (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : ( (i_datatype_size == 2) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S: LIBXSMM_AARCH64_ASIMD_WIDTH_Q);
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, (i_datatype_size == 4) ? l_gpload_instr : l_gploadh_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 2, (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_H);
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G,
                                                      i_gp_reg_scratch, i_vec_reg, 2, (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_H );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, (i_datatype_size == 4) ? l_gpstore_instr : l_gpstoreh_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
        }
        break;
      }
      case 4: {
        const unsigned char l_offset = LIBXSMM_CAST_UCHAR(i_adv_gpr == 0 ? 0 : i_datatype_size * 4);
        libxsmm_aarch64_asimd_width l_move_asimd_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;

        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        break;
      }
      case 5: {
        const unsigned char l_offset  = ( i_adv_gpr == 0 ? 0 : LIBXSMM_CAST_UCHAR(i_datatype_size * 4) );
        const unsigned char l_offset2 = ( i_adv_gpr == 0 ? LIBXSMM_CAST_UCHAR(i_datatype_size * 4) : LIBXSMM_CAST_UCHAR(i_datatype_size) );
        libxsmm_aarch64_asimd_width l_move_asimd_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gploadh_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 4, LIBXSMM_AARCH64_ASIMD_WIDTH_H);
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G,
                                                      i_gp_reg_scratch, i_vec_reg, 4, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gpstoreh_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
        }
        break;
      }
      case 6: {
        const unsigned char l_offset  = ( i_adv_gpr == 0 ? 0 : LIBXSMM_CAST_UCHAR(i_datatype_size * 4) );
        const unsigned char l_offset2 = ( i_adv_gpr == 0 ? LIBXSMM_CAST_UCHAR(i_datatype_size * 4) : LIBXSMM_CAST_UCHAR(i_datatype_size * 2) );
        libxsmm_aarch64_asimd_width l_move_asimd_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gpload_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 2, LIBXSMM_AARCH64_ASIMD_WIDTH_S);
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G,
                                                      i_gp_reg_scratch, i_vec_reg, 2, LIBXSMM_AARCH64_ASIMD_WIDTH_S );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gpstore_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
        }
        break;
      }
      case 7: {
        const unsigned char l_offset  = ( i_adv_gpr == 0 ? 0 : LIBXSMM_CAST_UCHAR(i_datatype_size * 4) );
        const unsigned char l_offset2 = ( i_adv_gpr == 0 ? LIBXSMM_CAST_UCHAR(i_datatype_size * 4) : LIBXSMM_CAST_UCHAR(i_datatype_size * 2) );
        const unsigned char l_offset3 = ( i_adv_gpr == 0 ? LIBXSMM_CAST_UCHAR(i_datatype_size * 6) : LIBXSMM_CAST_UCHAR(i_datatype_size) );
        libxsmm_aarch64_asimd_width l_move_asimd_width = LIBXSMM_AARCH64_ASIMD_WIDTH_D;
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                l_move_asimd_width );
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gpload_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 2, LIBXSMM_AARCH64_ASIMD_WIDTH_S);
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G,
                                                      i_gp_reg_scratch, i_vec_reg, 2, LIBXSMM_AARCH64_ASIMD_WIDTH_S );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gpstore_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset2,
                                                0x1f & i_gp_reg_scratch );
        }
        if ( i_is_store == 0 ) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gploadh_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset3,
                                                0x1f & i_gp_reg_scratch );
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 6, LIBXSMM_AARCH64_ASIMD_WIDTH_H);
        } else {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G,
                                                      i_gp_reg_scratch, i_vec_reg, 6, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, l_gpstoreh_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset3,
                                                0x1f & i_gp_reg_scratch );
        }
        break;
      }
      default: {
        unsigned char l_offset = ( i_adv_gpr == 0 ) ? 0 : 16;
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, l_vmove_instr,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                            const unsigned int      i_gp_reg_addr,
                                                            const unsigned int      i_gp_reg_scratch,
                                                            const unsigned int      i_vec_reg,
                                                            const unsigned int      i_datatype_size,
                                                            const unsigned int      i_masked_elems,
                                                            const unsigned int      i_adv_gpr ) {
  unsigned char l_offset = (unsigned char)(( i_adv_gpr == 0 ) ? 0 : i_datatype_size);
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);

  if ( l_is_sve ) {
    int l_pred_reg = 0; /* TODO: find suitable predicate register */
    /* different element sizes use different instructions; load a single element, broadcast it, and set the rest to zero */
    int l_instr = i_datatype_size == 1 ? LIBXSMM_AARCH64_INSTR_SVE_LD1RB_I_OFF :
                  i_datatype_size == 2 ? LIBXSMM_AARCH64_INSTR_SVE_LD1RH_I_OFF :
                  i_datatype_size == 4 ? LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF :
                                         LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF ;
     /* mark the predicate element to only load i_masked_elems elements, or all if -1 */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_pred_reg, i_masked_elems ? (int)(i_masked_elems * i_datatype_size) : -1, i_gp_reg_scratch );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, l_instr, i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_vec_reg, l_pred_reg );
    if ( l_offset ) {/* post increment address by offset */
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, i_gp_reg_addr, i_gp_reg_addr, l_offset, 0 );
    }
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_pred_reg, -1, i_gp_reg_scratch );
  } else {
    if ( i_masked_elems != 1 ) {
      if ( i_adv_gpr == 0 ) {
        libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                         i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, i_vec_reg,
                                                        (i_datatype_size == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else {
        libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                        i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_XZR, i_vec_reg,
                                                        (i_datatype_size == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
      if ( i_masked_elems != 0 ) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_scratch, 0x0 );
        /* todo: this is currently only implemented for fp32; the asimd_widths need to be adjusted for other types */
        if ( i_masked_elems == 2 ) {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 1, LIBXSMM_AARCH64_ASIMD_WIDTH_D );
        } else if ( i_masked_elems == 3 ) {
          libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                      i_gp_reg_scratch, i_vec_reg, 3, LIBXSMM_AARCH64_ASIMD_WIDTH_S );
        } else {
          /* should not happen */
        }
      }
    } else {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_vec_reg, i_vec_reg, 0, i_vec_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, l_offset, i_vec_reg,
                                              (i_datatype_size == 4) ? LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_addr,
                                                      const unsigned int      i_gp_reg_scratch,
                                                      const unsigned int      i_vec_length,
                                                      const unsigned int      i_vec_reg_count,
                                                      const unsigned int      i_m_blocking,
                                                      const unsigned int      i_n_blocking,
                                                      const unsigned int      i_ld,
                                                      const unsigned int      i_zero ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;
  unsigned int l_a_load_instruction = 0;
  unsigned int l_add_load = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);
  /* set load type */
  if( l_m_blocks[0] == 4 ){
    l_a_load_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LD1_4;
    l_add_load = 64;
  } else if( l_m_blocks[0] == 3 ){
    l_a_load_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LD1_3;
    l_add_load = 48;
  } else if( l_m_blocks[0] == 2 ){
    l_a_load_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LD1_2;
    l_add_load = 32;
  } else if( l_m_blocks[0] == 1 ){
    l_a_load_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_LD1_1;
    l_add_load = 16;
  }

  /* load C accumulator */
  if (i_zero == 0) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( l_m_blocks[0] > 0 ) {
        libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code,
                                                         l_a_load_instruction,
                                                         i_gp_reg_addr,
                                                         LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                         l_vec_reg_acc_start + (l_m_total_blocks * l_n),
                                                         LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_addr,
                                                     l_add_load,
                                                     0 );

      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_S );
      }
      if ( i_ld-l_m_bytes > 0 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                       ((long long)i_ld - l_m_bytes) );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                   (long long)i_ld*i_n_blocking );
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
    }
  }
}

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
                                                           const unsigned char     i_zip_row_major ) {
  unsigned int l_vec_reg_acc_start;
  unsigned int l_vec_reg_tmp[2];
  /* register blocking counter in n */
  unsigned int l_n = 0;
  unsigned int l_n_blocks = i_n_blocking / 2;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m iterations */
  unsigned int l_m_blocks[2];  /* 0: 128bit, 1: 64bit */
  unsigned int l_m_bytes = 0;
  unsigned int l_vr_c[24] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  /* derive zip instructions */
  unsigned int l_instr_zip[2] = { LIBXSMM_AARCH64_INSTR_ASIMD_ZIP1, LIBXSMM_AARCH64_INSTR_ASIMD_ZIP2 };
  libxsmm_aarch64_asimd_tupletype  l_type_zip = LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;

  LIBXSMM_UNUSED( i_micro_kernel_config );
  LIBXSMM_UNUSED( i_xgemm_desc );
  LIBXSMM_UNUSED( i_vec_reg_count );

  if ( i_zip_row_major ) {
    l_instr_zip[0] = LIBXSMM_AARCH64_INSTR_ASIMD_UZP1;
    l_instr_zip[1] = LIBXSMM_AARCH64_INSTR_ASIMD_UZP2;
    l_type_zip = LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  }

  /* derive register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8;

  /* start register of accumulator */
  l_vec_reg_acc_start = l_vr_c[0]; /*i_vec_reg_count - (i_n_blocking * l_m_total_blocks)*/;

  /* temporary vector registers used to load values to before zipping */
  l_vec_reg_tmp[0] = l_vec_reg_acc_start - 2;
  l_vec_reg_tmp[1] = l_vec_reg_acc_start - 1;

  /* load C accumulator */
  if ( i_zero == 0 ) {
    for ( l_n = 0; l_n < l_n_blocks; l_n++ ) {
      /* second address register for loads */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_addr,
                                                    i_gp_reg_scratch,
                                                    i_gp_reg_scratch,
                                                    i_ld );

      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        /* load first part */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                16,
                                                l_vec_reg_tmp[0],
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        /* load second part */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_scratch,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                16,
                                                l_vec_reg_tmp[1],
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        /* zip data to target vector registers */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   l_instr_zip[0],
                                                   l_vec_reg_tmp[0],
                                                   l_vec_reg_tmp[1],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   l_type_zip );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   l_instr_zip[1],
                                                   l_vec_reg_tmp[0],
                                                   l_vec_reg_tmp[1],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m+1],
                                                   l_type_zip );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        /* load first part */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                8,
                                                l_vec_reg_tmp[0],
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_D );
        /* load second part */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_scratch,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                8,
                                                l_vec_reg_tmp[1],
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_D );

        /* zip data to target vector register */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   l_instr_zip[0],
                                                   l_vec_reg_tmp[0],
                                                   l_vec_reg_tmp[1],
                                                   0,
                                                   l_vr_c[8*l_n + 2*(l_m+l_m_blocks[0])],
                                                   l_type_zip );
      }
      /* adjust address register for next two columns */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_addr,
                                                     2ull*i_ld-l_m_bytes );
    }
    /* reset address register */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_addr,
                                                   (long long)i_ld*i_n_blocking );
  } else {
    for ( l_n = 0; l_n < l_n_blocks; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vr_c[8*l_n + 2*(l_m_blocks[0]+l_m)],
                                                   l_vr_c[8*l_n + 2*(l_m_blocks[0]+l_m)],
                                                   0,
                                                   l_vr_c[8*l_n + 2*(l_m_blocks[0]+l_m)],
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vr_c[8*l_n + 2*(l_m_blocks[0]+l_m)+1],
                                                   l_vr_c[8*l_n + 2*(l_m_blocks[0]+l_m)+1],
                                                   0,
                                                   l_vr_c[8*l_n + 2*(l_m_blocks[0]+l_m)+1],
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
    }
  }
}

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
                                                    const unsigned int      i_zero ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector loads, 1: #predicated loads (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_m_bytes_full = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned int l_datatype_size = LIBXSMM_TYPESIZE(i_datatype);
  unsigned int l_full_vector_mask = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_SVE_REG_P2 : LIBXSMM_AARCH64_SVE_REG_UNDEF;
  unsigned int l_load_instr = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF;
  unsigned int l_masked_load_instr = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF;
  unsigned int l_partial_vector_mask = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_SVE_REG_P4 : LIBXSMM_AARCH64_SVE_REG_P1;

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * l_datatype_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* loads C accumulator from memory */
  if ( i_zero == 0 ) {
    /* this is the jump size to be performed after a n-block is complete */
    unsigned long long l_jump_block_n_last = 0;

    /* full vector loads */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* this is the jump size to be performed after a m-block is complete */
      unsigned long long l_jump_block_m_last = 0;

      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              l_load_instr,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                              l_full_vector_mask );
        if ( LIBXSMM_DATATYPE_BF16 == i_datatype ) {
          libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code,
                                                      l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                      LIBXSMM_AARCH64_SVE_REG_UNDEF );
        }
        /* increase pointer in m-dimension.
           but only if
             1) remainder follows
             or
             2) we are not at the end of the m-loop
        */
        if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
          libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                         i_gp_reg_addr,
                                                         i_gp_reg_addr,
                                                         i_vec_length * l_datatype_size,
                                                         0 );
        }
        /* combine the m-jump with the n one*/
        else {
          l_jump_block_m_last = (long long)i_vec_length * l_datatype_size;
        }
      }

      if ( l_m_blocks[1] != 0 ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              l_masked_load_instr,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                              l_partial_vector_mask );
        if ( LIBXSMM_DATATYPE_BF16 == i_datatype ) {
          libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code,
                                                      l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                                      LIBXSMM_AARCH64_SVE_REG_UNDEF );
        }
      }

      l_jump_block_m_last += (long long)i_ld - l_m_bytes_full;

      if ( l_n != i_n_blocking - 1 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr,
                                                       i_gp_reg_scratch,
                                                       i_gp_reg_addr,
                                                       l_jump_block_m_last );
      }
      else {
        l_jump_block_n_last = l_jump_block_m_last;
      }
    }

    /* reset C-ptr to original address */
    l_jump_block_n_last = (long long)i_ld * i_n_blocking - l_jump_block_n_last;
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_addr,
                                                   l_jump_block_n_last );
  }
  /* init C accumulator to zero */
  else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 (unsigned char)-1,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 LIBXSMM_AARCH64_SVE_TYPE_D );
      }
    }
  }
}

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
                                                         const unsigned char     i_zip_row_major ) {
  unsigned int l_vec_reg_acc_start;
  unsigned int l_vec_reg_tmp[2];
  /* register blocking counter in n */
  unsigned int l_n = 0;
  unsigned int l_n_blocks = (i_n_blocking + 1) / 2;

  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_m_blocks[2] = {0, 0}; /* 0: #full vector loads, 1: #predicated loads (0 or 1) */
  unsigned int l_m_bytes_full = 0;
  unsigned int l_remainder_size = 0;
  /* derive zip instructions */
  unsigned int l_instr_zip[2] = { LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V, LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V };
  libxsmm_aarch64_sve_type  l_type_zip = LIBXSMM_AARCH64_SVE_TYPE_D;
  /* vector registers holding C's values */
  unsigned int l_vr_c[24] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  unsigned int l_is_output_bf16 = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_is_i8f32_gemm  = ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_output_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P2;
  unsigned int l_is_colbias_bf16 = (i_micro_kernel_config->fused_bcolbias > 0) ? 1 : 0;
  unsigned int l_bias_tsize = (i_micro_kernel_config->fused_bcolbias > 0) ? 2 : 4;
  unsigned int l_gp_reg_bias = i_gp_reg_scratch1;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(sizeof(float)));

  LIBXSMM_UNUSED( i_vec_reg_count );
  LIBXSMM_UNUSED( i_zip_row_major );

  /* TODO (MMLA): implement */
  /* if ( i_zip_row_major ) {
    l_instr_zip[0] = LIBXSMM_AARCH64_INSTR_SVE_UZP1_V;
    l_instr_zip[1] = LIBXSMM_AARCH64_INSTR_SVE_UZP2_V;
    l_type_zip = LIBXSMM_AARCH64_SVE_TYPE_S;
  } */

  /* derive register blocking from kernel config */
  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * i_data_size + l_m_blocks[1] * (i_m_blocking % i_vec_length) * i_data_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = l_vr_c[0]; /*i_vec_reg_count - (i_n_blocking * l_m_total_blocks);*/

  /* temporary vector registers used to load values to before zipping */
  l_vec_reg_tmp[0] = l_vec_reg_acc_start - 2;
  l_vec_reg_tmp[1] = l_vec_reg_acc_start - 1;

  if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_bias);
  }

  /* load C accumulator */
  if ( (i_zero == 0) && (l_is_i8f32_gemm == 0)) {
    for ( l_n = 0; l_n < l_n_blocks; l_n++ ) {
      /* second address register for loads */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_scratch,
                                                     i_ld );

      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        /* Load column bias if need be */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          if (l_n == 0) {
            if (l_is_colbias_bf16 == 0) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  LIBXSMM_AARCH64_SVE_REG_UNDEF );
            } else {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  l_output_bf16_mask );
              libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_m, 0);
            }
            if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
              libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                             LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                             l_gp_reg_bias,
                                                             l_gp_reg_bias,
                                                             i_vec_length * l_bias_tsize,
                                                             0 );
            }
          }
        }

        /* load first part */
        if (l_is_output_bf16 == 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[0],
                                                LIBXSMM_AARCH64_SVE_REG_UNDEF );
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[0],
                                                l_output_bf16_mask );
          libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
        }

        /* Add column bias if requested */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
              l_vec_reg_tmp[0], l_m, 0, l_vec_reg_tmp[0], 0, l_sve_type );
        }

        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       i_gp_reg_addr,
                                                       i_gp_reg_addr,
                                                       i_vec_length * i_data_size,
                                                       0 );
        /* load second part */
        if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
              l_vec_reg_tmp[1], l_vec_reg_tmp[1], 0, l_vec_reg_tmp[1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          if (l_is_output_bf16 == 0) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                  i_gp_reg_scratch,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  LIBXSMM_AARCH64_SVE_REG_UNDEF );
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                                  i_gp_reg_scratch,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  l_output_bf16_mask );
            libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
          }
        }

        /* Add column bias if requested */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
              l_vec_reg_tmp[1], l_m, 0, l_vec_reg_tmp[1], 0, l_sve_type );
        }

        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       i_gp_reg_scratch,
                                                       i_gp_reg_scratch,
                                                       i_vec_length * i_data_size,
                                                       0 );

        /* zip data to target vector registers */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_zip[0],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vr_c[8*l_n + 2*l_m],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );

        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_zip[1],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vr_c[8*l_n + 2*l_m + 1],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );
      }
      for ( l_m = l_m_blocks[0]; l_m < l_m_blocks[0] + l_m_blocks[1]; l_m++ ) {
        /* Load column bias if need be */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          if (l_n == 0) {
            if (l_is_colbias_bf16 == 0) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  LIBXSMM_AARCH64_SVE_REG_P1 );
            } else {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  LIBXSMM_AARCH64_SVE_REG_P1 );
              libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_m, 0);
            }
          }
        }

        /* load first part */
        if (l_is_output_bf16 == 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[0],
                                                LIBXSMM_AARCH64_SVE_REG_P1 );
        } else {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[0],
                                                LIBXSMM_AARCH64_SVE_REG_P1 );
          libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
        }

        /* Add column bias if requested */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
              l_vec_reg_tmp[0], l_m, 0, l_vec_reg_tmp[0], 0, l_sve_type );
        }

        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       i_gp_reg_addr,
                                                       i_gp_reg_addr,
                                                       (i_m_blocking % i_vec_length) * i_data_size,
                                                       0 );
        /* load second part */
        if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
              l_vec_reg_tmp[1], l_vec_reg_tmp[1], 0, l_vec_reg_tmp[1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        } else {
          if (l_is_output_bf16 == 0) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                                  i_gp_reg_scratch,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  LIBXSMM_AARCH64_SVE_REG_P1 );
          } else {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                                                  i_gp_reg_scratch,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  LIBXSMM_AARCH64_SVE_REG_P1 );
            libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
          }
        }

        /* Add column bias if requested */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
              l_vec_reg_tmp[1], l_m, 0, l_vec_reg_tmp[1], 0, l_sve_type );
        }

        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       i_gp_reg_scratch,
                                                       i_gp_reg_scratch,
                                                       (i_m_blocking % i_vec_length) * i_data_size,
                                                       0 );

        /* zip data to target vector registers */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_zip[0],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vr_c[8*l_n + 2*l_m],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );

        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 l_instr_zip[1],
                                                 l_vec_reg_tmp[0],
                                                 l_vec_reg_tmp[1],
                                                 0,
                                                 l_vr_c[8*l_n + 2*l_m + 1],
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 l_type_zip );

      }
      /* adjust address register for next two columns */
      if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr,
                                                       i_gp_reg_scratch,
                                                       i_gp_reg_addr,
                                                       1ull*i_ld-l_m_bytes_full );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr,
                                                       i_gp_reg_scratch,
                                                       i_gp_reg_addr,
                                                       2ull*i_ld-l_m_bytes_full );
      }
    }
    /* reset address register */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_addr,
                                                   (long long)i_ld*i_n_blocking );
  } else {
    for ( l_n = 0; l_n < l_n_blocks; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        /* Load column bias if need be */
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          if (l_n == 0) {
            if (l_is_colbias_bf16 == 0) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  LIBXSMM_AARCH64_SVE_REG_UNDEF );
            } else {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  l_output_bf16_mask );
              libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_m, 0);
            }
            if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
              libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                             LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                             l_gp_reg_bias,
                                                             l_gp_reg_bias,
                                                             i_vec_length * l_bias_tsize,
                                                             0 );
            }
          }
          /* zip data to target vector registers */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   l_instr_zip[0],
                                                   l_m,
                                                   l_m,
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   l_type_zip );

          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   l_instr_zip[1],
                                                   l_m,
                                                   l_m,
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   l_type_zip );
        } else {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   l_vr_c[8*l_n + 2*l_m], l_vr_c[8*l_n + 2*l_m], 0, l_vr_c[8*l_n + 2*l_m], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   l_vr_c[8*l_n + 2*l_m+1], l_vr_c[8*l_n + 2*l_m+1], 0, l_vr_c[8*l_n + 2*l_m+1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        }
      }
      for ( l_m = l_m_blocks[0]; l_m < l_m_blocks[0] + l_m_blocks[1]; l_m++ ) {
        if ((i_micro_kernel_config->fused_bcolbias > 0) || (i_micro_kernel_config->fused_scolbias > 0)) {
          if (l_n == 0) {
            if (l_is_colbias_bf16 == 0) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  LIBXSMM_AARCH64_SVE_REG_P1 );
            } else {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                  LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF,
                  l_gp_reg_bias,
                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                  0,
                  l_m,
                  LIBXSMM_AARCH64_SVE_REG_P1 );
              libxsmm_generator_vcvt_bf16f32_aarch64_sve( io_generated_code, l_m, 0);
            }
          }
          /* zip data to target vector registers */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   l_instr_zip[0],
                                                   l_m,
                                                   l_m,
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m],
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   l_type_zip );

          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   l_instr_zip[1],
                                                   l_m,
                                                   l_m,
                                                   0,
                                                   l_vr_c[8*l_n + 2*l_m + 1],
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   l_type_zip );
        } else {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   l_vr_c[8*l_n + 2*l_m], l_vr_c[8*l_n + 2*l_m], 0, l_vr_c[8*l_n + 2*l_m], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   l_vr_c[8*l_n + 2*l_m+1], l_vr_c[8*l_n + 2*l_m+1], 0, l_vr_c[8*l_n + 2*l_m+1], 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_addr,
                                                       const unsigned int      i_gp_reg_scratch,
                                                       const unsigned int      i_vec_length,
                                                       const unsigned int      i_vec_reg_count,
                                                       const unsigned int      i_m_blocking,
                                                       const unsigned int      i_n_blocking,
                                                       const unsigned int      i_ld ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;
  unsigned int l_a_store_instruction = 0;
  unsigned int l_add_store = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);
  /* set store instruction */
  if( l_m_blocks[0] == 4 ){
    l_a_store_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_ST1_4;
    l_add_store = 64;
  } else if( l_m_blocks[0] == 3 ){
    l_a_store_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_ST1_3;
    l_add_store = 48;
  } else if( l_m_blocks[0] == 2 ){
    l_a_store_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_ST1_2;
    l_add_store = 32;
  } else if( l_m_blocks[0] == 1 ){
    l_a_store_instruction = LIBXSMM_AARCH64_INSTR_ASIMD_ST1_1;
    l_add_store = 16;
  }

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    if( l_m_blocks[0] > 0){
      libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code,
                                                       l_a_store_instruction,
                                                       i_gp_reg_addr,
                                                       LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                       l_vec_reg_acc_start + (l_m_total_blocks * l_n),
                                                       LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                    i_gp_reg_addr,
                                                    i_gp_reg_addr,
                                                    l_add_store,
                                                    0 );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                              l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                              l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_S );
    }
    if ( i_ld-l_m_bytes > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                     ((long long)i_ld - l_m_bytes) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                 (long long)i_ld*i_n_blocking );
}

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
                                                            const unsigned char     i_zip_row_major ) {
  unsigned int l_vec_reg_acc_start;
  unsigned int l_vec_reg_tmp[2];
  /* register blocking counter in n */
  unsigned int l_n = 0;
  unsigned int l_n_blocks = i_n_blocking / 2;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m iteration */
  unsigned int l_m_blocks[2];  /* 0: 128bit, 1: 64bit */
  unsigned int l_m_bytes = 0;
  unsigned int l_vr_c[24] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  /* derive zip instructions */
  unsigned int l_instr_zip[2] = { LIBXSMM_AARCH64_INSTR_ASIMD_ZIP1, LIBXSMM_AARCH64_INSTR_ASIMD_ZIP2 };
  libxsmm_aarch64_asimd_tupletype  l_type_zip = LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;

  LIBXSMM_UNUSED( i_micro_kernel_config );
  LIBXSMM_UNUSED( i_xgemm_desc );
  LIBXSMM_UNUSED( i_vec_reg_count );

  if ( i_zip_row_major ) {
    l_instr_zip[0] = LIBXSMM_AARCH64_INSTR_ASIMD_UZP1;
    l_instr_zip[1] = LIBXSMM_AARCH64_INSTR_ASIMD_UZP2;
    l_type_zip = LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  }

  /* derive register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8;

  /* start register of accumulator */
  l_vec_reg_acc_start = l_vr_c[0]; /*i_vec_reg_count - (i_n_blocking * l_m_total_blocks);*/

  /* temporary vector registers used to zip values before storing */
  l_vec_reg_tmp[0] = l_vec_reg_acc_start - 2;
  l_vec_reg_tmp[1] = l_vec_reg_acc_start - 1;

  for ( l_n = 0; l_n < l_n_blocks; l_n++ ) {
    /* second address register for stores */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  i_gp_reg_addr,
                                                  i_gp_reg_scratch,
                                                  i_gp_reg_scratch,
                                                  i_ld );

    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      /* zip data to temporary vector registers */
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                 l_instr_zip[0],
                                                 l_vr_c[8*l_n + 2*l_m],
                                                 l_vr_c[8*l_n + 2*l_m+1],
                                                 0,
                                                 l_vec_reg_tmp[0],
                                                 l_type_zip );

      libxsmm_aarch64_instruction_asimd_compute( io_generated_code,
                                                 l_instr_zip[1],
                                                 l_vr_c[8*l_n + 2*l_m],
                                                 l_vr_c[8*l_n + 2*l_m+1],
                                                 0,
                                                 l_vec_reg_tmp[1],
                                                 l_type_zip );

      /* store first part */
      libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              16,
                                              l_vec_reg_tmp[0],
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      /* store second part */
      libxsmm_aarch64_instruction_asimd_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_scratch,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              16,
                                              l_vec_reg_tmp[1],
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
    for (l_m = 0; l_m < l_m_blocks[1]; l_m++) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      LIBXSMM_ASSERT_MSG(0, "Not implemented yet");
    }
    /* adjust address register for next two columns */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_addr,
                                                   2ull*i_ld-l_m_bytes );
  }
  /* reset address register */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_addr,
                                                 i_gp_reg_scratch,
                                                 i_gp_reg_addr,
                                                 (long long)i_ld*i_n_blocking );
}

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
                                                          const unsigned char     i_zip_row_major ) {
  unsigned int l_vec_reg_acc_start;
  unsigned int l_vec_reg_tmp[2];
  /* register blocking counter in n */
  unsigned int l_n = 0;
  unsigned int l_n_blocks = (i_n_blocking + 1) / 2;

  /* register blocking counter in m */
  unsigned int l_m = 0;
  unsigned int l_m_blocks[2] = {0, 0}; /* 0: #full vector loads, 1: #predicated loads (0 or 1) */
  unsigned int l_m_bytes_full = 0;
  unsigned int l_remainder_size = 0;

  /* derive zip instructions */
  unsigned int l_instr_zip[2] = { LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, LIBXSMM_AARCH64_INSTR_SVE_UZP2_V };
  libxsmm_aarch64_sve_type l_type_zip = LIBXSMM_AARCH64_SVE_TYPE_D;
  unsigned int l_vr_c[24] = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  unsigned int l_is_output_bf16 = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ? 1 : 0;
  unsigned int l_is_i8f32_gemm  = ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_output_bf16_mask = LIBXSMM_AARCH64_SVE_REG_P2;
  unsigned int l_m_total_blocks = 0;
  unsigned char l_pred_reg = 0;
  unsigned char l_blend_reg = 6;
  unsigned char l_tmp_pred_reg0 = 5;
  unsigned char l_tmp_pred_reg1 = 4;
  unsigned char l_tmp_pred_reg0_2 = 7;
  unsigned char l_tmp_pred_reg1_2 = 3;
  unsigned int l_tmp_vreg = 0;
  unsigned int l_zero_vreg = 0;
  unsigned int l_gp_reg_scf = i_gp_reg_scratch3;
  unsigned int gp_reg_relumask = i_gp_reg_scratch2, gp_reg_relumask2 = i_gp_reg_scratch3;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_CAST_UCHAR(sizeof(float)));

  LIBXSMM_UNUSED( i_vec_reg_count );
  LIBXSMM_UNUSED( i_zero );
  LIBXSMM_UNUSED( i_zip_row_major );

  /* TODO (MMLA): implement */
  /* if ( i_zip_row_major ) {
    l_instr_zip[0] = LIBXSMM_AARCH64_INSTR_SVE_UZP1_V;
    l_instr_zip[1] = LIBXSMM_AARCH64_INSTR_SVE_UZP2_V;
    l_type_zip = LIBXSMM_AARCH64_SVE_TYPE_S;
  } */

  /* derive register blocking from kernel config */
  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * i_data_size + l_m_blocks[1] * (i_m_blocking % i_vec_length) * i_data_size;
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];

  /* start register of accumulator */
  l_vec_reg_acc_start = l_vr_c[0]; /*i_vec_reg_count - (i_n_blocking * l_m_total_blocks);*/

  /* temporary vector registers used to zip values to before storing */
  l_vec_reg_tmp[0] = l_vec_reg_acc_start - 2;
  l_vec_reg_tmp[1] = l_vec_reg_acc_start - 1;
  l_tmp_vreg = l_vec_reg_acc_start - 3;
  l_zero_vreg = l_vec_reg_acc_start - 4;

  if ((i_micro_kernel_config->fused_relu > 0) || (i_micro_kernel_config->fused_relu_nobitmask > 0))  {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           l_zero_vreg, l_zero_vreg, 0, l_zero_vreg,
                                           l_pred_reg, l_sve_type );
  }
  if (i_micro_kernel_config->fused_relu > 0) {
    libxsmm_generator_gemm_getval_stack_var_aarch64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, gp_reg_relumask);
  }
  if (l_is_i8f32_gemm > 0) {
    /* Broadcast scf to l_tmp_vreg */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                          l_gp_reg_scf, 0, 0, l_tmp_vreg, 0 );
  }

  /* load C accumulator */
  for ( l_n = 0; l_n < l_n_blocks; l_n++ ) {
    /* this is the jump size to be performed after a m-block is complete */
    unsigned int l_mask_adv = 0;
    unsigned int l_mask_adv2 = 0;

    /* second address register for stores */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_scratch,
                                                   i_ld );
    if (i_micro_kernel_config->fused_relu > 0) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     gp_reg_relumask,
                                                     gp_reg_relumask2,
                                                     gp_reg_relumask2,
                                                     i_xgemm_desc->ldcp/8 );
    }

    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      /* zip data to target vector registers */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               l_instr_zip[0],
                                               l_vr_c[8*l_n + 2*l_m],
                                               l_vr_c[8*l_n + 2*l_m + 1],
                                               0,
                                               l_vec_reg_tmp[0],
                                               LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                               l_type_zip );

      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               l_instr_zip[1],
                                               l_vr_c[8*l_n + 2*l_m],
                                               l_vr_c[8*l_n + 2*l_m + 1],
                                               0,
                                               l_vec_reg_tmp[1],
                                               LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                               l_type_zip );

      if (i_micro_kernel_config->fused_relu > 0) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V, l_vec_reg_tmp[0], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0,
            l_blend_reg, l_pred_reg, l_sve_type );
        libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_m_blocking, l_m, l_m_total_blocks,
            LIBXSMM_CAST_UCHAR(l_tmp_vreg),  LIBXSMM_CAST_UCHAR(gp_reg_relumask),
            l_blend_reg, l_tmp_pred_reg0, l_tmp_pred_reg1, LIBXSMM_CAST_UCHAR(i_gp_reg_scratch1), &l_mask_adv );

      }

      if ((i_micro_kernel_config->fused_relu > 0) || (i_micro_kernel_config->fused_relu_nobitmask > 0))  {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P,
                                                 l_vec_reg_tmp[0], l_zero_vreg, 0, l_vec_reg_tmp[0],
                                                 l_pred_reg, l_sve_type );
      }

      /* store first part */
      if (l_is_output_bf16 == 0) {
        if (l_is_i8f32_gemm > 0) {
          /* Convert result to F32  */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
              l_vec_reg_tmp[0], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_vec_reg_tmp[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
          /* Multiply with scaling factor */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                   l_vec_reg_tmp[0], l_tmp_vreg, 0, l_vec_reg_tmp[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
          /* For beta = 1 add original C value  */
          if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                  i_gp_reg_addr,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_zero_vreg,
                                                  LIBXSMM_AARCH64_SVE_REG_UNDEF );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                     l_zero_vreg, l_vec_reg_tmp[0], 0, l_vec_reg_tmp[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
          }
        }
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[0],
                                              LIBXSMM_AARCH64_SVE_REG_UNDEF );
      } else {
        libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[0],
                                              l_output_bf16_mask );
      }

      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_addr,
                                                     i_vec_length * i_data_size,
                                                     0 );

      if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
        /* Do nothing... */
      } else {
        if (i_micro_kernel_config->fused_relu > 0) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V, l_vec_reg_tmp[1], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0,
              l_blend_reg, l_pred_reg, l_sve_type );
          libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_m_blocking, l_m, l_m_total_blocks,
              LIBXSMM_CAST_UCHAR(l_tmp_vreg),  LIBXSMM_CAST_UCHAR(gp_reg_relumask2),
              l_blend_reg, l_tmp_pred_reg0_2, l_tmp_pred_reg1_2, LIBXSMM_CAST_UCHAR(i_gp_reg_scratch1), &l_mask_adv2 );

        }

        if ((i_micro_kernel_config->fused_relu > 0) || (i_micro_kernel_config->fused_relu_nobitmask > 0))  {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P,
                                                   l_vec_reg_tmp[1], l_zero_vreg, 0, l_vec_reg_tmp[1],
                                                   l_pred_reg, l_sve_type );
        }

        /* store second part */
        if (l_is_output_bf16 == 0) {
          if (l_is_i8f32_gemm > 0) {
            /* Convert result to F32  */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
                l_vec_reg_tmp[1], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_vec_reg_tmp[1], 0, libxsmm_generator_aarch64_get_sve_type(4) );
            /* Multiply with scaling factor */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     l_vec_reg_tmp[1], l_tmp_vreg, 0, l_vec_reg_tmp[1], 0, libxsmm_generator_aarch64_get_sve_type(4) );
            /* For beta = 1 add original C value  */
            if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                    i_gp_reg_scratch,
                                                    LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                    0,
                                                    l_zero_vreg,
                                                    LIBXSMM_AARCH64_SVE_REG_UNDEF );
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                       l_zero_vreg, l_vec_reg_tmp[1], 0, l_vec_reg_tmp[1], 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
          }
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                                  i_gp_reg_scratch,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  LIBXSMM_AARCH64_SVE_REG_UNDEF );
        } else {
          libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                                i_gp_reg_scratch,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[1],
                                                l_output_bf16_mask );
        }
      }

      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_scratch,
                                                     i_vec_length * i_data_size,
                                                     0 );
    }
    for ( l_m = l_m_blocks[0]; l_m < l_m_blocks[0] + l_m_blocks[1]; l_m++ ) {
      /* zip data to target vector registers */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               l_instr_zip[0],
                                               l_vr_c[8*l_n + 2*l_m],
                                               l_vr_c[8*l_n + 2*l_m + 1],
                                               0,
                                               l_vec_reg_tmp[0],
                                               LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                               l_type_zip );

      libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                               l_instr_zip[1],
                                               l_vr_c[8*l_n + 2*l_m],
                                               l_vr_c[8*l_n + 2*l_m + 1],
                                               0,
                                               l_vec_reg_tmp[1],
                                               LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                               l_type_zip );

      if (i_micro_kernel_config->fused_relu > 0) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V, l_vec_reg_tmp[0], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0,
            l_blend_reg, l_pred_reg, l_sve_type );
        libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_m_blocking, l_m, l_m_total_blocks,
            LIBXSMM_CAST_UCHAR(l_tmp_vreg),  LIBXSMM_CAST_UCHAR(gp_reg_relumask),
            l_blend_reg, l_tmp_pred_reg0, l_tmp_pred_reg1, LIBXSMM_CAST_UCHAR(i_gp_reg_scratch1), &l_mask_adv );

      }

      if ((i_micro_kernel_config->fused_relu > 0) || (i_micro_kernel_config->fused_relu_nobitmask > 0))  {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P,
                                                 l_vec_reg_tmp[0], l_zero_vreg, 0, l_vec_reg_tmp[0],
                                                 l_pred_reg, l_sve_type );
      }

      /* store first part */
      if (l_is_output_bf16 == 0) {
        if (l_is_i8f32_gemm > 0) {
          /* Convert result to F32  */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
              l_vec_reg_tmp[0], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_vec_reg_tmp[0], LIBXSMM_AARCH64_SVE_REG_P1, libxsmm_generator_aarch64_get_sve_type(4) );
          /* Multiply with scaling factor */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                   l_vec_reg_tmp[0], l_tmp_vreg, 0, l_vec_reg_tmp[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
          /* For beta = 1 add original C value  */
          if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
            libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                                  i_gp_reg_addr,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_zero_vreg,
                                                  LIBXSMM_AARCH64_SVE_REG_P1 );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                     l_zero_vreg, l_vec_reg_tmp[0], 0, l_vec_reg_tmp[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
          }
        }
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[0],
                                              LIBXSMM_AARCH64_SVE_REG_P1 );
      } else {
        libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[0], 0);
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_tmp[0],
                                              LIBXSMM_AARCH64_SVE_REG_P1 );
      }

      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_addr,
                                                     (i_m_blocking % i_vec_length) * i_data_size,
                                                     0 );


      if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
        /* Do nothing... */
      } else {
        if (i_micro_kernel_config->fused_relu > 0) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V, l_vec_reg_tmp[1], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0,
              l_blend_reg, l_pred_reg, l_sve_type );
          libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_m_blocking, l_m, l_m_total_blocks,
              LIBXSMM_CAST_UCHAR(l_tmp_vreg),  LIBXSMM_CAST_UCHAR(gp_reg_relumask2),
              l_blend_reg, l_tmp_pred_reg0_2, l_tmp_pred_reg1_2, LIBXSMM_CAST_UCHAR(i_gp_reg_scratch1), &l_mask_adv2 );

        }

        if ((i_micro_kernel_config->fused_relu > 0) || (i_micro_kernel_config->fused_relu_nobitmask > 0))  {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P,
                                                   l_vec_reg_tmp[1], l_zero_vreg, 0, l_vec_reg_tmp[1],
                                                   l_pred_reg, l_sve_type );
        }


        /* store second part */
        if (l_is_output_bf16 == 0) {
          if (l_is_i8f32_gemm > 0) {
            /* Convert result to F32  */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
                l_vec_reg_tmp[1], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_vec_reg_tmp[1], LIBXSMM_AARCH64_SVE_REG_P1, libxsmm_generator_aarch64_get_sve_type(4) );
            /* Multiply with scaling factor */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     l_vec_reg_tmp[1], l_tmp_vreg, 0, l_vec_reg_tmp[1], 0, libxsmm_generator_aarch64_get_sve_type(4) );
            /* For beta = 1 add original C value  */
            if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
              libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                    LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                                    i_gp_reg_scratch,
                                                    LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                    0,
                                                    l_zero_vreg,
                                                    LIBXSMM_AARCH64_SVE_REG_P1 );
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                       l_zero_vreg, l_vec_reg_tmp[1], 0, l_vec_reg_tmp[1], 0, libxsmm_generator_aarch64_get_sve_type(4) );
            }
          }
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF,
                                                  i_gp_reg_scratch,
                                                  LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                  0,
                                                  l_vec_reg_tmp[1],
                                                  LIBXSMM_AARCH64_SVE_REG_P1 );
        } else {
          libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, l_vec_reg_tmp[1], 0);
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF,
                                                i_gp_reg_scratch,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_vec_reg_tmp[1],
                                                LIBXSMM_AARCH64_SVE_REG_P1 );
        }
      }

      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_scratch,
                                                     (i_m_blocking % i_vec_length) * i_data_size,
                                                     0 );
    }

    /* adjust address register for next two columns */
    if ((i_n_blocking % 2 == 1) && (l_n == l_n_blocks-1)) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_addr,
                                                     1ull*i_ld-l_m_bytes_full );
      if (i_micro_kernel_config->fused_relu > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       gp_reg_relumask, i_gp_reg_scratch1, gp_reg_relumask,
                                                       ( 1ull * i_xgemm_desc->ldcp - (8ull * l_mask_adv))/8 );
      }
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_addr,
                                                     2ull*i_ld-l_m_bytes_full );
      if (i_micro_kernel_config->fused_relu > 0) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       gp_reg_relumask, i_gp_reg_scratch1, gp_reg_relumask,
                                                       ( 2ull * i_xgemm_desc->ldcp - (8ull * l_mask_adv))/8 );
      }
    }
  }
  /* reset address register */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_addr,
                                                 i_gp_reg_scratch,
                                                 i_gp_reg_addr,
                                                 (long long)i_ld*i_n_blocking );

}

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
                                                     const unsigned int      i_reduce_on_output  ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector stores, 1: #predicate stores (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_m_bytes_full = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned long long l_jump_block_n_last = 0; /* this is the jump size to be performed after a n-block is complete */
  unsigned int l_datatype_size = LIBXSMM_TYPESIZE(i_datatype);
  unsigned int l_full_vector_mask = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_SVE_REG_P2 : LIBXSMM_AARCH64_SVE_REG_UNDEF;
  unsigned int l_store_instr = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF;
  unsigned int l_masked_store_instr = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF;
  unsigned int l_partial_vector_mask = (LIBXSMM_DATATYPE_BF16 == i_datatype) ? LIBXSMM_AARCH64_SVE_REG_P4 : LIBXSMM_AARCH64_SVE_REG_P1;
  unsigned int l_cvt_i32f32 = (i_datatype == LIBXSMM_DATATYPE_F32 && i_inp_datatype == LIBXSMM_DATATYPE_I32) ? 1 : 0;
  unsigned int l_tmp_vreg = 0;
  unsigned int l_tmp_vreg2 = 0;

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * l_datatype_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);
  l_tmp_vreg = l_vec_reg_acc_start - 1;
  l_tmp_vreg2 = l_vec_reg_acc_start - 2;

  if (l_cvt_i32f32 > 0) {
    /* Broadcast scf to l_tmp_vreg */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                          i_aux_gp_reg, 0, 0, l_tmp_vreg, 0 );
  }

  /* stores C accumulator to memory */
  /* full vector stores */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* this is the jump size to be performed after a m-block is complete */
    unsigned long long l_jump_block_m_last = 0;

    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      if ( LIBXSMM_DATATYPE_BF16 == i_datatype ) {
        libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code,
                                                    l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                    LIBXSMM_AARCH64_SVE_REG_P0 );
      }
      if (l_cvt_i32f32 > 0) {
        /* Convert result to F32  */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 0, libxsmm_generator_aarch64_get_sve_type(4) );
        /* Multiply with scaling factor */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, l_tmp_vreg, 0, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 0, libxsmm_generator_aarch64_get_sve_type(4) );
        /* For beta = 1 add original C value  */
        if (i_reduce_on_output > 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_tmp_vreg2,
                                                LIBXSMM_AARCH64_SVE_REG_UNDEF );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                   l_tmp_vreg2, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 0, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m, 0, libxsmm_generator_aarch64_get_sve_type(4) );
        }
      }
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            l_store_instr,
                                            i_gp_reg_addr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                            l_full_vector_mask );
      /* increase pointer in m-dimension.
          but only if
            1) remainder follows
            or
            2) we are not at the end of the m-loop
      */
      if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                        i_gp_reg_addr,
                                                        i_gp_reg_addr,
                                                        i_vec_length * l_datatype_size,
                                                        0 );
      }
      /* combine the m-jump with the n one */
      else {
        l_jump_block_m_last = (long long)i_vec_length * l_datatype_size;
      }
    }

    if ( l_m_blocks[1] != 0 ) {
      if ( LIBXSMM_DATATYPE_BF16 == i_datatype ) {
        libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code,
                                                    l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                                    LIBXSMM_AARCH64_SVE_REG_P0 );
      }
      if (l_cvt_i32f32 > 0) {
        /* Convert result to F32  */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS,
            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
        /* Multiply with scaling factor */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], l_tmp_vreg, 0, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
        /* For beta = 1 add original C value  */
        if (i_reduce_on_output > 0) {
          libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                                i_gp_reg_addr,
                                                LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                0,
                                                l_tmp_vreg2,
                                                LIBXSMM_AARCH64_SVE_REG_UNDEF );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                   l_tmp_vreg2, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], 0, l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0], 0, libxsmm_generator_aarch64_get_sve_type(4) );
        }
      }
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            l_masked_store_instr,
                                            i_gp_reg_addr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                            l_partial_vector_mask );
    }

    l_jump_block_m_last += (long long)i_ld - l_m_bytes_full;

    if ( l_n != i_n_blocking - 1 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_addr,
                                                     l_jump_block_m_last );
    }
    else {
      l_jump_block_n_last = l_jump_block_m_last;
    }
  }

  /* reset C-ptr to original address */
  l_jump_block_n_last = (long long)i_ld * i_n_blocking - l_jump_block_n_last;
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_addr,
                                                  i_gp_reg_scratch,
                                                  i_gp_reg_addr,
                                                  l_jump_block_n_last );
}

LIBXSMM_API_INTERN
void libxsmm_generator_load_prng_state_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_prng_state_ptr,
                                                      const unsigned int      prng_state0_vreg,
                                                      const unsigned int      prng_state1_vreg,
                                                      const unsigned int      prng_state2_vreg,
                                                      const unsigned int      prng_state3_vreg ) {
  /* load RNG state */
  /* The memory layout is currently designed for architectures with upto 512 bit vector length.
   * If an architecture has a wider vector length, a lot of things need to be fixed (for both aarch64 and x86).
   * Load the first VL in fp32 values, the rest doesn't matter */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
      (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    /* The offset for the LDR instruction is not in bytes, it's in vector lengths;
     * Therefore, currently only architectures with a power-of-2-vector length are suppored by this code. */
    unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
    unsigned int l_increment = 64 / l_vector_length;
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,               0, prng_state0_vreg, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,     l_increment, prng_state1_vreg, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 2 * l_increment, prng_state2_vreg, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 3 * l_increment, prng_state3_vreg, 0 );
  } else {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,   0, prng_state0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,  64, prng_state1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 128, prng_state2_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 192, prng_state3_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_prng_state_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_prng_state_ptr,
                                                       const unsigned int      prng_state0_vreg,
                                                       const unsigned int      prng_state1_vreg,
                                                       const unsigned int      prng_state2_vreg,
                                                       const unsigned int      prng_state3_vreg ) {
  /* store RNG state */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
      (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    /* the same notes as in libxsmm_generator_load_prng_state_aarch64_asimd() apply here */
    unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
    unsigned int l_increment = 64 / l_vector_length;
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,               0, prng_state0_vreg, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,     l_increment, prng_state1_vreg, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 2 * l_increment, prng_state2_vreg, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 3 * l_increment, prng_state3_vreg, 0 );
  } else {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,   0, prng_state0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,  64, prng_state1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 128, prng_state2_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 192, prng_state3_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_tmp,
                                                      const unsigned int      i_gp_reg_prob_ptr,
                                                      const unsigned int      dropout_vreg_one,
                                                      const unsigned int      dropout_prob_vreg,
                                                      const unsigned int      dropout_invprob_vreg ) {
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
      (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    /* predicate register 0 must be ptrue; is ensured in generator_mateltwise_unary_binary_aarch64.c */
    unsigned char l_pred_reg = 0;

    /* load constant "1.0f" into register */
    libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve( io_generated_code, LIBXSMM_CAST_UCHAR(dropout_vreg_one), i_gp_reg_tmp,
                                                             LIBXSMM_AARCH64_SVE_TYPE_S, l_pred_reg, 0x3f800000 );

    /* load probability */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                          i_gp_reg_prob_ptr, 0, 0, dropout_prob_vreg, l_pred_reg );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                             dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg,
                                             l_pred_reg, LIBXSMM_AARCH64_SVE_TYPE_S );


    /* load 1/prob */
    /* there is only destructive division, so we first copy the value, and then divide in-place */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                             dropout_prob_vreg, dropout_prob_vreg, 0, dropout_invprob_vreg,
                                             l_pred_reg, LIBXSMM_AARCH64_SVE_TYPE_S );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FDIVR_V_P, /* div = a/b, divr = b/a, src0 must be dst */
                                             dropout_invprob_vreg, dropout_vreg_one, 0, dropout_invprob_vreg,
                                             l_pred_reg, LIBXSMM_AARCH64_SVE_TYPE_S );
  } else {
    /* load constant register */
    libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd( io_generated_code, LIBXSMM_CAST_UCHAR(dropout_vreg_one), i_gp_reg_tmp,
                                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S, 0x3f800000 );

    /* load probability */
    libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R, i_gp_reg_prob_ptr, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                     dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                               dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

    /* load 1/prob */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V,
                                               dropout_vreg_one, dropout_prob_vreg, 0, dropout_invprob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_inv_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_tmp,
                                                          const unsigned int      i_gp_reg_prob_ptr,
                                                          const unsigned int      dropout_vreg_one,
                                                          const unsigned int      dropout_vreg_zero,
                                                          const unsigned int      dropout_prob_vreg ) {
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) &&
      (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    /* predicate register 0 must be ptrue; is ensured in generator_mateltwise_unary_binary_aarch64.c */
    unsigned char l_pred_reg = 0;

    /* load constant register */
    libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve( io_generated_code, LIBXSMM_CAST_UCHAR(dropout_vreg_one), i_gp_reg_tmp,
                                                             LIBXSMM_AARCH64_SVE_TYPE_S, l_pred_reg, 0x3f800000 );

    /* load probability */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                          i_gp_reg_prob_ptr, 0, 0, dropout_prob_vreg, l_pred_reg );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                             dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg,
                                             l_pred_reg, LIBXSMM_AARCH64_SVE_TYPE_S );

    /* load 1/prob */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FDIVR_V_P, /* div = a/b, divr = b/a, src0 must be dst */
                                             dropout_prob_vreg, dropout_vreg_one, 0, dropout_prob_vreg,
                                             l_pred_reg, LIBXSMM_AARCH64_SVE_TYPE_S );

    /* load zero, for masking */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                             dropout_vreg_zero, dropout_vreg_zero, 0, dropout_vreg_zero,
                                             l_pred_reg, LIBXSMM_AARCH64_SVE_TYPE_S );
  } else {
    /* load constant register */
    libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd( io_generated_code, LIBXSMM_CAST_UCHAR(dropout_vreg_one), i_gp_reg_tmp,
                                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S, 0x3f800000 );

    /* load probability */
    libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R, i_gp_reg_prob_ptr, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                     dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                               dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

    /* load 1/prob */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V,
                                               dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

    /* load zero, for masking */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                               dropout_vreg_zero, dropout_vreg_zero, 0, dropout_vreg_zero, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128pp_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_reg_rng_state_0,
                                                   const unsigned int      i_vec_reg_rng_state_1,
                                                   const unsigned int      i_vec_reg_rng_state_2,
                                                   const unsigned int      i_vec_reg_rng_state_3,
                                                   const unsigned int      i_vec_reg_rng_tmp_0,
                                                   const unsigned int      i_vec_reg_rng_tmp_1,
                                                   const unsigned int      o_vec_reg_rng ) {
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_0);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_1);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_2);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_3);
  LIBXSMM_UNUSED(i_vec_reg_rng_tmp_0);
  LIBXSMM_UNUSED(i_vec_reg_rng_tmp_1);
  LIBXSMM_UNUSED(o_vec_reg_rng);
  /* TODO: needs validation */
#if 0
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 7, i_vec_reg_rng_tmp_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 25, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_0, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 9, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, 0, i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_state_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 11, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 21, i_vec_reg_rng_tmp_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128p_f32_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_vec_reg_rng_state_0,
                                                      const unsigned int      i_vec_reg_rng_state_1,
                                                      const unsigned int      i_vec_reg_rng_state_2,
                                                      const unsigned int      i_vec_reg_rng_state_3,
                                                      const unsigned int      i_vec_reg_rng_tmp_0,
                                                      const unsigned int      i_vec_reg_rng_tmp_1,
                                                      const unsigned int      i_vec_reg_rng_one,
                                                      const unsigned int      o_vec_reg_rng ) {
  /* generates random values */
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 9, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             o_vec_reg_rng, i_vec_reg_rng_one, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                             o_vec_reg_rng, i_vec_reg_rng_one, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 9, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, 0, i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_state_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 11, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 21, i_vec_reg_rng_tmp_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
}


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
                                                    const unsigned char     i_pred_reg ) {
  /* generates random values */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ADD_V,
                                               i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, o_vec_reg_rng, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V, /* USHR = LSR */
                                             o_vec_reg_rng, LIBXSMM_AARCH64_SVE_REG_UNDEF, 9, o_vec_reg_rng, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                             o_vec_reg_rng, i_vec_reg_rng_one, 0, o_vec_reg_rng, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                             o_vec_reg_rng, i_vec_reg_rng_one, 0, o_vec_reg_rng, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V,
                                             i_vec_reg_rng_state_1, LIBXSMM_AARCH64_SVE_REG_UNDEF, 9, i_vec_reg_rng_tmp_0, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, 0, i_vec_reg_rng_state_2, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, 0, i_vec_reg_rng_state_3, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, 0, i_vec_reg_rng_state_1, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_state_0, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                           i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, 0, i_vec_reg_rng_state_2, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V,
                                           i_vec_reg_rng_state_3, LIBXSMM_AARCH64_SVE_REG_UNDEF, 11, i_vec_reg_rng_tmp_0, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V, /* USHR = LSR */
                                           i_vec_reg_rng_state_3, LIBXSMM_AARCH64_SVE_REG_UNDEF, 21, i_vec_reg_rng_tmp_1, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_state_3, i_pred_reg, i_sve_type );
}


LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64_asimd( libxsmm_generated_code*               io_generated_code,
                                                                      const unsigned int                    i_vec_thres,
                                                                      const unsigned int                    i_vec_absmask,
                                                                      const unsigned int                    i_vec_scale,
                                                                      const unsigned int                    i_vec_shifter,
                                                                      const unsigned int                    i_vec_half,
                                                                      const unsigned int                    i_vec_c0,
                                                                      const unsigned int                    i_vec_c01,
                                                                      const unsigned int                    i_vec_c02,
                                                                      const unsigned int                    i_vec_c03,
                                                                      const unsigned int                    i_vec_c1,
                                                                      const unsigned int                    i_vec_c11,
                                                                      const unsigned int                    i_vec_c12,
                                                                      const unsigned int                    i_vec_c13,
                                                                      const unsigned int                    i_vec_c2,
                                                                      const unsigned int                    i_vec_c21,
                                                                      const unsigned int                    i_vec_c22,
                                                                      const unsigned int                    i_vec_c23,
                                                                      const unsigned int                    i_vec_tmp,
                                                                      const unsigned int                    i_vec_tmp1,
                                                                      const unsigned int                    i_gp_reg_tmp,
                                                                      const unsigned int                    i_gp_reg_tmp1,
                                                                      const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  unsigned long long thres_array = 0x40879fff;
  unsigned long long absmask_array = 0x7fffffff;
  unsigned long long scale_array = 0x406a0ea1;
  unsigned long long shifter_array = 0x4b400000;
  unsigned long long half_array = 0x3f000000;
  unsigned int c0_array[/*16*/] = { 0x3ecc4231u, 0x3ecc541cu, 0x3ecd6c48u, 0x3ed174c3u, 0x3ed9bd5du, 0x3ee5acd5u, 0x3ef2aeddu, 0x3efd5384u, 0x3f016724u, 0x3f00f778u, 0x3efb389eu, 0x3ef0464du, 0x3ee3014fu, 0x3ed50a78u, 0x3ec779dbu, 0x3ebae363u };
  unsigned int c1_array[/*16*/] = { 0xb7c7fb58u, 0xbacb9740u, 0xbc3e4b3au, 0xbd0d292au, 0xbd8bc5d0u, 0xbdd9978fu, 0xbe0f92d3u, 0xbe27b66du, 0xbe328ce7u, 0xbe3125bfu, 0xbe26dc9du, 0xbe17a056u, 0xbe06bdebu, 0xbdecc593u, 0xbdcf57aau, 0xbdb5ea3au };
  unsigned int c2_array[/*16*/] = { 0xbd877b85u, 0xbd7d9780u, 0xbd4cb70eu, 0xbd08a1e9u, 0xbc808857u, 0xb9476fd2u, 0x3c36f765u, 0x3c924160u, 0x3ca7b1fcu, 0x3ca5732cu, 0x3c95af63u, 0x3c8079f7u, 0x3c55fa4fu, 0x3c2fa86bu, 0x3c0fbb00u, 0x3bec178cu };
  unsigned int idx_array[/*4*/] = { 0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c};
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_thres),   i_gp_reg_tmp, i_tupletype, thres_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_absmask), i_gp_reg_tmp, i_tupletype, absmask_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_scale),   i_gp_reg_tmp, i_tupletype, scale_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_shifter), i_gp_reg_tmp, i_tupletype, shifter_array  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_half),    i_gp_reg_tmp, i_tupletype, half_array );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0),  i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 0 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c01), i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 2 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c02), i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 4 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c03), i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 6 );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1),  i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 0 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c11), i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 2 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c12), i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 4 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c13), i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 6 );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2),  i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 0 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c21), i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 2 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c22), i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 4 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c23), i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 6 );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_tmp),  i_gp_reg_tmp,  i_gp_reg_tmp1, idx_array,0);
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_tmp1), i_gp_reg_tmp, i_tupletype, 0x03020100 );
}


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
                                                                    const unsigned char            i_pred_reg ) {

  unsigned long long thres_array = 0x40879fff;
  unsigned long long absmask_array = 0x7fffffff;
  unsigned long long scale_array = 0x406a0ea1;
  unsigned long long shifter_array = 0x4b400000;
  unsigned long long half_array = 0x3f000000;
  unsigned long long expmask_array = 0x0000000f;

  unsigned int c0_array[/*16*/] = { 0x3ecc4231u, 0x3ecc541cu, 0x3ecd6c48u, 0x3ed174c3u, 0x3ed9bd5du, 0x3ee5acd5u, 0x3ef2aeddu, 0x3efd5384u, 0x3f016724u, 0x3f00f778u, 0x3efb389eu, 0x3ef0464du, 0x3ee3014fu, 0x3ed50a78u, 0x3ec779dbu, 0x3ebae363u };
  unsigned int c1_array[/*16*/] = { 0xb7c7fb58u, 0xbacb9740u, 0xbc3e4b3au, 0xbd0d292au, 0xbd8bc5d0u, 0xbdd9978fu, 0xbe0f92d3u, 0xbe27b66du, 0xbe328ce7u, 0xbe3125bfu, 0xbe26dc9du, 0xbe17a056u, 0xbe06bdebu, 0xbdecc593u, 0xbdcf57aau, 0xbdb5ea3au };
  unsigned int c2_array[/*16*/] = { 0xbd877b85u, 0xbd7d9780u, 0xbd4cb70eu, 0xbd08a1e9u, 0xbc808857u, 0xb9476fd2u, 0x3c36f765u, 0x3c924160u, 0x3ca7b1fcu, 0x3ca5732cu, 0x3c95af63u, 0x3c8079f7u, 0x3c55fa4fu, 0x3c2fa86bu, 0x3c0fbb00u, 0x3bec178cu };

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_thres),    i_gp_reg_tmp, i_sve_type, i_pred_reg, thres_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_absmask),  i_gp_reg_tmp, i_sve_type, i_pred_reg, absmask_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_scale),    i_gp_reg_tmp, i_sve_type, i_pred_reg, scale_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_shifter),  i_gp_reg_tmp, i_sve_type, i_pred_reg, shifter_array  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_half),     i_gp_reg_tmp, i_sve_type, i_pred_reg, half_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_exp_mask), i_gp_reg_tmp, i_sve_type, i_pred_reg, expmask_array );

  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE512) && (io_generated_code->arch < LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array, 64 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array, 64 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array, 64 );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) )  {
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array,     32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c01), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 8, 32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array,     32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c11), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 8, 32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array,     32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c21), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 8, 32 );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE256) )  {
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array,     16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c01), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 4, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c02), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 8, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c03), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 12, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array,     16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c11), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 4, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c12), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 8, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c13), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 12, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array,     16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c21), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 4, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c22), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 8, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c23), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 12, 16 );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64_asimd( libxsmm_generated_code*               io_generated_code,
                                                                          const unsigned int                    i_vec_thres,
                                                                          const unsigned int                    i_vec_absmask,
                                                                          const unsigned int                    i_vec_scale,
                                                                          const unsigned int                    i_vec_shifter,
                                                                          const unsigned int                    i_vec_half,
                                                                          const unsigned int                    i_vec_c0,
                                                                          const unsigned int                    i_vec_c01,
                                                                          const unsigned int                    i_vec_c02,
                                                                          const unsigned int                    i_vec_c03,
                                                                          const unsigned int                    i_vec_c1,
                                                                          const unsigned int                    i_vec_c11,
                                                                          const unsigned int                    i_vec_c12,
                                                                          const unsigned int                    i_vec_c13,
                                                                          const unsigned int                    i_vec_c2,
                                                                          const unsigned int                    i_vec_c21,
                                                                          const unsigned int                    i_vec_c22,
                                                                          const unsigned int                    i_vec_c23,
                                                                          const unsigned int                    i_vec_tmp,
                                                                          const unsigned int                    i_vec_tmp1,
                                                                          const unsigned int                    i_gp_reg_tmp,
                                                                          const unsigned int                    i_gp_reg_tmp1,
                                                                          const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  unsigned long long thres_array = 0x408f5fff;
  unsigned long long absmask_array = 0x7fffffff;
  unsigned long long scale_array = 0x405d67c9;
  unsigned long long shifter_array = 0x4b400000;
  unsigned long long half_array = 0x3f000000;

  unsigned int c0_array[/*16*/] = { 0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu, 0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu };
  unsigned int c1_array[/*16*/] = { 0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u, 0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u };
  unsigned int c2_array[/*16*/] = { 0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u, 0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u };
  unsigned int idx_array[/*4*/] = { 0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c };

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_thres),   i_gp_reg_tmp, i_tupletype, thres_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_absmask), i_gp_reg_tmp, i_tupletype, absmask_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_scale),   i_gp_reg_tmp, i_tupletype, scale_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_shifter), i_gp_reg_tmp, i_tupletype, shifter_array  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_half),    i_gp_reg_tmp, i_tupletype, half_array );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0),  i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 0 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c01), i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 2 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c02), i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 4 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c03), i_gp_reg_tmp, i_gp_reg_tmp1, c0_array, 6 );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1),  i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 0 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c11), i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 2 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c12), i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 4 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c13), i_gp_reg_tmp, i_gp_reg_tmp1, c1_array, 6 );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2),  i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 0 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c21), i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 2 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c22), i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 4 );
  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c23), i_gp_reg_tmp, i_gp_reg_tmp1, c2_array, 6 );

  libxsmm_aarch64_instruction_load16bytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_tmp),  i_gp_reg_tmp, i_gp_reg_tmp1, idx_array, 0);
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_tmp1), i_gp_reg_tmp, i_tupletype, 0x03020100 );
}


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
                                                                        const unsigned char            i_pred_reg ) {

  unsigned long long thres_array    = 0x408f5fff;
  unsigned long long absmask_array  = 0x7fffffff;
  unsigned long long scale_array    = 0x405d67c9;
  unsigned long long shifter_array  = 0x4b400000;
  unsigned long long half_array     = 0x3f000000;
  unsigned long long expmask_array  = 0x0000000f;

  unsigned int c0_array[/*16*/] = { 0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu, 0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu };
  unsigned int c1_array[/*16*/] = { 0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u, 0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u };
  unsigned int c2_array[/*16*/] = { 0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u, 0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u };


  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_thres),    i_gp_reg_tmp, i_sve_type, i_pred_reg, thres_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_absmask),  i_gp_reg_tmp, i_sve_type, i_pred_reg, absmask_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_scale),    i_gp_reg_tmp, i_sve_type, i_pred_reg, scale_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_shifter),  i_gp_reg_tmp, i_sve_type, i_pred_reg, shifter_array  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_half),     i_gp_reg_tmp, i_sve_type, i_pred_reg, half_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_exp_mask), i_gp_reg_tmp, i_sve_type, i_pred_reg, expmask_array );

  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE512) && (io_generated_code->arch < LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array, 64 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array, 64 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array, 64 );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) )  {
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array,     32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c01), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 8, 32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array,     32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c11), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 8, 32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array,     32 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c21), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 8, 32 );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE256) )  {
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array,     16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c01), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 4, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c02), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 8, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c03), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c0_array + 12, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array,     16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c11), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 4, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c12), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 8, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c13), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c1_array + 12, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2),  i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array,     16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c21), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 4, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c22), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 8, 16 );
    libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c23), i_gp_reg_tmp, i_gp_reg_tmp1, i_pred_reg, c2_array + 12, 16 );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_aarch64_asimd( libxsmm_generated_code*               io_generated_code,
                                                       const unsigned int                    i_vec_x,
                                                       const unsigned int                    i_vec_xr,
                                                       const unsigned int                    i_vec_xa,
                                                       const unsigned int                    i_vec_index,
                                                       const unsigned int                    i_vec_C0,
                                                       const unsigned int                    i_vec_C1,
                                                       const unsigned int                    i_vec_C2,
                                                       const unsigned int                    i_vec_thres,
                                                       const unsigned int                    i_vec_absmask,
                                                       const unsigned int                    i_vec_scale,
                                                       const unsigned int                    i_vec_shifter,
                                                       const unsigned int                    i_vec_half,
                                                       const unsigned int                    i_vec_c0,
                                                       const unsigned int                    i_vec_c1,
                                                       const unsigned int                    i_vec_c2,
                                                       const unsigned int                    i_vec_tmp,
                                                       const unsigned int                    i_vec_tmp1,
                                                       const libxsmm_aarch64_asimd_tupletype i_tupletype ) {
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_x, i_vec_x, 0, i_vec_xr,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SSHR_I_V,
                                             i_vec_xr, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 31, i_vec_xr,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_xr, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 31, i_vec_xr,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V,
                                             i_vec_x, i_vec_absmask, 0, i_vec_xa,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V,
                                             i_vec_xa, i_vec_thres, 0, i_vec_xa,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_xr, i_vec_xa, 0, i_vec_xr,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_xa, i_vec_xa, 0, i_vec_index,
                                             i_tupletype );

#ifndef LIBXSMM_AARCH64_SPLIT_FMA
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_shifter, i_vec_shifter, 0, i_vec_C0,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_index, i_vec_scale, 0, i_vec_shifter,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_shifter, i_vec_shifter, 0, i_vec_index,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C0, i_vec_C0, 0, i_vec_shifter,
                                             i_tupletype );

#else

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_index, i_vec_scale, 0, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_index, i_vec_shifter, 0, i_vec_index,
                                             i_tupletype );
#endif
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_index, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 2, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_1,
                                             i_vec_index, i_vec_tmp, 0, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_index, i_vec_tmp1, 0, i_vec_index,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4,
                                             i_vec_c0, i_vec_index, 0, i_vec_C0,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4,
                                              i_vec_c1, i_vec_index, 0, i_vec_C1,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4,
                                             i_vec_c2, i_vec_index, 0, i_vec_C2,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_xa, i_vec_C2, 0, i_vec_C1,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C1, i_vec_C1, 0, i_vec_C2,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                              i_vec_xa, i_vec_C2, 0, i_vec_C0,
                                              i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C0, i_vec_C0, 0, i_vec_C2,
                                             i_tupletype );

#ifndef LIBXSMM_AARCH64_SPLIT_FMA

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_half, i_vec_half, 0, i_vec_C0,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                              i_vec_xr, i_vec_C2, 0, i_vec_half,
                                              i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_half, i_vec_half, 0, i_vec_C2,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C0, i_vec_C0, 0, i_vec_half,
                                             i_tupletype );

#else
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_xr, i_vec_C2, 0, i_vec_C2,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_C2, i_vec_half, 0, i_vec_x,
                                             i_tupletype );
#endif
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_x, i_vec_C2, 0, i_vec_x,
                                             i_tupletype );
}


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
                                                     const unsigned int             i_vec_exp_mask, /* contains 0x000000ff in every element for masking the exponent */
                                                     const unsigned int             i_vec_temp,
                                                     const libxsmm_aarch64_sve_type i_sve_type,
                                                     const unsigned char            i_pred_reg ) {

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_x, i_vec_x, 0, i_vec_xr,
                                           i_pred_reg, i_sve_type );

    /* isolate sign bit */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V,
                                           i_vec_xr, LIBXSMM_AARCH64_SVE_REG_UNDEF, 31, i_vec_xr,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V,
                                           i_vec_xr, LIBXSMM_AARCH64_SVE_REG_UNDEF, 31, i_vec_xr,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_AND_V,
                                           i_vec_x, i_vec_absmask, 0, i_vec_xa,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P,
                                           i_vec_xa, i_vec_thres, 0, i_vec_xa,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_xr, i_vec_xa, 0, i_vec_xr,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_xa, i_vec_scale, 0, i_vec_index,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_index, i_vec_shifter, 0, i_vec_index,
                                           i_pred_reg, i_sve_type );

  /* index &= 0xff -> extract the exponent */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_AND_V,
                                           i_vec_index, i_vec_exp_mask, 0, i_vec_index,
                                           i_pred_reg, i_sve_type );


  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                           i_vec_c0, i_vec_index, 0, i_vec_C0,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                           i_vec_c1, i_vec_index, 0, i_vec_C1,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                           i_vec_c2, i_vec_index, 0, i_vec_C2,
                                           i_pred_reg, i_sve_type );

  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 8, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c01, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c11, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c21, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );
  }
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE256) ) {
    /* first lookup */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 4, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c01, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c11, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c21, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );

    /* second lookup */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 4, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c02, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c12, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c22, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );

    /* third lookup */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 4, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c03, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c13, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c23, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );
  }

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_xa, i_vec_C2, 0, i_vec_C1,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_xa, i_vec_C1, 0, i_vec_C0,
                                           i_pred_reg, i_sve_type );


  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                          i_vec_half, i_vec_half, 0, i_vec_C1,
                                          i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_C0, i_vec_xr, 0, i_vec_C1,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_x, i_vec_C1, 0, i_vec_x,
                                           i_pred_reg, i_sve_type );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_aarch64_asimd(  libxsmm_generated_code*                        io_generated_code,
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
                                                            const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_x, i_vec_x, 0, i_vec_xr,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SSHR_I_V,
                                             i_vec_xr, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 31, i_vec_xr,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_xr, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 31, i_vec_xr,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V,
                                             i_vec_x, i_vec_absmask, 0, i_vec_xa,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V,
                                             i_vec_xa, i_vec_thres, 0, i_vec_xa,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_xr, i_vec_xa, 0, i_vec_xr,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_xa, i_vec_xa, 0, i_vec_index,
                                             i_tupletype );

#ifndef LIBXSMM_AARCH64_SPLIT_FMA
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_shifter, i_vec_shifter, 0, i_vec_C0,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_index, i_vec_scale, 0, i_vec_shifter,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_shifter, i_vec_shifter, 0, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C0, i_vec_C0, 0, i_vec_shifter,
                                             i_tupletype );

#else
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_index, i_vec_scale, 0, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_index, i_vec_shifter, 0, i_vec_index,
                                             i_tupletype );
#endif

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_index, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 2, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_1,
                                             i_vec_index, i_vec_tmp, 0, i_vec_index,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_index, i_vec_tmp1, 0, i_vec_index,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4,
                                             i_vec_c0, i_vec_index, 0, i_vec_C0,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4,
                                              i_vec_c1, i_vec_index, 0, i_vec_C1,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4,
                                             i_vec_c2, i_vec_index, 0, i_vec_C2,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_xa, i_vec_C2, 0, i_vec_C1,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C1, i_vec_C1, 0, i_vec_C2,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                              i_vec_xa, i_vec_C2, 0, i_vec_C0,
                                              i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C0, i_vec_C0, 0, i_vec_C2,
                                             i_tupletype );

#ifndef LIBXSMM_AARCH64_SPLIT_FMA
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_half, i_vec_half, 0, i_vec_C0,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                              i_vec_xr, i_vec_C2, 0, i_vec_half,
                                              i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_half, i_vec_half, 0, i_vec_x,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_C0, i_vec_C0, 0, i_vec_half,
                                             i_tupletype );

#else
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_xr, i_vec_C2, 0, i_vec_C2,
                                             i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_C2, i_vec_half, 0, i_vec_x,
                                             i_tupletype );
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_aarch64_sve(  libxsmm_generated_code*        io_generated_code,
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
                                                          const unsigned int             i_vec_exp_mask, /* contains 0x000000ff for masking the exponent */
                                                          const unsigned int             i_vec_temp,
                                                          const libxsmm_aarch64_sve_type i_sve_type,
                                                          const unsigned char            i_pred_reg ) {

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                             i_vec_x, i_vec_x, 0, i_vec_xr,
                                             i_pred_reg, i_sve_type );


  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V,
                                           i_vec_xr, LIBXSMM_AARCH64_SVE_REG_UNDEF, 31, i_vec_xr,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V,
                                           i_vec_xr, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 31, i_vec_xr,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_AND_V,
                                             i_vec_x, i_vec_absmask, 0, i_vec_xa,
                                             i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P,
                                             i_vec_xa, i_vec_thres, 0, i_vec_xa,
                                             i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                             i_vec_xr, i_vec_xa, 0, i_vec_xr,
                                             i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_xa, i_vec_scale, 0, i_vec_index,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_index, i_vec_shifter, 0, i_vec_index,
                                           i_pred_reg, i_sve_type );

  /* index &= 0xff -> extract the exponent */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_AND_V,
                                           i_vec_index, i_vec_exp_mask, 0, i_vec_index,
                                           i_pred_reg, i_sve_type );


  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                           i_vec_c0, i_vec_index, 0, i_vec_C0,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                           i_vec_c1, i_vec_index, 0, i_vec_C1,
                                           i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                           i_vec_c2, i_vec_index, 0, i_vec_C2,
                                           i_pred_reg, i_sve_type );

  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512) ) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 8, i_vec_index,
                                            i_pred_reg, i_sve_type );


    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c01, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c11, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c21, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );
  }
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE256) ) {
    /* first lookup */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 4, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c01, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c11, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c21, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );

    /* second lookup */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 4, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c02, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c12, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c22, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );

    /* third lookup */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I,
                                            i_vec_index, LIBXSMM_AARCH64_SVE_REG_UNDEF, 4, i_vec_index,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c03, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C0, i_vec_temp, 0, i_vec_C0,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c13, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C1, i_vec_temp, 0, i_vec_C1,
                                            i_pred_reg, i_sve_type );

    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_TBL,
                                            i_vec_c23, i_vec_index, 0, i_vec_temp,
                                            i_pred_reg, i_sve_type );
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                            i_vec_C2, i_vec_temp, 0, i_vec_C2,
                                            i_pred_reg, i_sve_type );
  }

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_xa, i_vec_C2, 0, i_vec_C1,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_xa, i_vec_C1, 0, i_vec_C0,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                          i_vec_half, i_vec_half, 0, i_vec_C1,
                                          i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_xr, i_vec_C0, 0, i_vec_C1,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                          i_vec_C1, i_vec_C1, 0, i_vec_x,
                                          i_pred_reg, i_sve_type );
}


LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64_asimd( libxsmm_generated_code*               io_generated_code,
                                                                 const unsigned int                    i_vec_c0,
                                                                 const unsigned int                    i_vec_c1,
                                                                 const unsigned int                    i_vec_c2,
                                                                 const unsigned int                    i_vec_c3,
                                                                 const unsigned int                    i_vec_halves,
                                                                 const unsigned int                    i_vec_log2e,
                                                                 const unsigned int                    i_vec_expmask,
                                                                 const unsigned int                    i_vec_hi_bound,
                                                                 const unsigned int                    i_vec_lo_bound,
                                                                 const unsigned int                    i_gp_reg_tmp,
                                                                 const libxsmm_aarch64_asimd_tupletype i_tupletype  ) {

  unsigned long long vec_c0 = 0x3f34e022;
  unsigned long long vec_c1 = 0x3efd357f;
  unsigned long long vec_c2 = 0x3e20bcd5;
  unsigned long long vec_c3 = 0x3d635847;
  unsigned long long vec_halves = 0x3f000000;
  unsigned long long vec_log2e = 0x3fb8aa3b;
  unsigned int vec_expmask = 0x7f;
  unsigned long long vec_hi_bound = 0x42b0c0a5;
  unsigned long long vec_lo_bound = 0xc2b0c0a5;

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_halves), i_gp_reg_tmp, i_tupletype, vec_halves );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_log2e), i_gp_reg_tmp, i_tupletype, vec_log2e );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0), i_gp_reg_tmp, i_tupletype, vec_c0 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1), i_gp_reg_tmp, i_tupletype, vec_c1  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2), i_gp_reg_tmp, i_tupletype, vec_c2 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c3), i_gp_reg_tmp, i_tupletype, vec_c3 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_expmask), i_gp_reg_tmp, i_tupletype, vec_expmask );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_hi_bound), i_gp_reg_tmp, i_tupletype, vec_hi_bound );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_lo_bound), i_gp_reg_tmp, i_tupletype, vec_lo_bound );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64_sve( libxsmm_generated_code*               io_generated_code,
                                                               const unsigned int                    i_vec_c0,
                                                               const unsigned int                    i_vec_c1,
                                                               const unsigned int                    i_vec_c2,
                                                               const unsigned int                    i_vec_c3,
                                                               const unsigned int                    i_vec_halves,
                                                               const unsigned int                    i_vec_log2e,
                                                               const unsigned int                    i_vec_expmask,
                                                               const unsigned int                    i_vec_hi_bound,
                                                               const unsigned int                    i_vec_lo_bound,
                                                               const unsigned int                    i_gp_reg_tmp,
                                                               const libxsmm_aarch64_sve_type        i_sve_type,
                                                               const unsigned char                   i_pred_reg ) {

  unsigned long long vec_c0 = 0x3f34e022;
  unsigned long long vec_c1 = 0x3efd357f;
  unsigned long long vec_c2 = 0x3e20bcd5;
  unsigned long long vec_c3 = 0x3d635847;
  unsigned long long vec_halves = 0x3f000000;
  unsigned long long vec_log2e = 0x3fb8aa3b;
  unsigned int vec_expmask = 0x7f;
  unsigned long long vec_hi_bound = 0x42b0c0a5;
  unsigned long long vec_lo_bound = 0xc2b0c0a5;

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_halves), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_halves );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_log2e), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_log2e );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_c0 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_c1  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_c2 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c3), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_c3 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_expmask), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_expmask );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_hi_bound), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_hi_bound );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_lo_bound), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_lo_bound );
}

LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_aarch64_asimd( libxsmm_generated_code*               io_generated_code,
                                                  const unsigned int                    i_vec_x,
                                                  const unsigned int                    i_vec_y,
                                                  const unsigned int                    i_vec_z,
                                                  const unsigned int                    i_vec_c0,
                                                  const unsigned int                    i_vec_c1,
                                                  const unsigned int                    i_vec_c2,
                                                  const unsigned int                    i_vec_c3,
                                                  const unsigned int                    i_vec_halves,
                                                  const unsigned int                    i_vec_log2e,
                                                  const unsigned int                    i_vec_expmask,
                                                  const unsigned int                    i_vec_hi_bound,
                                                  const unsigned int                    i_vec_lo_bound,
                                                  const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V,
                                             i_vec_x, i_vec_lo_bound, 0, i_vec_x, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V,
                                             i_vec_x, i_vec_hi_bound, 0, i_vec_x, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_x, i_vec_log2e, 0, i_vec_x,
                                             i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_halves, i_vec_x, 0, i_vec_x,
                                             i_tupletype  );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRINTM_V,
                                             i_vec_x, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_y,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                             i_vec_x, i_vec_y, 0, i_vec_y,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_y, i_vec_y, 0, i_vec_z,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_z, i_vec_c3, 0, i_vec_z,
                                             i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_z, i_vec_c2, 0, i_vec_z,
                                             i_tupletype  );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_z, i_vec_y, 0, i_vec_z,
                                             i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_c1, i_vec_z, 0, i_vec_z,
                                             i_tupletype  );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_z, i_vec_y, 0, i_vec_z,
                                             i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_c0, i_vec_z, 0, i_vec_z,
                                             i_tupletype  );

  libxsmm_generator_scalefps_aarch64_asimd( io_generated_code, i_vec_z, i_vec_x, i_vec_x, i_vec_expmask, i_tupletype );
}


LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_aarch64_sve( libxsmm_generated_code*               io_generated_code,
                                                const unsigned int                    i_vec_x,
                                                const unsigned int                    i_vec_y,
                                                const unsigned int                    i_vec_z,
                                                const unsigned int                    i_vec_c0,
                                                const unsigned int                    i_vec_c1,
                                                const unsigned int                    i_vec_c2,
                                                const unsigned int                    i_vec_c3,
                                                const unsigned int                    i_vec_halves,
                                                const unsigned int                    i_vec_log2e,
                                                const unsigned int                    i_vec_expmask,
                                                const unsigned int                    i_vec_hi_bound,
                                                const unsigned int                    i_vec_lo_bound,
                                                const libxsmm_aarch64_sve_type        i_sve_type,
                                                const unsigned char                   i_pred_reg ) {

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P,
                                           i_vec_x, i_vec_lo_bound, 0, i_vec_x, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P,
                                           i_vec_x, i_vec_hi_bound, 0, i_vec_x, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_x, i_vec_log2e, 0, i_vec_x, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_halves, i_vec_x, 0, i_vec_x, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRINTM_V_P,
                                           i_vec_x, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_y, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                           i_vec_x, i_vec_y, 0, i_vec_y, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_y, i_vec_y, 0, i_vec_z, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_z, i_vec_c3, 0, i_vec_z, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_z, i_vec_c2, 0, i_vec_z, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_z, i_vec_y, 0, i_vec_z, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_c1, i_vec_z, 0, i_vec_z, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_z, i_vec_y, 0, i_vec_z, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_c0, i_vec_z, 0, i_vec_z, i_pred_reg, i_sve_type );

  libxsmm_generator_scalefps_aarch64_sve( io_generated_code, i_vec_z, i_vec_x, i_vec_x, i_vec_expmask, i_sve_type, i_pred_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_aarch64_asimd( libxsmm_generated_code*               io_generated_code,
                                               const unsigned int                    i_vec_x,
                                               const unsigned int                    i_vec_y,
                                               const unsigned int                    i_vec_z,
                                               const unsigned int                    i_vec_expmask,
                                               const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRINTM_V,
                                             i_vec_y, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_y,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCVTMS_V,
                                             i_vec_y, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_y,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_y, i_vec_expmask, 0, i_vec_y,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_y, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 23, i_vec_y,
                                             i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_x, i_vec_y, 0, i_vec_z,
                                             i_tupletype );
}

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_aarch64_sve( libxsmm_generated_code*               io_generated_code,
                                             const unsigned int                    i_vec_x,
                                             const unsigned int                    i_vec_y,
                                             const unsigned int                    i_vec_z,
                                             const unsigned int                    i_vec_expmask,
                                             const libxsmm_aarch64_sve_type        i_sve_type,
                                             const unsigned char                   i_pred_reg ) {

  /* SVE has an instruction called "fexpa". Maybe it could be used */

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRINTM_V_P,
                                           i_vec_y, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_y,
                                           i_pred_reg, i_sve_type );

  /* LIBXSMM_AARCH64_INSTR_ASIMD_FCVTMS_V: floating point convert to signed integer, rounding towards minus infinity */
  /* rounding to minus infinity should not be necessary (there is no direct instruction in SVE for that), because we already rounded previously */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCVTZS_V_P_SS,
                                           i_vec_y, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_y,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ADD_V,
                                           i_vec_y, i_vec_expmask, 0, i_vec_y,
                                           i_pred_reg, i_sve_type );

  /* logical shift left should be the same as shift left */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V,
                                           i_vec_y, LIBXSMM_AARCH64_SVE_REG_UNDEF, 23, i_vec_y,
                                           i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_x, i_vec_y, 0, i_vec_z,
                                           i_pred_reg, i_sve_type );
}


LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_asimd(  libxsmm_generated_code*                        io_generated_code,
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
                                                                    const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {
  unsigned long long c0_array = 0x49f77088;
  unsigned long long c1_array = 0x4883f7c0;
  unsigned long long c2_array = 0x45d89000;
  unsigned long long c3_array = 0x42100000;
  unsigned long long c1_d_array = 0x4966f190;
  unsigned long long c2_d_array = 0x474b0700;
  unsigned long long c3_d_array = 0x441d8000;
  unsigned long long hi_b_array = 0x409f0a3d;
  unsigned long long lo_b_array = 0xc09f0a3d;
  unsigned long long ones_array = 0x3f800000;
  unsigned int neg_ones_array = 0xbf800000;

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0), i_gp_reg_tmp, i_tupletype, c0_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1), i_gp_reg_tmp, i_tupletype, c1_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2), i_gp_reg_tmp, i_tupletype, c2_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c3), i_gp_reg_tmp, i_tupletype, c3_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1_d), i_gp_reg_tmp, i_tupletype, c1_d_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2_d), i_gp_reg_tmp, i_tupletype, c2_d_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c3_d), i_gp_reg_tmp, i_tupletype, c3_d_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_hi_bound), i_gp_reg_tmp, i_tupletype, hi_b_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_lo_bound), i_gp_reg_tmp, i_tupletype, lo_b_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_ones), i_gp_reg_tmp, i_tupletype, ones_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_neg_ones), i_gp_reg_tmp, i_tupletype, neg_ones_array );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_sve(  libxsmm_generated_code*                        io_generated_code,
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
                                                                        const unsigned char                            i_pred_reg ) {
  unsigned long long c0_array = 0x49f77088;
  unsigned long long c1_array = 0x4883f7c0;
  unsigned long long c2_array = 0x45d89000;
  unsigned long long c3_array = 0x42100000;
  unsigned long long c1_d_array = 0x4966f190;
  unsigned long long c2_d_array = 0x474b0700;
  unsigned long long c3_d_array = 0x441d8000;
  unsigned long long hi_b_array = 0x409f0a3d;
  unsigned long long lo_b_array = 0xc09f0a3d;
  unsigned long long ones_array = 0x3f800000;
  unsigned int neg_ones_array = 0xbf800000;

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c0), i_gp_reg_tmp, i_sve_type, i_pred_reg, c0_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1), i_gp_reg_tmp, i_sve_type, i_pred_reg, c1_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2), i_gp_reg_tmp, i_sve_type, i_pred_reg, c2_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c3), i_gp_reg_tmp, i_sve_type, i_pred_reg, c3_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c1_d), i_gp_reg_tmp, i_sve_type, i_pred_reg, c1_d_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c2_d), i_gp_reg_tmp, i_sve_type, i_pred_reg, c2_d_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_c3_d), i_gp_reg_tmp, i_sve_type, i_pred_reg, c3_d_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_hi_bound), i_gp_reg_tmp, i_sve_type, i_pred_reg, hi_b_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_lo_bound), i_gp_reg_tmp, i_sve_type, i_pred_reg, lo_b_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_ones), i_gp_reg_tmp, i_sve_type, i_pred_reg, ones_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_neg_ones), i_gp_reg_tmp, i_sve_type, i_pred_reg, neg_ones_array );
}

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
                                                                            const libxsmm_aarch64_asimd_tupletype          i_tupletype  ) {
  unsigned long long vec_halves = 0x3f000000;
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_halves), i_gp_reg_tmp, i_tupletype, vec_halves );
  libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_asimd(
        io_generated_code, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3, i_vec_c1_d, i_vec_c2_d, i_vec_c3_d,
        i_vec_hi_bound, i_vec_lo_bound, i_vec_ones, i_vec_neg_ones, i_gp_reg_tmp, i_tupletype );
}

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
                                                                          const unsigned char                            i_pred_reg ) {
  unsigned long long vec_halves = 0x3f000000;
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_vec_halves), i_gp_reg_tmp, i_sve_type, i_pred_reg, vec_halves );
  libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_sve(
        io_generated_code, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3, i_vec_c1_d, i_vec_c2_d, i_vec_c3_d,
        i_vec_hi_bound, i_vec_lo_bound, i_vec_ones, i_vec_neg_ones, i_gp_reg_tmp, i_sve_type, i_pred_reg );
}

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
                                                          const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_x, i_vec_x, 0, i_vec_x2, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V,
                                             i_vec_x, i_vec_hi_bound, 0, i_mask_hi, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V,
                                             i_vec_lo_bound, i_vec_x, 0, i_mask_lo, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_x2, i_vec_x2, 0, i_vec_nom, i_tupletype );


  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c2, i_vec_c2, 0, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_nom, i_vec_c3, 0, i_vec_c2, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c2, i_vec_c2, 0, i_vec_nom, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_tmp, i_vec_tmp, 0, i_vec_c2, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c1, i_vec_c1, 0, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_nom, i_vec_x2, 0, i_vec_c1, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c1, i_vec_c1, 0, i_vec_nom, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_tmp, i_vec_tmp, 0, i_vec_c1, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c0, i_vec_c0, 0, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_nom, i_vec_x2, 0, i_vec_c0, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c0, i_vec_c0, 0, i_vec_nom, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_tmp, i_vec_tmp, 0, i_vec_c0, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_nom, i_vec_x, 0, i_vec_nom, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_x2, i_vec_c3_d, 0, i_vec_denom, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c2_d, i_vec_c2_d, 0, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_denom, i_vec_x2, 0, i_vec_c2_d, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c2_d, i_vec_c2_d, 0, i_vec_denom, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_tmp, i_vec_tmp, 0, i_vec_c2_d, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c1_d, i_vec_c1_d, 0, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_denom, i_vec_x2, 0, i_vec_c1_d, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c1_d, i_vec_c1_d, 0, i_vec_denom, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_tmp, i_vec_tmp, 0, i_vec_c1_d, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c0, i_vec_c0, 0, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                             i_vec_denom, i_vec_x2, 0, i_vec_c0, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_c0, i_vec_c0, 0, i_vec_denom, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_tmp, i_vec_tmp, 0, i_vec_c0, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRECPE_V,
                                             i_vec_denom, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_tmp, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRECPS_V,
                                             i_vec_denom, i_vec_tmp, 0, i_vec_denom, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_denom, i_vec_tmp, 0, i_vec_denom, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_denom, i_vec_nom, 0, i_vec_x, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V,
                                             i_vec_ones,     i_mask_hi, 0, i_vec_x, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V,
                                             i_vec_neg_ones, i_mask_lo, 0, i_vec_x, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
}


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
                                                        const unsigned char                            i_pred_reg ) {

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_x, i_vec_x, 0, i_vec_x2, i_pred_reg, i_sve_type );

  /* fp compare greater than vec; sets all bits of the result to the answer (true = -1, false = 0) */
  /* i_mask_hi and i_mask_lo need to be predicate registers (0-7) instead of vector registers (0-31) */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V,
                                           i_vec_x, i_vec_hi_bound, 0, i_mask_hi, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V,
                                           i_vec_lo_bound, i_vec_x, 0, i_mask_lo, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_x2, i_vec_x2, 0, i_vec_nom, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c2, i_vec_c2, 0, i_vec_tmp, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_nom, i_vec_c3, 0, i_vec_c2, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c2, i_vec_c2, 0, i_vec_nom, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_tmp, i_vec_tmp, 0, i_vec_c2, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c1, i_vec_c1, 0, i_vec_tmp, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_nom, i_vec_x2, 0, i_vec_c1, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c1, i_vec_c1, 0, i_vec_nom, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_tmp, i_vec_tmp, 0, i_vec_c1, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c0, i_vec_c0, 0, i_vec_tmp, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_nom, i_vec_x2, 0, i_vec_c0, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c0, i_vec_c0, 0, i_vec_nom, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_tmp, i_vec_tmp, 0, i_vec_c0, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_nom, i_vec_x, 0, i_vec_nom, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_x2, i_vec_c3_d, 0, i_vec_denom, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c2_d, i_vec_c2_d, 0, i_vec_tmp, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_denom, i_vec_x2, 0, i_vec_c2_d, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c2_d, i_vec_c2_d, 0, i_vec_denom, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_tmp, i_vec_tmp, 0, i_vec_c2_d, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c1_d, i_vec_c1_d, 0, i_vec_tmp, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_denom, i_vec_x2, 0, i_vec_c1_d, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c1_d, i_vec_c1_d, 0, i_vec_denom, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_tmp, i_vec_tmp, 0, i_vec_c1_d, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c0, i_vec_c0, 0, i_vec_tmp, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                           i_vec_denom, i_vec_x2, 0, i_vec_c0, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_c0, i_vec_c0, 0, i_vec_denom, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                           i_vec_tmp, i_vec_tmp, 0, i_vec_c0, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPE_V,
                                           i_vec_denom, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_vec_tmp, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPS_V,
                                           i_vec_denom, i_vec_tmp, 0, i_vec_denom, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_denom, i_vec_tmp, 0, i_vec_denom, i_pred_reg, i_sve_type );

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_denom, i_vec_nom, 0, i_vec_x, i_pred_reg, i_sve_type );

  /* override too high values with 1, and too low values with -1 */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                           i_vec_ones,     i_vec_x, 0, i_vec_x, i_mask_hi, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                           i_vec_neg_ones, i_vec_x, 0, i_vec_x, i_mask_lo, i_sve_type );
}


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
                                                             const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {
  /* (tanh(x*0.5)+1)*0.5 = 1/(1+exp(-x)) */
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_x, i_vec_halves, 0, i_vec_x, i_tupletype );
  libxsmm_generator_tanh_ps_rational_78_aarch64_asimd(
    io_generated_code, i_vec_x, i_vec_x2, i_vec_nom, i_vec_denom,
    i_mask_hi, i_mask_lo, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3,
    i_vec_c1_d, i_vec_c2_d, i_vec_c3_d, i_vec_hi_bound, i_vec_lo_bound,
    i_vec_ones, i_vec_neg_ones, i_vec_tmp, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                             i_vec_x, i_vec_ones, 0, i_vec_x, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                             i_vec_x, i_vec_halves, 0, i_vec_x, i_tupletype );
}


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
                                                           const unsigned char                            i_pred_reg ) {
  /* (tanh(x*0.5)+1)*0.5 = 1/(1+exp(-x)) */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_x, i_vec_halves, 0, i_vec_x, i_pred_reg, i_sve_type );
  libxsmm_generator_tanh_ps_rational_78_aarch64_sve(
    io_generated_code, i_vec_x, i_vec_x2, i_vec_nom, i_vec_denom,
    i_mask_hi, i_mask_lo, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3,
    i_vec_c1_d, i_vec_c2_d, i_vec_c3_d, i_vec_hi_bound, i_vec_lo_bound,
    i_vec_ones, i_vec_neg_ones, i_vec_tmp, i_sve_type, i_pred_reg );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                           i_vec_x, i_vec_ones,  0, i_vec_x, i_pred_reg, i_sve_type );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                           i_vec_x, i_vec_halves, 0, i_vec_x, i_pred_reg, i_sve_type );
}


LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( libxsmm_generated_code*               io_generated_code,
                                                                 const unsigned char                   i_vec_reg,
                                                                 const unsigned int                    i_gp_reg_tmp,
                                                                 const libxsmm_aarch64_asimd_tupletype i_tupletype,
                                                                 const unsigned long long              imm64 ) {

libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, imm64 );
libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                               LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF,
                                      LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR,
                                      0, i_gp_reg_tmp);
libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, i_vec_reg,
                                                 i_tupletype );
libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                               LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
}


LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( libxsmm_generated_code*               io_generated_code,
                                                               const unsigned char                   i_vec_reg,
                                                               const unsigned int                    i_gp_reg_tmp,
                                                               const libxsmm_aarch64_sve_type        i_sve_type,
                                                               const unsigned char                   i_pred_reg,
                                                               const unsigned long long              imm64 ) {

  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, imm64 );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_MOV_R_P,
                                           i_gp_reg_tmp, 0, 0, i_vec_reg, i_pred_reg, i_sve_type );
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_load16bytes_const_to_vec( libxsmm_generated_code *io_generated_code,
                                                           const unsigned char     i_vec_reg,
                                                           const unsigned int      i_gp_reg_tmp0,
                                                           const unsigned int      i_gp_reg_tmp1,
                                                           void*                   imm64_array,
                                                           const unsigned int      i_start_index){
  /* fills an ASIMD register with 128 bits of data, which later will be used using indexed access */
  unsigned long long *imm_array_ptr =  (unsigned long long *)(imm64_array);
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, imm_array_ptr[i_start_index] );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, imm_array_ptr[i_start_index + 1] );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                  LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );

  libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                              i_gp_reg_tmp0, i_gp_reg_tmp1 );

  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                          i_vec_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                  LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
}


LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_rep16bytes_const_to_vec( libxsmm_generated_code* io_generated_code,
                                                              const unsigned int      i_vec_reg,
                                                              const unsigned int      i_gp_reg_tmp0,
                                                              const unsigned int      i_gp_reg_tmp1,
                                                              const unsigned int      i_pred_tmp,
                                                              void*                   imm64_array,
                                                              const unsigned int      i_start_index) {
  /* fills an SVE register with 128 bits of data, which later will be used using indexed access */
  unsigned long long* imm_array_ptr = (unsigned long long *)(imm64_array);

  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_pred_tmp, -1, i_gp_reg_tmp0 );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, imm_array_ptr[i_start_index] );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, imm_array_ptr[i_start_index + 1] );

  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );

  libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                             LIBXSMM_AARCH64_GP_REG_XSP, 0, i_gp_reg_tmp0, i_gp_reg_tmp1 );
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF,
                                        LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_vec_reg, i_pred_tmp );

  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_loadbytes_const_to_vec( libxsmm_generated_code* io_generated_code,
                                                              const unsigned char     i_vec_reg,
                                                              const unsigned int      i_gp_reg_tmp0,
                                                              const unsigned int      i_gp_reg_tmp1,
                                                              const unsigned int      i_pred_tmp,
                                                              void*                   imm64_array,
                                                              const unsigned int      i_bytes) {
  unsigned long long* imm_array_ptr = (unsigned long long *)(imm64_array);
  unsigned int i;

  /* fills an SVE register with i_bytes of data, which later will be used using indexed access */
  if ( !( i_bytes % 16 == 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_SIZE );
    return;
  }

  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_pred_tmp, -1, i_gp_reg_tmp0 );

  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 64, 0 );

  for (i=0;i<i_bytes/16;i++){
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, imm_array_ptr[2 * i] );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, imm_array_ptr[2 * i + 1] );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                               LIBXSMM_AARCH64_GP_REG_XSP, 16 * i, i_gp_reg_tmp0, i_gp_reg_tmp1 );
  }

  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                        LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_vec_reg, i_pred_tmp );

  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 64, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_memcpy( libxsmm_generated_code*        io_generated_code,
                                             const unsigned char            i_gp_reg_src,
                                             const unsigned char            i_gp_reg_dst,
                                             const unsigned char            i_gp_reg_tmp,
                                             const unsigned char            i_vec_reg_tmp,
                                             const unsigned char            i_pred_reg_tmp,
                                             const unsigned int             i_element_count,
                                             const libxsmm_aarch64_sve_type i_sve_type ) {

  unsigned int l_bytes_length = (1 << (unsigned int) i_sve_type) * i_element_count;
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch); /* number of bytes per vector */
  unsigned int l_remainder = l_bytes_length % l_vector_length;
  unsigned int l_i;
  unsigned int l_incremented_size = l_bytes_length - l_remainder;

  /* the offset of the instructions could be used, but that would make things more complicated and
   * we're L1 cache bound probably anyways */
  for (l_i=0;l_i+l_vector_length<=l_bytes_length;l_i+=l_vector_length){

    /* load all bytes */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                          i_gp_reg_src, 0, 0, i_vec_reg_tmp, i_pred_reg_tmp );
    /* store all bytes */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                          i_gp_reg_dst, 0, 0, i_vec_reg_tmp, i_pred_reg_tmp );

    /* inc src, inc dst */
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   i_gp_reg_src, i_gp_reg_src, l_vector_length, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                   i_gp_reg_dst, i_gp_reg_dst, l_vector_length, 0 );
  }

  if (l_remainder != 0) {
    /* remainder handling, set predicate first */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_pred_reg_tmp, l_remainder, i_gp_reg_tmp );
    /* load all bytes */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                          i_gp_reg_src, 0, 0, i_vec_reg_tmp, i_pred_reg_tmp );
    /* store all bytes */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF,
                                          i_gp_reg_dst, 0, 0, i_vec_reg_tmp, i_pred_reg_tmp );
    /* increment not needed, as this is the last copying instruction */
  }

  /* decrement source and destination registers */
  if (l_incremented_size > 0){
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   i_gp_reg_src, i_gp_reg_src, l_incremented_size, 0 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                   i_gp_reg_dst, i_gp_reg_dst, l_incremented_size, 0 );
  }
}
