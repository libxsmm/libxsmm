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
#include "generator_mateltwise_transform_rv64.h"
#include "generator_common_rv64.h"
#include "generator_rv64_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_rv64.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_rv64_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const unsigned int                      i_gp_reg_scratch,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int  load_instr;
  unsigned int  store_instr;

  load_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_GP_FLD : LIBXSMM_RV64_INSTR_GP_FLW;
  store_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_GP_FSD : LIBXSMM_RV64_INSTR_GP_FSW;

  /* m loop header */
  libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* n loop header */
  libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* actual transpose */
  libxsmm_rv64_instruction_alu_move( io_generated_code, load_instr, i_gp_reg_in, LIBXSMM_RV64_GP_REG_X5, 0 );

  libxsmm_rv64_instruction_alu_move( io_generated_code, store_instr, i_gp_reg_out, LIBXSMM_RV64_GP_REG_X5, 0 );

  /* advance input pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                              i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                              (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                              i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                              i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n) ? 1 : 0 );

  /* advance output pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                              i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                              ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

  /* advance input pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                              ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - ((long long)i_micro_kernel_config->datatype_size_in) );

  /* close m loop */
  libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, (i_mateltwise_desc->m) ? 1 : 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_load_regblock_8x8_rv64( libxsmm_generated_code*  io_generated_code,
                                                         const unsigned int       i_gp_reg_addr,
                                                         const unsigned int       i_gp_reg_dst,
                                                         const unsigned int       i_gp_reg_scratch,
                                                         const unsigned int       i_reg_count,
                                                         const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                         const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  /* Consecutive registers are divided into even and odd sequence */
  int e_reg = i_gp_reg_dst;
  int o_reg = i_gp_reg_dst + 4;

  unsigned int i;

  unsigned int l_valid_e_regs = i_reg_count / 2 + i_reg_count % 2;
  unsigned int l_valid_o_regs = i_reg_count / 2;

  for (i = 0; i < 4; i++){
    /* Load even register */
    if (i < l_valid_e_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VLE32_V,
          i_gp_reg_addr, i_gp_reg_scratch, e_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
    } else {
      /* Fill zero in the register */
      libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X0,
        LIBXSMM_RV64_GP_REG_V0, e_reg + i, 1);
    }

    /* Load odd register */
    if (i < l_valid_o_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VLE32_V,
          i_gp_reg_addr, i_gp_reg_scratch, o_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
    } else {
      /* Fill zero in the register */
      libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X0,
        LIBXSMM_RV64_GP_REG_V0, o_reg + i, 1);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_store_regblock_8x8_rv64( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_addr,
                                                          const unsigned int      i_gp_reg_dst,
                                                          const unsigned int      i_gp_reg_scratch,
                                                          const unsigned int      i_reg_count,
                                                          const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                          const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  int e_reg = i_gp_reg_dst;
  int o_reg = i_gp_reg_dst + 4;

  unsigned int l_valid_e_regs = i_reg_count / 2 + i_reg_count % 2;
  unsigned int l_valid_o_regs = i_reg_count / 2;
  unsigned int i;

  LIBXSMM_UNUSED(e_reg);
  LIBXSMM_UNUSED(o_reg);

  for (i = 0; i < 4; i++){
    /* store even register */
    if (i < l_valid_e_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSE32_V,
          i_gp_reg_addr, i_gp_reg_scratch, e_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
    }

    /* store odd register */
    if (i < l_valid_o_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSE32_V,
          i_gp_reg_addr, i_gp_reg_scratch, o_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_load_regblock_4x4_rv64( libxsmm_generated_code*  io_generated_code,
                                                         const unsigned int       i_gp_reg_addr,
                                                         const unsigned int       i_gp_reg_dst,
                                                         const unsigned int       i_gp_reg_scratch,
                                                         const unsigned int       i_reg_count,
                                                         const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                         const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  /* Consecutive registers are divided into even and odd sequence */
  int e_reg = i_gp_reg_dst;
  int o_reg = i_gp_reg_dst + 2;

  unsigned int l_valid_e_regs = i_reg_count / 2 + i_reg_count % 2;
  unsigned int l_valid_o_regs = i_reg_count / 2;
  unsigned int i;

  for (i = 0; i < 2; i++){
    /* Load even register */
    if (i < l_valid_e_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VLE64_V,
          i_gp_reg_addr, i_gp_reg_scratch, e_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
    } else {
      /* Fill zero in the register */
      libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X0,
        LIBXSMM_RV64_GP_REG_V0, e_reg + i, 1);
    }

    /* Load odd register */
    if (i < l_valid_o_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VLE64_V,
          i_gp_reg_addr, i_gp_reg_scratch, o_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );
    } else {
      /* Fill zero in the register */
      libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X0,
        LIBXSMM_RV64_GP_REG_V0, o_reg + i, 1);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_store_regblock_4x4_rv64( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_addr,
                                                          const unsigned int      i_gp_reg_dst,
                                                          const unsigned int      i_gp_reg_scratch,
                                                          const unsigned int      i_reg_count,
                                                          const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                          const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  int e_reg = i_gp_reg_dst;
  int o_reg = i_gp_reg_dst + 2;

  unsigned int l_valid_e_regs = i_reg_count / 2 + i_reg_count % 2;
  unsigned int l_valid_o_regs = i_reg_count / 2;
  unsigned int i;

  LIBXSMM_UNUSED(e_reg);
  LIBXSMM_UNUSED(o_reg);

  for (i = 0; i < 2; i++){
    /* store even register */
    if (i < l_valid_e_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSE64_V,
          i_gp_reg_addr, i_gp_reg_scratch, e_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
    }

    /* store odd register */
    if (i < l_valid_o_regs) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSE64_V,
          i_gp_reg_addr, i_gp_reg_scratch, o_reg + i, 1 );

      /* Move to next in address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr, (unsigned long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out );
    }
  }
}

/* Performs 32bit 8x8 transpose */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_shuffle_regblock_32bit_8x8_rvv( libxsmm_generated_code* io_generated_code,
                                                                               const unsigned int      i_gp_reg_dst_e,
                                                                               const unsigned int      i_gp_reg_dst_o,
                                                                               const unsigned int      i_gp_reg_scratch,
                                                                               const unsigned int      i_mask_e,
                                                                               const unsigned int      i_mask_o,
                                                                               const unsigned int      i_shuffle_stride ) {
  int e_reg = i_gp_reg_dst_e;
  int o_reg = i_gp_reg_dst_o;
  int s_reg = i_gp_reg_scratch;

  unsigned int i;

  /* Store the even mask */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X0,
    LIBXSMM_RV64_GP_REG_X28, i_mask_e);

  /* Store the odd mask */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X0,
    LIBXSMM_RV64_GP_REG_X29, i_mask_o);

  libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X28,
    LIBXSMM_RV64_GP_REG_V0, LIBXSMM_RV64_GP_REG_V0, 1);

  /* Copy the even registers (0, 2, 4, 6) to scratch */
  for (i = 0; i < 4; i++){
    libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_V,
      e_reg, LIBXSMM_RV64_GP_REG_X0, s_reg + i, 0);

    if ((e_reg + i_shuffle_stride) > 12){
      e_reg = i_gp_reg_dst_e + 1;
    } else {
      e_reg += i_shuffle_stride;
    }
  }

  e_reg = i_gp_reg_dst_e;

  /* Do a vslide up */
  for (i = 0; i < 4; i++){
    libxsmm_rv64_instruction_rvv_compute_imm( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSLIDEUP_VI,
      o_reg, i_shuffle_stride, e_reg, 0);

    /* For long strides, move to next start register once it exceeds max register count */
    if ((o_reg + i_shuffle_stride) > 12){
      o_reg = i_gp_reg_dst_o + 1;
    } else {
      o_reg += i_shuffle_stride;
    }

    /* For long strides, move to next start register once it exceeds max register count */
    if ((e_reg + i_shuffle_stride) > 12){
      e_reg = i_gp_reg_dst_e + 1;
    } else {
      e_reg += i_shuffle_stride;
    }
  }

  /* Set the odd mask */
  libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X29,
    LIBXSMM_RV64_GP_REG_V0, LIBXSMM_RV64_GP_REG_V0, 1);

  o_reg = i_gp_reg_dst_o;

  /* Do a vslide down */
  for (i = 0; i < 4; i++){
    libxsmm_rv64_instruction_rvv_compute_imm( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSLIDEDOWN_VI,
      s_reg + i, i_shuffle_stride, o_reg, 0);

    if ((o_reg + i_shuffle_stride) > 12){
      o_reg = i_gp_reg_dst_o + 1;
    } else {
      o_reg += i_shuffle_stride;
    }
  }
}

/* Performs 64bit 4x8 transpose */
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_shuffle_regblock_64bit_4x4_rvv( libxsmm_generated_code* io_generated_code,
                                                                               const unsigned int      i_gp_reg_dst_e,
                                                                               const unsigned int      i_gp_reg_dst_o,
                                                                               const unsigned int      i_gp_reg_scratch,
                                                                               const unsigned int      i_mask_e,
                                                                               const unsigned int      i_mask_o,
                                                                               const unsigned int      i_shuffle_stride ) {
  int e_reg = i_gp_reg_dst_e;
  int o_reg = i_gp_reg_dst_o;
  int s_reg = i_gp_reg_scratch;

  unsigned int i;

  /* Store the even mask */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X0,
    LIBXSMM_RV64_GP_REG_X28, i_mask_e);

  /* Store the even mask */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X0,
    LIBXSMM_RV64_GP_REG_X29, i_mask_o);

  libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X28,
    LIBXSMM_RV64_GP_REG_V0, LIBXSMM_RV64_GP_REG_V0, 1);

  /* Copy the even registers (0, 2) to scratch */
  for (i = 0; i < 2; i++){
    libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_V,
      e_reg, LIBXSMM_RV64_GP_REG_X0, s_reg + i, 0);

    if ((e_reg + i_shuffle_stride) > 8){
      e_reg = i_gp_reg_dst_e + 1;
    } else {
      e_reg += i_shuffle_stride;
    }
  }

  e_reg = i_gp_reg_dst_e;

  /* Do a vslide up */
  for (i = 0; i < 2; i++){
    libxsmm_rv64_instruction_rvv_compute_imm( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSLIDEUP_VI,
      o_reg, i_shuffle_stride, e_reg, 0);

    /* For long strides, move to next start register once it exceeds max register count */
    if ((o_reg + i_shuffle_stride) > 8){
      o_reg = i_gp_reg_dst_o + 1;
    } else {
      o_reg += i_shuffle_stride;
    }

    /* For long strides, move to next start register once it exceeds max register count */
    if ((e_reg + i_shuffle_stride) > 8){
      e_reg = i_gp_reg_dst_e + 1;
    } else {
      e_reg += i_shuffle_stride;
    }
  }

  /* Set the odd mask */
  libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VMV_V_X, LIBXSMM_RV64_GP_REG_X29,
    LIBXSMM_RV64_GP_REG_V0, LIBXSMM_RV64_GP_REG_V0, 1);

  o_reg = i_gp_reg_dst_o;

  /* Do a vslide down */
  for (i = 0; i < 2; i++){
    libxsmm_rv64_instruction_rvv_compute_imm( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSLIDEDOWN_VI,
      s_reg + i, i_shuffle_stride, o_reg, 0);

    if ((o_reg + i_shuffle_stride) > 12){
      o_reg = i_gp_reg_dst_o + 1;
    } else {
      o_reg += i_shuffle_stride;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_rvv( libxsmm_generated_code*     io_generated_code,
                                                                             libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                                             const unsigned int          i_gp_reg_in,
                                                                             const unsigned int          i_gp_reg_out,
                                                                             const unsigned int          i_gp_reg_scratch,
                                                                             const unsigned int          i_m,
                                                                             const unsigned int          i_n,
                                                                             const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  unsigned int l_reg_in_start       = LIBXSMM_RV64_GP_REG_V5;
  unsigned int l_reg_scratch_start  = LIBXSMM_RV64_GP_REG_V1;

  LIBXSMM_UNUSED(l_reg_scratch_start);

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_m, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_D, LIBXSMM_RV64_LMUL_M1);

  /* Load tensor from the src registers */
  libxsmm_generator_transform_load_regblock_8x8_rv64(io_generated_code, i_gp_reg_in, l_reg_in_start, i_gp_reg_scratch, i_n, i_micro_kernel_config, i_mateltwise_desc);

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, 8, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_D, LIBXSMM_RV64_LMUL_M1);

  /* Pass through shuffle network */
  libxsmm_generator_transform_norm_to_normt_shuffle_regblock_32bit_8x8_rvv(io_generated_code, l_reg_in_start, l_reg_in_start + 4, l_reg_scratch_start, 0xaa, 0x55, 1);
  libxsmm_generator_transform_norm_to_normt_shuffle_regblock_32bit_8x8_rvv(io_generated_code, l_reg_in_start, l_reg_in_start + 1, l_reg_scratch_start, 0xcc, 0x33, 2);
  libxsmm_generator_transform_norm_to_normt_shuffle_regblock_32bit_8x8_rvv(io_generated_code, l_reg_in_start, l_reg_in_start + 2, l_reg_scratch_start, 0xf0, 0x0f, 4);

  /* Set SEW and VL for store */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_n, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_D, LIBXSMM_RV64_LMUL_M1);

  /* Store the registers to the dst */
  libxsmm_generator_transform_store_regblock_8x8_rv64( io_generated_code, i_gp_reg_out, l_reg_in_start, i_gp_reg_scratch, i_m, i_micro_kernel_config, i_mateltwise_desc );

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, 8, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_D, LIBXSMM_RV64_LMUL_M1);

  /* Reset output pointer for next m block */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                              ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out * i_m) - (i_n * (long long)i_micro_kernel_config->datatype_size_out) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_rvv( libxsmm_generated_code*     io_generated_code,
                                                                             libxsmm_loop_label_tracker* io_loop_label_tracker,
                                                                             const unsigned int          i_gp_reg_in,
                                                                             const unsigned int          i_gp_reg_out,
                                                                             const unsigned int          i_gp_reg_scratch,
                                                                             const unsigned int          i_m,
                                                                             const unsigned int          i_n,
                                                                             const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  unsigned int l_reg_in_start       = LIBXSMM_RV64_GP_REG_V5;
  unsigned int l_reg_scratch_start  = LIBXSMM_RV64_GP_REG_V1;

  LIBXSMM_UNUSED(l_reg_scratch_start);

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_m, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_Q, LIBXSMM_RV64_LMUL_M1);

  /* Load tensor from the src registers */
  libxsmm_generator_transform_load_regblock_4x4_rv64( io_generated_code, i_gp_reg_in, l_reg_in_start, i_gp_reg_scratch, i_n, i_micro_kernel_config, i_mateltwise_desc );

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, 4, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_Q, LIBXSMM_RV64_LMUL_M1);

  /* Pass through shuffle network */
  libxsmm_generator_transform_norm_to_normt_shuffle_regblock_64bit_4x4_rvv( io_generated_code, l_reg_in_start, l_reg_in_start + 2, l_reg_scratch_start, 0xaa, 0x55, 1 );

  libxsmm_generator_transform_norm_to_normt_shuffle_regblock_64bit_4x4_rvv( io_generated_code, l_reg_in_start, l_reg_in_start + 1, l_reg_scratch_start, 0xcc, 0x33, 2 );

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_n, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_Q, LIBXSMM_RV64_LMUL_M1);

  /* Store tensor to the dst */
  libxsmm_generator_transform_store_regblock_4x4_rv64( io_generated_code, i_gp_reg_out, l_reg_in_start, i_gp_reg_scratch, i_m, i_micro_kernel_config, i_mateltwise_desc );

  /* Set SEW and VL */
  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, 4, LIBXSMM_RV64_GP_REG_X28, LIBXSMM_RV64_SEW_Q, LIBXSMM_RV64_LMUL_M1);

  /* Reset output pointer for next m block */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                              ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out * i_m) - (i_n * (long long)i_micro_kernel_config->datatype_size_out) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_rvv_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  /* For small matrices invoke scalar transpose */
  if ( (i_mateltwise_desc->m < 4) && (i_mateltwise_desc->n < 4) ) {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_rv64_microkernel( io_generated_code, io_loop_label_tracker,
                                                                            i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                            i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    if ( i_mateltwise_desc->m >= 8 ) {
      /* open m loop */
      libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, (i_mateltwise_desc->m/8)*8 );

      if ( i_mateltwise_desc->n >= 8 ) {
        /* open n loop */
        libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/8)*8 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, i_gp_reg_scratch, 8, 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
        /* close n footer */
        libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 8 );
      }

      if ( i_mateltwise_desc->n % 8 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, i_gp_reg_scratch, 8, i_mateltwise_desc->n % 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }

      /* advance output pointer */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
        i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out, (8LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

      /* advance input pointer */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                       i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in, (((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (8LL * i_micro_kernel_config->datatype_size_in)) );
      /* close m loop */
      libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 8 );
    }

    if ( i_mateltwise_desc->m % 8 != 0 ) {
      if ( i_mateltwise_desc->n >= 8 ) {
        /* open n loop */
        libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/8)*8 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, i_gp_reg_scratch, i_mateltwise_desc->m % 8, 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
        /* close n footer */
        libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 8 );
      }

      if ( i_mateltwise_desc->n % 8 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_32bit_8x8_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, i_gp_reg_scratch, i_mateltwise_desc->m % 8, i_mateltwise_desc->n % 8,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_rvv_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  /* For small matrices invoke scalar transpose */
  if ( (i_mateltwise_desc->m < 2) && (i_mateltwise_desc->n < 4) ) {
    libxsmm_generator_transform_norm_to_normt_mbit_scalar_rv64_microkernel( io_generated_code, io_loop_label_tracker,
                                                                          i_gp_reg_in, i_gp_reg_out,
                                                                          i_gp_reg_m_loop, i_gp_reg_n_loop,
                                                                          i_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    if ( i_mateltwise_desc->m >= 4 ) {
      /* open m loop */
      libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, (i_mateltwise_desc->m/4)*4 );

      if ( i_mateltwise_desc->n >= 4 ) {
        /* open n loop */
        libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/4)*4 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, LIBXSMM_RV64_GP_REG_X17, 4, 4,
                                                                                i_micro_kernel_config, i_mateltwise_desc );

        /* close n footer */
        libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );
      }

      if ( i_mateltwise_desc->n % 4 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, LIBXSMM_RV64_GP_REG_X17, 4, i_mateltwise_desc->n % 4,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }

      /* advance output pointer */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                       i_gp_reg_out, LIBXSMM_RV64_GP_REG_X28, i_gp_reg_out, (4LL * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) );

      /* advance input pointer */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                       i_gp_reg_in, LIBXSMM_RV64_GP_REG_X28, i_gp_reg_in, -1 * (((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - (4LL * i_micro_kernel_config->datatype_size_in)) );

      /* close m loop */
      libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, 4 );
    }

    if ( i_mateltwise_desc->m % 4 != 0 ) {
      if ( i_mateltwise_desc->n >= 4 ) {
        /* open n loop */
        libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n/4)*4 );

        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, LIBXSMM_RV64_GP_REG_X17, i_mateltwise_desc->m % 4, 4,
                                                                                i_micro_kernel_config, i_mateltwise_desc );

        /* close n footer */
        libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, 4 );
      }
      if ( i_mateltwise_desc->n % 4 != 0 ) {
        /* call shuffle network */
        libxsmm_generator_transform_norm_to_normt_64bit_4x4_shufflenetwork_rvv( io_generated_code, io_loop_label_tracker, i_gp_reg_in,
                                                                                i_gp_reg_out, LIBXSMM_RV64_GP_REG_X17, i_mateltwise_desc->m % 4, i_mateltwise_desc->n % 4,
                                                                                i_micro_kernel_config, i_mateltwise_desc );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_rv64_microkernel( libxsmm_generated_code*                  io_generated_code,
                                                   libxsmm_loop_label_tracker*              io_loop_label_tracker,
                                                   libxsmm_mateltwise_gp_reg_mapping*       i_gp_reg_mapping,
                                                   const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config,
                                                   const libxsmm_meltw_descriptor*          i_mateltwise_desc ) {
  int l_gp_reg_in;
  int l_gp_reg_out;
  int l_gp_reg_m_loop;
  int l_gp_reg_n_loop;
  int l_gp_reg_scratch;

  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));

  i_gp_reg_mapping->gp_reg_in        = LIBXSMM_RV64_GP_REG_X18;
  i_gp_reg_mapping->gp_reg_out       = LIBXSMM_RV64_GP_REG_X19;
  i_gp_reg_mapping->gp_reg_m_loop    = LIBXSMM_RV64_GP_REG_X20;
  i_gp_reg_mapping->gp_reg_n_loop    = LIBXSMM_RV64_GP_REG_X21;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_RV64_GP_REG_X22;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_RV64_GP_REG_X23;

  l_gp_reg_in       = i_gp_reg_mapping->gp_reg_in;
  l_gp_reg_out      = i_gp_reg_mapping->gp_reg_out;
  l_gp_reg_m_loop   = i_gp_reg_mapping->gp_reg_m_loop;
  l_gp_reg_n_loop   = i_gp_reg_mapping->gp_reg_n_loop;
  l_gp_reg_scratch  = i_gp_reg_mapping->gp_reg_scratch_0;

  /* load pointers from struct */
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
      i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in, l_offset_ptr_a );
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
      i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_out, l_offset_ptr_b );

  /* check leading dimnesions and sizes */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) ||
      (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T)    ) {
    /* coverity[copy_paste_error] */
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->n > i_mateltwise_desc->ldo) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2)     ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)         ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)         ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)          ) {
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
        (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      if ( (i_mateltwise_desc->m + i_mateltwise_desc->m%2) > i_mateltwise_desc->ldo ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
        return;
      }
    } else {
      if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
        return;
      }
    }
  } else {
    /* should not happen */
  }

  if ( ( LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
      ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_64bit_rvv_microkernel( io_generated_code, io_loop_label_tracker,
          l_gp_reg_in, l_gp_reg_out, l_gp_reg_m_loop, l_gp_reg_n_loop, l_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
      ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ) {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
      libxsmm_generator_transform_norm_to_normt_32bit_rvv_microkernel( io_generated_code, io_loop_label_tracker,
          l_gp_reg_in, l_gp_reg_out, l_gp_reg_m_loop, l_gp_reg_n_loop, l_gp_reg_scratch, i_micro_kernel_config, i_mateltwise_desc );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}
