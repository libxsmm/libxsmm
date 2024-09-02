/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_gemm_amx_microkernel.h"
#include "generator_gemm_amx.h"
#include "generator_mateltwise_transform_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common_x86.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_decompress_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, cnt_reg, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_decompress_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       n_iters) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, cnt_reg, 8);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, cnt_reg, n_iters );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_decompress_dyn_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       n_iters,
    unsigned int                       step) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, cnt_reg, step);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, cnt_reg, n_iters );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_fill_array_4_entries(int *array, int v0, int v1, int v2, int v3) {
  array[0] = v0;
  array[1] = v1;
  array[2] = v2;
  array[3] = v3;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_prefetch_tile_in_L2(libxsmm_generated_code*     io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int tile_cols,
    unsigned int LD,
    unsigned int base_reg,
    long long    offset) {
  unsigned int i;
  const char *const l_env_AB_prefetch_type = getenv("LIBXSMM_X86_AMX_GEMM_PRIMARY_PF_INPUTS_TYPE");
  LIBXSMM_UNUSED( i_micro_kernel_config );
  for (i=0; i<tile_cols; i++) {
    /* TODO: cacht overflow in offset */
    libxsmm_x86_instruction_prefetch(io_generated_code,
        (l_env_AB_prefetch_type == 0) ? LIBXSMM_X86_INSTR_PREFETCHT1 : ( atoi(l_env_AB_prefetch_type) > 0 ? LIBXSMM_X86_INSTR_PREFETCHT1 : LIBXSMM_X86_INSTR_PREFETCHT0 ),
        base_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        ((int)offset + i * LD * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_prefetch_tile_in_L1(libxsmm_generated_code*     io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int tile_cols,
    unsigned int LD,
    unsigned int base_reg,
    long long    offset) {
  unsigned int i;
  LIBXSMM_UNUSED( i_micro_kernel_config );
  for (i=0; i<tile_cols; i++) {
    /* TODO: cacht overflow in offset */
    libxsmm_x86_instruction_prefetch(io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0,
        base_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        ((int)offset + i * LD * 2 /*(i_micro_kernel_config->datatype_size/2)*/) );
  }
}
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_prefetch_output( libxsmm_generated_code*            io_generated_code,
    unsigned int                       gpr_base,
    unsigned int                       ldc,
    unsigned int                       dtype_size,
    unsigned int                       offset,
    int                                n_cols ) {
  unsigned int col = 0;
  for (col = 0; col < (unsigned int)n_cols; col++) {
    libxsmm_x86_instruction_prefetch(io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0,
        gpr_base,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        (int)((col * ldc * dtype_size + offset)) );
  }
}

LIBXSMM_API_INTERN
int libxsmm_is_tile_in_last_tilerow(const libxsmm_micro_kernel_config* i_micro_kernel_config, int tile) {
  int result = 0, i = 0;
  int max_im = 0, pos = 0;
  for (i = 0; i < 4; i++) {
    if (i_micro_kernel_config->_im[i] > max_im) {
      max_im = i_micro_kernel_config->_im[i];
    }
    if (i_micro_kernel_config->_C_tile_id[i] == tile ) {
      pos = i;
    }
  }
  if (i_micro_kernel_config->_im[pos] == max_im) {
    result = 1;
  } else {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_x86_cvtstore_tile_from_I32_to_F32( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols) {
  unsigned int col = 0, cur_vreg = 0;
  unsigned int gp_reg_gemm_scratch = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_0 : i_gp_reg_mapping->gp_reg_help_1;
  unsigned int reserved_zmms       = i_micro_kernel_config->reserved_zmms;
  /*unsigned int fused_eltwise       = ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_sigmoid == 1)) ? 1 : 0;*/
  unsigned int scf_vreg            = i_micro_kernel_config->scf_vreg;
  unsigned int batch_vstores       = (reserved_zmms + n_cols < 32) ? 1 : 0;
  int tile_in_last_tilerow         = libxsmm_is_tile_in_last_tilerow(i_micro_kernel_config, tile);
  int maskid                       = ((tile_in_last_tilerow > 0) && (i_micro_kernel_config->m_remainder > 0)) ? i_micro_kernel_config->mask_m_fp32 : 0;

  /* Check if we have to save the tmp registers */
  if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
  }
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );

  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, (i_micro_kernel_config->gemm_scratch_ld * 4/*l_micro_kernel_config.datatype_size*/)/4);
  libxsmm_x86_instruction_tile_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_TILESTORED,
      gp_reg_gemm_scratch,
      i_gp_reg_mapping->gp_reg_ldc,
      4,
      0,
      tile);
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, (i_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);

  for (col = 0; col < (unsigned int)n_cols; col++) {
    if (batch_vstores > 0) {
      cur_vreg = reserved_zmms + col;
    } else {
      cur_vreg = reserved_zmms + col % (32 - reserved_zmms);
    }
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        gp_reg_gemm_scratch,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        col * i_micro_kernel_config->gemm_scratch_ld * 4 ,
        i_micro_kernel_config->vector_name,
        cur_vreg, maskid, 1, 0 );

    /* Convert result to F32 */
    libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
        LIBXSMM_X86_INSTR_VCVTDQ2PS,
        i_micro_kernel_config->vector_name,
        cur_vreg,
        cur_vreg );

    /* Multiply with scaling factor */
    libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
        LIBXSMM_X86_INSTR_VMULPS,
        i_micro_kernel_config->vector_name,
        cur_vreg,
        scf_vreg,
        cur_vreg );

    /* Add C value in case beta == 1 */
    if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg_mapping->gp_reg_c,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ( ((in_offset+col) * i_xgemm_desc->ldc) + im_offset) * 4,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->aux_vreg, maskid, 1, 0 );

      libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
          LIBXSMM_X86_INSTR_VADDPS,
          i_micro_kernel_config->vector_name,
          cur_vreg,
          i_micro_kernel_config->aux_vreg,
          cur_vreg );
    }

    if (batch_vstores == 0) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg_mapping->gp_reg_c,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ( ((in_offset+col) * i_xgemm_desc->ldc) + im_offset) * 4,
          i_micro_kernel_config->vector_name,
          cur_vreg, maskid, 0, 1 );
    }
  }

  if (batch_vstores > 0) {
    for (col = 0; col < (unsigned int)n_cols; col++) {
      cur_vreg = reserved_zmms + col;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg_mapping->gp_reg_c,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ( ((in_offset+col) * i_xgemm_desc->ldc) + im_offset) * 4,
          i_micro_kernel_config->vector_name,
          cur_vreg, maskid, 0, 1 );
    }
  }

  if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_paired_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile0,
    int                                tile1,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols) {
  unsigned int col = 0, reg_0 = 0, reg_1 = 0, prev_reg_0 = 0, copy_prev_reg_0 = 0;
  unsigned int gp_reg_gemm_scratch = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_0 : i_gp_reg_mapping->gp_reg_help_1;
  unsigned int gp_reg_relu         = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_1 : i_gp_reg_mapping->gp_reg_help_0;
  unsigned int gp_reg_outptr       = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_1 : i_gp_reg_mapping->gp_reg_help_0;
  unsigned int fuse_relu           = i_micro_kernel_config->fused_relu;
  unsigned int fuse_relu_nobitmask = i_micro_kernel_config->fused_relu_nobitmask;
  unsigned int reserved_zmms       = i_micro_kernel_config->reserved_zmms;
  unsigned int max_unrolling       = 31;
  unsigned int eager_result_store  = 0;
  unsigned int reserved_mask_regs  = i_micro_kernel_config->reserved_mask_regs;
  unsigned int current_mask_reg    = 1;
  unsigned int overwrite_C         = i_micro_kernel_config->overwrite_C;
  unsigned int gp_reg_C            = (overwrite_C == 1) ? i_gp_reg_mapping->gp_reg_c : gp_reg_outptr;
  unsigned int gp_vnni_out_ext_buf = gp_reg_outptr;
  unsigned int vnni_cvt_output_ext_buf  = i_micro_kernel_config->vnni_cvt_output_ext_buf;
  unsigned int gp_reg_relu_bwd      = LIBXSMM_X86_GP_REG_R10;
  unsigned int fuse_relu_bwd        = i_micro_kernel_config->fused_relu_bwd;
  unsigned int is_output_bf16       = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ? 1 : 0;
  unsigned int is_output_f16        = (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ? 1 : 0;
  unsigned int out_tsize = (is_output_bf16 > 0 || is_output_f16 > 0) ? 2 : 4;
  int tile_in_last_tilerow         = libxsmm_is_tile_in_last_tilerow(i_micro_kernel_config, tile0);

  /* Check if we have to save the tmp registers */
  if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
  }

  if ((fuse_relu == 1) && (overwrite_C == 1)) {
    if ( (gp_reg_relu == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if ( (gp_reg_relu == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }

  if ((overwrite_C == 0) || (vnni_cvt_output_ext_buf == 1)) {
    if ( (gp_reg_outptr == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if ( (gp_reg_outptr == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }

  if (fuse_relu_bwd == 1) {
    libxsmm_x86_instruction_push_reg( io_generated_code, gp_reg_relu_bwd );
  }

  /* Load the gemm scratch/relu ptr */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );
  if (fuse_relu == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, gp_reg_relu );
  }
  if (vnni_cvt_output_ext_buf == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, gp_vnni_out_ext_buf );
  }
  if (fuse_relu_bwd == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR, gp_reg_relu_bwd);
  }

  /* Store FP32 tiles to scratch */
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, (i_micro_kernel_config->gemm_scratch_ld * 4/*l_micro_kernel_config.datatype_size*/)/4);
  libxsmm_x86_instruction_tile_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_TILESTORED,
      gp_reg_gemm_scratch,
      i_gp_reg_mapping->gp_reg_ldc,
      4,
      0,
      tile0);

  if (tile1 >= 0) {
    libxsmm_x86_instruction_tile_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_TILESTORED,
        gp_reg_gemm_scratch,
        i_gp_reg_mapping->gp_reg_ldc,
        4,
        i_micro_kernel_config->gemm_scratch_ld * n_cols * 4 ,
        tile1);
  }
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, (i_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);

  /* Fully unroll in N dimension */
  eager_result_store = ( ((unsigned int)n_cols > (max_unrolling - reserved_zmms)) || ((fuse_relu == 0) && (fuse_relu_bwd == 0)) || (tile1 < 0)) ? 1 : 0;

  for (col = 0; col < (unsigned int)n_cols; col++) {
    if (tile1 >= 0) {
      if (col + reserved_zmms < 16) {
        reg_0 = col % (16-reserved_zmms) + reserved_zmms;
      } else {
        reg_0 = 16 + ((col-16+reserved_zmms) % 15);
      }

      if (i_micro_kernel_config->fused_sigmoid == 1) {
        if (col + reserved_zmms < 16) {
          reg_1 = col % (16-reserved_zmms) + reserved_zmms + 16;
        } else {
          reg_1 = reg_0 + 1;
        }
      }
    } else {
      reg_0 = col % (16-reserved_zmms) + reserved_zmms;
      reg_1 = reg_0 + 1;
    }

    if (tile1 >= 0) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          gp_reg_gemm_scratch,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (col + n_cols) * i_micro_kernel_config->gemm_scratch_ld * 4 ,
          i_micro_kernel_config->vector_name,
          reg_0, i_micro_kernel_config->mask_m_fp32, 1, 0 );
    }

    /* In this case also save the result before doing any eltwise */
    if ((i_micro_kernel_config->fused_sigmoid == 1) && (overwrite_C == 0) ) {
      char vname = (char)((tile1 >= 0) ? i_micro_kernel_config->vector_name : (is_output_bf16 > 0 || is_output_f16 > 0) ? 'y' : 'z');
      unsigned int mask_id =  (tile1 >= 0) ? i_micro_kernel_config->mask_m_bf16 : ((tile_in_last_tilerow > 0) ? i_micro_kernel_config->mask_m_fp32 : 0);

      if ((i_micro_kernel_config->m_remainder == 0 && is_output_bf16 > 0) || (tile1 >= 0)) {
        if (is_output_bf16 > 0) {
          libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                                        i_micro_kernel_config->vector_name,
                                                        gp_reg_gemm_scratch,
                                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                        col * i_micro_kernel_config->gemm_scratch_ld * 4 , 0,
                                                        /* coverity[copy_paste_error] */
                                                        reg_0,
                                                        reg_1);
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            gp_reg_gemm_scratch,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            col * i_micro_kernel_config->gemm_scratch_ld * 4 ,
            i_micro_kernel_config->vector_name,
            reg_1, (tile_in_last_tilerow > 0) ? i_micro_kernel_config->mask_m_fp32 : 0, 1, 0 );
        if (is_output_bf16 > 0) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_1, reg_1 );
        }
        if (is_output_f16 > 0) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                                  i_micro_kernel_config->vector_name, reg_1, reg_1, 0, 1, 1, 0x00 );
        }
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          (is_output_bf16 > 0 || is_output_f16 > 0) ? LIBXSMM_X86_INSTR_VMOVDQU16 : LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg_mapping->gp_reg_c,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * out_tsize/*(i_micro_kernel_config->datatype_size/2)*/,
          vname,
          reg_1, mask_id, 0, 1 );
    }

    if (i_micro_kernel_config->fused_sigmoid == 1) {
      if (tile1 >= 0) {
        libxsmm_generator_sigmoid_ps_rational_78_avx512( io_generated_code, reg_0, i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom, i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi, i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0, i_micro_kernel_config->vec_c1, i_micro_kernel_config->vec_c2, i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d, i_micro_kernel_config->vec_c2_d, i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound, i_micro_kernel_config->vec_lo_bound, i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones, i_micro_kernel_config->vec_halves, 'z' );
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          gp_reg_gemm_scratch,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          col * i_micro_kernel_config->gemm_scratch_ld * 4 ,
          i_micro_kernel_config->vector_name,
          reg_1, (tile1 < 0) ? ((tile_in_last_tilerow > 0) ? i_micro_kernel_config->mask_m_fp32 : 0) : 0, 1, 0 );

      libxsmm_generator_sigmoid_ps_rational_78_avx512( io_generated_code, reg_1, i_micro_kernel_config->vec_x2,
          i_micro_kernel_config->vec_nom, i_micro_kernel_config->vec_denom,
          i_micro_kernel_config->mask_hi, i_micro_kernel_config->mask_lo,
          i_micro_kernel_config->vec_c0, i_micro_kernel_config->vec_c1, i_micro_kernel_config->vec_c2, i_micro_kernel_config->vec_c3,
          i_micro_kernel_config->vec_c1_d, i_micro_kernel_config->vec_c2_d, i_micro_kernel_config->vec_c3_d,
          i_micro_kernel_config->vec_hi_bound, i_micro_kernel_config->vec_lo_bound, i_micro_kernel_config->vec_ones,
          i_micro_kernel_config->vec_neg_ones, i_micro_kernel_config->vec_halves, 'z' );

      if (is_output_bf16 > 0) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                                  i_micro_kernel_config->vector_name, reg_1, reg_0, reg_0 );
      } else if (is_output_f16 > 0) {
        libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                                i_micro_kernel_config->vector_name, reg_1, reg_0, 0, 1, 1, 0x00 );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            LIBXSMM_X86_INSTR_VMOVDQU64,
            i_micro_kernel_config->vector_name,
            reg_1, LIBXSMM_X86_VEC_REG_UNDEF, reg_0 );
      }
    } else {
      char vname = (char)((tile1 >= 0) ? i_micro_kernel_config->vector_name : (is_output_bf16 > 0 || is_output_f16 > 0) ? 'y' : 'z');
      unsigned int mask_id =  (tile1 >= 0) ? i_micro_kernel_config->mask_m_bf16 : ((tile_in_last_tilerow > 0) ? i_micro_kernel_config->mask_m_fp32 : 0);

      if ((i_micro_kernel_config->m_remainder == 0 && is_output_bf16 > 0) || (tile1 >= 0)) {
        if (is_output_bf16 > 0) {
          libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                                        i_micro_kernel_config->vector_name,
                                                        gp_reg_gemm_scratch,
                                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                        col * i_micro_kernel_config->gemm_scratch_ld * 4 , 0,
                                                        reg_0,
                                                        reg_0);
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            gp_reg_gemm_scratch,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            col * i_micro_kernel_config->gemm_scratch_ld * 4 ,
            i_micro_kernel_config->vector_name,
            reg_0, ((tile_in_last_tilerow > 0) ? i_micro_kernel_config->mask_m_fp32 : 0), 1, 0 );
        if (is_output_bf16 > 0) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, i_micro_kernel_config->vector_name, reg_0, reg_0 );
        }
        if (is_output_f16 > 0) {
          libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                                  i_micro_kernel_config->vector_name, reg_0, reg_0, 0, 1, 1, 0x00 );
        }
      }

      /* Also store the result before any eltwise to original C */
      if (overwrite_C == 0) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            (is_output_bf16 > 0 || is_output_f16 > 0) ? LIBXSMM_X86_INSTR_VMOVDQU16 : LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * out_tsize /*(i_micro_kernel_config->datatype_size/2)*/,
            vname,
            reg_0, mask_id, 0, 1 );
      }
    }

    if (fuse_relu == 1) {
      current_mask_reg = reserved_mask_regs + (col % (8-reserved_mask_regs));

      libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
          (is_output_bf16 > 0 || is_output_f16 > 0) ? LIBXSMM_X86_INSTR_VPCMPW : LIBXSMM_X86_INSTR_VPCMPD,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->zero_reg,
          reg_0,
          current_mask_reg, 6 );

      /* Store relu mask */
      if ( overwrite_C == 1 ) {
        unsigned int mask_mov_instr = (tile1 >= 0) ? LIBXSMM_X86_INSTR_KMOVD_ST: LIBXSMM_X86_INSTR_KMOVW_ST;
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            mask_mov_instr,
            gp_reg_relu,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            ((in_offset+col) * i_xgemm_desc->ldcp/8 + LIBXSMM_UPDIV(im_offset,8)),
            current_mask_reg );
      }

      /* Blend output result with zero reg based on relu mask */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
          (is_output_bf16 > 0 || is_output_f16 > 0) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD,
          i_micro_kernel_config->vector_name,
          reg_0,
          i_micro_kernel_config->zero_reg,
          reg_0,
          current_mask_reg,
          0 );
    }

    if (fuse_relu_nobitmask == 1) {
      libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
          (is_output_bf16 > 0 || is_output_f16 > 0) ? LIBXSMM_X86_INSTR_VPMAXSW : LIBXSMM_X86_INSTR_VPMAXSD,
          i_micro_kernel_config->vector_name,
          reg_0,
          i_micro_kernel_config->zero_reg,
          reg_0);
    }

    if (fuse_relu_bwd == 1) {
      /* Load relu mask */
      unsigned int mask_mov_instr = (tile1 >= 0) ? LIBXSMM_X86_INSTR_KMOVD_LD: LIBXSMM_X86_INSTR_KMOVW_LD;
      current_mask_reg = reserved_mask_regs + (col % (8-reserved_mask_regs));
      libxsmm_x86_instruction_mask_move_mem( io_generated_code,
          mask_mov_instr,
          gp_reg_relu_bwd,
          LIBXSMM_X86_GP_REG_UNDEF,
          0,
          ((in_offset+col) * i_xgemm_desc->ldcp/8 + LIBXSMM_UPDIV(im_offset,8)),
          current_mask_reg );

      /* Blend output result with zero reg based on relu mask */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
          LIBXSMM_X86_INSTR_VPBLENDMW,
          i_micro_kernel_config->vector_name,
          reg_0,
          i_micro_kernel_config->zero_reg,
          reg_0,
          current_mask_reg,
          0 );
    }

    if (eager_result_store == 1) {
      char vname = (char)((tile1 >= 0) ? i_micro_kernel_config->vector_name : (is_output_bf16 > 0 || is_output_f16 > 0) ? 'y' : 'z');
      unsigned int mask_id =  (tile1 >= 0) ? i_micro_kernel_config->mask_m_bf16 : ((tile_in_last_tilerow > 0) ? i_micro_kernel_config->mask_m_fp32 : 0);

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          (is_output_bf16 > 0 || is_output_f16 > 0) ? LIBXSMM_X86_INSTR_VMOVDQU16 : LIBXSMM_X86_INSTR_VMOVUPS,
          gp_reg_C,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * out_tsize /*(i_micro_kernel_config->datatype_size/2)*/,
          vname,
          reg_0, mask_id, 0, 1 );
    }

    if ((vnni_cvt_output_ext_buf == 1) && (eager_result_store == 1)) {
      if (col % 2 == 1) {
        if ((col-1) + reserved_zmms < 16) {
          prev_reg_0 = (col-1) % (16-reserved_zmms) + reserved_zmms;
        } else {
          prev_reg_0 = 16 + (((col-1)-16+reserved_zmms) % 15);
        }
        copy_prev_reg_0 = (prev_reg_0 + 16 < 32) ? prev_reg_0 + 16 : 31;

        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
            LIBXSMM_X86_INSTR_VMOVDQU64,
            i_micro_kernel_config->vector_name,
            prev_reg_0, LIBXSMM_X86_VEC_REG_UNDEF, copy_prev_reg_0 );

        libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
            LIBXSMM_X86_INSTR_VPERMT2W,
            i_micro_kernel_config->vector_name,
            reg_0,
            i_micro_kernel_config->perm_table_vnni_lo,
            copy_prev_reg_0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            gp_vnni_out_ext_buf,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (((in_offset/2+col/2)) * i_xgemm_desc->ldc + im_offset) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
            i_micro_kernel_config->vector_name,
            copy_prev_reg_0, i_micro_kernel_config->mask_m_fp32, 0, 1 );

        libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
            LIBXSMM_X86_INSTR_VPERMT2W,
            i_micro_kernel_config->vector_name,
            reg_0,
            i_micro_kernel_config->perm_table_vnni_hi,
            prev_reg_0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            gp_vnni_out_ext_buf,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (((in_offset/2+col/2)) * i_xgemm_desc->ldc  + im_offset + 16) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
            i_micro_kernel_config->vector_name,
            prev_reg_0, i_micro_kernel_config->mask_m_fp32, 0, 1 );
      }
    }
  }

  /* Store all the downconverted results if we are in "lazy" store mode ... */
  if (eager_result_store == 0) {
    for (col = 0; col < (unsigned int)n_cols; col++) {
      if (col + reserved_zmms < 16) {
        reg_0 = col % (16-reserved_zmms) + reserved_zmms;
      } else {
        reg_0 = 16 + ((col-16+reserved_zmms) % 15);
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVDQU16,
          gp_reg_C,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
          i_micro_kernel_config->vector_name,
          reg_0, i_micro_kernel_config->mask_m_bf16, 0, 1 );

      if (vnni_cvt_output_ext_buf == 1) {
        if (col % 2 == 1) {
          if ((col-1) + reserved_zmms < 16) {
            prev_reg_0 = (col-1) % (16-reserved_zmms) + reserved_zmms;
          } else {
            prev_reg_0 = 16 + (((col-1)-16+reserved_zmms) % 15);
          }
          copy_prev_reg_0 = (prev_reg_0 + n_cols < 32) ? prev_reg_0 + n_cols : 31;

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VMOVDQU64,
              i_micro_kernel_config->vector_name,
              prev_reg_0, LIBXSMM_X86_VEC_REG_UNDEF, copy_prev_reg_0 );

          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VPERMT2W,
              i_micro_kernel_config->vector_name,
              reg_0,
              i_micro_kernel_config->perm_table_vnni_lo,
              copy_prev_reg_0);

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_vnni_out_ext_buf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (((in_offset/2+col/2)) * i_xgemm_desc->ldc + im_offset) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
              i_micro_kernel_config->vector_name,
              copy_prev_reg_0, i_micro_kernel_config->mask_m_fp32, 0, 1 );

          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VPERMT2W,
              i_micro_kernel_config->vector_name,
              reg_0,
              i_micro_kernel_config->perm_table_vnni_hi,
              prev_reg_0);

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              gp_vnni_out_ext_buf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (((in_offset/2+col/2)) * i_xgemm_desc->ldc  + im_offset + 16) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
              i_micro_kernel_config->vector_name,
              prev_reg_0, i_micro_kernel_config->mask_m_fp32, 0, 1 );
        }
      }
    }
  }

  /* Check if we have to restore the tmp registers */
  if (fuse_relu_bwd == 1) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, gp_reg_relu_bwd );
  }

  if ((fuse_relu == 1) && (overwrite_C == 1)) {
    if ( (gp_reg_relu == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if ( (gp_reg_relu == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }

  if ((overwrite_C == 0) || (vnni_cvt_output_ext_buf == 1)) {
    if ( (gp_reg_outptr == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if ( (gp_reg_outptr == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }

  if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_single_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols) {

  unsigned int col = 0, reg_0 = 0;
  unsigned int gp_reg_gemm_scratch = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_0 : i_gp_reg_mapping->gp_reg_help_1;
  unsigned int reserved_zmms       = i_micro_kernel_config->reserved_zmms;
  unsigned int fused_eltwise       = ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->fused_relu_nobitmask == 1) || (i_micro_kernel_config->fused_sigmoid == 1)) ? 1 : 0;
  int tile_in_last_tilerow         = libxsmm_is_tile_in_last_tilerow(i_micro_kernel_config, tile);
  int maskid                       = ((tile_in_last_tilerow > 0) && (i_micro_kernel_config->m_remainder > 0)) ? i_micro_kernel_config->mask_m_fp32 : 0;
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);

  if ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (fused_eltwise == 0)) ||
      (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || ((l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) {
    libxsmm_x86_instruction_tile_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_TILESTORED,
        i_gp_reg_mapping->gp_reg_c,
        i_gp_reg_mapping->gp_reg_ldc,
        4,
        (in_offset * i_xgemm_desc->ldc + im_offset) * 4 ,
        tile);
  } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
    libxsmm_x86_cvtstore_tile_from_I32_to_F32(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, tile, im_offset, in_offset, n_cols);
  } else {
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ||  LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      /* If we have some fusion, then we call the paired tilestore code generation with tile1 = -1 and we modify the tile1 manipulaiton */
      if (fused_eltwise == 1) {
        libxsmm_generator_gemm_amx_paired_tilestore( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, tile, -1, im_offset, in_offset, n_cols);
      } else {
        /* Potentially push aux register */
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }

        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, (i_micro_kernel_config->gemm_scratch_ld * 4/*l_micro_kernel_config.datatype_size*/)/4);
        libxsmm_x86_instruction_tile_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_TILESTORED,
            gp_reg_gemm_scratch,
            i_gp_reg_mapping->gp_reg_ldc,
            4,
            0,
            tile);
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_ldc, (i_xgemm_desc->ldc * 4/*l_micro_kernel_config.datatype_size*/)/4);

        if (i_micro_kernel_config->vnni_format_C == 0) {
          for (col = 0; col < (unsigned int)n_cols; col++) {
            reg_0 = col % (16-reserved_zmms) + reserved_zmms;

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                gp_reg_gemm_scratch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                col *i_micro_kernel_config->gemm_scratch_ld * 4 ,
                i_micro_kernel_config->vector_name,
                reg_0, maskid, 1, 0 );

            if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                        i_micro_kernel_config->vector_name, reg_0, reg_0 );
            } else if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                                      i_micro_kernel_config->vector_name, reg_0, reg_0, 0, 1, 1, 0x00 );
            } else {

            }

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
                'y',
                reg_0, maskid, 0, 1 );
          }
        } else {
          for (col = 0; col < (unsigned int)n_cols; col += 2) {
            unsigned int reg_1;
            reg_0 = col % (16-reserved_zmms) + reserved_zmms;
            reg_1 = reg_0 + 1;

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                gp_reg_gemm_scratch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (col + 1) * i_micro_kernel_config->gemm_scratch_ld * 4 ,
                i_micro_kernel_config->vector_name,
                reg_0, maskid, 1, 0 );

            if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
              if (i_micro_kernel_config->m_remainder == 0 || tile_in_last_tilerow == 0) {
                libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                                              i_micro_kernel_config->vector_name,
                                                              gp_reg_gemm_scratch,
                                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                              col * i_micro_kernel_config->gemm_scratch_ld * 4 , 0,
                                                              reg_0,
                                                              reg_0);
              } else {
                libxsmm_x86_instruction_vec_move( io_generated_code,
                    i_micro_kernel_config->instruction_set,
                    LIBXSMM_X86_INSTR_VMOVUPS,
                    gp_reg_gemm_scratch,
                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                    col * i_micro_kernel_config->gemm_scratch_ld * 4 ,
                    i_micro_kernel_config->vector_name,
                    reg_1, i_micro_kernel_config->mask_m_fp32, 1, 0 );
                libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16, i_micro_kernel_config->vector_name, reg_1, reg_0, reg_0 );

              }
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMW,
                                                       i_micro_kernel_config->vector_name,
                                                       reg_0,
                                                       i_micro_kernel_config->vnni_perm_reg,
                                                       reg_0);
            }
            if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_micro_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VMOVUPS,
                  gp_reg_gemm_scratch,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  col * i_micro_kernel_config->gemm_scratch_ld * 4 ,
                  i_micro_kernel_config->vector_name,
                  reg_1, maskid, 1, 0 );
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                                      i_micro_kernel_config->vector_name, reg_0, reg_0, 0, 1, 1, 0x00 );
              libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                                      i_micro_kernel_config->vector_name, reg_1, reg_1, 0, 1, 1, 0x00 );
              libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
                  LIBXSMM_X86_INSTR_VPERMT2W,
                  i_micro_kernel_config->vector_name,
                  reg_1,
                  i_micro_kernel_config->vnni_perm_reg,
                  reg_0);
            }

             libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (((in_offset+col)/2) * i_xgemm_desc->ldc + im_offset) * 2 * 2 /*(i_micro_kernel_config->datatype_size/2)*/,
                i_micro_kernel_config->vector_name,
                reg_0, maskid, 0, 1 );
          }
        }

        /* Potentially pop aux register */
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
      }
    } else {
      /* Should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}

/* TODO: catch overflows caused by a_offs */
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_decompress_32x32_A_block(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    long long                          a_offs,
    long long                          a_lookahead_offs,
    long long                          a_lookahead_br_index) {

  unsigned int expanded_cl, current_mask_reg, current_zmm;
  unsigned int reserved_mask_regs       = i_micro_kernel_config->reserved_mask_regs;
  unsigned int reserved_zmms            = i_micro_kernel_config->reserved_zmms;

  unsigned int n_elts_decompressed_reg  = i_gp_reg_mapping->gp_reg_help_0;
  unsigned int popcnt_reg               = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int decompress_loop_reg      = LIBXSMM_X86_GP_REG_R14;

  libxsmm_x86_instruction_push_reg( io_generated_code, n_elts_decompressed_reg);
  libxsmm_x86_instruction_push_reg( io_generated_code, popcnt_reg);
  libxsmm_x86_instruction_push_reg( io_generated_code, decompress_loop_reg);

  if (a_lookahead_br_index > 0) {
    unsigned int help_gpr = n_elts_decompressed_reg;
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_bitmap_a);
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_decompressed_a);
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a_offset,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        (int)(a_lookahead_br_index*8),
        i_gp_reg_mapping->gp_reg_a,
        0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_a, help_gpr);
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR, i_gp_reg_mapping->gp_reg_bitmap_a );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF, i_gp_reg_mapping->gp_reg_decompressed_a);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, help_gpr, i_micro_kernel_config->sparsity_factor_A);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, help_gpr, i_gp_reg_mapping->gp_reg_decompressed_a);
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, help_gpr, 4);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, help_gpr, i_gp_reg_mapping->gp_reg_bitmap_a);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a_base, i_gp_reg_mapping->gp_reg_a);
  }

  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, n_elts_decompressed_reg, 0);
  libxsmm_generator_gemm_header_decompress_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, decompress_loop_reg );

  for (expanded_cl = 0; expanded_cl < 8; expanded_cl++ ) {
    current_mask_reg = reserved_mask_regs + (expanded_cl % (8-reserved_mask_regs));
    current_zmm      = expanded_cl % (32-reserved_zmms) + reserved_zmms;

    /* Load bit mask for current expand operation */
    libxsmm_x86_instruction_mask_move_mem( io_generated_code,
        LIBXSMM_X86_INSTR_KMOVD_LD,
        i_gp_reg_mapping->gp_reg_bitmap_a,
        decompress_loop_reg,
        1,
        (int)((a_offs*i_micro_kernel_config->sparsity_factor_A)/16 + expanded_cl * 4 + (a_lookahead_offs * i_micro_kernel_config->sparsity_factor_A)/16),
        current_mask_reg );

    /* Expand operation */
    libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                                                 LIBXSMM_X86_INSTR_VPEXPANDW,
                                                 i_micro_kernel_config->vector_name,
                                                 i_gp_reg_mapping->gp_reg_a,
                                                 n_elts_decompressed_reg,
                                                 2,
                                                 (int)(a_offs + a_lookahead_offs),
                                                 0,
                                                 LIBXSMM_X86_VEC_REG_UNDEF,
                                                 current_zmm,
                                                 current_mask_reg,
                                                 1,
                                                 0);
    /* Move zmm to reg */
    libxsmm_x86_instruction_mask_move( io_generated_code,
      LIBXSMM_X86_INSTR_KMOVD_GPR_ST,
      popcnt_reg,
      current_mask_reg );

    /* Popcount */
    libxsmm_x86_instruction_alu_reg( io_generated_code,
        LIBXSMM_X86_INSTR_POPCNT,
        popcnt_reg,
        popcnt_reg);

    /* Adjust count of decompressed elements */
    libxsmm_x86_instruction_alu_reg( io_generated_code,
        LIBXSMM_X86_INSTR_ADDQ,
        popcnt_reg,
        n_elts_decompressed_reg);
  }

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SALQ, decompress_loop_reg, 1);

  for (expanded_cl = 0; expanded_cl < 8; expanded_cl++ ) {
    current_mask_reg = reserved_mask_regs + (expanded_cl % (8-reserved_mask_regs));
    current_zmm      = expanded_cl % (32-reserved_zmms) + reserved_zmms;

    /* Store zmm to scratch */
    libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VMOVUPS,
      i_gp_reg_mapping->gp_reg_decompressed_a,
      decompress_loop_reg, 8,
      (int)(a_offs * i_micro_kernel_config->sparsity_factor_A + expanded_cl * 64 + a_lookahead_offs * i_micro_kernel_config->sparsity_factor_A),
      i_micro_kernel_config->vector_name,
      current_zmm, 0, 0, 1 );
  }

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, decompress_loop_reg, 1);
  libxsmm_generator_gemm_footer_decompress_loop_amx( io_generated_code,  io_loop_label_tracker, i_micro_kernel_config, decompress_loop_reg, 128);

  if (a_lookahead_br_index > 0) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_decompressed_a);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_bitmap_a);
  }

  libxsmm_x86_instruction_pop_reg( io_generated_code, decompress_loop_reg);
  libxsmm_x86_instruction_pop_reg( io_generated_code, popcnt_reg);
  libxsmm_x86_instruction_pop_reg( io_generated_code, n_elts_decompressed_reg);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_decompress_Kx32_A_block_kloop(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    long long                          i_k_iter,
    unsigned int                       i_k_CL) {

  unsigned int expanded_cl, current_mask_reg, current_zmm;
  unsigned int reserved_mask_regs       = i_micro_kernel_config->reserved_mask_regs;
  unsigned int reserved_zmms            = i_micro_kernel_config->reserved_zmms;

  unsigned int n_elts_decompressed_reg  = i_gp_reg_mapping->gp_reg_help_0;
  unsigned int popcnt_reg               = i_gp_reg_mapping->gp_reg_help_1;
  unsigned int decompress_loop_reg      = LIBXSMM_X86_GP_REG_R14;

  unsigned int unroll_factor = ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ))) ? 1 : 2;
  unsigned int expand_dtype_size = ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ))) ? 1 : 2;
  unsigned int maskload_instruction = ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ))) ? LIBXSMM_X86_INSTR_KMOVQ_LD : LIBXSMM_X86_INSTR_KMOVD_LD;
  unsigned int maskmove_instruction = ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ))) ? LIBXSMM_X86_INSTR_KMOVQ_GPR_ST : LIBXSMM_X86_INSTR_KMOVD_GPR_ST;
  unsigned int expand_instruction = ((LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ))) ? LIBXSMM_X86_INSTR_VPEXPANDB : LIBXSMM_X86_INSTR_VPEXPANDW;

  libxsmm_x86_instruction_push_reg( io_generated_code, decompress_loop_reg);
  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, n_elts_decompressed_reg, 0);
  libxsmm_generator_gemm_header_decompress_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, decompress_loop_reg );

  for (expanded_cl = 0; expanded_cl < unroll_factor; expanded_cl++ ) {
    current_mask_reg = reserved_mask_regs + (expanded_cl % (8-reserved_mask_regs));
    current_zmm      = expanded_cl % (32-reserved_zmms) + reserved_zmms;

    /* Load bit mask for current expand operation */
    libxsmm_x86_instruction_mask_move_mem( io_generated_code,
        maskload_instruction,
        i_gp_reg_mapping->gp_reg_bitmap_a,
        decompress_loop_reg,
        1,
        expanded_cl * 4,
        current_mask_reg );

    libxsmm_x86_instruction_prefetch(io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0,
        i_gp_reg_mapping->gp_reg_bitmap_a,
        decompress_loop_reg, 1,
        expanded_cl * 4 +  16 * 64);

    /* Expand operation */
    libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                                                 expand_instruction,
                                                 i_micro_kernel_config->vector_name,
                                                 i_gp_reg_mapping->gp_reg_a,
                                                 n_elts_decompressed_reg,
                                                 expand_dtype_size,
                                                 0,
                                                 0,
                                                 LIBXSMM_X86_VEC_REG_UNDEF,
                                                 current_zmm,
                                                 current_mask_reg,
                                                 1,
                                                 0);

    libxsmm_x86_instruction_prefetch(io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT0,
        i_gp_reg_mapping->gp_reg_a,
        n_elts_decompressed_reg, expand_dtype_size,
        16 * 64);

    /* Move zmm to reg */
    libxsmm_x86_instruction_mask_move( io_generated_code,
      maskmove_instruction,
      popcnt_reg,
      current_mask_reg );

    /* Popcount */
    libxsmm_x86_instruction_alu_reg( io_generated_code,
        LIBXSMM_X86_INSTR_POPCNT,
        popcnt_reg,
        popcnt_reg);

    /* Adjust count of decompressed elements */
    libxsmm_x86_instruction_alu_reg( io_generated_code,
        LIBXSMM_X86_INSTR_ADDQ,
        popcnt_reg,
        n_elts_decompressed_reg);

    if (expand_dtype_size == 1) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, LIBXSMM_X86_INSTR_VEXTRACTI32X8, i_micro_kernel_config->vector_name, current_zmm, LIBXSMM_X86_VEC_REG_UNDEF, current_zmm+1, 0, 0, 0, 1 );
      libxsmm_generator_cvt_8bit_to_16bit_lut_prepped_regs_avx512( io_generated_code, i_micro_kernel_config->vector_name,
        current_zmm, current_zmm,
        i_micro_kernel_config->luth_reg0,
        i_micro_kernel_config->luth_reg1,
        i_micro_kernel_config->lutl_reg0,
        i_micro_kernel_config->lutl_reg1,
        i_micro_kernel_config->sign_reg,
        i_micro_kernel_config->blend_reg,
        i_micro_kernel_config->tmp_reg0,
        i_micro_kernel_config->tmp_reg1 );
      libxsmm_generator_cvt_8bit_to_16bit_lut_prepped_regs_avx512( io_generated_code, i_micro_kernel_config->vector_name,
        current_zmm+1, current_zmm+1,
        i_micro_kernel_config->luth_reg0,
        i_micro_kernel_config->luth_reg1,
        i_micro_kernel_config->lutl_reg0,
        i_micro_kernel_config->lutl_reg1,
        i_micro_kernel_config->sign_reg,
        i_micro_kernel_config->blend_reg,
        i_micro_kernel_config->tmp_reg0,
        i_micro_kernel_config->tmp_reg1 );
    }
  }

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SALQ, decompress_loop_reg, 1);

  for (expanded_cl = 0; expanded_cl < unroll_factor; expanded_cl++ ) {
    current_mask_reg = reserved_mask_regs + (expanded_cl % (8-reserved_mask_regs));
    current_zmm      = expanded_cl % (32-reserved_zmms) + reserved_zmms;

    /* Store zmm to scratch */
    libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VMOVUPS,
      i_gp_reg_mapping->gp_reg_decompressed_a,
      decompress_loop_reg, 8,
      expanded_cl * 64 + (i_k_iter % 2)*i_k_CL*32*2 + 64 * i_micro_kernel_config->gemm_scratch_ld * 4,
      i_micro_kernel_config->vector_name,
      current_zmm, 0, 0, 1 );
    if (expand_dtype_size == 1) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        i_gp_reg_mapping->gp_reg_decompressed_a,
        decompress_loop_reg, 8,
        1 * 64 + (i_k_iter % 2)*i_k_CL*32*2 + 64 * i_micro_kernel_config->gemm_scratch_ld * 4,
        i_micro_kernel_config->vector_name,
        current_zmm+1, 0, 0, 1 );
    }
  }

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, decompress_loop_reg, 1);

  libxsmm_generator_gemm_footer_decompress_dyn_loop_amx( io_generated_code,  io_loop_label_tracker, i_micro_kernel_config, decompress_loop_reg, (32*i_k_CL)/8, 8);

  /* Advance bitmap reg and compressed reg  */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_bitmap_a, (i_k_CL*32)/8);
  if (expand_dtype_size == 2) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SALQ, n_elts_decompressed_reg, 1);
  }
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, n_elts_decompressed_reg, i_gp_reg_mapping->gp_reg_a);
  libxsmm_x86_instruction_pop_reg( io_generated_code, decompress_loop_reg);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_normT_32x16_bf16_ext_buf(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_micro_kernel_config*       i_micro_kernel_config_gemm,
    unsigned int                       i_gp_reg_in,
    long long                          i_offset_in,
    long long                          i_offset_out) {

  int i = 0, reserved_zmms = i_micro_kernel_config_gemm->reserved_zmms;
  libxsmm_mateltwise_kernel_config  config_struct;
  libxsmm_meltw_descriptor          desc_struct;
  libxsmm_mateltwise_kernel_config  *i_micro_kernel_config = &config_struct;
  libxsmm_meltw_descriptor          *i_mateltwise_desc = &desc_struct;
  unsigned int i_mask_reg_0 = i_micro_kernel_config_gemm->norm_to_normT_mask_reg_0;
  unsigned int i_mask_reg_1 = i_micro_kernel_config_gemm->norm_to_normT_mask_reg_1;

  unsigned int i_gp_reg_zmm_scratch = LIBXSMM_X86_GP_REG_R9;
  unsigned int i_gp_reg_out         = LIBXSMM_X86_GP_REG_R10;
  unsigned int i_gp_reg_mask        = LIBXSMM_X86_GP_REG_R12;
  unsigned int i_gp_reg_m_loop      = LIBXSMM_X86_GP_REG_R13;
  unsigned int i_gp_reg_n_loop      = LIBXSMM_X86_GP_REG_RAX;

  /* Initialize mateltwise config struct and descriptor */
  i_micro_kernel_config->vector_name = 'z';
  i_micro_kernel_config->datatype_size_in = 2;
  i_micro_kernel_config->datatype_size_out = 2;
  i_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
  i_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
  i_mateltwise_desc->m  = 32;
  i_mateltwise_desc->n  = 16;
  i_mateltwise_desc->ldi= i_xgemm_desc->ldb;
  i_mateltwise_desc->ldo= i_xgemm_desc->n;

  /* Save gp registers */
  if (reserved_zmms > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_zmm_scratch);
  }
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_in );
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_out );
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mask );
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_m_loop );
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_n_loop );

  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config_gemm, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B,  i_gp_reg_out );
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config_gemm, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_zmm_scratch );

  if (i_offset_in > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_in, i_offset_in);
  }
  if (i_offset_out > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_out, i_offset_out);
  }

  /* Store reserved ZMMs if any */
  for (i = 0; i < reserved_zmms; i++) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config_gemm->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        i_gp_reg_zmm_scratch,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        i*64,
        i_micro_kernel_config_gemm->vector_name,
        i, 0, 1, 1 );
  }

  libxsmm_generator_transform_norm_to_normt_16bit_avx512_microkernel( io_generated_code, io_loop_label_tracker,
      i_gp_reg_in, i_gp_reg_out, i_gp_reg_m_loop, i_gp_reg_n_loop, i_gp_reg_mask, i_gp_reg_mask,
      i_mask_reg_0, i_mask_reg_1, 0, 0, 0, 0, 0,
      i_micro_kernel_config, i_mateltwise_desc );

  /* Restore reserved ZMMs if any */
  for (i = 0; i < reserved_zmms; i++) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config_gemm->instruction_set,
        LIBXSMM_X86_INSTR_VMOVUPS,
        i_gp_reg_zmm_scratch,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        i*64,
        i_micro_kernel_config_gemm->vector_name,
        i, 0, 1, 0 );
  }

  /* Store gp registers */
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_n_loop );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_m_loop  );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mask );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_out  );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_in  );
  if (reserved_zmms > 0) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_zmm_scratch );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_microkernel( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info,
    long long                          offset_A,
    long long                          offset_B,
    unsigned int                       is_last_k,
    long long                          i_brgemm_loop,
    unsigned int                       fully_unrolled_brloop  ) {
  int m_tiles = m_blocking_info->tiles;
  int n_tiles = n_blocking_info->tiles;
  int i, im, in;
  int pf_dist = 0;
  int pf_dist_l1 = 0;
  int emit_tilestores = ((i_brgemm_loop == (i_micro_kernel_config->cur_unroll_factor - 1)) && (is_last_k == 1) && (fully_unrolled_brloop == 1) && (i_micro_kernel_config->is_peeled_br_loop == 1)) ? 1 : (((is_last_k == 1) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ? 1 : 0);
  unsigned int l_enforce_Mx1_amx_tile_blocking = (libxsmm_cpuid_x86_amx_gemm_enforce_mx1_tile_blocking() > 0) ? 1 : (i_xgemm_desc->n <= 16) ? 1 : 0;
  int use_paired_tilestores = ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (m_tiles % 2 == 0) && (i_micro_kernel_config->vnni_format_C == 0) && (l_enforce_Mx1_amx_tile_blocking == 0)) ? 1 : 0;
  int n_CL_to_pf;
  unsigned int tile_compute_instr = 0;
  unsigned int gp_reg_a;
  const char *const l_env_pf_c_scratch_dist = getenv("LIBXSMM_X86_AMX_GEMM_OUTPUT_SCR_PF_DIST");
  const char *const l_env_pf_c_matrix_dist  = getenv("LIBXSMM_X86_AMX_GEMM_OUTPUT_MAT_PF_DIST");
  /*unsigned int prefetch_C_scratch       = ((i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_C_SCRATCH) > 0) ? 1 : 0;*/
  unsigned int prefetch_C_scratch       = (l_env_pf_c_scratch_dist == 0) ? 0 : 1;
  unsigned int prefetch_C_scratch_dist  = (l_env_pf_c_scratch_dist == 0) ? 4 : atoi(l_env_pf_c_scratch_dist);
  /*unsigned int prefetch_C_matrix        = ((i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_C) > 0) ? 1 : 0;*/
  unsigned int prefetch_C_matrix        = (l_env_pf_c_matrix_dist == 0) ? 0 : 1;
  unsigned int prefetch_C_matrix_dist   = (l_env_pf_c_matrix_dist == 0) ? 3 : atoi(l_env_pf_c_matrix_dist);
  unsigned int streaming_tileload = 0, l_tileload_instr = LIBXSMM_X86_INSTR_TILELOADD;
  const char *const env_streaming_tileload= getenv("LIBXSMM_X86_AMX_GEMM_STREAMING_TILELOAD");
  unsigned int decompress_via_bitmap = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0) ? 1 : 0;
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);

  /* Tiles in the kernel are organized as indicated below when we use 2x2 2D blocking
   *
   *       C            A        B
   *
   *    N0  N1          Ki     N0  N1
   *
   * M0 T0  T1      M0  T4     T6  T7  Ki
   *           +=           X
   * M1 T2  T3      M1  T5
   *
   * */


  /* Tiles in the kernel are organized as indicated below when we use 1x4 1D blocking
   *
   *            C                A               B
   *
   *    N0  N1  N2  N3           Ki     N0  N1  N2  N3
   *
   * M0 T0  T1  T2  T3  +=   M0  T4     T6  T6  T6  T7  Ki
   *
   *
   */
  /* Arrays that indicate the tile traversals in the microkernel */
  int _im[4] = { 0 };
  int _in[4] = { 0 };
  int _in_tileloads_B[4] = { 0 };
  int _A_tile_id[4] = { 0 };
  int _B_tile_id[4] = { 0 };
  int _C_tile_id[4] = { 0 };
  int _C_tile_mate_id[4] = { 0 };
  int _C_tile_done[4] = { 0 };
  int _A_tile_id_load[4] = { 0 };
  int _B_tile_id_load[4] = { 0 };
  int _A_tileload_instr[4] = { 0 };
  int _B_tileload_instr[4] = { 0 };
  int _C_tilecomp_instr[4] = { 0 };
  long long _A_offsets[4] = { 0 };
  long long _B_offsets[4] = { 0 };
  int _im_offset_prefix_sums[4] = { 0 };
  int _in_offset_prefix_sums[4] = { 0 };

  const char *const env_pf_dist = getenv("LIBXSMM_X86_AMX_GEMM_PRIMARY_PF_INPUTS_DIST");
  const char *const env_pf_dist_l1 = getenv("LIBXSMM_X86_AMX_GEMM_SECONDARY_PF_INPUTS_DIST");

  if ( 0 == env_streaming_tileload ) {
    l_tileload_instr = LIBXSMM_X86_INSTR_TILELOADD;
  } else {
    streaming_tileload = atoi(env_streaming_tileload);
    if (streaming_tileload > 0) {
      l_tileload_instr = LIBXSMM_X86_INSTR_TILELOADDT1;
    }
  }

  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }
  if ( 0 == env_pf_dist_l1 ) {
  } else {
    pf_dist_l1 = atoi(env_pf_dist_l1);
  }

  /* Disable prefetched if not strided BRGEMM... */
  if (i_brgemm_loop == -2) {
    pf_dist += i_micro_kernel_config->cur_unroll_factor + 2;
    pf_dist_l1 += i_micro_kernel_config->cur_unroll_factor + 2;
  }

  for (i = 1; i < 4; i++) {
    _im_offset_prefix_sums[i] = _im_offset_prefix_sums[i-1] + m_blocking_info->sizes[i-1];
  }
  for (i = 1; i < 4; i++) {
    _in_offset_prefix_sums[i] = _in_offset_prefix_sums[i-1] + n_blocking_info->sizes[i-1];
  }

  if ((m_tiles == 2) && (n_tiles == 2)) {
    libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tile_mate_id, 2, 3, 0, 1);
  } else if ((m_tiles == 2) && (n_tiles == 1)) {
    libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tile_mate_id, 2, -1, 0, -1);
    if (l_enforce_Mx1_amx_tile_blocking > 0) {
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tile_mate_id, 1, 0, -1, -1);
    }
  } else if ((m_tiles == 4) && (n_tiles == 1)) {
    libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tile_mate_id, 1, 0, 3, 2);
  } else {
    /* In this case we cannot do paired tilestores */
    libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tile_mate_id, -1, -1, -1, -1);
  }

  /* Pick the proper tile compute instruction */
  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) {
    tile_compute_instr = LIBXSMM_X86_INSTR_TDPBF16PS;
  } else if (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
    tile_compute_instr = LIBXSMM_X86_INSTR_TDPFP16PS;
  } else if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
    if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBUUD;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBUSD;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBSUD;
    } else {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBSSD;
    }
  } else {
    /* Should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Some checks for this functionality... */
  if (i_micro_kernel_config->vnni_cvt_output_ext_buf == 1) {
    if (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) {
      fprintf(stderr, "For now we support C norm->vnni to external buffer only when C output is in BF16...\n");
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    if (use_paired_tilestores == 0) {
      fprintf(stderr, "For now we support C norm->vnni to external buffer only when microkernel performs paired-tilestores...\n");
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0)  && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0)) {
    if (m_tiles == 2 && n_tiles == 2) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, 1, 0, 1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, 0, 1, 1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, 5, 4, 5);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, 5, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, 6, 7, 7);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, -1, 7, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, -1, 1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, l_tileload_instr, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, -1, l_tileload_instr, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

    if (m_tiles == 1 && n_tiles == 4) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, 0, 0, 0);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, 1, 2, 3);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, 4, 4, 4);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, 6, 6, 7);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, 6, 6, 7);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, 1, 2, 3);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, l_tileload_instr, l_tileload_instr, l_tileload_instr);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _in[i];
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

    if (m_tiles == 1 && n_tiles == 2) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, 0, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, 1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, 4, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, 7, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, 7, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, 1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, l_tileload_instr, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, -1, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

    if (m_tiles == 2 && n_tiles == 1) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, 1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, 0, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, 5, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, 5, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, 6, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, l_tileload_instr, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, -1, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        if (l_enforce_Mx1_amx_tile_blocking > 0) {
          _C_tile_id[i] = _im[i];
        } else {
          _C_tile_id[i] = _im[i] * 2 + _in[i];
        }
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

    if (m_tiles == 3 && n_tiles == 1) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, 1, 2, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, 0, 0, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, 4, 5, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, 4, 5, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, 6, 6, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, l_tileload_instr, l_tileload_instr, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i];
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

    if (m_tiles == 4 && n_tiles == 1) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, 1, 2, 3);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, 0, 0, 0);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, 4, 4, 5);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, 4, 4, 5);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, 6, 6, 6);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, l_tileload_instr, l_tileload_instr, l_tileload_instr);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i];
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

    if (m_tiles == 1 && n_tiles == 1) {
      /* Encode loop iterations */
      libxsmm_generator_gemm_amx_fill_array_4_entries(_im, 0, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in, 0, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id, 4, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tile_id_load, 4, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id, 6, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tile_id_load, 6, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_in_tileloads_B, 0, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_A_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_B_tileload_instr, l_tileload_instr, -1, -1, -1);
      libxsmm_generator_gemm_amx_fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, -1, -1, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + ((long long)_im_offset_prefix_sums[_im[i]] * 4 ) / i_micro_kernel_config->sparsity_factor_A;
        _B_offsets[i] = offset_B + ((long long)_in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 );
      }
    }

  }

  /* Now copy over the GEMM traversal meta-data for later usage by the store function if needed */
  i_micro_kernel_config->use_paired_tilestores = use_paired_tilestores;
  for (i = 0; i < 4; i++) {
    i_micro_kernel_config->_im[i]                     = _im[i];
    i_micro_kernel_config->_in[i]                     = _in[i];
    i_micro_kernel_config->_C_tile_id[i]              = _C_tile_id[i];
    i_micro_kernel_config->_C_tile_mate_id[i]         = _C_tile_mate_id[i];
    i_micro_kernel_config->_im_offset_prefix_sums[i]  = _im_offset_prefix_sums[i];
    i_micro_kernel_config->_in_offset_prefix_sums[i]  = _in_offset_prefix_sums[i];
  }

  for (i = 0; i < m_tiles*n_tiles; i++) {
    im = _im[i];
    in = _in[i];
    gp_reg_a = (i_micro_kernel_config->decompress_A == 0) ? i_gp_reg_mapping->gp_reg_a : i_gp_reg_mapping->gp_reg_decompressed_a;

    if (i_micro_kernel_config->decompress_A == 1) {
      if ((m_tiles == 1) && (_A_tile_id_load[i] > 0) && (_A_tile_id_load[i] % 2 == 0)) {
        libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_mloop, 16);
        libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JNE, i_micro_kernel_config->loop_label_id, i_micro_kernel_config->p_jump_label_tracker);
      }

      /* Decompress first block of A */
      if ((_A_tile_id_load[i] > 0) && (_A_tile_id_load[i] % 2 == 0) && (i_brgemm_loop <= 0)) {
        libxsmm_generator_gemm_amx_decompress_32x32_A_block(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, _A_offsets[i], 0, 0);
      }
      /* Check if SW pipelining is doable for the A decompression... */
      if ((_A_tile_id_load[i] > 0) && (_A_tile_id_load[i] % 2 == 0) && (i_brgemm_loop >= 0) && (i_brgemm_loop < i_micro_kernel_config->cur_unroll_factor - 1) && (fully_unrolled_brloop == 1)) {
        if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
          libxsmm_generator_gemm_amx_decompress_32x32_A_block(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, _A_offsets[i], i_xgemm_desc->c1, 0);
        } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
          libxsmm_generator_gemm_amx_decompress_32x32_A_block(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, _A_offsets[i], 0, i_brgemm_loop+1);
        }
      }

      if ((m_tiles == 1) && (_A_tile_id_load[i] > 0) && (_A_tile_id_load[i] % 2 == 0)) {
        libxsmm_x86_instruction_register_jump_label(io_generated_code, i_micro_kernel_config->loop_label_id, i_micro_kernel_config->p_jump_label_tracker);
        i_micro_kernel_config->loop_label_id++;
      }
    }

    if ((decompress_via_bitmap > 0) && (_A_tile_id_load[i] > 0) && (_A_tile_id_load[i] % 2 == 0)) {
      /* Decompress also the next 32x32 block  */
      unsigned int K_CL = (m_tiles == 1) ? 16 : 32;
      if (i_xgemm_desc->m == 8 || i_xgemm_desc->m == 4 || i_xgemm_desc->m == 2 || i_xgemm_desc->m == 1) {
        K_CL = K_CL/(16/i_xgemm_desc->m);
      } else {
        /* Do nothing */
      }

      if (i_brgemm_loop == 0) {
        libxsmm_generator_gemm_amx_decompress_Kx32_A_block_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_xgemm_desc, i_micro_kernel_config, i_brgemm_loop, K_CL);
      }
      if (is_last_k == 0) {
        libxsmm_generator_gemm_amx_decompress_Kx32_A_block_kloop(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_xgemm_desc, i_micro_kernel_config, i_brgemm_loop+1, K_CL);
      }
    }

    if (_A_tile_id_load[i] > 0) {
      /* TODO: catch int overdlow in A_offset */
      if (decompress_via_bitmap > 0) {
        unsigned int K_CL = (m_tiles == 1) ? 16 : 32;
        if (i_xgemm_desc->m == 8 || i_xgemm_desc->m == 4 || i_xgemm_desc->m == 2 || i_xgemm_desc->m == 1) {
          K_CL = K_CL/(16/i_xgemm_desc->m);
        } else {
          /* Do nothing */
        }
        libxsmm_x86_instruction_tile_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            _A_tileload_instr[i],
            i_gp_reg_mapping->gp_reg_decompressed_a,
            i_gp_reg_mapping->gp_reg_lda,
            4,
            (int)(_A_offsets[i] + (i_brgemm_loop % 2)*K_CL *32*2 + 64 * i_micro_kernel_config->gemm_scratch_ld * 4),
            _A_tile_id_load[i]);

      } else {
        libxsmm_x86_instruction_tile_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            _A_tileload_instr[i],
            gp_reg_a,
            i_gp_reg_mapping->gp_reg_lda,
            4,
            (int)(_A_offsets[i] * i_micro_kernel_config->sparsity_factor_A),
            _A_tile_id_load[i]);
      }

      if ((fully_unrolled_brloop == 1 && i_brgemm_loop >= 0 && pf_dist > 0) && (((i_brgemm_loop + pf_dist < i_micro_kernel_config->cur_unroll_factor) && (i_micro_kernel_config->is_peeled_br_loop == 1)) || i_micro_kernel_config->is_peeled_br_loop == 0)) {
        unsigned int n_tile_rows, n_tile_cols;
        libxsmm_get_tileinfo( _A_tile_id_load[i], &n_tile_rows, &n_tile_cols, &i_micro_kernel_config->tile_config );
        n_CL_to_pf = n_tile_cols;
        libxsmm_generator_gemm_amx_prefetch_tile_in_L2(  io_generated_code,
            i_micro_kernel_config,
            n_CL_to_pf,
            i_xgemm_desc->lda * 2,
            i_gp_reg_mapping->gp_reg_a,
            _A_offsets[i] + pf_dist * i_xgemm_desc->c1 );
      }
      if ((fully_unrolled_brloop == 1 && i_brgemm_loop >= 0 && pf_dist_l1 > 0) && (((i_brgemm_loop + pf_dist_l1 < i_micro_kernel_config->cur_unroll_factor) && (i_micro_kernel_config->is_peeled_br_loop == 1)) || i_micro_kernel_config->is_peeled_br_loop == 0)) {
        unsigned int n_tile_rows, n_tile_cols;
        libxsmm_get_tileinfo( _A_tile_id_load[i], &n_tile_rows, &n_tile_cols, &i_micro_kernel_config->tile_config );
        n_CL_to_pf = n_tile_cols;
        libxsmm_generator_gemm_amx_prefetch_tile_in_L1(  io_generated_code,
            i_micro_kernel_config,
            n_CL_to_pf,
            i_xgemm_desc->lda * 2,
            i_gp_reg_mapping->gp_reg_a,
            _A_offsets[i] + pf_dist_l1 * i_xgemm_desc->c1 );
      }
    }

    if (_B_tile_id_load[i] > 0) {
      /* TODO: catch int overdlow in B_offset */
      libxsmm_x86_instruction_tile_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            _B_tileload_instr[i],
            i_gp_reg_mapping->gp_reg_b,
            i_gp_reg_mapping->gp_reg_ldb,
            4,
            (int)_B_offsets[i],
            _B_tile_id_load[i]);

      if (i_micro_kernel_config->norm_to_normT_B_ext_buf == 1) {
        unsigned int _B_trans_offset = i_micro_kernel_config->B_offs_trans + i_micro_kernel_config->k_amx_microkernel * i_xgemm_desc->ldb  * i_micro_kernel_config->datatype_size_in  + _in_offset_prefix_sums[_in_tileloads_B[i]] * 2 /*(i_micro_kernel_config->datatype_size/2)*/;
        libxsmm_generator_gemm_amx_normT_32x16_bf16_ext_buf(io_generated_code, io_loop_label_tracker, i_xgemm_desc, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_b, _B_offsets[i], _B_trans_offset);
      }

      if ((fully_unrolled_brloop == 1 && i_brgemm_loop >= 0 && pf_dist > 0) && (((i_brgemm_loop + pf_dist < i_micro_kernel_config->cur_unroll_factor) && (i_micro_kernel_config->is_peeled_br_loop == 1)) || i_micro_kernel_config->is_peeled_br_loop == 0)) {
        unsigned int n_tile_rows, n_tile_cols;
        libxsmm_get_tileinfo( _B_tile_id_load[i], &n_tile_rows, &n_tile_cols, &i_micro_kernel_config->tile_config );
        n_CL_to_pf = n_tile_cols;
        libxsmm_generator_gemm_amx_prefetch_tile_in_L2(  io_generated_code,
            i_micro_kernel_config,
            n_CL_to_pf,
            i_xgemm_desc->ldb,
            i_gp_reg_mapping->gp_reg_b,
            _B_offsets[i] + pf_dist * i_xgemm_desc->c2 );
      }
      if ((fully_unrolled_brloop == 1 && i_brgemm_loop >= 0 && pf_dist_l1 > 0) && (((i_brgemm_loop + pf_dist_l1 < i_micro_kernel_config->cur_unroll_factor) && (i_micro_kernel_config->is_peeled_br_loop == 1)) || i_micro_kernel_config->is_peeled_br_loop == 0)) {
        unsigned int n_tile_rows, n_tile_cols;
        libxsmm_get_tileinfo( _B_tile_id_load[i], &n_tile_rows, &n_tile_cols, &i_micro_kernel_config->tile_config );
        n_CL_to_pf = n_tile_cols;
        libxsmm_generator_gemm_amx_prefetch_tile_in_L1(  io_generated_code,
            i_micro_kernel_config,
            n_CL_to_pf,
            i_xgemm_desc->ldb,
            i_gp_reg_mapping->gp_reg_b,
            _B_offsets[i] + pf_dist_l1 * i_xgemm_desc->c2 );
      }
    }

    libxsmm_x86_instruction_tile_compute( io_generated_code,
        i_micro_kernel_config->instruction_set,
        _C_tilecomp_instr[i],
        _A_tile_id[i],
        _B_tile_id[i],
        _C_tile_id[i]);

    if ((prefetch_C_scratch > 0) && (is_last_k == 1) &&  (i_micro_kernel_config->is_peeled_br_loop == 1)  &&
        (i_brgemm_loop + prefetch_C_scratch_dist == i_micro_kernel_config->cur_unroll_factor) &&
        (((i < 2) && (use_paired_tilestores == 1)) || ((i < 1) && (use_paired_tilestores == 0)))) {
      unsigned int offset = 0;
      unsigned int gp_reg_gemm_scratch = i_gp_reg_mapping->gp_reg_help_0;
      if ((use_paired_tilestores == 1) && (i == 1)) {
        offset = n_blocking_info->sizes[_in[0]] * i_micro_kernel_config->gemm_scratch_ld * 4;
      }
      libxsmm_x86_instruction_push_reg( io_generated_code, gp_reg_gemm_scratch );
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );
      libxsmm_generator_gemm_amx_prefetch_output( io_generated_code, gp_reg_gemm_scratch, i_micro_kernel_config->gemm_scratch_ld, 4, offset, n_blocking_info->sizes[in] );
      libxsmm_x86_instruction_pop_reg( io_generated_code, gp_reg_gemm_scratch);
    }

    _C_tile_done[_C_tile_id[i]] = 1;

    if ((prefetch_C_matrix > 0) && (is_last_k == 1) && (i_micro_kernel_config->is_peeled_br_loop == 1)  &&
        (i_brgemm_loop + prefetch_C_matrix_dist == i_micro_kernel_config->cur_unroll_factor)) {
      unsigned int offset = 0;
      if (use_paired_tilestores == 1) {
        if (_C_tile_done[_C_tile_mate_id[_C_tile_id[i]]] == 1) {
          int min_mate_C_id = (_C_tile_id[i] < _C_tile_mate_id[_C_tile_id[i]]) ? _C_tile_id[i] : _C_tile_mate_id[_C_tile_id[i]];
          int im_store = min_mate_C_id / n_tiles;
          int in_store = min_mate_C_id % n_tiles;
          offset = (_in_offset_prefix_sums[in_store] * i_xgemm_desc->ldc + _im_offset_prefix_sums[im_store]) * 2;
          libxsmm_generator_gemm_amx_prefetch_output( io_generated_code, i_gp_reg_mapping->gp_reg_c, i_xgemm_desc->ldc, 2, offset, n_blocking_info->sizes[in_store] );
        }
      } else {
        offset = (_in_offset_prefix_sums[in] * i_xgemm_desc->ldc + _im_offset_prefix_sums[im]) * 2;
        libxsmm_generator_gemm_amx_prefetch_output( io_generated_code, i_gp_reg_mapping->gp_reg_c, i_xgemm_desc->ldc, 2, offset, n_blocking_info->sizes[in] );
      }
    }

    if (emit_tilestores == 1) {
      if (use_paired_tilestores == 1) {
        /* If mate C tile is also ready, then two paired tilestore */
        if (_C_tile_done[_C_tile_mate_id[_C_tile_id[i]]] == 1) {
          int min_mate_C_id = (_C_tile_id[i] < _C_tile_mate_id[_C_tile_id[i]]) ? _C_tile_id[i] : _C_tile_mate_id[_C_tile_id[i]];
          int im_store = min_mate_C_id / n_tiles;
          int in_store = min_mate_C_id % n_tiles;
          libxsmm_generator_gemm_amx_paired_tilestore( io_generated_code,
              i_gp_reg_mapping,
              i_micro_kernel_config,
              i_xgemm_desc,
              min_mate_C_id,
              _C_tile_mate_id[min_mate_C_id],
              _im_offset_prefix_sums[im_store],
              _in_offset_prefix_sums[in_store],
              n_blocking_info->sizes[in_store]);
        }
      } else {
        libxsmm_generator_gemm_amx_single_tilestore( io_generated_code,
            i_gp_reg_mapping,
            i_micro_kernel_config,
            i_xgemm_desc,
            _C_tile_id[i],
            _im_offset_prefix_sums[im],
            _in_offset_prefix_sums[in],
            n_blocking_info->sizes[in]);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_k_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, cnt_reg, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_k_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       n_iters) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, cnt_reg, 1);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, cnt_reg, n_iters );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_kloop( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info,
    long long                          A_offs,
    long long                          B_offs,
    unsigned int                       fully_unrolled_brloop ) {

  unsigned int l_k_blocking = 16;
  unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_k_pack_factor = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) ? 2 : libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  unsigned int k;
  long long offset_A = 0;
  long long offset_B = 0;
  long long i_brgemm_loop = -2;
  int is_last_k = 0;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_lda = (l_is_Ai4_Bi8_gemm > 0 || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) ? m_blocking_info->blocking : i_xgemm_desc->lda;
  unsigned int l_a_dtype_size = (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) ? 2 : i_micro_kernel_config->datatype_size_in;

  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0) {
    l_k_blocking = 32;
  } else if (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) {
    l_k_blocking = 64;
  }

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
    i_brgemm_loop = (long long)i_micro_kernel_config->br_loop_index;
  }

  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) && (fully_unrolled_brloop == 1)) {
    i_brgemm_loop = (long long)i_micro_kernel_config->br_loop_index;
  }

  while (i_xgemm_desc->k % l_k_blocking != 0) {
    l_k_blocking -= l_k_pack_factor;
  }

  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0) && (i_xgemm_desc->k > 1024) && ((i_xgemm_desc->k/l_k_blocking) % 2 == 0)) {
    unsigned int k_gp_reg = LIBXSMM_X86_GP_REG_R9;
    /* Peel off first iteration */
    libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, 0, 0, 0, 0, fully_unrolled_brloop);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, (long long)l_k_blocking*i_micro_kernel_config->datatype_size_in2);

    /* K loop here */
    libxsmm_generator_gemm_header_k_loop_amx( io_generated_code,  io_loop_label_tracker, i_micro_kernel_config, k_gp_reg);

    libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, 0, 0, 0, 1, fully_unrolled_brloop);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, (long long)l_k_blocking*i_micro_kernel_config->datatype_size_in2);
    libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, 0, 0, 0, 2, fully_unrolled_brloop);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, (long long)l_k_blocking*i_micro_kernel_config->datatype_size_in2);

    libxsmm_generator_gemm_footer_k_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, k_gp_reg, ((i_xgemm_desc->k/l_k_blocking)-2)/2 );

    /* Peel off last iteration  */
    libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, 0, 0, 1, 1, fully_unrolled_brloop);
  } else {
    /* Limit unrolling in case of large number of trips counts */
    unsigned int l_unroll_kernel_limit = 256;
    unsigned int l_brgemm_unroll_factor = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0) ||
                                           ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0) ||
                                           ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0)) ? LIBXSMM_MAX(1,i_xgemm_desc->c3) : 1;
    unsigned int l_k_trips = (i_xgemm_desc->k + l_k_blocking - 1)/l_k_blocking;
    if (((l_k_trips * l_brgemm_unroll_factor) > l_unroll_kernel_limit) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) == 0)) {
      unsigned int l_k_gp_reg = i_gp_reg_mapping->gp_reg_help_1;
      unsigned int l_k_unroll_limit = 4;
      /* Adjust l_k_unroll_limit */
      while (l_k_unroll_limit * l_brgemm_unroll_factor > l_unroll_kernel_limit) {
        if (l_k_unroll_limit == 1) break;
        l_k_unroll_limit--;
      }
      while ((l_k_trips-1) % l_k_unroll_limit != 0) {
        l_k_unroll_limit--;
      }
      /* K loop here */
      is_last_k = 0;
      libxsmm_x86_instruction_push_reg( io_generated_code, l_k_gp_reg );
      libxsmm_generator_gemm_header_k_loop_amx( io_generated_code,  io_loop_label_tracker, i_micro_kernel_config, l_k_gp_reg);
      for (k = 0; k < l_k_unroll_limit * l_k_blocking; k+= l_k_blocking) {
        offset_A = ((long long)k * l_lda * l_a_dtype_size )/i_micro_kernel_config->sparsity_factor_A + A_offs;
        offset_B = (long long)k * i_micro_kernel_config->datatype_size_in2  + B_offs;
        libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, offset_A, offset_B, is_last_k, i_brgemm_loop, fully_unrolled_brloop);
      }
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)l_lda * l_k_unroll_limit * l_k_blocking * l_a_dtype_size )/i_micro_kernel_config->sparsity_factor_A);
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, (long long) l_k_unroll_limit * l_k_blocking * i_micro_kernel_config->datatype_size_in2);
      libxsmm_generator_gemm_footer_k_loop_amx( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, l_k_gp_reg, (l_k_trips-1) / l_k_unroll_limit );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)l_lda * l_a_dtype_size * (l_k_trips-1) * l_k_blocking)/i_micro_kernel_config->sparsity_factor_A);
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b, (long long)(l_k_trips-1) * l_k_blocking * i_micro_kernel_config->datatype_size_in2);
      libxsmm_x86_instruction_pop_reg( io_generated_code, l_k_gp_reg );
      /* Peel off last iteration */
      is_last_k = 1;
      k = (l_k_trips-1) * l_k_blocking;
      offset_A = ((long long)k * l_lda * l_a_dtype_size )/i_micro_kernel_config->sparsity_factor_A + A_offs;
      offset_B = (long long)k * i_micro_kernel_config->datatype_size_in2  + B_offs;
      libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, offset_A, offset_B, is_last_k, i_brgemm_loop, fully_unrolled_brloop);
    } else {
      /* For now fully unroll the k loop */
      for (k = 0; k < i_xgemm_desc->k; k+= l_k_blocking) {
        i_micro_kernel_config->k_amx_microkernel = k;
        is_last_k = (k + l_k_blocking >= i_xgemm_desc->k) ? 1 : 0;
        offset_A = ((long long)k * l_lda * l_a_dtype_size )/i_micro_kernel_config->sparsity_factor_A + A_offs;
        offset_B = (long long)k * i_micro_kernel_config->datatype_size_in2  + B_offs;
        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0) {
          i_brgemm_loop = k/l_k_blocking;
          offset_A = 0;
        }
        libxsmm_generator_gemm_amx_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, offset_A, offset_B, is_last_k, i_brgemm_loop, fully_unrolled_brloop);
      }
    }
  }
}

