/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_gemm_amx_microkernel.h"
#include "generator_x86_instructions.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
    const libxsmm_micro_kernel_config*             i_micro_kernel_config,
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
    const unsigned int                             i_vec_neg_ones
    ) {
  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        i_micro_kernel_config->vector_name,
                                        i_vec_x, i_vec_x, i_vec_x2 );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_hi_bound,
                                                i_vec_x,
                                                i_vec_x,
                                                17, i_mask_hi, 0 );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_lo_bound,
                                                i_vec_x,
                                                i_vec_x,
                                                30, i_mask_lo, 0 );

   libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_x2, i_vec_nom, LIBXSMM_X86_VEC_REG_UNDEF );

   libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c2, i_vec_c3, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c1, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c0, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        i_micro_kernel_config->vector_name,
                                        i_vec_nom, i_vec_x, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VADDPS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_x2, i_vec_c3_d, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c2_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c1_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c0, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VRCP14PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_denom, i_vec_denom, LIBXSMM_X86_VEC_REG_UNDEF );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        i_micro_kernel_config->vector_name,
                                        i_vec_denom, i_vec_nom, i_vec_x );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VBLENDMPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_x,
                                                i_vec_ones,
                                                i_vec_x,
                                                LIBXSMM_X86_IMM_UNDEF,
                                                i_mask_hi, 0 );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VBLENDMPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_x,
                                                i_vec_neg_ones,
                                                i_vec_x,
                                                LIBXSMM_X86_IMM_UNDEF,
                                                i_mask_lo, 0 );
}

LIBXSMM_API_INTERN
void fill_array_4_entries(int *array, int v0, int v1, int v2, int v3){
  array[0] = v0;
  array[1] = v1;
  array[2] = v2;
  array[3] = v3;
}

LIBXSMM_API_INTERN
void prefetch_tile_in_L2(libxsmm_generated_code*     io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int tile_cols,
    unsigned int LD,
    unsigned int base_reg,
    unsigned int offset) {
  unsigned int i;
  for (i=0; i<tile_cols; i++) {
    libxsmm_x86_instruction_prefetch(io_generated_code,
        LIBXSMM_X86_INSTR_PREFETCHT1,
        base_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        offset + i * LD * (i_micro_kernel_config->datatype_size/2));
  }
}

LIBXSMM_API_INTERN
void paired_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile0,
    int                                tile1,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols) {
  unsigned int col = 0, reg_0 = 0, reg_1 = 0;
  unsigned int gp_reg_gemm_scratch = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_0 : i_gp_reg_mapping->gp_reg_help_1;
  unsigned int gp_reg_relu         = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_1 : i_gp_reg_mapping->gp_reg_help_0;
  unsigned int gp_reg_outptr       = (i_micro_kernel_config->m_loop_exists == 0) ? i_gp_reg_mapping->gp_reg_help_1 : i_gp_reg_mapping->gp_reg_help_0;
  unsigned int fuse_relu           = i_micro_kernel_config->fused_relu ;
  unsigned int reserved_zmms       = i_micro_kernel_config->reserved_zmms;
  unsigned int max_unrolling       = 31;
  unsigned int eager_result_store  = 0;
  unsigned int reserved_mask_regs  = i_micro_kernel_config->reserved_mask_regs;
  unsigned int current_mask_reg    = 1;
  unsigned int overwrite_C         = i_micro_kernel_config->overwrite_C;
  unsigned int gp_reg_C            = (overwrite_C == 1) ? i_gp_reg_mapping->gp_reg_c : gp_reg_outptr;

  /* Check if we have to save the tmp registers  */
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

  if (overwrite_C == 0) {
    if ( (gp_reg_outptr == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if ( (gp_reg_outptr == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }

  /* Load the gemm scratch/relu ptr  */
  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );
  if (fuse_relu == 1) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, gp_reg_relu );
  }

  /* Store FP32 tiles to scratch  */
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
        i_xgemm_desc->ldc * n_cols * i_micro_kernel_config->datatype_size,
        tile1);
  }

  /* Fully unroll in N dimension  */
  eager_result_store = ( ((unsigned int)n_cols > (max_unrolling - reserved_zmms)) || (fuse_relu == 0) || (tile1 < 0) ) ? 1 : 0;

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
          (col + n_cols) * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
          i_micro_kernel_config->vector_name,
          reg_0, 0, 1, 0 );
    }

    /* In this case also save the result before doing any eltwise */
    if ((i_micro_kernel_config->fused_sigmoid == 1) && (overwrite_C == 0) ) {
      char vname = (char)((tile1 >= 0) ? i_micro_kernel_config->vector_name : 'y');
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                               0,
                                               gp_reg_gemm_scratch,
                                               LIBXSMM_X86_GP_REG_UNDEF, 0,
                                               col * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
                                               i_micro_kernel_config->vector_name,
                                               reg_0,
                                               reg_1);

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          i_gp_reg_mapping->gp_reg_c,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * (i_micro_kernel_config->datatype_size/2),
          vname,
          reg_1, 0, 1, 1 );
    }

    if (i_micro_kernel_config->fused_sigmoid == 1) {
      if (tile1 >= 0) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              reg_0, i_micro_kernel_config->vec_halves, reg_0 );

        libxsmm_generator_gemm_tanh_ps_rational_78_avx512(io_generated_code, i_micro_kernel_config, reg_0, i_micro_kernel_config->vec_x2,
          i_micro_kernel_config->vec_nom, i_micro_kernel_config->vec_denom, i_micro_kernel_config->mask_hi, i_micro_kernel_config->mask_lo,
          i_micro_kernel_config->vec_c0, i_micro_kernel_config->vec_c1, i_micro_kernel_config->vec_c2, i_micro_kernel_config->vec_c3,
          i_micro_kernel_config->vec_c1_d, i_micro_kernel_config->vec_c2_d, i_micro_kernel_config->vec_c3_d,
          i_micro_kernel_config->vec_hi_bound, i_micro_kernel_config->vec_lo_bound, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_neg_ones);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VADDPS,
                                              i_micro_kernel_config->vector_name,
                                              reg_0, i_micro_kernel_config->vec_ones, reg_0 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              reg_0, i_micro_kernel_config->vec_halves, reg_0 );
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          gp_reg_gemm_scratch,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          col * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
          i_micro_kernel_config->vector_name,
          reg_1, 0, 1, 0 );

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            LIBXSMM_X86_INSTR_VMULPS,
                                            i_micro_kernel_config->vector_name,
                                            reg_1, i_micro_kernel_config->vec_halves, reg_1 );

      libxsmm_generator_gemm_tanh_ps_rational_78_avx512(io_generated_code, i_micro_kernel_config, reg_1, i_micro_kernel_config->vec_x2,
        i_micro_kernel_config->vec_nom, i_micro_kernel_config->vec_denom, i_micro_kernel_config->mask_hi, i_micro_kernel_config->mask_lo,
        i_micro_kernel_config->vec_c0, i_micro_kernel_config->vec_c1, i_micro_kernel_config->vec_c2, i_micro_kernel_config->vec_c3,
        i_micro_kernel_config->vec_c1_d, i_micro_kernel_config->vec_c2_d, i_micro_kernel_config->vec_c3_d,
        i_micro_kernel_config->vec_hi_bound, i_micro_kernel_config->vec_lo_bound, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_neg_ones);

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            LIBXSMM_X86_INSTR_VADDPS,
                                            i_micro_kernel_config->vector_name,
                                            reg_1, i_micro_kernel_config->vec_ones, reg_1 );

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            LIBXSMM_X86_INSTR_VMULPS,
                                            i_micro_kernel_config->vector_name,
                                            reg_1, i_micro_kernel_config->vec_halves, reg_1 );

      libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
              i_micro_kernel_config->vector_name,
              reg_1, reg_0, reg_0, 0);

    } else {
      char vname = (char)((tile1 >= 0) ? i_micro_kernel_config->vector_name : 'y');
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                               0,
                                               gp_reg_gemm_scratch,
                                               LIBXSMM_X86_GP_REG_UNDEF, 0,
                                               col * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
                                               i_micro_kernel_config->vector_name,
                                               reg_0,
                                               reg_0);

      /* Also store the result before any eltwise to original C  */
      if (overwrite_C == 0) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * (i_micro_kernel_config->datatype_size/2),
            vname,
            reg_0, 0, 1, 1 );
      }
    }

    if (fuse_relu == 1) {
      current_mask_reg = reserved_mask_regs + (col % (8-reserved_mask_regs));

      libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VPCMPW,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->zero_reg,
          reg_0,
          reg_0,
          6, current_mask_reg, 0 );

      /* Store relu mask */
      if ( overwrite_C == 1 ) {
        unsigned int mask_mov_instr = (tile1 >= 0) ? LIBXSMM_X86_INSTR_KMOVD: LIBXSMM_X86_INSTR_KMOVW;
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            mask_mov_instr,
            gp_reg_relu,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            ((in_offset+col) * i_xgemm_desc->ldc + im_offset)/8,
            current_mask_reg,
            1 );
      }

      /* Blend output result with zero reg based on relu mask */
      libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VPBLENDMW,
          i_micro_kernel_config->vector_name,
          reg_0,
          i_micro_kernel_config->zero_reg,
          reg_0,
          LIBXSMM_X86_IMM_UNDEF,
          current_mask_reg,
          0 );
    }

    if (eager_result_store == 1) {
      char vname = (char)((tile1 >= 0) ? i_micro_kernel_config->vector_name : 'y');
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VMOVUPS,
          gp_reg_C,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * (i_micro_kernel_config->datatype_size/2),
          vname,
          reg_0, 0, 1, 1 );
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
          LIBXSMM_X86_INSTR_VMOVUPS,
          gp_reg_C,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * (i_micro_kernel_config->datatype_size/2),
          i_micro_kernel_config->vector_name,
          reg_0, 0, 1, 1 );
    }
  }

  /* Check if we have to restore the tmp registers  */
  if ((fuse_relu == 1) && (overwrite_C == 1)) {
    if ( (gp_reg_relu == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
    }
    if ( (gp_reg_relu == i_gp_reg_mapping->gp_reg_help_0) && (i_micro_kernel_config->m_loop_exists == 1)  ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    }
  }

  if (overwrite_C == 0) {
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
void single_tilestore( libxsmm_generated_code*            io_generated_code,
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
  unsigned int fused_eltwise       = ((i_micro_kernel_config->fused_relu == 1) || (i_micro_kernel_config->fused_sigmoid == 1)) ? 1 : 0;

  if (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) || LIBXSMM_GEMM_PRECISION_I32 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) {
    libxsmm_x86_instruction_tile_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_TILESTORED,
        i_gp_reg_mapping->gp_reg_c,
        i_gp_reg_mapping->gp_reg_ldc,
        4,
        (in_offset * i_xgemm_desc->ldc + im_offset) * i_micro_kernel_config->datatype_size,
        tile);
  } else {
    if (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) {
      /* If we have some fusion, then we call the paired tilestore code generation with tile1 = -1 and we modify the tile1 manipulaiton  */
      if (fused_eltwise == 1) {
        paired_tilestore( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, tile, -1, im_offset, in_offset, n_cols);
      } else {
        /* Potentially push aux register */
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }

        libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, gp_reg_gemm_scratch );

        libxsmm_x86_instruction_tile_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_TILESTORED,
            gp_reg_gemm_scratch,
            i_gp_reg_mapping->gp_reg_ldc,
            4,
            0,
            tile);

        if (i_micro_kernel_config->vnni_format_C == 0) {
          for (col = 0; col < (unsigned int)n_cols; col++) {
            reg_0 = col % (16-reserved_zmms) + reserved_zmms;

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                gp_reg_gemm_scratch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                col * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
                i_micro_kernel_config->vector_name,
                reg_0, 0, 1, 0 );

            libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                i_micro_kernel_config->vector_name,
                reg_0, LIBXSMM_X86_VEC_REG_UNDEF,
                reg_0,
                0);

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((in_offset+col) * i_xgemm_desc->ldc + im_offset) * (i_micro_kernel_config->datatype_size/2),
                'y',
                reg_0, 0, 0, 1 );
          }
        } else {
          for (col = 0; col < (unsigned int)n_cols; col += 2) {
            reg_0 = col % (32-reserved_zmms) + reserved_zmms;

            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                gp_reg_gemm_scratch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (col + 1) * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
                i_micro_kernel_config->vector_name,
                reg_0, 0, 1, 0 );

            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
                                                     0,
                                                     gp_reg_gemm_scratch,
                                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                     col * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size,
                                                     i_micro_kernel_config->vector_name,
                                                     reg_0,
                                                     reg_0);

            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VPERMW,
                                                     i_micro_kernel_config->vector_name,
                                                     reg_0,
                                                     i_micro_kernel_config->vnni_perm_reg,
                                                     reg_0);

             libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVUPS,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (((in_offset+col)/2) * i_xgemm_desc->ldc + im_offset) * 2 * (i_micro_kernel_config->datatype_size/2),
                i_micro_kernel_config->vector_name,
                reg_0, 0, 1, 1 );
          }
        }

        /* Potentially pop aux register */
        if ( (gp_reg_gemm_scratch == i_gp_reg_mapping->gp_reg_help_1) && (i_micro_kernel_config->n_loop_exists == 1)  ) {
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_1 );
        }
      }
    } else {
      /* Should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_microkernel( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info,
    unsigned int                       offset_A,
    unsigned int                       offset_B,
    unsigned int                       is_last_k,
    int                                i_brgemm_loop  ) {
  int m_tiles = m_blocking_info->tiles;
  int n_tiles = n_blocking_info->tiles;
  int i, im, in;
  int pf_dist = i_xgemm_desc->c3;
  int emit_tilestores = ((i_brgemm_loop == (i_xgemm_desc->c3 - 1)) && (is_last_k == 1)) ? 1 : 0;
  int use_paired_tilestores = ((LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) && (m_tiles % 2 == 0) && (i_micro_kernel_config->vnni_format_C == 0)) ? 1 : 0;
  int n_CL_to_pf;
  unsigned int tile_compute_instr = 0;

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
  int _A_offsets[4] = { 0 };
  int _B_offsets[4] = { 0 };
  int _im_offset_prefix_sums[4] = { 0 };
  int _in_offset_prefix_sums[4] = { 0 };

  const char *const env_pf_dist = getenv("PF_DIST");
  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }

  /* Disable prefetched if not strided BRGEMM...  */
  if (i_brgemm_loop == -2) {
    pf_dist += i_xgemm_desc->c3 + 2;
  }

  for (i = 1; i < 4; i++) {
    _im_offset_prefix_sums[i] = _im_offset_prefix_sums[i-1] + m_blocking_info->sizes[i-1];
  }
  for (i = 1; i < 4; i++) {
    _in_offset_prefix_sums[i] = _in_offset_prefix_sums[i-1] + n_blocking_info->sizes[i-1];
  }

  if ((m_tiles == 2) && (n_tiles == 2)) {
    fill_array_4_entries(_C_tile_mate_id, 2, 3, 0, 1);
  } else if ((m_tiles == 2) && (n_tiles == 1)) {
    fill_array_4_entries(_C_tile_mate_id, 2, -1, 0, -1);
  } else {
    /* In this case we can't do paired tilestores */
    fill_array_4_entries(_C_tile_mate_id, -1, -1, -1, -1);
  }

  /* Pick the proper tile compute instruction  */
  if (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) {
    tile_compute_instr = LIBXSMM_X86_INSTR_TDPBF16PS;
  } else if (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) {
    if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBUUD;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0) ) {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBSUD;
    } else if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBUSD;
    } else {
      tile_compute_instr = LIBXSMM_X86_INSTR_TDPBSSD;
    }
  } else {
    /* Should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  if (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0)  && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0)) {
    if (m_tiles == 2 && n_tiles == 2) {
      /* Encode loop iterations */
      fill_array_4_entries(_im, 0, 1, 0, 1);
      fill_array_4_entries(_in, 0, 0, 1, 1);
      fill_array_4_entries(_A_tile_id, 4, 5, 4, 5);
      fill_array_4_entries(_A_tile_id_load, 4, 5, -1, -1);
      fill_array_4_entries(_B_tile_id, 6, 6, 7, 7);
      fill_array_4_entries(_B_tile_id_load, 6, -1, 7, -1);
      fill_array_4_entries(_in_tileloads_B, 0, -1, 1, -1);
      fill_array_4_entries(_A_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, LIBXSMM_X86_INSTR_TILELOADD, -1, -1);
      fill_array_4_entries(_B_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, -1, LIBXSMM_X86_INSTR_TILELOADD, -1);
      fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + _im_offset_prefix_sums[_im[i]] * i_micro_kernel_config->datatype_size;
        _B_offsets[i] = offset_B + _in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
      }
    }

    if (m_tiles == 1 && n_tiles == 4) {
      /* Encode loop iterations */
      fill_array_4_entries(_im, 0, 0, 0, 0);
      fill_array_4_entries(_in, 0, 1, 2, 3);
      fill_array_4_entries(_A_tile_id, 4, 4, 4, 4);
      fill_array_4_entries(_A_tile_id_load, 4, -1, -1, -1);
      fill_array_4_entries(_B_tile_id, 6, 6, 6, 7);
      fill_array_4_entries(_B_tile_id_load, 6, 6, 6, 7);
      fill_array_4_entries(_in_tileloads_B, 0, 1, 2, 3);
      fill_array_4_entries(_A_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, -1, -1, -1);
      fill_array_4_entries(_B_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, LIBXSMM_X86_INSTR_TILELOADD, LIBXSMM_X86_INSTR_TILELOADD, LIBXSMM_X86_INSTR_TILELOADD);
      fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr, tile_compute_instr);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _in[i];
        _A_offsets[i] = offset_A + _im_offset_prefix_sums[_im[i]] * i_micro_kernel_config->datatype_size;
        _B_offsets[i] = offset_B + _in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
      }
    }

    if (m_tiles == 1 && n_tiles == 2) {
      /* Encode loop iterations */
      fill_array_4_entries(_im, 0, 0, -1, -1);
      fill_array_4_entries(_in, 0, 1, -1, -1);
      fill_array_4_entries(_A_tile_id, 4, 4, -1, -1);
      fill_array_4_entries(_A_tile_id_load, 4, -1, -1, -1);
      fill_array_4_entries(_B_tile_id, 6, 7, -1, -1);
      fill_array_4_entries(_B_tile_id_load, 6, 7, -1, -1);
      fill_array_4_entries(_in_tileloads_B, 0, 1, -1, -1);
      fill_array_4_entries(_A_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, -1, -1, -1);
      fill_array_4_entries(_B_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, LIBXSMM_X86_INSTR_TILELOADD, -1, -1);
      fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, -1, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + _im_offset_prefix_sums[_im[i]] * i_micro_kernel_config->datatype_size;
        _B_offsets[i] = offset_B + _in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
      }
    }

    if (m_tiles == 2 && n_tiles == 1) {
      /* Encode loop iterations */
      fill_array_4_entries(_im, 0, 1, -1, -1);
      fill_array_4_entries(_in, 0, 0, -1, -1);
      fill_array_4_entries(_A_tile_id, 4, 5, -1, -1);
      fill_array_4_entries(_A_tile_id_load, 4, 5, -1, -1);
      fill_array_4_entries(_B_tile_id, 6, 6, -1, -1);
      fill_array_4_entries(_B_tile_id_load, 6, -1, -1, -1);
      fill_array_4_entries(_in_tileloads_B, 0, -1, -1, -1);
      fill_array_4_entries(_A_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, LIBXSMM_X86_INSTR_TILELOADD, -1, -1);
      fill_array_4_entries(_B_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, -1, -1, -1);
      fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, tile_compute_instr, -1, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + _im_offset_prefix_sums[_im[i]] * i_micro_kernel_config->datatype_size;
        _B_offsets[i] = offset_B + _in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
      }
    }

    if (m_tiles == 1 && n_tiles == 1) {
      /* Encode loop iterations */
      fill_array_4_entries(_im, 0, -1, -1, -1);
      fill_array_4_entries(_in, 0, -1, -1, -1);
      fill_array_4_entries(_A_tile_id, 4, -1, -1, -1);
      fill_array_4_entries(_A_tile_id_load, 4, -1, -1, -1);
      fill_array_4_entries(_B_tile_id, 6, -1, -1, -1);
      fill_array_4_entries(_B_tile_id_load, 6, -1, -1, -1);
      fill_array_4_entries(_in_tileloads_B, 0, -1, -1, -1);
      fill_array_4_entries(_A_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, -1, -1, -1);
      fill_array_4_entries(_B_tileload_instr, LIBXSMM_X86_INSTR_TILELOADD, -1, -1, -1);
      fill_array_4_entries(_C_tilecomp_instr, tile_compute_instr, -1, -1, -1);
      /* Fill in the accumulator IDs properly and the A/B offsets*/
      for (i = 0; i < 4; i++) {
        _C_tile_id[i] = _im[i] * 2 + _in[i];
        _A_offsets[i] = offset_A + _im_offset_prefix_sums[_im[i]] * i_micro_kernel_config->datatype_size;
        _B_offsets[i] = offset_B + _in_offset_prefix_sums[_in_tileloads_B[i]] * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
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

    if (_A_tile_id_load[i] > 0) {
      libxsmm_x86_instruction_tile_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          _A_tileload_instr[i],
          i_gp_reg_mapping->gp_reg_a,
          i_gp_reg_mapping->gp_reg_lda,
          4,
          _A_offsets[i],
          _A_tile_id_load[i]);

      if (i_brgemm_loop + pf_dist < i_xgemm_desc->c3) {
        n_CL_to_pf = 16;
        prefetch_tile_in_L2(  io_generated_code,
            i_micro_kernel_config,
            n_CL_to_pf,
            i_xgemm_desc->lda * 2,
            i_gp_reg_mapping->gp_reg_a,
            (unsigned int)(_A_offsets[i] + pf_dist * i_xgemm_desc->c1));
      }
    }

    if (_B_tile_id_load[i] > 0) {
        libxsmm_x86_instruction_tile_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            _B_tileload_instr[i],
            i_gp_reg_mapping->gp_reg_b,
            i_gp_reg_mapping->gp_reg_ldb,
            4,
            _B_offsets[i],
            _B_tile_id_load[i]);

      if (i_brgemm_loop + pf_dist < i_xgemm_desc->c3) {
        n_CL_to_pf = 16;
        prefetch_tile_in_L2(  io_generated_code,
            i_micro_kernel_config,
            n_CL_to_pf,
            i_xgemm_desc->ldb * 2,
            i_gp_reg_mapping->gp_reg_b,
            (unsigned int)(_B_offsets[i] + pf_dist * i_xgemm_desc->c2));
      }
    }

    libxsmm_x86_instruction_tile_compute( io_generated_code,
        i_micro_kernel_config->instruction_set,
        _C_tilecomp_instr[i],
        _A_tile_id[i],
        _B_tile_id[i],
        _C_tile_id[i]);

    _C_tile_done[_C_tile_id[i]] = 1;

    if (emit_tilestores == 1) {
      if (use_paired_tilestores == 1) {
        /* If mate C tile is also ready, then two paired tilestore  */
        if (_C_tile_done[_C_tile_mate_id[_C_tile_id[i]]] == 1) {
          int min_mate_C_id = (_C_tile_id[i] < _C_tile_mate_id[_C_tile_id[i]]) ? _C_tile_id[i] : _C_tile_mate_id[_C_tile_id[i]];
          int im_store = min_mate_C_id / n_tiles;
          int in_store = min_mate_C_id % n_tiles;
          paired_tilestore( io_generated_code,
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
        single_tilestore( io_generated_code,
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
void libxsmm_generator_gemm_amx_kernel_kloop( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info,
    unsigned int                       A_offs,
    unsigned int                       B_offs ) {

  unsigned int l_k_blocking = 16;
  unsigned int k;
  unsigned int offset_A = 0;
  unsigned int offset_B = 0;
  int i_brgemm_loop = -2;
  int is_last_k = 0;
  LIBXSMM_UNUSED( io_loop_label_tracker );

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
    i_brgemm_loop = A_offs/((unsigned int)i_xgemm_desc->c1);
  }

  while (i_xgemm_desc->k % l_k_blocking != 0) {
    l_k_blocking--;
  }

  /* For now fully unroll the k loop  */
  for (k = 0; k < i_xgemm_desc->k; k+= l_k_blocking) {
    is_last_k = (k + l_k_blocking >= i_xgemm_desc->k) ? 1 : 0;
    offset_A = k * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size + A_offs;
    offset_B = k * i_micro_kernel_config->datatype_size + B_offs;
    libxsmm_generator_gemm_amx_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, n_blocking_info, m_blocking_info, offset_A, offset_B, is_last_k, i_brgemm_loop);
  }
}

