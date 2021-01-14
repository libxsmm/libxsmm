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

#include "generator_common_x86.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#define MN_LOOP_ORDER 0
#define NM_LOOP_ORDER 1
#define LOOP_TYPE_M 0
#define LOOP_TYPE_N 1

LIBXSMM_API_INTERN
void adjust_after_microkernel_addr_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg,
                                                 unsigned int                            i_adjust_instr,
                                                 unsigned int                            m_microkernel,
                                                 unsigned int                            n_microkernel,
                                                 unsigned int                            i_loop_type ) {

  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) )) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out) ? 1 : 0;

  if ((is_inp_gp_reg > 0) || (is_out_gp_reg > 0)) {
    unsigned int tsize  = (is_inp_gp_reg > 0) ? i_micro_kernel_config->datatype_size_in : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? i_mateltwise_desc->ldi: i_mateltwise_desc->ldo;

    if (i_loop_type == LOOP_TYPE_M) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, m_microkernel * tsize);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ld * n_microkernel * tsize);
    }
  } else {
    /* Advance relumasks if need be */
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, m_microkernel/8);
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (i_mateltwise_desc->ldo * n_microkernel)/8);
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, m_microkernel/8);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (i_mateltwise_desc->ldi * n_microkernel)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, m_microkernel * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in );
          }
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void adjust_in_microkernel_addr_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg,
                                                 unsigned int                            i_adjust_instr,
                                                 unsigned int                            i_adjust_param,
                                                 unsigned int                            i_loop_type ) {
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) )) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out) ? 1 : 0;

  if ((is_inp_gp_reg > 0) || (is_out_gp_reg > 0)) {
    unsigned int vlen   = (is_inp_gp_reg > 0) ? i_micro_kernel_config->vlen_in : i_micro_kernel_config->vlen_out;
    unsigned int tsize  = (is_inp_gp_reg > 0) ? i_micro_kernel_config->datatype_size_in : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? i_mateltwise_desc->ldi: i_mateltwise_desc->ldo;

    if (i_loop_type == LOOP_TYPE_M) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, vlen * i_adjust_param * tsize);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ld * i_adjust_param * tsize);
    }
  } else {
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, (i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, (i_mateltwise_desc->ldo * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (i_mateltwise_desc->ldi * i_adjust_param)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, i_micro_kernel_config->vlen_in * i_adjust_param * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, i_mateltwise_desc->ldi * i_adjust_param * i_micro_kernel_config->datatype_size_in );
          }
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_avx512_vlens(const libxsmm_meltw_descriptor* i_mateltwise_desc, libxsmm_mateltwise_kernel_config* i_micro_kernel_config) {
  /* First, determine the vlen compute based on the operation */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 )) {
    i_micro_kernel_config->vlen_comp = 64;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 )) {
    i_micro_kernel_config->vlen_comp = 32;
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ))   {
    i_micro_kernel_config->vlen_comp = 16;
  } else if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ))   {
    i_micro_kernel_config->vlen_comp = 8;
  }

  /* The vlen_in is aligned with the vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;

  /* The vlen_out depends on the output datatype */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    i_micro_kernel_config->vlen_out = 64;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {

    /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
    if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
        LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) )
    {
      i_micro_kernel_config->vlen_out= 16;
    } else {
      i_micro_kernel_config->vlen_out = 32;
    }
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   {
    i_micro_kernel_config->vlen_out = 16;
  } else if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   {
    i_micro_kernel_config->vlen_out = 8;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_M_N_blocking(unsigned int m, unsigned int n, unsigned int vlen, unsigned int *m_blocking, unsigned int *n_blocking) {
  /* The m blocking is done in chunks of vlen */
  unsigned int m_chunks = (m+vlen-1)/vlen;
  /* TODO: Make m chunk remainder depend on number of available zmm registers */
  unsigned int m_chunk_remainder = 8;
  unsigned int m_range, m_block_size, foo1, foo2;

  if (m % vlen == 0) {
    /* If there is not remainder in M, then we block M in order to limit block size */
    if (m_chunks > 32) {
      libxsmm_compute_equalized_blocking(m_chunks, (m_chunks+1)/2, &m_range, &m_block_size, &foo1, &foo2);
      *m_blocking = m_range * vlen;
    } else {
      *m_blocking = m;
    }
  } else {
    /* If there is remainder we make sure we can fully unroll the kernel with masks */
    if (m_chunks > 16) {
      *m_blocking = (m_chunks - m_chunk_remainder) * vlen;
    } else {
      *m_blocking = m;
    }
  }

  /* For now not any additional blocking in N */
  *n_blocking = n;
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_loop_order(const libxsmm_meltw_descriptor* i_mateltwise_desc, unsigned int *loop_order, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int *out_blocking, unsigned int *inner_blocking, unsigned int *out_bound, unsigned int *inner_bound) {
  unsigned int _loop_order = NM_LOOP_ORDER;

  /* TODO: Potentially reorder loops given the kernel type */
  *loop_order = _loop_order;

  if (_loop_order == NM_LOOP_ORDER) {
    *out_blocking = *n_blocking;
    *out_bound = i_mateltwise_desc->n;
    *inner_blocking = *m_blocking;
    *inner_bound = i_mateltwise_desc->m;
  } else {
    *out_blocking = *m_blocking;
    *out_bound = i_mateltwise_desc->m;
    *inner_blocking = *n_blocking;
    *inner_bound = i_mateltwise_desc->n;
  }
}

LIBXSMM_API_INTERN
void libxsmm_load_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  char vname = (i_vlen * i_micro_kernel_config->datatype_size_in == 64) ? 'z' : 'y';

  /* In this case we don't have to load any data  */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) return;

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * i_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          vname,
          cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1, 0 );

      /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
      if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype)) ||
           (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype  ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype)) ) {
        libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_store_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  char vname = (i_vlen * i_micro_kernel_config->datatype_size_out == 64) ? 'z' : 'y';

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      /* In the XOR case we have a constnt vreg  */
      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) {
        cur_vreg = i_micro_kernel_config->zero_vreg;
      }

      /* If compute is in F32 and output is BF16 (or input is F32 and output is BF16), then downconvert BF16 -> FP32 */
      if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
           LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
        if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX ) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                    i_micro_kernel_config->vector_name,
                                                    cur_vreg, cur_vreg );
        } else {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                               cur_vreg, cur_vreg,
                                                               i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1);
        }
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          vname,
          cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0, 1 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_op( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(i_vlen);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_X2) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg, cur_vreg );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VXORPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->neg_signs_vreg, cur_vreg );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
        /* TODO:rewrite once added missing instruction */
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,LIBXSMM_X86_INSTR_VRSQRT14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,LIBXSMM_X86_INSTR_VRCP14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ) {
        libxsmm_generator_tanh_ps_rational_78_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones);

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VFNMSUB213PS, 'z', i_micro_kernel_config->vec_neg_ones, cur_vreg, cur_vreg );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
        libxsmm_generator_sigmoid_ps_rational_78_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones,
            i_micro_kernel_config->vec_halves );

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VSUBPS, 'z', cur_vreg, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_x2 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS, 'z', i_micro_kernel_config->vec_x2, cur_vreg, cur_vreg );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
        libxsmm_generator_gelu_ps_minimax3_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
        libxsmm_generator_gelu_inv_ps_minimax3_avx512( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_relu( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ) ? 1 : 0;
      unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
      unsigned int cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
      unsigned int l_vcmp_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPCMPW : LIBXSMM_X86_INSTR_VCMPPS;
      unsigned int l_vblend_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD;
      unsigned int l_mask_st_instr = ( l_bf16_compute > 0  ) ? LIBXSMM_X86_INSTR_KMOVD_ST : LIBXSMM_X86_INSTR_KMOVW_ST;
      unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      /* Compare to generate mask  */
      libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, l_vcmp_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 6 );

      /* Store mask relu  */
      if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) > 0 ) {
        libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_st_instr, i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_mateltwise_desc->ldo)/8, cur_mask_reg );
      }

      /* Blend output result with zero reg based on relu mask */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vblend_instr, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 0 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_relu_inv( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ) ? 1 : 0;
      unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
      unsigned int cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
      unsigned int l_vcmp_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPCMPW : LIBXSMM_X86_INSTR_VCMPPS;
      unsigned int l_vblend_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD;
      unsigned int l_mask_ld_instr = ( l_bf16_compute > 0  ) ? LIBXSMM_X86_INSTR_KMOVD_LD : LIBXSMM_X86_INSTR_KMOVW_LD;
      unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) > 0 ) {
        libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_ld_instr, i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_X86_GP_REG_UNDEF,  0, (im * l_vlen + in * i_mateltwise_desc->ldi)/8,  cur_mask_reg );
      } else {
        if ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
             (l_bf16_compute == 0)     ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_relumask,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          'y',
          i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 1, 0 );

          libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_relumask,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 1, 0 );
        }

        /* Compare to generate mask  */
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
          l_vcmp_instr,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->zero_vreg,
          i_micro_kernel_config->tmp_vreg,
          cur_mask_reg,
          6 );
      }

      /* Blend output result with zero reg based on relu mask */
      libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
          l_vblend_instr,
          i_micro_kernel_config->vector_name,
          cur_vreg,
          i_micro_kernel_config->zero_vreg,
          cur_vreg,
          cur_mask_reg,
          0 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_binary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int binary_op_instr = 0;

  LIBXSMM_UNUSED(i_vlen);

  switch (i_mateltwise_desc->param) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = LIBXSMM_X86_INSTR_VADDPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = LIBXSMM_X86_INSTR_VMULPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = LIBXSMM_X86_INSTR_VSUBPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = LIBXSMM_X86_INSTR_VDIVPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MULADD: {
      binary_op_instr = LIBXSMM_X86_INSTR_VFMADD213PS;
    } break;
    default:;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ) ? 1 : 0;
      unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in2,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          'y',
          i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 1, 0 );

        libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in2,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 1, 0 );
      }

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            'y',
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 1, 0 );

          libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 1, 0 );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, binary_op_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, cur_vreg );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, binary_op_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, cur_vreg, cur_vreg );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_binary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_start_vreg,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    switch (i_mateltwise_desc->param) {
      case LIBXSMM_MELTW_TYPE_UNARY_TANH:
      case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID:
      case LIBXSMM_MELTW_TYPE_UNARY_TANH_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_GELU:
      case LIBXSMM_MELTW_TYPE_UNARY_GELU_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_SQRT:
      case LIBXSMM_MELTW_TYPE_UNARY_NEGATE:
      case LIBXSMM_MELTW_TYPE_UNARY_X2: {
        libxsmm_compute_unary_2d_reg_block_op( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_RELU: {
        libxsmm_compute_unary_2d_reg_block_relu( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_RELU_INV: {
        libxsmm_compute_unary_2d_reg_block_relu_inv( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      default: /* Perform no compute */ ;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    libxsmm_compute_binary_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
  }
}

LIBXSMM_API_INTERN
void libxsmm_setup_input_output_masks( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_tmp_reg,
                                                 unsigned int                            i_m,
                                                 unsigned int*                           i_use_m_input_masking,
                                                 unsigned int*                           i_mask_reg_in,
                                                 unsigned int*                           i_use_m_output_masking,
                                                 unsigned int*                           i_mask_reg_out) {

  unsigned int mask_in_count, mask_out_count, mask_reg_in = 0, mask_reg_out = 0, use_m_input_masking, use_m_output_masking;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;

  use_m_input_masking   = (i_m % i_vlen_in == 0 ) ? 0 : 1;
  use_m_output_masking  = (i_m % i_vlen_out == 0 ) ? 0 : 1;

  if (use_m_input_masking == 1) {
    mask_in_count = i_vlen_in - i_m % i_vlen_in;
    mask_reg_in   = reserved_mask_regs;
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_in, mask_in_count, LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype));
    reserved_mask_regs++;
  }

  if (use_m_output_masking == 1) {
    if (i_vlen_in == i_vlen_out) {
      mask_reg_out = mask_reg_in;
    } else {
      mask_out_count = i_vlen_out - i_m % i_vlen_out;
      mask_reg_out   = reserved_mask_regs;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_out, mask_out_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
      reserved_mask_regs++;
    }
  }

  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
  *i_mask_reg_in = mask_reg_in;
  *i_use_m_input_masking = use_m_input_masking;
  *i_mask_reg_out = mask_reg_out;
  *i_use_m_output_masking = use_m_output_masking;
}

LIBXSMM_API_INTERN
void libxsmm_configure_microkernel_loops( libxsmm_generated_code*                        io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            i_use_m_input_masking,
                                                 unsigned int*                           i_m_trips,
                                                 unsigned int*                           i_n_trips,
                                                 unsigned int*                           i_m_unroll_factor,
                                                 unsigned int*                           i_n_unroll_factor,
                                                 unsigned int*                           i_m_assm_trips,
                                                 unsigned int*                           i_n_assm_trips,
                                                 unsigned int*                           i_out_loop_trips,
                                                 unsigned int*                           i_inner_loop_trips,
                                                 unsigned int*                           i_out_loop_bound,
                                                 unsigned int*                           i_inner_loop_bound,
                                                 unsigned int*                           i_out_loop_reg,
                                                 unsigned int*                           i_inner_loop_reg,
                                                 unsigned int*                           i_out_unroll_factor,
                                                 unsigned int*                           i_inner_unroll_factor) {

  unsigned int m_trips, n_trips, m_unroll_factor, n_unroll_factor, m_assm_trips, n_assm_trips, out_loop_trips, inner_loop_trips, out_loop_bound, inner_loop_bound, out_loop_reg, inner_loop_reg, out_unroll_factor, inner_unroll_factor;
  unsigned int max_nm_unrolling = 32;
  unsigned int i_loop_order = i_micro_kernel_config->loop_order;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_mateltwise_desc);

  m_trips               = (i_m + i_vlen_in - 1) / i_vlen_in;
  n_trips               = i_n;

  max_nm_unrolling  = max_nm_unrolling - reserved_zmms;

  if (i_use_m_input_masking == 1) {
    m_unroll_factor = m_trips;
  } else {
    m_unroll_factor = LIBXSMM_MIN(m_trips,16);
  }

  if (m_unroll_factor > max_nm_unrolling) {
    m_unroll_factor = max_nm_unrolling;
  }

  while (m_trips % m_unroll_factor != 0) {
    m_unroll_factor--;
  }

  n_unroll_factor = n_trips;
  while (m_unroll_factor * n_unroll_factor > max_nm_unrolling) {
    n_unroll_factor--;
  }

  while (n_trips % n_unroll_factor != 0) {
    n_unroll_factor--;
  }

  m_assm_trips = m_trips/m_unroll_factor;
  n_assm_trips = n_trips/n_unroll_factor;

  out_loop_trips      = (i_loop_order == NM_LOOP_ORDER) ? n_assm_trips : m_assm_trips;
  out_loop_bound      = (i_loop_order == NM_LOOP_ORDER) ? n_trips : m_trips;
  out_loop_reg        = (i_loop_order == NM_LOOP_ORDER) ? i_gp_reg_mapping->gp_reg_n_loop : i_gp_reg_mapping->gp_reg_m_loop;
  out_unroll_factor   = (i_loop_order == NM_LOOP_ORDER) ? n_unroll_factor : m_unroll_factor;

  inner_loop_trips    = (i_loop_order == MN_LOOP_ORDER) ? n_assm_trips : m_assm_trips;
  inner_loop_bound    = (i_loop_order == MN_LOOP_ORDER) ? n_trips : m_trips;
  inner_loop_reg      = (i_loop_order == MN_LOOP_ORDER) ? i_gp_reg_mapping->gp_reg_n_loop : i_gp_reg_mapping->gp_reg_m_loop;
  inner_unroll_factor = (i_loop_order == MN_LOOP_ORDER) ? n_unroll_factor : m_unroll_factor;

  *i_m_trips = m_trips;
  *i_n_trips = n_trips;
  *i_m_unroll_factor = m_unroll_factor;
  *i_n_unroll_factor = n_unroll_factor;
  *i_m_assm_trips = m_assm_trips;
  *i_n_assm_trips = n_assm_trips;
  *i_out_loop_trips = out_loop_trips;
  *i_inner_loop_trips = inner_loop_trips;
  *i_out_loop_bound = out_loop_bound;
  *i_inner_loop_bound = inner_loop_bound;
  *i_out_loop_reg = out_loop_reg;
  *i_inner_loop_reg = inner_loop_reg;
  *i_out_unroll_factor = out_unroll_factor;
  *i_inner_unroll_factor = inner_unroll_factor;
}

LIBXSMM_API_INTERN
void libxsmm_configure_kernel_vregs_masks( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg_tmp) {
  /* initialize some values */
  i_micro_kernel_config->reserved_zmms = 0;
  i_micro_kernel_config->reserved_mask_regs = 1;
  i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;

  /* if we need FP32->BF16 downconverts and we don't have native instruction, then prepare stack */
  if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
       LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX)) {
    i_micro_kernel_config->use_fp32bf16_cvt_replacement = 1;
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );
    i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 2;
    i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
      i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

      /* Set zero register needed for relu  */
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg );

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK) == 0) ) {
        i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      }
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
      i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg );
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
      float neg_array[16] = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
      i_micro_kernel_config->neg_signs_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) neg_array, "neg_array", 'z', i_micro_kernel_config->neg_signs_vreg );
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
      unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
      reserved_zmms += 14;

      i_micro_kernel_config->vec_xr         = reserved_zmms - 1;
      i_micro_kernel_config->vec_xa         = reserved_zmms - 2;
      i_micro_kernel_config->vec_index      = reserved_zmms - 3;
      i_micro_kernel_config->vec_C0         = reserved_zmms - 4;
      i_micro_kernel_config->vec_C1         = reserved_zmms - 5;
      i_micro_kernel_config->vec_C2         = reserved_zmms - 6;
      i_micro_kernel_config->vec_thres      = reserved_zmms - 7;
      i_micro_kernel_config->vec_absmask    = reserved_zmms - 8;
      i_micro_kernel_config->vec_scale      = reserved_zmms - 9;
      i_micro_kernel_config->vec_shifter    = reserved_zmms - 10;
      i_micro_kernel_config->vec_halves     = reserved_zmms - 11;
      i_micro_kernel_config->vec_c0         = reserved_zmms - 12;
      i_micro_kernel_config->vec_c1         = reserved_zmms - 13;
      i_micro_kernel_config->vec_c2         = reserved_zmms - 14;

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU ) {
        libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx512( io_generated_code,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      }

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ) {
        libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx512( io_generated_code,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      }

      i_micro_kernel_config->reserved_zmms = reserved_zmms;
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
      unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
      unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
      reserved_zmms += 14;
      reserved_mask_regs += 2;

      i_micro_kernel_config->mask_hi       = reserved_mask_regs - 1;
      i_micro_kernel_config->mask_lo       = reserved_mask_regs - 2;
      i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
      i_micro_kernel_config->vec_nom       = reserved_zmms - 2;
      i_micro_kernel_config->vec_denom     = reserved_zmms - 3;
      i_micro_kernel_config->vec_c0        = reserved_zmms - 4;
      i_micro_kernel_config->vec_c1        = reserved_zmms - 5;
      i_micro_kernel_config->vec_c2        = reserved_zmms - 6;
      i_micro_kernel_config->vec_c3        = reserved_zmms - 7;
      i_micro_kernel_config->vec_c1_d      = reserved_zmms - 8;
      i_micro_kernel_config->vec_c2_d      = reserved_zmms - 9;
      i_micro_kernel_config->vec_c3_d      = reserved_zmms - 10;
      i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 11;
      i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 12;
      i_micro_kernel_config->vec_ones      = reserved_zmms - 13;
      i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 14;

      libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx512( io_generated_code,
          i_micro_kernel_config->vec_c0,
          i_micro_kernel_config->vec_c1,
          i_micro_kernel_config->vec_c2,
          i_micro_kernel_config->vec_c3,
          i_micro_kernel_config->vec_c1_d,
          i_micro_kernel_config->vec_c2_d,
          i_micro_kernel_config->vec_c3_d,
          i_micro_kernel_config->vec_hi_bound,
          i_micro_kernel_config->vec_lo_bound,
          i_micro_kernel_config->vec_ones,
          i_micro_kernel_config->vec_neg_ones );

      i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
      i_micro_kernel_config->reserved_zmms = reserved_zmms;
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
      unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
      unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
      reserved_zmms += 15;
      reserved_mask_regs += 2;

      i_micro_kernel_config->mask_hi       = reserved_mask_regs - 1;
      i_micro_kernel_config->mask_lo       = reserved_mask_regs - 2;
      i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
      i_micro_kernel_config->vec_nom       = reserved_zmms - 2;
      i_micro_kernel_config->vec_denom     = reserved_zmms - 3;
      i_micro_kernel_config->vec_c0        = reserved_zmms - 4;
      i_micro_kernel_config->vec_c1        = reserved_zmms - 5;
      i_micro_kernel_config->vec_c2        = reserved_zmms - 6;
      i_micro_kernel_config->vec_c3        = reserved_zmms - 7;
      i_micro_kernel_config->vec_c1_d      = reserved_zmms - 8;
      i_micro_kernel_config->vec_c2_d      = reserved_zmms - 9;
      i_micro_kernel_config->vec_c3_d      = reserved_zmms - 10;
      i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 11;
      i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 12;
      i_micro_kernel_config->vec_ones      = reserved_zmms - 13;
      i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 14;
      i_micro_kernel_config->vec_halves    = reserved_zmms - 15;

      libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx512( io_generated_code,
          i_micro_kernel_config->vec_c0,
          i_micro_kernel_config->vec_c1,
          i_micro_kernel_config->vec_c2,
          i_micro_kernel_config->vec_c3,
          i_micro_kernel_config->vec_c1_d,
          i_micro_kernel_config->vec_c2_d,
          i_micro_kernel_config->vec_c3_d,
          i_micro_kernel_config->vec_hi_bound,
          i_micro_kernel_config->vec_lo_bound,
          i_micro_kernel_config->vec_ones,
          i_micro_kernel_config->vec_neg_ones,
          i_micro_kernel_config->vec_halves );

      i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
      i_micro_kernel_config->reserved_zmms = reserved_zmms;
    }
  }

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    /* This is the temp register used to load the second input */
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
      i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_2d_microkernel( libxsmm_generated_code*                     io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n) {

  unsigned int use_m_input_masking, use_m_output_masking, m_trips, m_unroll_factor, m_assm_trips, n_trips, n_unroll_factor, n_assm_trips;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int out_loop_trips, inner_loop_trips, out_loop_reg, inner_loop_reg, out_loop_bound, inner_loop_bound, out_unroll_factor, inner_unroll_factor;
  unsigned int mask_reg_in, mask_reg_out;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int loop_type;

  /* Configure microkernel masks */
  libxsmm_setup_input_output_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc,
      LIBXSMM_X86_GP_REG_R11, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);

  /* Configure microkernel loops */
  libxsmm_configure_microkernel_loops( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, i_m, i_n, use_m_input_masking,
    &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips,
    &out_loop_trips, &inner_loop_trips, &out_loop_bound, &inner_loop_bound, &out_loop_reg, &inner_loop_reg, &out_unroll_factor, &inner_unroll_factor );

  /* Headers of microkernel loops */
  if (out_loop_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, out_loop_reg, 0, out_unroll_factor);
  }

  if (inner_loop_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, inner_loop_reg, 0, inner_unroll_factor);
  }

  /* Load block of registers */
  libxsmm_load_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Compute on registers */
  libxsmm_compute_unary_binary_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Store block of registers */
  libxsmm_store_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_out, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out);

  /* Footers of microkernel loops  */
  if (inner_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);

    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
        adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
      }
    }

    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_loop_bound);

    /* Reset input/output pointers  */
    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);

    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
        adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
      }
    }
  }

  if (out_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (out_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);

    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
        adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
      }
    }

    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, out_loop_reg, out_loop_bound);

    /* Reset input/output pointers  */
    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);

    adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
        adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int loop_order, m_blocking, out_blocking, out_bound, out_block = 0, n_blocking, inner_blocking, inner_block, inner_bound, n_microkernel = 0, m_microkernel = 0;
  unsigned int out_ind, inner_ind, reset_regs, loop_type;
  unsigned int i_gp_reg_tmp = LIBXSMM_X86_GP_REG_R11;

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* check datatype */
  if ( ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ||
         LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )    )
       &&
       ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
         LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )    ) ) {
    /* fine */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Configure vlens */
  libxsmm_generator_configure_avx512_vlens(i_mateltwise_desc, i_micro_kernel_config);

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R11;
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    i_gp_reg_mapping->gp_reg_in2  = LIBXSMM_X86_GP_REG_RAX;
  } else {
    i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_RAX;
  }

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      16,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        8,
        i_gp_reg_mapping->gp_reg_in2,
        0 );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          i_gp_reg_mapping->gp_reg_relumask,
          0 );
    }
  } else {
    /* This hsould not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }

  /* Configure M and N blocking factors */
  libxsmm_generator_configure_M_N_blocking(i_mateltwise_desc->m, i_mateltwise_desc->n, i_micro_kernel_config->vlen_in, &m_blocking, &n_blocking);
  libxsmm_generator_configure_loop_order(i_mateltwise_desc, &loop_order, &m_blocking, &n_blocking, &out_blocking, &inner_blocking, &out_bound, &inner_bound);
  i_micro_kernel_config->loop_order = loop_order;

  /* Based on kernel type reserve zmms and mask registers  */
  libxsmm_configure_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, i_gp_reg_tmp);

  out_ind = 0;
  while ( out_ind != out_bound ) {
    inner_ind = 0;
    reset_regs = 0;
    while( inner_ind != inner_bound ) {

      out_block = (out_ind < out_blocking) ? out_blocking : out_bound - out_ind;
      inner_block  = (inner_ind < inner_blocking ) ? inner_blocking : inner_bound - inner_ind;
      n_microkernel = (loop_order == NM_LOOP_ORDER) ? out_block : inner_block;
      m_microkernel = (loop_order == MN_LOOP_ORDER) ? out_block : inner_block;

      libxsmm_generator_unary_binary_2d_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, m_microkernel, n_microkernel);

      inner_ind += inner_block;

      if (inner_ind != inner_bound) {
        reset_regs = 1;
        /* Advance input/output pointers */
        loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

        adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

        adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
          adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
          if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
            adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
          }
        }
      }
    }

    /* If needed, readjust the registers */
    if (reset_regs == 1) {
      loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );

      adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
        adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
          adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
        }
      }
    }

    out_ind += out_block;
    if (out_ind != out_bound) {
      /* Advance input/output pointers */
      loop_type = (loop_order == MN_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

      adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
        adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) {
          adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        }
      }
    }
  }

  if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  }
}

#undef MN_LOOP_ORDER
#undef NM_LOOP_ORDER
#undef LOOP_TYPE_M
#undef LOOP_TYPE_N

