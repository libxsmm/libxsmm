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
#include "generator_mateltwise_unary_avx_avx512.h"
#include "generator_mateltwise_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#define MN_LOOP_ORDER 0
#define NM_LOOP_ORDER 1

LIBXSMM_API_INTERN
void libxsmm_generator_configure_avx512_vlens(const libxsmm_meltw_descriptor* i_mateltwise_desc,
    unsigned int *vlen_in,
    unsigned int *vlen_out,
    unsigned int *vlen_comp) {
  /* First, determine the vlen compute based on the operation */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 )) {
    *vlen_comp = 64;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 )) {
    *vlen_comp = 32;
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ))   {
    *vlen_comp = 16;
  } else if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ))   {
    *vlen_comp = 8;
  }

  /* The vlen_in is aligned with the vlen compute */
  *vlen_in = *vlen_comp;

  /* The vlen_out depends on the output datatype */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    *vlen_out = 64;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {

    /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
    if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
        LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) )
    {
      *vlen_out = 16;
    } else {
      *vlen_out = 32;
    }
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   {
    *vlen_out = 16;
  } else if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   {
    *vlen_out = 8;
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
void load_2D_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
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
void store_2D_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
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
void compute_2D_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
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

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_X2){
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg, cur_vreg );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_2D_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_loop_order,
                                                 unsigned int                            i_vlen_in,
                                                 unsigned int                            i_vlen_out,
                                                 unsigned int                            i_vlen_comp,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n) {

  unsigned int use_m_input_masking, use_m_output_masking, m_trips, m_unroll_factor, m_assm_trips, n_trips, n_unroll_factor, n_assm_trips,  max_nm_unrolling = 32, mask_in_count, mask_out_count;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
  unsigned int out_loop_trips, inner_loop_trips, out_loop_reg, inner_loop_reg, out_loop_bound, inner_loop_bound, out_unroll_factor, inner_unroll_factor, i_out, i_inner;
  unsigned int mask_reg_in, mask_reg_out;
  unsigned int use_replacement_fp32bf16_downncvt = 0;

  /* Calculate M/N trips and unroll factors */
  use_m_input_masking   = (i_m % i_vlen_in == 0 ) ? 0 : 1;
  use_m_output_masking  = (i_m % i_vlen_out == 0 ) ? 0 : 1;
  m_trips               = (i_m + i_vlen_in - 1) / i_vlen_in;
  n_trips               = i_n;

  /* Setup read/write masks if need be */
  if (use_m_input_masking == 1) {
    mask_in_count = i_vlen_in - i_m % i_vlen_in;
    mask_reg_in   = reserved_mask_regs;
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_in, mask_in_count, LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype));
    reserved_mask_regs++;
  }

  if (use_m_output_masking == 1) {
    if (i_vlen_in == i_vlen_out) {
      mask_reg_out = mask_reg_in;
    } else {
      mask_out_count = i_vlen_out - i_m % i_vlen_out;
      mask_reg_out   = reserved_mask_regs;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_out, mask_out_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
      reserved_mask_regs++;
    }
  }
  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;

  /* Configure microkernel loops  */
  max_nm_unrolling  = max_nm_unrolling - reserved_zmms;
  if (use_m_input_masking == 1) {
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


  if (out_loop_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, out_loop_reg, 0);
  }

  if (inner_loop_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, inner_loop_reg, 0);
  }

  /* Load block of registers */
  load_2D_reg_block(io_generated_code,
      io_loop_label_tracker,
      i_gp_reg_mapping,
      i_micro_kernel_config,
      i_mateltwise_desc,
      i_vlen_in,
      reserved_zmms,
      m_unroll_factor,
      n_unroll_factor,
      use_m_input_masking,
      mask_reg_in);

  /* Compute on registers */
  compute_2D_reg_block(io_generated_code,
      io_loop_label_tracker,
      i_gp_reg_mapping,
      i_micro_kernel_config,
      i_mateltwise_desc,
      i_vlen_in,
      reserved_zmms,
      m_unroll_factor,
      n_unroll_factor,
      use_m_input_masking,
      mask_reg_in);

  /* Store block of registers */
  store_2D_reg_block(io_generated_code,
      io_loop_label_tracker,
      i_gp_reg_mapping,
      i_micro_kernel_config,
      i_mateltwise_desc,
      i_vlen_out,
      reserved_zmms,
      m_unroll_factor,
      n_unroll_factor,
      use_m_output_masking,
      mask_reg_out);

  if (inner_loop_trips > 1) {
    /* Advance input/output pointers */
    if (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, i_vlen_in * inner_unroll_factor * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, i_vlen_out * inner_unroll_factor * i_micro_kernel_config->datatype_size_out);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * inner_unroll_factor * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * inner_unroll_factor * i_micro_kernel_config->datatype_size_out);
    }

    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_loop_bound, inner_unroll_factor);

    /* Reset input/output pointers  */
    if (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_in, i_vlen_in * inner_unroll_factor * inner_loop_trips * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_out, i_vlen_out * inner_unroll_factor * inner_loop_trips * i_micro_kernel_config->datatype_size_out);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * inner_unroll_factor * inner_loop_trips * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * inner_unroll_factor * inner_loop_trips * i_micro_kernel_config->datatype_size_out);
    }
  }

  if (out_loop_trips > 1) {
    /* Advance input/output pointers */
    if (out_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, i_vlen_in * out_unroll_factor * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, i_vlen_out * out_unroll_factor * i_micro_kernel_config->datatype_size_out);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * out_unroll_factor * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * out_unroll_factor * i_micro_kernel_config->datatype_size_out);
    }

    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, out_loop_reg, out_loop_bound, out_unroll_factor);

    /* Reset input/output pointers  */
    if (out_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_in, i_vlen_in * out_unroll_factor * out_loop_trips * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_out, i_vlen_out * out_unroll_factor * out_loop_trips * i_micro_kernel_config->datatype_size_out);
    } else {
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * out_unroll_factor * out_loop_trips * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * out_unroll_factor * out_loop_trips * i_micro_kernel_config->datatype_size_out);
    }
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_avx512_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int loop_order, m_blocking, out_blocking, out_bound, out_block, n_blocking, inner_blocking, inner_block, inner_bound, n_microkernel, m_microkernel;
  unsigned int vlen_in, vlen_out, vlen_comp, out_ind, inner_ind, reset_regs;
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
  libxsmm_generator_configure_avx512_vlens(i_mateltwise_desc, &vlen_in, &vlen_out, &vlen_comp);

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_RBX;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_RCX;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_RSI;
  i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_RDX;  /* this we might want to rename */

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

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_relumask,
      0 );

  /* Configure M and N blocking factors */
  libxsmm_generator_configure_M_N_blocking(i_mateltwise_desc->m, i_mateltwise_desc->n, vlen_in, &m_blocking, &n_blocking);
  libxsmm_generator_configure_loop_order(i_mateltwise_desc, &loop_order, &m_blocking, &n_blocking, &out_blocking, &inner_blocking, &out_bound, &inner_bound);

  /* TODO: Based on kernel type reserve zmms and mask registers  */
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

  out_ind = 0;
  while ( out_ind != out_bound ) {
    inner_ind = 0;
    reset_regs = 0;
    while( inner_ind != inner_bound ) {

      out_block = (out_ind < out_blocking) ? out_blocking : out_bound - out_ind;
      inner_block  = (inner_ind < inner_blocking ) ? inner_blocking : inner_bound - inner_ind;
      n_microkernel = (loop_order == NM_LOOP_ORDER) ? out_block : inner_block;
      m_microkernel = (loop_order == MN_LOOP_ORDER) ? out_block : inner_block;

      libxsmm_generator_2D_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          loop_order, vlen_in, vlen_out, vlen_comp, m_microkernel, n_microkernel);

      inner_ind += inner_block;

      if (inner_ind != inner_bound) {
        reset_regs = 1;
        /* Advance input/output pointers */
        if (loop_order == NM_LOOP_ORDER) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, m_microkernel * i_micro_kernel_config->datatype_size_in);
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, m_microkernel * i_micro_kernel_config->datatype_size_out);
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in);
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * n_microkernel * i_micro_kernel_config->datatype_size_out);
        }
      }
    }
    /* If needed, readjust the registers */
    if (reset_regs == 1) {
      if (loop_order == NM_LOOP_ORDER) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_in, m_microkernel * i_micro_kernel_config->datatype_size_in);
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_out, m_microkernel * i_micro_kernel_config->datatype_size_out);
      } else {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in);
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * n_microkernel * i_micro_kernel_config->datatype_size_out);
      }
    }
    out_ind += out_block;
    if (out_ind != out_bound) {
      /* Advance input/output pointers */
      if (loop_order == MN_LOOP_ORDER) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, m_blocking * i_micro_kernel_config->datatype_size_in);
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, m_blocking * i_micro_kernel_config->datatype_size_out);
      } else {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * n_blocking * i_micro_kernel_config->datatype_size_in);
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, i_mateltwise_desc->ldo * n_blocking * i_micro_kernel_config->datatype_size_out);
      }
    }
  }

  if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  }
}

#undef MN_LOOP_ORDER
#undef NM_LOOP_ORDER

