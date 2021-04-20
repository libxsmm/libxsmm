/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_gemm_common.h"
#include "generator_common.h"
#include "generator_x86_instructions.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_get_rbp_relative_offset( libxsmm_gemm_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      10th param (if applicable)                <-- RBP+40
   *      9th param (if applicable)                 <-- RBP+32
   *      8th param (if applicable)                 <-- RBP+24
   *      7th param (if applicable)                 <-- RBP+16
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      prefetch A ptr                            <-- RBP-8
   *      prefetch B ptr                            <-- RBP-16
   *      Offset A array ptr                        <-- RBP-24
   *      Offset B array ptr                        <-- RBP-32
   *      Int8 scaling factor                       <-- RBP-40
   *      GEMM_scratch ptr in stack (to be filled)  <-- RBP-48
   *      Eltwise bias ptr                          <-- RBP-56
   *      Eltwise output_ptr                        <-- RBP-64
   *      Eltwise buf1_ptr                          <-- RBP-72
   *      Eltwise buf2_ptr                          <-- RBP-80
   *
   * */

  switch ( stack_var ) {
    case LIBXSMM_GEMM_STACK_VAR_NONE:
      return 0;
    case LIBXSMM_GEMM_STACK_VAR_PFA_PTR:
      return -8;
    case LIBXSMM_GEMM_STACK_VAR_PFB_PTR:
      return -16;
    case LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR:
      return -24;
    case LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR:
      return -32;
    case LIBXSMM_GEMM_STACK_VAR_INT8_SCF:
      return -40;
    case LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR:
      return -48;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR:
      return -56;
    case LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR:
      return -64;
    case LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BUF1:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BUF2:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR:
      return -72;
    case LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF:
      return -80;
    case LIBXSMM_GEMM_STACK_VAR_ARG_7:
      return 16;
    case LIBXSMM_GEMM_STACK_VAR_ARG_8:
      return 24;
    case LIBXSMM_GEMM_STACK_VAR_ARG_9:
      return 32;
    case LIBXSMM_GEMM_STACK_VAR_ARG_10:
      return 40;
    default:
      return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_getval_stack_var( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              libxsmm_gemm_stack_var              stack_var,
                                              unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setval_stack_var( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              libxsmm_gemm_stack_var              stack_var,
                                              unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_gemm_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_fullvector( libxsmm_micro_kernel_config*    io_micro_kernel_config,
    const unsigned int             i_arch,
    const libxsmm_gemm_descriptor* i_xgemm_desc,
    const unsigned int             i_use_masking_a_c ) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( (i_arch <= LIBXSMM_TARGET_ARCH_GENERIC) || (i_arch > LIBXSMM_X86_ALLFEAT) ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
    io_micro_kernel_config->vector_reg_count = 0;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'a';
    io_micro_kernel_config->vector_length = 0;
    io_micro_kernel_config->datatype_size_in = 0;
    io_micro_kernel_config->datatype_size_out = 0;
    io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
  } else if ( i_arch  <= LIBXSMM_X86_SSE42  ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'x';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPD;
      }
      if ( i_arch == LIBXSMM_X86_GENERIC ) {
        io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSD;
        io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPD;
      } else {
        io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVDDUP;
        io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULPD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDPD;
    } else {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_SHUFPS;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVAPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPS;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULPS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDPS;
    }
  } else if ( i_arch <= LIBXSMM_X86_AVX2 ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'y';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
      }
    } else {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      }
    }
  } else if ( i_arch <= LIBXSMM_X86_ALLFEAT ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 32;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'z';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
        } else {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
    } else if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      if ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPWSSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 1;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VPDPBUSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    } else if ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 4;
      if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VDPBF16PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else if ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
                ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
      /* C is 32bit, so we treat all 3 matrices as 32bit element arrays */
      io_micro_kernel_config->vector_length = 16;
      io_micro_kernel_config->datatype_size_in = 2;
      if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) {
        io_micro_kernel_config->datatype_size_out = 2;
      } else {
        io_micro_kernel_config->datatype_size_out = 4;
      }
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        if ( (i_use_masking_a_c == 0) ) {
          io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
        } else {
          io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        }
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    } else {
      /* shouldn't happen as we caught this case earlier */
      io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
      io_micro_kernel_config->vector_reg_count = 0;
      io_micro_kernel_config->use_masking_a_c = 0;
      io_micro_kernel_config->vector_name = 'a';
      io_micro_kernel_config->vector_length = 0;
      io_micro_kernel_config->datatype_size_in = 0;
      io_micro_kernel_config->datatype_size_out = 0;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else {
    /* that should no happen */
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_halfvector( libxsmm_micro_kernel_config*    io_micro_kernel_config,
    const unsigned int             i_arch,
    const libxsmm_gemm_descriptor* i_xgemm_desc,
    const unsigned int             i_use_masking_a_c ) {
  if ( (i_arch <= LIBXSMM_TARGET_ARCH_GENERIC) || (i_arch > LIBXSMM_X86_ALLFEAT) ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
    io_micro_kernel_config->vector_reg_count = 0;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'a';
    io_micro_kernel_config->vector_length = 0;
    io_micro_kernel_config->datatype_size_in = 0;
    io_micro_kernel_config->datatype_size_out = 0;
    io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
  } else if ( i_arch <= LIBXSMM_X86_SSE42 ) {
#if !defined(NDEBUG)
    fprintf(stderr, "LIBXSMM WARNING, libxsmm_generator_gemm_init_micro_kernel_config_halfvector, redirecting to scalar, please fix the generation code!!!\n");
#endif
    libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, i_use_masking_a_c );
  } else if ( i_arch <= LIBXSMM_X86_AVX2 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'x';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VMOVDDUP;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPD;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
      }
    } else {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( (LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
        io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULPS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
      }
    }
  } else if ( i_arch <= LIBXSMM_X86_ALLFEAT ) {
#if !defined(NDEBUG)
    fprintf(stderr, "LIBXSMM WARNING, libxsmm_generator_gemm_init_micro_kernel_config_halfvector, AVX512 redirecting to fullvector!\n");
#endif
    libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, i_use_masking_a_c );
  } else {
    /* should not happen */
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_scalar( libxsmm_micro_kernel_config*    io_micro_kernel_config,
    const unsigned int             i_arch,
    const libxsmm_gemm_descriptor* i_xgemm_desc,
    const unsigned int             i_use_masking_a_c ) {
  if ( ( i_arch <= LIBXSMM_TARGET_ARCH_GENERIC ) || ( i_arch > LIBXSMM_X86_ALLFEAT ) ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_TARGET_ARCH_GENERIC;
    io_micro_kernel_config->vector_reg_count = 0;
    io_micro_kernel_config->use_masking_a_c = 0;
    io_micro_kernel_config->vector_name = 'a';
    io_micro_kernel_config->vector_length = 0;
    io_micro_kernel_config->datatype_size_in = 0;
    io_micro_kernel_config->datatype_size_out = 0;
    io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_UNDEF;
    io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
  } else if ( i_arch <= LIBXSMM_X86_SSE42 ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'x';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 1;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVSD;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVSD;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVSD;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULSD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDSD;
    } else {
      io_micro_kernel_config->vector_length = 1;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_MOVSS;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPS;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_MULSS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_ADDSS;
    }
  } else if ( i_arch <= LIBXSMM_X86_ALLFEAT ) {
    io_micro_kernel_config->instruction_set = i_arch;
    io_micro_kernel_config->vector_reg_count = 16;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'x';
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      io_micro_kernel_config->vector_length = 1;
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULSD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDSD;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231SD;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
      }
    } else {
      io_micro_kernel_config->vector_length = 1;
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->c_vmove_nts_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      if ( i_arch == LIBXSMM_X86_AVX ) {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VMULSS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_VADDSS;
      } else {
        io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231SS;
        io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
      }
    }
  } else {
    /* should not happen */
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_add_flop_counter( libxsmm_generated_code*         io_generated_code,
    const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  if ( io_generated_code->code_type == 0 ) {
    char l_new_code[512];
    const unsigned int l_max_code_length = sizeof(l_new_code) - 1;
    int l_code_length = 0;

    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifndef NDEBUG\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef _OPENMP\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma omp atomic\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "libxsmm_num_total_flops += %u;\n", 2u * i_xgemm_desc->m * i_xgemm_desc->n * i_xgemm_desc->k);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_kloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_k_blocking ) {
  LIBXSMM_UNUSED(i_m_blocking);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_kloop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_kloop, i_k_blocking);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_kloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_max_blocked_k,
    const unsigned int                 i_kloop_complete ) {
  LIBXSMM_UNUSED(i_m_blocking);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_kloop, i_max_blocked_k );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
  if ( i_kloop_complete != 0 ) {
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_reduceloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc) {
  LIBXSMM_UNUSED(i_xgemm_desc);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, 1);
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_reduce_loop);
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_nloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_n_init,
    const unsigned int                 i_n_blocking) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_init );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_blocking );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_nloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_n_blocking,
    const unsigned int                 i_n_done ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
      (i_n_blocking*(i_xgemm_desc->ldc)*(i_micro_kernel_config->datatype_size_out)) - ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_out)) );

  /* B prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_prefetch,
          (i_n_blocking*(i_xgemm_desc->ldc)*i_micro_kernel_config->datatype_size_in) - ((i_xgemm_desc->m)*i_micro_kernel_config->datatype_size_in) );
    }
  }

#if 0
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c_prefetch,
        (i_n_blocking*(i_xgemm_desc->ldc)*(i_micro_kernel_config->datatype_size_out)) - ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_out)) );
  }
#endif

  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    }

    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_help_0, ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in)) );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        1 );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_0, l_b_offset );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_b,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        1 );
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_prefetch,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          0 );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_help_0, ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in)) );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_a_prefetch,
          i_gp_reg_mapping->gp_reg_reduce_loop, 8,
          0,
          i_gp_reg_mapping->gp_reg_help_0,
          1 );
    }
    libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  } else {
    /* handle trans B */
    int l_b_offset = 0;
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_n_blocking * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_n_blocking * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_a, ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in)) );

    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_a_prefetch, ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size_in)) );
    }
  }
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_done );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_mloop( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_m_init,
    const unsigned int                 i_m_blocking ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_init );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_blocking );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_mloop( libxsmm_generated_code*            io_generated_code,
                                          libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                          const unsigned int                 i_m_blocking,
                                          const unsigned int                 i_m_done ) {
  /* advance C pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
      i_gp_reg_mapping->gp_reg_c, i_m_blocking*(i_micro_kernel_config->datatype_size_out) );

  /* C prefetch */
#if 0
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_c_prefetch, i_m_blocking*(i_micro_kernel_config->datatype_size_out) );

  }
#endif

  /* B prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       ||
       i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    ) {
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_b_prefetch, i_m_blocking*i_micro_kernel_config->datatype_size_in );
    }
  }

  /* A prefetch */
  if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
      i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
        libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_help_0,
            0 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0,
            ((i_xgemm_desc->k) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) -
            (i_m_blocking * (i_micro_kernel_config->datatype_size_in)) );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            i_gp_reg_mapping->gp_reg_reduce_loop, 8,
            0,
            i_gp_reg_mapping->gp_reg_help_0,
            1 );
        libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
        libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
      }
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a_prefetch,
          ((i_xgemm_desc->k) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) -
          (i_m_blocking * (i_micro_kernel_config->datatype_size_in)) );
    }
  }

  /* advance A pointer */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        0 );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0,
        ((i_xgemm_desc->k) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) - (i_m_blocking * (i_micro_kernel_config->datatype_size_in)) );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_a,
        i_gp_reg_mapping->gp_reg_reduce_loop, 8,
        0,
        i_gp_reg_mapping->gp_reg_help_0,
        1 );
    libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc);
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_reduce_loop );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0 );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
        ((i_xgemm_desc->k) * (i_micro_kernel_config->datatype_size_in) * (i_xgemm_desc->lda) ) - (i_m_blocking * (i_micro_kernel_config->datatype_size_in)) );
  }

  /* loop handling */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_done );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_n_blocking ) {
  unsigned int l_m_blocking, l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  assert(0 < i_micro_kernel_config->vector_length);
  /* deriving register blocking from kernel config */
  l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);

#if !defined(NDEBUG)
  /* Do some test if it is possible to generate the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
  if (i_micro_kernel_config->instruction_set == LIBXSMM_X86_GENERIC ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_SSE3    ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_SSE42   ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX     ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX2 ) {
    if ( (i_n_blocking > 3) || (i_n_blocking < 1) || (i_m_blocking < 1) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
      return;
    }
  } else if ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_CORE ) {
    if ( (i_n_blocking > 30) || (i_n_blocking < 1) || (l_m_blocking != 1) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
      return;
    }
  } else if ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CORE ) {
    if ( (i_n_blocking > 30) || (i_n_blocking < 1) || (l_m_blocking < 1) || (l_m_blocking > 6) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
      return;
    }
  } else {}
#if 0
  if ( i_m_blocking % i_micro_kernel_config->vector_length != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
#endif
#endif /*!defined(NDEBUG)*/

  /* load C accumulator */
  if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=1 */
    /* pure BF16 kernel */
    if ( ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CORE) && (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) ) &&
                ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) ) {
      /* we add when scaling during conversion to FP32 */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* load 16 bit values into ymm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'z',
                0, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'y',
                0, 0, 1, 0 );
          }
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
              16);
        }
      }
    /* pure int8 kernel */
    } else if ( ( (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CORE) && (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) ) &&
                ( (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) ) {
      /* we need to up convert int8 to int32 */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* load 16 bit values into xmm portion of the register */
          if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU8,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'z',
                0, 2, 1, 0 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->c_vmove_instruction,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'x',
                0, 0, 1, 0 );
          }
          /* convert 8 bit values into 32 bit (integer convert) */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_C_UNSIGNED) != 0 ) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                LIBXSMM_X86_INSTR_VPMOVZXBD,
                i_micro_kernel_config->vector_name,
                0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                LIBXSMM_X86_INSTR_VPMOVSXBD,
                i_micro_kernel_config->vector_name,
                0, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          }
        }
      }
    } else {
      /* adding to C, so let's load C */
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
          /* we only mask the last m-blocked load */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->c_vmove_instruction,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              i_micro_kernel_config->vector_name,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
        }
#if 0
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
            i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C )  {
          for (l_m = 0; l_m < l_m_blocking; l_m += l_m++ ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                i_micro_kernel_config->prefetch_instruction,
                i_gp_reg_mapping->gp_reg_c_prefetch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out));
          }
        }
#endif
      }
    }
  } else {
    /* overwriting C, so let's xout accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        /* @TODO: cannot migrate to new encoder as this is also SSE */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            io_generated_code->arch,
            i_micro_kernel_config->vxor_instruction,
            i_micro_kernel_config->vector_name,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
#if 0
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_CL2 ||
          i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C )  {
        for (l_m = 0; l_m < l_m_blocking; l_m += l_m++ ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              i_micro_kernel_config->prefetch_instruction,
              i_gp_reg_mapping->gp_reg_c_prefetch,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out));
        }
      }
#endif
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_n_blocking )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* select store instruction */
  unsigned int l_vstore = (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT == (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags)) ? i_micro_kernel_config->c_vmove_nts_instruction : i_micro_kernel_config->c_vmove_instruction;

  /* @TODO fix this test */
#if !defined(NDEBUG)
  if (i_micro_kernel_config->instruction_set == LIBXSMM_X86_GENERIC ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_SSE3    ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_SSE42   ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX     ||
      i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX2 ) {
    if ( (i_n_blocking > 3) || (i_n_blocking < 1) || (i_m_blocking < 1) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
      return;
    }
  } else if ( i_micro_kernel_config->instruction_set < LIBXSMM_X86_AVX512_CORE ) {
    if ( (i_n_blocking > 30) || (i_n_blocking < 1) || (i_m_blocking != i_micro_kernel_config->vector_length) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
      return;
    }
  } else if ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CORE ) {
    if ( (i_n_blocking > 30) || (i_n_blocking < 1) || (l_m_blocking < 1) || (l_m_blocking > 6) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
      return;
    }
  } else {}
#if 0
  if ( i_m_blocking % i_micro_kernel_config->vector_length != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
#endif
#endif

  if ( ( (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE) || (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CLX) ) &&
       ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) ) {
    /* init stack with helper variables for SW-based RNE rounding */
    /* push 0x7f800000 on the stack, naninf masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_2, 0x7f800000);
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );

    /* push 0x00010000 on the stack, fixup masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_2, 0x00010000);
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );

    /* push 0x00007fff on the stack, rneadd */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_2, 0x00007fff);
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );

    /* push 0x00000001 on the stack, fixup */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_2, 0x00000001);
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );

    /* storing downconverted and rounded C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

        /* and with naninf */
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VPANDD,
                                                 1,
                                                 LIBXSMM_X86_GP_REG_RSP,
                                                 LIBXSMM_X86_GP_REG_UNDEF,
                                                 0,
                                                 24,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_X,
                                                 0 );

        /* and with fixup */
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VPANDD,
                                                 1,
                                                 LIBXSMM_X86_GP_REG_RSP,
                                                 LIBXSMM_X86_GP_REG_UNDEF,
                                                 0,
                                                 16,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_X,
                                                 1 );

        /* compute naninf mask k7 */
        libxsmm_x86_instruction_vec_compute_mem_2reg_imm8( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VPCMPD,
                                                      i_micro_kernel_config->vector_name,
                                                      LIBXSMM_X86_GP_REG_RSP,
                                                      LIBXSMM_X86_GP_REG_UNDEF,
                                                      0,
                                                      24,
                                                      1,
                                                      0,
                                                      7,
                                                      4 );

        /* compute fixup mask k6 */
        libxsmm_x86_instruction_vec_compute_mem_2reg_imm8( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VPCMPD,
                                                      i_micro_kernel_config->vector_name,
                                                      LIBXSMM_X86_GP_REG_RSP,
                                                      LIBXSMM_X86_GP_REG_UNDEF,
                                                      0,
                                                      16,
                                                      1,
                                                      1,
                                                      6,
                                                      0 );

        /* load rneadd */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          LIBXSMM_X86_INSTR_VBROADCASTSS,
                                          LIBXSMM_X86_GP_REG_RSP,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          8,
                                          i_micro_kernel_config->vector_name,
                                          0, 0, 1, 0 );

        /* load fixup */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          LIBXSMM_X86_INSTR_VBROADCASTSS,
                                          LIBXSMM_X86_GP_REG_RSP,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          0,
                                          i_micro_kernel_config->vector_name,
                                          1, 0, 1, 0 );

        /* compute fixup */
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VPADDD,
                                                      i_micro_kernel_config->vector_name,
                                                      1,
                                                      0,
                                                      0,
                                                      6,
                                                      0 );

        /* compute fixup */
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VPADDD,
                                                      i_micro_kernel_config->vector_name,
                                                      0,
                                                      reg_X,
                                                      reg_X,
                                                      7,
                                                      0 );

        /* shift FP32 by 16bit to right */
        libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code,
            LIBXSMM_X86_INSTR_VPSRAD_I,
            i_micro_kernel_config->vector_name,
            reg_X,
            reg_X,
            16);

        /* shift FP32 by 16bit to right */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
            LIBXSMM_X86_INSTR_VPMOVDW,
            i_micro_kernel_config->vector_name,
            reg_X,
            0 );

        /* store 16 bit values into ymm portion of the register */
        if ( (i_micro_kernel_config->use_masking_a_c != 0) && ( l_m == (l_m_blocking - 1) ) ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              'z',
              0, 2, 0, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_vstore,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              'y',
              0, 0, 0, 1 );
        }
      }
    }
    /* clean stack and restore help5 */
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  } else if ( ( (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) && (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CPX) ) &&
              ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) ) {
    /* storing downconverted and rounded C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      unsigned int l_m_2_blocking = (l_m_blocking/2)*2;
      l_m = 0;

      if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
        for ( l_m = 0 ; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              reg_X, 0 );

          /* store 16 bit values into ymm portion of the register */
          if ( l_m == (l_m_blocking - 1) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU16,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'z',
                0, 2, 0, 1 );
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                l_vstore,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'y',
                0, 0, 0, 1 );
          }
        }
      } else {
        for (; l_m < l_m_2_blocking; l_m+=2 ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
          unsigned int reg_X2 = l_vec_reg_acc_start + l_m+1 + (l_m_blocking * l_n);

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
              i_micro_kernel_config->vector_name,
              reg_X, reg_X2, 0 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_vstore,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              'z',
              0, 0, 0, 1 );
        }
        for (; l_m < l_m_blocking; l_m++ ) {
          unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              reg_X, 0 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              l_vstore,
              i_gp_reg_mapping->gp_reg_c,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
              'y',
              0, 0, 0, 1 );
        }
      }
    }
  } else if ( ( (i_micro_kernel_config->instruction_set <= LIBXSMM_X86_ALLFEAT) || (i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CORE) ) &&
              ( (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) ) {
    /* pick the right instrucitons */
    unsigned int inst_f32_i32 = ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_C_UNSIGNED ) != 0 ) ? LIBXSMM_X86_INSTR_VCVTPS2UDQ : LIBXSMM_X86_INSTR_VCVTPS2DQ;
    unsigned int inst_i32_i8  = ( ( i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_C_UNSIGNED ) != 0 ) ? LIBXSMM_X86_INSTR_VPMOVUSDB  : LIBXSMM_X86_INSTR_VPMOVSDB;

    /* there are case where we need to load the scaling factor's address from the stack argument list */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) != 0 ) {
      libxsmm_x86_instruction_load_arg_to_reg( io_generated_code, 0, i_gp_reg_mapping->gp_reg_scf );
    }

    /* loading scf into register 3 */
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VBROADCASTSS,
        i_gp_reg_mapping->gp_reg_scf,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
        i_micro_kernel_config->vector_name,
        3, 0, 1, 0 );

    /* Zero out register 0 to perform relu */
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
        i_micro_kernel_config->vxor_instruction,
        i_micro_kernel_config->vector_name,
        0,
        0,
        0);

    /* storing downconverted and rounded C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        unsigned int reg_X = l_vec_reg_acc_start + l_m + (l_m_blocking * l_n);
        /* Convert result to F32  */
        libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
            LIBXSMM_X86_INSTR_VCVTDQ2PS,
            i_micro_kernel_config->vector_name,
            reg_X,
            reg_X );

        /* Multiply with scaling factor */
        libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
            LIBXSMM_X86_INSTR_VMULPS,
            i_micro_kernel_config->vector_name,
            reg_X,
            3,
            reg_X );

        /* Perform RELU */
        libxsmm_x86_instruction_vec_compute_3reg(  io_generated_code,
            LIBXSMM_X86_INSTR_VMAXPS,
            i_micro_kernel_config->vector_name,
            reg_X,
            0,
            reg_X);

        /* Round result to int32  */
        libxsmm_x86_instruction_vec_compute_2reg(  io_generated_code,
            inst_f32_i32,
            i_micro_kernel_config->vector_name,
            reg_X, reg_X );

        /* down-convert to int8 */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            inst_i32_i8,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
            i_micro_kernel_config->vector_name,
            reg_X, ( ( l_m == (l_m_blocking - 1)) && ( i_micro_kernel_config->use_masking_a_c != 0 ) ) ? 2 : 0, 0, 1 );
      }
    }
  } else {
    /* storing C accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            l_vstore,
            i_gp_reg_mapping->gp_reg_c,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
            i_micro_kernel_config->vector_name,
            l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 0, 1 );
      }

      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    )  {
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
          /* determining how many prefetches we need in M direction as we just need one prefetch per cache line */
          unsigned int l_m_advance = 64 / ((i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size_out)); /* 64: hardcoded cache line length */

          for (l_m = 0; l_m < l_m_blocking; l_m += l_m_advance ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                i_micro_kernel_config->prefetch_instruction,
                i_gp_reg_mapping->gp_reg_b_prefetch,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out));
          }
        }
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                 i_gp_reg_tmp,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_mask_count ) {
  unsigned int l_mask;

  /* init full mask */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_mask = 0xff;
  } else {
    l_mask = 0xffff;
  }

  /* shift right by "inverse" remainder */
  l_mask = l_mask >> i_mask_count;

  /* move mask to GP register */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
      LIBXSMM_X86_INSTR_MOVQ,
      i_gp_reg_tmp,
      l_mask );

  if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    libxsmm_x86_instruction_mask_move( io_generated_code,
        LIBXSMM_X86_INSTR_KMOVW_GPR_LD,
        i_gp_reg_tmp,
        LIBXSMM_X86_AVX512_MASK );
    if ( ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
          i_gp_reg_tmp,
          2 );
    } else if ( ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) && ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) ) ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
          i_gp_reg_tmp,
          2 );
    } else {
      /* no addtional mask is needed */
    }
  } else {
    /* shouldn't happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

