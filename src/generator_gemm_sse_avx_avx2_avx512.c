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
#include "generator_common.h"
#include "generator_common_x86.h"
#include "generator_x86_instructions.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse_avx_avx2_avx512.h"
#include "generator_gemm_sse_microkernel.h"
#include "generator_gemm_avx_microkernel.h"
#include "generator_gemm_avx2_microkernel.h"
#include "generator_gemm_avx512_microkernel.h"
#include "generator_mateltwise_transform_avx512.h"
#include "generator_mateltwise_transform_avx.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"


LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper( libxsmm_generated_code*        io_generated_code,
                                                                                   const libxsmm_gemm_descriptor* i_xgemm_desc       ) {
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif
  l_gp_reg_mapping.gp_reg_a = l_gp_reg_mapping.gp_reg_param_struct;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_R9;
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (l_is_Amxfp4_Bi8_gemm == 0) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    if (l_is_Amxfp4_Bi8_gemm > 0) {
      l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_RBX;
    }
  } else if ( ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
              (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
              (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ||
              ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
              (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
              (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RBX;
    l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  }
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
    l_gp_reg_mapping.gp_reg_bitmap_a = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_decompressed_elts = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_popcnt = LIBXSMM_X86_GP_REG_R9;
  }

  /* If we are generating the batchreduce kernel, then we rename the registers */
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (l_is_Amxfp4_Bi8_gemm == 0) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      if (l_is_Amxfp4_Bi8_gemm > 0) {
        l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_RBX;
      }
    } else if (((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ||
               ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )))) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RBX;
      l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_R9;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_RAX;
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (l_is_Amxfp4_Bi8_gemm == 0) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
    } else if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
      if (l_is_Amxfp4_Bi8_gemm > 0) {
        l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_RBX;
      }
    } else if( ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ||
               ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
               (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RBX;
      l_gp_reg_mapping.gp_reg_zpt = LIBXSMM_X86_GP_REG_RAX;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
  }
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RBX;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );

  /* call Intel SIMD kernel */
  libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, i_xgemm_desc );

  /* close asm */
  libxsmm_x86_instruction_close_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                                           libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                           const libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                           const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  libxsmm_gemm_descriptor *l_xgemm_desc = (libxsmm_gemm_descriptor*) &l_xgemm_desc_mod;
  unsigned int l_is_Ai4_Bf16_gemm = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_avnni_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0) &&
                                                   ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                   ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) &&
                                                   ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                   (l_xgemm_desc->k % 2 == 0) &&
                                                   ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                   (io_generated_code->arch >= LIBXSMM_X86_AVX));
  unsigned int l_atvnni_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) != 0) &&
                                                    ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                    ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0) &&
                                                    ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                    (l_xgemm_desc->k % 2 == 0) &&
                                                    ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                    (io_generated_code->arch >= LIBXSMM_X86_AVX));
  unsigned int l_avnni_btrans_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0) &&
                                                          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) != 0) &&
                                                          ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                          (l_xgemm_desc->k % 2 == 0) &&
                                                          ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                          (io_generated_code->arch >= LIBXSMM_X86_AVX));
  unsigned int l_atvnni_btrans_gemm_stack_alloc_tensors = (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) != 0) &&
                                                           ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)  &&
                                                           ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) != 0) &&
                                                           ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0)  &&
                                                           (l_xgemm_desc->k % 2 == 0) &&
                                                           ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) &&
                                                           (io_generated_code->arch >= LIBXSMM_X86_AVX));
  /* initialize n-blocking */
  unsigned int l_n_count = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_n[2] = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2] = {0,0};       /* size of blocks */

  unsigned int adjust_A_pf_ptrs = 0;
  unsigned int adjust_B_pf_ptrs = 0;
  unsigned int l_max_n_blocking = 0;
  unsigned int a_in_vnni = ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ? 1 : 0;
  unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) && (l_is_Amxfp4_Bbf16_gemm == 0)) &&
                                      (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ) ? 1 : 0;

  /* TODO: we need to implement a consolidate solution for callee save stuff
   * here we need to handle AMX stuff to allow AMX optimized TPPs to run lower platforms */
  if ( !( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) == 0)) ||
          (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & l_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & l_xgemm_desc->flags) != 0))    ) ) {
    return;
  }

  /* Make sure we properly adjust A,B prefetch pointers in case of batch-reduce gemm kernel */
  if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    if ( l_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         l_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      adjust_A_pf_ptrs = 1;
    }
  }

  /* In case of F16 and IMPLICIT compute set proper compute */
  if ( (((LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ))) ||
        ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )))) &&
        LIBXSMM_DATATYPE_IMPLICIT == LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc->datatype ) ) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
      LIBXSMM_GEMM_SET_DESC_DATATYPE((libxsmm_datatype)LIBXSMM_GEMM_GETENUM_A_PREC(l_xgemm_desc->datatype),
                                     (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_B_PREC(l_xgemm_desc->datatype),
                                     (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC(l_xgemm_desc->datatype),
                                     LIBXSMM_DATATYPE_F16,
                                     (unsigned char *)l_xgemm_desc->datatype);
    } else {
      LIBXSMM_GEMM_SET_DESC_DATATYPE((libxsmm_datatype)LIBXSMM_GEMM_GETENUM_A_PREC(l_xgemm_desc->datatype),
                                     (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_B_PREC(l_xgemm_desc->datatype),
                                     (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC(l_xgemm_desc->datatype),
                                     LIBXSMM_DATATYPE_F32,
                                     (unsigned char *)l_xgemm_desc->datatype);
    }
  }

  /* in case of BF8/HFS we might need set different precisions */
  if ( ( (libxsmm_cpuid_x86_bf8_gemm_via_stack() > 0) || ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (a_in_vnni == 0) ) ) ||
         ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    /* Adjust descriptor to perform GEMM with bf16 or f32 inputs */
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX)) {
      LIBXSMM_GEMM_SET_DESC_DATATYPE(LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC(l_xgemm_desc->datatype), LIBXSMM_DATATYPE_F32, (unsigned char *)l_xgemm_desc->datatype);
      l_xgemm_desc->flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
    } else {
      LIBXSMM_GEMM_SET_DESC_DATATYPE(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC(l_xgemm_desc->datatype), LIBXSMM_DATATYPE_F32, (unsigned char *)l_xgemm_desc->datatype);
    }
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
  }

  /* @TODO check if we can make this smarter and don't need two times the same if */
  if ( (l_avnni_gemm_stack_alloc_tensors != 0) || (l_atvnni_gemm_stack_alloc_tensors != 0) || (l_avnni_btrans_gemm_stack_alloc_tensors != 0) || (l_atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    l_xgemm_desc->flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }
  if ( (l_atvnni_gemm_stack_alloc_tensors != 0) || (l_atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    l_xgemm_desc->flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A));
    if ( l_xgemm_desc->lda%2 != 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
  }
  if ( (l_avnni_btrans_gemm_stack_alloc_tensors != 0) || (l_atvnni_btrans_gemm_stack_alloc_tensors != 0) ) {
    l_xgemm_desc->flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_B));
  }

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config( &l_micro_kernel_config, io_generated_code->arch, l_xgemm_desc, 0 );

  /* setup hf8 / bf8 conversion on stack before GEMM, we need to recheck as we now can update the field in ukernel config, need to use the original GEMM descriptor */
  if ( (libxsmm_cpuid_x86_bf8_gemm_via_stack() > 0) || ( (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && (a_in_vnni == 0) ) ) {
    l_micro_kernel_config.bf8_gemm_via_stack_alloc_tensors = 1;
  }

  if (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    l_micro_kernel_config.hf8_gemm_via_stack_alloc_tensors = 1;
  }

  /* in case when A needs to be transposed, we need to change temporarily the descriptor dimensions for gemm */
  if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
    if ((LIBXSMM_DATATYPE_F32 == (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_ABC_COMMON_PREC(l_xgemm_desc->datatype)) || (LIBXSMM_DATATYPE_F64 == (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_ABC_COMMON_PREC(l_xgemm_desc->datatype))) {
      l_xgemm_desc->lda = l_xgemm_desc->m;
      l_xgemm_desc->flags = (unsigned int)((unsigned int)(l_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A));
      l_micro_kernel_config.atrans_gemm_stack_alloc_tensors = 1;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if (((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) && ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0)) {
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) {
      unsigned int aux_flags = (unsigned int)((unsigned int)l_xgemm_desc->flags & (~LIBXSMM_GEMM_FLAG_TRANS_B));
      l_xgemm_desc->ldb = l_xgemm_desc->k;
      l_xgemm_desc->flags = (unsigned int)((unsigned int)aux_flags & (~LIBXSMM_GEMM_FLAG_VNNI_B));
      l_micro_kernel_config.bvnni_btrans_gemm_stack_alloc_tensors = 1;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  /* handle A VNNI on stack */
  if ( l_avnni_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_micro_kernel_config.avnni_gemm_stack_alloc_tensors = 1;
  }
  if ( l_atvnni_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_micro_kernel_config.atvnni_gemm_stack_alloc_tensors = 1;
  }
  if ( l_avnni_btrans_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    l_micro_kernel_config.avnni_btrans_gemm_stack_alloc_tensors = 1;
  }
  if ( l_atvnni_btrans_gemm_stack_alloc_tensors != 0 ) {
    l_xgemm_desc->lda = l_xgemm_desc->m;
    l_xgemm_desc->ldb = l_xgemm_desc->k;
    l_micro_kernel_config.atvnni_btrans_gemm_stack_alloc_tensors = 1;
  }

  /* block according to the number of available registers or given limits */
  l_max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch );
#if 1
  if (3 < l_max_n_blocking)
#endif
  {
    const unsigned int init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch, 0 );
    const unsigned int init_m_blocks = LIBXSMM_UPDIV(init_m_blocking, l_micro_kernel_config.vector_length);
    unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) &&
                                            (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
                                            (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ) ? 1 : 0;

    if (l_is_Ai8_Bf16_gemm > 0) {
      int l_m_scf_vregs = ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) ? ((l_is_Ai4_Bf16_gemm > 0) ? 3 + init_m_blocks : 1) : (  (l_is_Ai4_Bf16_gemm > 0) ? 2 + 2*init_m_blocks : init_m_blocks);
      if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
        l_m_scf_vregs += init_m_blocks;
      }
      /* In this case we need m vec regs for the scaling factors... */
      if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (l_is_Ai4_Bf16_gemm == 0) ) {
        while ((init_m_blocks * l_max_n_blocking + l_max_n_blocking + 1 + l_m_scf_vregs) > l_micro_kernel_config.vector_reg_count) {
          l_max_n_blocking--;
        }
      } else {
        while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1 + l_m_scf_vregs) > l_micro_kernel_config.vector_reg_count) {
          l_max_n_blocking--;
        }
      }
    } else if (l_is_Ai8_Bbf16_gemm > 0) {
      int l_m_scf_vregs = ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) ? 1 : init_m_blocks;
      /* In this case we need m vec regs for the scaling factors... */
      while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 2 + l_m_scf_vregs) > l_micro_kernel_config.vector_reg_count) {
        l_max_n_blocking--;
      }
    } else if (l_is_Ai4_Bi8_gemm > 0) {
      int l_m_zpt_vregs = 3 + init_m_blocks;
      if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
        l_m_zpt_vregs += init_m_blocks;
      }
      while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1 + l_m_zpt_vregs) > l_micro_kernel_config.vector_reg_count) {
        l_max_n_blocking--;
      }
    } else {
      if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2_SRF) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
        while ((init_m_blocks * l_max_n_blocking + l_max_n_blocking + 1) > l_micro_kernel_config.vector_reg_count) {
          l_max_n_blocking--;
        }
      } else {
        while ((init_m_blocks * l_max_n_blocking + init_m_blocks + 1) > l_micro_kernel_config.vector_reg_count) {
          l_max_n_blocking--;
        }
      }
    }
  }
  if ( l_max_n_blocking == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  libxsmm_compute_equalized_blocking( l_xgemm_desc->n, l_max_n_blocking, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );

  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
       ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_help_1 );
    /* A pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, i_gp_reg_mapping->gp_reg_a, 0 );
    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 40, i_gp_reg_mapping->gp_reg_bitmap_a, 0 );
    }
    if ( l_is_Amxfp4_Bfp32_gemm > 0  || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 48, i_gp_reg_mapping->gp_reg_scf, 0 );
    }
    /* B pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, i_gp_reg_mapping->gp_reg_b, 0 );
    if (l_is_Amxfp4_Bi8_gemm > 0) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 80, i_gp_reg_mapping->gp_reg_zpt, 0 );
    }

    /* C pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, i_gp_reg_mapping->gp_reg_c, 0 );
    if ( l_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, i_gp_reg_mapping->gp_reg_a_prefetch, 0 );
      /* B prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, i_gp_reg_mapping->gp_reg_b_prefetch, 0 );
    }
    /* batch reduce count & offset arrays*/
    if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET)) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_reduce_count, 0 );

      if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 40, i_gp_reg_mapping->gp_reg_a_offset, 0 );
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 72, i_gp_reg_mapping->gp_reg_b_offset, 0 );
      }
    }
    /* loading scaling factor for ternary C */
    if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) && (l_is_Amxfp4_Bi8_gemm == 0) ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 112, i_gp_reg_mapping->gp_reg_scf, 0 );
    }

    /* Load scaling factor for A  */
    if (((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype )) &&
        (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
        (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))) ||
        ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) && (l_is_Amxfp4_Bbf16_gemm == 0) ) &&
        (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype )) &&
        (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )))) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 48, i_gp_reg_mapping->gp_reg_scf, 0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, i_gp_reg_mapping->gp_reg_zpt, 0 );
    }

    if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, i_gp_reg_mapping->gp_reg_zpt, 0 );
    }
  }

  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ||
       (l_micro_kernel_config.vnni_format_C > 0) ) {
    /* Illegal ext_abi when precision is not fp32 or bf16 */
    if (!(LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ||
          LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
      return;
    }
  }

  /* Setting up the stack frame */
  libxsmm_generator_gemm_setup_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config);

  /* In this case we store C to scratch */
  if (l_micro_kernel_config.vnni_format_C > 0) {
    libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C, i_gp_reg_mapping->gp_reg_c );
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_c );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_1, 32LL * 64LL );
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_c);
    l_xgemm_desc->ldc = l_xgemm_desc->m;
  }

  /* Apply potential opA / opB */
  libxsmm_generator_gemm_apply_opA_opB( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, i_xgemm_desc);

  libxsmm_reset_loop_label_tracker( io_loop_label_tracker );

  /* generate hoisted BF16 emulation mask for AVX512 */
  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) &&
         ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) &&
         (io_generated_code->arch != LIBXSMM_X86_AVX512_CPX) &&
         (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
         (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) &&
         (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0xaaaaaaaa );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }

  /* generate hoisted UU SS i8 emulation mask for AVX512 */
  if ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) &&
         ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) &&
         ( ( ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) ||
           ( ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) >  0) && ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) >  0) ) ) &&
         (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
         (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) {
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x55555555 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }

  if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 ) {
    /* Set to 0 lo mask and to 1 hi mask */
    float lut_mant[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    unsigned int mask_sign[8] = { 8, 8, 8, 8, 8, 8, 8, 8 };
    l_micro_kernel_config.io_loop_label_tracker = io_loop_label_tracker;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) lut_mant ,
                                                         "vperm_mant",
                                                         'y',
                                                         0 );
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) mask_sign ,
                                                         "vperm_sign",
                                                         'y',
                                                         1 );
  }

  if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
    /* Set to 0 lo mask and to 1 hi mask */
    char lut_mxfp4[32] = { 0, 11, 21, 32, 42, 64, 85, 127, 0, (char)-11, (char)-21, (char)-32, (char)-42, (char)-64, (char)-85, (char)-127,
                           0, 11, 21, 32, 42, 64, 85, 127, 0, (char)-11, (char)-21, (char)-32, (char)-42, (char)-64, (char)-85, (char)-127 };
    unsigned int mask_idx[8] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f };
    l_micro_kernel_config.io_loop_label_tracker = io_loop_label_tracker;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) lut_mxfp4 ,
                                                         "vperm_lut",
                                                         'y',
                                                         0 );
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) mask_idx ,
                                                         "vmask_idx",
                                                         'y',
                                                         1 );
  }

  if (l_is_Ai4_Bf16_gemm > 0) {
    unsigned int l_use_perm_based_cvt = (io_generated_code->arch > LIBXSMM_X86_AVX2 && io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1;
    if (l_use_perm_based_cvt > 0) {
      unsigned int lut_f32[16] = {0x00000000, 0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000, 0x40e00000, 0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000};
      unsigned int signed_lut_f32[16] = {0x00000000, 0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000, 0x40e00000, 0xc1000000, 0xc0e00000, 0xc0c00000, 0xc0a00000, 0xc0800000, 0xc0400000, 0xc0000000, 0xbf800000};
      unsigned short lut[32] = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80,
                                0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80};
      unsigned short signed_lut[32] = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0xc800, 0xc700, 0xc600, 0xc500, 0xc400, 0xc200, 0xc000, 0xbc00,
                                       0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0xc800, 0xc700, 0xc600, 0xc500, 0xc400, 0xc200, 0xc000, 0xbc00};
      if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype) || io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) {
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? lut_f32 : signed_lut_f32), "my_lut", 'z', 0 );
      } else {
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) ? lut : signed_lut), "my_lut", 'z', 0 );
      }
    } else {
      /* Set to 0 lo mask and to 1 hi mask */
      unsigned int mask_lo_i4[8] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f};
      unsigned int mask_hi_i4[8] = { 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0};
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                           (const unsigned char *) mask_lo_i4 ,
                                                           "my_i4_lo",
                                                           'y',
                                                           0 );
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                           (const unsigned char *) mask_hi_i4 ,
                                                           "my_i4_hi",
                                                           'y',
                                                           1 );
    }
  }

  if (l_is_Ai4_Bi8_gemm > 0) {
    /* Set 2 to vperm reg */
    unsigned char perm_rpt_zpt[64] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                                      8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15};
    /* Set to 0 lo mask and to 1 hi mask */
    unsigned int mask_lo_i4[16] = { 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f,
                                    0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f};
    unsigned int mask_hi_i4[16] = { 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0,
                                    0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0};
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) mask_lo_i4 ,
                                                         "my_i4_lo",
                                                         'z',
                                                         0 );
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) mask_hi_i4 ,
                                                         "my_i4_hi",
                                                         'z',
                                                         1 );
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,  (const unsigned char *) perm_rpt_zpt, "my_vperm_i4", 'z', 2 );
    }
  }

  if (l_is_Ai8_Bbf16_gemm > 0) {
    unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = (l_micro_kernel_config.vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS) ? 1 : 0;
    if (l_is_Ai8_Bbf16_gemm_bf16fma > 0) {
      unsigned short l_bf16_zip_512[32] = { 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47 };
      unsigned short l_bf16_zip_256[16] = { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 };
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                           (const unsigned char *) ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? l_bf16_zip_512 : l_bf16_zip_256 ),
                                                           "my_bf16_zip",
                                                           l_micro_kernel_config.vector_name,
                                                           0 );
    }
  }

  /* generated hoisted helpers for BF8 emulation */
  if ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) {
    unsigned short bf8_perm_512[32] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };
    unsigned short bf8_perm_256[16] = { 0, 2, 4, 6, 8, 10, 12, 14,                                 1, 3, 5, 7, 9, 11, 13, 15 };

    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *) ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? bf8_perm_512 : bf8_perm_256 ),
                                                         "my_bf8_perm",
                                                         l_micro_kernel_config.vector_name,
                                                         1 );
    libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x2222222222222222 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 3 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x4444444444444444 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 4 );
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,  LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_mapping->gp_reg_help_2, 0x8888888888888888 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD, i_gp_reg_mapping->gp_reg_help_2, 5 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }

  if ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && (( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) || (LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) )) ) {
    /* In this case we have one scaling factor per full tensor, load it here */
    if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
      libxsmm_x86_instruction_vec_move( io_generated_code,
          l_micro_kernel_config.instruction_set,
          LIBXSMM_X86_INSTR_VPBROADCASTW,
          i_gp_reg_mapping->gp_reg_scf,
          LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
          l_micro_kernel_config.vector_name,
          (l_is_Ai4_Bf16_gemm > 0) ? 2 : 0, 0, 1, 0 );
      if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc->datatype) || io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) {
        char vname_cvt = (l_micro_kernel_config.vector_name == 'y') ? 'z' : ((l_micro_kernel_config.vector_name == 'x') ? 'y' : 'z');
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, (l_is_Ai4_Bf16_gemm > 0) ? 2 : 0, (l_is_Ai4_Bf16_gemm > 0) ? 2 : 0 );
      }
    }
  }

  if ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) && (l_is_Amxfp4_Bbf16_gemm == 0)) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && (( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) || (LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) )) ) {
    /* In this case we have one scaling factor per full tensor, load it here */
    if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) {
      libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_INT8_SCF, i_gp_reg_mapping->gp_reg_scf );
      libxsmm_x86_instruction_vec_move( io_generated_code,
          l_micro_kernel_config.instruction_set,
          LIBXSMM_X86_INSTR_VPBROADCASTD,
          i_gp_reg_mapping->gp_reg_scf,
          LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
          l_micro_kernel_config.vector_name,
          1, 0, 1, 0 );
    }
  }

  /* Load the actual batch-reduce trip count */
  if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_micro_kernel_config.alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_reduce_count,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        i_gp_reg_mapping->gp_reg_reduce_count,
        0 );
  }

  /* apply n_blocking */
  while (l_n_done != (unsigned int)l_xgemm_desc->n) {
    unsigned int l_n_blocking = l_n_n[l_n_count];
    unsigned int l_m_done = 0;
    unsigned int l_m_done_old = 0;
    unsigned int l_m_blocking = 0;

    /* open N loop */
    libxsmm_generator_gemm_header_nloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_n_done, l_n_blocking );
    if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_decompressed_elts, 0 );
    }

    /* advance N */
    l_n_done += l_n_N[l_n_count];
    l_n_count++;

    /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
    l_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch, 0 );
    l_micro_kernel_config.m_bitmask_advance = 0; /* @TODO: FOR SSE ONLY and relumask */

    /* apply m_blocking */
    while (l_m_done != (unsigned int)l_xgemm_desc->m) {
      if ( l_m_blocking == 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }

      l_m_done_old = l_m_done;
      l_micro_kernel_config.current_m = l_m_done_old;
      LIBXSMM_ASSERT(0 != l_m_blocking);
      /* coverity[divide_by_zero] */
      l_m_done = l_m_done + (((l_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
      l_micro_kernel_config.m_bitmask_advance += ((l_m_done-l_m_done_old+3)/8); /* @TODO: FOR SSE ONLY and relumask */
      /*printf(" advance: %i %i %i\n", l_m_done, l_m_blocking, l_micro_kernel_config.m_bitmask_advance);*/

      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        /* when on AVX512, load mask, if needed */
        if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
          /* compute the mask count, depends on vlen as block in M */
          unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
          unsigned int l_mask_count = l_corrected_vlen - ( l_m_blocking % l_corrected_vlen );

          if ( ( ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ||
               ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) ) ) {
            unsigned int l_is_Ai8_Bf16_gemm = ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) )  ? 1 : 0;
            unsigned int l_is_Abf8_Bf16_gemm = ( ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) ) ) ? 1 : 0;
            unsigned int l_is_Af16_Bf16_gemm = ( ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc->datatype ) ) && ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc->datatype ) )) ? 1 : 0;
            unsigned int l_is_compute_f16_gemm = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc->datatype) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) )) ? 1 : 0;
            libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, l_mask_count, (libxsmm_datatype) ((l_is_compute_f16_gemm > 0) ? LIBXSMM_DATATYPE_F16 : LIBXSMM_DATATYPE_I32) );
            if (l_is_compute_f16_gemm > 0 && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) {
              /* Adjust mask for C handling */
              unsigned int l_c_vlen_adjusted = (l_is_compute_f16_gemm > 0) ? l_corrected_vlen/2 : l_corrected_vlen;
              libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 2, (l_m_blocking % l_corrected_vlen >= l_c_vlen_adjusted) ? 0 : l_c_vlen_adjusted - ( l_m_blocking % l_c_vlen_adjusted ), (libxsmm_datatype)LIBXSMM_DATATYPE_I32 );
              libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 3, (l_m_blocking % l_corrected_vlen >= l_c_vlen_adjusted) ? l_c_vlen_adjusted - ( l_m_blocking % l_c_vlen_adjusted ) : l_c_vlen_adjusted, (libxsmm_datatype)LIBXSMM_DATATYPE_I32 );
            } else {
              /* we have to adjust mask count as for now we are using ymm for 16bit and xmm for 8bit */
              if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) ) {
                l_mask_count = ( (((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )  ) && (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))) || ( l_is_Ai8_Bbf16_gemm > 0)) ) ? l_mask_count + 8 : l_mask_count + 24;
              } else {
                l_mask_count = ( (((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_HF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )  ) && (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))) || ( l_is_Ai8_Bbf16_gemm > 0)) ) ? l_mask_count + 16 : l_mask_count + 48;
              }
              libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, 2, l_mask_count, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype) );
            }
          } else {
            libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, l_mask_count, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype) );
          }
        } else if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX )  ) {
          unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
          unsigned int l_mask_count = l_m_blocking % l_corrected_vlen;

          if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ||
               (LIBXSMM_DATATYPE_I32 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype )) ||
               (LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ))    ) {
            libxsmm_generator_initialize_avx_mask( io_generated_code, (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 0, l_mask_count, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) );
          } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc->datatype ) ) {
            libxsmm_generator_initialize_avx_mask( io_generated_code, (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 0, l_mask_count, LIBXSMM_DATATYPE_I32 );
          } else {
            /* should not happen */
          }
          /* store mask into stack frame */
          libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_1 );
          libxsmm_x86_instruction_vec_move( io_generated_code, l_micro_kernel_config.instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                            i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) ? 2 : 0, 0, 0, 1 );
        }

        libxsmm_generator_gemm_header_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_m_done_old, l_m_blocking );
        libxsmm_generator_gemm_load_C( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_m_blocking, l_n_blocking );

        if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
          if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
          } else if ( l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 32 );
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 24, i_gp_reg_mapping->gp_reg_a, 1);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_b, 1);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 1);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 1);
          } else {
            /* nothing to do */
          }
          /* This is the reduce loop */
          libxsmm_generator_gemm_header_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config );
          if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);

            if (adjust_A_pf_ptrs) {
              /* coverity[dead_error_line] */
              libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a_prefetch );
            }
            if (adjust_B_pf_ptrs) {
              libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
            }
            /* load to reg_a the proper array based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_a,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_a,
                0 );
            if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_help_1,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_scf,
                  0 );
              if (l_is_Amxfp4_Bi8_gemm > 0) {
                libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
                libxsmm_x86_instruction_alu_mem( io_generated_code,
                    l_micro_kernel_config.alu_mov_instruction,
                    i_gp_reg_mapping->gp_reg_help_1,
                    i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                    0,
                    i_gp_reg_mapping->gp_reg_help_1,
                    0 );
                libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              }
            }
            if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_help_1,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_help_1,
                  0 );
              libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_1 );
            }
            /* load to reg_b the proper array based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_b,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_b,
                0 );
            if (adjust_A_pf_ptrs) {
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_prefetch,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_a_prefetch,
                  0 );
            }
            if (adjust_B_pf_ptrs) {
              /* coverity[dead_error_line] */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  0 );
            }
          } else if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_a the proper address based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_a_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);

            if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_offset,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_scf,
                  0 );
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_scf, 4);
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_scf);
              if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
                libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
                libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
                libxsmm_x86_instruction_alu_mem( io_generated_code,
                    l_micro_kernel_config.alu_mov_instruction,
                    i_gp_reg_mapping->gp_reg_b_offset,
                    i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                    0,
                    LIBXSMM_X86_GP_REG_RDX,
                    0 );
                libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, LIBXSMM_X86_GP_REG_RDX, 3);
                libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_RDX);
                libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR, LIBXSMM_X86_GP_REG_RDX );
                libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
              }
            }

            if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
              libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
              libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
              libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, LIBXSMM_X86_GP_REG_RDX, 0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_a_offset,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  LIBXSMM_X86_GP_REG_RAX,
                  0 );
              libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_1, i_xgemm_desc->k);
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SARQ, i_gp_reg_mapping->gp_reg_help_1, 1);
              libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_IDIVQ, LIBXSMM_X86_GP_REG_UNDEF, i_gp_reg_mapping->gp_reg_help_1);
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, LIBXSMM_X86_GP_REG_RAX, i_gp_reg_mapping->gp_reg_help_1);
              libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
              libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
            }


            /* Calculate to reg_b the proper address based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_b_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
          } else if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
            /* reloading A and B from stack */
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 24, i_gp_reg_mapping->gp_reg_a, 0);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, i_gp_reg_mapping->gp_reg_b, 0);
            /* Calculate to reg_a the proper address based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 0);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper address based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 0);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            if (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_MXSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, i_gp_reg_mapping->gp_reg_scf);
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_scf, i_xgemm_desc->c1/16);
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_scf);
              if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
                libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
                libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_PTR, i_gp_reg_mapping->gp_reg_help_1 );
                libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, LIBXSMM_X86_GP_REG_RAX);
                libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RAX, i_xgemm_desc->c2/8);
                libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_RAX);
                libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_BSCALE_BRGEMM_PTR, LIBXSMM_X86_GP_REG_RAX );
                libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
              }
            }

            if (l_is_Ai4_Bi8_gemm > 0 && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 ) {
              libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
              libxsmm_generator_gemm_getval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_PTR, i_gp_reg_mapping->gp_reg_help_1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_loop, LIBXSMM_X86_GP_REG_RAX);
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RAX, i_xgemm_desc->lda);
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_RAX);
              libxsmm_generator_gemm_setval_stack_var( io_generated_code, &l_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, LIBXSMM_X86_GP_REG_RAX );
              libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
            }
          }
        }

        libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config,
                                                           l_xgemm_desc, l_m_blocking, l_n_blocking );

        if ((l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
          if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
            if (adjust_B_pf_ptrs) {
              /* coverity[dead_error_begin] */
              libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_help_0,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_b_prefetch,
                  1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b_prefetch);
            }
            if (adjust_A_pf_ptrs) {
              libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  i_gp_reg_mapping->gp_reg_help_0,
                  i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                  0,
                  i_gp_reg_mapping->gp_reg_a_prefetch,
                  1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a_prefetch);
            }
            /* Pop address of B_array to help_0 and store proper address of B */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_help_0,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_b,
                1 );
            /* Move to reg_b the address of B_array */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            /* Pop address of A_array to help_0 and store proper address of A */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_help_0,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                0,
                i_gp_reg_mapping->gp_reg_a,
                1 );
            /* Move to reg_a the address of A_array */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
          } else if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
           /* Calculate to reg_a the proper address based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_help_0, 1);
            /* Calculate to reg_b the proper address based on the reduce loop index */
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_help_0, 1);
          }
          libxsmm_generator_gemm_footer_reduceloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc);
          if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
            /* Calculate to reg_a the proper A advance from the microkernel */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_a_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                -8,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper B advance from the microkernel */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                i_gp_reg_mapping->gp_reg_b_offset,
                i_gp_reg_mapping->gp_reg_reduce_loop, 8,
                -8,
                i_gp_reg_mapping->gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            /* Consume the last two pushes from the stack */
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_help_0);
          }
          if (l_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
            /* Calculate to reg_a the proper A advance from the microkernel */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a);
            /* Calculate to reg_b the proper B advance from the microkernel */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, i_gp_reg_mapping->gp_reg_reduce_count, i_gp_reg_mapping->gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, l_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_b);
            /* reset stack */
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, LIBXSMM_X86_GP_REG_RSP, 32 );
          }
        }

        libxsmm_generator_gemm_store_C( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_m_blocking, l_n_blocking );
        libxsmm_generator_gemm_footer_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_m_blocking, l_m_done );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( &l_micro_kernel_config, l_xgemm_desc, io_generated_code->arch, l_m_blocking );
    }
    libxsmm_generator_gemm_footer_nloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc, l_n_blocking, l_n_done );
  } /* while l_n_done */

  /* In this case we vnni-format C from scratch */
  if (l_micro_kernel_config.vnni_format_C > 0) {
    l_xgemm_desc->ldc = i_xgemm_desc->ldc;
    libxsmm_generator_gemm_vnni_store_C_from_scratch( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc);
  }

  /* destroy stack frame */
  libxsmm_generator_gemm_destroy_stack_frame( io_generated_code, l_xgemm_desc, i_gp_reg_mapping, &l_micro_kernel_config );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse_avx_avx2_avx512_kloop( libxsmm_generated_code*            io_generated_code,
                                                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const unsigned int                 i_m_blocking,
                                                                           const unsigned int                 i_n_blocking ) {
  void (*l_generator_kloop_kernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                   const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int, const unsigned int);
  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 0;
  unsigned int l_k_threshold = 0;
  unsigned int l_k_pack_factor = 1;
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) && (l_is_Amxfp4_Bbf16_gemm == 0)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
                                     ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
                                      ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
  unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = (i_micro_kernel_config->vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS) ? 1 : 0;
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_i8_uu_ss_gemm = (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                                    ( ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) ||
                                      ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) >  0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) >  0) ) );

  /* a very simple k unrolling model */
  l_k_blocking = 4;
  l_k_threshold = 23;

  /* VNNI kernel should maintain the same amount of unrolled instructions */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );;
    l_k_blocking = l_k_blocking*l_k_pack_factor;
    l_k_threshold = ((l_k_threshold+1)*l_k_pack_factor)-1;
  }
  /* for BF8 we need to limit the unrolling, software emulation code is very large */
  if ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) || LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) ) {
    l_k_blocking = 8;
    l_k_threshold = 23;
  }

  /* for BF8 we need to limit the unrolling, software emulation code is very large */
  if ( ( l_is_i8_uu_ss_gemm != 0) && ((io_generated_code->arch == LIBXSMM_X86_AVX512_SKX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_SKX)) ) {
    l_k_blocking = 8;
    l_k_threshold = 23;
  }

  if (l_is_Ai8_Bbf16_gemm > 0) {
    if (l_is_Ai8_Bbf16_gemm_bf16fma > 0) {
      l_k_blocking = 4;
      l_k_threshold = 13;
    } else {
      l_k_blocking = 2;
      l_k_threshold = 5;
    }
  }

  if ( l_k_threshold < l_k_blocking ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_K_BLOCK );
    return;
  }

  if (l_is_Ai4_Bf16_gemm > 0) {
    l_k_blocking = 4;
    l_k_threshold = 8;
  }

  if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0 ) {
    l_k_blocking = 32;
    l_k_threshold = i_xgemm_desc->k;
  }

  /* set up architecture dependent compute micro kernel generator */
  if ( io_generated_code->arch < LIBXSMM_TARGET_ARCH_GENERIC ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  } else if ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) {
    l_generator_kloop_kernel = libxsmm_generator_gemm_avx2_kloop_kernel;
  } else if ( io_generated_code->arch <= LIBXSMM_X86_SSE42 ) {
    l_generator_kloop_kernel = libxsmm_generator_gemm_sse_kloop_kernel;
  } else if ( io_generated_code->arch == LIBXSMM_X86_AVX ) {
    l_generator_kloop_kernel = libxsmm_generator_gemm_avx_kloop_kernel;
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) ) {
    l_generator_kloop_kernel = libxsmm_generator_gemm_avx2_kloop_kernel;
  } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    l_generator_kloop_kernel = libxsmm_generator_gemm_avx512_kloop_kernel;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  if (l_is_Ai4_Bi8_gemm > 0 &&  ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0 )) {
    unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
    unsigned int l_m;
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_ZPT_BRGEMM_PTR, i_gp_reg_mapping->gp_reg_help_2 );
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      char vname_ld = 'x';
      unsigned int l_vreg = 3 + l_m;
      unsigned int l_vperm_reg = 2;
      libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU8,
          i_gp_reg_mapping->gp_reg_help_2, LIBXSMM_X86_GP_REG_UNDEF, 0,
          (l_m * i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size_in2),
          vname_ld,
          l_vreg, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
        /* Permute with vperm register 2 to get partially brodcasted vector: [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 ...] */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPERMB, i_micro_kernel_config->vector_name, l_vreg, l_vperm_reg, l_vreg);
      } else {
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXBD, i_micro_kernel_config->vector_name, l_vreg, l_vreg);
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg, l_vperm_reg, 8 );
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_micro_kernel_config->vector_name, l_vperm_reg, l_vreg, l_vreg );
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg, l_vperm_reg, 16 );
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_micro_kernel_config->vector_name, l_vperm_reg, l_vreg, l_vreg );
      }
    }
  }

  if (l_is_Ai8_Bbf16_gemm > 0 && l_is_Ai8_Bbf16_gemm_bf16fma == 0) {
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }

  /* apply multiple k_blocking strategies */
  /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
  if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
    libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_m_blocking, l_k_blocking);

    l_generator_kloop_kernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
      i_xgemm_desc, i_m_blocking, i_n_blocking, l_k_blocking);

    libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
      i_xgemm_desc, i_m_blocking, i_xgemm_desc->k, 1 );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      l_generator_kloop_kernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
        i_xgemm_desc, i_m_blocking, i_n_blocking, (unsigned int)i_xgemm_desc->k);
    /* 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
    } else {
      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
      int l_b_offset = 0;

      /* we can block as k is large enough */
      if ( l_max_blocked_k > 0 ) {
        libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_m_blocking, l_k_blocking);

        l_generator_kloop_kernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_m_blocking, i_n_blocking, l_k_blocking);

        libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_m_blocking, l_max_blocked_k, 0 );
      }

      /* now we handle the remainder handling */
      l_generator_kloop_kernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
        i_xgemm_desc, i_m_blocking, i_n_blocking, ((unsigned int)i_xgemm_desc->k) - l_max_blocked_k );

      /* reset B pointer */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in2;
      } else {
        l_b_offset = i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in2;
      }

      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
    }
  }

  if (l_is_Ai8_Bbf16_gemm > 0 && l_is_Ai8_Bbf16_gemm_bf16fma == 0) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_mapping->gp_reg_help_2 );
  }
}

LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                                           const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                                           const unsigned int             i_arch,
                                                                                           const unsigned int             i_current_m_blocking ) {
  unsigned int l_use_masking_a_c = 0;
  unsigned int l_m_blocking = i_current_m_blocking;
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);

  if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    if ( io_micro_kernel_config->fused_relu == 1 ) {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
    } else {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 12, 4, &l_m_blocking, &l_use_masking_a_c );
    }
  } else if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    if ( io_micro_kernel_config->fused_relu == 1 ) {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 4, 2, &l_m_blocking, &l_use_masking_a_c );
    } else {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 6, 2, &l_m_blocking, &l_use_masking_a_c );
    }
  } else if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch <= LIBXSMM_X86_SSE42 ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX )      && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 24, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch == LIBXSMM_X86_AVX )      && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 12, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch >= LIBXSMM_X86_AVX2 ) && ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0  ) ) {
    if (i_xgemm_desc->n == 1) {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
    } else {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
    }
  } else if ((i_arch == LIBXSMM_X86_AVX2_SRF) && ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ||
                                                                                    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ||
                                                                                    ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )    ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 3) ? 32 : ( (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 5) ? 16 : 8), 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX2) && (i_arch < LIBXSMM_X86_AVX512_VL128_SKX) ) && ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ||
                                                                                    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ||
                                                                                    ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )    ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ((i_arch == LIBXSMM_X86_AVX2_SRF) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 3) ? 16 : ( (libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking() <= 5) ? 8 : 4), 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX2) && (i_arch < LIBXSMM_X86_AVX512_VL128_SKX) ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ((i_arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (i_arch < LIBXSMM_X86_AVX512_SKX)) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )  ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ((i_arch >= LIBXSMM_X86_AVX512_SKX) && (i_arch <= LIBXSMM_X86_ALLFEAT)) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) == 0 )) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ( (i_arch == LIBXSMM_X86_AVX512_VL256_SKX) && ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )  ||
                                                        ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )      ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 8, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( i_arch >= LIBXSMM_X86_AVX512_SKX) && ( i_arch <= LIBXSMM_X86_AVX512_SKX ) &&
                             ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )  ||
                               ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) )      ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 16, 16, &l_m_blocking, &l_use_masking_a_c );
  } else if ((( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) &&
              ( LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ))  ||
             ((LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) &&
              (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ))) ) {
    if (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) {
      if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SPR ) ) {
        libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 128, 32, &l_m_blocking, &l_use_masking_a_c );
      } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SKX ) ) {
        libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
      } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) ) {
        libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
      } else {
        /* Do nothing  */
      }
    } else {
      if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SPR ) ) {
        libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
      } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SKX ) ) {
        libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
      } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) ) {
        libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
      } else {
        /* Do nothing  */
      }
    }
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && (i_arch < LIBXSMM_X86_AVX512_SKX) ) &&
              ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 8, &l_m_blocking, &l_use_masking_a_c );
  } else if ( ( (i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && (i_arch < LIBXSMM_X86_AVX512_SKX ) )
              && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 4, &l_m_blocking, &l_use_masking_a_c );
  } else if ( (LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) &&
              (LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) ) {
    if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_SKX ) ) {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
    } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX ) ) {
      libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
    } else {
      /* Do nothing  */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_I32  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 )||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) ||
                ( LIBXSMM_DATATYPE_HF8  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) && (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 64, 16, &l_m_blocking, &l_use_masking_a_c );
  }  else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
    libxsmm_generator_gemm_get_blocking_and_mask( i_xgemm_desc->m, 32, 8, &l_m_blocking, &l_use_masking_a_c );
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  io_micro_kernel_config->use_masking_a_c = l_use_masking_a_c;
  if ( l_use_masking_a_c != 0 ) {
    io_micro_kernel_config->c_vmove_nts_instruction = ( io_micro_kernel_config->c_vmove_nts_instruction == LIBXSMM_X86_INSTR_VMOVNTPS ) ? LIBXSMM_X86_INSTR_VMOVAPS : io_micro_kernel_config->c_vmove_nts_instruction;
    io_micro_kernel_config->c_vmove_nts_instruction = ( io_micro_kernel_config->c_vmove_nts_instruction == LIBXSMM_X86_INSTR_VMOVNTPD ) ? LIBXSMM_X86_INSTR_VMOVAPD : io_micro_kernel_config->c_vmove_nts_instruction;
  } else {
    io_micro_kernel_config->c_vmove_nts_instruction = ( io_micro_kernel_config->c_vmove_nts_instruction == LIBXSMM_X86_INSTR_VMOVAPS ) ? LIBXSMM_X86_INSTR_VMOVNTPS : io_micro_kernel_config->c_vmove_nts_instruction;
    io_micro_kernel_config->c_vmove_nts_instruction = ( io_micro_kernel_config->c_vmove_nts_instruction == LIBXSMM_X86_INSTR_VMOVAPD ) ? LIBXSMM_X86_INSTR_VMOVNTPD : io_micro_kernel_config->c_vmove_nts_instruction;
  }

  return l_m_blocking;
}


LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                                               const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                                               const unsigned int                  i_arch ) {
  unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
  unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
  if ( i_arch >= LIBXSMM_X86_GENERIC && (l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0)  ) {
    return 4;
  } else if ( i_arch >= LIBXSMM_X86_GENERIC && i_arch < LIBXSMM_X86_AVX512_VL256_SKX ) {
    LIBXSMM_UNUSED(i_micro_kernel_config);
    if (i_arch == LIBXSMM_X86_AVX2_SRF) {
      return libxsmm_cpuid_x86_srf_gemm_set_n_max_blocking();
    } else {
      return 3;
    }
  } else if ( i_arch >= LIBXSMM_X86_AVX512_VL256_SKX && i_arch < LIBXSMM_X86_AVX512_SKX ) {
    if ( ( i_arch == LIBXSMM_X86_AVX512_VL256_CPX ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    if ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ||
         ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    if ( ( (i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    return 30;
  } else if ( i_arch >= LIBXSMM_X86_AVX512_SKX && i_arch <= LIBXSMM_X86_ALLFEAT) {
    /* handle int16 on SKX */
    if ( ( i_arch == LIBXSMM_X86_AVX512_SKX ) && ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle int8 on all AVX512 */
    if ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    if ( ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    if ((LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_F16  == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) || LIBXSMM_DATATYPE_BF8  == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) ) {
      return 28;
    }
    /* handle bfoat8 on all AVX512 */
    if ( ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) || ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    /* handle bfoat8 on all AVX512 */
    if ( ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) || ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
     /* handle bf16 */
    if ( ( i_arch < LIBXSMM_X86_AVX512_CPX ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) && ( i_arch != LIBXSMM_X86_AVX512_VL256_CPX ) ) ) {
      return 28;
    }
    if ( ( (i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) ) {
      return 28;
    }
    return 30;
  } else {
    /* shouldn’t happen */
  }
  return 0;
}
