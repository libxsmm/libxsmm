/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_rv64_instructions.h"
#include "generator_common_rv64.h"
#include "generator_gemm_common_rv64.h"
#include "generator_gemm_rv64.h"
#include "generator_mateltwise_rv64.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_common.h"

#define MAX_FP_REG (20)
#define MAX_UIMM   (0x7ff)
#define REUSE_A    (libxsmm_cpuid_rv64_gemm_prefetch_reuse_a())
#define REUSE_B    (libxsmm_cpuid_rv64_gemm_prefetch_reuse_b())
#define REUSE_C    (libxsmm_cpuid_rv64_gemm_prefetch_reuse_c())
#define PREFETCH_A (libxsmm_cpuid_rv64_gemm_prefetch_a())
#define PREFETCH_B (libxsmm_cpuid_rv64_gemm_prefetch_b())
#define REG_GP(i)  (((i)->arch == LIBXSMM_RV64_MVL128_LMUL) || (((i)->arch == LIBXSMM_RV64_MVL256_LMUL)))

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_microkernel_rvv( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                  const unsigned int                 i_m_blocking,
                                                  const unsigned int                 i_n_blocking,
                                                  const int                          i_loop_index,
                                                  const int                          i_fma_index,
                                                  const unsigned int                 i_pipelined,
                                                  const unsigned int                 i_reg_gp ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: full vector ops, 1: remainder ops */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned int l_b_stride = i_xgemm_desc->ldb;
  unsigned int l_b_next = 0;

  int u_loop_index_local = 0;
  int nld;

  unsigned long imm_offset;

  /* prep of B-ptr for next k-iteration */
  unsigned int l_b_next_k = 0;
  unsigned int l_b_next_k_inst = 0;
  unsigned int l_k_pack_factor = 1;

  /* datatype dependent instructions */
  unsigned int l_a_part_load_instr = LIBXSMM_RV64_INSTR_UNDEF;
  unsigned int l_a_4reg_load_instr = LIBXSMM_RV64_INSTR_UNDEF;
  unsigned int l_a_8reg_load_instr = LIBXSMM_RV64_INSTR_UNDEF;

  unsigned int l_b_load_instr = LIBXSMM_RV64_INSTR_RVV_VRGATHER_VV;
  unsigned int l_b_load_scalar_instr = LIBXSMM_RV64_INSTR_UNDEF;
  unsigned int l_b_load_bcast_instr = LIBXSMM_RV64_INSTR_RVV_VFMV_V_F;

  unsigned int l_compute_instr = LIBXSMM_RV64_INSTR_UNDEF;
  unsigned int l_compute_is_pred = 1;

  int max_loads = 1;

  static int fp_regid[MAX_FP_REG] = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30, 31};

  LIBXSMM_UNUSED(l_compute_is_pred);
  LIBXSMM_UNUSED(l_b_load_instr);
  LIBXSMM_UNUSED(l_b_load_scalar_instr);
  LIBXSMM_UNUSED(l_b_load_bcast_instr);
  LIBXSMM_UNUSED(l_b_next_k_inst);
  LIBXSMM_UNUSED(imm_offset);
  LIBXSMM_UNUSED(l_b_next);
  LIBXSMM_UNUSED(u_loop_index_local);
  LIBXSMM_UNUSED(l_a_part_load_instr);
  LIBXSMM_UNUSED(l_a_4reg_load_instr);
  LIBXSMM_UNUSED(l_a_8reg_load_instr);
  LIBXSMM_UNUSED(l_compute_instr);
  LIBXSMM_UNUSED(l_vec_reg_acc_start);

  l_a_part_load_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_RVV_VLE64_V : LIBXSMM_RV64_INSTR_RVV_VLE32_V;
  l_a_4reg_load_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_RVV_VL4RE64_V : LIBXSMM_RV64_INSTR_RVV_VL4RE32_V;
  l_a_8reg_load_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_RVV_VL8RE64_V : LIBXSMM_RV64_INSTR_RVV_VL8RE32_V;

  l_b_load_scalar_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_GP_FLD : LIBXSMM_RV64_INSTR_GP_FLW;

  /* Set the register index to be used */
  if (i_pipelined == 1) {
    if (i_loop_index == 0){
      max_loads = 2;
      u_loop_index_local = i_loop_index;
    } else if (i_loop_index == -1){
      max_loads = -1;
      u_loop_index_local = 0;
    }else{
      max_loads = 1;
      u_loop_index_local = i_loop_index + 1;
    }
  } else {
    max_loads = 1;
    u_loop_index_local = (i_loop_index < 0) ? 0 : i_loop_index;
  }

  assert(u_loop_index_local >= 0);

  if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) ||
       (LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))    ) {
    l_compute_instr = LIBXSMM_RV64_INSTR_RVV_VFMACC_VF;
    l_compute_is_pred = 1;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  l_m_blocks[0] = i_m_blocking / i_micro_kernel_config->vector_length;
  l_remainder_size = i_m_blocking % i_micro_kernel_config->vector_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];

  /* stride when accessing B */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    l_b_stride = 1;
  }

  l_b_stride *= i_micro_kernel_config->datatype_size_in;

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
    if ( i_n_blocking == 1 ) {
      l_b_next_k = l_k_pack_factor;
      l_b_next_k_inst = LIBXSMM_RV64_INSTR_GP_ADD;
    }
    else {
      l_b_next_k = ( (i_n_blocking - 1) * i_xgemm_desc->ldb - l_k_pack_factor);
      l_b_next_k_inst = LIBXSMM_RV64_INSTR_GP_SUB;
    }
  }
  else {
    l_b_next_k = i_xgemm_desc->ldb - (i_n_blocking - 1);
    l_b_next_k_inst = LIBXSMM_RV64_INSTR_GP_ADD;
  }

  l_b_next_k *= i_micro_kernel_config->datatype_size_in;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_total_blocks);

  /* Full vector loads on a */
  /* If sw pipeline is enabled, fetch twice for first iteration and skip the last. */
  for (nld = 0; nld < max_loads; nld++){
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m += i_reg_gp ) {
      if (PREFETCH_A){
        int m_stride = libxsmm_cpuid_rv64_gemm_m_prefetch_stride();
        int stride = i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in;

        /* If m_stride is set prefetch next m block else next k iteration of current m block */
        if (m_stride >= 1) {
          stride = i_m_blocking * m_stride * i_micro_kernel_config->datatype_size_in;
        }

        /* Calculate prefetch address for A from acutal address */
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
            i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_a_prefetch,
            stride);

        /* Prefetch A for next k loop */
        libxsmm_rv64_instruction_prefetch( io_generated_code, LIBXSMM_RV64_INSTR_GP_PREFETCH_R, i_gp_reg_mapping->gp_reg_a_prefetch, 0 );
      }

      libxsmm_rv64_instruction_rvv_move( io_generated_code,
          l_a_part_load_instr,
          i_gp_reg_mapping->gp_reg_a,
          0,
          ((u_loop_index_local + nld) % 2) * l_m_blocks[0] + l_m,
          1 );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
          LIBXSMM_RV64_INSTR_GP_ADDI,
          i_gp_reg_mapping->gp_reg_a,
          i_gp_reg_mapping->gp_reg_a,
          i_micro_kernel_config->vector_length * i_micro_kernel_config->datatype_size_in * l_k_pack_factor * i_reg_gp );
    }

    if ( l_m_blocks[1] > 0) {
         /* Set vector length */
         libxsmm_rv64_instruction_rvv_move( io_generated_code,
             l_a_part_load_instr,
             i_gp_reg_mapping->gp_reg_a,
             0,
             (u_loop_index_local + nld) % 2 + l_m,
             1);


         libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
             LIBXSMM_RV64_INSTR_GP_ADD,
             i_gp_reg_mapping->gp_reg_a,
             i_gp_reg_mapping->gp_reg_help_0,
             i_gp_reg_mapping->gp_reg_a,
             (long long)l_remainder_size * i_micro_kernel_config->datatype_size_in * l_k_pack_factor );
    }

    if ((i_xgemm_desc->lda > i_m_blocking)){
      /* move immediate to the rgister */
      libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
          ((long long)i_xgemm_desc->lda - i_m_blocking) * i_micro_kernel_config->datatype_size_in );

      libxsmm_rv64_instruction_alu_compute( io_generated_code,
          LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_mapping->gp_reg_a,
          i_gp_reg_mapping->gp_reg_help_0,
          i_gp_reg_mapping->gp_reg_a );
    }
  }

  /* Prefetch B for the next k iteration */

  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* Prefetch B for the next k iteration */
    if (PREFETCH_B){
      /* Calculate prefetch address for A from acutal address */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
          i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b_prefetch,
          (long long)i_micro_kernel_config->vector_length * i_micro_kernel_config->datatype_size_in);

      /* Prefetch A for next k loop */
      libxsmm_rv64_instruction_prefetch( io_generated_code, LIBXSMM_RV64_INSTR_GP_PREFETCH_R, i_gp_reg_mapping->gp_reg_b_prefetch, 0 );
    }

    /* Load scalar and then broadcast on b */
    libxsmm_rv64_instruction_alu_move( io_generated_code,
                                       l_b_load_scalar_instr,
                                       i_gp_reg_mapping->gp_reg_b,
                                       fp_regid[l_n % MAX_FP_REG],
                                       l_b_next  );

    if ( l_n != i_n_blocking - 1 ) {
      /* move on to next entry of B */
      l_b_next += l_b_stride;

      /* If immidiate exceeds 12 bit */
      if (l_b_next >= ((1 << 11) - 1)) {
        /* move on to next entry of B */
        if (l_b_stride > ((1 << 11) - 1)){
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_RV64_INSTR_GP_ADD,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      i_gp_reg_mapping->gp_reg_help_0,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      l_b_stride );
          l_b_next = 0;
      } else {
          libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
                                                      LIBXSMM_RV64_INSTR_GP_ADDI,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      (l_b_next - l_b_stride) );
          l_b_next = l_b_stride;
        }
      }
    }
    else {
      if ((l_b_next_k_inst == LIBXSMM_RV64_INSTR_GP_SUB) && (l_b_next > l_b_next_k)){

        imm_offset = l_b_next - l_b_next_k;

        /* move on to next entry of B */
        if (imm_offset <= MAX_UIMM) {
          libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
                                                      LIBXSMM_RV64_INSTR_GP_ADDI,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      imm_offset );
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_RV64_INSTR_GP_ADD,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      i_gp_reg_mapping->gp_reg_help_0,
                                                      i_gp_reg_mapping->gp_reg_b,
                                                      imm_offset );
        }
      }else{
        /* move on to next entry of B */
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_RV64_INSTR_GP_ADD,
                                                    i_gp_reg_mapping->gp_reg_b,
                                                    i_gp_reg_mapping->gp_reg_help_0,
                                                    i_gp_reg_mapping->gp_reg_b,
                                                    l_b_next );
        /* prepare for next call of kernel */
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                    l_b_next_k_inst,
                                                    i_gp_reg_mapping->gp_reg_b,
                                                    i_gp_reg_mapping->gp_reg_help_0,
                                                    i_gp_reg_mapping->gp_reg_b,
                                                    l_b_next_k );
      }

      l_b_next = 0;
    }
  }

  /* Load A and B matrices only if there is a valid loop index */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* issue FMAs */
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m += i_reg_gp ) {
      libxsmm_rv64_instruction_rvv_compute( io_generated_code,
                                            l_compute_instr,
                                            fp_regid[l_n % MAX_FP_REG],
                                            (i_fma_index % 2) * l_m_blocks[0] + l_m,
                                            l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                            1
                                            );
    }

    if ( l_m_blocks[1] > 0 ) {
      libxsmm_rv64_instruction_rvv_compute( io_generated_code,
                                            l_compute_instr,
                                            fp_regid[l_n % MAX_FP_REG],
                                            (i_fma_index % 2) + l_m,
                                            l_vec_reg_acc_start + (l_m_total_blocks * l_n) + l_m_blocks[0],
                                            1
                                            );
    }
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_kloop( libxsmm_generated_code*            io_generated_code,
                                        libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                        const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                        const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                        const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                        const unsigned int                 i_m_blocking,
                                        const unsigned int                 i_n_blocking,
                                        const unsigned int                 i_reg_gp ) {
  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 4;
  unsigned int l_k_threshold = 8;
  unsigned int l_k_stride = 1;
  int u_loop_index = 0;

  void (*l_generator_microkernel)( libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*, const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int , const int, const int, const unsigned int, unsigned int);

  LIBXSMM_UNUSED(u_loop_index);

  /* select micro kernel based on rv64 variant */
  l_generator_microkernel = libxsmm_generator_gemm_rv64_microkernel_rvv;

  /* apply multiple k_blocking strategies */
  /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
  if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
    unsigned int l_k;

    libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, (unsigned int)i_xgemm_desc->k );

    /* TODO (MMLA): strided k loop breaks with original idea */
    for ( l_k = 0; l_k < l_k_blocking; l_k+=l_k_stride ) {
      l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking, (l_k < (l_k_blocking - 1)) ? (int)l_k : -1, l_k, (l_k_blocking == 1) ? 0 : 1, i_reg_gp);
    }

    libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      unsigned int l_k;

      /* TODO (MMLA): strided k loop breaks with original idea */
      for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k+=l_k_stride ) {
        l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking, (l_k < (i_xgemm_desc->k - 1)) ? (int)l_k : -1, l_k, (i_xgemm_desc->k == 1) ? 0 : 1, i_reg_gp);
      }
    /* 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
    } else {

      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
      unsigned int l_k;

      /* we can block as k is large enough */
      if ( l_max_blocked_k > 0 ) {
        libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_max_blocked_k );

        /* TODO (MMLA): strided k loop breaks with original idea */
        for ( l_k = 0; l_k < l_k_blocking; l_k+=l_k_stride ) {
          l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking, (l_k < (l_k_blocking - 1)) ? (int)l_k : -1, l_k, (l_k_blocking == 1) ? 0 : 1, i_reg_gp);
        }

        libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, l_k_blocking );
      }

      /* now we handle the remainder handling */
      for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k+=l_k_stride) {
        l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_m_blocking, i_n_blocking, l_k, l_k, 0, i_reg_gp);
      }
    }
  }

  /* reset A pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_a,
                                              ((long long)i_xgemm_desc->k) * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in );

  /* reset B pointer */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                (long long)i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size_in );
  } else {
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                ((long long)i_xgemm_desc->k) * i_micro_kernel_config->datatype_size_in );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_kernel( libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  unsigned int l_n_n[2]     = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2]     = {0,0};       /* size of blocks */
  unsigned int l_n_count    = 0;           /* array counter for blocking arrays */
  unsigned int l_n_done     = 0;           /* progress tracker */
  unsigned int l_n_done_old = 0;
  unsigned int a_vnni_factor  = 1;
  unsigned int l_ldc_saved = 0;
  unsigned int l_is_i8f32_gemm  = ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
  unsigned int l_sew = (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) ? LIBXSMM_RV64_SEW_B : (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) ? LIBXSMM_RV64_SEW_W : (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype )) ? LIBXSMM_RV64_SEW_D : LIBXSMM_RV64_SEW_Q;

  unsigned int l_reg_gp   = 1; /* Register grouped together */
  unsigned int l_lmul     = LIBXSMM_RV64_LMUL_M1; /* Register grouped together */
  unsigned int l_lmul_new = LIBXSMM_RV64_LMUL_M1; /* Register grouped together */

  /* Local variables used for A transpose case */
  libxsmm_gemm_descriptor* l_xgemm_desc_opa;
  libxsmm_gemm_descriptor  l_new_xgemm_desc_opa;
  unsigned int             lda_transpose;

  LIBXSMM_UNUSED(l_ldc_saved);

  /* define gp register mapping */

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_RV64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_RV64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_RV64_GP_REG_X12;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_RV64_GP_REG_X13;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_RV64_GP_REG_X14;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_RV64_GP_REG_X15;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_RV64_GP_REG_X16;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_RV64_GP_REG_X17;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_RV64_GP_REG_X18;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_RV64_GP_REG_X19;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_RV64_GP_REG_X20;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_RV64_GP_REG_X21;
  l_gp_reg_mapping.gp_reg_scf    = LIBXSMM_RV64_GP_REG_X22;
  l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_RV64_GP_REG_X23;
  l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_RV64_GP_REG_X24;      /* Offset reg are same as used for stride */
  l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_RV64_GP_REG_X25;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_RV64_GP_REG_X26;        /* storing forward counting BRGEMM interations */
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_RV64_GP_REG_X27;        /* for a ptr updates in BRGEMM */
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_RV64_GP_REG_X28;        /* for b ptr updates in BRGEMM */
  l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_RV64_GP_REG_X29;   /* BRGEMM loop */

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* compute n blocking, based on m blocking */
  libxsmm_generator_gemm_rv64_setup_n_blocking( io_generated_code, &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch, l_n_N, l_n_n );

  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_rv64_instruction_open_stream( io_generated_code, 0x3fff );

  /* ensuring compatibility with X86 AMX */
  if ( !( (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) == 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) == 0)) ||
          (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & i_xgemm_desc->flags) != 0) && ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & i_xgemm_desc->flags) != 0))    ) ) {
    /* close asm */
    libxsmm_rv64_instruction_close_stream( io_generated_code, 0x3fff );
    return;
  }

  /* in case when A needs to be transposed, we need to change temporarily the descriptor dimensions for gemm, hence the local descriptor */
  lda_transpose = i_xgemm_desc->m;
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) {
    if ((LIBXSMM_DATATYPE_F32 == (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_ABC_COMMON_PREC(i_xgemm_desc->datatype)) || (LIBXSMM_DATATYPE_F64 == (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_ABC_COMMON_PREC(i_xgemm_desc->datatype))) {
      l_new_xgemm_desc_opa = *i_xgemm_desc;
      l_new_xgemm_desc_opa.lda = lda_transpose;
      l_new_xgemm_desc_opa.flags = (unsigned int)((unsigned int)(i_xgemm_desc->flags) & (~LIBXSMM_GEMM_FLAG_TRANS_A));
      l_xgemm_desc_opa = (libxsmm_gemm_descriptor*) &l_new_xgemm_desc_opa;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    l_new_xgemm_desc_opa = *i_xgemm_desc;
    l_xgemm_desc_opa = (libxsmm_gemm_descriptor*) &l_new_xgemm_desc_opa;
  }

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ||
       ((LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI & l_xgemm_desc_opa->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI) ) {
    int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
    int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
    int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));

    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
      l_gp_reg_mapping.gp_reg_param_struct, LIBXSMM_RV64_GP_REG_X0, l_gp_reg_mapping.gp_reg_help_1);

    /* A pointer */
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_a, l_offset_ptr_a );
    /* B pointer */
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b, l_offset_ptr_b );
    /* C pointer */
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_c, l_offset_ptr_c );

    /* Load scaling factor gpr if need be */
    if ( l_is_i8f32_gemm > 0 ) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                            l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_scf, l_offset_ptr_c + 16);

    }

    if ( l_xgemm_desc_opa->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                       l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_a_prefetch, l_offset_ptr_a + LIBXSMM_MATRIX_ARG_OFFSET_PREFETCH );
      /* B prefetch pointer */
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                       l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b_prefetch, l_offset_ptr_b + LIBXSMM_MATRIX_ARG_OFFSET_PREFETCH );
    }

    /* batch reduce count & offset arrays */
    if ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) || (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET)) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                       l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_reduce_count, 16 );

      if ( l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) {
        libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                         l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_a_offset, l_offset_ptr_a + 8 );
        libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD,
                                         l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b_offset, l_offset_ptr_b + 8 );
      }
    }
  }

  /* setting up the stack frame */
  libxsmm_generator_gemm_setup_stack_frame_rv64( io_generated_code, i_xgemm_desc, &l_gp_reg_mapping, &l_micro_kernel_config);

  /* Apply potential opA / opB */
  libxsmm_generator_gemm_apply_opA_opB_rv64( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_xgemm_desc_opa, i_xgemm_desc);

  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* calling gemm kernel with the modified pointer to the first matrix (now trans_a on the stack) should go here */

  /* apply n_blocking */
  while (l_n_done != (unsigned int)l_xgemm_desc_opa->n) {
    unsigned int l_n_blocking = l_n_n[l_n_count];
    unsigned int l_m_done = 0;
    unsigned int l_m_done_old = 0;
    unsigned int l_m_blocking = 0;
    unsigned int l_m_blocking_old = 0;

    /* advance N */
    l_n_done_old = l_n_done;
    l_n_done += l_n_N[l_n_count];
    l_n_count++;

    /* open N loop */
    libxsmm_generator_loop_header_rv64( io_generated_code, &l_loop_label_tracker,
                                        l_gp_reg_mapping.gp_reg_nloop, l_n_done - l_n_done_old );

    /* Set vector length to full vector */
    if (io_generated_code->arch != LIBXSMM_RV64_MVL128_LMUL && io_generated_code->arch != LIBXSMM_RV64_MVL128_LMUL)
      libxsmm_rv64_instruction_rvv_setivli( io_generated_code, l_micro_kernel_config.vector_length, l_gp_reg_mapping.gp_reg_help_5, l_sew, LIBXSMM_RV64_LMUL_M1);

    l_lmul = LIBXSMM_RV64_LMUL_M1;

    /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
    l_m_blocking = libxsmm_generator_gemm_rv64_get_initial_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch );

    l_m_blocking_old = 0;

    /* apply m_blocking */
    while (l_m_done != (unsigned int)l_xgemm_desc_opa->m) {
      if ( l_m_blocking == 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }

      l_lmul_new = l_lmul;

      if (REG_GP(io_generated_code)) {
        if ((l_m_blocking/l_micro_kernel_config.vector_length) == 2) {
          l_reg_gp = 2;
        }
        else if ((l_m_blocking/l_micro_kernel_config.vector_length) == 4) {
          l_reg_gp = 4;
        }
        else if ((l_m_blocking/l_micro_kernel_config.vector_length) == 8){
          l_reg_gp = 8;
        }
        else {
          l_reg_gp = 1;
        }
      }
      else {
        l_reg_gp = 1;
      }

      switch (l_reg_gp) {
        case 1:
          l_lmul = LIBXSMM_RV64_LMUL_M1;
          break;

        case 2:
          l_lmul = LIBXSMM_RV64_LMUL_M2;
          break;

        case 4:
          l_lmul = LIBXSMM_RV64_LMUL_M3;
          break;

        case 8:
          l_lmul = LIBXSMM_RV64_LMUL_M4;
          break;
      }

      if (l_lmul_new != l_lmul){
        libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_5, (unsigned long long)l_micro_kernel_config.vector_length * l_reg_gp );
        libxsmm_rv64_instruction_rvv_setvli( io_generated_code,  l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_5, l_sew, l_lmul);
      }

      if (l_m_blocking && (l_m_blocking < l_micro_kernel_config.vector_length)){
        libxsmm_rv64_instruction_rvv_setivli( io_generated_code, l_m_blocking, l_gp_reg_mapping.gp_reg_help_5, l_sew, LIBXSMM_RV64_LMUL_M1);
      }

      l_m_done_old = l_m_done;
      LIBXSMM_ASSERT(0 != l_m_blocking);
      /* coverity[divide_by_zero] */
      l_m_done = l_m_done + (((l_xgemm_desc_opa->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        /* open M loop */
        libxsmm_generator_loop_header_rv64( io_generated_code, &l_loop_label_tracker,
                                               l_gp_reg_mapping.gp_reg_mloop, l_m_done - l_m_done_old );
        /* load block of C */
        if ( io_generated_code->arch >= LIBXSMM_RV64_MVL128 ) {
          if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
            libxsmm_generator_gemm_load_add_colbias_2dregblock_rv64( io_generated_code, l_xgemm_desc_opa, l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, (l_micro_kernel_config.fused_scolbias == 1) ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16, l_m_blocking, l_n_blocking, l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out );
          }
          else {
            libxsmm_generator_load_2dregblock_rv64_rvv( io_generated_code, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_0,
                l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                (l_is_i8f32_gemm > 0) ? 1 : (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags), l_reg_gp );
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
          return;
        }

        /* handle BRGEMM */
        if ( ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) >  0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) >  0) ) {
          /* we need to load the real address */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_4, 0 );
            libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_5, 0 );
          } else {
            libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_3, 0 );
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                        l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_4, 0 );
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                        l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_5, 0 );
          }

          /* open BR loop */
          libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_reduce_count,
                                                l_gp_reg_mapping.gp_reg_reduce_loop, 0 );

          libxsmm_rv64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );

          /* we need to load the real address of A and B for this reduce operation */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
            /* Compute the effective address */
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_4,
                                                  l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

            libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
              0);


            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_5,
                                                  l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

            libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b,
              0);
          }

          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) > 0 ) {
            /* Compute the effective address */
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_a_offset,
                                                                 l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

            libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_0,
              0);

            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_4,
                                                                 l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a );


            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_b_offset,
                                                                 l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

            libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
              0);

            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_5,
                                                                 l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b );
          }
        }

        /* compute outer product */
        libxsmm_generator_gemm_rv64_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                              l_xgemm_desc_opa, l_m_blocking, l_n_blocking, l_reg_gp );

        if ( ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) >  0) ||
             ((l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) >  0)    ) {
          /* increment forward counting BRGEMM count */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            /* nothing to do */
          } else {
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                        l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8 );
          }

          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_0, l_xgemm_desc_opa->c1 );
            libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, l_xgemm_desc_opa->c2 );
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_0,
                                                                 l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_a );
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_1,
                                                                 l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_b );

            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_0,
                                                                 l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_4 );
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_1,
                                                                 l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_help_5 );
          }

          /* close BRGEMM loop */
          libxsmm_generator_loop_footer_rv64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_reduce_loop, 1 );

          /* restore A and B register */
          if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) > 0 ) {
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, l_gp_reg_mapping.gp_reg_a,
                                                                 l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a );
            libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, l_gp_reg_mapping.gp_reg_b,
                                                                 l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_b );
          } else {
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                           l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0 );
            libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                           l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_b, 0 );
          }
        }

        /* store block of C */
        if ( io_generated_code->arch >= LIBXSMM_RV64_MVL128 ) {
           libxsmm_generator_gemm_apply_fusion_2dregblock_rv64( io_generated_code, l_xgemm_desc_opa, &l_micro_kernel_config, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1, l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking  );

          libxsmm_generator_store_2dregblock_rv64_rvv( io_generated_code, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ), l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_4,
                                                       l_micro_kernel_config.vector_length, l_micro_kernel_config.vector_reg_count, l_m_blocking, l_n_blocking,
                                                       l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out,
                                                       ( l_is_i8f32_gemm > 0 ) ? (libxsmm_datatype)LIBXSMM_DATATYPE_I32 : (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ),
                                                       ( l_is_i8f32_gemm > 0 ) ? l_gp_reg_mapping.gp_reg_scf : 0,
                                                       ( l_is_i8f32_gemm > 0 ) ? (LIBXSMM_GEMM_FLAG_BETA_0 & l_xgemm_desc_opa->flags) == 0 ? 1 : 0 : 0, l_reg_gp );
        }

        /* advance C pointer */
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                    l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                    (long long)l_m_blocking*l_micro_kernel_config.datatype_size_out );

        /* Adjust colbias ptr */
        if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
          libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_1);
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                      l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                      (l_micro_kernel_config.fused_scolbias == 1) ? (long long)l_m_blocking*4 : (long long)l_m_blocking*2 );
          libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
        }

        /* advance A pointer */
        if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
          libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_3, 0 );
          libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                         l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_4, 0 );

          /* open BR loop */
          libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_reduce_count,
            l_gp_reg_mapping.gp_reg_reduce_loop, 0 );
          libxsmm_rv64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );

          /* update A pointer */
          libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_4,
                                                l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

          libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a, 0 );

          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                         l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                         (long long)l_m_blocking*l_micro_kernel_config.datatype_size_in*a_vnni_factor );

          libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_4,
                                                l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

          libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                0 );

          libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                         l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8 );
          /* close BRGEMM loop */
          libxsmm_generator_loop_footer_rv64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_reduce_loop, 1 );

          /* reset A */
          libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                         l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0 );
        } else {
          if ((l_m_done != (unsigned int)l_xgemm_desc_opa->m) || (l_m_blocking < (unsigned int)l_xgemm_desc_opa->m)){
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,  (long long)l_m_blocking*l_micro_kernel_config.datatype_size_in );
            l_m_blocking_old += l_m_done - l_m_done_old;
          }
        }
        /* close M loop */
        libxsmm_generator_loop_footer_rv64( io_generated_code, &l_loop_label_tracker,
                                               l_gp_reg_mapping.gp_reg_mloop, l_m_blocking );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_rv64_update_m_blocking( &l_micro_kernel_config, l_xgemm_desc_opa, io_generated_code->arch, l_m_blocking );
    }

    /* reset C pointer */
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                   ((long long)l_n_blocking * l_xgemm_desc_opa->ldc * l_micro_kernel_config.datatype_size_out) -
                                                   ((long long)l_xgemm_desc_opa->m * l_micro_kernel_config.datatype_size_out) );

    /* Adjust relu bitmask pointer */
    if ((l_micro_kernel_config.fused_relu == 1) && (l_micro_kernel_config.overwrite_C == 1) ) {
      libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                     (((long long)l_n_blocking*i_xgemm_desc->ldcp)/8) - (((long long)l_xgemm_desc_opa->m+7)/8));
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
    }

    /* Adjust colbias ptr */
    if ((l_micro_kernel_config.fused_scolbias == 1) || (l_micro_kernel_config.fused_bcolbias == 1)) {
      libxsmm_generator_gemm_getval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                     l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1,
                                                     (l_micro_kernel_config.fused_scolbias == 1) ? (long long)l_xgemm_desc_opa->m*4 : (long long)l_xgemm_desc_opa->m *2 );
      libxsmm_generator_gemm_setval_stack_var_rv64( io_generated_code, LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_help_1);
    }

    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
      libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_3, 0 );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                    l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_4, 0 );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                    l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_5, 0 );

      /* open BR loop */
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_reduce_count,
                                          l_gp_reg_mapping.gp_reg_reduce_loop, 0);
      libxsmm_rv64_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );

      /* update A pointer */
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_4,
                                            l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0,
                                          l_gp_reg_mapping.gp_reg_a, 0 );

      /* update B pointer */
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_5,
                                            l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );

      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, l_gp_reg_mapping.gp_reg_help_0,
                                          l_gp_reg_mapping.gp_reg_b, 0 );
    }

    /* advance B pointer */
    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 &&  (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_VNNI_B) == 0 ) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (long long)l_n_blocking * l_micro_kernel_config.datatype_size_in );
    } else if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 && (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (long long)l_n_blocking * a_vnni_factor * l_micro_kernel_config.datatype_size_in );
    } else {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_b,
                                                     (long long)l_n_blocking * l_xgemm_desc_opa->ldb * l_micro_kernel_config.datatype_size_in );
    }

    /* reset A pointer */
    if (l_m_blocking_old){
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
          l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,  (long long)l_m_blocking_old*l_micro_kernel_config.datatype_size_in );
    }

    if ( (l_xgemm_desc_opa->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) > 0 ) {
      /* reset A pointer */
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                  l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                  (long long)l_xgemm_desc_opa->m*l_micro_kernel_config.datatype_size_in*a_vnni_factor );
      /* Compute the effective address and store A and B */

      /* Store A addresses */
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_4,
                                                           l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
              0);

      /* Store B addresses */
      libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD, l_gp_reg_mapping.gp_reg_help_5,
                                                           l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_0 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b,
              0);

      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                     l_gp_reg_mapping.gp_reg_help_3, l_gp_reg_mapping.gp_reg_help_3, 8 );
      /* close BRGEMM loop */
      libxsmm_generator_loop_footer_rv64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_reduce_loop, 1 );

      /* reset A and B */
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                     l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_a, 0 );
      libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                     l_gp_reg_mapping.gp_reg_help_5, l_gp_reg_mapping.gp_reg_b, 0 );
    }

    /* close N loop */
    libxsmm_generator_loop_footer_rv64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_nloop, l_n_blocking );
  }

  libxsmm_generator_gemm_destroy_stack_frame_rv64( io_generated_code );

  /* close asm */
  libxsmm_rv64_instruction_close_stream( io_generated_code, 0x3fff );
}

#undef MAX_FP_REG
#undef MAX_UIMM
#undef REUSE_A
#undef REUSE_B
#undef REUSE_C
#undef PREFETCH_A
#undef PREFETCH_B
