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
#include "generator_common.h"
#include "generator_x86_instructions.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse3_avx_avx2_avx512.h"
#include "generator_gemm_sse3_microkernel.h"
#include "generator_gemm_avx_microkernel.h"
#include "generator_gemm_avx2_microkernel.h"
#include "generator_gemm_avx512_microkernel.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse3_avx_avx2_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                                            const libxsmm_gemm_descriptor* i_xgemm_desc       ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* initialize n-blocking */
  unsigned int l_n_count = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_n[2] = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2] = {0,0};       /* size of blocks */

  unsigned int adjust_A_pf_ptrs = 0;
  unsigned int adjust_B_pf_ptrs = 0;

  /* Make sure we properly adjust A,B prefetch pointers in case of batch-reduce gemm kernel  */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      adjust_A_pf_ptrs = 1;
    }
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  if ( (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) ) {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
  } else {
    l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  }
  /* If we are generating the batchreduce kernel, then we rename the registers  */
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    if ( (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R9;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
      l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
      l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
    l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R14;
  } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_offset = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_offset = LIBXSMM_X86_GP_REG_R9;
    if ( (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype )) ) {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_RAX;
    } else {
      l_gp_reg_mapping.gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
    }
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R13;
    l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R14;
  }
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RBX;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* block according to the number of available registers or given limits */
  if ( i_xgemm_desc->n == 7 && io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE && io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
    libxsmm_compute_equalized_blocking( i_xgemm_desc->n, 7, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );
  } else {
    unsigned int max_n_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_max_n_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch );
#if 1
    if (3 < max_n_blocking)
#endif
    {
      const unsigned int init_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_initial_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch );
      const unsigned int init_m_blocks = (init_m_blocking + l_micro_kernel_config.vector_length - 1) / l_micro_kernel_config.vector_length;
      while ((init_m_blocks * max_n_blocking + init_m_blocks + 1) > l_micro_kernel_config.vector_reg_count) {
        max_n_blocking--;
      }
    }
    libxsmm_compute_equalized_blocking( i_xgemm_desc->n, max_n_blocking, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );
  }
  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );

  /* Load the actual batch-reduce trip count */
  if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_micro_kernel_config.alu_mov_instruction,
        l_gp_reg_mapping.gp_reg_reduce_count,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        l_gp_reg_mapping.gp_reg_reduce_count,
        0 );
  }

  /* apply n_blocking */
  while (l_n_done != (unsigned int)i_xgemm_desc->n) {
    unsigned int l_n_blocking = l_n_n[l_n_count];
    unsigned int l_m_done = 0;
    unsigned int l_m_done_old = 0;
    unsigned int l_m_blocking = 0;

    /* advance N */
    l_n_done += l_n_N[l_n_count];
    l_n_count++;

    /* open N loop */
    libxsmm_generator_gemm_header_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_n_blocking );

    /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
    l_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_initial_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch );

    /* apply m_blocking */
    while (l_m_done != (unsigned int)i_xgemm_desc->m) {
      if ( l_m_blocking == 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
        return;
      }

      if (l_m_done == 0) {
        /* This is a SeisSol Order 6, HSW, DP performance fix */
        if ( ( io_generated_code->arch == LIBXSMM_X86_AVX2 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
          l_m_done_old = l_m_done;
          if (i_xgemm_desc->m == 56) {
            l_m_done = 32;
          } else {
            LIBXSMM_ASSERT(0 != l_m_blocking);
            /* coverity[divide_by_zero] */
            l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
          }
        } else {
          l_m_done_old = l_m_done;
          LIBXSMM_ASSERT(0 != l_m_blocking);
          /* coverity[divide_by_zero] */
          l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
        }
      } else {
        l_m_done_old = l_m_done;
        LIBXSMM_ASSERT(0 != l_m_blocking);
        /* coverity[divide_by_zero] */
        l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
      }

      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        /* when on AVX512, load mask, if needed */
        if ( ( l_micro_kernel_config.use_masking_a_c != 0 ) && ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
          /* compute the mask count, depends on vlen as block in M */
          unsigned int l_corrected_vlen = l_micro_kernel_config.vector_length;
          unsigned int l_mask_count = l_corrected_vlen - ( l_m_blocking % l_corrected_vlen );

          libxsmm_generator_gemm_initialize_avx512_mask( io_generated_code, l_gp_reg_mapping.gp_reg_help_1, i_xgemm_desc, l_mask_count );
        }

        libxsmm_generator_gemm_header_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking );
        libxsmm_generator_gemm_load_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );

        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
          if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
          }
          /* This is the reduce loop  */
          libxsmm_generator_gemm_header_reduceloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config );
          if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
            if (adjust_A_pf_ptrs) {
              /* coverity[dead_error_line] */
              libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a_prefetch );
            }
            if (adjust_B_pf_ptrs) {
              libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b_prefetch );
            }
            /* load to reg_a the proper array based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_a,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_a,
                0 );
            /* load to reg_b the proper array based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_b,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_b,
                0 );
            if (adjust_A_pf_ptrs) {
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  l_gp_reg_mapping.gp_reg_a_prefetch,
                  l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                  0,
                  l_gp_reg_mapping.gp_reg_a_prefetch,
                  0 );
            }
            if (adjust_B_pf_ptrs) {
              /* coverity[dead_error_line] */
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  l_gp_reg_mapping.gp_reg_b_prefetch,
                  l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                  0,
                  l_gp_reg_mapping.gp_reg_b_prefetch,
                  0 );
            }
          } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
            /* Calculate to reg_a the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_a_offset,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a);
            /* Calculate to reg_b the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_b_offset,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b);
          } else if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
            libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
            /* Calculate to reg_a the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_reduce_loop, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_gp_reg_mapping.gp_reg_help_0, i_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a);
            /* Calculate to reg_b the proper address based on the reduce loop index  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_reduce_loop, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_gp_reg_mapping.gp_reg_help_0, i_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b);
          }
        }

        libxsmm_generator_gemm_sse3_avx_avx2_avx512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                           i_xgemm_desc, l_m_blocking, l_n_blocking );

        if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
          if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) {
            if (adjust_B_pf_ptrs) {
              /* coverity[dead_error_begin] */
              libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  l_gp_reg_mapping.gp_reg_help_0,
                  l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                  0,
                  l_gp_reg_mapping.gp_reg_b_prefetch,
                  1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b_prefetch);
            }
            if (adjust_A_pf_ptrs) {
              libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
              libxsmm_x86_instruction_alu_mem( io_generated_code,
                  l_micro_kernel_config.alu_mov_instruction,
                  l_gp_reg_mapping.gp_reg_help_0,
                  l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                  0,
                  l_gp_reg_mapping.gp_reg_a_prefetch,
                  1 );
              libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a_prefetch);
            }
            /* Pop address of B_array to help_0 and store proper address of B   */
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_help_0,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_b,
                1 );
            /* Move to reg_b the address of B_array  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b);
            /* Pop address of A_array to help_0 and store proper address of A   */
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_help_0,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_a,
                1 );
            /* Move to reg_a the address of A_array  */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a);
          }
          libxsmm_generator_gemm_footer_reduceloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc);
          if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) {
            /* Calculate to reg_a the proper A advance form the microkernel */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_a_offset,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                -8,
                l_gp_reg_mapping.gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a);
            /* Calculate to reg_b the proper B advance form the microkernel */
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_b_offset,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                -8,
                l_gp_reg_mapping.gp_reg_help_0,
                0 );
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b);
            /* Consume the last two pushes form the stack */
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
          }
          if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE) {
            /* Calculate to reg_a the proper A advance form the microkernel */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_reduce_count, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_gp_reg_mapping.gp_reg_help_0, i_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_help_0, i_xgemm_desc->c1);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a);
            /* Calculate to reg_b the proper B advance form the microkernel */
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_reduce_count, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_IMUL, l_gp_reg_mapping.gp_reg_help_0, i_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_help_0, i_xgemm_desc->c2);
            libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_b);
            /* Consume the last two pushes form the stack */
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
            libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0);
          }
        }

        libxsmm_generator_gemm_store_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );
        libxsmm_generator_gemm_footer_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( &l_micro_kernel_config, i_xgemm_desc, io_generated_code->arch, l_m_blocking );
    }
    libxsmm_generator_gemm_footer_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n_blocking, l_n_done );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_sse3_avx_avx2_avx512_kloop( libxsmm_generated_code*            io_generated_code,
                                                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const unsigned int                 i_m_blocking,
                                                                           const unsigned int                 i_n_blocking ) {
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                  const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int, const int);

  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 0;
  unsigned int l_k_threshold = 0;

  /* calculate m_blocking such that we choose the right AVX512 kernel */
  unsigned int l_m_vector = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;

  /* in case of 1d blocking and KNL/KNM we unroll aggressively */
  if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_AVX512_KNM ) && ( l_m_vector == 1 ) ) {
    l_k_blocking = 16;
    l_k_threshold = 47;
  } else {
    l_k_blocking = 4;
    l_k_threshold = 23;
  }

  /* set up architecture dependent compute micro kernel generator */
  if ( io_generated_code->arch < LIBXSMM_X86_SSE3 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  } else if ( io_generated_code->arch <= LIBXSMM_X86_SSE4 ) {
    l_generator_microkernel = libxsmm_generator_gemm_sse3_microkernel;
  } else if ( io_generated_code->arch == LIBXSMM_X86_AVX ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx_microkernel;
  } else if ( io_generated_code->arch == LIBXSMM_X86_AVX2 ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel;
  } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* apply multiple k_blocking strategies */
  /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
  if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
    unsigned int l_k;
    libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_m_blocking, l_k_blocking);

    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( l_m_vector == 1 ) ) {
      if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
        libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_n_blocking, l_k_blocking );
      } else {
        libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_n_blocking, l_k_blocking );
      }
    } else {
      for ( l_k = 0; l_k < l_k_blocking; l_k++) {
        l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_m_blocking, i_n_blocking, -1);
      }
    }

    libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
      i_xgemm_desc, i_m_blocking, i_xgemm_desc->k, 1 );
  } else {
    /* 2. we want to fully unroll below the threshold */
    if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
      unsigned int l_k;

      if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( l_m_vector == 1 ) ) {
        if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, (unsigned int)i_xgemm_desc->k );
        } else {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, (unsigned int)i_xgemm_desc->k );
        }
      } else {
        for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
          l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_m_blocking, i_n_blocking, l_k);
        }
      }
      /* 3. we are larger than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
    } else {
      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
      unsigned int l_k;
      int l_b_offset = 0;

      /* we can block as k is large enough */
      if ( l_max_blocked_k > 0 ) {
        libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_m_blocking, l_k_blocking);

        if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( l_m_vector == 1 ) ) {
          if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
            libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              i_xgemm_desc, i_n_blocking, l_k_blocking );
          } else {
            libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              i_xgemm_desc, i_n_blocking, l_k_blocking );
          }
        } else {
          for ( l_k = 0; l_k < l_k_blocking; l_k++) {
            l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
              i_xgemm_desc, i_m_blocking, i_n_blocking, -1);
          }
        }

        libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
          i_xgemm_desc, i_m_blocking, l_max_blocked_k, 0 );
      }

      /* now we handle the remainder handling */
      if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( l_m_vector == 1 ) ) {
        if ( io_generated_code->arch != LIBXSMM_X86_AVX512_KNM ) {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, ((unsigned int)i_xgemm_desc->k) - l_max_blocked_k );
        } else {
          libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_n_blocking, ((unsigned int)i_xgemm_desc->k) - l_max_blocked_k );
        }
      } else {
        for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
          l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
            i_xgemm_desc, i_m_blocking, i_n_blocking, -1);
        }
      }

      /* reset B pointer */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = i_xgemm_desc->ldb * i_xgemm_desc->k * i_micro_kernel_config->datatype_size;
      } else {
        l_b_offset = i_xgemm_desc->k * i_micro_kernel_config->datatype_size;
      }

      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction,
        i_gp_reg_mapping->gp_reg_b, l_b_offset );
    }
  }
}


LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                                                    const unsigned int              i_arch ) {
  unsigned int l_use_masking_a_c = 0;
  unsigned int l_m_blocking = 0;

  if ( ( i_arch <= LIBXSMM_X86_SSE4 )           && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 12;
  } else if ( ( i_arch <= LIBXSMM_X86_SSE4 )    && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 6;
  } else if ( ( i_arch == LIBXSMM_X86_AVX )     && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 24;
  } else if ( ( i_arch == LIBXSMM_X86_AVX )     && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 12;
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 )    && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 32;
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 )    && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 16;
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
    }
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 8 ) {
      l_m_blocking = 8;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if ( l_m_blocking % 8 != 0 ) {
        l_use_masking_a_c = 1;
      }
    }
  } else if ( ( i_arch <= LIBXSMM_X86_AVX512_CLX ) && ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
    }
  } else if ( ( i_arch <= LIBXSMM_X86_AVX512_CORE ) && ( ( LIBXSMM_GEMM_PRECISION_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )  ||
                                                         ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )      ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 16 ) {
      l_m_blocking = 16;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
    }
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                                                     ( LIBXSMM_GEMM_PRECISION_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                                                     ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                                                     ( LIBXSMM_GEMM_PRECISION_I8   == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )     ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32/Int32 out kernel with the same logic */
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 64 ) {
      l_m_blocking = 64;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
    }
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    /* @TODO check if there is a better blocking strategy */
    if ( i_xgemm_desc->m >= 32 ) {
      l_m_blocking = 32;
    } else {
      l_m_blocking = i_xgemm_desc->m;
      /* in case we don't have a full vector length, we use masking */
      if ( l_m_blocking % 8 != 0 ) {
        l_use_masking_a_c = 1;
      }
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );

  return l_m_blocking;
}

LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                                               const unsigned int             i_arch,
                                                                                               const unsigned int             i_current_m_blocking ) {
  unsigned int l_use_masking_a_c = 0;
  unsigned int l_m_blocking = 0;

  if ( ( i_arch <= LIBXSMM_X86_SSE4 ) && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_SSE4 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
    } else if (i_current_m_blocking == 6) {
      l_m_blocking = 4;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_X86_AVX ) && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_X86_AVX ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 ) && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 24;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch == LIBXSMM_X86_AVX2 ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_arch, i_xgemm_desc, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 12;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
      libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( (i_arch == LIBXSMM_X86_AVX512_MIC) || (i_arch == LIBXSMM_X86_AVX512_KNM) ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = i_xgemm_desc->m % 8;
      if ( l_m_blocking % 8 != 0 ) {
        l_use_masking_a_c = 1;
      }
      libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_AVX512_CLX ) && ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
      libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_AVX512_CORE ) && ( ( LIBXSMM_GEMM_PRECISION_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )  ||
                                                         ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )      ) ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = i_xgemm_desc->m % 16;
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
      libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( ( LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                                                     ( LIBXSMM_GEMM_PRECISION_I32  == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                                                     ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )  ||
                                                     ( LIBXSMM_GEMM_PRECISION_I8   == LIBXSMM_GETENUM_OUT( i_xgemm_desc->datatype ) )      ) ) {
    /* Remark switching ti OUT datatype check here to cover BF16 in, Fp32 out kernel with the same logic */
    if (i_current_m_blocking == 64) {
      l_m_blocking = i_xgemm_desc->m % 64;
      if ( l_m_blocking % 16 != 0 ) {
        l_use_masking_a_c = 1;
      }
      libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
    } else {
      /* we are done with m_blocking */
    }
  } else if ( ( i_arch <= LIBXSMM_X86_ALLFEAT ) && ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 32) {
      l_m_blocking = i_xgemm_desc->m % 32;
      if ( l_m_blocking % 8 != 0 ) {
        l_use_masking_a_c = 1;
      }
      libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_arch, i_xgemm_desc, l_use_masking_a_c );
    } else {
      /* we are done with m_blocking */
    }
  } else {
    /* we should never end up here, if we do let the user know */
    /*LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return 0;*/
  }

  return l_m_blocking;
}


LIBXSMM_API_INTERN unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                                                const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                                                const unsigned int                  i_arch ) {
  if ( i_arch <= LIBXSMM_X86_ALLFEAT ) {
    if ( i_arch >= LIBXSMM_X86_AVX512 ) {
      /* handle KNM qmadd */
      if ( ( i_arch == LIBXSMM_X86_AVX512_KNM ) && ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
        return 28;
      }
      /* handle KNM qvnni */
      if ( ( i_arch == LIBXSMM_X86_AVX512_KNM ) && ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
        return 28;
      }
      /* handle int16 on SKX */
      if ( ( i_arch == LIBXSMM_X86_AVX512_CORE ) && ( LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
        return 28;
      }
      /* handle int8 on all AVX512 */
      if ( ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
        return 28;
      }
      /* handle bf16 */
      if ( ( i_arch < LIBXSMM_X86_AVX512_CPX ) && ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
        return 28;
      }
      return 30;
    }
    else {
#if 1
      LIBXSMM_UNUSED(i_micro_kernel_config);
      return 3;
#else
      return i_micro_kernel_config->vector_reg_count - 2;
#endif
    }
  }
  else {
    return 3;
  }
}

