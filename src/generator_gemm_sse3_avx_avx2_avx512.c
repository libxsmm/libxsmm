/******************************************************************************
** Copyright (c) 2015-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
#include "generator_gemm_avx512_microkernel_nofsdbcst.h"
#include "libxsmm_main.h"

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
void libxsmm_generator_gemm_sse3_avx_avx2_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                         const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                         const char*                    i_arch ) {
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                  const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int, const int);
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 4;
  unsigned int l_k_threshold = 23;

  /* initialize n-blocking */
  unsigned int l_n_count = 0;          /* array counter for blocking arrays */
  unsigned int l_n_done = 0;           /* progress tracker */
  unsigned int l_n_n[2] = {0,0};       /* blocking sizes for blocks */
  unsigned int l_n_N[2] = {0,0};       /* size of blocks */

  unsigned int adjust_A_pf_ptrs = 0;
  unsigned int adjust_B_pf_ptrs = 0;

  /* as we have 32 registers, we can block more aggressively */
  /* @TODO: take M blocking into account */
  if ( (strcmp(i_arch, "skx") == 0) ||
       (strcmp(i_arch, "clx") == 0) ||
       (strcmp(i_arch, "cpx") == 0)   ) {
    libxsmm_compute_equalized_blocking( i_xgemm_desc->n, 6, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );
  } else {
    libxsmm_compute_equalized_blocking( i_xgemm_desc->n, 3, &(l_n_N[0]), &(l_n_n[0]), &(l_n_N[1]), &(l_n_n[1]) );
  }
  /* check that l_n_N1 is non-zero */
  if ( l_n_N[0] == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* Make sure we properly adjust A,B prefetch pointers in case of batch-reduce gemm kernel  */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE) {
    if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_AL1 ||
        i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_JPST ||
        i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_JPST ||
        i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
        i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
      adjust_A_pf_ptrs = 1;
    }
    if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
      adjust_B_pf_ptrs = 1;
    }
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  /* If we are generating the batchreduce kernel, then we rename the registers  */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE) {
    l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
    l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
    l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
    l_gp_reg_mapping.gp_reg_reduce_count = LIBXSMM_X86_GP_REG_RCX;
    l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R8;
    l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R9;
    l_gp_reg_mapping.gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_R10;
    l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R12;
  }
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* set up architecture dependent compute micro kernel generator */
  if ( (strcmp(i_arch, "wsm") == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_sse3_microkernel;
  } else if ( (strcmp(i_arch, "snb") == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx_microkernel;
  } else if ( (strcmp(i_arch, "hsw") == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel;
  } else if ( (strcmp(i_arch, "skx") == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
  } else if ( (strcmp(i_arch, "clx") == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
  } else if ( (strcmp(i_arch, "cpx") == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* Load the actual batch-reduce trip count */
  if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE) {
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
    l_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_initial_m_blocking( &l_micro_kernel_config, i_xgemm_desc, i_arch );

    /* apply m_blocking */
    while (l_m_done != (unsigned int)i_xgemm_desc->m) {
      if (l_m_done == 0) {
        /* This is a SeisSol Order 6, HSW, DP performance fix */
        if ( (strcmp( i_arch, "hsw" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
          l_m_done_old = l_m_done;
          if (i_xgemm_desc->m == 56) {
            l_m_done = 32;
          } else {
            assert(0 != l_m_blocking);
            l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
          }
        } else {
          l_m_done_old = l_m_done;
          assert(0 != l_m_blocking);
          l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
        }
      } else {
        l_m_done_old = l_m_done;
        assert(0 != l_m_blocking);
        l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
      }

      if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
        libxsmm_generator_gemm_header_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking );
        libxsmm_generator_gemm_load_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );

        if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE) {
          libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_mloop);
          /* This is the reduce loop  */
          libxsmm_generator_gemm_header_reduceloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config );
          libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_a);
          libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_b);
          if (adjust_A_pf_ptrs) {
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
            libxsmm_x86_instruction_alu_mem( io_generated_code,
                l_micro_kernel_config.alu_mov_instruction,
                l_gp_reg_mapping.gp_reg_b_prefetch,
                l_gp_reg_mapping.gp_reg_reduce_loop, 8,
                0,
                l_gp_reg_mapping.gp_reg_b_prefetch,
                0 );
          }
        }

        /* apply multiple k_blocking strategies */
        /* 1. we are larger the k_threshold and a multiple of a predefined blocking parameter */
        if ((i_xgemm_desc->k % l_k_blocking) == 0 && (l_k_threshold < (unsigned int)i_xgemm_desc->k)) {
          unsigned int l_k;
          libxsmm_generator_gemm_header_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking, l_k_blocking);

          for ( l_k = 0; l_k < l_k_blocking; l_k++) {
            l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config,
                i_xgemm_desc, l_m_blocking, l_n_blocking, -1);
          }

          libxsmm_generator_gemm_footer_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
              i_xgemm_desc, l_m_blocking, i_xgemm_desc->k, 1 );
        } else {
          /* 2. we want to fully unroll below the threshold */
          if ((unsigned int)i_xgemm_desc->k <= l_k_threshold) {
            unsigned int l_k;
            for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
              l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config,
                  i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
            }
            /* 3. we are large than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
          } else {
            unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
            unsigned int l_k;
            if ( l_max_blocked_k > 0 ) {
              libxsmm_generator_gemm_header_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking, l_k_blocking);

              for ( l_k = 0; l_k < l_k_blocking; l_k++) {
                l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config,
                    i_xgemm_desc, l_m_blocking, l_n_blocking, -1);
              }

              libxsmm_generator_gemm_footer_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                  i_xgemm_desc, l_m_blocking, l_max_blocked_k, 0 );
            }
            if (l_max_blocked_k > 0 ) {
              /* handle trans B */
              int l_b_offset = 0;
              if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
                l_b_offset = i_xgemm_desc->ldb * l_max_blocked_k * l_micro_kernel_config.datatype_size;
              } else {
                l_b_offset = l_max_blocked_k * l_micro_kernel_config.datatype_size;
              }

              libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction,
                  l_gp_reg_mapping.gp_reg_b, l_b_offset );
            }
            for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
              l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config,
                  i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
            }
          }
        }

        if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE) {
          if (adjust_B_pf_ptrs) {
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
          libxsmm_generator_gemm_footer_reduceloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc);
          libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_mloop );
        }

        libxsmm_generator_gemm_store_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );
        libxsmm_generator_gemm_footer_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done, 0 );
      }

      /* switch to next smaller m_blocking */
      l_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( &l_micro_kernel_config, i_xgemm_desc, i_arch, l_m_blocking );
    }
    libxsmm_generator_gemm_footer_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n_blocking, l_n_done );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
    const libxsmm_gemm_descriptor* i_xgemm_desc,
    const char*                    i_arch ) {
  unsigned int l_m_blocking = 0;

  if ( (strcmp( i_arch, "wsm" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_m_blocking = 12;
  } else if ( (strcmp( i_arch, "wsm" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 6;
  } else if ( (strcmp( i_arch, "snb" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_m_blocking = 24;
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 12;
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_m_blocking = 32;
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 16;
  } else if ( (strcmp( i_arch, "skx" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_m_blocking = 64;
  } else if ( (strcmp( i_arch, "skx" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 32;
  } else if ( (strcmp( i_arch, "clx" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_m_blocking = 64;
  } else if ( (strcmp( i_arch, "clx" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 32;
  } else if ( (strcmp( i_arch, "cpx" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_m_blocking = 64;
  } else if ( (strcmp( i_arch, "cpx" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    l_m_blocking = 32;
  } else { }

  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  return l_m_blocking;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
    const libxsmm_gemm_descriptor* i_xgemm_desc,
    const char*                    i_arch,
    const unsigned int             i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( (strcmp( i_arch, "wsm" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "wsm" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
    } else if (i_current_m_blocking == 6) {
      l_m_blocking = 4;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "snb" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 24;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
      libxsmm_generator_gemm_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 12;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "skx" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 48) {
      l_m_blocking = 32;
    } else if (i_current_m_blocking == 64) {
      l_m_blocking = 48;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "skx" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 24;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "clx" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 48) {
      l_m_blocking = 32;
    } else if (i_current_m_blocking == 64) {
      l_m_blocking = 48;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "clx" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 24;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "cpx" ) == 0) && LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if (i_current_m_blocking == 16) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 48) {
      l_m_blocking = 32;
    } else if (i_current_m_blocking == 64) {
      l_m_blocking = 48;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "cpx" ) == 0) && (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) ) {
    if (i_current_m_blocking == 8) {
      l_m_blocking = 1;
      libxsmm_generator_gemm_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 24;
    } else {
      /* we are done with m_blocking */
    }
  } else { }

  return l_m_blocking;
}

