/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "generator_common.h"
#include "generator_x86_instructions.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse3_avx_avx2_avx512.h"
#include "generator_gemm_sse3_microkernel.h"
#include "generator_gemm_avx_microkernel.h"
#include "generator_gemm_avx2_microkernel.h"
#include "generator_gemm_avx512_microkernel_nofsdbcst.h"

LIBXSMM_INTERNAL_API_DEFINITION
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
  unsigned int l_k_threshold = 30;

  /* initialize n-blocking */
  unsigned int l_n_done = 0;
  unsigned int l_n_done_old = 0;
  unsigned int l_n_blocking = 3;

  /* as we have 32 registers, we can block more aggessively */
  if ( (strcmp(i_arch, "skx") == 0) ) {
    l_n_blocking = 6;
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
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
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* apply n_blocking */
  while (l_n_done != (unsigned int)i_xgemm_desc->n) {
    l_n_done_old = l_n_done;
    l_n_done = l_n_done + (((i_xgemm_desc->n - l_n_done_old) / l_n_blocking) * l_n_blocking);

    if (l_n_done != l_n_done_old && l_n_done > 0) {
      unsigned int l_m_done = 0;
      unsigned int l_m_done_old = 0;
      unsigned int l_m_blocking = 0;

      libxsmm_generator_gemm_header_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_n_blocking );
      /* define the micro kernel code gen properties, especially m-blocking affects the vector instruction length */
      l_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_inital_m_blocking( &l_micro_kernel_config, i_xgemm_desc, i_arch );

      /* apply m_blocking */
      while (l_m_done != (unsigned int)i_xgemm_desc->m) {
        if (l_m_done == 0) {
          /* This is a SeisSol Order 6, HSW, DP performance fix */
          if ( (strcmp( i_arch, "hsw" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
            l_m_done_old = l_m_done;
            if (i_xgemm_desc->m == 56) {
              l_m_done = 32;
            } else {
              l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
            }
          } else {
            l_m_done_old = l_m_done;
            l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
          }
        } else {
          l_m_done_old = l_m_done;
          l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
        }

        if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
          libxsmm_generator_gemm_header_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking );
          libxsmm_generator_gemm_load_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );

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
                libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction,
                                             l_gp_reg_mapping.gp_reg_b, l_max_blocked_k * l_micro_kernel_config.datatype_size );
              }
              for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
                l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config,
                                        i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
              }
            }
          }

          libxsmm_generator_gemm_store_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );
          libxsmm_generator_gemm_footer_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done, 0 );
        }

        /* switch to next smaller m_blocking */
        l_m_blocking = libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( &l_micro_kernel_config, i_xgemm_desc, i_arch, l_m_blocking );
      }
      libxsmm_generator_gemm_footer_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n_blocking, l_n_done );
    }

    /* switch to a different, smaller n_blocking */
    if (l_n_blocking == 2) {
      l_n_blocking = 1;
    } else if (l_n_blocking == 3) {
      l_n_blocking = 2;
    } else if (l_n_blocking == 4) {
      l_n_blocking = 3;
    } else if (l_n_blocking == 5) {
      l_n_blocking = 4;
    } else if (l_n_blocking == 6) {
      l_n_blocking = 5;
    } else {
      /* we are done with n_blocking */
    }
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}

LIBXSMM_INTERNAL_API_DEFINITION
unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_get_inital_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                                const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                                const char*                    i_arch ) {
  unsigned int l_m_blocking = 0;

  if ( (strcmp( i_arch, "wsm" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
    l_m_blocking = 12;
  } else if ( (strcmp( i_arch, "wsm" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
    l_m_blocking = 6;
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
    l_m_blocking = 24;
  } else if ( (strcmp( i_arch, "snb" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
    l_m_blocking = 12;
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
    l_m_blocking = 32;
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
    l_m_blocking = 16;
  } else if ( (strcmp( i_arch, "skx" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
    l_m_blocking = 64;
  } else if ( (strcmp( i_arch, "skx" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
    l_m_blocking = 32;
  } else { }

  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  return l_m_blocking;
}

LIBXSMM_INTERNAL_API_DEFINITION
unsigned int libxsmm_generator_gemm_sse3_avx_avx2_avx512_update_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                            const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                            const char*                    i_arch,
                                                                            const unsigned int             i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( (strcmp( i_arch, "wsm" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
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
  } else if ( (strcmp( i_arch, "wsm" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
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
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
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
  } else if ( (strcmp( i_arch, "snb" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
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
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
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
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
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
  } else if ( (strcmp( i_arch, "skx" ) == 0) && (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) != 0 ) {
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
  } else if ( (strcmp( i_arch, "skx" ) == 0) && ((LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0) ) {
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

