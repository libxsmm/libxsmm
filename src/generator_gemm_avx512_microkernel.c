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
#include "generator_gemm_avx512_microkernel.h"
#include "generator_x86_instructions.h"
#include <libxsmm_macros.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_avx512_microkernel( libxsmm_generated_code*             io_generated_code,
                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                 i_n_blocking,
                                                 const unsigned int                 i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_b_reg;
  unsigned int l_b_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k_b = 0;
  unsigned int l_k_b_updates = 0;
  unsigned int l_displacement_k_a = 0;
  unsigned int l_k_a_update = 0;
  unsigned int l_n_accs = 0;

#if !defined(NDEBUG)
  if ( i_n_blocking > 30 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
#endif

  /* compute number of n accumulators to hide FMA latencies */
  if (i_n_blocking >= 12) {
    l_n_accs = 1;
  } else if (i_n_blocking >= 6) {
    l_n_accs = 2;
  } else {
    l_n_accs = 4;
  }

  /* Intialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_0, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_1, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_2, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_3, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb * 7 );

  /* helper 4: B + 9*ldb, additional base address
     helper 5: B + 18*ldb, additional base address
     helper 6: B + 27*ldb, additional base address, using the the prefetch b register, which was save to stack */
  if ( i_n_blocking > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_4,  9 * i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
  }
  if ( i_n_blocking > 18 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_5);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_5, 18 * i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
  }
  if ( i_n_blocking > 27 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_b_prefetch);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_b_prefetch, 27 * i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
  }

  l_displacement_k_b = 0;
  l_displacement_k_a = 0;

  /* xor additional accumulator, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vxor_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n );
    }
  }

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
    /* advance b pointer if needed */
    if ( (l_k > 0) && ((l_k%128) == 0) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b, 128*i_micro_kernel_config->datatype_size );
      /* advance the second base pointer only if it's needed */
      if ( i_n_blocking > 9 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_help_4, 128*i_micro_kernel_config->datatype_size );
      }
      /* advance the third base pointer only if it's needed */
      if ( i_n_blocking > 18 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_help_5, 128*i_micro_kernel_config->datatype_size );
      }
      /* advance the fourth base pointer only if it's needed */
      if ( i_n_blocking > 27 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_b_prefetch, 128*i_micro_kernel_config->datatype_size );
      }

      l_displacement_k_b = 0;
      l_k_b_updates++;
    }

    if ( l_k == 0 ) {
       /* load A */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * l_displacement_k_a * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        0,
                                        i_micro_kernel_config->use_masking_a_c, 0 );
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * l_displacement_k_a * i_micro_kernel_config->datatype_size) + 64 );
      }
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }
      if ( i_k_blocking > 1 ) {
        /* second A load in first iteration, in case of large blockings -> hiding L1 latencies */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->a_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          i_xgemm_desc->lda * l_displacement_k_a * i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          1,
                                          i_micro_kernel_config->use_masking_a_c, 0 );
        /* current A prefetch, next 8 rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
             i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_micro_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            (i_xgemm_desc->lda * l_displacement_k_a * i_micro_kernel_config->datatype_size) + 64 );
        }
        /* handle large displacements */
        if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
          l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
          l_displacement_k_a = 0;
        } else {
          l_displacement_k_a++;
        }
      }
    } else if ( l_k < (i_k_blocking - 1) ) {
      /* pipelined load of A, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * l_displacement_k_a * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        (l_k+1)%2,
                                        i_micro_kernel_config->use_masking_a_c, 0 );
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * l_displacement_k_a * i_micro_kernel_config->datatype_size) + 64 );
      }
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                        i_micro_kernel_config->prefetch_instruction,
                                        i_gp_reg_mapping->gp_reg_a_prefetch,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) && (i_k_blocking != (unsigned int)i_xgemm_desc->k) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a_prefetch,
                                         i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( (l_k == (i_k_blocking - 1)) && (i_k_blocking != (unsigned int)i_xgemm_desc->k) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      /* determining base, idx and scale values */
      l_b_reg = i_gp_reg_mapping->gp_reg_b;
      l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
      l_scale = 0;
      l_disp = l_displacement_k_b*i_micro_kernel_config->datatype_size;
      /* select the base register */
      if ( l_n > 26 ) {
        l_b_reg = i_gp_reg_mapping->gp_reg_b_prefetch;
      } else if ( l_n > 17 ) {
        l_b_reg = i_gp_reg_mapping->gp_reg_help_5;
      } else if ( l_n > 8 ) {
        l_b_reg = i_gp_reg_mapping->gp_reg_help_4;
      } else {
        l_b_reg = i_gp_reg_mapping->gp_reg_b;
      }
      /* Select SIB */
      if ( l_n % 9 == 0 ) {
        l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
        l_scale = 0;
      } else if ( l_n % 9 == 1 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_0;
        l_scale = 1;
      } else if ( l_n % 9 == 2 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_0;
        l_scale = 2;
      } else if ( l_n % 9 == 3 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_1;
        l_scale = 1;
      } else if ( l_n % 9 == 4 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_0;
        l_scale = 4;
      } else if ( l_n % 9 == 5 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_2;
        l_scale = 1;
      } else if ( l_n % 9 == 6 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_1;
        l_scale = 2;
      } else if ( l_n % 9 == 7 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_3;
        l_scale = 1;
      } else if ( l_n % 9 == 8 ) {
        l_b_idx = i_gp_reg_mapping->gp_reg_help_0;
        l_scale = 8;
      } else {
        /* shouldn't happen.... */
      }

#if 1
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               1,
                                               l_b_reg,
                                               l_b_idx,
                                               l_scale,
                                               l_disp,
                                               i_micro_kernel_config->vector_name,
                                               l_k%2,
                                               i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
#else
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               1,
                                               i_gp_reg_mapping->gp_reg_b,
                                               LIBXSMM_X86_GP_REG_UNDEF,
                                               0,
                                               (l_k * i_micro_kernel_config->datatype_size)+(i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                               i_micro_kernel_config->vector_name,
                                               l_k%2,
                                               i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
#endif
    }
    l_displacement_k_b++;
  }

  /* Adjust a pointer */
  if (l_k_a_update > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
                                     l_k_a_update );
  }

  /* advance pointers of B only when we are not fully unrolling K and taking care of intermediate advances */
  if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
    /* advance pointers of B */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_b,
                                     (i_k_blocking * i_micro_kernel_config->datatype_size) - (128*(i_micro_kernel_config->datatype_size)*l_k_b_updates) );
  } else {
    /* we have to make sure that we are reseting the pointer to its original value in case a full unroll */
    if ( l_k_b_updates > 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_sub_instruction,
                                       i_gp_reg_mapping->gp_reg_b, 128*(i_micro_kernel_config->datatype_size)*l_k_b_updates );
    }
  }

  /* add additional accumulators, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vadd_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
    }
  }
}

#if 0
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_avx512_microkernel_k_large( libxsmm_generated_code*             io_generated_code,
                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const unsigned int                 i_n_blocking,
                                                         const unsigned int                 i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;

#if !defined(NDEBUG)
  if ( i_n_blocking > 24 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( i_k_blocking < 8 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_K_BLOCK );
    return;
  }
#endif

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
    if ( l_k == 0 ) {
      /* load A, zmm0 + 1 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+0) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    0,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    1,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
    } else if ( l_k == 1 ) {
      /* load A, zmm2 + 3 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    2,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+2) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    3,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
    } else if ( l_k == 2 ) {
      /* load A, zmm4 + 5 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+2) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    4,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+3) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    5,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
    } else if ( l_k == 3 ) {
      /* load A, zmm6 + 7 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+3) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    6,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+4) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    7,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
    } else if ( l_k < (i_k_blocking - 4) ) {
      /* pipelined load of A, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+4) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    (l_k+4)%8,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
    }

    /* current A prefetch, next 8 rows for the current column */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) + 64 );
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a_prefetch,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( l_k == (i_k_blocking - 1) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vmul_instruction,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           (l_k * i_micro_kernel_config->datatype_size)+(i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
    }
  }

  /* advance pointers of B only when we are not fully unrolling K*/
  if ( i_k_blocking < i_xgemm_desc->k ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_b,
                                 i_k_blocking * i_micro_kernel_config->datatype_size );
  }
}
#endif

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_avx512_microkernel_k_large_n_nine( libxsmm_generated_code*             io_generated_code,
                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                const unsigned int                 i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;
  const unsigned int l_n_blocking = 9;
  unsigned int l_displacement_k_b = 0;
  unsigned int l_k_b_updates = 0;
  unsigned int l_displacement_k_a = 0;
  unsigned int l_k_a_update = 0;

#if !defined(NDEBUG)
  if ( i_k_blocking < 8 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_K_BLOCK );
    return;
  }
#endif

  /* Intialize helper registers for SIB addressing */
  if ( i_k_blocking != 9 ) {
    /* helper 0: Index register holding ldb*datatype_size */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                                     i_gp_reg_mapping->gp_reg_help_0, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
    /* helper 1: B + 3*ldb, additional base address
      helper 2: B + 6*ldb, additional base adrress */
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1);
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_2);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_1, 3 * i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_2, 6 * i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
  }

  /* init a displacement for k unrolling */
  l_displacement_k_a = 0;
  l_k_a_update = 0;

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
    unsigned int l_vcompute = 0;
    unsigned int l_register_offset = 0;

    if ( i_k_blocking != 9 ) {
      if ( (l_k > 0) && ((l_k%128) == 0) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, 128*i_micro_kernel_config->datatype_size );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_2, 128*i_micro_kernel_config->datatype_size );
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, 128*i_micro_kernel_config->datatype_size );

        l_displacement_k_b = 0;
        l_k_b_updates++;
      }
    } else {
      l_displacement_k_b = 0;
      l_k_b_updates = 0;
    }

    if ( l_k == 0 ) {
      /* load A, zmm0 + 1 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    0,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    1,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }

      /* handle prefetch */


    } else if ( l_k == 1 ) {
      /* load A, zmm2 + 3 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    2,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    3,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }
    } else if ( l_k == 2 ) {
      /* load A, zmm4 + 5 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    4,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    5,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }
    } else if ( l_k == 3 ) {
      /* load A, zmm6 + 7 */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    6,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    7,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }
    } else if ( l_k < (i_k_blocking - 4) ) {
      /* pipelined load of A, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    (l_k+4)%8,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_displacement_k_a) * i_micro_kernel_config->datatype_size) + 64 );
      }
#endif
      /* handle large displacements */
      if ( ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) >= 8192 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size) );
        l_k_a_update += ((l_displacement_k_a+1)*i_xgemm_desc->lda*i_micro_kernel_config->datatype_size);
        l_displacement_k_a = 0;
      } else {
        l_displacement_k_a++;
      }
    }

    /* current A prefetch, next 8 rows for the current column */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                        i_micro_kernel_config->prefetch_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) + 64 - l_k_a_update );
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) && (i_k_blocking != (unsigned int)i_xgemm_desc->k) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a_prefetch,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a_prefetch,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( l_k == (i_k_blocking - 1) && (i_k_blocking != (unsigned int)i_xgemm_desc->k) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    /* defning the compute routine */
    l_vcompute = i_micro_kernel_config->vmul_instruction;
    l_register_offset = l_n_blocking;

    if ( i_k_blocking != 9 ) {
      if (l_k == 1) {
        if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
          l_vcompute = LIBXSMM_X86_INSTR_VMULPD;
        } else {
          l_vcompute = LIBXSMM_X86_INSTR_VMULPS;
        }
      }
      l_register_offset = (l_n_blocking*((l_k%2)+1));
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    /* we just use displacements for very small GEMMS to save GPR instructions */
    if ( i_k_blocking == 9 ) {
      for ( l_n = 0; l_n < 9; l_n++) {
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               1,
                                               i_gp_reg_mapping->gp_reg_b,
                                               LIBXSMM_X86_GP_REG_UNDEF,
                                               0,
                                               (l_k * i_micro_kernel_config->datatype_size)+(i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                               i_micro_kernel_config->vector_name,
                                               l_k%8,
                                               i_micro_kernel_config->vector_reg_count - 9 + l_n );
      }
    } else {
      /* l_n = 0 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 0 );
      /* l_n = 1 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           i_gp_reg_mapping->gp_reg_help_0,
                                           1,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 1 );
      /* l_n = 2 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           i_gp_reg_mapping->gp_reg_help_0,
                                           2,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 2 );
      /* l_n = 3 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_help_1,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 3 );
      /* l_n = 4 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           i_gp_reg_mapping->gp_reg_help_0,
                                           4,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 4 );
      /* l_n = 5 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_help_1,
                                           i_gp_reg_mapping->gp_reg_help_0,
                                           2,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 5 );
      /* l_n = 6 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_help_2,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 6 );
      /* l_n = 7 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_help_1,
                                           i_gp_reg_mapping->gp_reg_help_0,
                                           4,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 7 );
      /* l_n = 8 */
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           l_vcompute,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           i_gp_reg_mapping->gp_reg_help_0,
                                           8,
                                           l_displacement_k_b*i_micro_kernel_config->datatype_size,
                                           i_micro_kernel_config->vector_name,
                                           l_k%8,
                                           i_micro_kernel_config->vector_reg_count - l_register_offset + 8 );

      l_displacement_k_b++;
    }
  }

  if (l_k_b_updates > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b,
                                     128*(i_micro_kernel_config->datatype_size)*l_k_b_updates );
  }

  if (l_k_a_update > 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
                                     l_k_a_update );
  }

  /* add C buffers */
  if ( i_k_blocking != 9 ) {
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vadd_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (l_n_blocking*2) + l_n,
                                           i_micro_kernel_config->vector_reg_count - l_n_blocking + l_n,
                                           i_micro_kernel_config->vector_reg_count - l_n_blocking + l_n );
    }
  }

  /* advance pointers of B only when we are not fully unrolling K*/
  if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_b,
                                 i_k_blocking * i_micro_kernel_config->datatype_size );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
unsigned int libxsmm_generator_gemm_avx512_kernel_kloop( libxsmm_generated_code*            io_generated_code,
                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const char*                        i_arch,
                                                         unsigned int                       i_n_blocking ) {
  /* l_k_blocking must be smaller than l_k_threshold */
  const unsigned int l_k_blocking = 8;
  const unsigned int l_k_threshold = 64;
  unsigned int l_k_unrolled = 0;

  LIBXSMM_UNUSED(i_arch);

  if ( (l_k_blocking >= l_k_threshold) ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_K_BLOCK );
    return 0;
  }

  /* Let's do something special for SeisSol high-order (N == 9 holds true) */
  if ((i_xgemm_desc->k >= 8) && (i_xgemm_desc->n == 9)) {
    libxsmm_generator_gemm_avx512_microkernel_k_large_n_nine( io_generated_code,
                                                               i_gp_reg_mapping,
                                                               i_micro_kernel_config,
                                                               i_xgemm_desc,
                                                               i_xgemm_desc->k );
    l_k_unrolled = 1;
  } else if ( (unsigned int)i_xgemm_desc->k <= l_k_threshold ) {
    libxsmm_generator_gemm_avx512_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                i_xgemm_desc->k);
    l_k_unrolled = 1;
  } else if ( (i_xgemm_desc->k % l_k_blocking == 0) && (l_k_threshold < (unsigned int)i_xgemm_desc->k) ) {
    libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                          i_micro_kernel_config->vector_length, l_k_blocking);

    libxsmm_generator_gemm_avx512_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                l_k_blocking);

    libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                          i_xgemm_desc, i_micro_kernel_config->vector_length, i_xgemm_desc->k, 1 );
  } else {
    unsigned int l_max_blocked_k = (i_xgemm_desc->k/l_k_blocking)*l_k_blocking;
    if (l_max_blocked_k > 0 ) {
      libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                            i_micro_kernel_config->vector_length, l_k_blocking);

      libxsmm_generator_gemm_avx512_microkernel( io_generated_code,
                                                  i_gp_reg_mapping,
                                                  i_micro_kernel_config,
                                                  i_xgemm_desc,
                                                  i_n_blocking,
                                                  l_k_blocking);

      libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                          i_xgemm_desc, i_micro_kernel_config->vector_length, l_max_blocked_k, 0 );
    }

    /* let's handle the remainder */
    libxsmm_generator_gemm_avx512_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                i_xgemm_desc->k-l_max_blocked_k);
    /* reset B manually */
    if (l_max_blocked_k > 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_sub_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       i_xgemm_desc->k * i_micro_kernel_config->datatype_size );
    }
  }

  return l_k_unrolled;
}

