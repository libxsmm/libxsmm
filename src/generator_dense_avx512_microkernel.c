/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
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
#include "generator_dense_avx512_microkernel.h"
#include "generator_dense_instructions.h"
#include <libxsmm_macros.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void libxsmm_generator_dense_avx512_microkernel( libxsmm_generated_code*             io_generated_code,
                                                 const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                  i_n_blocking,
                                                 const unsigned int                  i_k_blocking,
                                                 const int                           i_offset )
{
  unsigned int l_n;
  unsigned int l_k;

#if !defined(NDEBUG)
  if ( i_n_blocking > 30 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( (i_offset >= 0) && (i_k_blocking != 1) ) {
    fprintf(stderr, "LIBXSMM WARNING, libxsmm_generator_dense_avx512_microkernel: i_k_blocking is ignored as offset is >=0\n");
  }
#endif

  /* if we have an offset greater-equal -> external k-unrolling */
  if (i_offset != (-1)) {
    /* load A */
    libxsmm_instruction_vec_move( io_generated_code,
                                  i_micro_kernel_config->instruction_set,
                                  i_micro_kernel_config->a_vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_a,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size,
                                  i_micro_kernel_config->vector_name,
                                  0,
                                  i_micro_kernel_config->use_masking_a_c, 0 );

    /* current A prefetch, next 8 rows for the current column */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size) + 64 );
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size) );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vmul_instruction,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           (i_offset * i_micro_kernel_config->datatype_size) + (i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                           i_micro_kernel_config->vector_name,
                                           0,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
    }
  } else {
    /* apply k blocking */
    for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
      if ( l_k == 0 ) {
        /* load A */
        libxsmm_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size,
                                      i_micro_kernel_config->vector_name,
                                      0,
                                      i_micro_kernel_config->use_masking_a_c, 0 );
        if ( i_k_blocking > 1 ) {
          /* second A load in first iteration, in case of large blockings -> hiding L1 latencies */
          libxsmm_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        1,
                                        i_micro_kernel_config->use_masking_a_c, 0 );
        }
      } else if ( l_k < (i_k_blocking - 1) ) {
        /* pipelined load of A, one k iteration ahead */
        libxsmm_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                      i_micro_kernel_config->vector_name,
                                      (l_k+1)%2,
                                      i_micro_kernel_config->use_masking_a_c, 0 );
      }

      /* current A prefetch, next 8 rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_instruction_prefetch( io_generated_code,
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
        libxsmm_instruction_prefetch( io_generated_code,
                                      i_micro_kernel_config->prefetch_instruction,
                                      i_gp_reg_mapping->gp_reg_a_prefetch,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
        if ( l_k == (i_k_blocking - 1) ) {
          libxsmm_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_a_prefetch,
                                       i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
        }
      }

      /* in last k-iteration: advance pointers */
      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }

      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        libxsmm_instruction_vec_compute_mem( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             i_micro_kernel_config->vmul_instruction,
                                             1,
                                             i_gp_reg_mapping->gp_reg_b,
                                             LIBXSMM_X86_GP_REG_UNDEF,
                                             0,
                                             (l_k * i_micro_kernel_config->datatype_size)+(i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                             i_micro_kernel_config->vector_name,
                                             l_k%2,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
      }
    }

    /* advance pointers of B only when we are not fully unrolling K*/
    if ( i_k_blocking < i_xgemm_desc->k ) {
      /* advance pointers of B */
      libxsmm_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   i_k_blocking * i_micro_kernel_config->datatype_size );
    }
  }
}

#if 0
void libxsmm_generator_dense_avx512_microkernel_k_large( libxsmm_generated_code*             io_generated_code,
                                                         const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const unsigned int                  i_n_blocking,
                                                         const unsigned int                  i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;

#if !defined(NDEBUG)
  if ( i_n_blocking > 24 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx512_microkernel_k_large: i_n_blocking needs to be 24 or smaller!\n");
    exit(-1);
  }
  if ( i_k_blocking < 8 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx512_microkernel_k_large: i_k_blocking needs to be at least 8!\n");
    exit(-1);
  }
#endif

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
    if ( l_k == 0 ) {
      /* load A, zmm0 + 1 */
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+0) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    0,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    2,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+2) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    4,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+3) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    6,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_prefetch( io_generated_code,
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
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a_prefetch,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( l_k == (i_k_blocking - 1) ) {
      libxsmm_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_instruction_vec_compute_mem( io_generated_code,
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
    libxsmm_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_b,
                                 i_k_blocking * i_micro_kernel_config->datatype_size );
  }
}
#endif

void libxsmm_generator_dense_avx512_microkernel_k_large_n_nine( libxsmm_generated_code*             io_generated_code,
                                                                const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                const unsigned int                  i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;
  const unsigned int l_n_blocking = 9;
  unsigned int l_displacement_k = 0;
  unsigned int l_k_updates = 0;

#if !defined(NDEBUG)
  if ( i_k_blocking < 8 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx512_microkernel_k_large_n_nine: i_k_blocking needs to be at least 8!\n");
    exit(-1);
  }
#endif

  /* Intialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                               i_gp_reg_mapping->gp_reg_help_0, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb );
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                               i_gp_reg_mapping->gp_reg_help_1, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                               i_gp_reg_mapping->gp_reg_help_2, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
                               i_gp_reg_mapping->gp_reg_help_3, i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb * 7 );

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
    unsigned int l_vcompute = 0;
    unsigned int l_register_offset = 0;

    if ( (l_k > 0) && (l_k%(128/i_micro_kernel_config->datatype_size) == 0) ) {
      libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, 128 );

      l_displacement_k = 0;
      l_k_updates++;
    }

    if ( l_k == 0 ) {
      /* load A, zmm0 + 1 */
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+0) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    0,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    2,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+2) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    4,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    i_micro_kernel_config->a_vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_xgemm_desc->lda * (l_k+3) * i_micro_kernel_config->datatype_size,
                                    i_micro_kernel_config->vector_name,
                                    6,
                                    i_micro_kernel_config->use_masking_a_c, 0 );
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_vec_move( io_generated_code,
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
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) + 64 );
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) && (i_k_blocking != i_xgemm_desc->k) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a_prefetch,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a_prefetch,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( l_k == (i_k_blocking - 1) && (i_k_blocking != i_xgemm_desc->k) ) {
      libxsmm_instruction_alu_imm( io_generated_code,
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

    /* l_n = 0 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         LIBXSMM_X86_GP_REG_UNDEF,
                                         0,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 0 );
    /* l_n = 1 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_0,
                                         1,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 1 );
    /* l_n = 2 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_0,
                                         2,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 2 );
    /* l_n = 3 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_1,
                                         1,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 3 );
    /* l_n = 4 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_0,
                                         4,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 4 );
    /* l_n = 5 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_2,
                                         1,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 5 );
    /* l_n = 6 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_1,
                                         2,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 6 );
    /* l_n = 7 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_3,
                                         1,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 7 );
    /* l_n = 8 */
    libxsmm_instruction_vec_compute_mem( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         l_vcompute,
                                         1,
                                         i_gp_reg_mapping->gp_reg_b,
                                         i_gp_reg_mapping->gp_reg_help_0,
                                         8,
                                         l_displacement_k*i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         l_k%8,
                                         i_micro_kernel_config->vector_reg_count - l_register_offset + 8 );

    l_displacement_k++;
  }

  if (l_k_updates > 0) {
    libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b, 128*l_k_updates );
  }

  /* add C buffers */
  if ( i_k_blocking != 9 ) {
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      libxsmm_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vadd_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (l_n_blocking*2) + l_n,
                                           i_micro_kernel_config->vector_reg_count - l_n_blocking + l_n,
                                           i_micro_kernel_config->vector_reg_count - l_n_blocking + l_n );
    }
  }

  /* advance pointers of B only when we are not fully unrolling K*/
  if ( i_k_blocking < i_xgemm_desc->k ) {
    libxsmm_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_b,
                                 i_k_blocking * i_micro_kernel_config->datatype_size );
  }
}

unsigned int libxsmm_generator_dense_avx512_kernel_kloop( libxsmm_generated_code*            io_generated_code,
                                                          libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_gemm_descriptor*    i_xgemm_desc,
                                                          const char*                        i_arch,
                                                          unsigned int                       i_n_blocking ) {
  const unsigned int l_k_blocking = 8;
  const unsigned int l_k_threshold = 8;
  unsigned int l_k_unrolled = 0;

  LIBXSMM_UNUSED(i_arch);

  /* Let's do something special for SeisSol high-order (N == 9 holds true) */
  if ((i_xgemm_desc->k >= 8) && (i_xgemm_desc->n == 9)) {
    libxsmm_generator_dense_avx512_microkernel_k_large_n_nine( io_generated_code,
                                                               i_gp_reg_mapping,
                                                               i_micro_kernel_config,
                                                               i_xgemm_desc,
                                                               i_xgemm_desc->k );
    l_k_unrolled = 1;
  } else if ( (i_xgemm_desc->k % l_k_blocking == 0) && (i_xgemm_desc->k >= l_k_threshold) ) {
    if ( i_xgemm_desc->k != l_k_blocking ) {
      libxsmm_generator_dense_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                            i_micro_kernel_config->vector_length, l_k_blocking);
    }

    libxsmm_generator_dense_avx512_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                l_k_blocking,
                                                -1 );
    if ( i_xgemm_desc->k != l_k_blocking ) {
      libxsmm_generator_dense_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                            i_xgemm_desc, i_micro_kernel_config->vector_length, i_xgemm_desc->k, 1 );
    }
  } else {
    unsigned int l_max_blocked_k = (i_xgemm_desc->k/l_k_blocking)*l_k_blocking;
    unsigned int l_k;
    if (l_max_blocked_k > 0 ) {
      libxsmm_generator_dense_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                            i_micro_kernel_config->vector_length, l_k_blocking);

      libxsmm_generator_dense_avx512_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                l_k_blocking,
                                                -1 );

      libxsmm_generator_dense_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                          i_xgemm_desc, i_micro_kernel_config->vector_length, l_max_blocked_k, 0 );
    }
    for ( l_k = l_max_blocked_k; l_k < i_xgemm_desc->k; l_k++) {
      libxsmm_generator_dense_avx512_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                1,
                                                l_k-l_max_blocked_k );
    }
    /* update A, B and a_prefetch pointers */
    libxsmm_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_a,
                                 (i_xgemm_desc->k - l_max_blocked_k) * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (i_xgemm_desc->k - l_max_blocked_k) * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }
    /* reset on B is just needed when we had more than iterations left */
    if (l_max_blocked_k > 0 ) {
      libxsmm_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   l_max_blocked_k * i_micro_kernel_config->datatype_size );
    }
  }

  return l_k_unrolled;
}

