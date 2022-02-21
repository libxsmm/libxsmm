/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena)
******************************************************************************/

#include "generator_packed_spgemm_csc_bsparse_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_bsparse_aarch64( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                          const unsigned int*             i_row_idx,
                                                          const unsigned int*             i_column_idx,
                                                          const void*                     i_values,
                                                          const unsigned int              i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_max_cols = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_iters_full = 0;
  unsigned int l_simd_packed_width = 0;
  unsigned int l_packed_done = 0;
  unsigned int l_packed_count = 0;
  unsigned int l_packed_reg_block[2] = {0,0};
  unsigned int l_packed_reg_range[2] = {0,0};
  unsigned int l_col_reg_block[2][2] = { {0,0}, {0,0} };
  unsigned int l_col_reg_range[2][2] = { {0,0}, {0,0} };

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    l_simd_packed_width = 2;
    l_max_reg_block = 28;
  } else {
    l_simd_packed_width = 4;
    l_max_reg_block = 28;
  }
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters_full = i_packed_width/l_simd_packed_width;
  l_simd_packed_iters = ( l_simd_packed_remainder > 0 ) ? l_simd_packed_iters_full+1 : l_simd_packed_iters_full;
  if ( l_simd_packed_remainder != 0 ) {
    /* this is for now a general error */
    fprintf( stderr, "libxsmm_generator_packed_spgemm_csc_bsparse_aarch64 right now only supports multiples of SIMD length!\n" );
    exit(-1);
  }

  /* get max column in C */
  l_max_cols = i_xgemm_desc->n;
  for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++ ) {
    if ( i_column_idx[l_n] == i_column_idx[i_xgemm_desc->n] ) {
      l_max_cols = l_n+1;
    }
  }

#if 0
  printf("packed parameters: %u, %u, %u, %u, %u\n", i_packed_width, l_simd_packed_remainder, l_simd_packed_iters, l_simd_packed_iters_full, l_simd_packed_width );
#endif
 /* packed blocking  */
  /* @TODO for 2^x for l_simd_packed iters we might want to todo something else */
  libxsmm_compute_equalized_blocking( l_simd_packed_iters, l_max_reg_block, &(l_packed_reg_range[0]), &(l_packed_reg_block[0]), &(l_packed_reg_range[1]), &(l_packed_reg_block[1]) );
#if 0
  printf("packed blocking (range0, block0, range1, block1): %u %u %u %u\n", l_packed_reg_range[0], l_packed_reg_block[0], l_packed_reg_range[1], l_packed_reg_block[1]);
#endif

  /* adjust max reg_blocking to allow for 2d blocking */
  if ( l_packed_reg_block[0] == 2 ) {
    l_max_reg_block = 20;
  }
  if ( l_packed_reg_block[0] == 4 ) {
    l_max_reg_block = 24;
  }

  /* N blocking for packed blocking */
  libxsmm_compute_equalized_blocking( l_max_cols, l_max_reg_block/l_packed_reg_block[0], &(l_col_reg_range[0][0]), &(l_col_reg_block[0][0]), &(l_col_reg_range[0][1]), &(l_col_reg_block[0][1]) );
  if ( l_packed_reg_block[1] != 0 ) {
    libxsmm_compute_equalized_blocking( l_max_cols, l_max_reg_block/l_packed_reg_block[1], &(l_col_reg_range[1][0]), &(l_col_reg_block[1][0]), &(l_col_reg_range[1][1]), &(l_col_reg_block[1][1]) );
  }
#if 0
  printf("n blocking 0    (range0, block0, range1, block1): %u %u %u %u\n",  l_col_reg_range[0][0],  l_col_reg_block[0][0],  l_col_reg_range[0][1],  l_col_reg_block[0][1]);
  printf("n blocking 1    (range0, block0, range1, block1): %u %u %u %u\n",  l_col_reg_range[1][0],  l_col_reg_block[1][0],  l_col_reg_range[1][1],  l_col_reg_block[1][1]);
#endif

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  /*l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_AARCH64_GP_REG_X5;*/
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X12;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the strcut, so lets first move this one into R15 */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* A pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mapping.gp_reg_a );
    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B preftech pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

  /* m loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );

  /* loop over packed blocks */
  while ( l_packed_done != l_simd_packed_iters ) {
    unsigned int l_packed_blocking = l_packed_reg_block[l_packed_count];
    unsigned int l_packed_remainder = 0;
    unsigned int l_n_done = 0;
    unsigned int l_n_count = 0;
    unsigned int l_n_processed = 0;

    if ( (l_simd_packed_remainder != 0) && (l_packed_count == 0) ) {
      if ( l_packed_reg_block[1] > 0 ) {
        l_packed_remainder = 0;
      } else {
         l_packed_remainder = l_simd_packed_remainder;
      }
    } else if (l_simd_packed_remainder != 0) {
      l_packed_remainder = l_simd_packed_remainder;
    }

    while ( l_n_done < l_max_cols ) {
      unsigned int l_n_blocking = l_col_reg_block[l_packed_count][l_n_count];

      for ( l_n_processed = l_n_done; l_n_processed < l_n_done + l_col_reg_range[l_packed_count][l_n_count]; l_n_processed += l_n_blocking ) {
        libxsmm_generator_packed_spgemm_csc_bsparse_aarch64_kloop( io_generated_code,
                                                                   &l_loop_label_tracker,
                                                                   &l_gp_reg_mapping,
                                                                   &l_micro_kernel_config,
                                                                   i_xgemm_desc,
                                                                   i_row_idx,
                                                                   i_column_idx,
                                                                   i_values,
                                                                   l_n_processed,
                                                                   l_n_processed + l_n_blocking,
                                                                   l_packed_done,
                                                                   l_packed_done + l_packed_reg_range[l_packed_count],
                                                                   l_packed_blocking,
                                                                   l_packed_remainder,
                                                                   i_packed_width );
      }

      l_n_done += l_col_reg_range[l_packed_count][l_n_count];
      l_n_count++;
    }

    /* advance N */
    l_packed_done += l_packed_reg_range[l_packed_count];
    l_packed_count++;
  }

  /* advance C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_2, l_gp_reg_mapping.gp_reg_c,
                                                 l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc );

  /* advance A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 l_gp_reg_mapping.gp_reg_a, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_a,
                                                 l_micro_kernel_config.datatype_size_in*i_packed_width*i_xgemm_desc->lda );

  /* close m loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_bsparse_aarch64_kloop( libxsmm_generated_code*            io_generated_code,
                                                                libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                const unsigned int*                i_row_idx,
                                                                const unsigned int*                i_column_idx,
                                                                const void*                        i_values,
                                                                const unsigned int                 i_n_processed,
                                                                const unsigned int                 i_n_limit,
                                                                const unsigned int                 i_packed_processed,
                                                                const unsigned int                 i_packed_range,
                                                                const unsigned int                 i_packed_blocking,
                                                                const unsigned int                 i_packed_remainder,
                                                                const unsigned int                 i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_p = 0;
  unsigned int l_k = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_max_reg_block = (i_n_limit - i_n_processed) * i_packed_blocking;
  unsigned int l_n_blocking = i_n_limit - i_n_processed;

  LIBXSMM_UNUSED(i_values);
  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_range/i_packed_blocking );
  }

  /* load k if packed remainder is non-zero */
  if ( i_packed_remainder != 0 ) {
    /* @TODO */
  }

  /* load C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
        unsigned int l_reg = (l_n*i_packed_blocking) + l_p;
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     l_reg, l_reg, 0, l_reg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      } else {
        unsigned int l_reg = (l_n*i_packed_blocking) + l_p;
        if ( (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ) {
          /* @TODO */
        } else {
          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                                     ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                                     ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ) );
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                  i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, 0,
                                                  l_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        }
      }
    }
  }

  /* do dense packed times sparse multiplication */
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++ ) {
    unsigned int l_col_k = 0;
    int l_nnz_idx[28][4] = { {0}, {0} };

    /* reset helpers */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      l_nnz_idx[l_n][0] = -1; l_nnz_idx[l_n][1] = -1; l_nnz_idx[l_n][2] = -1; l_nnz_idx[l_n][3] = -1;
    }
    l_found_mul = 0;

    /* loop over the columns of B/C */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      unsigned int l_col_elements = i_column_idx[i_n_processed+l_n+1] - i_column_idx[i_n_processed+l_n];
      unsigned int l_cur_column = i_column_idx[i_n_processed+l_n];
      /* search for entries matching that k */
      for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
        if ( l_k == i_row_idx[l_cur_column + l_col_k] ) {
          l_nnz_idx[l_n][0] = l_cur_column + l_col_k;
          l_col_k = l_col_elements;
        }
      }
      /* let's check if we have an entry in the column that matches the k from A */
      if ( (l_nnz_idx[l_n][0] != -1) ) {
        l_found_mul = 1;
      }
    }

    if ( l_found_mul != 0 ) {
      for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
        if ( (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ) {
          /* @TODO */
        } else {
          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                     (l_k*i_packed_width*i_micro_kernel_config->datatype_size_in) +
                                                     ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in ) );
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                  i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, 0,
                                                  l_max_reg_block, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        }
        /* loop over the columns of B/C */
        for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
          if ( l_nnz_idx[l_n][0] != -1 ) {
            libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_1,
                                                       l_nnz_idx[l_n][0] * i_micro_kernel_config->datatype_size_in );
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                    i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, 0,
                                                    l_max_reg_block+1, (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_WIDTH_S : LIBXSMM_AARCH64_ASIMD_WIDTH_D );
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V,
                                                       l_max_reg_block, l_max_reg_block+1, 0, (l_n*i_packed_blocking) + l_p,
                                                       (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
          }
        }
      }
    } else {
      /* shouldn't happen */
    }
  }

  /* store C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      if ( (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ) {
        /* @TODO */
      } else {
        unsigned int l_reg = (l_n*i_packed_blocking) + l_p;
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_mapping->gp_reg_help_2,
                                                   ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                                   ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ) );
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_R,
                                                i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, 0,
                                                l_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
    }
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, 1 );

    /* reset A and C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );
  }
}

