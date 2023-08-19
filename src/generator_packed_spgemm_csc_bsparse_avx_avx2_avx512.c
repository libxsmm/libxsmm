/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_packed_spgemm_csc_bsparse_avx_avx2_avx512.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_common_x86.h"


LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_bsparse_avx_avx2_avx512( libxsmm_generated_code*         io_generated_code,
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
  if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )  ) {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_simd_packed_width = 8;
      l_max_reg_block = 28;
    } else {
      l_simd_packed_width = 4;
      l_max_reg_block = 14;
    }
  } else {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_simd_packed_width = 16;
      l_max_reg_block = 28;
    } else {
      l_simd_packed_width = 8;
      l_max_reg_block = 14;
    }
  }
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters_full = i_packed_width/l_simd_packed_width;
  l_simd_packed_iters = ( l_simd_packed_remainder > 0 ) ? l_simd_packed_iters_full+1 : l_simd_packed_iters_full;

  /* get max column in C */
  l_max_cols = i_xgemm_desc->n;
  for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++ ) {
    if ( i_column_idx[l_n] == i_column_idx[i_xgemm_desc->n] ) {
      l_max_cols = l_n+1;
    }
  }

  /* when we have remainder on lower than AVX512 we need one spare register for a mask */
  if ( ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) && ( l_simd_packed_remainder != 0 ) ) {
    l_max_reg_block = 13;
  }

#if 0
  printf("packed parameters: %u, %u, %u, %u, %u\n", i_packed_width, l_simd_packed_remainder, l_simd_packed_iters, l_simd_packed_iters_full, l_simd_packed_width );
#endif
  /* packed blocking */
  /* TODO: for 2^x for l_simd_packed iters we might want to do something else */
  libxsmm_compute_equalized_blocking( l_simd_packed_iters, l_max_reg_block, &(l_packed_reg_range[0]), &(l_packed_reg_block[0]), &(l_packed_reg_range[1]), &(l_packed_reg_block[1]) );
#if 0
  printf("packed blocking (range0, block0, range1, block1): %u %u %u %u\n", l_packed_reg_range[0], l_packed_reg_block[0], l_packed_reg_range[1], l_packed_reg_block[1]);
#endif

  /* adjust max reg_blocking to allow for 2d blocking */
  if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    if ( l_packed_reg_block[0] == 2 ) {
      l_max_reg_block = 20;
    }
    if ( l_packed_reg_block[0] == 4 ) {
      l_max_reg_block = 24;
    }
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
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* open asm */
  libxsmm_x86_instruction_open_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1 );
    /* A pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 32, l_gp_reg_mapping.gp_reg_a, 0 );
    /* B pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 64, l_gp_reg_mapping.gp_reg_b, 0 );
    /* C pointer */
    libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 96, l_gp_reg_mapping.gp_reg_c, 0 );
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 56, l_gp_reg_mapping.gp_reg_a_prefetch, 0 );
      /* B prefetch pointer */
      libxsmm_x86_instruction_alu_mem( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 88, l_gp_reg_mapping.gp_reg_b_prefetch, 0 );
    }
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

  /* m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_mloop, 0 );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_mloop, 1 );

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
        libxsmm_generator_packed_spgemm_csc_bsparse_avx_avx2_avx512_kloop( io_generated_code,
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
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                   (long long)l_micro_kernel_config.datatype_size_out*i_packed_width*i_xgemm_desc->ldc);

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_a,
                                   (long long)l_micro_kernel_config.datatype_size_in*i_packed_width*i_xgemm_desc->lda);

  /* close m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

  /* close asm */
  libxsmm_x86_instruction_close_stream_gemm( io_generated_code, &l_gp_reg_mapping, 0, i_xgemm_desc->prefetch );
}

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_bsparse_avx_avx2_avx512_kloop( libxsmm_generated_code*            io_generated_code,
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
  unsigned int l_max_reg_block = (i_n_limit - i_n_processed) * i_packed_blocking;
  unsigned int l_n_blocking = i_n_limit - i_n_processed;
  unsigned int l_avx_mask_instr = 0;

  LIBXSMM_UNUSED(i_values);
  LIBXSMM_ASSERT( i_packed_blocking > 0 );

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, 1 );
  }

  /* load k if packed remainder is non-zero */
  if ( i_packed_remainder != 0 ) {
    /* on AVX512 we can use mask registers */
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      libxsmm_generator_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_AVX512_MASK, i_micro_kernel_config->vector_length-i_packed_remainder, (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) );
    } else {
      const char l_id = LIBXSMM_CAST_CHAR(128 > l_n_blocking ? ((int)l_n_blocking) : 127);
      unsigned char l_data[32];
      unsigned int l_count;

      if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        unsigned long long* l_i64_ptr = (unsigned long long*)l_data;
        for ( l_count = 0; l_count < 4; ++l_count ) {
          if ( l_count < i_packed_remainder ) {
            l_i64_ptr[l_count] = 0xffffffffffffffff;
          } else {
            l_i64_ptr[l_count] = 0x0;
          }
        }
        l_avx_mask_instr = LIBXSMM_X86_INSTR_VMASKMOVPD;
      } else {
        unsigned int* l_i32_ptr = (unsigned int*)l_data;
        for ( l_count = 0; l_count < 8; ++l_count ) {
          if ( l_count < i_packed_remainder ) {
            l_i32_ptr[l_count] = 0xffffffff;
          } else {
            l_i32_ptr[l_count] = 0x0;
          }
        }
        l_avx_mask_instr = LIBXSMM_X86_INSTR_VMASKMOVPS;
      }

      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, l_data, &l_id, 'y', 15 );
    }
  }

  /* load C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  i_micro_kernel_config->vxor_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  (l_n*i_packed_blocking) + l_p, (l_n*i_packed_blocking) + l_p, (l_n*i_packed_blocking) + l_p );
      } else {
        if ( (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ) {
          if ( l_avx_mask_instr > 0 ) {
            libxsmm_x86_instruction_vec_mask_move( io_generated_code,
                                                   l_avx_mask_instr,
                                                   i_gp_reg_mapping->gp_reg_c,
                                                   LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                   ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                                   ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ),
                                                   i_micro_kernel_config->vector_name,
                                                   (l_n*i_packed_blocking) + l_p, 15, 0);
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              i_micro_kernel_config->c_vmove_instruction,
                                              i_gp_reg_mapping->gp_reg_c,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                              ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ),
                                              i_micro_kernel_config->vector_name,
                                              (l_n*i_packed_blocking) + l_p, 1, 1, 0 );
          }
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->c_vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                            ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ),
                                            i_micro_kernel_config->vector_name,
                                            (l_n*i_packed_blocking) + l_p, 0, 1, 0 );
        }
      }
    }
  }

  /* do dense packed times sparse multiplication */
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++ ) {
    unsigned int l_col_k = 0;
    int l_nnz_idx[28] = { 0 };
    unsigned int l_preload_b = 0;
    unsigned int l_avx_max_reg = 0;

    /* reset helpers */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      l_nnz_idx[l_n] = -1;
    }

    /* loop over the columns of B/C */
    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      unsigned int l_col_elements = i_column_idx[i_n_processed+l_n+1] - i_column_idx[i_n_processed+l_n];
      unsigned int l_cur_column = i_column_idx[i_n_processed+l_n];
      /* search for entries matching that k */
      for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
        if ( l_k == i_row_idx[l_cur_column + l_col_k] ) {
          l_nnz_idx[l_n] = l_cur_column + l_col_k;
          l_col_k = l_col_elements;
        }
      }
    }


    l_preload_b = ( (i_packed_blocking > 1) &&
                    ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT )  ) &&
                    ( (1 + l_n_blocking + (i_packed_blocking * l_n_blocking)) < i_micro_kernel_config->vector_reg_count ) ) ? 1 : 0;
    l_avx_max_reg = ( l_avx_mask_instr > 0 ) ? 14 : 15;

    if ( l_preload_b ) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        if ( l_nnz_idx[l_n] != -1 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->b_vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_b,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_nnz_idx[l_n] * i_micro_kernel_config->datatype_size_in,
                                            i_micro_kernel_config->vector_name,
                                            l_max_reg_block + l_n + 1, 0, 1, 0 );
         }
      }
    }

    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      if ( (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ) {
        if ( l_avx_mask_instr > 0 ) {
          libxsmm_x86_instruction_vec_mask_move( io_generated_code,
                                                 l_avx_mask_instr,
                                                 i_gp_reg_mapping->gp_reg_a,
                                                 LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                 (l_k*i_packed_width*i_micro_kernel_config->datatype_size_in) +
                                                 ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in ),
                                                 i_micro_kernel_config->vector_name,
                                                 l_max_reg_block, 15, 0 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->a_vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            (l_k*i_packed_width*i_micro_kernel_config->datatype_size_in) +
                                            ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in ),
                                            i_micro_kernel_config->vector_name,
                                            l_max_reg_block, 1, 1, 0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->a_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_k*i_packed_width*i_micro_kernel_config->datatype_size_in) +
                                          ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in ),
                                          i_micro_kernel_config->vector_name,
                                          l_max_reg_block, 0, 1, 0 );
      }
      /* loop over the columns of B/C */
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        if ( l_nnz_idx[l_n] != -1 ) {
          if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
            if ( l_preload_b ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        i_micro_kernel_config->vmul_instruction,
                                                        i_micro_kernel_config->vector_name,
                                                        l_max_reg_block,
                                                        l_max_reg_block + l_n + 1,
                                                        (l_n*i_packed_blocking) + l_p );
            } else {
              libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                            i_micro_kernel_config->vmul_instruction,
                                                            i_micro_kernel_config->vector_name,
                                                            i_gp_reg_mapping->gp_reg_b,
                                                            LIBXSMM_X86_GP_REG_UNDEF,
                                                            0,
                                                            l_nnz_idx[l_n] * i_micro_kernel_config->datatype_size_in, 1,
                                                            l_max_reg_block,
                                                            (l_n*i_packed_blocking) + l_p );
            }
          } else if ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              i_micro_kernel_config->b_vmove_instruction,
                                              i_gp_reg_mapping->gp_reg_b,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              l_nnz_idx[l_n] * i_micro_kernel_config->datatype_size_in,
                                              i_micro_kernel_config->vector_name,
                                              l_avx_max_reg, 0, 1, 0 );
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vmul_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_max_reg_block,
                                                      l_avx_max_reg,
                                                      (l_n*i_packed_blocking) + l_p );
          } else if ( io_generated_code->arch == LIBXSMM_X86_AVX ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              i_micro_kernel_config->b_vmove_instruction,
                                              i_gp_reg_mapping->gp_reg_b,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              l_nnz_idx[l_n] * i_micro_kernel_config->datatype_size_in,
                                              i_micro_kernel_config->vector_name,
                                              l_avx_max_reg, 0, 1, 0 );
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vmul_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_max_reg_block,
                                                      l_avx_max_reg,
                                                      l_avx_max_reg );
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vadd_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_avx_max_reg,
                                                      (l_n*i_packed_blocking) + l_p,
                                                      (l_n*i_packed_blocking) + l_p );
          } else {
          }
        }
      }
    }
  }

  /* store C accumulator */
  for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
    for ( l_p = 0; l_p < i_packed_blocking; l_p++ ) {
      if ( (l_p == i_packed_blocking-1) && (i_packed_remainder != 0) ) {
        if ( l_avx_mask_instr > 0 ) {
          libxsmm_x86_instruction_vec_mask_move( io_generated_code,
                                                 l_avx_mask_instr,
                                                 i_gp_reg_mapping->gp_reg_c,
                                                 LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                 ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                                 ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ),
                                                 i_micro_kernel_config->vector_name,
                                                 (l_n*i_packed_blocking) + l_p, 15, 1);
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->c_vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                            ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ),
                                            i_micro_kernel_config->vector_name,
                                            (l_n*i_packed_blocking) + l_p, 1, 0, 1 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->c_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          ( (i_n_processed + l_n)*i_packed_width*i_micro_kernel_config->datatype_size_out ) +
                                          ( (i_packed_processed + l_p)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out ),
                                          i_micro_kernel_config->vector_name,
                                          (l_n*i_packed_blocking) + l_p, 0, 0, 1 );
      }
    }
  }

  /* packed loop */
  if ( i_packed_range/i_packed_blocking > 1 ) {
    /* advance A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, (long long)i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, (long long)i_packed_blocking*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );

    /* packed loop footer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, (long long)i_packed_range/i_packed_blocking );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* reset A and C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, ((long long)i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_in );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_c, ((long long)i_packed_range/i_packed_blocking)*i_micro_kernel_config->vector_length*i_micro_kernel_config->datatype_size_out );
  }
}
