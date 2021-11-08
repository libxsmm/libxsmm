/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena)
******************************************************************************/

#include "generator_gemm_power.h"
#include "generator_power_instructions.h"

LIBXSMM_API_INTERN
unsigned char libxsmm_generator_gemm_power_load_store_vsx( libxsmm_generated_code * io_generated_code,
                                                           unsigned int             i_m_blocking_full,
                                                           unsigned int             i_n_blocking,
                                                           unsigned int             i_remainder_size,
                                                           unsigned int             i_stride,
                                                           unsigned char            i_load_store,
                                                           unsigned char            i_precision,
                                                           unsigned char            i_endianness,
                                                           unsigned char            i_gpr_ptr,
                                                           unsigned char          * i_gpr_scratch,
                                                           unsigned char            i_vsr_first ) {
  unsigned int l_vsr = i_vsr_first;

  unsigned int l_ops_vsx[3] = {0};
  if( i_load_store == 0 ) {
    l_ops_vsx[0] = LIBXSMM_POWER_INSTR_VSX_LXVW4X;
    l_ops_vsx[1] = LIBXSMM_POWER_INSTR_VSX_LXVLL;
    l_ops_vsx[2] = LIBXSMM_POWER_INSTR_VSX_XXBRW;
  }
  else {
    l_ops_vsx[0] = LIBXSMM_POWER_INSTR_VSX_STXVW4X;
    l_ops_vsx[1] = LIBXSMM_POWER_INSTR_VSX_STXVLL;
    l_ops_vsx[2] = LIBXSMM_POWER_INSTR_VSX_XXBRW;
  }

  libxsmm_power_instruction_3( io_generated_code,
                               LIBXSMM_POWER_INSTR_FIP_ORI,
                               i_gpr_scratch[0],
                               i_gpr_ptr,
                               0 );

  if( i_remainder_size > 0 ) {
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 i_gpr_scratch[1],
                                 0,
                                 i_remainder_size );
    libxsmm_power_instruction_4( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_RLDICR,
                                 i_gpr_scratch[1],
                                 i_gpr_scratch[1],
                                 64-8,
                                 63 );
  }

  for( unsigned int l_bn = 0; l_bn < i_n_blocking; l_bn++ ) {
    /* prep the GPRs for the storage accesses */
    unsigned int l_n_ops = i_m_blocking_full;
    if( i_remainder_size > 0 ) {
      l_n_ops++;
    }
    for( unsigned int l_gp = 0; l_gp < l_n_ops; l_gp++ ) {
      libxsmm_power_instruction_3( io_generated_code,
                                   LIBXSMM_POWER_INSTR_FIP_ADDI,
                                   i_gpr_scratch[2+l_gp],
                                   i_gpr_scratch[0],
                                   16*l_gp );
    }

    if( l_bn != i_n_blocking - 1 ) {
      libxsmm_power_instruction_3( io_generated_code,
                                  LIBXSMM_POWER_INSTR_FIP_ADDI,
                                  i_gpr_scratch[0],
                                  i_gpr_scratch[0],
                                  i_stride );
    }

    /* 128bit loads/stores */
    for( unsigned int l_bm = 0; l_bm < i_m_blocking_full; l_bm++ ) {
      libxsmm_power_instruction_3( io_generated_code,
                                   l_ops_vsx[0],
                                   l_vsr,
                                   0,
                                   i_gpr_scratch[2+l_bm] );
      l_vsr++;
    }

    /* remainder load/store (if required) */
    if( i_remainder_size > 0 ) {
      /* reverse byte-order before storing for LE */
      if( i_endianness == 0 &&
          i_load_store == 1 ) {
        libxsmm_power_instruction_2( io_generated_code,
                                     l_ops_vsx[2],
                                     l_vsr,
                                     l_vsr );
      }

      libxsmm_power_instruction_3( io_generated_code,
                                   l_ops_vsx[1],
                                   l_vsr,
                                   i_gpr_scratch[2+i_m_blocking_full],
                                   i_gpr_scratch[1] );

      /* reverse byte-order after loading for LE */
      if( i_endianness == 0 &&
          i_load_store == 0 ) {
        libxsmm_power_instruction_2( io_generated_code,
                                     l_ops_vsx[2],
                                     l_vsr,
                                     l_vsr );
      }
      l_vsr++;
    }
  }

  return l_vsr - i_vsr_first;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_power_microkernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                   libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                   unsigned char                   i_gpr_a,
                                                   unsigned char                   i_gpr_b,
                                                   unsigned char                   i_gpr_c,
                                                   unsigned int                    i_m_blocking,
                                                   unsigned int                    i_n_blocking,
                                                   unsigned int                    i_k_blocking ) {
  unsigned int l_bytes_per_val = 4;
  unsigned int l_vector_length = 4;

  unsigned char l_gpr_a = LIBXSMM_POWER_GPR_R21;
  unsigned char l_gpr_b = LIBXSMM_POWER_GPR_R22;
  unsigned char l_gpr_k = 20;

  unsigned char l_gpr_scratch[14] = {  6,  7,  8,  9,
                                      10, 11, 12, 13,
                                      14, 15, 16, 17,
                                      18, 19 };

  unsigned int l_stride_a = i_xgemm_desc->lda * l_bytes_per_val;
  unsigned int l_stride_b = i_xgemm_desc->ldb * l_bytes_per_val;
  unsigned int l_stride_c = i_xgemm_desc->ldc * l_bytes_per_val;

  /* derive blocking in terms of 128bit and remainder ops */
  unsigned int l_m_blocking_full = i_m_blocking / l_vector_length;
  unsigned int l_remainder_size = i_m_blocking % l_vector_length;
               l_remainder_size *= l_bytes_per_val;

  /* derive k-unrolling */
  unsigned short l_k_unroll = 1;
  if( i_k_blocking%4 == 0 ) {
    l_k_unroll = 4;
  }

  /* loop labels */
  libxsmm_loop_label_tracker l_loop_labels;

  /* locally store addresses of A and C */
  libxsmm_power_instruction_3( io_generated_code,
                               LIBXSMM_POWER_INSTR_FIP_ORI,
                               l_gpr_a,
                               i_gpr_a,
                               0 );

  libxsmm_power_instruction_3( io_generated_code,
                               LIBXSMM_POWER_INSTR_FIP_ORI,
                               l_gpr_b,
                               i_gpr_b,
                               0 );

  /* load accumulator block */
  unsigned char l_vsr_c_first = 0;
  unsigned char l_n_vsr_c = 0;
  l_n_vsr_c = libxsmm_generator_gemm_power_load_store_vsx( io_generated_code,
                                                           l_m_blocking_full,
                                                           i_n_blocking,
                                                           l_remainder_size,
                                                           l_stride_c,
                                                           0,
                                                           l_bytes_per_val == 4 ? 0 : 1,
                                                           0,
                                                           i_gpr_c,
                                                           l_gpr_scratch,
                                                           l_vsr_c_first );

  unsigned char l_vsr_a_first = l_vsr_c_first + l_n_vsr_c;

  /* iterate over k if necessary */
  if( l_k_unroll != i_k_blocking ) {
    libxsmm_reset_loop_label_tracker( &l_loop_labels );
    libxsmm_power_instruction_3( io_generated_code,
                                LIBXSMM_POWER_INSTR_FIP_ADDI,
                                l_gpr_k,
                                0,
                                i_k_blocking / l_k_unroll );
    /* use count register for inner-loop */
    libxsmm_power_instruction_2( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_MTSPR,
                                 l_gpr_k,
                                 LIBXSMM_POWER_SPR_CTR );
    libxsmm_power_instruction_register_jump_back_label( io_generated_code,
                                                        &l_loop_labels );
  }

  /* unrolled k */
  for( unsigned short l_k = 0; l_k < l_k_unroll; l_k++ ) {
    /* load (partial) column of A */
    unsigned int l_n_vsr_a = 0;
    l_n_vsr_a = libxsmm_generator_gemm_power_load_store_vsx( io_generated_code,
                                                             l_m_blocking_full,
                                                             1,
                                                             l_remainder_size,
                                                             l_stride_a,
                                                             0,
                                                             l_bytes_per_val == 4 ? 0 : 1,
                                                             0,
                                                             l_gpr_a,
                                                             l_gpr_scratch,
                                                             l_vsr_a_first );

    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_a,
                                 l_gpr_a,
                                 l_stride_a );

    unsigned char l_vsr_b = l_vsr_a_first + l_n_vsr_a;

    /* prepare the GPRs for the loads of B */
    for( unsigned short l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_power_instruction_3( io_generated_code,
                                   LIBXSMM_POWER_INSTR_FIP_ADDI,
                                   l_gpr_scratch[l_n],
                                   l_gpr_b,
                                   l_stride_b*l_n );
    }
    if( l_k != i_k_blocking-1 ) {
      libxsmm_power_instruction_3( io_generated_code,
                                   LIBXSMM_POWER_INSTR_FIP_ADDI,
                                   l_gpr_b,
                                   l_gpr_b,
                                   l_bytes_per_val );
    }

    for( unsigned short l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* bcast entry of B */
      libxsmm_power_instruction_3( io_generated_code,
                                   LIBXSMM_POWER_INSTR_VSX_LXVWSX,
                                   l_vsr_b,
                                   0,
                                   l_gpr_scratch[l_n] );

      /* perform FMAs */
      unsigned int l_n_ops = l_m_blocking_full;
      if( l_remainder_size > 1 ) {
        l_n_ops++;
      }
      for( unsigned short l_op = 0; l_op < l_n_ops; l_op++ ) {
        unsigned int l_vsr_c_off = l_n * l_n_ops;
                     l_vsr_c_off += l_op;

        libxsmm_power_instruction_3( io_generated_code,
                                     LIBXSMM_POWER_INSTR_VSX_XVMADDASP,
                                     l_vsr_c_first + l_vsr_c_off,
                                     l_vsr_a_first + l_op,
                                     l_vsr_b );
      }
    }
  }

  /* end of k-loop */
  if( l_k_unroll != i_k_blocking ) {
    libxsmm_power_instruction_cond_jump_back_to_label_ctr( io_generated_code,
                                                           &l_loop_labels );
  }

  /* store accumulator block */
  l_n_vsr_c = libxsmm_generator_gemm_power_load_store_vsx( io_generated_code,
                                                           l_m_blocking_full,
                                                           i_n_blocking,
                                                           l_remainder_size,
                                                           l_stride_c,
                                                           1,
                                                           l_bytes_per_val == 4 ? 0 : 1,
                                                           0,
                                                           i_gpr_c,
                                                           l_gpr_scratch,
                                                           l_vsr_c_first );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_power_m_loop_vsx( libxsmm_generated_code        * io_generated_code,
                                              libxsmm_gemm_descriptor const * i_xgemm_desc,
                                              unsigned int                    i_bytes_per_val,
                                              unsigned char                   i_gpr_a,
                                              unsigned char                   i_gpr_b,
                                              unsigned char                   i_gpr_c,
                                              unsigned char                   i_gpr_scratch,
                                              unsigned int                    i_max_block_m,
                                              unsigned int                    i_n ) {
  unsigned short l_gpr_m = i_gpr_scratch;
  unsigned int l_n_max_blocks_m = i_xgemm_desc->m / i_max_block_m;

  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* loop over maximum-sized m-blocks if necessary */
  if( l_n_max_blocks_m > 1 ) {
    libxsmm_power_instruction_3( io_generated_code,
                                LIBXSMM_POWER_INSTR_FIP_ADDI,
                                l_gpr_m,
                                0,
                                l_n_max_blocks_m );
    libxsmm_power_instruction_register_jump_back_label( io_generated_code,
                                                        &l_loop_labels );
  }

  if( l_n_max_blocks_m > 0 ) {
    /* micro-kernel for maximum sizes block w.r.t. m */
    libxsmm_generator_gemm_power_microkernel_vsx( io_generated_code,
                                                  i_xgemm_desc,
                                                  i_gpr_a,
                                                  i_gpr_b,
                                                  i_gpr_c,
                                                  i_max_block_m,
                                                  i_n,
                                                  i_xgemm_desc->k );

    /* increase A and B pointers */
    unsigned int l_n_bytes_per_block_m = i_max_block_m*i_bytes_per_val;
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 i_gpr_a,
                                 i_gpr_a,
                                 l_n_bytes_per_block_m );

    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 i_gpr_c,
                                 i_gpr_c,
                                 l_n_bytes_per_block_m );
  }

  /* end of max-size m-block loop */
  if( l_n_max_blocks_m > 1 ) {
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_m,
                                 l_gpr_m,
                                 -1 );

    libxsmm_power_instruction_cond_jump_back_to_label( io_generated_code,
                                                       l_gpr_m,
                                                       &l_loop_labels );
  }

  /* generate microkernel for remainder if necessary */
  unsigned int l_m_remaining = i_xgemm_desc->m % i_max_block_m;

  if( l_m_remaining > 0 ) {
    libxsmm_generator_gemm_power_microkernel_vsx( io_generated_code,
                                                  i_xgemm_desc,
                                                  i_gpr_a,
                                                  i_gpr_b,
                                                  i_gpr_c,
                                                  l_m_remaining,
                                                  i_n,
                                                  i_xgemm_desc->k );
  }
}

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_power_kernel_vsx( libxsmm_generated_code        * io_generated_code,
                                             libxsmm_gemm_descriptor const * i_xgemm_desc ) {
  /* derive if FP32 or FP64 and respective vector length */
  unsigned int l_bytes_per_val = 0;
  unsigned int l_vector_length = 0;
  if( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 4;
    l_vector_length = 4;
  }
  else if( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 8;
    l_vector_length = 2;
  }
  else {
    return 1;
  }

  /* loop labels */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* derive maximum block-size bound by the number of available registers */
  unsigned int l_max_block_m  = i_xgemm_desc->m;
               l_max_block_m += l_vector_length-1;
               l_max_block_m /= l_vector_length;
  if( l_max_block_m > 12 ) {
    l_max_block_m = 12;
  }
  l_max_block_m *= l_vector_length;

  unsigned int l_max_block_n  = 64 - (l_max_block_m/l_vector_length) - 1;
               l_max_block_n /= (l_max_block_m/l_vector_length);
  if( l_max_block_n > i_xgemm_desc->n ) {
    l_max_block_n = i_xgemm_desc->n;
  }
  if( l_max_block_n > 12 ) {
    l_max_block_n = 12;
  }

  /* derive number of required GP-, FP-, and VS-registers */
  unsigned int l_n_vs_all = (l_max_block_m/l_vector_length) * (l_max_block_n + 1); /* acc block and A-col */
  l_n_vs_all += 1; /* B bcast-register */

  unsigned short l_gprMax = 24;
  unsigned short l_fprMax = (l_n_vs_all < 32) ? l_n_vs_all - 1 : 31;
  unsigned short l_vsrMax = (l_n_vs_all < 33) ? 0 : l_n_vs_all - 33;

  /* save registers on stack as required by ABI */
  libxsmm_power_instruction_open_stream( io_generated_code,
                                         l_gprMax,
                                         l_fprMax,
                                         l_vsrMax );

  /* function-input holding A's, B's and C's location in memory */
  unsigned char l_gpr_a = LIBXSMM_POWER_GPR_R3;
  unsigned char l_gpr_b = LIBXSMM_POWER_GPR_R4;
  unsigned char l_gpr_c = LIBXSMM_POWER_GPR_R5;

  /* GPR mapping */
  unsigned char l_gpr_scratch = 23;
  unsigned char l_gpr_n = 24;
  unsigned int l_n_max_blocks_n = i_xgemm_desc->n / l_max_block_n;

  /* loop over max-sized blocks in N-dimension */
  if( l_n_max_blocks_n > 1 ) {
    libxsmm_power_instruction_3( io_generated_code,
                                LIBXSMM_POWER_INSTR_FIP_ADDI,
                                l_gpr_n,
                                0,
                                l_n_max_blocks_n );
    libxsmm_power_instruction_register_jump_back_label( io_generated_code,
                                                        &l_loop_labels );
  }

  /* generate M-loop for max-sized block in N-dimension */
  if( l_n_max_blocks_n > 0 ) {
    libxsmm_generator_gemm_power_m_loop_vsx( io_generated_code,
                                             i_xgemm_desc,
                                             l_bytes_per_val,
                                             l_gpr_a,
                                             l_gpr_b,
                                             l_gpr_c,
                                             l_gpr_scratch,
                                             l_max_block_m,
                                             l_max_block_n );

    /* adjust matrix pointers */
    unsigned int l_off  = 0;

    /* A */
    unsigned int l_n_max_blocks_m = i_xgemm_desc->m / l_max_block_m;
    if( l_n_max_blocks_m > 0 ) {
      l_off -= l_n_max_blocks_m*l_max_block_m;
    }
    l_off *= l_bytes_per_val;
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_a,
                                 l_gpr_a,
                                 l_off );

    /* B */
    l_off = i_xgemm_desc->ldb * l_max_block_n;
    l_off *= l_bytes_per_val;
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_b,
                                 l_gpr_b,
                                 l_off );

    /* C */
    l_off = i_xgemm_desc->ldc * l_max_block_n;
    if( l_n_max_blocks_m > 0 ) {
      l_off -= l_n_max_blocks_m*l_max_block_m;
    }
    l_off *= l_bytes_per_val;
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_c,
                                 l_gpr_c,
                                 l_off );
  }

  /* end of n-loop */
  if( l_n_max_blocks_n > 1 ) {
    libxsmm_power_instruction_3( io_generated_code,
                                 LIBXSMM_POWER_INSTR_FIP_ADDI,
                                 l_gpr_n,
                                 l_gpr_n,
                                 -1 );
    libxsmm_power_instruction_cond_jump_back_to_label( io_generated_code,
                                                       l_gpr_n,
                                                       &l_loop_labels );
  }

  /* generate M-loop for remaining entries in N-dimension */
  unsigned int l_n_remaining = i_xgemm_desc->n % l_max_block_n;
  if( l_n_remaining > 0 ) {
    libxsmm_generator_gemm_power_m_loop_vsx( io_generated_code,
                                             i_xgemm_desc,
                                             l_bytes_per_val,
                                             l_gpr_a,
                                             l_gpr_b,
                                             l_gpr_c,
                                             l_gpr_scratch,
                                             l_max_block_m,
                                             l_n_remaining );
  }

  /* restore registers from stack according to ABI */
  libxsmm_power_instruction_close_stream( io_generated_code,
                                          l_gprMax,
                                          l_fprMax,
                                          l_vsrMax );
  return 0;
}

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_power_kernel( libxsmm_generated_code        * io_generated_code,
                                         libxsmm_gemm_descriptor const * i_xgemm_desc ) {
  return libxsmm_generator_gemm_power_kernel_vsx( io_generated_code,
                                                  i_xgemm_desc );
}