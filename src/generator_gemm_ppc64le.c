/******************************************************************************
* Copyright (c) 2021, Friedrich Schiller University Jena                      *
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_gemm_ppc64le.h"


unsigned char libxsmm_generator_gemm_ppc64le_load_vsx( libxsmm_generated_code * io_generated_code,
                                                       unsigned int             i_m_blocking_full,
                                                       unsigned int             i_n_blocking,
                                                       unsigned int             i_remainder_size,
                                                       unsigned int             i_stride,
                                                       unsigned char            i_precision,
                                                       unsigned char            i_gpr_ptr,
                                                       unsigned char          * i_gpr_scratch,
                                                       unsigned char            i_vsr_first ) {
  unsigned int l_vsr = i_vsr_first;

  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ORI,
                           i_gpr_scratch[0],
                           i_gpr_ptr,
                           0 );

  if( i_remainder_size > 0 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_gpr_scratch[1],
                             0,
                             i_remainder_size );
    libxsmm_ppc64le_instr_4( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_RLDICR,
                             i_gpr_scratch[1],
                             i_gpr_scratch[1],
                             64 - 8,
                             63 );
  }

  for( unsigned int l_bn = 0; l_bn < i_n_blocking; l_bn++ ) {
    /* prep the GPRs for the storage accesses */
    unsigned int l_n_ops = i_m_blocking_full;
    if( i_remainder_size > 0 ) {
      l_n_ops++;
    }
    for( unsigned int l_gp = 0; l_gp < l_n_ops; l_gp++ ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               i_gpr_scratch[2 + l_gp],
                               i_gpr_scratch[0],
                               16*l_gp );
    }

    if( l_bn != i_n_blocking - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               i_gpr_scratch[0],
                               i_gpr_scratch[0],
                               i_stride );
    }

    /* 128bit loads/stores */
    for( unsigned int l_bm = 0; l_bm < i_m_blocking_full; l_bm++ ) {
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVW4X,
                               l_vsr,
                               0,
                               i_gpr_scratch[2 + l_bm],
                               (0x0020 & l_vsr) >> 5 );
      l_vsr++;
    }

    /* remainder load (if required) */
    if( i_remainder_size > 0 ) {
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVLL,
                               l_vsr,
                               i_gpr_scratch[2 + i_m_blocking_full],
                               i_gpr_scratch[1],
                               (0x0020 & l_vsr) >> 5 );

      /* reverse byte-order after loading for LE */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXBRW,
                               l_vsr,
                               l_vsr,
                               (0x0020 & l_vsr) >> 5,
                               (0x0020 & l_vsr) >> 5 );
      l_vsr++;
    }
  }

  return l_vsr - i_vsr_first;

}


LIBXSMM_API_INTERN
unsigned char libxsmm_generator_gemm_ppc64le_store_vsx( libxsmm_generated_code * io_generated_code,
                                                        unsigned int             i_m_blocking_full,
                                                        unsigned int             i_n_blocking,
                                                        unsigned int             i_remainder_size,
                                                        unsigned int             i_stride,
                                                        unsigned char            i_precision,
                                                        unsigned char            i_gpr_ptr,
                                                        unsigned char          * i_gpr_scratch,
                                                        unsigned char            i_vsr_first ) {
  unsigned int l_vsr = i_vsr_first;

  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ORI,
                           i_gpr_scratch[0],
                           i_gpr_ptr,
                           0 );

  if( i_remainder_size > 0 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_gpr_scratch[1],
                             0,
                             i_remainder_size );
    libxsmm_ppc64le_instr_4( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_RLDICR,
                             i_gpr_scratch[1],
                             i_gpr_scratch[1],
                             64 - 8,
                             63 );
  }

  for( unsigned int l_bn = 0; l_bn < i_n_blocking; l_bn++ ) {
    /* prep the GPRs for the storage accesses */
    unsigned int l_n_ops = i_m_blocking_full;
    if( i_remainder_size > 0 ) {
      l_n_ops++;
    }
    for( unsigned int l_gp = 0; l_gp < l_n_ops; l_gp++ ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               i_gpr_scratch[2 + l_gp],
                               i_gpr_scratch[0],
                               16*l_gp );
    }

    if( l_bn != i_n_blocking - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               i_gpr_scratch[0],
                               i_gpr_scratch[0],
                               i_stride );
    }

    /* 128bit loads/stores */
    for( unsigned int l_bm = 0; l_bm < i_m_blocking_full; l_bm++ ) {
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVW4X,
                               l_vsr,
                               0,
                               i_gpr_scratch[2 + l_bm],
                               (0x0020 & l_vsr) >> 5 );
      l_vsr++;
    }

    /* remainder load/store (if required) */
    if( i_remainder_size > 0 ) {
      /* reverse byte-order before storing for LE */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXBRW,
                               l_vsr,
                               l_vsr,
                               (0x0020 & l_vsr) >> 5,
                               (0x0020 & l_vsr) >> 5 );

      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVLL,
                               l_vsr,
                               i_gpr_scratch[2 + i_m_blocking_full],
                               i_gpr_scratch[1],
                               (0x0020 & l_vsr) >> 5 );
      l_vsr++;
    }
  }

  return l_vsr - i_vsr_first;
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_vsx_m_loop( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker,
                                                libxsmm_loop_label_tracker    *io_loop_labels,
                                                unsigned char const            i_a,
                                                unsigned char const            i_b,
                                                unsigned char const            i_c ) {
  unsigned int l_m_iters = i_xgemm_desc->m / io_blocking->block_m;
  unsigned int l_m_loop;
  unsigned int l_packed = ( ( io_blocking->block_m % io_blocking->vector_len_comp ) == 0 &&
                            ( i_xgemm_desc->m % io_blocking->block_m ) == 0 ) ? 1 : 0;

  /* Create local pointer for a */
  unsigned int l_a, l_c;
  if ( l_m_iters > 1 || !l_packed ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_c, l_c, i_c );
  } else {
    l_a = i_a;
    l_c = i_c;
  }

  /* Set jump point if required */
  if ( l_m_iters > 1 ) {
    l_m_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_loop,
                             0,
                             l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );
  }

  /* Call k-loop */
  if ( l_m_iters > 0 ) {
    libxsmm_generator_vsx_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       l_a,
                                       i_b,
                                       l_c );
  }

  /* Increment a and c pointers if required */
  if ( l_m_iters > 1 || ( l_m_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_a,
                             l_a,
                             io_blocking->comp_bytes*io_blocking->block_m );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_c,
                             l_c,
                             io_blocking->comp_bytes*io_blocking->block_m );
  }

  /* Compare and jump if required */
  if ( l_m_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_loop,
                             l_m_loop,
                             -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   l_m_loop,
                                                   io_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
  }

  /* Partial M block */
  if ( !l_packed ) {
    unsigned int l_block_m = io_blocking->block_m;
    io_blocking->block_m = ( i_xgemm_desc->m % l_block_m );

    libxsmm_generator_vsx_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       l_a,
                                       i_b,
                                       l_c );

    io_blocking->block_m = l_block_m;
  }

  /* Free a and c pointers if required */
  if ( l_m_iters > 1 || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }

  return;
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker ) {

  /* loop labels reset */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  unsigned char i_a = LIBXSMM_PPC64LE_GPR_R3;
  unsigned char i_b = LIBXSMM_PPC64LE_GPR_R4;
  unsigned char i_c = LIBXSMM_PPC64LE_GPR_R5;

  /* n loop values */
  unsigned int l_n_iters = i_xgemm_desc->n / io_blocking->block_n;
  unsigned int l_n_loop;
  unsigned int l_packed = ( ( io_blocking->block_n % io_blocking->vector_len_comp ) == 0 &&
                            ( i_xgemm_desc->n % io_blocking->block_n ) == 0 ) ? 1 : 0;

  /* Set up local pointers for b and c if required */
  unsigned char l_b, l_c;
  if ( l_n_iters > 1 || !l_packed ) {
    l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_b, l_b, i_b );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_c, l_c, i_c );
  } else {
    l_b = i_b;
    l_c = i_c;
  }

  /* Set jump point if required */
  if ( l_n_iters > 1 ) {
    l_n_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, 0, l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, &l_loop_labels );
  }

  /* Call packed m-loop */
  if ( l_n_iters > 0 ) {
    libxsmm_generator_gemm_ppc64le_vsx_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               i_a,
                                               l_b,
                                               l_c );
  }

  /* Increment b and c pointers if required */
  if ( ( l_n_iters > 1 ) || ( l_n_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_b,
                             l_b,
                             i_xgemm_desc->ldb*io_blocking->comp_bytes*io_blocking->block_n );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_c,
                             l_c,
                             i_xgemm_desc->ldc*io_blocking->comp_bytes*io_blocking->block_n );
  }

  /* Compare and jump if required */
  if ( l_n_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, l_n_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );
  }

  /* Partial N block */
  if ( !l_packed ) {
    unsigned int l_block_n = io_blocking->block_n;
    io_blocking->block_n = i_xgemm_desc->n % io_blocking->block_n;
    libxsmm_generator_gemm_ppc64le_vsx_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               i_a,
                                               l_b,
                                               l_c );
    io_blocking->block_n = l_block_n;
  }

  /* Free b and c pointers if required */
  if ( l_n_iters > 1 || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }

  /* Colapse stack frame */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, io_reg_tracker );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_mma_m_loop( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker,
                                                libxsmm_loop_label_tracker    *io_loop_labels,
                                                unsigned int                  *i_acc,
                                                unsigned char const            i_a,
                                                unsigned char const            i_b,
                                                unsigned char const            i_c ) {
  unsigned int l_m_iters = i_xgemm_desc->m / io_blocking->block_m;
  unsigned int l_m_loop;
  unsigned int l_packed = ( ( io_blocking->block_m % io_blocking->vector_len_comp ) == 0 &&
                            ( i_xgemm_desc->m % io_blocking->block_m ) == 0 ) ? 1 : 0;

  /* Create local pointer for a */
  unsigned int l_a, l_c;
  if ( l_m_iters > 1 || !l_packed ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_c, l_c, i_c );
  } else {
    l_a = i_a;
    l_c = i_c;
  }

  /* Set jump point if required */
  if ( l_m_iters > 1 ) {
    l_m_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_loop,
                             0,
                             l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, io_loop_labels );
  }

  /* Call k-loop */
  libxsmm_generator_mma_microkernel( io_generated_code,
                                     i_xgemm_desc,
                                     io_blocking,
                                     io_reg_tracker,
                                     io_loop_labels,
                                     i_acc,
                                     l_a,
                                     i_b,
                                     l_c );

  /* Increment a and c pointers if required */
  if ( l_m_iters > 1 || ( l_m_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_a,
                             l_a,
                             io_blocking->comp_bytes*io_blocking->block_m );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_c,
                             l_c,
                             io_blocking->comp_bytes*io_blocking->block_m );
  }

  /* Compare and jump if required */
  if ( l_m_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_loop,
                             l_m_loop,
                             -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   l_m_loop,
                                                   io_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
  }

  /* Partial M block */
  if ( !l_packed ) {
    unsigned int l_block_m = io_blocking->block_m;
    io_blocking->block_m = ( i_xgemm_desc->m % l_block_m );

    libxsmm_generator_mma_microkernel( io_generated_code,
                                       i_xgemm_desc,
                                       io_blocking,
                                       io_reg_tracker,
                                       io_loop_labels,
                                       i_acc,
                                       l_a,
                                       i_b,
                                       l_c );

    io_blocking->block_m = l_block_m;
  }

  /* Free a and c pointers if required */
  if ( l_m_iters > 1 || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }

  return;
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_mma( libxsmm_generated_code        *io_generated_code,
                                                libxsmm_gemm_descriptor const *i_xgemm_desc,
                                                libxsmm_ppc64le_blocking      *io_blocking,
                                                libxsmm_ppc64le_reg           *io_reg_tracker ) {

  /* loop labels reset */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* GPRs holding pointers to A, B, and C */
  unsigned char i_a = LIBXSMM_PPC64LE_GPR_R3;
  unsigned char i_b = LIBXSMM_PPC64LE_GPR_R4;
  unsigned char i_c = LIBXSMM_PPC64LE_GPR_R5;

  /* n loop values */
  unsigned int l_n_iters = i_xgemm_desc->n / io_blocking->block_n;
  unsigned int l_n_loop;
  unsigned int l_packed = ( ( io_blocking->block_n % io_blocking->vector_len_comp ) == 0 &&
                            ( i_xgemm_desc->n % io_blocking->block_n ) == 0 ) ? 1 : 0;

  /* Allocate C accumaltors early as contiguous VSR are needed */
  unsigned int l_c_acc[io_blocking->n_acc_c];
  for ( int l_i = 0; l_i < io_blocking->n_acc_c; ++l_i ) {
    l_c_acc[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_ACC );
  }

  /* Create local pointers for b and c */
  unsigned char l_b, l_c;
  if ( l_n_iters > 1 || !l_packed ) {
    l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_c = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_b, l_b, i_b );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_c, l_c, i_c );
  } else {
    l_b = i_b;
    l_c = i_c;
  }

  /* Set jump point for N-loop if required */
  if ( l_n_iters > 1 ) {
    l_n_loop = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, 0, l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code, &l_loop_labels );
  } else {
    l_b = i_b;
    l_c = i_c;
  }

  /* Call m-loop */
  if ( l_n_iters > 0 ) {
    libxsmm_generator_gemm_ppc64le_mma_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               l_c_acc,
                                               i_a,
                                               l_b,
                                               l_c );
  }

  /* Increment b and c pointers if required */
  if ( ( l_n_iters > 1 ) || ( l_n_iters > 0 && !l_packed ) ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_b,
                             l_b,
                             i_xgemm_desc->ldb*io_blocking->comp_bytes*io_blocking->block_n );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_c,
                             l_c,
                             i_xgemm_desc->ldc*io_blocking->comp_bytes*io_blocking->block_n );
  }

  /* Compare and jump if required */
  if ( l_n_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_n_loop, l_n_loop, -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code, l_n_loop, &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );
  }

  /* Partial N block */
  if ( !l_packed ) {
    unsigned int l_block_n = io_blocking->block_n;
    io_blocking->block_n = i_xgemm_desc->n % io_blocking->block_n;
    libxsmm_generator_gemm_ppc64le_mma_m_loop( io_generated_code,
                                               i_xgemm_desc,
                                               io_blocking,
                                               io_reg_tracker,
                                               &l_loop_labels,
                                               l_c_acc,
                                               i_a,
                                               l_b,
                                               l_c );
    io_blocking->block_n = l_block_n;
  }

  /* Free b and c pointers if required */
  if ( l_n_iters > 1 || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c );
  }

  /* Free accumulator */
  for ( int l_i = 0; l_i < io_blocking->n_acc_c; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_ACC, l_c_acc[l_i] );
  }

  /* Colapse stack frame */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, io_reg_tracker );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel( libxsmm_generated_code        *io_generated_code,
                                            const libxsmm_gemm_descriptor *i_xgemm_desc ) {
  void (*l_generator_kernel)( libxsmm_generated_code *,
                              libxsmm_gemm_descriptor const *,
                              libxsmm_ppc64le_blocking *,
                              libxsmm_ppc64le_reg * );

  if (io_generated_code->arch == LIBXSMM_PPC64LE_VSX) {
    l_generator_kernel = &libxsmm_generator_gemm_ppc64le_kernel_vsx;
  } else if (io_generated_code->arch == LIBXSMM_PPC64LE_MMA) {
    l_generator_kernel = &libxsmm_generator_gemm_ppc64le_kernel_mma;
    //l_generator_kernel = &libxsmm_generator_gemm_ppc64le_kernel_vsx;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Initialise reg tracker */
  libxsmm_ppc64le_reg l_reg_tracker = libxsmm_ppc64le_reg_init();

  /* Initialise blocking */
  libxsmm_ppc64le_blocking l_blocking;
  libxsmm_generator_gemm_ppc64le_setup_blocking( io_generated_code,
                                                 i_xgemm_desc,
                                                 &l_blocking );

  /*libxsmm_generator_gemm_ppc64le_kernel_mma( io_generated_code, i_xgemm_desc, &l_blocking, &l_reg_tracker );*/
  l_generator_kernel( io_generated_code, i_xgemm_desc, &l_blocking, &l_reg_tracker );
  return;
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_setup_blocking( libxsmm_generated_code        *io_generated_code,
                                                    const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                    libxsmm_ppc64le_blocking      *io_blocking ) {

  unsigned int l_v_bytes = LIBXSMM_PPC64LE_VSR_WIDTH / 8;

  io_blocking->vector_len_a = l_v_bytes / libxsmm_ppc64le_instr_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_b = l_v_bytes / libxsmm_ppc64le_instr_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_c = l_v_bytes / libxsmm_ppc64le_instr_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) );
  io_blocking->comp_bytes = libxsmm_ppc64le_instr_bytes( io_generated_code, LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) );
  io_blocking->vector_len_comp = l_v_bytes / io_blocking->comp_bytes;

  if ( io_generated_code->arch == LIBXSMM_PPC64LE_VSX ) {
    io_blocking->block_m = ( 64 / io_blocking->comp_bytes < i_xgemm_desc->m ) ?
      64 / io_blocking->comp_bytes :
      io_blocking->vector_len_comp*( ( i_xgemm_desc->m + io_blocking->vector_len_comp - 1) / io_blocking->vector_len_comp );
    io_blocking->block_n = ( 16 / io_blocking->comp_bytes < i_xgemm_desc->n ) ?
      16 / io_blocking->comp_bytes :
      io_blocking->vector_len_comp*( ( i_xgemm_desc->n + io_blocking->vector_len_comp - 1) / io_blocking->vector_len_comp );
    io_blocking->block_k = ( 16 / io_blocking->comp_bytes < i_xgemm_desc->k ) ?
      16 / io_blocking->comp_bytes :
      io_blocking->vector_len_comp*( ( i_xgemm_desc->k + io_blocking->vector_len_comp - 1) / io_blocking->vector_len_comp );

    io_blocking->n_block_m_full = i_xgemm_desc->m / io_blocking->block_m;
    io_blocking->n_block_n_full = i_xgemm_desc->n / io_blocking->block_n;
    io_blocking->n_block_k_full = i_xgemm_desc->k / io_blocking->block_k;

    if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) ) {
      io_blocking->reg_lda = (io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
      io_blocking->reg_ldb = io_blocking->block_k;
      io_blocking->reg_ldc = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;

      io_blocking->n_reg_a = ( ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp )*io_blocking->block_k;
      io_blocking->n_reg_b = io_blocking->block_n*io_blocking->block_k;
      io_blocking->n_reg_c = ( ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp )*io_blocking->block_n;

    }
  } else if ( io_generated_code->arch == LIBXSMM_PPC64LE_MMA ) {
    io_blocking->block_m = ( 64 / io_blocking->comp_bytes < i_xgemm_desc->m ) ?
      64 / io_blocking->comp_bytes :
      io_blocking->vector_len_comp*( ( i_xgemm_desc->m + io_blocking->vector_len_comp - 1) / io_blocking->vector_len_comp );

    io_blocking->block_n = ( 32 / io_blocking->comp_bytes < i_xgemm_desc->n ) ?
      32 / io_blocking->comp_bytes :
      io_blocking->vector_len_comp*( ( i_xgemm_desc->n + io_blocking->vector_len_comp - 1) / io_blocking->vector_len_comp );

    io_blocking->block_k = ( 16 / io_blocking->comp_bytes < i_xgemm_desc->k ) ?
      16 / io_blocking->comp_bytes :
      io_blocking->vector_len_comp*( ( i_xgemm_desc->k + io_blocking->vector_len_comp - 1) / io_blocking->vector_len_comp );

    io_blocking->n_block_m_full = i_xgemm_desc->m / io_blocking->block_m;
    io_blocking->n_block_n_full = i_xgemm_desc->n / io_blocking->block_n;
    io_blocking->n_block_k_full = i_xgemm_desc->k / io_blocking->block_k;

    if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ) ) {
      io_blocking->reg_lda = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;
      io_blocking->reg_ldb = io_blocking->block_k;
      io_blocking->reg_ldc = ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp;

      io_blocking->n_reg_a = ( ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp )*io_blocking->block_k;
      io_blocking->n_reg_b = ( ( io_blocking->block_n + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp )*io_blocking->block_k;
      io_blocking->n_acc_c = ( ( ( io_blocking->block_n + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp ) *
                               ( ( io_blocking->block_m + io_blocking->vector_len_comp - 1 ) / io_blocking->vector_len_comp ) );
      io_blocking->n_reg_c = io_blocking->n_acc_c*4;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  unsigned int l_n_reg = io_blocking->n_reg_a + io_blocking->n_reg_b + io_blocking->n_reg_c;
  if ( l_n_reg > LIBXSMM_PPC64LE_VSR_NMAX ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
    exit(-1);
  }
}
