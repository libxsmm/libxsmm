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
void libxsmm_generator_gemm_ppc64le_vsx_m_loop( libxsmm_generated_code        * io_generated_code,
                                                libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                libxsmm_ppc64le_reg           * io_reg_tracker,
                                                libxsmm_loop_label_tracker    * io_loop_labels,
                                                unsigned int                  * i_blocking,
                                                unsigned char const             i_a_ptr_gpr,
                                                unsigned char const             i_b_ptr_gpr,
                                                unsigned char const             i_c_ptr_gpr ) {

  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  //libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_bytes = libxsmm_ppc64le_instr_bytes( io_generated_code, l_a_datatype );
  unsigned int l_m_iters = i_xgemm_desc->m / i_blocking[1];
  unsigned int l_m_loop;

  /* Create local pointer for a */
  unsigned int l_a_ptr_gpr, l_c_ptr_gpr;

  /* Set jump point if required */
  if ( l_m_iters > 1 ) {
    l_a_ptr_gpr = libxsmm_ppc64le_get_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             i_a_ptr_gpr,
                             l_a_ptr_gpr,
                             i_a_ptr_gpr );
    l_c_ptr_gpr = libxsmm_ppc64le_get_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             i_c_ptr_gpr,
                             l_c_ptr_gpr,
                             i_c_ptr_gpr );

    l_m_loop = libxsmm_ppc64le_get_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_loop,
                             0,
                             l_m_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code,
                                                    io_loop_labels );
  } else {
    l_a_ptr_gpr = i_a_ptr_gpr;
    l_c_ptr_gpr = i_c_ptr_gpr;
  }

  /* Call k-loop */
  libxsmm_generator_vsx_microkernel( io_generated_code,
                                     i_xgemm_desc,
                                     io_reg_tracker,
                                     io_loop_labels,
                                     i_blocking,
                                     l_a_ptr_gpr,
                                     i_b_ptr_gpr,
                                     l_c_ptr_gpr );

  /* Compare and jump if required */
  if ( l_m_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_a_ptr_gpr,
                             l_a_ptr_gpr,
                             l_bytes*i_blocking[1] );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_c_ptr_gpr,
                             l_c_ptr_gpr,
                             l_bytes*i_blocking[1] );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_loop,
                             l_m_loop,
                             -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   l_m_loop,
                                                   io_loop_labels );
    libxsmm_ppc64le_free_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_loop );
    libxsmm_ppc64le_free_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_ptr_gpr );
    libxsmm_ppc64le_free_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_ptr_gpr );
  }

  return;
}


/*

m n k

A[m * k] * B[k * n] = C[m * n]

*/

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                libxsmm_ppc64le_reg           * io_reg_tracker ) {

  /* loop labels reset */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, io_reg_tracker );

  /* create blocking */
  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  //libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_bytes = libxsmm_ppc64le_instr_bytes( io_generated_code, l_a_datatype );
  unsigned int l_v_len = ( LIBXSMM_PPC64LE_VSR_WIDTH / 8 ) / l_bytes;
  unsigned int l_blocking[3];
  l_blocking[0] = 16 / l_bytes < i_xgemm_desc->n ? 16 / l_bytes : i_xgemm_desc->n; /* n-blocking */
  l_blocking[1] = 64 / l_bytes < i_xgemm_desc->m ? 64 / l_bytes : i_xgemm_desc->m; /* m-blocking */
  l_blocking[2] = 16 / l_bytes < i_xgemm_desc->k ? 16 / l_bytes : i_xgemm_desc->k; /* k-blocking */

  unsigned int l_n_a_reg = ( l_blocking[1] / l_v_len )*l_blocking[2]; /* a registers */
  unsigned int l_n_c_reg = ( l_blocking[1] / l_v_len )*l_blocking[0]; /* c registers */
  unsigned int l_n_b_reg =   l_blocking[0]*l_blocking[2];             /* b registers */
  unsigned int l_n_vsr = l_n_a_reg + l_n_b_reg + l_n_c_reg;

  if ( l_n_vsr > LIBXSMM_PPC64LE_VSR_NMAX ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }

  /* GPRs holding pointers to A, B, and C */
  unsigned char i_a_ptr_gpr = LIBXSMM_PPC64LE_GPR_R3;
  unsigned char i_b_ptr_gpr = LIBXSMM_PPC64LE_GPR_R4;
  unsigned char i_c_ptr_gpr = LIBXSMM_PPC64LE_GPR_R5;

  /* n loop values */
  unsigned int l_n_iters = i_xgemm_desc->n / l_blocking[0];
  unsigned int l_n_loop;

  /* Create local pointers for b and c */
  unsigned char l_b_ptr_gpr;
  unsigned char l_c_ptr_gpr;

  /* Set jump point if required */
  if ( l_n_iters > 1 ) {
    l_b_ptr_gpr = libxsmm_ppc64le_get_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_c_ptr_gpr = libxsmm_ppc64le_get_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             i_b_ptr_gpr,
                             l_b_ptr_gpr,
                             i_b_ptr_gpr );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             i_c_ptr_gpr,
                             l_c_ptr_gpr,
                             i_c_ptr_gpr );

    l_n_loop = libxsmm_ppc64le_get_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_n_loop,
                             0,
                             l_n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code,
                                                    &l_loop_labels );
  } else {
    l_b_ptr_gpr = i_b_ptr_gpr;
    l_c_ptr_gpr = i_c_ptr_gpr;
  }

  /* Call m-loop */
  libxsmm_generator_gemm_ppc64le_vsx_m_loop( io_generated_code,
                                             i_xgemm_desc,
                                             io_reg_tracker,
                                             &l_loop_labels,
                                             l_blocking,
                                             i_a_ptr_gpr,
                                             l_b_ptr_gpr,
                                             l_c_ptr_gpr );

  /* Compare and jump if required */
  if ( l_n_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_b_ptr_gpr,
                             l_b_ptr_gpr,
                             i_xgemm_desc->ldb*l_bytes*l_blocking[0] );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_c_ptr_gpr,
                             l_c_ptr_gpr,
                             i_xgemm_desc->ldc*l_bytes*l_blocking[0] );

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_n_loop,
                             l_n_loop,
                             -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   l_n_loop,
                                                   &l_loop_labels );
    libxsmm_ppc64le_free_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_n_loop );

    libxsmm_ppc64le_free_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_ptr_gpr );
    libxsmm_ppc64le_free_reg( io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_c_ptr_gpr );
  }

  /* close stream */
  libxsmm_ppc64le_instr_close_stream( io_generated_code, io_reg_tracker );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel( libxsmm_generated_code        * io_generated_code,
                                            const libxsmm_gemm_descriptor * i_xgemm_desc ) {
  //void (*l_generator_kernel)( libxsmm_generated_code * io_generated_code,
  //                            libxsmm_gemm_descriptor const * i_xgemm_desc);


  if (io_generated_code->arch == LIBXSMM_PPC64LE_VSX) {
    //l_generator_kernel = libxsmm_generator_gemm_ppc64le_kernel_vsx;
  } else if (io_generated_code->arch == LIBXSMM_PPC64LE_MMA) {
    //l_generator_kernel = libxsmm_generator_gemm_ppc64le_kernel_vsx;
    //LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    //return;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Initialise reg tracker */
  libxsmm_ppc64le_reg l_reg_tracker = libxsmm_ppc64le_reg_init();

  libxsmm_generator_gemm_ppc64le_kernel_vsx( io_generated_code, i_xgemm_desc, &l_reg_tracker );
  return;
}
