/******************************************************************************
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_gemm_vsx_microkernel.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_block_load_vsr( libxsmm_generated_code *io_generated_code,
                                                libxsmm_ppc64le_reg    *io_reg_tracker,
                                                libxsmm_datatype const  i_datatype,
                                                libxsmm_datatype const  i_comptype,
                                                unsigned int            i_a,
                                                unsigned int            i_m,
                                                unsigned int            i_n,
                                                unsigned int            i_lda,
                                                unsigned int           *io_t,
                                                unsigned int            i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_part;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  unsigned int l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m;

  /* Partial length */
  if ( !l_packed ) {
    unsigned int l_temp = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, 0, l_temp, ( i_m % l_vec_ele )*l_databytes );
    libxsmm_ppc64le_instr_set_shift_left( io_generated_code, l_temp, l_m_part, 56 );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_temp );
  }

  /* Offset registers for large lda and unaligned offset as imm limited to 0xffff */
  libxsmm_ppc64le_ptr_reg_alloc( io_generated_code,
                                 io_reg_tracker,
                                 i_a,
                                 i_n,
                                 i_lda*l_databytes,
                                 l_m_blocks*l_vec_len,
                                 l_a_ptr,
                                 l_offsets );

  for ( l_n = 0; l_n < i_n; ++l_n ) {
    /* Full width load */
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {
      long l_offset = l_m*l_vec_len + l_offsets[l_n];
      unsigned int l_t = io_t[l_n*i_ldt + l_m];
      libxsmm_ppc64le_instr_load( io_generated_code, l_a_ptr[l_n], l_offset, l_t );
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      long l_offset = l_m_blocks*l_vec_len + l_offsets[l_n];
      libxsmm_ppc64le_instr_load_part( io_generated_code, io_reg_tracker, l_a_ptr[l_n], l_offset, l_m_part, l_t );
    }
  }

  /* Free GPR */
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_block_store_vsr( libxsmm_generated_code *io_generated_code,
                                                 libxsmm_ppc64le_reg    *io_reg_tracker,
                                                 libxsmm_datatype const  i_datatype,
                                                 libxsmm_datatype const  i_comptype,
                                                 unsigned int            i_a,
                                                 unsigned int            i_m,
                                                 unsigned int            i_n,
                                                 unsigned int            i_lda,
                                                 unsigned int           *io_t,
                                                 unsigned int            i_ldt ) {

  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_part;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  unsigned int l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m;

  /* Partial length */
  if ( !l_packed ) {
    unsigned int l_temp = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, 0, l_temp, ( i_m % l_vec_ele )*l_databytes );
    libxsmm_ppc64le_instr_set_shift_left( io_generated_code, l_temp, l_m_part, 56 );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_temp );
  }

  /* Offset registers for large lda and unaligned offset as imm limited to 0xffff */
  libxsmm_ppc64le_ptr_reg_alloc( io_generated_code,
                                 io_reg_tracker,
                                 i_a,
                                 i_n,
                                 i_lda*l_databytes,
                                 l_m_blocks*l_vec_len,
                                 l_a_ptr,
                                 l_offsets );

  for ( l_n = 0; l_n < i_n; ++l_n ) {
    /* Full width store */
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m];
      long l_offset = l_m*l_vec_len + l_offsets[l_n];
      libxsmm_ppc64le_instr_store( io_generated_code, l_a_ptr[l_n], l_offset, l_t );
    }

    /* Partial store */
    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      long l_offset = l_m_blocks*l_vec_len + l_offsets[l_n];
      libxsmm_ppc64le_instr_store_part( io_generated_code, io_reg_tracker, l_a_ptr[l_n], l_offset, l_m_part, l_t );
    }
  }

  /* Free GPR */
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_block_load_vsr_splat( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      libxsmm_datatype const  i_datatype,
                                                      libxsmm_datatype const  i_comptype, /* currently unsuded */
                                                      unsigned int            i_a,
                                                      unsigned int            i_m,
                                                      unsigned int            i_n,
                                                      unsigned int            i_lda,
                                                      unsigned int           *io_t,
                                                      unsigned int            i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  unsigned int l_scratch, l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m, l_i;

  /* Offset registers for large lda and unaligned offset as imm limited to 0xffff */
  libxsmm_ppc64le_ptr_reg_alloc( io_generated_code,
                                 io_reg_tracker,
                                 i_a,
                                 i_n,
                                 i_lda*l_databytes,
                                 l_m_blocks*l_vec_len,
                                 l_a_ptr,
                                 l_offsets );

  /* Vector scratch register */
  l_scratch = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );

  for ( l_n = 0; l_n < i_n; ++l_n ){
    /* Full width load */
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {
      long l_offset = l_vec_len*l_m + l_offsets[l_n];
      libxsmm_ppc64le_instr_load( io_generated_code, l_a_ptr[l_n], l_offset, l_scratch );

      for ( l_i = 0; l_i < l_vec_ele; ++l_i ) {
        unsigned int l_t = io_t[l_i + l_m*l_vec_ele + l_n*i_ldt];
        libxsmm_ppc64le_instr_vec_splat( io_generated_code, i_datatype, l_scratch, l_vec_ele - 1 - l_i, l_t );
      }
    }

    /* Partial load and splat */
    if ( !l_packed ) {
      for ( l_i = 0; l_i < ( i_m % l_vec_ele ); ++l_i ) {
        unsigned int l_t = io_t[l_i + l_m_blocks*l_vec_ele + l_n*i_ldt];
        long l_offset = l_m_blocks*l_vec_len + l_i*l_databytes + l_offsets[l_n];
        libxsmm_ppc64le_instr_load_splat( io_generated_code, io_reg_tracker, i_datatype, l_a_ptr[l_n], l_offset, l_t );
      }
    }
  }

  /* Free registers */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch );
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_alu( libxsmm_generated_code *io_generated_code,
                                libxsmm_datatype const  i_datatype,
                                unsigned int            i_a,
                                unsigned int            i_b,
                                unsigned int            i_c,
                                char                    i_alpha,
                                char                    i_beta ) {
  /*
    VSX has 5 possible instructions in the A-form variant:
    XVMADDA  c, a, b -> c = a*b + c       | alpha =  1, beta = 1
    XVMSUBA  c, a, b -> c = a*b - c       | alpha =  1, beta = -1
    XVNMADDA c, a, b -> c = - ( a*b + c ) | alpha = -1, beta = -1
    XVNMSUBA c, a, b -> c = -( a*b - c )  | alpha = -1, beta = 1
    XVMUL    c, a, b -> c = a*b           | alpha = -1, beta = 0
  */
  unsigned int l_op;

  /* If beta is zero we only suport positive multiplication */
  if ( ( i_beta == 0 ) && ( i_alpha != 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  } else if ( i_alpha == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_op = LIBXSMM_PPC64LE_INSTR_XVMADDASP;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      l_op = LIBXSMM_PPC64LE_INSTR_XVMADDADP;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  if ( i_beta != 0 ) {
    l_op += 0x40 * ( 1 - i_beta ) + 0x0200 * ( 1 - i_alpha );
  } else {
    l_op += 0x78;
  }

  libxsmm_ppc64le_instr_6( io_generated_code, l_op, i_c, i_a, i_b, ( 0x20 & i_a ) >> 5, ( 0x20 & i_b ) >> 5, ( 0x20 & i_c ) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_splat( libxsmm_generated_code *io_generated_code,
                                              libxsmm_datatype const  i_datatype,
                                              unsigned int            i_m,
                                              unsigned int            i_n,
                                              unsigned int            i_k,
                                              unsigned int           *i_a,
                                              unsigned int            i_lda,
                                              unsigned int           *i_b,
                                              unsigned int            i_ldb,
                                              unsigned int            i_beta,
                                              unsigned int           *io_c,
                                              unsigned int            i_ldc ) {
  unsigned int l_m, l_n, l_k;
  for ( l_k = 0; l_k < i_k; ++l_k ) {
    for ( l_n = 0; l_n < i_n; ++l_n ) {
      for ( l_m = 0; l_m < i_m; ++l_m ) {
        char l_beta = ( ( l_k == 0 ) && ( i_beta == 0 ) ) ? 0 : 1;
        libxsmm_generator_vsx_alu( io_generated_code,
                                   i_datatype,
                                   i_a[l_m + l_k*i_lda],
                                   i_b[l_k + l_n*i_ldb],
                                   io_c[l_m + l_n*i_ldc],
                                   1,
                                   l_beta );
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code        *io_generated_code,
                                        libxsmm_gemm_descriptor const *i_xgemm_desc,
                                        libxsmm_ppc64le_blocking      *i_blocking,
                                        libxsmm_ppc64le_reg           *io_reg_tracker,
                                        libxsmm_loop_label_tracker    *io_loop_labels,
                                        unsigned char                  i_a,
                                        unsigned char                  i_b,
                                        unsigned char                  i_c ) {
  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x0004 ) >> 2;
  unsigned int l_v_len = i_blocking->vector_len_comp;
  unsigned int l_n_k_blocks = ( i_xgemm_desc->k + i_blocking->block_k - 1 ) / i_blocking->block_k;
  unsigned int l_a, l_b, l_a_last, l_b_last, l_a_pipe[2], l_b_pipe[2];
  unsigned l_a_reg[LIBXSMM_PPC64LE_VSR_NMAX], l_b_reg[LIBXSMM_PPC64LE_VSR_NMAX], l_c_reg[LIBXSMM_PPC64LE_VSR_NMAX];
  unsigned int l_i, l_k_block;

  /* Local pointers registers for A and B */
  for ( l_i = 0; l_i < 2; ++l_i ) {
    l_a_pipe[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_pipe[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  }

  /* Allocate vector registers */
  libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, i_blocking->n_reg_a, 1, l_a_reg );
  libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, i_blocking->n_reg_b, 1, l_b_reg );
  libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, i_blocking->n_reg_c, 1, l_c_reg );


  /* Load C if required */
  if ( !l_beta_zero ) {
    libxsmm_generator_gemm_vsx_block_load_vsr( io_generated_code,
                                               io_reg_tracker,
                                               l_c_datatype,
                                               l_comptype,
                                               i_c,
                                               i_blocking->block_m,
                                               i_blocking->block_n,
                                               i_xgemm_desc->ldc,
                                               l_c_reg,
                                               i_blocking->reg_ldc );
  }

  /* Fully unrolled k-loop */
  for ( l_k_block = 0; l_k_block < l_n_k_blocks; ++l_k_block ) {
    unsigned int l_k_rem = i_xgemm_desc->k - l_k_block*i_blocking->block_k;
    unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;
    unsigned int l_beta = ( ( l_k_block == 0 ) && ( l_beta_zero ) ) ? 0 : 1;

    /* Make pipeline of pointers to reduce hazards */
    if ( l_k_block == 0 ) {
      l_a = i_a;
      l_b = i_b;
    } else {
      l_a = l_a_pipe[l_k_block % 2];
      l_b = l_b_pipe[l_k_block % 2];

      libxsmm_ppc64le_instr_add_value(io_generated_code, io_reg_tracker, l_a_last, l_a, i_blocking->block_k*i_xgemm_desc->lda*i_blocking->comp_bytes );
      libxsmm_ppc64le_instr_add_value(io_generated_code, io_reg_tracker, l_b_last, l_b, i_blocking->block_k*i_blocking->comp_bytes );
    }
    l_a_last = l_a;
    l_b_last = l_b;

    /* Load block of A */
    libxsmm_generator_gemm_vsx_block_load_vsr( io_generated_code,
                                               io_reg_tracker,
                                               l_a_datatype,
                                               l_comptype,
                                               l_a,
                                               i_blocking->block_m,
                                               l_block_k,
                                               i_xgemm_desc->lda,
                                               l_a_reg,
                                               i_blocking->reg_lda );

    /* Load block of B and broadcast values to vector */
    libxsmm_generator_gemm_vsx_block_load_vsr_splat( io_generated_code,
                                                     io_reg_tracker,
                                                     l_b_datatype,
                                                     l_comptype,
                                                     l_b,
                                                     l_block_k,
                                                     i_blocking->block_n,
                                                     i_xgemm_desc->ldb,
                                                     l_b_reg,
                                                     i_blocking->reg_ldb );

    /* Block FMA */
    libxsmm_generator_vsx_block_fma_b_splat( io_generated_code,
                                             l_comptype,
                                             ( i_blocking->block_m + l_v_len - 1 ) / l_v_len,
                                             i_blocking->block_n,
                                             l_block_k,
                                             l_a_reg,
                                             i_blocking->reg_lda,
                                             l_b_reg,
                                             i_blocking->reg_ldb,
                                             l_beta,
                                             l_c_reg,
                                             i_blocking->reg_ldc );
  }

  /* Store result block in C */
  libxsmm_generator_gemm_vsx_block_store_vsr( io_generated_code,
                                              io_reg_tracker,
                                              l_c_datatype,
                                              l_comptype,
                                              i_c,
                                              i_blocking->block_m,
                                              i_blocking->block_n,
                                              i_xgemm_desc->ldc,
                                              l_c_reg,
                                              i_blocking->reg_ldc );

  /* Free A and B registers */
  for ( l_i = 0; l_i < i_blocking->n_reg_a; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( l_i = 0; l_i < i_blocking->n_reg_b; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }

  /* Free pointer pipeline */
  for ( l_i = 0; l_i < 2; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_pipe[l_i] );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_pipe[l_i] );
  }

  /* Free C registers */
  for ( l_i = 0; l_i < i_blocking->n_reg_c; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_c_reg[l_i] );
  }

  return;
}
