/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_gemm_vxrs_microkernel.h"

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_microkernel( libxsmm_generated_code        *io_generated_code,
                                         const libxsmm_gemm_descriptor *i_xgemm_desc,
                                         libxsmm_s390x_reg             *io_reg_tracker,
                                         libxsmm_loop_label_tracker    *io_loop_labels,
                                         const libxsmm_s390x_blocking  *i_blocking,
                                         unsigned int                   i_a,
                                         unsigned int                   i_b,
                                         unsigned int                   i_c ) {
  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int *l_a_reg, *l_b_reg, *l_c_reg;
  unsigned int k, l_n_k_blocks = ( i_xgemm_desc->k + i_blocking->block_k - 1 ) / i_blocking->block_k;
  unsigned int l_vec_ele = i_blocking->vector_len_comp;
  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x04 ) >> 2;
  unsigned int l_a, l_b, l_a_temp = 0, l_b_temp = 0;
  unsigned int l_n_reg_a, l_reg_lda, l_n_reg_b, l_reg_ldb, l_n_reg_c, l_reg_ldc;

  l_reg_lda = ( i_blocking->block_m + l_vec_ele - 1 ) / l_vec_ele;
  l_n_reg_a = l_reg_lda * i_blocking->block_k ;
  l_reg_ldb = i_blocking->block_k;
  l_n_reg_b = l_reg_ldb*i_blocking->block_n;
  l_reg_ldc = ( i_blocking->block_m + l_vec_ele - 1 ) / l_vec_ele;
  l_n_reg_c = l_reg_ldc * i_blocking->block_n ;

  /* Allocate space for a, b, and c registers */
  l_a_reg = malloc(l_n_reg_a*sizeof(unsigned int));
  l_b_reg = malloc(l_n_reg_b*sizeof(unsigned int));
  l_c_reg = malloc(l_n_reg_c*sizeof(unsigned int));
  libxsmm_s390x_reg_alloc_vr_mat( io_generated_code, io_reg_tracker, l_reg_lda, i_blocking->block_k, l_a_reg );
  libxsmm_s390x_reg_alloc_vr_mat( io_generated_code, io_reg_tracker, l_reg_ldc, i_blocking->block_n, l_c_reg );
  libxsmm_s390x_reg_alloc( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR, l_n_reg_b, l_b_reg );

  /* Load C is beta != 0 */
  if ( !l_beta_zero ) {
    libxsmm_generator_vxrs_block_load_mult( io_generated_code,
                                            i_xgemm_desc,
                                            io_reg_tracker,
                                            l_c_datatype,
                                            l_comptype,
                                            i_c,
                                            i_blocking->block_m,
                                            i_blocking->block_n,
                                            i_xgemm_desc->ldc,
                                            l_c_reg,
                                            l_reg_ldc );
  }

  /* Decide if we need temporary pointers */
  if ( l_n_k_blocks > 1 ) {
    l_a_temp = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    l_b_temp = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
  }
  l_a = i_a;
  l_b = i_b;

  /* Unroll k-loop */
  for ( k = 0 ; k < l_n_k_blocks ; ++k ) {
    unsigned int l_k_rem = i_xgemm_desc->k - k*i_blocking->block_k;
    unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;
    unsigned int l_beta = ( !k && l_beta_zero ) ? 0 : 1;

    /* Load vectors of A */
    libxsmm_generator_vxrs_block_load_mult( io_generated_code,
                                            i_xgemm_desc,
                                            io_reg_tracker,
                                            l_a_datatype,
                                            l_comptype,
                                            l_a,
                                            i_blocking->block_m,
                                            l_block_k,
                                            i_xgemm_desc->lda,
                                            l_a_reg,
                                            l_reg_lda );

    /* Broadcasted load of B */
    libxsmm_generator_vxrs_block_load_bcast( io_generated_code,
                                             i_xgemm_desc,
                                             io_reg_tracker,
                                             l_b_datatype,
                                             l_comptype,
                                             l_b,
                                             l_block_k,
                                             i_blocking->block_n,
                                             i_xgemm_desc->ldb,
                                             l_b_reg,
                                             l_reg_ldb );

    /* Call FMA */
    libxsmm_generator_vxrs_block_fma_b_splat( io_generated_code,
                                              l_comptype,
                                              ( i_blocking->block_m + l_vec_ele - 1 ) / l_vec_ele,
                                              i_blocking->block_n,
                                              l_block_k,
                                              l_a_reg,
                                              l_reg_lda,
                                              l_b_reg,
                                              l_reg_ldb,
                                              l_beta,
                                              l_c_reg,
                                              l_reg_ldc );

    if ( k < l_n_k_blocks - 1 ) {
      libxsmm_s390x_instr_gpr_add_value( io_generated_code, l_a, l_a_temp, i_blocking->block_k*i_xgemm_desc->lda*i_blocking->comp_bytes );
      libxsmm_s390x_instr_gpr_add_value( io_generated_code, l_b, l_b_temp, i_blocking->block_k*i_blocking->comp_bytes );
    }
    if ( k == 0 && l_n_k_blocks > 1 ) {
      l_a = l_a_temp;
      l_b = l_b_temp;
    }
  }

  /* Store result */
  libxsmm_generator_vxrs_block_store_mult( io_generated_code,
                                           i_xgemm_desc,
                                           io_reg_tracker,
                                           l_c_datatype,
                                           l_comptype,
                                           i_c,
                                           i_blocking->block_m,
                                           i_blocking->block_n,
                                           i_xgemm_desc->ldc,
                                           l_c_reg,
                                           l_reg_ldc );

  /* Free a, b, and c register */
  libxsmm_s390x_reg_dealloc( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR, l_n_reg_a, l_a_reg );
  free(l_a_reg);
  libxsmm_s390x_reg_dealloc( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR, l_n_reg_b, l_b_reg );
  free(l_b_reg);
  libxsmm_s390x_reg_dealloc( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR, l_n_reg_c, l_c_reg );
  free(l_c_reg);

  for ( k=0 ; k<io_reg_tracker->nvr; ++k ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR, k);
  }

  /* Free pointers */
  if ( l_n_k_blocks > 1 ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_a_temp );
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_b_temp );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_fma_b_splat( libxsmm_generated_code *io_generated_code,
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
  for ( unsigned int l_k = 0; l_k < i_k; ++l_k ) {
    for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
      for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
        char l_beta = ( ( l_k == 0 ) && ( i_beta == 0 ) ) ? 0 : 1;
        libxsmm_s390x_instr_vxrs_alu( io_generated_code, i_datatype, i_a[l_m + l_k*i_lda], i_b[l_k + l_n*i_ldb], io_c[l_m + l_n*i_ldc], 1, l_beta );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_load_mult( libxsmm_generated_code        *io_generated_code,
                                             const libxsmm_gemm_descriptor *i_xgemm_desc,
                                             libxsmm_s390x_reg             *io_reg_tracker,
                                             const libxsmm_datatype         i_datatype,
                                             const libxsmm_datatype         i_comptype,
                                             unsigned int                   i_a,
                                             unsigned int                   i_m,
                                             unsigned int                   i_n,
                                             unsigned int                   i_lda,
                                             unsigned int                  *io_t,
                                             unsigned int                   i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_s390x_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_S390X_VR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_rem = i_m % l_vec_ele;

  unsigned int l_col;

  for ( l_col = 0; l_col < i_n; ++l_col ) {
    /* If packed, use multiload */
    if ( l_m_blocks > 1 ) {
      unsigned int l_offset = i_lda*l_col*l_databytes;
      unsigned int l_t = io_t[i_ldt*l_col];
      libxsmm_s390x_instr_vec_load_mult( io_generated_code, io_reg_tracker, i_a, l_m_blocks, l_offset, l_t );
    } else if ( l_m_blocks == 1 ) {
      unsigned int l_offset = i_lda*l_col*l_databytes;
      unsigned int l_t = io_t[i_ldt*l_col];
      libxsmm_s390x_instr_vec_load( io_generated_code, io_reg_tracker, i_a, l_offset, l_t );
    }

    /* Partial loads */
    if ( l_m_rem != 0 ) {
      unsigned int l_offset = i_lda*l_col*l_databytes + l_vec_len*l_m_blocks;
      unsigned int l_t = io_t[i_ldt*l_col + l_m_blocks];
      libxsmm_s390x_instr_vec_load_part( io_generated_code, io_reg_tracker, i_datatype, i_a, l_m_rem, l_offset, l_t );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_load_bcast( libxsmm_generated_code        *io_generated_code,
                                              const libxsmm_gemm_descriptor *i_xgemm_desc,
                                              libxsmm_s390x_reg             *io_reg_tracker,
                                              const libxsmm_datatype         i_datatype,
                                              const libxsmm_datatype         i_comptype,
                                              unsigned int                   i_a,
                                              unsigned int                   i_m,
                                              unsigned int                   i_n,
                                              unsigned int                   i_lda,
                                              unsigned int                  *io_t,
                                              unsigned int                   i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_s390x_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_S390X_VR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_load_blocks = i_m / l_vec_ele;
  unsigned int l_m_load_rem = i_m % l_vec_ele;

  unsigned int l_col, l_row, l_ele;

  unsigned int l_scratch = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR );

  for ( l_col = 0; l_col < i_n; ++l_col ) {
    /* Packed load into scratch and then broadcast */
    for ( l_row = 0; l_row < l_m_load_blocks; ++l_row ) {
      unsigned int l_offset = i_lda*l_col*l_databytes + l_vec_len*l_row;
      libxsmm_s390x_instr_vec_load( io_generated_code, io_reg_tracker, i_a, l_offset, l_scratch );

      for ( l_ele = 0; l_ele < l_vec_ele; ++l_ele ) {
        unsigned int l_t = io_t[l_col*i_ldt + l_row*l_vec_ele + l_ele];
        libxsmm_s390x_instr_vec_bcast( io_generated_code, i_datatype, l_scratch, l_ele, l_t );
      }
    }

    /* For partial, just perform load and broadcast op */
    for ( l_ele = 0; l_ele < l_m_load_rem; ++l_ele ) {
      unsigned int l_offset = i_lda*l_col*l_databytes + l_vec_len*l_m_load_blocks + l_ele*l_databytes;
      unsigned int l_t = io_t[l_col*i_ldt + l_vec_ele*l_m_load_blocks + l_ele];
      libxsmm_s390x_instr_vec_load_bcast( io_generated_code, io_reg_tracker, i_datatype, i_a, l_offset, l_t );
    }
  }
  libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_VR, l_scratch );

}

LIBXSMM_API_INTERN
void libxsmm_generator_vxrs_block_store_mult( libxsmm_generated_code        *io_generated_code,
                                              const libxsmm_gemm_descriptor *i_xgemm_desc,
                                              libxsmm_s390x_reg             *io_reg_tracker,
                                              const libxsmm_datatype         i_datatype,
                                              const libxsmm_datatype         i_comptype,
                                              unsigned int                   i_a,
                                              unsigned int                   i_m,
                                              unsigned int                   i_n,
                                              unsigned int                   i_lda,
                                              unsigned int                  *io_t,
                                              unsigned int                   i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_s390x_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_S390X_VR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_rem = i_m % l_vec_ele;

  unsigned int l_col;

  for ( l_col = 0; l_col < i_n; ++l_col ) {
    /* If packed, use multistore */
    if ( l_m_blocks > 1 ) {
      unsigned int l_offset = i_lda*l_col*l_databytes;
      unsigned int l_t = io_t[i_ldt*l_col];
      libxsmm_s390x_instr_vec_store_mult( io_generated_code, io_reg_tracker, i_a, l_m_blocks, l_offset, l_t );
    } else if ( l_m_blocks == 1 ) {
      unsigned int l_offset = i_lda*l_col*l_databytes;
      unsigned int l_t = io_t[i_ldt*l_col];
      libxsmm_s390x_instr_vec_store( io_generated_code, io_reg_tracker, i_a, l_offset, l_t );
    }

    /* Partial store */
    if ( l_m_rem != 0 ) {
      unsigned int l_offset = i_lda*l_col*l_databytes + l_vec_len*l_m_blocks;
      unsigned int l_t = io_t[i_ldt*l_col + l_m_blocks];
      libxsmm_s390x_instr_vec_store_part( io_generated_code, io_reg_tracker, i_datatype, i_a, l_m_rem, l_offset, l_t );
    }
  }
}
