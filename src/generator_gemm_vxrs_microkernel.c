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
                                         const libxsmm_s390x_blocking  *i_blocking ) {

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
    if ( l_m_blocks > 0 ) {
      unsigned int l_offset = i_lda*l_col;
      unsigned int l_t = io_t[i_ldt*l_col];
      libxsmm_s390x_instr_vec_load_mult( io_generated_code, io_reg_tracker, i_a, l_m_blocks, l_offset, l_t );
    }

    /* Partial loads */
    if ( l_m_rem != 0 ) {
      unsigned int l_offset = i_lda*l_col + l_vec_len*l_m_blocks;
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
      unsigned int l_offset = i_lda*l_col + l_vec_len*l_row;
      libxsmm_s390x_instr_vec_load( io_generated_code, io_reg_tracker, i_a, l_offset, l_scratch );

      for ( l_ele = 0; l_ele < l_vec_ele; ++l_ele ) {
        unsigned int l_t = io_t[l_col*i_ldt + l_row*l_vec_ele + l_ele];
        libxsmm_s390x_instr_vec_bcast( io_generated_code, i_datatype, l_scratch, l_ele, l_t );
      }
    }

    /* For partial, just perform load and broadcast op */
    for ( l_ele = 0; l_ele < l_m_load_rem; ++l_ele ) {
      unsigned int l_offset = i_lda*l_col + l_vec_len*l_m_load_blocks + l_ele*l_databytes;
      unsigned int l_t = io_t[l_col*i_ldt + l_vec_ele*l_m_load_blocks + l_ele];
      libxsmm_s390x_instr_vec_load_bcast( io_generated_code, io_reg_tracker, i_datatype, i_a, l_offset, l_t );
    }
  }
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
    if ( l_m_blocks > 0 ) {
      unsigned int l_offset = i_lda*l_col;
      unsigned int l_t = io_t[i_ldt*l_col];
      libxsmm_s390x_instr_vec_store_mult( io_generated_code, io_reg_tracker, i_a, l_m_blocks, l_offset, l_t );
    }

    /* Partial store */
    if ( l_m_rem != 0 ) {
      unsigned int l_offset = i_lda*l_col + l_vec_len*l_m_blocks;
      unsigned int l_t = io_t[i_ldt*l_col + l_m_blocks];
      libxsmm_s390x_instr_vec_store_part( io_generated_code, io_reg_tracker, i_datatype, i_a, l_m_rem, l_offset, l_t );
    }
  }
}
