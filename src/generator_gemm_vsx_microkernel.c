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
void libxsmm_generator_gemm_vsx_mk_load_trans( libxsmm_generated_code * io_generated_code,
                                               libxsmm_datatype const   i_datatype,
                                               libxsmm_datatype const   i_comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * io_reg_tracker,
                                               unsigned int           * i_loaded_regs,
                                               unsigned int const       i_ptr_gpr,
                                               unsigned int const       i_n_rows,
                                               unsigned int const       i_n_cols,
                                               unsigned int const       i_stride ) {

  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (l_databytes * 8);
  unsigned int l_n_col_blocks = i_n_cols / l_vec_len;
  unsigned int l_n_row_blocks = i_n_rows / l_vec_len;
  unsigned int l_block_ld = ( i_n_cols + l_vec_len - 1 ) / l_vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_col_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_ptr,
                           i_ptr_gpr );

  if( l_databytes*i_stride > (unsigned int)(0xffff) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* load the fully packed part */
  for ( unsigned int l_row_block = 0; l_row_block < l_n_row_blocks; ++l_row_block ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_ptr,
                             l_col_ptr,
                             l_ptr );

    for ( unsigned int l_col_block = 0; l_col_block < l_n_col_blocks; ++l_col_block ) {
      /* vector load */
      if ( i_datatype == LIBXSMM_DATATYPE_F32 ) {
        for ( unsigned int l_vec = 0; l_vec < l_vec_len; ++l_vec ) {
          unsigned int l_reg_idx = l_col_block + (l_row_block*l_vec_len + l_vec)*l_block_ld;
          libxsmm_ppc64le_instr_4( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                   i_loaded_regs[l_reg_idx],
                                   0,
                                   l_col_ptr,
                                   (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );
          if ( l_vec < l_vec_len - 1 || l_col_block < l_n_col_blocks - 1){
            libxsmm_ppc64le_instr_3( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_ADDI,
                                     l_col_ptr,
                                     l_col_ptr,
                                     l_databytes*i_stride );
          }
        }
        unsigned int l_v_reg[l_vec_len];
        for ( unsigned int l_i = 0; l_i < l_vec_len; ++l_i) {
          l_v_reg[l_i] = i_loaded_regs[l_col_block + (l_row_block*l_vec_len + l_i)*l_block_ld];
        }

        libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( io_generated_code,
                                                         io_reg_tracker,
                                                         l_v_reg );
      } else if ( i_datatype == LIBXSMM_DATATYPE_F64 ) {
        for ( unsigned int l_vec = 0; l_vec < l_vec_len; ++l_vec ) {
          unsigned int l_reg_idx = l_col_block + (l_row_block*l_vec_len + l_vec)*l_block_ld;
          libxsmm_ppc64le_instr_4( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                   i_loaded_regs[l_reg_idx],
                                   0,
                                   l_col_ptr,
                                   (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );
          libxsmm_ppc64le_instr_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_ADDI,
                                   l_col_ptr,
                                   l_col_ptr,
                                   l_databytes*i_stride );
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    /* increament if not last */
    if ( l_row_block < l_n_row_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               l_vec_len*l_databytes);
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_col_ptr );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load( libxsmm_generated_code * io_generated_code,
                                         libxsmm_datatype const   i_datatype,
                                         libxsmm_datatype const   i_comptype, /* currently unsuded */
                                         libxsmm_ppc64le_reg    * io_reg_tracker,
                                         unsigned int           * i_loaded_regs,
                                         unsigned int const       i_ptr_gpr,
                                         unsigned int const       i_n_rows,
                                         unsigned int const       i_n_cols,
                                         unsigned int const       i_stride ) {

  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / ( l_databytes * 8 );
  unsigned int l_n_col_blocks = i_n_cols;
  unsigned int l_n_row_blocks = i_n_rows / l_vec_len;
  unsigned int l_block_ld = ( i_n_rows + l_vec_len - 1 ) / l_vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_row_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_ptr,
                           i_ptr_gpr );

  /* load the fully packed part */
  for ( unsigned int l_col_block = 0; l_col_block < l_n_col_blocks; ++l_col_block ) {

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_ptr,
                             l_row_ptr,
                             l_ptr );

    for ( unsigned int l_row_block = 0; l_row_block < l_n_row_blocks; ++l_row_block ) {
      unsigned int l_reg_idx = l_col_block*l_block_ld + l_row_block;

      /* vector load */
      if ( i_datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                 i_loaded_regs[l_reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );

      } else if ( i_datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                 i_loaded_regs[l_reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );

      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      /* increment if not last */
      if ( l_row_block < l_n_row_blocks - 1) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_row_ptr,
                                 l_row_ptr,
                                 l_vec_len*l_databytes );
      }
    }

    /* increament if not last */
    if ( l_col_block < l_n_col_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               i_stride*l_databytes );
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_row_ptr );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load_bcast( libxsmm_generated_code * io_generated_code,
                                               libxsmm_datatype const   i_datatype,
                                               libxsmm_datatype const   i_comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * io_reg_tracker,
                                               unsigned int           * i_loaded_regs,
                                               unsigned int const       i_ptr_gpr,
                                               unsigned int const       i_n_rows,
                                               unsigned int const       i_n_cols,
                                               unsigned int const       i_stride ) {

  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (l_databytes * 8);
  unsigned int l_n_col_blocks = i_n_cols;
  unsigned int l_n_row_blocks = i_n_rows / l_vec_len;

  unsigned int l_ldr = i_n_rows;

  unsigned int l_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_col_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned l_v_scratch = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );

  /* local pointer for columns */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_col_ptr,
                           i_ptr_gpr );

  for ( unsigned int l_col_block = 0; l_col_block < l_n_col_blocks; ++l_col_block ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_col_ptr,
                             l_ptr,
                             l_col_ptr );
    for ( unsigned int l_row_block = 0; l_row_block < l_n_row_blocks; ++l_row_block ) {
      if ( i_datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                 l_v_scratch,
                                 0,
                                 l_ptr,
                                 (0x0020 & l_v_scratch) >> 5 );

        for ( unsigned int l_i = 0; l_i < l_vec_len; ++l_i ) {
          unsigned int l_reg_idx = l_i + l_row_block*l_vec_len + l_col_block*l_ldr;
          /* broadcast */
          libxsmm_ppc64le_instr_5( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_XXSPLTW,
                                   i_loaded_regs[l_reg_idx],
                                   l_i,
                                   l_v_scratch,
                                   (0x0020 & l_v_scratch) >> 5,
                                   (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );
        }
      } else if ( i_datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                 l_v_scratch,
                                 0,
                                 l_ptr,
                                 (0x0020 & l_v_scratch) >> 5 );
        for ( unsigned int l_i = 0; l_i < l_vec_len; ++l_i ) {
          unsigned int l_reg_idx = l_i + l_row_block*l_vec_len + l_col_block*l_ldr;
          /* broadcast */
          libxsmm_ppc64le_instr_7( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                                   i_loaded_regs[l_reg_idx],
                                   l_v_scratch,
                                   l_v_scratch,
                                   l_i*3, /* ie: 0b00 or 0b11 */
                                   (0x0020 & l_v_scratch) >> 5,
                                   (0x0020 & l_v_scratch) >> 5,
                                   (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      /* increament pointer if required */
      if ( l_row_block < l_n_row_blocks - 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_ptr,
                                 l_ptr,
                                 LIBXSMM_PPC64LE_VSR_WIDTH / 8 );
      }
    }

    /* increament column pointer if required */
    if ( l_col_block < l_n_col_blocks - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_col_ptr,
                               l_col_ptr,
                               i_stride*l_databytes );
    }
  }

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_v_scratch );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_col_ptr );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_store( libxsmm_generated_code * io_generated_code,
                                          libxsmm_datatype const   i_datatype,
                                          libxsmm_datatype const   i_comptype, /* currently unsuded */
                                          libxsmm_ppc64le_reg    * io_reg_tracker,
                                          unsigned int           * i_loaded_regs,
                                          unsigned int const       i_ptr_gpr,
                                          unsigned int const       i_n_rows,
                                          unsigned int const       i_n_cols,
                                          unsigned int const       i_stride ) {

  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (l_databytes * 8);
  unsigned int l_n_col_blocks = i_n_cols;
  unsigned int l_n_row_blocks = i_n_rows / l_vec_len;
  unsigned int l_block_ld = ( i_n_rows + l_vec_len - 1 ) / l_vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_row_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_ptr,
                           i_ptr_gpr );

  /* store the fully packed part */
  for ( unsigned int l_col_block = 0; l_col_block < l_n_col_blocks; ++l_col_block ) {

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_ptr,
                             l_row_ptr,
                             l_ptr );

    for ( unsigned int l_row_block = 0; l_row_block < l_n_row_blocks; ++l_row_block ) {
      unsigned int l_reg_idx = l_col_block*l_block_ld + l_row_block;
      /* vector store */
      if ( i_datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_STXVW4X,
                                 i_loaded_regs[l_reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );
      } else if ( i_datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_STXVD2X,
                                 i_loaded_regs[l_reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & i_loaded_regs[l_reg_idx]) >> 5 );
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      /* increment if not last */
      if ( l_row_block < l_n_row_blocks - 1) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_row_ptr,
                                 l_row_ptr,
                                 l_vec_len*l_databytes );
      }
    }

    /* increament if not last */
    if ( l_col_block < l_n_col_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               i_stride*l_databytes );
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_row_ptr );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_bcast( libxsmm_generated_code * io_generated_code,
                                              libxsmm_datatype const   i_datatype,
                                              unsigned int const       i_m,
                                              unsigned int const       i_n,
                                              unsigned int const       i_k,
                                              unsigned int           * i_a,
                                              unsigned int const       i_lda,
                                              unsigned int           * i_b,
                                              unsigned int const       i_ldb,
                                              unsigned int const       i_beta,
                                              unsigned int           * io_c,
                                              unsigned int const       i_ldc ) {

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      for ( unsigned int l_k = 0; l_k < i_k; ++l_k ) {
        if ( ( l_k == 0 ) & ( i_beta == 0 ) ) {
          if ( i_datatype == LIBXSMM_DATATYPE_F32 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMULSP,
                                     io_c[l_m + l_n*i_ldc],
                                     i_a[l_m + l_k*i_lda],
                                     i_b[l_k + l_n*i_ldb],
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5,
                                     (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                     (0x0020 & io_c[l_m + l_n*i_ldc]) >> 5 );
          } else if ( i_datatype == LIBXSMM_DATATYPE_F64 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMULDP,
                                     io_c[l_m + l_n*i_ldc],
                                     i_a[l_m + l_k*i_lda],
                                     i_b[l_k + l_n*i_ldb],
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5,
                                     (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                     (0x0020 & io_c[l_m + l_n*i_ldc]) >> 5 );
          } else {
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
            return;
          }
        } else {
          if ( i_datatype == LIBXSMM_DATATYPE_F32 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMADDASP,
                                     io_c[l_m + l_n*i_ldc],
                                     i_a[l_m + l_k*i_lda],
                                     i_b[l_k + l_n*i_ldb],
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5,
                                     (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                     (0x0020 & io_c[l_m + l_n*i_ldc]) >> 5 );
          } else if ( i_datatype == LIBXSMM_DATATYPE_F64 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMADDADP,
                                     io_c[l_m + l_n*i_ldc],
                                     i_a[l_m + l_k*i_lda],
                                     i_b[l_k + l_n*i_ldb],
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5,
                                     (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                     (0x0020 & io_c[l_m + l_n*i_ldc]) >> 5 );
          } else {
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
            return;
          }
        }
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code        * io_generated_code,
                                        libxsmm_gemm_descriptor const * i_xgemm_desc,
                                        libxsmm_ppc64le_reg           * io_reg_tracker,
                                        libxsmm_loop_label_tracker    * io_loop_labels,
                                        unsigned int                  * i_blocking,
                                        unsigned char const             i_a_ptr_gpr,
                                        unsigned char const             i_b_ptr_gpr,
                                        unsigned char const             i_c_ptr_gpr ) {

  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_compbytes = libxsmm_ppc64le_instr_bytes( io_generated_code, l_comptype );
  unsigned int l_v_len = ( LIBXSMM_PPC64LE_VSR_WIDTH / 8 ) / l_compbytes;

  unsigned int l_n_k_blocks = i_xgemm_desc->k / i_blocking[2];

  /* Local pointers registers for A and B */
  unsigned int l_a_ptr, l_b_ptr;

  if ( l_n_k_blocks > 1 ) {
    l_a_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             i_a_ptr_gpr,
                             l_a_ptr,
                             i_a_ptr_gpr );

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             i_b_ptr_gpr,
                             l_b_ptr,
                             i_b_ptr_gpr );
  } else {
    l_a_ptr = i_a_ptr_gpr;
    l_b_ptr = i_b_ptr_gpr;
  }

  /* Allocate A, B, and C registers */
  unsigned int l_n_a_reg = ( i_blocking[1] / l_v_len ) * i_blocking[2];
  unsigned int l_n_b_reg = i_blocking[0] * i_blocking[2];
  unsigned int l_n_c_reg = ( i_blocking[1] / l_v_len ) * i_blocking[0];

  unsigned l_a_reg[l_n_a_reg], l_b_reg[l_n_b_reg], l_c_reg[l_n_c_reg];

  for ( unsigned int l_i = 0; l_i < l_n_a_reg; ++l_i ) {
    l_a_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( unsigned int l_i = 0; l_i < l_n_b_reg; ++l_i ) {
    l_b_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( unsigned int l_i = 0; l_i < l_n_c_reg; ++l_i ) {
    l_c_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  unsigned int l_beta_zero = i_xgemm_desc->flags & 0x0004 >> 2;

  /* Load C if required */
  if ( !l_beta_zero ) {
    libxsmm_generator_gemm_vsx_mk_load( io_generated_code,
                                        l_c_datatype,
                                        l_comptype,
                                        io_reg_tracker,
                                        l_c_reg,
                                        i_c_ptr_gpr,
                                        i_blocking[1],
                                        i_blocking[0],
                                        i_xgemm_desc->ldc );
  }

  for ( unsigned int l_k_block = 0; l_k_block < l_n_k_blocks; ++l_k_block) {
    /* Load block of A */
    libxsmm_generator_gemm_vsx_mk_load( io_generated_code,
                                        l_a_datatype,
                                        l_comptype,
                                        io_reg_tracker,
                                        l_a_reg,
                                        l_a_ptr,
                                        i_blocking[1],
                                        i_blocking[2],
                                        i_xgemm_desc->lda );

    /* Load block of B and broadcast values to vector */
    libxsmm_generator_gemm_vsx_mk_load_bcast( io_generated_code,
                                              l_b_datatype,
                                              l_comptype,
                                              io_reg_tracker,
                                              l_b_reg,
                                              l_b_ptr,
                                              i_blocking[2],
                                              i_blocking[0],
                                              i_xgemm_desc->ldb );

    /* block FMA */
    unsigned int l_beta = ( ( l_k_block == 0 ) & ( l_beta_zero ) ) ? 0 : 1;
    libxsmm_generator_vsx_block_fma_b_bcast( io_generated_code,
                                             l_comptype,
                                             i_blocking[1] / l_v_len,
                                             i_blocking[0],
                                             i_blocking[2],
                                             l_a_reg,
                                             i_blocking[1] / l_v_len,
                                             l_b_reg,
                                             i_blocking[2],
                                             l_beta,
                                             l_c_reg,
                                             i_blocking[1] / l_v_len );

    if ( l_k_block < l_n_k_blocks - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_a_ptr,
                               l_a_ptr,
                               i_blocking[2]*i_xgemm_desc->lda*l_compbytes );

      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_b_ptr,
                               l_b_ptr,
                               i_blocking[2]*l_compbytes );
    }
  }

  /* Free A and B registers */
  for ( unsigned int l_i = 0; l_i < l_n_a_reg; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( unsigned int l_i = 0; l_i < l_n_b_reg; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }

  if ( l_n_k_blocks > 1 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_ptr );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_ptr );
  }

  /* Store result block in C */
  libxsmm_generator_gemm_vsx_mk_store( io_generated_code,
                                       l_c_datatype,
                                       l_comptype,
                                       io_reg_tracker,
                                       l_c_reg,
                                       i_c_ptr_gpr,
                                       i_blocking[1],
                                       i_blocking[0],
                                       i_xgemm_desc->ldc );

  /* Free C registers */
  for ( unsigned int l_i = 0; l_i < l_n_c_reg; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_c_reg[l_i] );
  }

  return;
}
