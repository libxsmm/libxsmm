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
                                               libxsmm_datatype const   datatype,
                                               libxsmm_datatype const   comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * reg_tracker,
                                               unsigned int           * loaded_regs,
                                               unsigned int             i_ptr_gpr,
                                               unsigned int             n_rows,
                                               unsigned int             n_cols,
                                               unsigned int             stride ) {

  unsigned int databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, datatype );
  unsigned int vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (databytes * 8);
  unsigned int n_col_blocks = n_cols / vec_len;
  unsigned int n_row_blocks = n_rows / vec_len;
  unsigned int block_ld = ( n_cols + vec_len - 1 ) / vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_col_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_ptr,
                           i_ptr_gpr );

  if( databytes*stride > (unsigned int)(0xffff) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* load the fully packed part */
  for ( unsigned int row_block = 0; row_block < n_row_blocks; ++row_block ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_ptr,
                             l_col_ptr,
                             l_ptr );

    for ( unsigned int col_block = 0; col_block < n_col_blocks; ++col_block ) {
      /* vector load */
      if ( datatype == LIBXSMM_DATATYPE_F32 ) {
        for ( unsigned int vec = 0; vec < vec_len; ++vec ) {
          unsigned int reg_idx = col_block + (row_block*vec_len + vec)*block_ld;
          libxsmm_ppc64le_instr_4( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                   loaded_regs[reg_idx],
                                   0,
                                   l_col_ptr,
                                   (0x0020 & loaded_regs[reg_idx]) >> 5 );
          if ( vec < vec_len - 1 || col_block < n_col_blocks - 1){
            libxsmm_ppc64le_instr_3( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_ADDI,
                                     l_col_ptr,
                                     l_col_ptr,
                                     databytes*stride );
          }
        }
        unsigned int v_reg[vec_len];
        for ( unsigned int i = 0; i < vec_len; ++i) {
          v_reg[i] = loaded_regs[col_block + (row_block*vec_len + i)*block_ld];
        }

        libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( io_generated_code,
                                                         reg_tracker,
                                                         v_reg );
      } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
        for ( unsigned int vec = 0; vec < vec_len; ++vec ) {
          unsigned int reg_idx = col_block + (row_block*vec_len + vec)*block_ld;
          libxsmm_ppc64le_instr_4( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                   loaded_regs[reg_idx],
                                   0,
                                   l_col_ptr,
                                   (0x0020 & loaded_regs[reg_idx]) >> 5 );
          libxsmm_ppc64le_instr_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_ADDI,
                                   l_col_ptr,
                                   l_col_ptr,
                                   databytes*stride );
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    /* increament if not last */
    if ( row_block < n_row_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               vec_len*databytes);
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_col_ptr );
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load( libxsmm_generated_code * io_generated_code,
                                         libxsmm_datatype const   datatype,
                                         libxsmm_datatype const   comptype, /* currently unsuded */
                                         libxsmm_ppc64le_reg    * reg_tracker,
                                         unsigned int           * loaded_regs,
                                         unsigned int             i_ptr_gpr,
                                         unsigned int             n_rows,
                                         unsigned int             n_cols,
                                         unsigned int             stride ) {

  unsigned int databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, datatype );
  unsigned int vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (databytes * 8);
  unsigned int n_col_blocks = n_cols;
  unsigned int n_row_blocks = n_rows / vec_len;
  unsigned int block_ld = ( n_rows + vec_len - 1 ) / vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_row_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_ptr,
                           i_ptr_gpr );

  /* load the fully packed part */
  for ( unsigned int col_block = 0; col_block < n_col_blocks; ++col_block ) {

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_ptr,
                             l_row_ptr,
                             l_ptr );

    for ( unsigned int row_block = 0; row_block < n_row_blocks; ++row_block ) {
      unsigned int reg_idx = col_block*block_ld + row_block;

      /* vector load */
      if ( datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                 loaded_regs[reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & loaded_regs[reg_idx]) >> 5 );

      } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                 loaded_regs[reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & loaded_regs[reg_idx]) >> 5 );

      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      /* increment if not last */
      if ( row_block < n_row_blocks - 1) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_row_ptr,
                                 l_row_ptr,
                                 vec_len*databytes );
      }
    }

    /* increament if not last */
    if ( col_block < n_col_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               stride*databytes );
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_row_ptr );
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_load_bcast( libxsmm_generated_code * io_generated_code,
                                               libxsmm_datatype const   datatype,
                                               libxsmm_datatype const   comptype, /* currently unsuded */
                                               libxsmm_ppc64le_reg    * reg_tracker,
                                               unsigned int           * loaded_regs,
                                               unsigned int             i_ptr_gpr,
                                               unsigned int             n_rows,
                                               unsigned int             n_cols,
                                               unsigned int             stride ) {

  unsigned int databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, datatype );
  unsigned int vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (databytes * 8);
  unsigned int n_col_blocks = n_cols;
  unsigned int n_row_blocks = n_rows / vec_len;

  unsigned int ldr = n_rows;

  unsigned int l_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_col_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned v_scratch = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_VSR );

  /* local pointer for columns */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_col_ptr,
                           i_ptr_gpr );

  for ( unsigned int col_block = 0; col_block < n_col_blocks; ++col_block ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_col_ptr,
                             l_ptr,
                             l_col_ptr );
    for ( unsigned int row_block = 0; row_block < n_row_blocks; ++row_block ) {
      if ( datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                 v_scratch,
                                 0,
                                 l_ptr,
                                 (0x0020 & v_scratch) >> 5 );

        for ( unsigned int i = 0; i < vec_len; ++i ) {
          unsigned int reg_idx = i + row_block*vec_len + col_block*ldr;
          /* broadcast */
          libxsmm_ppc64le_instr_5( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_XXSPLTW,
                                   loaded_regs[reg_idx],
                                   i,
                                   v_scratch,
                                   (0x0020 & v_scratch) >> 5,
                                   (0x0020 & loaded_regs[reg_idx]) >> 5 );
        }
      } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                 v_scratch,
                                 0,
                                 l_ptr,
                                 (0x0020 & v_scratch) >> 5 );
        for ( unsigned int i = 0; i < vec_len; ++i ) {
          unsigned int reg_idx = i + row_block*vec_len + col_block*ldr;
          /* broadcast */
          libxsmm_ppc64le_instr_7( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                                   loaded_regs[reg_idx],
                                   v_scratch,
                                   v_scratch,
                                   i*3, /* ie: 0b00 or 0b11 */
                                   (0x0020 & v_scratch) >> 5,
                                   (0x0020 & v_scratch) >> 5,
                                   (0x0020 & loaded_regs[reg_idx]) >> 5 );
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      /* increament pointer if required */
      if ( row_block < n_row_blocks - 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_ptr,
                                 l_ptr,
                                 LIBXSMM_PPC64LE_VSR_WIDTH / 8 );
      }
    }

    /* increament column pointer if required */
    if ( col_block < n_col_blocks - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_col_ptr,
                               l_col_ptr,
                               stride*databytes );
    }
  }

  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_VSR, v_scratch );
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_col_ptr );
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vsx_mk_store( libxsmm_generated_code * io_generated_code,
                                          libxsmm_datatype const   datatype,
                                          libxsmm_datatype const   comptype, /* currently unsuded */
                                          libxsmm_ppc64le_reg    * reg_tracker,
                                          unsigned int           * loaded_regs,
                                          unsigned int             i_ptr_gpr,
                                          unsigned int             n_rows,
                                          unsigned int             n_cols,
                                          unsigned int             stride ) {

  unsigned int databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, datatype );
  unsigned int vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (databytes * 8);
  unsigned int n_col_blocks = n_cols;
  unsigned int n_row_blocks = n_rows / vec_len;
  unsigned int block_ld = ( n_rows + vec_len - 1 ) / vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_row_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           i_ptr_gpr,
                           l_ptr,
                           i_ptr_gpr );

  /* store the fully packed part */
  for ( unsigned int col_block = 0; col_block < n_col_blocks; ++col_block ) {

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_OR,
                             l_ptr,
                             l_row_ptr,
                             l_ptr );

    for ( unsigned int row_block = 0; row_block < n_row_blocks; ++row_block ) {
      unsigned int reg_idx = col_block*block_ld + row_block;
      /* vector store */
      if ( datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_STXVW4X,
                                 loaded_regs[reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & loaded_regs[reg_idx]) >> 5 );
      } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_STXVD2X,
                                 loaded_regs[reg_idx],
                                 0,
                                 l_row_ptr,
                                 (0x0020 & loaded_regs[reg_idx]) >> 5 );
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }

      /* increment if not last */
      if ( row_block < n_row_blocks - 1) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_row_ptr,
                                 l_row_ptr,
                                 vec_len*databytes );
      }
    }

    /* increament if not last */
    if ( col_block < n_col_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               stride*databytes );
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_row_ptr );
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_bcast( libxsmm_generated_code * io_generated_code,
                                              libxsmm_datatype         datatype,
                                              unsigned int             m,
                                              unsigned int             n,
                                              unsigned int             k,
                                              unsigned int           * a,
                                              unsigned int             lda,
                                              unsigned int           * b,
                                              unsigned int             ldb,
                                              unsigned int             beta,
                                              unsigned int           * c,
                                              unsigned int             ldc ) {

  for ( unsigned int i_n = 0; i_n < n; ++i_n ) {
    for ( unsigned int i_m = 0; i_m < m; ++i_m ) {
      for ( unsigned int i_k = 0; i_k < k; ++i_k ) {
        if ( ( i_k == 0 ) & ( beta == 0 ) ) {
          if ( datatype == LIBXSMM_DATATYPE_F32 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMULSP,
                                     c[i_m + i_n*ldc],
                                     a[i_m + i_k*lda],
                                     b[i_k + i_n*ldb],
                                     (0x0020 & a[i_m + i_k*lda]) >> 5,
                                     (0x0020 & b[i_k + i_n*ldb]) >> 5,
                                     (0x0020 & c[i_m + i_n*ldc]) >> 5 );
          } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMULDP,
                                     c[i_m + i_n*ldc],
                                     a[i_m + i_k*lda],
                                     b[i_k + i_n*ldb],
                                     (0x0020 & a[i_m + i_k*lda]) >> 5,
                                     (0x0020 & b[i_k + i_n*ldb]) >> 5,
                                     (0x0020 & c[i_m + i_n*ldc]) >> 5 );
          } else {
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
            return;
          }
        } else {
          if ( datatype == LIBXSMM_DATATYPE_F32 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMADDASP,
                                     c[i_m + i_n*ldc],
                                     a[i_m + i_k*lda],
                                     b[i_k + i_n*ldb],
                                     (0x0020 & a[i_m + i_k*lda]) >> 5,
                                     (0x0020 & b[i_k + i_n*ldb]) >> 5,
                                     (0x0020 & c[i_m + i_n*ldc]) >> 5 );
          } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
            libxsmm_ppc64le_instr_6( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XVMADDADP,
                                     c[i_m + i_n*ldc],
                                     a[i_m + i_k*lda],
                                     b[i_k + i_n*ldb],
                                     (0x0020 & a[i_m + i_k*lda]) >> 5,
                                     (0x0020 & b[i_k + i_n*ldb]) >> 5,
                                     (0x0020 & c[i_m + i_n*ldc]) >> 5 );
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
                                        libxsmm_ppc64le_reg           * reg_tracker,
                                        libxsmm_loop_label_tracker    * loop_labels,
                                        unsigned int                  * blocking,
                                        unsigned char const             i_a_ptr_gpr,
                                        unsigned char const             i_b_ptr_gpr,
                                        unsigned char const             i_c_ptr_gpr ) {

  libxsmm_datatype a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int bytes = libxsmm_ppc64le_instr_bytes( io_generated_code, comptype );
  unsigned int v_len = ( LIBXSMM_PPC64LE_VSR_WIDTH / 8 ) / bytes;

  unsigned int n_k_blocks = i_xgemm_desc->k / blocking[2];

  /* Local pointers registers for A and B */
  unsigned int l_a_ptr, l_b_ptr;

  if ( n_k_blocks > 1 ) {
    l_a_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );

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
  unsigned int n_a_reg = (blocking[1] / v_len ) * blocking[2];
  unsigned int n_b_reg = blocking[0] * blocking[2];
  unsigned int n_c_reg = ( blocking[1] / v_len ) * blocking[0];

  unsigned a_reg[n_a_reg], b_reg[n_b_reg], c_reg[n_c_reg];

  for ( unsigned int i = 0; i < n_a_reg; ++i ) {
    a_reg[i] = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( unsigned int i = 0; i < n_b_reg; ++i ) {
    b_reg[i] = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( unsigned int i = 0; i < n_c_reg; ++i ) {
    c_reg[i] = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_VSR );
  }


  unsigned int beta_zero = i_xgemm_desc->flags & 0x0004 >> 2;

  /* Load C if required */
  if ( !beta_zero ) {
    libxsmm_generator_gemm_vsx_mk_load( io_generated_code,
                                        c_datatype,
                                        comptype,
                                        reg_tracker,
                                        c_reg,
                                        i_c_ptr_gpr,
                                        blocking[1],
                                        blocking[0],
                                        i_xgemm_desc->ldc );
  }

  for ( unsigned int k_block = 0; k_block < n_k_blocks; ++k_block) {
    /* Load block of A */
    libxsmm_generator_gemm_vsx_mk_load( io_generated_code,
                                        a_datatype,
                                        comptype,
                                        reg_tracker,
                                        a_reg,
                                        l_a_ptr,
                                        blocking[1],
                                        blocking[2],
                                        i_xgemm_desc->lda );

    /* Load block of B and broadcast values to vector */
    libxsmm_generator_gemm_vsx_mk_load_bcast( io_generated_code,
                                              b_datatype,
                                              comptype,
                                              reg_tracker,
                                              b_reg,
                                              l_b_ptr,
                                              blocking[2],
                                              blocking[0],
                                              i_xgemm_desc->ldb );

    /* block FMA */
    unsigned int beta = ( ( k_block == 0 ) & ( beta_zero ) ) ? 0 : 1;
    libxsmm_generator_vsx_block_fma_b_bcast( io_generated_code,
                                             comptype,
                                             blocking[1] / v_len,
                                             blocking[0],
                                             blocking[2],
                                             a_reg,
                                             blocking[1] / v_len,
                                             b_reg,
                                             blocking[2],
                                             beta,
                                             c_reg,
                                             blocking[1] / v_len );

    if ( k_block < n_k_blocks - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_a_ptr,
                               l_a_ptr,
                               blocking[2]*i_xgemm_desc->lda*bytes );
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_b_ptr,
                               l_b_ptr,
                               blocking[2]*bytes );
    }
  }

  /* Free A and B registers */
  for ( unsigned int i = 0; i < n_a_reg; ++i ) {
    libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_VSR, a_reg[i] );
  }
  for ( unsigned int i = 0; i < n_b_reg; ++i ) {
    libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_VSR, b_reg[i] );
  }

  if ( n_k_blocks > 1 ) {
    libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_ptr );
    libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_ptr );
  }

  /* Store result block in C */
  libxsmm_generator_gemm_vsx_mk_store( io_generated_code,
                                       c_datatype,
                                       comptype,
                                       reg_tracker,
                                       c_reg,
                                       i_c_ptr_gpr,
                                       blocking[1],
                                       blocking[0],
                                       i_xgemm_desc->ldc );

  /* Free C registers */
  for ( unsigned int i = 0; i < n_c_reg; ++i ) {
    libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_VSR, c_reg[i] );
  }

  return;
}
