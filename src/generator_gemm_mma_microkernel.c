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

#include "generator_gemm_mma_microkernel.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_load_trans_f32_f32( libxsmm_generated_code   *io_generated_code,
                                                       libxsmm_ppc64le_reg      *io_reg_tracker,
                                                       libxsmm_ppc64le_blocking *i_blocking,
                                                       unsigned int const        i_a,
                                                       unsigned int const        i_m,
                                                       unsigned int const        i_n,
                                                       unsigned int const        i_lda,
                                                       unsigned int             *io_t,
                                                       unsigned int const        i_ldt ) {

  /* local copy of pointer */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_t_idx = (l_n / 4)*i_ldt + 4*l_m + ( l_n % 4 );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVW4X,
                               io_t[l_t_idx],
                               0,
                               l_a_row,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
      if ( l_m < i_m - 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, 16 );
      }
    }

    /* increament if not last */
    if ( l_n < i_n - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*4 );
    }
  }

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );

  /* Transpose blocks */
  for ( unsigned int l_n = 0; l_n < ( i_n + 3 ) / 4; ++l_n ) {
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_reg[4];
      for ( unsigned int l_i = 0; l_i < 4; ++l_i ) {
        l_reg[l_i] = io_t[l_n*i_ldt + 4*l_m + l_i];
      }
      libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( io_generated_code,
                                                       io_reg_tracker,
                                                       l_reg );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_store_trans_f32_f32( libxsmm_generated_code   *io_generated_code,
                                                        libxsmm_ppc64le_reg      *io_reg_tracker,
                                                        libxsmm_ppc64le_blocking *i_blocking,
                                                        unsigned int const        i_a,
                                                        unsigned int const        i_m,
                                                        unsigned int const        i_n,
                                                        unsigned int const        i_lda,
                                                        unsigned int             *io_t,
                                                        unsigned int const        i_ldt ) {

  /* Transpose blocks */
  for ( unsigned int l_n = 0; l_n < ( i_n + 3 ) / 4; ++l_n ) {
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_reg[4];
      for ( unsigned int l_i = 0; l_i < 4; ++l_i ) {
        l_reg[l_i] = io_t[l_n*i_ldt + 4*l_m + l_i];
      }
      libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( io_generated_code,
                                                       io_reg_tracker,
                                                       l_reg );
    }
  }

  /* local copy of pointer */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_t_idx = ((l_n + 3) / 4)*i_ldt + 4*l_m + ( l_n % 4 );

      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVW4X,
                               io_t[l_t_idx],
                               0,
                               l_a_row,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
      if ( l_m < i_m - 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, 16 );
      }
    }

    /* increament if not last */
    if ( l_n < i_n - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*4 );
    }
  }

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_load_f32_f32( libxsmm_generated_code   *io_generated_code,
                                                 libxsmm_ppc64le_reg      *io_reg_tracker,
                                                 libxsmm_ppc64le_blocking *i_blocking,
                                                 unsigned int const        i_a,
                                                 unsigned int const        i_m,
                                                 unsigned int const        i_n,
                                                 unsigned int const        i_lda,
                                                 unsigned int             *io_t,
                                                 unsigned int const        i_ldt ) {

  /* local copy of pointer */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );

    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_t_idx = l_n*i_ldt + l_m;

      /* vector load */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVW4X,
                               io_t[l_t_idx],
                               0,
                               l_a_row,
                               (0x0020 & io_t[l_t_idx]) >> 5 );

      /* increment if not last */
      if ( l_m < i_m - 1) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, 16 );
      }
    }

    /* increament if not last */
    if ( l_n < i_n - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*4 );
    }
  }

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_store_f32_f32( libxsmm_generated_code   *io_generated_code,
                                                  libxsmm_ppc64le_reg      *io_reg_tracker,
                                                  libxsmm_ppc64le_blocking *i_blocking,
                                                  unsigned int const        i_a,
                                                  unsigned int const        i_m,
                                                  unsigned int const        i_n,
                                                  unsigned int const        i_lda,
                                                  unsigned int             *io_t,
                                                  unsigned int const        i_ldt ) {

  /* local copy of pointer */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );

    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_t_idx = l_n*i_ldt + l_m;

      /* vector load */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVW4X,
                               io_t[l_t_idx],
                               0,
                               l_a_row,
                               (0x0020 & io_t[l_t_idx]) >> 5 );

      /* increment if not last */
      if ( l_m < i_m - 1) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, 16 );
      }
    }

    /* increament if not last */
    if ( l_n < i_n - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*4 );
    }
  }

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_load_acc_f32_f32( libxsmm_generated_code   * io_generated_code,
                                                     libxsmm_ppc64le_reg      * io_reg_tracker,
                                                     libxsmm_ppc64le_blocking * i_blocking,
                                                     unsigned int const         i_a,
                                                     unsigned int const         i_m,
                                                     unsigned int const         i_n,
                                                     unsigned int const         i_lda,
                                                     unsigned int             * io_t,
                                                     unsigned int const         i_ldt ) {
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_t_idx = l_m + ( l_n / 4 )*i_ldt ;
      unsigned int l_t = 4*io_t[l_t_idx] + ( l_n % 4 );

      libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_LXVW4X, l_t, 0, l_a_row, 0 );

      if ( l_m < i_m - 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, 16 );
      }
    }
    if ( ( l_n < i_n - 1 ) ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*4 );
    }
  }

  /* Move C from VSR to ACC */
  for ( unsigned int l_i = 0; l_i < i_blocking->n_acc_c ; ++l_i ) {
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMTACC, io_t[l_i] );
  }

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_mk_store_acc_f32_f32( libxsmm_generated_code   * io_generated_code,
                                                      libxsmm_ppc64le_reg      * io_reg_tracker,
                                                      libxsmm_ppc64le_blocking * i_blocking,
                                                      unsigned int const         i_a,
                                                      unsigned int const         i_m,
                                                      unsigned int const         i_n,
                                                      unsigned int const         i_lda,
                                                      unsigned int             * io_t,
                                                      unsigned int const         i_ldt ) {

  /* Move C from ACC to VSR */
  for ( unsigned int l_i = 0; l_i < i_blocking->n_acc_c ; ++l_i ) {
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMFACC, io_t[l_i] );
  }

  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      unsigned int l_t_idx = l_m + ( l_n / 4 )*i_ldt ;
      unsigned int l_t = 4*io_t[l_t_idx] + ( l_n % 4 );

      libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_STXVW4X, l_t, 0, l_a_row, 0 );

      if ( l_m < i_m - 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, 16 );
      }
    }
    if ( ( l_n < i_n - 1 ) ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*4 );
    }
  }

  /* free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f32_f32( libxsmm_generated_code * io_generated_code,
                                              unsigned int const       i_m,
                                              unsigned int const       i_n,
                                              unsigned int const       i_k,
                                              unsigned int           * i_a,
                                              unsigned int const       i_lda,
                                              unsigned int           * i_b,
                                              unsigned int const       i_ldb,
                                              unsigned int           * io_c,
                                              unsigned int const       i_ldc ) {
  for ( int l_n = 0; l_n < i_n; ++l_n ) {
    for ( int l_m = 0; l_m < i_m; ++l_m ) {
      for ( int l_k = 0 ; l_k < i_k; ++l_k ) {
        libxsmm_ppc64le_instr_5( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_XVF32GERPP,
                                 io_c[l_m + l_n*i_ldc],
                                 i_b[l_k + l_n*i_ldb],
                                 i_a[l_m + l_k*i_lda],
                                 (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                 (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_microkernel( libxsmm_generated_code        * io_generated_code,
                                        libxsmm_gemm_descriptor const * i_xgemm_desc,
                                        libxsmm_ppc64le_blocking      * i_blocking,
                                        libxsmm_ppc64le_reg           * io_reg_tracker,
                                        libxsmm_loop_label_tracker    * io_loop_labels,
                                        unsigned int                  * i_acc,
                                        unsigned char const             i_a_ptr_gpr,
                                        unsigned char const             i_b_ptr_gpr,
                                        unsigned char const             i_c_ptr_gpr ) {

  /* Local pointers registers for A and B */
  unsigned int l_a_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_b_ptr = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

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

  /* Allocate A and B registers */
  unsigned int l_a_reg[i_blocking->n_reg_a], l_b_reg[i_blocking->n_reg_b];
  for ( int l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    l_a_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( int l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
    l_b_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  /* Load C into accumulators or zero */
  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x0004 ) >> 2;

  if ( l_beta_zero ) {
    for ( unsigned int l_i = 0; l_i < i_blocking->n_acc_c; ++l_i ) {
      libxsmm_ppc64le_instr_1( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXSETACCZ,
                               i_acc[l_i] );
    }
  } else {
    libxsmm_generator_gemm_mma_mk_load_acc_f32_f32( io_generated_code,
                                                    io_reg_tracker,
                                                    i_blocking,
                                                    i_c_ptr_gpr,
                                                    i_blocking->block_m / 4,
                                                    i_blocking->block_n,
                                                    i_xgemm_desc->ldc,
                                                    i_acc,
                                                    i_blocking->reg_ldc );
  }

  for ( unsigned int l_k_block = 0; l_k_block < i_blocking->n_block_k_full; ++l_k_block ) {

    /* Load block of A */
    libxsmm_generator_gemm_mma_mk_load_f32_f32( io_generated_code,
                                                io_reg_tracker,
                                                i_blocking,
                                                l_a_ptr,
                                                i_blocking->block_m / 4,
                                                i_blocking->block_k,
                                                i_xgemm_desc->lda,
                                                l_a_reg,
                                                i_blocking->reg_lda );

    /* Load block of B and transpose in place */
    libxsmm_generator_gemm_mma_mk_load_trans_f32_f32( io_generated_code,
                                                      io_reg_tracker,
                                                      i_blocking,
                                                      l_b_ptr,
                                                      i_blocking->block_k / 4,
                                                      i_blocking->block_n,
                                                      i_xgemm_desc->ldb,
                                                      l_b_reg,
                                                      i_blocking->reg_ldb );

    /* GER call */
    libxsmm_generator_mma_block_ger_f32_f32( io_generated_code,
                                             i_blocking->block_m / 4,
                                             i_blocking->block_n / 4,
                                             i_blocking->block_k,
                                             l_a_reg,
                                             i_blocking->block_m / 4,
                                             l_b_reg,
                                             i_blocking->block_k,
                                             i_acc,
                                             i_blocking->block_m / 4 );

    /* Increment pointers if not last */
    if ( l_k_block < i_blocking->n_block_k_full - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_a_ptr,
                               l_a_ptr,
                               i_blocking->block_k*i_xgemm_desc->lda*i_blocking->comp_bytes );

      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_b_ptr,
                               l_b_ptr,
                               i_blocking->block_k*i_blocking->comp_bytes );
    }
  }

  /* Free A and B registers */
  for ( int l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( int l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }

  /* Store blocks of C */
  libxsmm_generator_gemm_mma_mk_store_acc_f32_f32( io_generated_code,
                                                   io_reg_tracker,
                                                   i_blocking,
                                                   i_c_ptr_gpr,
                                                   i_blocking->block_m / 4,
                                                   i_blocking->block_n,
                                                   i_xgemm_desc->ldc,
                                                   i_acc,
                                                   i_blocking->reg_ldc );

  return;
}
