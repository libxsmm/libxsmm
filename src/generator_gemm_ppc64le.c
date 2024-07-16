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
#include "generator_ppc64le_instructions.h"


/*
void libxsmm_gnerator_gemm_ppc64le_vsx_load_trans( libxsmm_generated_code * io_generated_code,
                                                   libxsmm_datatype const   datatype,
                                                   libxsmm_datatype const   comptype,
                                                   libxsmm_ppc64le_reg    * reg_tracker,
                                                   unsigned char          * loaded_regs,
                                                   unsigned char            i_ptr_gpr,
                                                   unsigned int             n_rows,
                                                   unsigned int             n_cols,
                                                   unsigned int             stride ) {

  unsigned int databytes = libxsmm_generator_gemm_ppc64le_bytes( io_generated_code, datatype );
  unsigned int vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / databytes;
  unsigned int n_col_blocks =0;

  unsigned int l_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ORI,
                           l_ptr,
                           i_ptr_gpr,
                           0 );

  for ( unsigned int col_block = 0; col_block < n_col_blocks; ++col_block ) {
    for ( unsigned int row_block = 0; row_block < n_row_blocks; ++row_block ) {
       if ( datatype == LIBXSMM_DATATYPE_F32 ) {
         for ( unsigned int i_vec = 0; i_vec < vec_len; ++i_vec) {
           libxsmm_ppc64le_instr_4( io_generated_code,
                                    LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                    l_vsr,
                                    0,
                                    l_ptr,
                                    (0x0020 & l_vsr) >> 5 );
         }


          transpose the block just loaded
         libxsmm_ppc64le_instr_6( io_generated_code,
                                  LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0 );

       } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {block ) {
         for ( unsigned int i_vec = 0; i_vec < vec_len; ++i_vec) {
           libxsmm_ppc64le_instr_4( io_generated_code,
                                    LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                    l_vsr,
                                    0,
                                    l_ptr,
                                    (0x0020 & l_vsr) >> 5 );
         }
       } else if ( datatype == LIBXSMM_DATAYPE_F16 || datatype == LIBXSMM_DATAYPE_B16 ) {
         for ( unsigned int i_vec = 0; i_vec < vec_len; ++i_vec) {
           libxsmm_ppc64le_instr_4( io_generated_code,
                                    LIBXSMM_PPC64LE_INSTR_LXVH8X,
                                    l_vsr,
                                    0,
                                    l_ptr,
                                    (0x0020 & l_vsr) >> 5 );
         }
      }
      if ( datatype == LIBXSMM_DATATYPE_F32 ) {

      } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                 l_vsr,
                                 0,
                                 l_ptr,
                                 (0x0020 & l_vsr) >> 5 );
      } else if ( datatype == LIBXSMM_DATAYPE_F16 || datatype == LIBXSMM_DATAYPE_B16 ) {

      }
    }
  }

  libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}
*/


LIBXSMM_API_INTERN
void libxsmm_gnerator_gemm_ppc64le_vsx_load( libxsmm_generated_code * io_generated_code,
                                             libxsmm_datatype const   datatype,
                                             libxsmm_datatype const   comptype, /* currently unsuded */
                                             libxsmm_ppc64le_reg    * reg_tracker,
                                             unsigned char          * loaded_regs,
                                             unsigned char            i_ptr_gpr,
                                             unsigned int             n_rows,
                                             unsigned int             n_cols,
                                             unsigned int             stride ) {

  unsigned int databytes = libxsmm_generator_gemm_ppc64le_bytes( io_generated_code, datatype );
  unsigned int vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / (databytes * 8);
  unsigned int n_col_blocks = n_cols;
  unsigned int n_row_blocks = n_rows / vec_len;
  unsigned int block_ld = ( n_rows + vec_len - 1 ) / vec_len;

  /* local copy of pointer */
  unsigned int l_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_row_ptr = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           l_ptr,
                           i_ptr_gpr,
                           0 );

  /* load the fully packed part */
  for ( unsigned int col_block = 0; col_block < n_col_blocks; ++col_block ) {
    for ( unsigned int row_block = 0; row_block < n_row_blocks; ++row_block ) {
      unsigned int reg_idx = col_block*block_ld + row_block;

      /* row pointer is only needed if row blocking requires it */
      unsigned int row_ptr = (n_row_blocks == 1) ? l_ptr : l_row_ptr;
      if (n_row_blocks > 1 ) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 l_row_ptr,
                                 l_ptr,
                                 row_block*vec_len*databytes );
      }

      /* vector load */
      if ( datatype == LIBXSMM_DATATYPE_F32 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVW4X,
                                 loaded_regs[reg_idx],
                                 0,
                                 row_ptr,
                                 (0x0020 & loaded_regs[reg_idx]) >> 5 );

      } else if ( datatype == LIBXSMM_DATATYPE_F64 ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVD2X,
                                 loaded_regs[reg_idx],
                                 0,
                                 row_ptr,
                                 (0x0020 & loaded_regs[reg_idx]) >> 5 );

      } else if ( datatype == LIBXSMM_DATATYPE_F16 || datatype == LIBXSMM_DATATYPE_BF16 ) {
         libxsmm_ppc64le_instr_4( io_generated_code,
                                  LIBXSMM_PPC64LE_INSTR_LXVH8X,
                                  loaded_regs[reg_idx],
                                  0,
                                  row_ptr,
                                  (0x0020 & loaded_regs[reg_idx]) >> 5 );
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    /* increament if not last */
    if ( col_block < n_col_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_ptr,
                               l_ptr,
                               stride*databytes);
    }
  }

  /* todo: the remainders */

  /* free GPR */
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_row_ptr );
  libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, l_ptr );
}


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
void libxsmm_generator_gemm_ppc64le_microkernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                     libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                     unsigned char                   i_gpr_a,
                                                     unsigned char                   i_gpr_b,
                                                     unsigned char                   i_gpr_c,
                                                     unsigned int                    i_m_blocking,
                                                     unsigned int                    i_n_blocking,
                                                     unsigned int                    i_k_blocking ) {
  unsigned int l_bytes_per_val = 4;
  unsigned int l_vector_length = 4;

  unsigned char l_gpr_a = LIBXSMM_PPC64LE_GPR_R21;
  unsigned char l_gpr_b = LIBXSMM_PPC64LE_GPR_R22;
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
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ORI,
                           l_gpr_a,
                           i_gpr_a,
                           0 );

  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ORI,
                           l_gpr_b,
                           i_gpr_b,
                           0 );

  /* load accumulator block */
  unsigned char l_vsr_c_first = 0;
  unsigned char l_n_vsr_c = 0;

  l_n_vsr_c = libxsmm_generator_gemm_ppc64le_load_vsx( io_generated_code,
                                                       l_m_blocking_full,
                                                       i_n_blocking,
                                                       l_remainder_size,
                                                       l_stride_c,
                                                       l_bytes_per_val == 4 ? 0 : 1,
                                                       i_gpr_c,
                                                       l_gpr_scratch,
                                                       l_vsr_c_first );

  unsigned char l_vsr_a_first = l_vsr_c_first + l_n_vsr_c;

  /* iterate over k if necessary */
  if( l_k_unroll != i_k_blocking ) {
    libxsmm_reset_loop_label_tracker( &l_loop_labels );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_k,
                             0,
                             i_k_blocking / l_k_unroll );
    /* use count register for inner-loop */
    libxsmm_ppc64le_instr_2( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_MTSPR,
                             l_gpr_k,
                             LIBXSMM_PPC64LE_SPR_CTR );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code,
                                                    &l_loop_labels );
  }

  /* unrolled k */
  for( unsigned short l_k = 0; l_k < l_k_unroll; l_k++ ) {
    /* load (partial) column of A */
    unsigned int l_n_vsr_a = 0;
    l_n_vsr_a = libxsmm_generator_gemm_ppc64le_load_vsx( io_generated_code,
                                                               l_m_blocking_full,
                                                               1,
                                                               l_remainder_size,
                                                               l_stride_a,
                                                               l_bytes_per_val == 4 ? 0 : 1,
                                                               l_gpr_a,
                                                               l_gpr_scratch,
                                                               l_vsr_a_first );

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_a,
                             l_gpr_a,
                             l_stride_a );

    unsigned char l_vsr_b = l_vsr_a_first + l_n_vsr_a;

    /* prepare the GPRs for the loads of B */
    for( unsigned short l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_gpr_scratch[l_n],
                               l_gpr_b,
                               l_stride_b*l_n );
    }
    if( l_k != i_k_blocking-1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_gpr_b,
                               l_gpr_b,
                               l_bytes_per_val );
    }

    for( unsigned short l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* bcast entry of B */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVWSX,
                               l_vsr_b,
                               0,
                               l_gpr_scratch[l_n],
                               (0x0020 & l_vsr_b) >> 5 );

      /* perform FMAs */
      unsigned int l_n_ops = l_m_blocking_full;
      if( l_remainder_size > 1 ) {
        l_n_ops++;
      }
      for( unsigned short l_op = 0; l_op < l_n_ops; l_op++ ) {
        unsigned int l_vsr_c_off = l_n * l_n_ops;
        l_vsr_c_off += l_op;

        unsigned int _xt = l_vsr_c_first + l_vsr_c_off;
        unsigned int _xa = l_vsr_a_first + l_op;
        unsigned int _xb = l_vsr_b;
        libxsmm_ppc64le_instr_6( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_XVMADDASP,
                                 _xt,
                                 _xa,
                                 _xb,
                                 (0x0020 & _xt) >> 5,
                                 (0x0020 & _xa) >> 5,
                                 (0x0020 & _xb) >> 5 );
      }
    }
  }

  /* end of k-loop */
  if( l_k_unroll != i_k_blocking ) {
    libxsmm_ppc64le_instr_cond_jump_back_to_label_ctr( io_generated_code,
                                                       &l_loop_labels );
  }

  /* store accumulator block */
  l_n_vsr_c = libxsmm_generator_gemm_ppc64le_store_vsx( io_generated_code,
                                                             l_m_blocking_full,
                                                             i_n_blocking,
                                                             l_remainder_size,
                                                             l_stride_c,
                                                             l_bytes_per_val == 4 ? 0 : 1,
                                                             i_gpr_c,
                                                             l_gpr_scratch,
                                                             l_vsr_c_first );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_m_loop_vsx( libxsmm_generated_code        * io_generated_code,
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
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_m,
                             0,
                             l_n_max_blocks_m );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code,
                                                    &l_loop_labels );
  }

  if( l_n_max_blocks_m > 0 ) {
    /* micro-kernel for maximum sizes block w.r.t. m */
    libxsmm_generator_gemm_ppc64le_microkernel_vsx( io_generated_code,
                                                    i_xgemm_desc,
                                                    i_gpr_a,
                                                    i_gpr_b,
                                                    i_gpr_c,
                                                    i_max_block_m,
                                                    i_n,
                                                    i_xgemm_desc->k );

    /* increase A and B pointers */
    unsigned int l_n_bytes_per_block_m = i_max_block_m*i_bytes_per_val;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_gpr_a,
                             i_gpr_a,
                             l_n_bytes_per_block_m );

    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_gpr_c,
                             i_gpr_c,
                             l_n_bytes_per_block_m );
  }

  /* end of max-size m-block loop */
  if( l_n_max_blocks_m > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_m,
                             l_gpr_m,
                             -1 );

    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   l_gpr_m,
                                                   &l_loop_labels );
  }

  /* generate microkernel for remainder if necessary */
  unsigned int l_m_remaining = i_xgemm_desc->m % i_max_block_m;

  if( l_m_remaining > 0 ) {
    libxsmm_generator_gemm_ppc64le_microkernel_vsx( io_generated_code,
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
void libxsmm_generator_gemm_ppc64le_kernel_vsx( libxsmm_generated_code        * io_generated_code,
                                                libxsmm_gemm_descriptor const * i_xgemm_desc ) {
  /* derive if FP32 or FP64 and respective vector length */
  unsigned int l_bytes_per_val = 0;
  unsigned int l_vector_length = 0;

  if( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 4;
    l_vector_length = 4;
  }
  else if( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype ) ) {
    l_bytes_per_val = 8;
    l_vector_length = 2;
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* loop labels */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* derive maximum block-size bound by the number of available registers */
  unsigned int l_max_block_m  = i_xgemm_desc->m;
               l_max_block_m += l_vector_length - 1;
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
  libxsmm_ppc64le_instr_open_stream( io_generated_code,
                                     l_gprMax,
                                     l_fprMax,
                                     l_vsrMax );

  /* function-input holding A's, B's and C's location in memory */
  unsigned char l_gpr_a = LIBXSMM_PPC64LE_GPR_R3;
  unsigned char l_gpr_b = LIBXSMM_PPC64LE_GPR_R4;
  unsigned char l_gpr_c = LIBXSMM_PPC64LE_GPR_R5;

  /* GPR mapping */
  unsigned char l_gpr_scratch = 23;
  unsigned char l_gpr_n = 24;
  unsigned int l_n_max_blocks_n = i_xgemm_desc->n / l_max_block_n;

  /* loop over max-sized blocks in N-dimension */
  if( l_n_max_blocks_n > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_n,
                             0,
                             l_n_max_blocks_n );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code,
                                                    &l_loop_labels );
  }

  /* generate M-loop for max-sized block in N-dimension */
  if( l_n_max_blocks_n > 0 ) {
    libxsmm_generator_gemm_ppc64le_m_loop_vsx( io_generated_code,
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
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_a,
                             l_gpr_a,
                             l_off );

    /* B */
    l_off = i_xgemm_desc->ldb * l_max_block_n;
    l_off *= l_bytes_per_val;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_b,
                             l_gpr_b,
                             l_off );

    /* C */
    l_off = i_xgemm_desc->ldc * l_max_block_n;
    if( l_n_max_blocks_m > 0 ) {
      l_off -= l_n_max_blocks_m*l_max_block_m;
    }
    l_off *= l_bytes_per_val;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_c,
                             l_gpr_c,
                             l_off );
  }


  /* end of n-loop */
  if( l_n_max_blocks_n > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_gpr_n,
                             l_gpr_n,
                             -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   l_gpr_n,
                                                   &l_loop_labels );
  }

  /* generate M-loop for remaining entries in N-dimension */
  unsigned int l_n_remaining = i_xgemm_desc->n % l_max_block_n;
  if( l_n_remaining > 0 ) {
    libxsmm_generator_gemm_ppc64le_m_loop_vsx( io_generated_code,
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
  libxsmm_ppc64le_instr_close_stream( io_generated_code,
                                      l_gprMax,
                                      l_fprMax,
                                      l_vsrMax );
  return;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_ppc64le_bytes( libxsmm_generated_code * io_generated_code,
                                                   libxsmm_datatype const   datatype ) {
  unsigned int bytes = 0;

  switch ( datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      bytes = 4;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      bytes = 8;
    } break;
    case LIBXSMM_DATATYPE_F16: {
      bytes = 2;
    } break;
    case LIBXSMM_DATATYPE_BF16: {
      bytes = 2;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return -1;
    }
  }

  return bytes;
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_vsx_m_loop( libxsmm_generated_code        * io_generated_code,
                                                libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                libxsmm_ppc64le_reg           * reg_tracker,
                                                unsigned int                  * blocking,
                                                unsigned char                   i_a_ptr_gpr,
                                                unsigned char                   i_b_ptr_gpr,
                                                unsigned char                   i_c_ptr_gpr ) {




  return;
}


/*

n m k

A[m * k] * B[k * n] = C[m * n]

*/

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel_vsx_wt( libxsmm_generated_code        * io_generated_code,
                                                   libxsmm_gemm_descriptor const * i_xgemm_desc,
                                                   libxsmm_ppc64le_reg           * reg_tracker ) {


  /* loop labels reset */
  libxsmm_loop_label_tracker l_loop_labels;
  libxsmm_reset_loop_label_tracker( &l_loop_labels );

  /* open stream */
  libxsmm_ppc64le_instr_open_stream_wt( io_generated_code, reg_tracker );

  /* create blocking */
  libxsmm_datatype a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int bytes = libxsmm_generator_gemm_ppc64le_bytes( io_generated_code, a_datatype );
  unsigned int v_len = (LIBXSMM_PPC64LE_VSR_WIDTH / 8) / bytes;
  unsigned int blocking[3];
  blocking[0] = 64 / bytes < i_xgemm_desc->n ? 16 / bytes : i_xgemm_desc->n; /* n-blocking */
  blocking[1] = 96 / bytes < i_xgemm_desc->m ? 32 / bytes : i_xgemm_desc->m; /* m-blocking */
  blocking[2] = 64 / bytes < i_xgemm_desc->k ? 64 / bytes : i_xgemm_desc->k; /* k-blocking */

  unsigned int n_vsr = ( ( blocking[0] + blocking[1] )*( blocking[2] / v_len ) +
                         blocking[1]*blocking[0] / v_len);

  if ( n_vsr > LIBXSMM_PPC64LE_VSR_NMAX ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }

  /* GPRs holding pointers to A, B, and C */
  unsigned char i_a_ptr_gpr = LIBXSMM_PPC64LE_GPR_R3;
  unsigned char i_b_ptr_gpr = LIBXSMM_PPC64LE_GPR_R4;
  unsigned char i_c_ptr_gpr = LIBXSMM_PPC64LE_GPR_R5;

  unsigned int n_iters = i_xgemm_desc->n / blocking[0];

  unsigned int i_n_reg;

  /* Set jump point if required */
  if ( n_iters > 1 ) {
    i_n_reg = libxsmm_ppc64le_get_reg( reg_tracker, LIBXSMM_PPC64LE_GPR );
    /* Load imediate */
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_n_reg,
                             LIBXSMM_PPC64LE_GPR_R0,
                             n_iters );
    libxsmm_ppc64le_instr_register_jump_back_label( io_generated_code,
                                                    &l_loop_labels );

  }

  /* Call m-loop */
  libxsmm_generator_gemm_ppc64le_vsx_m_loop( io_generated_code,
                                            i_xgemm_desc,
                                            reg_tracker,
                                            blocking,
                                            i_a_ptr_gpr,
                                            i_b_ptr_gpr,
                                            i_c_ptr_gpr );

  /* Compare and jump if required */
  if ( n_iters > 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_n_reg,
                             i_n_reg,
                             -1 );
    libxsmm_ppc64le_instr_cond_jump_back_to_label( io_generated_code,
                                                   i_n_reg,
                                                   &l_loop_labels );
    libxsmm_ppc64le_free_reg( reg_tracker, LIBXSMM_PPC64LE_GPR, i_n_reg );
  }

  /* close stream */
  libxsmm_ppc64le_instr_close_stream_wt( io_generated_code, reg_tracker );

}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_kernel( libxsmm_generated_code * io_generated_code,
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
  libxsmm_ppc64le_reg i_reg = libxsmm_ppc64le_reg_init();

  libxsmm_generator_gemm_ppc64le_kernel_vsx_wt( io_generated_code, i_xgemm_desc, &i_reg );
  return;
}
