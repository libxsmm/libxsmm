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
void libxsmm_generator_gemm_mma_block_load_pair( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_m_blocks = i_m / l_vec_ele ;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  unsigned int l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m;

  /* Partial load length */
  if ( !l_packed ) {
    unsigned int l_temp = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_temp, ( i_m % l_vec_ele )*l_databytes );
    libxsmm_ppc64le_instr_set_shift_left( io_generated_code, l_temp, l_m_part, 56 );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_temp );
  }

  /* Offset registers for large lda and unaligned offset, plxv[p] can handle it, but this is faster */
  libxsmm_ppc64le_ptr_reg_alloc( io_generated_code,
                                 io_reg_tracker,
                                 i_a,
                                 i_n,
                                 i_lda*l_databytes,
                                 l_m_blocks*l_vec_len,
                                 l_a_ptr,
                                 l_offsets );

  for ( l_n = 0; l_n < i_n; ++l_n ) {
    unsigned int l_a = l_a_ptr[l_n];
    long l_col_offset = l_offsets[l_n];

    /* Full width pair load */
    for ( l_m = 0; l_m < l_m_blocks / 2; ++l_m ) {
      unsigned int l_t = io_t[l_n*i_ldt + 2*l_m + 1];
      long l_offset = 2*l_m*l_vec_len + l_col_offset;
      libxsmm_ppc64le_instr_load_pair( io_generated_code, l_a, l_offset, l_t );
    }

    /* Unparied registers */
    if ( l_m_blocks % 2 == 1 ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks - 1];
      long l_offset = (l_m_blocks - 1)*l_vec_len + l_col_offset;
      libxsmm_ppc64le_instr_load( io_generated_code, l_a, l_offset, l_t );
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      long l_offset = l_m_blocks*l_vec_len + l_col_offset;
      libxsmm_ppc64le_instr_load_part( io_generated_code, io_reg_tracker, l_a, l_offset, l_m_part, l_t );
    }
  }

  /* Free GPR */
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_trans( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_n_blocks = (i_n + l_vec_ele - 1) / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( 0 == ( i_m % l_vec_ele ) ) ? 1 : 0;

  unsigned int l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX], l_scratch[LIBXSMM_PPC64LE_VSR_WIDTH/8], l_t[LIBXSMM_PPC64LE_VSR_WIDTH/8];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m, l_i;

  /* Partial load length */
  if ( 0 == l_packed ) {
    unsigned int l_temp = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_temp, ( i_m % l_vec_ele )*l_databytes );
    libxsmm_ppc64le_instr_set_shift_left( io_generated_code, l_temp, l_m_part, 56 );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_temp );
  }

  /* Scratch registers */
  for ( l_i = 0; l_i < l_vec_ele; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  /* Offset registers for large lda and unaligned offset, plxv[p] can handle it, but this is faster */
  libxsmm_ppc64le_ptr_reg_alloc( io_generated_code,
                                 io_reg_tracker,
                                 i_a,
                                 i_n,
                                 i_lda*l_databytes,
                                 l_m_blocks*l_vec_len,
                                 l_a_ptr,
                                 l_offsets );

  for ( l_n = 0; l_n < l_n_blocks; ++l_n ) {
    /* Full width load */
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {

      /* Number of full width vector load blocks */
      unsigned int l_n_ele = ( i_n - l_vec_ele*l_n < l_vec_ele) ? i_n - l_vec_ele*l_n : l_vec_ele;

      /* Output registers for transposed block */
      for ( l_i = 0; l_i < l_vec_ele; ++l_i ) {
        l_t[l_i] = io_t[i_ldt*l_n + l_m*l_vec_ele + l_i];
      }

      /* Load full width vector block */
      for ( l_i = 0; l_i < l_n_ele; ++l_i ) {
        unsigned int l_a = l_a_ptr[l_vec_ele*l_n + l_i];
        long l_offset = l_m*l_vec_len + l_offsets[l_vec_ele*l_n + l_i];
        libxsmm_ppc64le_instr_load( io_generated_code, l_a, l_offset, l_scratch[l_i]);
      }

      /* Transpose vector block */
      libxsmm_ppc64le_instr_transpose( io_generated_code, io_reg_tracker, i_datatype, l_scratch, l_vec_ele, l_t, l_n_ele );
    }

    /* Partial load */
    if ( 0 == l_packed ) {
      unsigned int l_m_ele = i_m % l_vec_ele;
      unsigned int l_n_ele = ( i_n - l_vec_ele*l_n < l_vec_ele) ? i_n - l_vec_ele*l_n : l_vec_ele;

      /* Output registers for transposed block */
      for ( l_i = 0 ; l_i < l_m_ele ; ++l_i ) {
        l_t[l_i] = io_t[i_ldt*l_n + l_m_blocks*l_vec_ele + l_i];
      }

      for ( l_i = 0; l_i < l_n_ele; ++l_i ) {
        unsigned int l_a = l_a_ptr[l_vec_ele*l_n + l_i];
        long l_offset = l_m_blocks*l_vec_len + l_offsets[l_vec_ele*l_n + l_i];
        libxsmm_ppc64le_instr_load_part( io_generated_code, io_reg_tracker, l_a, l_offset, l_m_part, l_scratch[l_i] );
      }

      /* Transpose vector block */
      libxsmm_ppc64le_instr_transpose( io_generated_code, io_reg_tracker, i_datatype, l_scratch, l_m_ele, l_t, l_n_ele );
    }
  }

  /* Free scratch */
  for ( l_i = 0; l_i < l_vec_ele; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }

  /* Free GPR */
  if ( 0 == l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_acc( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_acc_vec = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  unsigned int l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m;

  /* Partial load length */
  if ( !l_packed ) {
    unsigned int l_temp = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_temp, ( i_m % l_vec_ele )*l_databytes );
    libxsmm_ppc64le_instr_set_shift_left( io_generated_code, l_temp, l_m_part, 56 );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_temp);
  }

  /* Offset registers for large lda and unaligned offset, plxv[p] can handle it, but this is faster */
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
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ) );
      long l_offset = l_m*l_vec_len + l_offsets[l_n];
      libxsmm_ppc64le_instr_load( io_generated_code, l_a_ptr[l_n], l_offset, l_t );
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m_blocks] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ));
      long l_offset = l_m_blocks*l_vec_len + l_offsets[l_n];
      libxsmm_ppc64le_instr_load_part( io_generated_code, io_reg_tracker, l_a_ptr[l_n], l_offset, l_m_part, l_t );
    }
  }

  /* Move from VSR to ACC */
  for ( l_n = 0; l_n < i_n / l_acc_vec ; ++l_n ) {
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMTACC, l_t );
    }

    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMTACC, l_t );
    }
  }

  /* Free GPR */
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_store_acc( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_acc_vec = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  unsigned int l_a_ptr[LIBXSMM_PPC64LE_GPR_NMAX];
  long l_offsets[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int l_n, l_m;

  /* Partial length */
  if ( !l_packed ) {
    unsigned int l_temp = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_temp, ( i_m % l_vec_ele )*l_databytes );
    libxsmm_ppc64le_instr_set_shift_left( io_generated_code, l_temp, l_m_part, 56 );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_temp );
  }

  /* Move from ACC to VSR */
  for ( l_n = 0; l_n < (i_n + l_acc_vec - 1 )/ l_acc_vec ; ++l_n ) {
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMFACC, l_t );
    }

    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMFACC, l_t );
    }
  }

  /* Offset registers for large lda and unaligned offset, pstxv[p] can handle it, but this is faster */
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
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ) );
      long l_offset = l_vec_len*l_m + l_offsets[l_n];
      libxsmm_ppc64le_instr_store( io_generated_code, l_a_ptr[l_n], l_offset, l_t);
    }

    /* Partial store */
    if ( !l_packed ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m_blocks] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ) );
      long l_offset = l_vec_len*l_m_blocks + l_offsets[l_n];
      libxsmm_ppc64le_instr_store_part( io_generated_code, io_reg_tracker, l_a_ptr[l_n], l_offset, l_m_part, l_t);
    }
  }

  /* Free GPR */
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  libxsmm_ppc64le_ptr_reg_free( io_generated_code, io_reg_tracker, l_a_ptr, i_n );
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger( libxsmm_generated_code *io_generated_code,
                                      libxsmm_datatype const  i_comptype,
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

  switch ( i_comptype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_generator_mma_block_ger_f32( io_generated_code, i_m, i_n, i_k, i_a, i_lda, i_b, i_ldb, i_beta, io_c, i_ldc );
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_generator_mma_block_ger_f64( io_generated_code, i_m, i_n, i_k, i_a, i_lda, i_b, i_ldb, i_beta, io_c, i_ldc );
    } break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      /*libxsmm_generator_mma_block_ger_xf16( io_generated_code, i_comptype, i_m, i_n, i_k, i_a, i_lda, i_B, i_ldb, i_beta, io_c, i_ldc );*/
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    }
  }

}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_alu( libxsmm_generated_code *io_generated_code,
                                libxsmm_datatype const  i_datatype,
                                unsigned int            i_a,
                                unsigned int            i_b,
                                unsigned int            i_c,
                                char                    i_alpha,
                                char                    i_beta,
                                unsigned int            i_amask,
                                unsigned int            i_bmask ) {
  char l_masked;
  /* Given alpha/beta, calculate opcode relative to mul */
  int l_sign = 0x0100 * ( 1 - i_alpha ) + 0x0200 * ( 1 - i_beta ) - 0x0008;

  /* If beta is zero we only suport positive multiplication */
  if ( ( i_beta == 0 ) && ( i_alpha != 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  } else if ( i_alpha == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  switch ( i_datatype ) {
      case LIBXSMM_DATATYPE_F32:
      case LIBXSMM_DATATYPE_F16:
      case LIBXSMM_DATATYPE_BF16: {
        l_masked = ( ( i_amask == 0x0f ) && ( i_bmask == 0x0f ) ) ? 0 : 1;
      } break;
      case LIBXSMM_DATATYPE_F64: {
        l_masked = ( ( i_amask == 0x0f ) && ( i_bmask == 0x03 ) ) ? 0 : 1;
      } break;
      default: {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

  /* If no masking */
  if ( l_masked == 0 ) {
    unsigned int l_op;
    switch ( i_datatype ) {
      case LIBXSMM_DATATYPE_F32: {
        l_op = LIBXSMM_PPC64LE_INSTR_XVF32GER;
      } break;
      case LIBXSMM_DATATYPE_F16: {
        l_op = LIBXSMM_PPC64LE_INSTR_XVF16GER2;
      } break;
      case LIBXSMM_DATATYPE_BF16: {
        l_op = LIBXSMM_PPC64LE_INSTR_XVBF16GER2;
      } break;
      case LIBXSMM_DATATYPE_F64: {
        l_op = LIBXSMM_PPC64LE_INSTR_XVF64GER;
      } break;
      default: {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if ( i_beta != 0 ) {
      l_op += l_sign;
    }

    libxsmm_ppc64le_instr_5( io_generated_code, l_op, i_c, i_a, i_b, (0x20 & i_a) >> 5, (0x20 & i_b) >> 5 );
  } else {
    unsigned long l_op;
    switch ( i_datatype ) {
      case LIBXSMM_DATATYPE_F32: {
        l_op = LIBXSMM_PPC64LE_INSTR_PMXVF32GER;
      } break;
      case LIBXSMM_DATATYPE_F16: {
        l_op = LIBXSMM_PPC64LE_INSTR_PMXVF16GER2;
      } break;
      case LIBXSMM_DATATYPE_BF16: {
        l_op = LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2;
      } break;
      case LIBXSMM_DATATYPE_F64: {
        l_op = LIBXSMM_PPC64LE_INSTR_PMXVF64GER;
      } break;
      default: {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      };
    }

    if ( i_beta != 0 ) {
      l_op += l_sign;
    }

    libxsmm_ppc64le_instr_prefix_7( io_generated_code, l_op, i_amask, i_bmask, i_c, i_a, i_b, (0x20 & i_a) >> 5, (0x20 & i_b) >> 5 );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f32( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / 4;
  unsigned int l_n_blocks = ( i_n + l_vec_ele - 1 ) / l_vec_ele;
  unsigned int l_m_blocks = ( i_m + l_vec_ele - 1 ) / l_vec_ele;
  unsigned int l_n, l_m, l_k;

  for ( l_n = 0; l_n < l_n_blocks; ++l_n ) {
    for ( l_m = 0; l_m < l_m_blocks; ++l_m ) {
      for ( l_k = 0 ; l_k < i_k; ++l_k ) {
        unsigned int l_n_rem = i_n - l_n*l_vec_ele;
        unsigned int l_m_rem = i_m - l_m*l_vec_ele;
        unsigned int l_n_subblock = ( l_n_rem < l_vec_ele ) ? l_n_rem : l_vec_ele;
        unsigned int l_m_subblock = ( l_m_rem < l_vec_ele ) ? l_m_rem : l_vec_ele;
        unsigned int l_bmsk = ( 1 << l_n_subblock ) - 1;
        unsigned int l_amsk = ( 1 << l_m_subblock ) - 1;

        unsigned int l_c = io_c[l_m + l_n*i_ldc];
        unsigned int l_b = i_b[l_k + l_n*i_ldb];
        unsigned int l_a = i_a[l_m + l_k*i_lda];

        char l_beta = ( i_beta == 0 && l_k == 0 ) ? 0 : 1;
        libxsmm_generator_mma_alu( io_generated_code, LIBXSMM_DATATYPE_F32, l_b, l_a, l_c, 1, l_beta, l_bmsk, l_amsk );
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f64( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / 8;
  unsigned int l_n_ele = 2*l_vec_ele;
  unsigned int l_m_ele = l_vec_ele;
  unsigned int l_n_blocks = ( i_n + l_n_ele - 1 ) / l_n_ele;
  unsigned int l_m_blocks = ( i_m + l_m_ele - 1 ) / l_m_ele;
  unsigned int l_n, l_m, l_k;

  for ( l_n = 0 ; l_n < l_n_blocks ; ++l_n ) {
    for ( l_m = 0 ; l_m < l_m_blocks ; ++l_m ) {
      for ( l_k = 0 ; l_k < i_k ; ++l_k ) {
        unsigned int l_n_rem = i_n - l_n*l_vec_ele;
        unsigned int l_m_rem = i_m - l_m*l_vec_ele;
        unsigned int l_n_subblock = ( l_n_rem < l_n_ele ) ? l_n_rem : l_n_ele;
        unsigned int l_m_subblock = ( l_m_rem < l_m_ele ) ? l_m_rem : l_m_ele;
        unsigned int l_bmsk = ( 1 << l_n_subblock ) - 1;
        unsigned int l_amsk = ( 1 << l_m_subblock ) - 1;

        unsigned int l_c = io_c[l_m + l_n*i_ldc];
        unsigned int l_b = i_b[l_k + (2*l_n + 1)*i_ldb];
        unsigned int l_a = i_a[l_m + l_k*i_lda];

        char l_beta = ( i_beta == 0 && l_k == 0 ) ? 0 : 1;
        libxsmm_generator_mma_alu( io_generated_code, LIBXSMM_DATATYPE_F64, l_b, l_a, l_c, 1, l_beta, l_bmsk, l_amsk );

      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_br_mma_microkernel( libxsmm_generated_code        *io_generated_code,
                                           libxsmm_gemm_descriptor const *i_xgemm_desc,
                                           libxsmm_ppc64le_blocking      *i_blocking,
                                           libxsmm_ppc64le_reg           *io_reg_tracker,
                                           libxsmm_loop_label_tracker    *io_loop_labels,
                                           unsigned char                  i_a,
                                           unsigned char                  i_b,
                                           unsigned char                  i_c,
                                           unsigned int                  *i_c_acc ) {
  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_n_k_blocks = ( i_xgemm_desc->k + i_blocking->block_k - 1 ) / i_blocking->block_k;
  unsigned int l_a, l_b, l_a_last, l_b_last, l_a_pipe[2], l_b_pipe[2];
  unsigned int l_a_reg[LIBXSMM_PPC64LE_VSR_NMAX], l_b_reg[LIBXSMM_PPC64LE_VSR_NMAX];
  unsigned int l_i, l_k_block;

  /* Local pointers registers for A and B */
  for ( l_i = 0; l_i < 2 ; ++l_i ) {
    l_a_pipe[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_pipe[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  }

  /* Allocate registers for A */
  libxsmm_ppc64le_alloc_mat( io_generated_code,
                             io_reg_tracker,
                             LIBXSMM_PPC64LE_ALLOC_ROW_PAIR,
                             LIBXSMM_PPC64LE_VSR,
                             i_blocking->reg_lda,
                             i_blocking->n_reg_a / i_blocking->reg_lda,
                             l_a_reg,
                             i_blocking->reg_lda );

  /* Allocate registers for B */
  if ( l_b_datatype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_ppc64le_alloc_mat( io_generated_code,
                               io_reg_tracker,
                               LIBXSMM_PPC64LE_ALLOC_COL_PAIR,
                               LIBXSMM_PPC64LE_VSR,
                               i_blocking->reg_ldb,
                               i_blocking->n_reg_b / i_blocking->reg_ldb,
                               l_b_reg,
                               i_blocking->reg_ldb );
  } else {
    libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, i_blocking->n_reg_b, 1, l_b_reg );
  }

  for ( l_k_block = 0 ; l_k_block < l_n_k_blocks ; ++l_k_block ) {
    unsigned int l_k_rem = i_xgemm_desc->k - l_k_block*i_blocking->block_k;
    unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;

    /* Make pipeline of pointers to reduce hazards */
    if ( 0 == l_k_block ) {
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
    libxsmm_generator_gemm_mma_block_load_pair( io_generated_code,
                                                io_reg_tracker,
                                                l_a_datatype,
                                                l_comptype,
                                                l_a,
                                                i_blocking->block_m,
                                                l_block_k,
                                                i_xgemm_desc->lda,
                                                l_a_reg,
                                                i_blocking->reg_lda );

    /* Load block of B and transpose */
    libxsmm_generator_gemm_mma_block_load_trans( io_generated_code,
                                                 io_reg_tracker,
                                                 l_b_datatype,
                                                 l_comptype,
                                                 l_b,
                                                 l_block_k,
                                                 i_blocking->block_n,
                                                 i_xgemm_desc->ldb,
                                                 l_b_reg,
                                                 i_blocking->reg_ldb );

    /* GER call */
    libxsmm_generator_mma_block_ger( io_generated_code,
                                     l_comptype,
                                     i_blocking->block_m,
                                     i_blocking->block_n,
                                     l_block_k,
                                     l_a_reg,
                                     i_blocking->reg_lda,
                                     l_b_reg,
                                     i_blocking->reg_ldb,
                                     1,
                                     i_c_acc,
                                     i_blocking->reg_ldc );
  }

  /* Free A and B registers */
  for ( l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }
  for ( l_i = 0; l_i < 2; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_pipe[l_i] );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_pipe[l_i] );
  }

  return;
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_microkernel( libxsmm_generated_code        *io_generated_code,
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
  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x04 ) >> 2;
  unsigned int l_n_k_blocks = ( i_xgemm_desc->k + i_blocking->block_k - 1 ) / i_blocking->block_k;
  unsigned int l_a, l_b, l_a_last, l_b_last, l_a_pipe[2], l_b_pipe[2];
  unsigned int l_a_reg[LIBXSMM_PPC64LE_VSR_NMAX], l_b_reg[LIBXSMM_PPC64LE_VSR_NMAX], l_c_acc[LIBXSMM_PPC64LE_ACC_NMAX];
  unsigned int l_i, l_k_block;

  /* Local pointers registers for A and B */
  for ( l_i = 0; l_i < 2 ; ++l_i ) {
    l_a_pipe[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_pipe[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  }

  /* Allocate acc for C, do this first to guarantee they can be allocated */
  for ( l_i = 0; l_i < i_blocking->n_acc_c; ++l_i ) {
    l_c_acc[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_ACC );
  }

  /* Allocate registers for A */
  libxsmm_ppc64le_alloc_mat( io_generated_code,
                             io_reg_tracker,
                             LIBXSMM_PPC64LE_ALLOC_ROW_PAIR,
                             LIBXSMM_PPC64LE_VSR,
                             i_blocking->reg_lda,
                             i_blocking->n_reg_a / i_blocking->reg_lda,
                             l_a_reg,
                             i_blocking->reg_lda );

  /* Allocate registers for B */
  if ( l_b_datatype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_ppc64le_alloc_mat( io_generated_code,
                               io_reg_tracker,
                               LIBXSMM_PPC64LE_ALLOC_COL_PAIR,
                               LIBXSMM_PPC64LE_VSR,
                               i_blocking->reg_ldb,
                               i_blocking->n_reg_b / i_blocking->reg_ldb,
                               l_b_reg,
                               i_blocking->reg_ldb );
  } else {
    libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, i_blocking->n_reg_b, 1, l_b_reg );
  }

  /* Load C into accumulators if beta != 0 */
  if ( !l_beta_zero ) {
    libxsmm_generator_gemm_mma_block_load_acc( io_generated_code,
                                               io_reg_tracker,
                                               l_c_datatype,
                                               l_comptype,
                                               i_c,
                                               i_blocking->block_m,
                                               i_blocking->block_n,
                                               i_xgemm_desc->ldc,
                                               l_c_acc,
                                               i_blocking->reg_ldc );
  }

  for ( l_k_block = 0 ; l_k_block < l_n_k_blocks ; ++l_k_block ) {
    unsigned int l_k_rem = i_xgemm_desc->k - l_k_block*i_blocking->block_k;
    unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;
    unsigned int l_beta = ( ( l_k_block == 0 ) && ( l_beta_zero ) ) ? 0 : 1;

    /* Make pipeline of pointers to reduce hazards */
    if ( 0 == l_k_block ) {
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
    libxsmm_generator_gemm_mma_block_load_pair( io_generated_code,
                                                io_reg_tracker,
                                                l_a_datatype,
                                                l_comptype,
                                                l_a,
                                                i_blocking->block_m,
                                                l_block_k,
                                                i_xgemm_desc->lda,
                                                l_a_reg,
                                                i_blocking->reg_lda );

    /* Load block of B and transpose */
    libxsmm_generator_gemm_mma_block_load_trans( io_generated_code,
                                                 io_reg_tracker,
                                                 l_b_datatype,
                                                 l_comptype,
                                                 l_b,
                                                 l_block_k,
                                                 i_blocking->block_n,
                                                 i_xgemm_desc->ldb,
                                                 l_b_reg,
                                                 i_blocking->reg_ldb );

    /* GER call */
    libxsmm_generator_mma_block_ger( io_generated_code,
                                     l_comptype,
                                     i_blocking->block_m,
                                     i_blocking->block_n,
                                     l_block_k,
                                     l_a_reg,
                                     i_blocking->reg_lda,
                                     l_b_reg,
                                     i_blocking->reg_ldb,
                                     l_beta,
                                     l_c_acc,
                                     i_blocking->reg_ldc );
  }

  /* Free A and B registers */
  for ( l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }
  for ( l_i = 0; l_i < 2; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_pipe[l_i] );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_pipe[l_i] );
  }

  /* Store blocks of C */
  libxsmm_generator_gemm_mma_block_store_acc( io_generated_code,
                                              io_reg_tracker,
                                              l_c_datatype,
                                              l_comptype,
                                              i_c,
                                              i_blocking->block_m,
                                              i_blocking->block_n,
                                              i_xgemm_desc->ldc,
                                              l_c_acc,
                                              i_blocking->reg_ldc );

  /* Free C acc registers */
  for ( l_i = 0; l_i < i_blocking->n_acc_c ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_ACC, l_c_acc[l_i] );
  }

  return;
}
