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
void libxsmm_generator_gemm_mma_mk_f64_b_reg_alloc( libxsmm_generated_code *io_generated_code,
                                                    libxsmm_ppc64le_reg    *io_reg_tracker,
                                                    unsigned int const      i_n_rows,
                                                    unsigned int const      i_n_cols,
                                                    unsigned int           *o_reg,
                                                    unsigned int const      i_ld ) {
  unsigned int l_col_pairs = i_n_cols / 2;

  for ( int l_i = 0; l_i < l_col_pairs; ++l_i ) {
    for ( int l_j = 0; l_j < i_n_rows; ++l_j ) {
      unsigned int l_rpair[2];
      libxsmm_ppc64le_get_sequential_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, 2, l_rpair );
      o_reg[l_j + (2*l_i + 0)*i_ld] = l_rpair[0];
      o_reg[l_j + (2*l_i + 1)*i_ld] = l_rpair[1];
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_micro_load_trans( libxsmm_generated_code *io_generated_code,
                                                  libxsmm_ppc64le_reg    *io_reg_tracker,
                                                  libxsmm_datatype const  i_datatype,
                                                  libxsmm_datatype const  i_comptype,
                                                  unsigned int const      i_a,
                                                  unsigned int const      i_m,
                                                  unsigned int const      i_n,
                                                  unsigned int const      i_lda,
                                                  unsigned int           *io_t,
                                                  unsigned int const      i_ldt ) {

  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  /* Local copy of pointer */
  unsigned int l_a = i_a;
  if ( i_n > 1 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );
  }

  /* Partial length */
  if ( !l_packed ) {
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_part,
                             0,
                             i_m*l_databytes );
    unsigned char l_shift = 56;
    libxsmm_ppc64le_instr_6( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_RLDICR,
                             l_m_part,
                             l_m_part,
                             l_shift,
                             63 - l_shift,
                             (0x20 & l_shift) >> 5,
                             0 );
  }

  /* Scratch registers */
  unsigned int l_scratch[l_vec_ele];
  for ( unsigned int l_i = 0; l_i < l_vec_ele; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  /* Vector load and reverse opcode */
  unsigned int l_vec_ld, l_vec_br;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVW4X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRW;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVD2X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRD;
    } break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVH8X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRH;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  for ( unsigned int l_vec = 0; l_vec < i_n; ++l_vec ) {
    if ( l_packed ) {
      /* Load full width vector block */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_ld,
                               l_scratch[l_vec],
                               0,
                               l_a,
                               (0x0020 & l_scratch[l_vec]) >> 5 );
    } else {
      /* Partial load */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVLL,
                               l_scratch[l_vec],
                               l_a,
                               l_m_part,
                               (0x0020 & l_scratch[l_vec]) >> 5 );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_br,
                               l_scratch[l_vec],
                               l_scratch[l_vec],
                               (0x0020 & l_scratch[l_vec]) >> 5,
                               (0x0020 & l_scratch[l_vec]) >> 5 );
    }
    if ( l_vec < i_n - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*l_databytes );
    }
  }

  /* Transpose vector block */
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_ppc64le_instr_transpose_f32( io_generated_code, io_reg_tracker, l_scratch, i_m, io_t, i_n);
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_ppc64le_instr_transpose_f64( io_generated_code, io_reg_tracker, l_scratch, i_m, io_t, i_n);
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  /* Free scratch */
  for ( unsigned int l_i = 0; l_i < l_vec_ele; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }

  /* Free GPR */
  if ( i_n > 1 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  }
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_trans( libxsmm_generated_code   *io_generated_code,
                                                  libxsmm_ppc64le_reg      *io_reg_tracker,
                                                  libxsmm_datatype const    i_datatype,
                                                  libxsmm_datatype const    i_comptype,
                                                  unsigned int const        i_a,
                                                  unsigned int const        i_m,
                                                  unsigned int const        i_n,
                                                  unsigned int const        i_lda,
                                                  unsigned int             *io_t,
                                                  unsigned int const        i_ldt ) {

  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_n_blocks = (i_n + l_vec_ele - 1) / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  /* Local copy of pointer */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_a_col = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );

  /* Partial length */
  if ( !l_packed ) {
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_part,
                             0,
                             ( i_m % l_vec_ele )*l_databytes );
    unsigned char l_shift = 56;
    libxsmm_ppc64le_instr_6( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_RLDICR,
                             l_m_part,
                             l_m_part,
                             l_shift,
                             63 - l_shift,
                             (0x20 & l_shift) >> 5,
                             0 );
  }

  /* Scratch registers */
  unsigned int l_scratch[l_vec_ele];
  for ( unsigned int l_i = 0; l_i < l_vec_ele; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  /* Vector load and reverse opcode */
  unsigned int l_vec_ld, l_vec_br;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVW4X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRW;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVD2X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRD;
    } break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVH8X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRH;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  for ( unsigned int l_n = 0; l_n < l_n_blocks; ++l_n ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );

    /* Full width load */
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a_row, l_a_col, l_a_row );
      unsigned int l_t[l_vec_ele];

      for ( unsigned int l_vec = 0; l_vec < l_vec_ele; ++l_vec ) {
        l_t[l_vec] = io_t[i_ldt*l_n + l_m*l_vec_ele + l_vec];
      }

      /* Load full width vector block */
      unsigned int l_n_ele = ( i_n - l_vec_ele*l_n < l_vec_ele) ? i_n - l_vec_ele*l_n : l_vec_ele;
      for ( unsigned int l_vec = 0; l_vec < l_n_ele; ++l_vec ) {
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 l_vec_ld,
                                 l_scratch[l_vec],
                                 0,
                                 l_a_col,
                                 (0x0020 & l_scratch[l_vec]) >> 5 );
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_col, l_a_col, i_lda*l_databytes );
      }

      /* Transpose vector block */
      switch ( i_datatype ) {
        case LIBXSMM_DATATYPE_F32: {
          libxsmm_ppc64le_instr_transpose_f32( io_generated_code, io_reg_tracker, l_scratch, l_vec_ele, l_t, l_n_ele);
        } break;
        case LIBXSMM_DATATYPE_F64: {
          libxsmm_ppc64le_instr_transpose_f64( io_generated_code, io_reg_tracker, l_scratch, l_vec_ele, l_t, l_n_ele);
        } break;
        default: {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }

      if ( ( l_m < l_m_blocks - 1 ) || !l_packed ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_vec_len );
      }
    }

    /* Partial load */
    if ( !l_packed ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a_row, l_a_col, l_a_row );
      unsigned int l_m_ele = ( i_m % l_vec_ele );
      unsigned int l_t[l_vec_ele];

      for ( unsigned int l_vec = 0; l_vec < l_m_ele; ++l_vec ) {
        l_t[l_vec] = io_t[i_ldt*l_n + l_m_blocks*l_vec_ele + l_vec];
      }

      unsigned int l_n_ele = ( i_n - l_vec_ele*l_n < l_vec_ele) ? i_n - l_vec_ele*l_n : l_vec_ele;
      for ( unsigned int l_vec = 0; l_vec < l_n_ele; ++l_vec ) {
        l_t[l_vec] = io_t[i_ldt*l_n + l_m_blocks*l_vec_ele + l_vec];
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVLL,
                                 l_scratch[l_vec],
                                 l_a_col,
                                 l_m_part,
                                 (0x0020 & l_scratch[l_vec]) >> 5 );
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 l_vec_br,
                                 l_scratch[l_vec],
                                 l_scratch[l_vec],
                                 (0x0020 & l_scratch[l_vec]) >> 5,
                                 (0x0020 & l_scratch[l_vec]) >> 5 );
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_col, l_a_col, i_lda*l_databytes );
      }

      /* Transpose vector block */
      switch ( i_datatype ) {
        case LIBXSMM_DATATYPE_F32: {
          libxsmm_ppc64le_instr_transpose_f32( io_generated_code, io_reg_tracker, l_scratch, l_m_ele, l_t, l_n_ele);
        } break;
        case LIBXSMM_DATATYPE_F64: {
          libxsmm_ppc64le_instr_transpose_f64( io_generated_code, io_reg_tracker, l_scratch, l_m_ele, l_t, l_n_ele);
        } break;
        default: {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
    }

    /* increament if not last */
    if ( l_n < l_n_blocks - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*l_databytes*l_vec_ele );
    }
  }

  /* Free scratch */
  for ( unsigned int l_i = 0; l_i < l_vec_ele; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }

  /* Free GPR */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_col );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_acc( libxsmm_generated_code *io_generated_code,
                                                libxsmm_ppc64le_reg    *io_reg_tracker,
                                                libxsmm_datatype const  i_datatype,
                                                libxsmm_datatype const  i_comptype,
                                                unsigned int const      i_a,
                                                unsigned int const      i_m,
                                                unsigned int const      i_n,
                                                unsigned int const      i_lda,
                                                unsigned int           *io_t,
                                                unsigned int const      i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_acc_vec = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  /* Local copies of pointer */
  unsigned int l_a, l_a_row;
  if ( i_n > 1 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );
  } else {
    l_a = i_a;
  }

  if ( ( l_m_blocks > 1 ) || ( l_m_blocks > 0 && !l_packed ) ) {
      l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  } else {
    l_a_row = l_a;
  }

  /* Partial length */
  if ( !l_packed ) {
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_part,
                             0,
                             ( i_m % l_vec_ele )*l_databytes );
    unsigned char l_shift = 56;
    libxsmm_ppc64le_instr_6( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_RLDICR,
                             l_m_part,
                             l_m_part,
                             l_shift,
                             63 - l_shift,
                             (0x20 & l_shift) >> 5,
                             0 );
  }

  /* Vector load and reverse opcode */
  unsigned int l_vec_ld, l_vec_br;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVW4X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRW;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVD2X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRD;
    } break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVH8X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRH;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    if ( ( l_m_blocks > 1 ) || ( l_m_blocks > 0 && !l_packed ) ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    }

    /* Full width load */
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m] + ( l_n % l_acc_vec );

      libxsmm_ppc64le_instr_4( io_generated_code, l_vec_ld, l_t, 0, l_a_row, (0x0020 & l_t) >> 5 );

      /* Increment if not last or not packed */
      if ( ( l_m < l_m_blocks - 1 ) || !l_packed ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_vec_len );
      }
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m_blocks] + ( l_n % l_acc_vec );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVLL,
                               l_t,
                               l_a_row,
                               l_m_part,
                               (0x0020 & l_t) >> 5 );
      libxsmm_ppc64le_instr_4( io_generated_code, l_vec_br, l_t, l_t, (0x0020 & l_t) >> 5, (0x0020 & l_t) >> 5 );
    }

    /* Increament if not last */
    if ( l_n < i_n - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*l_databytes );
    }
  }

  /* Move from VSR to ACC */
  for ( unsigned int l_n = 0; l_n < i_n / l_acc_vec ; ++l_n ) {
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMTACC, l_t );
    }

    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMTACC, l_t );
    }
  }

  /* Free GPR */
  if ( i_n > 1 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  }
  if ( ( l_m_blocks > 1 ) || ( l_m_blocks > 0 && !l_packed ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  }
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_store_acc( libxsmm_generated_code *io_generated_code,
                                                 libxsmm_ppc64le_reg    *io_reg_tracker,
                                                 libxsmm_datatype const  i_datatype,
                                                 libxsmm_datatype const  i_comptype,
                                                 unsigned int const      i_a,
                                                 unsigned int const      i_m,
                                                 unsigned int const      i_n,
                                                 unsigned int const      i_lda,
                                                 unsigned int           *io_t,
                                                 unsigned int const      i_ldt ) {

  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_acc_vec = LIBXSMM_PPC64LE_ACC_WIDTH / LIBXSMM_PPC64LE_VSR_WIDTH;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  /* Local copies of pointer */
  unsigned int l_a, l_a_row;
  if ( i_n > 1 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );
  } else {
    l_a = i_a;
  }

  if ( ( l_m_blocks > 1 ) || ( l_m_blocks > 0 && !l_packed ) ) {
      l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  } else {
    l_a_row = l_a;
  }

  /* Partial length */
  if ( !l_packed ) {
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             l_m_part,
                             0,
                             ( i_m % l_vec_ele )*l_databytes );
    unsigned char l_shift = 56;
    libxsmm_ppc64le_instr_6( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_RLDICR,
                             l_m_part,
                             l_m_part,
                             l_shift,
                             63 - l_shift,
                             (0x20 & l_shift) >> 5,
                             0 );
  }

  /* Move from ACC to VSR */
  for ( unsigned int l_n = 0; l_n < i_n / l_acc_vec ; ++l_n ) {
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMFACC, l_t );
    }

    if ( !l_packed ) {
      unsigned int l_t = io_t[l_n*i_ldt + l_m_blocks];
      libxsmm_ppc64le_instr_1( io_generated_code, LIBXSMM_PPC64LE_INSTR_XXMFACC, l_t );
    }
  }

  /* Vector store and reverse opcode */
  unsigned int l_vec_st, l_vec_br;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_vec_st = LIBXSMM_PPC64LE_INSTR_STXVW4X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRW;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_st = LIBXSMM_PPC64LE_INSTR_STXVD2X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRD;
    } break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      l_vec_st = LIBXSMM_PPC64LE_INSTR_STXVH8X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRH;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    if ( ( l_m_blocks > 1 ) || ( l_m_blocks > 0 && !l_packed ) ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    }

    /* Full width store */
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m] + ( l_n % l_acc_vec );

      libxsmm_ppc64le_instr_4( io_generated_code, l_vec_st, l_t, 0, l_a_row, (0x0020 & l_t) >> 5 );

      /* Increment if not last or not packed */
      if ( ( l_m < l_m_blocks - 1 ) || !l_packed ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_vec_len );
      }
    }

    /* Partial store */
    if ( !l_packed ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m_blocks] + ( l_n % l_acc_vec );
      libxsmm_ppc64le_instr_4( io_generated_code, l_vec_br, l_t, l_t, (0x0020 & l_t) >> 5, (0x0020 & l_t) >> 5 );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVLL,
                               l_t,
                               l_a_row,
                               l_m_part,
                               (0x0020 & l_t) >> 5 );
    }

    /* Increament if not last */
    if ( l_n < i_n - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*l_databytes );
    }
  }

  /* Free GPR */
  if ( i_n > 1 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  }
  if ( ( l_m_blocks > 1 ) || ( l_m_blocks > 0 && !l_packed ) ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  }
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger( libxsmm_generated_code *io_generated_code,
                                      libxsmm_datatype const  i_comptype,
                                      unsigned int const      i_m,
                                      unsigned int const      i_n,
                                      unsigned int const      i_k,
                                      unsigned int           *i_a,
                                      unsigned int const      i_lda,
                                      unsigned int           *i_b,
                                      unsigned int const      i_ldb,
                                      unsigned int const      i_beta,
                                      unsigned int           *io_c,
                                      unsigned int const      i_ldc ) {

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
void libxsmm_generator_mma_micro_ger( libxsmm_generated_code *io_generated_code,
                                      libxsmm_datatype const  i_comptype,
                                      unsigned int const      i_m,
                                      unsigned int const      i_n,
                                      unsigned int const      i_k,
                                      unsigned int           *i_a,
                                      unsigned int const      i_lda,
                                      unsigned int           *i_b,
                                      unsigned int const      io_c ) {
  switch ( i_comptype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_generator_mma_micro_ger_f32( io_generated_code, i_m, i_n, i_k, i_a, i_lda, i_b, io_c );
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_generator_mma_micro_ger_f64( io_generated_code, i_m, i_n, i_k, i_a, i_lda, i_b, io_c );
    } break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_micro_ger_f32( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      io_c ) {
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / 4;

  /* vector mul and fma opcodes */
  unsigned int l_acc_fma = LIBXSMM_PPC64LE_INSTR_XVF32GERPP;
  unsigned long l_acc_mfma = LIBXSMM_PPC64LE_INSTR_PMXVF32GERPP;

  if ( i_n == l_vec_ele && i_m == l_vec_ele ) {
    for ( unsigned int l_k = 0; l_k < i_k; ++l_k ) {
      unsigned int l_a_reg = i_a[l_k*i_lda];
      unsigned int l_b_reg = i_b[l_k];
      libxsmm_ppc64le_instr_5( io_generated_code,
                               l_acc_fma,
                               io_c,
                               l_b_reg,
                               l_a_reg,
                               (0x0020 & l_b_reg) >> 5,
                               (0x0020 & l_a_reg) >> 5 );
    }
  } else {
    unsigned int l_bmsk = ( ( 1 << i_n ) - 1 ) << ( 4 - i_n );
    unsigned int l_amsk = ( ( 1 << i_m ) - 1 ) << ( 4 - i_m );
    for ( unsigned int l_k = 0; l_k < i_k ; ++l_k ) {
      unsigned int l_a_reg = i_a[l_k*i_lda];
      unsigned int l_b_reg = i_b[l_k];
      libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                      l_acc_mfma,
                                      l_bmsk,
                                      l_amsk,
                                      io_c,
                                      l_b_reg,
                                      l_a_reg,
                                      (0x0020 & l_b_reg) >> 5,
                                      (0x0020 & l_a_reg) >> 5 );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_micro_ger_f64( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      io_c ) {
  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / 8;
  unsigned int l_n_ele = 2*l_vec_ele;
  unsigned int l_m_ele = l_vec_ele;

  /* vector mul and fma opcodes */
  unsigned int l_acc_fma = LIBXSMM_PPC64LE_INSTR_XVF64GERPP;
  unsigned long l_acc_mfma = LIBXSMM_PPC64LE_INSTR_PMXVF64GERPP;

  if ( i_n < l_n_ele || i_m < l_m_ele ) {
    for ( unsigned int l_k = 0; l_k < i_k ; ++l_k ) {
      unsigned int l_a_reg = i_a[l_k*i_lda];
      unsigned int l_b_reg = i_b[l_k];
      libxsmm_ppc64le_instr_5( io_generated_code,
                               l_acc_fma,
                               io_c,
                               l_b_reg,
                               l_a_reg,
                               (0x0020 & l_b_reg) >> 5,
                               (0x0020 & l_a_reg) >> 5 );
    }
  } else {
    unsigned int l_bmsk = ( ( 1 << i_n ) - 1 ) << ( 4 - i_n );
    unsigned int l_amsk = ( ( 1 << i_m ) - 1 ) << ( 2 - i_m );

    for ( unsigned int l_k = 0; l_k < i_k ; ++l_k ) {
      unsigned int l_a_reg = i_a[l_k*i_lda];
      unsigned int l_b_reg = i_b[l_k];
      libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                      l_acc_mfma,
                                      l_bmsk,
                                      l_amsk,
                                      io_c,
                                      l_b_reg,
                                      l_a_reg,
                                      (0x0020 & l_b_reg) >> 5,
                                      (0x0020 & l_a_reg) >> 5 );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f32( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      i_ldb,
                                          unsigned int const      i_beta,
                                          unsigned int           *io_c,
                                          unsigned int const      i_ldc ) {

  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / 4;
  unsigned int l_n_blocks = ( i_n + l_vec_ele - 1 ) / l_vec_ele;
  unsigned int l_m_blocks = ( i_m + l_vec_ele - 1 ) / l_vec_ele;

  /* vector mul and fma opcodes */
  unsigned int l_acc_mul = LIBXSMM_PPC64LE_INSTR_XVF32GER;
  unsigned int l_acc_fma = LIBXSMM_PPC64LE_INSTR_XVF32GERPP;
  unsigned long l_acc_mmul = LIBXSMM_PPC64LE_INSTR_PMXVF32GER;
  unsigned long l_acc_mfma = LIBXSMM_PPC64LE_INSTR_PMXVF32GERPP;

  for ( int l_n = 0; l_n < l_n_blocks; ++l_n ) {
    unsigned int l_n_rem = i_n - l_n*l_vec_ele;
    for ( int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_m_rem = i_m - l_m*l_vec_ele;
      for ( int l_k = 0 ; l_k < i_k; ++l_k ) {
        unsigned int l_full = ( l_n_rem < l_vec_ele || l_m_rem < l_vec_ele ) ? 0 : 1;
        if ( l_full ) {
          if ( i_beta == 0 && l_k == 0 ) {
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     l_acc_mul,
                                     io_c[l_m + l_n*i_ldc],
                                     i_b[l_k + l_n*i_ldb],
                                     i_a[l_m + l_k*i_lda],
                                     (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          } else {
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     l_acc_fma,
                                     io_c[l_m + l_n*i_ldc],
                                     i_b[l_k + l_n*i_ldb],
                                     i_a[l_m + l_k*i_lda],
                                     (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          }
        } else {
          unsigned int l_n_subblock = ( l_n_rem < l_vec_ele ) ? l_n_rem : l_vec_ele;
          unsigned int l_m_subblock = ( l_m_rem < l_vec_ele ) ? l_m_rem : l_vec_ele;
          unsigned int l_bmsk = ( ( 1 << l_n_subblock ) - 1 ) << ( 4 - l_n_subblock );
          unsigned int l_amsk = ( ( 1 << l_m_subblock ) - 1 ) << ( 4 - l_m_subblock );

          if ( i_beta == 0 && l_k == 0 ) {
            libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                            l_acc_mmul,
                                            l_bmsk,
                                            l_amsk,
                                            io_c[l_m + l_n*i_ldc],
                                            i_b[l_k + l_n*i_ldb],
                                            i_a[l_m + l_k*i_lda],
                                            (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                            (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          } else {
            libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                            l_acc_mfma,
                                            l_bmsk,
                                            l_amsk,
                                            io_c[l_m + l_n*i_ldc],
                                            i_b[l_k + l_n*i_ldb],
                                            i_a[l_m + l_k*i_lda],
                                            (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                            (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          }
        }
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_block_ger_f64( libxsmm_generated_code *io_generated_code,
                                          unsigned int const      i_m,
                                          unsigned int const      i_n,
                                          unsigned int const      i_k,
                                          unsigned int           *i_a,
                                          unsigned int const      i_lda,
                                          unsigned int           *i_b,
                                          unsigned int const      i_ldb,
                                          unsigned int const      i_beta,
                                          unsigned int           *io_c,
                                          unsigned int const      i_ldc ) {

  unsigned int l_vec_len =  LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / 8;
  unsigned int l_n_ele = 2*l_vec_ele;
  unsigned int l_m_ele = l_vec_ele;
  unsigned int l_n_blocks = ( i_n + l_n_ele - 1 ) / l_n_ele;
  unsigned int l_m_blocks = ( i_m + l_m_ele - 1 ) / l_m_ele;

  /* vector mul and fma opcodes */
  unsigned int l_acc_mul = LIBXSMM_PPC64LE_INSTR_XVF64GER;
  unsigned int l_acc_fma = LIBXSMM_PPC64LE_INSTR_XVF64GERPP;
  unsigned long l_acc_mmul = LIBXSMM_PPC64LE_INSTR_PMXVF64GER;
  unsigned long l_acc_mfma = LIBXSMM_PPC64LE_INSTR_PMXVF64GERPP;

  for ( int l_n = 0; l_n < l_n_blocks; ++l_n ) {
    unsigned int l_n_rem = i_n - l_n*l_vec_ele;
    for ( int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_m_rem = i_m - l_m*l_vec_ele;

      for ( int l_k = 0 ; l_k < i_k; ++l_k ) {
        unsigned int l_full = ( l_n_rem < l_vec_ele || l_m_rem < l_vec_ele ) ? 0 : 1;

        if ( l_full ) {
          if ( i_beta == 0 && l_k == 0 ) {
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     l_acc_mul,
                                     io_c[l_m + l_n*i_ldc],
                                     i_b[l_k + 2*l_n*i_ldb],
                                     i_a[l_m + l_k*i_lda],
                                     (0x0020 & i_b[l_k + 2*l_n*i_ldb]) >> 5,
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          } else {
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     l_acc_fma,
                                     io_c[l_m + l_n*i_ldc],
                                     i_b[l_k + 2*l_n*i_ldb],
                                     i_a[l_m + l_k*i_lda],
                                     (0x0020 & i_b[l_k + 2*l_n*i_ldb]) >> 5,
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          }
        } else {
          unsigned int l_n_subblock = ( l_n_rem < l_n_ele ) ? l_n_rem : l_n_ele;
          unsigned int l_m_subblock = ( l_m_rem < l_m_ele ) ? l_m_rem : l_m_ele;
          unsigned int l_bmsk = ( ( 1 << l_n_subblock ) - 1 ) << ( 4 - l_n_subblock );
          unsigned int l_amsk = ( ( 1 << l_m_subblock ) - 1 ) << ( 2 - l_m_subblock );

          if ( i_beta == 0 && l_k == 0 ) {
            libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                            l_acc_mmul,
                                            l_bmsk,
                                            l_amsk,
                                            io_c[l_m + l_n*i_ldc],
                                            i_b[l_k + 2*l_n*i_ldb],
                                            i_a[l_m + l_k*i_lda],
                                            (0x0020 & i_b[l_k + 2*l_n*i_ldb]) >> 5,
                                            (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          } else {
            libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                            l_acc_mfma,
                                            l_bmsk,
                                            l_amsk,
                                            io_c[l_m + l_n*i_ldc],
                                            i_b[l_k + 2*l_n*i_ldb],
                                            i_a[l_m + l_k*i_lda],
                                            (0x0020 & i_b[l_k + 2*l_n*i_ldb]) >> 5,
                                            (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          }
        }
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_microkernel( libxsmm_generated_code        *io_generated_code,
                                        libxsmm_gemm_descriptor const *i_xgemm_desc,
                                        libxsmm_ppc64le_blocking      *i_blocking,
                                        libxsmm_ppc64le_reg           *io_reg_tracker,
                                        libxsmm_loop_label_tracker    *io_loop_labels,
                                        unsigned int                  *i_acc,
                                        unsigned char const            i_a,
                                        unsigned char const            i_b,
                                        unsigned char const            i_c ) {
  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );

  unsigned int l_n_k_blocks = ( i_xgemm_desc->k + i_blocking->block_k - 1 ) / i_blocking->block_k;

  /* Local pointers registers for A and B */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_b, l_b, i_b );

  /* Allocate A and B registers */
  unsigned int l_a_reg[i_blocking->n_reg_a], l_b_reg[i_blocking->n_reg_b];

  /* Allocate B first as they might need to be sequential for F64 */
  if ( l_b_datatype == LIBXSMM_DATATYPE_F64 ) {
    unsigned int l_n_b_cols = i_blocking->n_reg_b / i_blocking->reg_ldb;
    libxsmm_generator_gemm_mma_mk_f64_b_reg_alloc( io_generated_code,
                                                   io_reg_tracker,
                                                   i_blocking->reg_ldb,
                                                   l_n_b_cols,
                                                   l_b_reg,
                                                   i_blocking->reg_ldb );
  } else {
    for ( int l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
      l_b_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
    }
  }

  for ( int l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    l_a_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  /* Load C into accumulators if beta != 0 */
  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x0004 ) >> 2;
  if ( !l_beta_zero ) {
    libxsmm_generator_gemm_mma_block_load_acc( io_generated_code,
                                               io_reg_tracker,
                                               l_c_datatype,
                                               l_comptype,
                                               i_c,
                                               i_blocking->block_m,
                                               i_blocking->block_n,
                                               i_xgemm_desc->ldc,
                                               i_acc,
                                               i_blocking->reg_ldc );
  }

  for ( unsigned int l_k_block = 0; l_k_block < l_n_k_blocks; ++l_k_block ) {
    unsigned int l_k_rem = i_xgemm_desc->k - l_k_block*i_blocking->block_k;
    unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;

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
    unsigned int l_beta = ( ( l_k_block == 0 ) && ( l_beta_zero ) ) ? 0 : 1;
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
                                     i_acc,
                                     i_blocking->reg_ldc );

    /* Increment pointers if not last */
    if ( l_k_block < l_n_k_blocks - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_a,
                               l_a,
                               i_blocking->block_k*i_xgemm_desc->lda*i_blocking->comp_bytes );

      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               l_b,
                               l_b,
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
  libxsmm_generator_gemm_mma_block_store_acc( io_generated_code,
                                              io_reg_tracker,
                                              l_c_datatype,
                                              l_comptype,
                                              i_c,
                                              i_blocking->block_m,
                                              i_blocking->block_n,
                                              i_xgemm_desc->ldc,
                                              i_acc,
                                              i_blocking->reg_ldc );
  return;
}


LIBXSMM_API_INTERN
void libxsmm_generator_mma_microkernel_sched( libxsmm_generated_code         *io_generated_code,
                                              libxsmm_gemm_descriptor const  *i_xgemm_desc,
                                              libxsmm_ppc64le_blocking       *i_blocking,
                                              libxsmm_ppc64le_reg            *io_reg_tracker,
                                              libxsmm_loop_label_tracker     *io_loop_labels,
                                              libxsmm_ppc64le_node          **i_schedule,
                                              unsigned int                   *i_acc,
                                              unsigned char const             i_a,
                                              unsigned char const             i_b,
                                              unsigned char const             i_c ) {
  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, l_comptype );

  unsigned int l_n_k_blocks = ( i_xgemm_desc->k + i_blocking->block_k - 1 ) / i_blocking->block_k;

  /* Local pointers registers for A and B */
  unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* Allocate A and B registers */
  unsigned int l_a_reg[i_blocking->n_reg_a], l_b_reg[i_blocking->n_reg_b];

  /* Allocate B first as they might need to be sequential for F64 */
  if ( l_b_datatype == LIBXSMM_DATATYPE_F64 ) {
    unsigned int l_n_b_cols = i_blocking->n_reg_b / i_blocking->reg_ldb;
    libxsmm_generator_gemm_mma_mk_f64_b_reg_alloc( io_generated_code,
                                                   io_reg_tracker,
                                                   i_blocking->reg_ldb,
                                                   l_n_b_cols,
                                                   l_b_reg,
                                                   i_blocking->reg_ldb );
  } else {
    for ( int l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
      l_b_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
    }
  }

  for ( int l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    l_a_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  /* Load C into accumulators if beta != 0 */
  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x0004 ) >> 2;
  if ( !l_beta_zero ) {
    libxsmm_generator_gemm_mma_block_load_acc( io_generated_code,
                                               io_reg_tracker,
                                               l_c_datatype,
                                               l_comptype,
                                               i_c,
                                               i_blocking->block_m,
                                               i_blocking->block_n,
                                               i_xgemm_desc->ldc,
                                               i_acc,
                                               i_blocking->reg_ldc );
  }

  for ( unsigned int l_k_block = 0; l_k_block < l_n_k_blocks; ++l_k_block ) {
    /*unsigned int l_k_rem = i_xgemm_desc->k - l_k_block*i_blocking->block_k;*/
    /*unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;*/

    unsigned int l_inode = 0;
    char l_scheduling = 1;
    while ( l_scheduling ) {
      libxsmm_ppc64le_node node = *i_schedule[l_inode];
      switch ( node.type ) {
        case LIBXSMM_PPC64LE_NODE_LOAD: {
          switch ( node.m ) {
            case LIBXSMM_PPC64LE_MAT_A: {
              unsigned int l_offset = (node.row * i_blocking->m_ele + (node.col * i_blocking->k_ele + l_k_block * i_blocking->block_k )* i_xgemm_desc->lda) * l_databytes;
              unsigned int l_a_idx = node.row + node.col * i_blocking->k_ele * i_blocking->reg_lda ;

              libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a, l_offset );
              libxsmm_generator_gemm_vsx_micro_load_vsr( io_generated_code,
                                                         io_reg_tracker,
                                                         l_a_datatype,
                                                         l_comptype,
                                                         l_a,
                                                         i_blocking->m_ele,
                                                         i_blocking->k_ele,
                                                         i_xgemm_desc->lda,
                                                         &l_a_reg[l_a_idx],
                                                         i_blocking->reg_lda );
            } break;
            case LIBXSMM_PPC64LE_MAT_B: {
              unsigned int l_offset = (node.row * i_blocking->k_ele + l_k_block * i_blocking->block_k + node.col * i_blocking->n_ele * i_xgemm_desc->ldb) * l_databytes;
              unsigned int l_b_idx = node.row*i_blocking->k_ele + node.col * i_blocking->reg_ldb;

              libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_b, l_b, l_offset );
              libxsmm_generator_gemm_mma_micro_load_trans( io_generated_code,
                                                           io_reg_tracker,
                                                           l_b_datatype,
                                                           l_comptype,
                                                           l_b,
                                                           i_blocking->k_ele,
                                                           i_blocking->n_ele,
                                                           i_xgemm_desc->ldb,
                                                           &l_b_reg[l_b_idx],
                                                           i_blocking->reg_ldb );
            } break;
            default: {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
            }
          }
        } break;
        case LIBXSMM_PPC64LE_NODE_PRODUCER: {
          unsigned int l_a_idx = node.row + node.k * i_blocking->k_ele * i_blocking->reg_lda ;
          unsigned int l_b_idx = node.k * i_blocking->k_ele + node.col * i_blocking->reg_ldb;

          libxsmm_generator_mma_micro_ger( io_generated_code,
                                           l_comptype,
                                           i_blocking->m_ele,
                                           i_blocking->n_ele,
                                           i_blocking->k_ele,
                                           &l_a_reg[l_a_idx],
                                           i_blocking->reg_lda,
                                           &l_b_reg[l_b_idx],
                                           i_acc[node.row + node.col*i_blocking->reg_ldc] );
        } break;
        case LIBXSMM_PPC64LE_NODE_RETURN: {
          l_scheduling = 0;
        } break;
        default: {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        }
      }
      l_inode += 1;
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
  libxsmm_generator_gemm_mma_block_store_acc( io_generated_code,
                                              io_reg_tracker,
                                              l_c_datatype,
                                              l_comptype,
                                              i_c,
                                              i_blocking->block_m,
                                              i_blocking->block_n,
                                              i_xgemm_desc->ldc,
                                              i_acc,
                                              i_blocking->reg_ldc );
  return;
}
