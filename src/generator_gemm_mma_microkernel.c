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
  unsigned int l_m_blocks = i_m / l_vec_ele ;
  unsigned int l_m_part;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

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

  /* Offset registers for large lda plxv[p] can handle it, but this is faster */
  unsigned int l_nld = ( 0x7fff - (l_m_blocks - 1)*l_vec_len ) / ( i_lda*l_databytes );
  unsigned int l_nptr = (i_n + l_nld ) / (l_nld + 1);
  unsigned int l_a[i_n];
  l_a[0] = i_a;
  for ( int i = 1; i < l_nptr; ++i ) {
    l_a[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    unsigned int l_offset = (l_nld + 1)*i*i_lda*l_databytes;
    libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a[i], l_offset );
  }

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    unsigned int l_aidx = l_n / ( l_nld + 1 );

    /* Full width pair load */
    for ( unsigned int l_m = 0; l_m < l_m_blocks / 2; ++l_m ) {
      unsigned int l_offset = 2*l_m*l_vec_len + (l_n - l_aidx*(l_nld + 1))*i_lda*l_databytes;
      unsigned int l_t_idx = l_n*i_ldt + 2*l_m;

      libxsmm_ppc64le_instr_load_pair( io_generated_code, l_a[l_aidx], l_offset, io_t[l_t_idx + 1] );
    }

    /* Unparied registers */
    if ( l_m_blocks % 2 == 1 ) {
      unsigned int l_offset = (l_m_blocks - 1)*l_vec_len + (l_n - l_aidx*(l_nld + 1))*i_lda*l_databytes;
      unsigned int l_t_idx = l_n*i_ldt + l_m_blocks - 1;

      libxsmm_ppc64le_instr_load( io_generated_code, l_a[l_aidx], l_offset, io_t[l_t_idx] );
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t_idx = l_n*i_ldt + l_m_blocks;
      unsigned int l_offset = l_m_blocks*l_vec_len + l_n*i_lda*l_databytes;
      unsigned int l_a_p = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR);
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a_p, l_offset);
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVL,
                               io_t[l_t_idx],
                               l_a_p,
                               l_m_part,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
      libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_p);
    }
  }

  /* Free GPR */
  if ( !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_m_part );
  }
  for ( int i = 1; i < l_nptr; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a[i] );
  }
}



LIBXSMM_API_INTERN
void libxsmm_generator_gemm_mma_block_load_trans( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_n_blocks = (i_n + l_vec_ele - 1) / l_vec_ele;
  unsigned int l_m_part = -1;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

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

  for ( unsigned int l_n = 0; l_n < l_n_blocks; ++l_n ) {
    /* Full width load */
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t[l_vec_ele];
      for ( unsigned int l_vec = 0; l_vec < l_vec_ele; ++l_vec ) {
        l_t[l_vec] = io_t[i_ldt*l_n + l_m*l_vec_ele + l_vec];
      }

      /* Load full width vector block */
      unsigned int l_n_ele = ( i_n - l_vec_ele*l_n < l_vec_ele) ? i_n - l_vec_ele*l_n : l_vec_ele;
      for ( unsigned int l_vec = 0; l_vec < l_n_ele; ++l_vec ) {
        unsigned int l_offset = l_vec*i_lda*l_databytes + l_m*l_vec_len + l_n*i_lda*l_databytes*l_vec_ele;
        libxsmm_ppc64le_instr_load( io_generated_code, i_a, l_offset, l_scratch[l_vec]);
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
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_m_ele = ( i_m % l_vec_ele );
      unsigned int l_t[l_vec_ele], l_a[l_vec_ele];
      unsigned int l_n_ele = ( i_n - l_vec_ele*l_n < l_vec_ele) ? i_n - l_vec_ele*l_n : l_vec_ele;

      for ( unsigned int l_vec = 0; l_vec < l_n_ele; ++l_vec ) {
        unsigned int l_offset =  l_vec*i_lda*l_databytes + l_m_blocks*l_vec_len + l_n*i_lda*l_databytes*l_vec_ele;
        l_a[l_vec] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
        libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a[l_vec], l_offset );

        l_t[l_vec] = io_t[i_ldt*l_n + l_m_blocks*l_vec_ele + l_vec];
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_LXVL,
                                 l_scratch[l_vec],
                                 l_a[l_vec],
                                 l_m_part,
                                 (0x0020 & l_scratch[l_vec]) >> 5 );
      }
      for ( unsigned int l_vec = 0; l_vec < l_n_ele; ++l_vec ) {
        libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a[l_vec] );
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
  }

  /* Free scratch */
  for ( unsigned int l_i = 0; l_i < l_vec_ele; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }

  /* Free GPR */
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

  /* Partial length */
  if ( !l_packed ) {
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     0,
                                     l_m_part,
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

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    /* Full width load */
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ) );
      unsigned int l_offset = l_n*i_lda*l_databytes + l_m*l_vec_len;

      libxsmm_ppc64le_instr_load( io_generated_code, i_a, l_offset, l_t);
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m_blocks] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ));
      unsigned int l_offset = l_n*i_lda*l_databytes + l_m_blocks*l_vec_len;
      unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR);

      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a, l_offset);
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVL,
                               l_t,
                               l_a,
                               l_m_part,
                               (0x0020 & l_t) >> 5 );
      libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a);
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

  /* Partial length */
  if ( !l_packed ) {
    l_m_part = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_add_value( io_generated_code,
                                     io_reg_tracker,
                                     0,
                                     l_m_part,
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

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    /* Full width store */
    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ) );
      unsigned int l_offset = l_n*i_lda*l_databytes + l_vec_len*l_m;

      libxsmm_ppc64le_instr_store( io_generated_code, i_a, l_offset, l_t);
    }

    /* Partial store */
    if ( !l_packed ) {
      unsigned int l_t = l_acc_vec*io_t[( l_n/l_acc_vec )*i_ldt + l_m_blocks] + ( l_acc_vec - 1 - ( l_n % l_acc_vec ) );
      unsigned int l_offset = l_n*i_lda*l_databytes + l_vec_len*l_m_blocks;

      unsigned int l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a, l_offset );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVL,
                               l_t,
                               l_a,
                               l_m_part,
                               (0x0020 & l_t) >> 5 );
      libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
    }

  }

  /* Free GPR */
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
void libxsmm_generator_mma_alu( libxsmm_generated_code *io_generated_code,
                                libxsmm_datatype const  i_datatype,
                                unsigned int const      i_a,
                                unsigned int const      i_b,
                                unsigned int const      i_c,
                                char const              i_alpha,
                                char const              i_beta,
                                unsigned int const      i_amask,
                                unsigned int const      i_bmask ) {
  /* If beta is zero we only suport positive multiplication */
  if ( ( i_beta == 0 ) && ( i_alpha != 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  } else if ( i_alpha == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Given alpha/beta, calculate opcode relative to mul */
  int l_sign = 0x0100 * ( 1 - i_alpha ) + 0x0200 * ( 1 - i_beta ) - 0x0008;

  /* If no masking */
  if (( i_amask == 0x0f ) && ( i_bmask == 0x0f )) {
    unsigned int l_op;
    switch ( i_datatype ) {
      case LIBXSMM_DATATYPE_F32: {
        l_op = LIBXSMM_PPC64LE_INSTR_XVF32GER;
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
    for ( int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      for ( int l_k = 0 ; l_k < i_k; ++l_k ) {
        unsigned int l_n_rem = i_n - l_n*l_vec_ele;
        unsigned int l_m_rem = i_m - l_m*l_vec_ele;
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
    for ( int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      for ( int l_k = 0 ; l_k < i_k; ++l_k ) {
        unsigned int l_n_rem = i_n - l_n*l_vec_ele;
        unsigned int l_m_rem = i_m - l_m*l_vec_ele;
        unsigned int l_full = ( l_n_rem < l_vec_ele || l_m_rem < l_vec_ele ) ? 0 : 1;

        if ( l_full ) {
          if ( i_beta == 0 && l_k == 0 ) {
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     l_acc_mul,
                                     io_c[l_m + l_n*i_ldc],
                                     i_b[l_k + (2*l_n + 1)*i_ldb],
                                     i_a[l_m + l_k*i_lda],
                                     (0x0020 & i_b[l_k + (2*l_n + 1)*i_ldb]) >> 5,
                                     (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          } else {
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     l_acc_fma,
                                     io_c[l_m + l_n*i_ldc],
                                     i_b[l_k + (2*l_n + 1)*i_ldb],
                                     i_a[l_m + l_k*i_lda],
                                     (0x0020 & i_b[l_k + (2*l_n + 1)*i_ldb]) >> 5,
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
                                            i_b[l_k + (2*l_n + 1)*i_ldb + 1],
                                            i_a[l_m + l_k*i_lda],
                                            (0x0020 & i_b[l_k + (2*l_n + 1)*i_ldb + 1]) >> 5,
                                            (0x0020 & i_a[l_m + l_k*i_lda]) >> 5 );
          } else {
            libxsmm_ppc64le_instr_prefix_7( io_generated_code,
                                            l_acc_mfma,
                                            l_bmsk,
                                            l_amsk,
                                            io_c[l_m + l_n*i_ldc],
                                            i_b[l_k + (2*l_n + 1)*i_ldb],
                                            i_a[l_m + l_k*i_lda],
                                            (0x0020 & i_b[l_k + (2*l_n + 1)*i_ldb]) >> 5,
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
  unsigned int l_a_pipe[2], l_b_pipe[2];
  for ( int i = 0; i < 2; ++i ) {
    l_a_pipe[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b_pipe[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  }

  /* Allocate space to store A and B indices */
  unsigned int l_a_reg[i_blocking->n_reg_a], l_b_reg[i_blocking->n_reg_b];

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

  unsigned int l_a, l_b, l_a_last, l_b_last;
  for ( unsigned int l_k_block = 0; l_k_block < l_n_k_blocks; ++l_k_block ) {
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

    unsigned int l_k_rem = i_xgemm_desc->k - l_k_block*i_blocking->block_k;
    unsigned int l_block_k = ( l_k_rem < i_blocking->block_k ) ? l_k_rem : i_blocking->block_k;

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
  }

  /* Free A and B registers */
  for ( int l_i = 0; l_i < i_blocking->n_reg_a ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( int l_i = 0; l_i < i_blocking->n_reg_b ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }
  for ( int i = 0; i < 2; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_pipe[i] );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b_pipe[i] );
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
