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
void libxsmm_generator_gemm_vsx_mk_load_vsr( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_m_part;
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
    }  break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVD2X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRD;
    }  break;
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVH8X;
      l_vec_br = LIBXSMM_PPC64LE_INSTR_XXBRH;
    }  break;
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
      unsigned int l_t_idx = l_n*i_ldt + l_m;

      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_ld,
                               io_t[l_t_idx],
                               0,
                               l_a_row,
                               (0x0020 & io_t[l_t_idx]) >> 5 );

      /* Increment if not last or not packed */
      if ( ( l_m < l_m_blocks - 1 ) || !l_packed ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_vec_len );
      }
    }

    /* Partial load */
    if ( !l_packed ) {
      unsigned int l_t_idx = l_n*i_ldt + l_m_blocks;
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LXVLL,
                               io_t[l_t_idx],
                               l_a_row,
                               l_m_part,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_br,
                               io_t[l_t_idx],
                               io_t[l_t_idx],
                               (0x0020 & io_t[l_t_idx]) >> 5,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
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
void libxsmm_generator_gemm_vsx_mk_store_vsr( libxsmm_generated_code *io_generated_code,
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
  unsigned int l_m_part;
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
      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_st,
                               io_t[l_n*i_ldt + l_m],
                               0,
                               l_a_row,
                               (0x0020 & io_t[l_n*i_ldt + l_m]) >> 5 );

      /* Increment if not last or not packed */
      if ( ( l_m < l_m_blocks - 1 ) || !l_packed ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_vec_len );
      }
    }
    /* Partial store */
    if ( !l_packed ) {
      unsigned int l_t_idx = l_n*i_ldt + l_m_blocks;
      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_br,
                               io_t[l_t_idx],
                               io_t[l_t_idx],
                               (0x0020 & io_t[l_t_idx]) >> 5,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
      libxsmm_ppc64le_instr_4( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STXVLL,
                               io_t[l_t_idx],
                               l_a_row,
                               l_m_part,
                               (0x0020 & io_t[l_t_idx]) >> 5 );
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
void libxsmm_generator_gemm_vsx_mk_load_vsr_splat( libxsmm_generated_code *io_generated_code,
                                                   libxsmm_ppc64le_reg    *io_reg_tracker,
                                                   libxsmm_datatype const  i_datatype,
                                                   libxsmm_datatype const  i_comptype, /* currently unsuded */
                                                   unsigned int const      i_a,
                                                   unsigned int const      i_m,
                                                   unsigned int const      i_n,
                                                   unsigned int const      i_lda,
                                                   unsigned int           *io_t,
                                                   unsigned int const      i_ldt ) {
  /* Data shape */
  unsigned int l_databytes = libxsmm_ppc64le_instr_bytes( io_generated_code, i_datatype );
  unsigned int l_vec_len = LIBXSMM_PPC64LE_VSR_WIDTH / 8;
  unsigned int l_vec_ele = l_vec_len / l_databytes;
  unsigned int l_m_blocks = i_m / l_vec_ele;
  unsigned int l_packed = ( ( i_m % l_vec_ele ) == 0 ) ? 1 : 0;

  /* Local copies of pointer */
  unsigned int l_a, l_a_row;
  if ( i_n > 1 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );
  } else {
    l_a = i_a;
  }

  if ( ( l_m_blocks > 1 ) || !l_packed ) {
    l_a_row = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  } else {
    l_a_row = l_a;
  }

  /* Vector scratch register */
  unsigned l_scratch = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );

  /* Vector load and load-splat opcode */
  unsigned int l_vec_ld, l_vec_sp;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVW4X;
      l_vec_sp = LIBXSMM_PPC64LE_INSTR_LXVWSX;
    }  break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_ld = LIBXSMM_PPC64LE_INSTR_LXVD2X;
      l_vec_sp = LIBXSMM_PPC64LE_INSTR_LXVDSX;
    }  break;
    default: {
      l_vec_ld = -1;
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    if ( ( l_m_blocks > 1 ) || !l_packed ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, l_a, l_a_row, l_a );
    }

    for ( unsigned int l_m = 0; l_m < l_m_blocks; ++l_m ) {
      /* Full width load */
      libxsmm_ppc64le_instr_4( io_generated_code,
                               l_vec_ld,
                               l_scratch,
                               0,
                               l_a_row,
                               (0x0020 & l_scratch) >> 5 );

      for ( unsigned int l_vec = 0; l_vec < l_vec_ele; ++l_vec ) {
        unsigned int l_t_idx = l_vec + l_m*l_vec_ele + l_n*i_ldt;
        /* Splat */
        switch ( i_datatype ) {
          case LIBXSMM_DATATYPE_F32:
            libxsmm_ppc64le_instr_5( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XXSPLTW,
                                     io_t[l_t_idx],
                                     l_vec,
                                     l_scratch,
                                     (0x0020 & l_scratch) >> 5,
                                     (0x0020 & io_t[l_t_idx]) >> 5 );
            break;
          case LIBXSMM_DATATYPE_F64:
            libxsmm_ppc64le_instr_7( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                                     io_t[l_t_idx],
                                     l_scratch,
                                     l_scratch,
                                     l_vec*3, /* ie: 0b00 or 0b11 */
                                     (0x0020 & l_scratch) >> 5,
                                     (0x0020 & l_scratch) >> 5,
                                     (0x0020 & io_t[l_t_idx]) >> 5 );
            break;
          default:
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
            return;
        }
      }

      /* Increament if not last or not packed */
      if ( ( l_m < l_m_blocks - 1 ) || !l_packed ) {
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_vec_len );
      }
    }
    /* Partial load and splat */
    if ( !l_packed ) {
      for ( unsigned int l_vec = 0; l_vec < ( i_m % l_vec_ele ); ++l_vec ) {
        unsigned int l_t_idx = l_vec + l_m_blocks*l_vec_ele + l_n*i_ldt;
        libxsmm_ppc64le_instr_4( io_generated_code,
                                 l_vec_sp,
                                 io_t[l_t_idx],
                                 0,
                                 l_a_row,
                                 (0x0020 & io_t[l_t_idx]) >> 5 );
        if ( l_vec < ( i_m % l_vec_ele ) - 1 ) {
          libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a_row, l_a_row, l_databytes );
        }
      }
    }

    /* Increament if not last */
    if ( l_n < i_n - 1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_a, l_a, i_lda*l_databytes );
    }
  }

  /* Free registers */
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch );
  if ( i_n > 1 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
  }
  if ( ( l_m_blocks > 1 ) || !l_packed ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a_row );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_block_fma_b_splat( libxsmm_generated_code * io_generated_code,
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
  /* vector mul and fma opcodes */
  unsigned int l_vec_fma, l_vec_mul;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_vec_mul = LIBXSMM_PPC64LE_INSTR_XVMULSP;
      l_vec_fma = LIBXSMM_PPC64LE_INSTR_XVMADDASP;
    }  break;
    case LIBXSMM_DATATYPE_F64: {
      l_vec_mul = LIBXSMM_PPC64LE_INSTR_XVMULDP;
      l_vec_fma = LIBXSMM_PPC64LE_INSTR_XVMADDADP;
    }  break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  for ( unsigned int l_n = 0; l_n < i_n; ++l_n ) {
    for ( unsigned int l_m = 0; l_m < i_m; ++l_m ) {
      for ( unsigned int l_k = 0; l_k < i_k; ++l_k ) {
        if ( ( l_k == 0 ) && ( i_beta == 0 ) ) {
          libxsmm_ppc64le_instr_6( io_generated_code,
                                   l_vec_mul,
                                   io_c[l_m + l_n*i_ldc],
                                   i_a[l_m + l_k*i_lda],
                                   i_b[l_k + l_n*i_ldb],
                                   (0x0020 & i_a[l_m + l_k*i_lda]) >> 5,
                                   (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                   (0x0020 & io_c[l_m + l_n*i_ldc]) >> 5 );
        } else {
          libxsmm_ppc64le_instr_6( io_generated_code,
                                   l_vec_fma,
                                   io_c[l_m + l_n*i_ldc],
                                   i_a[l_m + l_k*i_lda],
                                   i_b[l_k + l_n*i_ldb],
                                   (0x0020 & i_a[l_m + l_k*i_lda]) >> 5,
                                   (0x0020 & i_b[l_k + l_n*i_ldb]) >> 5,
                                   (0x0020 & io_c[l_m + l_n*i_ldc]) >> 5 );
        }
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
                                        unsigned char const            i_a,
                                        unsigned char const            i_b,
                                        unsigned char const            i_c ) {

  libxsmm_datatype l_a_datatype = LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_b_datatype = LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_c_datatype = LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype );
  libxsmm_datatype l_comptype = LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype );
  unsigned int l_v_len = i_blocking->vector_len_comp;

  unsigned int l_n_k_blocks = i_xgemm_desc->k / i_blocking->block_k;
  unsigned int l_packed = ( ( i_blocking->block_k % l_v_len ) == 0 &&
                            ( i_xgemm_desc->k % i_blocking->block_k ) == 0 ) ? 1 : 0;

  /* Local pointers registers for A and B */
  unsigned int l_a, l_b;

  if ( l_n_k_blocks > 1 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    l_b = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_a, l_a, i_a );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_b, l_b, i_b );
  } else {
    l_a = i_a;
    l_b = i_b;
  }

  /* Allocate vector registers */
  unsigned l_a_reg[i_blocking->n_reg_a], l_b_reg[i_blocking->n_reg_b], l_c_reg[i_blocking->n_reg_c];

  for ( unsigned int l_i = 0; l_i < i_blocking->n_reg_a; ++l_i ) {
    l_a_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( unsigned int l_i = 0; l_i < i_blocking->n_reg_b; ++l_i ) {
    l_b_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }
  for ( unsigned int l_i = 0; l_i < i_blocking->n_reg_c; ++l_i ) {
    l_c_reg[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  unsigned int l_beta_zero = ( i_xgemm_desc->flags & 0x0004 ) >> 2;

  /* Load C if required */
  if ( !l_beta_zero ) {
    libxsmm_generator_gemm_vsx_mk_load_vsr( io_generated_code,
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

  for ( unsigned int l_k_block = 0; l_k_block < l_n_k_blocks; ++l_k_block) {
    /* Load block of A */
    libxsmm_generator_gemm_vsx_mk_load_vsr( io_generated_code,
                                            io_reg_tracker,
                                            l_a_datatype,
                                            l_comptype,
                                            l_a,
                                            i_blocking->block_m,
                                            i_blocking->block_k,
                                            i_xgemm_desc->lda,
                                            l_a_reg,
                                            i_blocking->reg_lda );

    /* Load block of B and broadcast values to vector */
    libxsmm_generator_gemm_vsx_mk_load_vsr_splat( io_generated_code,
                                                  io_reg_tracker,
                                                  l_b_datatype,
                                                  l_comptype,
                                                  l_b,
                                                  i_blocking->block_k,
                                                  i_blocking->block_n,
                                                  i_xgemm_desc->ldb,
                                                  l_b_reg,
                                                  i_blocking->reg_ldb );

    /* block FMA */
    unsigned int l_beta = ( ( l_k_block == 0 ) && ( l_beta_zero ) ) ? 0 : 1;
    libxsmm_generator_vsx_block_fma_b_splat( io_generated_code,
                                             l_comptype,
                                             ( i_blocking->block_m + l_v_len - 1 ) / l_v_len,
                                             i_blocking->block_n,
                                             i_blocking->block_k,
                                             l_a_reg,
                                             i_blocking->reg_lda,
                                             l_b_reg,
                                             i_blocking->reg_ldb,
                                             l_beta,
                                             l_c_reg,
                                             i_blocking->reg_ldc );

    /* Increament if not last or not packed */
    if ( l_k_block < l_n_k_blocks - 1 || !l_packed ) {
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
  /* Partial dot product */
  if ( !l_packed ) {
    unsigned int l_k_part = i_xgemm_desc->k % i_blocking->block_k;
    /* Load block of A */
    libxsmm_generator_gemm_vsx_mk_load_vsr( io_generated_code,
                                            io_reg_tracker,
                                            l_a_datatype,
                                            l_comptype,
                                            l_a,
                                            i_blocking->block_m,
                                            l_k_part,
                                            i_xgemm_desc->lda,
                                            l_a_reg,
                                            i_blocking->reg_lda );

    /* Load block of B and broadcast values to vector */
    libxsmm_generator_gemm_vsx_mk_load_vsr_splat( io_generated_code,
                                                  io_reg_tracker,
                                                  l_b_datatype,
                                                  l_comptype,
                                                  l_b,
                                                  l_k_part,
                                                  i_blocking->block_n,
                                                  i_xgemm_desc->ldb,
                                                  l_b_reg,
                                                  i_blocking->reg_ldb );

    /* block FMA */
    unsigned int l_beta = ( ( l_n_k_blocks == 0 ) && ( l_beta_zero ) ) ? 0 : 1;
    libxsmm_generator_vsx_block_fma_b_splat( io_generated_code,
                                             l_comptype,
                                             ( i_blocking->block_m + l_v_len - 1 ) / l_v_len,
                                             i_blocking->block_n,
                                             l_k_part,
                                             l_a_reg,
                                             i_blocking->reg_lda,
                                             l_b_reg,
                                             i_blocking->reg_ldb,
                                             l_beta,
                                             l_c_reg,
                                             i_blocking->reg_ldc );
  }


  /* Store result block in C */
  libxsmm_generator_gemm_vsx_mk_store_vsr( io_generated_code,
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
  for ( unsigned int l_i = 0; l_i < i_blocking->n_reg_a; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_a_reg[l_i] );
  }
  for ( unsigned int l_i = 0; l_i < i_blocking->n_reg_b; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_b_reg[l_i] );
  }

  if ( l_n_k_blocks > 1 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a );
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_b );
  }

  /* Free C registers */
  for ( unsigned int l_i = 0; l_i < i_blocking->n_reg_c; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_c_reg[l_i] );
  }

  return;
}
