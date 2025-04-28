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

#include "generator_ppc64le_instructions.h"

/*
 * Source:
 *   Power ISA
 *   Version 3.1
 */


LIBXSMM_API_INTERN
libxsmm_ppc64le_reg libxsmm_ppc64le_reg_init() {
  libxsmm_ppc64le_reg reg_tracker = LIBXSMM_PPC64LE_REG_DEFAULT;
  return reg_tracker;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_get_reg( libxsmm_generated_code  *io_generated_code,
                                      libxsmm_ppc64le_reg     *io_reg_tracker,
                                      libxsmm_ppc64le_reg_type i_reg_type ) {
  int i, n_reg;

  switch (i_reg_type) {
    case LIBXSMM_PPC64LE_GPR: {
      n_reg = LIBXSMM_PPC64LE_GPR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_FPR: {
      n_reg = LIBXSMM_PPC64LE_FPR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_VR: {
      n_reg = LIBXSMM_PPC64LE_VR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_VSR: {
      n_reg = LIBXSMM_PPC64LE_VSR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_ACC: {
      n_reg = LIBXSMM_PPC64LE_ACC_NMAX;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return 0xfffffff;
    }
  }

  for ( i = n_reg - 1; i >= 0; --i ) {
    if ( libxsmm_ppc64le_isfree_reg( io_generated_code, io_reg_tracker, i_reg_type, i) ) {
      libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, i_reg_type, i);
      return i;
    }
  }

  LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  return 0xffffffff;
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_alloc_reg( libxsmm_generated_code  *io_generated_code,
                                libxsmm_ppc64le_reg     *io_reg_tracker,
                                libxsmm_ppc64le_reg_type i_reg_type,
                                unsigned int             i_n,
                                unsigned int             i_contig,
                                unsigned int            *i_a ) {
  unsigned int i, j;

  if ( 1 == i_contig ) {
    for ( i = 0 ; i < i_n ; ++i ) {
      i_a[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, i_reg_type );
    }
  } else {
    unsigned int l_groups = i_n / i_contig;
    for ( i = 0 ; i < l_groups ; ++i ) {
      unsigned int l_reg[LIBXSMM_PPC64LE_REG_NMAX];
      libxsmm_ppc64le_get_sequential_reg( io_generated_code, io_reg_tracker, i_reg_type, i_contig, l_reg );
      for ( j = 0 ; j < i_contig ; ++j ) {
        i_a[i_contig*i + j] = l_reg[j];
      }
    }
    for ( i = 0 ; i < i_n % i_contig ; ++i ) {
      i_a[i_contig*l_groups + i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, i_reg_type );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_alloc_mat( libxsmm_generated_code    *io_generated_code,
                                libxsmm_ppc64le_reg       *io_reg_tracker,
                                libxsmm_ppc64le_alloc_type i_type,
                                libxsmm_ppc64le_reg_type   i_reg_type,
                                unsigned int const         i_n_rows,
                                unsigned int const         i_n_cols,
                                unsigned int              *o_reg,
                                unsigned int const         i_ld ) {
  unsigned int i, j;

  switch ( i_type ) {
    case LIBXSMM_PPC64LE_ALLOC_NONE: {
      libxsmm_ppc64le_alloc_reg( io_generated_code, io_reg_tracker, i_reg_type, i_n_rows*i_n_cols, 1, o_reg );
    } break;
    case LIBXSMM_PPC64LE_ALLOC_COL_PAIR: {
      for ( j = 0; j < i_n_rows; ++j ) {
        for ( i = 0; i < i_n_cols / 2 ; ++i ) {
          unsigned int l_rpair[2];
          libxsmm_ppc64le_get_sequential_reg( io_generated_code, io_reg_tracker, i_reg_type, 2, l_rpair );
          /* Order is reversed for GER */
          o_reg[j + (2*i + 0)*i_ld] = l_rpair[1];
          o_reg[j + (2*i + 1)*i_ld] = l_rpair[0];
        }
        if ( 1 == i_n_cols % 2 ) {
          o_reg[j + (i_n_cols - 1)*i_ld] =  libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, i_reg_type );
        }
      }
    } break;
    case LIBXSMM_PPC64LE_ALLOC_ROW_PAIR: {
      for ( i = 0; i < i_n_cols ; ++i ) {
        for ( j = 0; j < i_n_rows / 2 ; ++j ) {
          unsigned int l_rpair[2];
          libxsmm_ppc64le_get_sequential_reg( io_generated_code, io_reg_tracker, i_reg_type, 2, l_rpair );
          /* Order is reversed for pair load */
          o_reg[2*j + 0 + i*i_ld] = l_rpair[1];
          o_reg[2*j + 1 + i*i_ld] = l_rpair[0];
        }
        if ( 1 == i_n_rows % 2 ) {
          o_reg[i_n_rows - 1 + i*i_ld] =  libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, i_reg_type );
        }
      }
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_ptr_reg_alloc( libxsmm_generated_code *io_generated_code,
                                    libxsmm_ppc64le_reg    *io_reg_tracker,
                                    unsigned int            i_a,
                                    unsigned int            i_n,
                                    unsigned int            i_ld,
                                    unsigned int            i_max_add,
                                    unsigned int           *o_ptr,
                                    long                   *o_offset ) {
  unsigned int i, j;

  o_ptr[0] = i_a;
  o_offset[0] = 0;
  for ( i = 1 ; i < i_n ; ++i ) {
    int l_new_required = 1;

    for ( j = 0; j < i ; ++j ) {
      long l_rel_offset = o_offset[j] + (i - j)*i_ld;
      long l_max_offset = l_rel_offset + i_max_add;

      if ( ( ( l_rel_offset % 16 ) == 0 ) && ( l_max_offset < 0x7fff ) ) {
        o_ptr[i] = o_ptr[j];
        o_offset[i] = l_rel_offset;
        l_new_required = 0;
        break;
      }
    }

    if ( 1 == l_new_required ) {
      long l_delta0;
      int l_shift;

      /* Try to make a shift that doesn't require extended addition */
      l_shift = 0x7ff0 - (o_offset[0] + i*i_ld);
      l_shift -= l_shift % 16;
      if ( 0 > l_shift ) {
        l_shift = 0x7ff0;
      }

      l_delta0 = o_offset[0] + i*i_ld + l_shift;
      o_offset[i] = -l_shift ;
      o_ptr[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
      libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, o_ptr[0], o_ptr[i], l_delta0 );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_ptr_reg_free( libxsmm_generated_code *io_generated_code,
                                   libxsmm_ppc64le_reg    *io_reg_tracker,
                                   unsigned int           *i_ptr,
                                   unsigned int            i_n ) {
  unsigned int i, l_base_ptr = i_ptr[0];
  for ( i = 1 ; i < i_n ; ++i ) {
    if ( i_ptr[i] != l_base_ptr ) {
      libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, i_ptr[i] );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_get_sequential_reg( libxsmm_generated_code  *io_generated_code,
                                         libxsmm_ppc64le_reg     *io_reg_tracker,
                                         libxsmm_ppc64le_reg_type i_reg_type,
                                         unsigned int             i_n,
                                         unsigned int            *o_reg ) {
  char l_alloc = 0;
  int i = 0, j, l_n_reg;
  switch (i_reg_type) {
    case LIBXSMM_PPC64LE_GPR: {
      l_n_reg = LIBXSMM_PPC64LE_GPR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_FPR: {
      l_n_reg = LIBXSMM_PPC64LE_FPR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_VR: {
      l_n_reg = LIBXSMM_PPC64LE_VR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_VSR: {
      l_n_reg = LIBXSMM_PPC64LE_VSR_NMAX;
    } break;
    case LIBXSMM_PPC64LE_ACC: {
      l_n_reg = LIBXSMM_PPC64LE_ACC_NMAX;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  while ( i <= (int)(l_n_reg - i_n) && 0 == l_alloc ) {
    char  l_search = 1;
    for ( j = 0; j < (int)i_n && 1 == l_search; ++j ) {
      if ( !libxsmm_ppc64le_isfree_reg( io_generated_code, io_reg_tracker, i_reg_type, i + j ) ) {
        i += j + 1;
        l_search = 0;
      }
    }
    l_alloc = l_search;

    /* Ensure register index is aligned */
    if ( 0 == l_alloc ) {
      i += ( i % i_n );
    }
  }

  if ( 1 == l_alloc ) {
    for ( j = 0 ; j < (int)i_n ; ++j ) {
      libxsmm_ppc64le_used_reg( io_generated_code, io_reg_tracker, i_reg_type, j + i );
      o_reg[j] = (unsigned int)(j + i);
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}


LIBXSMM_API_INTERN
char libxsmm_ppc64le_isfree_reg( libxsmm_generated_code  *io_generated_code,
                                 libxsmm_ppc64le_reg     *io_reg_tracker,
                                 libxsmm_ppc64le_reg_type i_reg_type,
                                 unsigned int             i_reg ) {
  char o_is_free = 0;
  switch (i_reg_type) {
    case LIBXSMM_PPC64LE_GPR: {
      if ( ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->gpr[i_reg] ) ) {
        o_is_free = 1;
      }
    } break;
    case LIBXSMM_PPC64LE_FPR: {
      if ( ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->fpr[i_reg] ) &&
           ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->vsr[i_reg] ) &&
           ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->acc[i_reg / 4] ) ) {
        o_is_free = 1;
      }
    } break;
    case LIBXSMM_PPC64LE_VR: {
      if ( ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->vr[i_reg]) &&
           ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->vsr[i_reg + LIBXSMM_PPC64LE_FPR_NMAX] ) ) {
        o_is_free = 1;
      }
    } break;
    case LIBXSMM_PPC64LE_VSR: {
      if ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->vsr[i_reg] ) {
        if ( LIBXSMM_PPC64LE_FPR_NMAX > i_reg ) {
          if ( ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->fpr[i_reg] ) &&
               ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->acc[i_reg / 4] ) ) {
            o_is_free = 1;
          }
        } else if ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->vr[i_reg - LIBXSMM_PPC64LE_FPR_NMAX] ) {
          o_is_free = 1;
        }
      }
    } break;
    case LIBXSMM_PPC64LE_ACC: {
      if ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->acc[i_reg] ) {
        int i;
        o_is_free = 1;
        for ( i = 0; i < 4; ++i ) {
          o_is_free &= ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->fpr[i_reg*4 + i] );
          o_is_free &= ( LIBXSMM_PPC64LE_REG_FREE <= io_reg_tracker->vsr[i_reg*4 + i] );
        }
      }
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  }
  return o_is_free;
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_set_reg( libxsmm_generated_code  *io_generated_code,
                              libxsmm_ppc64le_reg     *io_reg_tracker,
                              libxsmm_ppc64le_reg_type i_reg_type,
                              unsigned int             i_reg,
                              libxsmm_ppc64le_reg_util i_value ) {
  unsigned int i;

  switch ( i_reg_type ) {
    case LIBXSMM_PPC64LE_GPR: {
      io_reg_tracker->gpr[i_reg] = i_value;
    } break;
    case LIBXSMM_PPC64LE_FPR: {
      io_reg_tracker->fpr[i_reg] = i_value;
      io_reg_tracker->vsr[i_reg] = i_value;
    } break;
    case LIBXSMM_PPC64LE_VR: {
      io_reg_tracker->vr[i_reg] =  i_value;
      io_reg_tracker->vsr[i_reg + LIBXSMM_PPC64LE_FPR_NMAX] = i_value;
    } break;
    case LIBXSMM_PPC64LE_VSR: {
      io_reg_tracker->vsr[i_reg] = i_value;
      if ( LIBXSMM_PPC64LE_FPR_NMAX > i_reg ) {
        io_reg_tracker->fpr[i_reg] = i_value;
      } else {
        io_reg_tracker->vr[i_reg - LIBXSMM_PPC64LE_FPR_NMAX] = i_value;
      }
    } break;
    case LIBXSMM_PPC64LE_ACC: {
      io_reg_tracker->acc[i_reg] = i_value;
      for ( i = 0; i < 4; ++i ) {
        io_reg_tracker->fpr[i_reg*4 + i] = i_value;
        io_reg_tracker->vsr[i_reg*4 + i] = i_value;
      }
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_used_reg( libxsmm_generated_code  *io_generated_code,
                               libxsmm_ppc64le_reg     *io_reg_tracker,
                               libxsmm_ppc64le_reg_type i_reg_type,
                               unsigned int             i_reg ) {
  libxsmm_ppc64le_set_reg( io_generated_code, io_reg_tracker, i_reg_type, i_reg, LIBXSMM_PPC64LE_REG_USED );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_free_reg( libxsmm_generated_code  *io_generated_code,
                               libxsmm_ppc64le_reg     *io_reg_tracker,
                               libxsmm_ppc64le_reg_type i_reg_type,
                               unsigned int const       i_reg ) {
  libxsmm_ppc64le_set_reg( io_generated_code, io_reg_tracker, i_reg_type, i_reg, LIBXSMM_PPC64LE_REG_ALTD );
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_b_form( unsigned int  i_instr,
                                           unsigned char i_bo,
                                           unsigned char i_bi,
                                           unsigned int  i_bd ) {
  unsigned int l_instr = i_instr;

  /* set BO */
  l_instr |= (unsigned int)( (0x1f & i_bo) << (31 - 6 - 4) );
  /* set BI */
  l_instr |= (unsigned int)( (0x1f & i_bi) << (31 - 11 - 4) );
  /* set BD */
  l_instr |= (unsigned int)( (0x00003fff & i_bd) << (31 - 16 - 13) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_b_form_al( unsigned int  i_instr,
                                              unsigned char i_bo,
                                              unsigned char i_bi,
                                              unsigned int  i_bd,
                                              unsigned char i_aa,
                                              unsigned char i_lk ) {
  unsigned int l_instr = i_instr;

  /* set BO */
  l_instr |= (unsigned int)( (0x1f & i_bo) << (31 - 6 - 4) );
  /* set BI */
  l_instr |= (unsigned int)( (0x1f & i_bi) << (31 - 11 - 4) );
  /* set BD */
  l_instr |= (unsigned int)( (0x00003fff & i_bd) << (31 - 16 - 13) );
  /* set AA */
  l_instr |= (unsigned int)( (0x01 & i_aa) << (31 - 30 - 0) );
  /* set LK */
  l_instr |= (unsigned int)( (0x01 & i_lk) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_d_form( unsigned int  i_instr,
                                           unsigned char i_t,
                                           unsigned char i_a,
                                           unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* set D */
  l_instr |= (unsigned int)( (0x0000ffff & i_d) << (31 - 16 - 15) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_d_form_bf( unsigned int  i_instr,
                                              unsigned char i_bf,
                                              unsigned char i_l,
                                              unsigned char i_a,
                                              unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set BF */
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 6 - 2) );
  /* set L */
  l_instr |= (unsigned int)( (0x01 & i_l) << (31 - 10 - 0) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* set D */
  l_instr |= (unsigned int)( (0x0000ffff & i_d) << (31 - 16 - 15) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_dq_form_p( unsigned int  i_instr,
                                              unsigned char i_tp,
                                              unsigned char i_tx,
                                              unsigned char i_ra,
                                              unsigned int  i_dq ) {
  unsigned int l_instr = i_instr;

  /* set Tp */
  l_instr |= (unsigned int)( (0x0f & i_tp) << (31 - 6 - 3) );
  /* set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 10 - 0) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31 - 11 - 4) );
  /* set DQ */
  l_instr |= (unsigned int)( (0x0fff & i_dq) << (31 - 16 - 11) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_dq_form_x( unsigned int  i_instr,
                                              unsigned char i_t,
                                              unsigned char i_ra,
                                              unsigned int  i_dq,
                                              unsigned char i_x ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31 - 11 - 4) );
  /* set DQ */
  l_instr |= (unsigned int)( (0x0fff & i_dq) << (31 - 16 - 11) );
  /* set X */
  l_instr |= (unsigned int)( (0x01 & i_x) << (31 - 28 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_ds_form( unsigned int  i_instr,
                                            unsigned char i_s,
                                            unsigned char i_a,
                                            unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set S */
  l_instr |= (unsigned int)( (0x1f & i_s) << (31 - 6 - 4) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* set D */
  l_instr |= (unsigned int)( (0x3fff & i_d) << (31 - 16 - 13) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_m_form( unsigned int  i_instr,
                                           unsigned char i_rs,
                                           unsigned char i_ra,
                                           unsigned char i_sh,
                                           unsigned char i_mb,
                                           unsigned char i_me,
                                           unsigned char i_rc ) {
  unsigned int l_instr = i_instr;

  /* Set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31 - 6 - 4) );
  /* Set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31 - 11 - 4) );
  /* Set SH */
  l_instr |= (unsigned int)( (0x1f & i_sh) << (31 - 16 - 4) );
  /* Set MB */
  l_instr |= (unsigned int)( (0x3f & i_mb) << (31 - 21 - 4) );
  /* Set ME */
  l_instr |= (unsigned int)( (0x01 & i_me) << (31 - 26 - 4) );
  /* Set Rc */
  l_instr |= (unsigned int)( (0x01 & i_rc) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_md_form( unsigned int  i_instr,
                                            unsigned char i_rs,
                                            unsigned char i_ra,
                                            unsigned char i_sh,
                                            unsigned char i_m,
                                            unsigned char i_sh2,
                                            unsigned char i_rc ) {
  unsigned int l_instr = i_instr;

  /* Set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31 - 6 - 4) );
  /* Set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31 - 11 - 4) );
  /* Set sh */
  l_instr |= (unsigned int)( (0x1f & i_sh) << (31 - 16 - 4) );
  /* Set mb/me */
  l_instr |= (unsigned int)( (0x3f & i_m) << (31 - 21 - 5) );
  /* Set sh */
  l_instr |= (unsigned int)( (0x01 & i_sh2) << (31 - 30 - 0) );
  /* Set Rc */
  l_instr |= (unsigned int)( (0x01 & i_rc) << (31 - 31 - 0) );

  return l_instr;

}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_va_form( unsigned int  i_instr,
                                            unsigned char i_rt,
                                            unsigned char i_ra,
                                            unsigned char i_rb,
                                            unsigned char i_rc ) {
  unsigned int l_instr = i_instr;

  /* Set RT */
  l_instr |= (unsigned int)( (0x1f & i_rt) << (31 - 6 - 4) );
  /* Set RA */
  l_instr |= (unsigned int)( (0x1f & i_rc) << (31 - 11 - 4) );
  /* Set RB */
  l_instr |= (unsigned int)( (0x1f & i_rb) << (31 - 16 - 4) );
  /* Set RC */
  l_instr |= (unsigned int)( (0x1f & i_rc) << (31 - 21 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_vx_form_vrb( unsigned int  i_instr,
                                                unsigned char i_vrb ) {
  unsigned int l_instr = i_instr;

  /* Set VRB */
  l_instr |= (unsigned int)( (0x1f & i_vrb) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_vx_form_vrt( unsigned int  i_instr,
                                                unsigned char i_vrt ) {
  unsigned int l_instr = i_instr;

  /* Set VRT */
  l_instr |= (unsigned int)( (0x1f & i_vrt) << (31 - 6 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form( unsigned int  i_instr,
                                           unsigned char i_t,
                                           unsigned char i_a,
                                           unsigned char i_b,
                                           unsigned char i_x ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set X */
  l_instr |= (unsigned int)( (0x01 & i_x) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_3( unsigned int  i_instr,
                                             unsigned char i_a ) {
  unsigned int l_instr = i_instr;

  /* Set A */
  l_instr |= (unsigned int)( (0x07 & i_a) << (31 - 6 - 2) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_33( unsigned int  i_instr,
                                              unsigned char i_bf,
                                              unsigned char i_bfa ) {
  unsigned int l_instr = i_instr;

  /* Set BF */
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 6 - 2) );
  /* Set BFA */
  l_instr |= (unsigned int)( (0x07 & i_bfa) << (31 - 11 - 2) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_355( unsigned int  i_instr,
                                               unsigned char i_bf,
                                               unsigned char i_a,
                                               unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set BF */
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 6 - 2) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_355_l( unsigned int  i_instr,
                                                 unsigned char i_l,
                                                 unsigned char i_a,
                                                 unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set L */
  l_instr |= (unsigned int)( (0x07 & i_l) << (31 - 8 - 2) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_4155( unsigned int  i_instr,
                                                unsigned char i_t,
                                                unsigned char i_x,
                                                unsigned char i_a,
                                                unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x0f & i_t) << (31 - 6 - 3) );
  /* Set X */
  l_instr |= (unsigned int)( (0x01 & i_x) << (31 - 10 - 0) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_55( unsigned int  i_instr,
                                              unsigned char i_a,
                                              unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_555( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_581( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_imm,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set IMM8 */
  l_instr |= (unsigned int)( (0xff & i_imm) << (31 - 13 - 7) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xl_form_2( unsigned int  i_instr,
                                              unsigned char i_ba,
                                              unsigned char i_bfa ) {
  unsigned int l_instr = i_instr;

  /* Set BA */
  l_instr |= (unsigned int)( (0x07 & i_ba) << (31 - 6 - 2) );
  /* Set BFA */
  l_instr |= (unsigned int)( (0x07 & i_bfa) << (31 - 11 - 2) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xl_form_4( unsigned int  i_instr,
                                              unsigned char i_bo,
                                              unsigned char i_bi,
                                              unsigned char i_bh,
                                              unsigned char i_lk ) {
  unsigned int l_instr = i_instr;

  /* Set BO */
  l_instr |= (unsigned int)( (0x1f & i_bo) << (31 - 6 - 4) );
  /* Set BI */
  l_instr |= (unsigned int)( (0x1f & i_bi) << (31 - 11 - 4) );
  /* Set BH */
  l_instr |= (unsigned int)( (0x03 & i_bh) << (31 - 19 - 1) );
  /* Set LK */
  l_instr |= (unsigned int)( (0x01 & i_lk) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xfx_form_2( unsigned int  i_instr,
                                               unsigned char i_rs,
                                               unsigned char i_fxm ) {
  unsigned int l_instr = i_instr;

  /* Set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31 - 6 - 4) );
  /* Set FXM */
  l_instr |= (unsigned int)( (0xff & i_fxm) << (31 - 12 - 7) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xfx_form_4( unsigned int  i_instr,
                                               unsigned char i_rs,
                                               unsigned int  i_spr ) {
  unsigned int l_instr = i_instr;

  /* Set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31 - 6 - 4) );
  /* Set spr */
  l_instr |= (unsigned int)( (0x03ff & i_spr) << (31 - 11 - 9) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xfx_form_5( unsigned int  i_instr,
                                               unsigned char i_rt ) {
  unsigned int l_instr = i_instr;

  /* Set  RT */
  l_instr |= (unsigned int)( (0x1f & i_rt) << (31 - 6 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_2( unsigned int  i_instr,
                                               unsigned char i_rt,
                                               unsigned char i_b,
                                               unsigned char i_bx ) {
  unsigned int l_instr = i_instr;

  /* Set RT */
  l_instr |= (unsigned int)( (0x1f & i_rt) << (31 - 6 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_3( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_b,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_4( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_uim,
                                               unsigned char i_b,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set UIM */
  l_instr |= (unsigned int)( (0x03 & i_uim) << (31 - 14 - 1) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned char i_a,
                                             unsigned char i_b) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_0( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_ax,
                                               unsigned char i_bx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x07 & i_t) << (31 - 6 - 2) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_3( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_w,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set W */
  l_instr |= (unsigned int)( (0x03 & i_w) << (31 - 22 - 1) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_5( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_rc,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set RC */
  l_instr |= (unsigned int)( (0x01 & i_rc) << (31 - 21 - 0) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_6( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx4_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned char i_a,
                                             unsigned char i_b,
                                             unsigned char i_c,
                                             unsigned char i_cx,
                                             unsigned char i_ax,
                                             unsigned char i_bx,
                                             unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set C */
  l_instr |= (unsigned int)( (0x1f & i_c) << (31 - 21 - 4) );
  /* Set CX */
  l_instr |= (unsigned int)( (0x01 & i_cx) << (31 - 28 - 0) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_8ls( unsigned long i_instr,
                                                unsigned char i_r,
                                                unsigned int  i_d0,
                                                unsigned char i_tx,
                                                unsigned char i_t,
                                                unsigned char i_a,
                                                unsigned int  i_d1 ) {
  unsigned long l_instr = i_instr;

  /* Set R */
  l_instr |= (unsigned long)(0x01 & i_r) << (32 + 31 - 11 - 0);
  /* Set d0 */
  l_instr |= (unsigned long)(0x0003ffff & i_d0) << (32 + 31 - 14 - 17);
  /* Set TX */
  l_instr |= (unsigned long)(0x01 & i_tx) << (31 - 5 - 0);
  /* Set T */
  l_instr |= (unsigned long)(0x1f & i_t) << (31 - 6 - 4);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set d1 */
  l_instr |= (unsigned long)(0x0000ffff & i_d1) << (31 - 16 - 15);
  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_8lsp( unsigned long i_instr,
                                                 unsigned char i_r,
                                                 unsigned int  i_d0,
                                                 unsigned char i_tp,
                                                 unsigned char i_tx,
                                                 unsigned char i_a,
                                                 unsigned int  i_d1 ) {
  unsigned long l_instr = i_instr;

  /* Set R */
  l_instr |= (unsigned long)(0x01 & i_r) << (32 + 31 - 11 - 0);
  /* Set d0 */
  l_instr |= (unsigned long)(0x03ffff & i_d0) << (32 + 31 - 14 - 17);
  /* Set Tp */
  l_instr |= (unsigned long)(0x0f & i_tp) << (31 - 6 - 3);
  /* Set TX */
  l_instr |= (unsigned long)(0x01 & i_tx) << (31 - 10 - 0);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set d1 */
  l_instr |= (unsigned long)(0xffff & i_d1) << (31 - 16 - 15);
  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_mls( unsigned long i_instr,
                                                unsigned char i_r,
                                                unsigned int  i_si0,
                                                unsigned char i_t,
                                                unsigned char i_a,
                                                unsigned int  i_si1 ) {
  unsigned long l_instr = i_instr;

  /* Set R */
  l_instr |= (unsigned long)(0x01 & i_r) << (32 + 31 - 11 - 0);
  /* Set SI0 */
  l_instr |= (unsigned long)(0x0003ffff & i_si0) << (32 + 31 - 14 - 17);
  /* Set T */
  l_instr |= (unsigned long)(0x1f & i_t) << (31 - 6 - 4);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set SI1 */
  l_instr |= (unsigned long)(0x0000ffff & i_si1) << (31 - 16 - 15);
  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_0_8rr3( unsigned long i_instr,
                                                   unsigned int  i_imm0,
                                                   unsigned char i_t,
                                                   unsigned char i_ix,
                                                   unsigned char i_tx,
                                                   unsigned int  i_imm1 ) {
  unsigned long l_instr = i_instr;

  /* Set IMM0 */
  l_instr |= (unsigned long)(0x0000ffff & i_imm0) << (32 + 31 - 16 - 15);
  /* Set T */
  l_instr |= (unsigned long)(0x1f & i_t) << (31 - 6 - 4);
  /* Set IX */
  l_instr |= (unsigned long)(0x01 & i_ix) << (31 - 14 - 0);
  /* Set TX */
  l_instr |= (unsigned long)(0x01 & i_tx) << (31 - 15 - 0);
  /* Set IMM1 */
  l_instr |= (unsigned long)(0x0000ffff & i_imm0) << (31 - 16 - 15);

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_1_8rr3( unsigned long i_instr,
                                                   unsigned int  i_imm0,
                                                   unsigned char i_t,
                                                   unsigned char i_tx,
                                                   unsigned int  i_imm1 ) {
  unsigned long l_instr = i_instr;

  /* Set IMM0 */
  l_instr |= (unsigned long)(0x0000ffff & i_imm0) << (32 + 31 - 16 - 15);
  /* Set T */
  l_instr |= (unsigned long)(0x1f & i_t) << (31 - 6 - 4);
  /* Set TX */
  l_instr |= (unsigned long)(0x01 & i_tx) << (31 - 15 - 0);
  /* Set IMM1 */
  l_instr |= (unsigned long)(0x0000ffff & i_imm0) << (31 - 16 - 15);

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx3_form_0_mmirr0( unsigned long i_instr,
                                                       unsigned char i_xmsk,
                                                       unsigned char i_ymsk,
                                                       unsigned char i_at,
                                                       unsigned char i_a,
                                                       unsigned char i_b,
                                                       unsigned char i_ax,
                                                       unsigned char i_bx ) {
  unsigned long l_instr = i_instr;

  /* Set XMSK */
  l_instr |= (unsigned long)(0x0f & i_xmsk) << (32 + 31 - 24 - 3);
  /* Set YMSK */
  l_instr |= (unsigned long)(0x0f & i_ymsk) << (32 + 31 - 28 - 3);
  /* Set AT */
  l_instr |= (unsigned long)(0x07 & i_at) << (31 - 6 - 2);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set B */
  l_instr |= (unsigned long)(0x1f & i_b) << (31 - 16 - 4);
  /* Set AX */
  l_instr |= (unsigned long)(0x01 & i_ax) << (31 - 29 - 0);
  /* Set BX */
  l_instr |= (unsigned long)(0x01 & i_bx) << (31 - 30 - 0);

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx3_form_0_mmirr1( unsigned long i_instr,
                                                       unsigned char i_xmsk,
                                                       unsigned char i_ymsk,
                                                       unsigned char i_at,
                                                       unsigned char i_a,
                                                       unsigned char i_b,
                                                       unsigned char i_ax,
                                                       unsigned char i_bx ) {
  unsigned long l_instr = i_instr;

  /* Set XMSK */
  l_instr |= (unsigned long)(0x0f & i_xmsk) << (32 + 31 - 24 - 3);
  /* Set YMSK */
  l_instr |= (unsigned long)(0x03 & i_ymsk) << (32 + 31 - 28 - 1);
  /* Set AT */
  l_instr |= (unsigned long)(0x07 & i_at) << (31 - 6 - 2);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set B */
  l_instr |= (unsigned long)(0x1f & i_b) << (31 - 16 - 4);
  /* Set AX */
  l_instr |= (unsigned long)(0x01 & i_ax) << (31 - 29 - 0);
  /* Set BX */
  l_instr |= (unsigned long)(0x01 & i_bx) << (31 - 30 - 0);

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx3_form_0_mmirr3( unsigned long i_instr,
                                                       unsigned char i_pmsk,
                                                       unsigned char i_xmsk,
                                                       unsigned char i_ymsk,
                                                       unsigned char i_at,
                                                       unsigned char i_a,
                                                       unsigned char i_b,
                                                       unsigned char i_ax,
                                                       unsigned char i_bx ) {
  unsigned long l_instr = i_instr;

  /* Set PMSK */
  l_instr |= (unsigned long)(0x03 & i_pmsk) << (32 + 31 - 16 - 1);
  /* Set XMSK */
  l_instr |= (unsigned long)(0x0f & i_xmsk) << (32 + 31 - 24 - 3);
  /* Set YMSK */
  l_instr |= (unsigned long)(0x0f & i_ymsk) << (32 + 31 - 28 - 3);
  /* Set AT */
  l_instr |= (unsigned long)(0x07 & i_at) << (31 - 6 - 2);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set B */
  l_instr |= (unsigned long)(0x1f & i_b) << (31 - 16 - 4);
  /* Set AX */
  l_instr |= (unsigned long)(0x01 & i_ax) << (31 - 29 - 0);
  /* Set BX */
  l_instr |= (unsigned long)(0x01 & i_bx) << (31 - 30 - 0);

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx4_form_8rr0( unsigned long i_instr,
                                                   unsigned char i_t,
                                                   unsigned char i_a,
                                                   unsigned char i_b,
                                                   unsigned char i_c,
                                                   unsigned char i_cx,
                                                   unsigned char i_ax,
                                                   unsigned char i_bx,
                                                   unsigned char i_tx ) {
  unsigned long l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned long)(0x1f & i_t) << (31 - 6 - 4);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set B */
  l_instr |= (unsigned long)(0x1f & i_b) << (31 - 16 - 4);
  /* Set C */
  l_instr |= (unsigned long)(0x1f & i_c) << (31 - 21 - 4);
  /* Set CX */
  l_instr |= (unsigned long)(0x01 & i_cx) << (31 - 28 - 0);
  /* Set AX */
  l_instr |= (unsigned long)(0x01 & i_ax) << (31 - 29 - 0);
  /* Set BX */
  l_instr |= (unsigned long)(0x01 & i_bx) << (31 - 30 - 0);
  /* Set TX */
  l_instr |= (unsigned long)(0x01 & i_tx) << (31 - 31 - 0);

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx4_form_8rr2( unsigned long i_instr,
                                                   unsigned char i_imm,
                                                   unsigned char i_t,
                                                   unsigned char i_a,
                                                   unsigned char i_b,
                                                   unsigned char i_c,
                                                   unsigned char i_cx,
                                                   unsigned char i_ax,
                                                   unsigned char i_bx,
                                                   unsigned char i_tx ) {
  unsigned long l_instr = i_instr;

  /* Set IMM */
  l_instr |= (unsigned long)(0xff & i_imm) << (32 + 31 - 24 - 7);
  /* Set T */
  l_instr |= (unsigned long)(0x1f & i_t) << (31 - 6 - 4);
  /* Set A */
  l_instr |= (unsigned long)(0x1f & i_a) << (31 - 11 - 4);
  /* Set B */
  l_instr |= (unsigned long)(0x1f & i_b) << (31 - 16 - 4);
  /* Set C */
  l_instr |= (unsigned long)(0x1f & i_c) << (31 - 21 - 4);
  /* Set CX */
  l_instr |= (unsigned long)(0x01 & i_cx) << (31 - 28 - 0);
  /* Set AX */
  l_instr |= (unsigned long)(0x01 & i_ax) << (31 - 29 - 0);
  /* Set BX */
  l_instr |= (unsigned long)(0x01 & i_bx) << (31 - 30 - 0);
  /* Set TX */
  l_instr |= (unsigned long)(0x01 & i_tx) << (31 - 31 - 0);

  return l_instr;
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_nop( libxsmm_generated_code *io_generated_code ) {
  libxsmm_ppc64le_instr_append( io_generated_code,
                                LIBXSMM_PPC64LE_INSTR_NOP );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_blr( libxsmm_generated_code *io_generated_code ) {
  libxsmm_ppc64le_instr_append( io_generated_code,
                                LIBXSMM_PPC64LE_INSTR_BLR );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_append( libxsmm_generated_code *io_generated_code,
                                   unsigned int            i_op ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;

    l_code[l_code_head] = i_op;
    io_generated_code->code_size += 4;
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_1( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* VX (vrb) form */
      case LIBXSMM_PPC64LE_FORM_VX_VRB: {
        l_op = libxsmm_ppc64le_instr_vx_form_vrb( l_instr, (unsigned char)i_0 );
      } break;
      /* VX (vrt) form */
      case LIBXSMM_PPC64LE_FORM_VX_VRT: {
        l_op = libxsmm_ppc64le_instr_vx_form_vrt( l_instr, (unsigned char)i_0 );
      } break;
      /* X (3) form */
      case LIBXSMM_PPC64LE_FORM_X_3: {
        l_op = libxsmm_ppc64le_instr_x_form_3( l_instr, (unsigned char)i_0 );
      } break;
      /* X (3) form */
      case LIBXSMM_PPC64LE_FORM_XFX_5: {
        l_op = libxsmm_ppc64le_instr_xfx_form_5( l_instr, (unsigned char)i_0 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_2( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* X (33) form */
      case LIBXSMM_PPC64LE_FORM_X_33: {
        l_op = libxsmm_ppc64le_instr_x_form_33( l_instr, (unsigned char)i_0, (unsigned char)i_1 );
      } break;
      /* X (55) form */
      case LIBXSMM_PPC64LE_FORM_X_55: {
        l_op = libxsmm_ppc64le_instr_x_form_55( l_instr, (unsigned char)i_0, (unsigned char)i_1 );
      } break;
      /* XL (2) form */
      case LIBXSMM_PPC64LE_FORM_XL_2: {
        l_op = libxsmm_ppc64le_instr_xl_form_2( l_instr, (unsigned char)i_0, (unsigned char)i_1 );
      } break;
      /* XFX (2) form */
      case LIBXSMM_PPC64LE_FORM_XFX_2: {
        l_op = libxsmm_ppc64le_instr_xfx_form_2( l_instr, (unsigned char)i_0, (unsigned char)i_1 );
      } break;
      /* XFX (4) form */
      case LIBXSMM_PPC64LE_FORM_XFX_4: {
        l_op = libxsmm_ppc64le_instr_xfx_form_4( l_instr, (unsigned char)i_0, (unsigned int)i_1 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_3( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* B form */
      case LIBXSMM_PPC64LE_FORM_B: {
        l_op = libxsmm_ppc64le_instr_b_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2 );
      } break;
      /* D form */
      case LIBXSMM_PPC64LE_FORM_D: {
        l_op = libxsmm_ppc64le_instr_d_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2 );
      } break;
      /* DS form */
      case LIBXSMM_PPC64LE_FORM_DS: {
        l_op = libxsmm_ppc64le_instr_ds_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2 );
      } break;
      /* X (355L) form */
      case LIBXSMM_PPC64LE_FORM_X_355L: {
        l_op = libxsmm_ppc64le_instr_x_form_355_l( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2 );
      } break;
      /* X (555) form */
      case LIBXSMM_PPC64LE_FORM_X_555: {
        l_op = libxsmm_ppc64le_instr_x_form_555( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2 );
      } break;
      /* XX2 (2) form */
      case LIBXSMM_PPC64LE_FORM_XX2_2: {
        l_op = libxsmm_ppc64le_instr_xx2_form_2( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_4( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* D (bf) form */
      case LIBXSMM_PPC64LE_FORM_D_BF: {
        l_op = libxsmm_ppc64le_instr_d_form_bf( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
      } break;
      /* DQ (p) form */
      case LIBXSMM_PPC64LE_FORM_DQ_P:{
        l_op = libxsmm_ppc64le_instr_dq_form_p( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
      } break;
      /* DQ (x) form */
      case LIBXSMM_PPC64LE_FORM_DQ_X:{
        l_op = libxsmm_ppc64le_instr_dq_form_x( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3 );
      } break;
      /* VA form */
      case LIBXSMM_PPC64LE_FORM_VA: {
        l_op = libxsmm_ppc64le_instr_va_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
      } break;
      /* X form */
      case LIBXSMM_PPC64LE_FORM_X: {
        l_op = libxsmm_ppc64le_instr_x_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
      } break;
      /* X (4155) form */
      case LIBXSMM_PPC64LE_FORM_X_4155: {
        l_op = libxsmm_ppc64le_instr_x_form_4155( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
      } break;
      /* XL (4) form */
      case LIBXSMM_PPC64LE_FORM_XL_4: {
        l_op = libxsmm_ppc64le_instr_xl_form_4( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
      } break;
      /* XX2 (3) form */
      case LIBXSMM_PPC64LE_FORM_XX2_3: {
        l_op = libxsmm_ppc64le_instr_xx2_form_3( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_5( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* XX2 (4) form */
      case LIBXSMM_PPC64LE_FORM_XX2_4: {
        l_op = libxsmm_ppc64le_instr_xx2_form_4( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
      } break;
      /* XX3 (0) form */
      case LIBXSMM_PPC64LE_FORM_XX3_0: {
        l_op = libxsmm_ppc64le_instr_xx3_form_0( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_6( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3,
                              unsigned int            i_4,
                              unsigned int            i_5 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* M form */
      case LIBXSMM_PPC64LE_FORM_M: {
        l_op = libxsmm_ppc64le_instr_m_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
      } break;
      /* MD form */
      case LIBXSMM_PPC64LE_FORM_MD: {
        l_op = libxsmm_ppc64le_instr_md_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
      } break;
      /* XX3 (6) form */
      case LIBXSMM_PPC64LE_FORM_XX3_6: {
        l_op = libxsmm_ppc64le_instr_xx3_form_6( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_7( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3,
                              unsigned int            i_4,
                              unsigned int            i_5,
                              unsigned int            i_6 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* XX3 (3) form */
      case LIBXSMM_PPC64LE_FORM_XX3_3: {
        l_op = libxsmm_ppc64le_instr_xx3_form_3( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_8( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3,
                              unsigned int            i_4,
                              unsigned int            i_5,
                              unsigned int            i_6,
                              unsigned int            i_7 ) {

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_op;

    unsigned int l_fid = i_instr & ~LIBXSMM_PPC64LE_32FMASK;
    unsigned int l_instr = i_instr & LIBXSMM_PPC64LE_32FMASK;

    switch( l_fid ) {
      /* XX4 form */
      case LIBXSMM_PPC64LE_FORM_XX4: {
        l_op = libxsmm_ppc64le_instr_xx4_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_append( libxsmm_generated_code *io_generated_code,
                                          unsigned long           i_op ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;

    unsigned int l_op_h = (unsigned int)( ( i_op >> 32 ) & 0x00000000ffffffff );
    unsigned int l_op_l = (unsigned int)( i_op & 0x00000000ffffffff );

    /* From ABI 8-byte 'prefix' ops cannot cross 64-byte boundaries */
    if ( ( l_code_head / 16 ) == ( ( l_code_head + 1 ) / 16 ) ) {
      l_code[l_code_head] = l_op_h;
      l_code[l_code_head + 1] = l_op_l;
      io_generated_code->code_size += 8;
    } else {
      l_code[l_code_head] = LIBXSMM_PPC64LE_INSTR_NOP;
      l_code[l_code_head + 1] = l_op_h;
      l_code[l_code_head + 2] = l_op_l;
      io_generated_code->code_size += 12;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}



LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_4( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned long l_op;

    unsigned long l_fid = i_instr & ~LIBXSMM_PPC64LE_64FMASK;
    unsigned long l_instr = i_instr & LIBXSMM_PPC64LE_64FMASK;

    switch( l_fid ) {
      /* D-8RR (1, 3) form */
      case LIBXSMM_PPC64LE_FORM_8RR_D_1_3: {
        l_op = libxsmm_ppc64le_instr_d_form_1_8rr3( l_instr, (unsigned int)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_prefix_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_5( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned long l_op;

    unsigned long l_fid = i_instr & ~LIBXSMM_PPC64LE_64FMASK;
    unsigned long l_instr = i_instr & LIBXSMM_PPC64LE_64FMASK;

    switch( l_fid ) {
      /* D-MLS form */
      case LIBXSMM_PPC64LE_FORM_MLS_D: {
        l_op = libxsmm_ppc64le_instr_d_form_mls( l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned int)i_4 );
      } break;
      /* D-8RR (0, 3) form */
      case LIBXSMM_PPC64LE_FORM_8RR_D_0_3: {
        l_op = libxsmm_ppc64le_instr_d_form_0_8rr3( l_instr, (unsigned int)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned int)i_4 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_prefix_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_6( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4,
                                     unsigned int            i_5 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned long l_op;

    unsigned long l_fid = i_instr & ~LIBXSMM_PPC64LE_64FMASK;
    unsigned long l_instr = i_instr & LIBXSMM_PPC64LE_64FMASK;

    switch( l_fid ) {
      /* D-8LS form */
      case LIBXSMM_PPC64LE_FORM_8LS_D: {
        l_op = libxsmm_ppc64le_instr_d_form_8ls( l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned int)i_5 );
      } break;
      /* D-8LS(P) form */
      case LIBXSMM_PPC64LE_FORM_8LS_D_P: {
        l_op = libxsmm_ppc64le_instr_d_form_8lsp( l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned int)i_5 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_prefix_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_7( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4,
                                     unsigned int            i_5,
                                     unsigned int            i_6 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned long l_op;

    unsigned long l_fid = i_instr & ~LIBXSMM_PPC64LE_64FMASK;
    unsigned long l_instr = i_instr & LIBXSMM_PPC64LE_64FMASK;

    switch( l_fid ) {
      /* XX3-MMIRR (0, 0) form */
      case LIBXSMM_PPC64LE_FORM_MMIRR_XX3_0_0: {
        l_op = libxsmm_ppc64le_instr_xx3_form_0_mmirr0( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
      } break;
      /* XX3-MMIRR (0, 1) form */
      case LIBXSMM_PPC64LE_FORM_MMIRR_XX3_0_1: {
        l_op = libxsmm_ppc64le_instr_xx3_form_0_mmirr1( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_prefix_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_8( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4,
                                     unsigned int            i_5,
                                     unsigned int            i_6,
                                     unsigned int            i_7 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned long l_op;

    unsigned long l_fid = i_instr & ~LIBXSMM_PPC64LE_64FMASK;
    unsigned long l_instr = i_instr & LIBXSMM_PPC64LE_64FMASK;

    switch( l_fid ) {
      /* XX3-MMIRR (0, 3) form */
      case LIBXSMM_PPC64LE_FORM_MMIRR_XX3_0_3: {
        l_op = libxsmm_ppc64le_instr_xx3_form_0_mmirr3( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
      } break;
      /* XX4-8RR (0) form */
      case LIBXSMM_PPC64LE_FORM_8RR_XX4_0: {
        l_op = libxsmm_ppc64le_instr_xx4_form_8rr0( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_prefix_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_9( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4,
                                     unsigned int            i_5,
                                     unsigned int            i_6,
                                     unsigned int            i_7,
                                     unsigned int            i_8 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned long l_op;

    unsigned long l_fid = i_instr & ~LIBXSMM_PPC64LE_64FMASK;
    unsigned long l_instr = i_instr & LIBXSMM_PPC64LE_64FMASK;

    switch( l_fid ) {
      /* XX4-8RR (2) form */
      case LIBXSMM_PPC64LE_FORM_8RR_XX4_2: {
        l_op =libxsmm_ppc64le_instr_xx4_form_8rr2( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7, (unsigned char)i_8 );
      } break;
      default: {
        l_op = 0;
      }
    }

    if ( l_op != 0 ) {
      libxsmm_ppc64le_instr_prefix_append( io_generated_code, l_op );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose( libxsmm_generated_code *io_generated_code,
                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                      libxsmm_datatype const  i_datatype,
                                      unsigned int           *i_v,
                                      unsigned int            i_n,
                                      unsigned int           *o_v,
                                      unsigned int            i_m ) {
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_ppc64le_instr_transpose_f32( io_generated_code, io_reg_tracker, i_v, i_n, o_v, i_m );
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_ppc64le_instr_transpose_f64( io_generated_code, io_reg_tracker, i_v, i_n, o_v, i_m );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker,
                                          unsigned int           *i_v,
                                          unsigned int            i_n,
                                          unsigned int           *o_v,
                                          unsigned int            i_m ) {
  unsigned char l_id = (0x0f & (unsigned char)i_m) | ((0x0f & (unsigned char)i_n) << 4);

  switch( l_id ) {
    case 0x44: {
      libxsmm_ppc64le_instr_transpose_f32_4x4( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x34: {
      libxsmm_ppc64le_instr_transpose_f32_3x4( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x24: {
      libxsmm_ppc64le_instr_transpose_f32_2x4( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x14: {
      libxsmm_ppc64le_instr_transpose_f32_1x4( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x43: {
      libxsmm_ppc64le_instr_transpose_f32_4x3( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x33: {
      libxsmm_ppc64le_instr_transpose_f32_3x3( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x23: {
      libxsmm_ppc64le_instr_transpose_f32_2x3( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x13: {
      libxsmm_ppc64le_instr_transpose_f32_1x3( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x42: {
      libxsmm_ppc64le_instr_transpose_f32_4x2( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x32: {
      libxsmm_ppc64le_instr_transpose_f32_3x2( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x22: {
      libxsmm_ppc64le_instr_transpose_f32_2x2( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x12: {
      libxsmm_ppc64le_instr_transpose_f32_1x2( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x41: {
      libxsmm_ppc64le_instr_transpose_f32_4x1( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x31: {
      libxsmm_ppc64le_instr_transpose_f32_3x1( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x21: {
      libxsmm_ppc64le_instr_transpose_f32_2x1( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x11: {
      libxsmm_ppc64le_instr_transpose_f32_1x1( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  }
}

/**
 * Transpose a 4x4 fp32 block in big-endian word ordering
 * ie.
 * VI0 = [ a3,  a2,  a1,  a0]
 * VI1 = [ a7,  a6,  a5,  a4]
 * VI2 = [a11, a10,  a9,  a8]
 * VI3 = [a15, a14, a13, a12]
 * goes to
 * VO0 = [a12,  a8,  a4,  a0]
 * VO1 = [a13,  a9,  a5,  a1]
 * VO2 = [a14, a10,  a6,  a2]
 * VO3 = [a15, a11,  a7,  a3]
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param io_reg_tracker pointer to register tracking structure.
 * @param i_v pointer to input register indices.
 * @param o_v pointer to output register indices.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  static unsigned int i, l_scratch[4];
  for ( i = 0; i < 4 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[3]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[2]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[0], o_v[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[3], l_scratch[1], o_v[3]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[3], l_scratch[1], o_v[2]);

  for ( i = 0 ; i < 4 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  unsigned int i, l_scratch[4];
  for ( i = 0; i < 4 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[3]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[2]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[0], o_v[1]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[3], l_scratch[1], o_v[2]);

  for ( i = 0 ; i < 4 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  unsigned int i, l_scratch[2];
  for ( i = 0; i < 2 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[1], l_scratch[0], o_v[1]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[1], l_scratch[0], o_v[0]);

  for ( i = 0 ; i < 2 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  unsigned int i, l_scratch[2];
  for ( i = 0; i < 2 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[1]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[1], l_scratch[0], o_v[0]);

  for ( i = 0 ; i < 2 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  unsigned int i, l_scratch[4];
  for ( i = 0; i < 4 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[3]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[3], i_v[1], l_scratch[2]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[0], o_v[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[3], l_scratch[1], o_v[3]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[3], l_scratch[1], o_v[2]);

  for ( i = 0 ; i < 4 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  unsigned int i, l_scratch[4];
  for ( i = 0; i < 4 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[1]);
  libxsmm_ppc64le_instr_vec_splat( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], 2, l_scratch[2]);
  libxsmm_ppc64le_instr_vec_splat( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], 1, l_scratch[3]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], l_scratch[1], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[2], l_scratch[1], o_v[1]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[3], l_scratch[0], o_v[2]);

  for ( i = 0 ; i < 4 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  unsigned int i, l_scratch[2];
  for ( i = 0; i < 2 ; ++i ) {
    l_scratch[i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], l_scratch[0]);
  libxsmm_ppc64le_instr_vec_splat( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], 2, l_scratch[1]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, l_scratch[1], l_scratch[0], o_v[1]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], l_scratch[0], o_v[0]);

  for ( i = 0 ; i < 2 ; ++i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[2], i_v[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], o_v[0], o_v[0]);
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], i_v[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], i_v[0], o_v[2]);
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, o_v[0], o_v[1], 2 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, o_v[2], o_v[3], 2 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], i_v[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], i_v[0], o_v[2]);
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, o_v[0], o_v[1], 2 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], i_v[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, o_v[0], o_v[1], 2 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[1], i_v[0], o_v[0]);
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[0], 0 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[1], 3 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[2], 2 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[3], 1 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[0], 0 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[1], 3 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[2], 2 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[0], 0 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[1], 3 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F32, i_v[0], o_v[0], 0 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker,
                                          unsigned int           *i_v,
                                          unsigned int            i_n,
                                          unsigned int           *o_v,
                                          unsigned int            i_m ) {
  unsigned char l_id = (0x0f & (unsigned char)i_m) | ((0x0f & (unsigned char)i_n) << 4);

  switch( l_id ) {
    case 0x22: {
      libxsmm_ppc64le_instr_transpose_f64_2x2( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x12: {
      libxsmm_ppc64le_instr_transpose_f64_1x2( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x21: {
      libxsmm_ppc64le_instr_transpose_f64_2x1( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    case 0x11: {
      libxsmm_ppc64le_instr_transpose_f64_1x1( io_generated_code, io_reg_tracker, i_v, o_v );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F64, i_v[1], i_v[0], o_v[0]);
  libxsmm_ppc64le_instr_vec_merge_high( io_generated_code, LIBXSMM_DATATYPE_F64, i_v[1], i_v[0], o_v[1]);
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_1x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_vec_merge_low( io_generated_code, LIBXSMM_DATATYPE_F64, i_v[1], i_v[0], o_v[0]);
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F64, i_v[0], o_v[0], 0 );
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F64, i_v[0], o_v[1], 1 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_1x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_vec_shift_left( io_generated_code, LIBXSMM_DATATYPE_F64, i_v[0], o_v[0], 0 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_copy_reg( libxsmm_generated_code *io_generated_code,
                                     unsigned int            i_src,
                                     unsigned int            i_dst ) {
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_OR, i_src, i_dst, i_src );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_add_value( libxsmm_generated_code *io_generated_code,
                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                      unsigned int            i_src,
                                      unsigned int            i_dst,
                                      long                    i_val ) {
  if ( 0 == i_src ) {
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, i_dst, i_val );
  } else if ( 0 == i_val ) {
    libxsmm_ppc64le_instr_copy_reg( io_generated_code, i_src, i_dst );
  } else if ( i_val <= 0x7fff && i_val >= -0x7fff ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, i_dst, i_src, i_val );
  } else {
    if ( io_generated_code->arch == LIBXSMM_PPC64LE_VSX ) {
      unsigned int l_low = (unsigned int)( 0xffff & i_val );
      unsigned int l_high = (unsigned int)( (0xffff & ( i_val >> 16 )) + ( 0x01 & ( l_low >> 15 ) ) );
      unsigned int l_reg;

      if ( l_low != 0 ) {
        l_reg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
        libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_reg, i_src, l_low );
        libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_reg );
      } else {
        l_reg = i_dst;
      }
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, i_dst, l_reg, l_high );
    } else if ( io_generated_code->arch == LIBXSMM_PPC64LE_MMA ) {
      unsigned int l_low = (unsigned int)( 0xffff & i_val );
      unsigned int l_high = (unsigned int)( 0x03ffff & ( i_val >> 16 ) );
      libxsmm_ppc64le_instr_prefix_5( io_generated_code, LIBXSMM_PPC64LE_INSTR_PADDI, 0, l_high, i_dst, i_src, l_low );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_set_imm64( libxsmm_generated_code *io_generated_code,
                                      unsigned int            i_dst,
                                      long                    i_val ) {
  if ( 0 == i_val ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, i_dst, 0, 0 );
  } else if ( 0 != i_val ) {
    unsigned int l_dst = 0;
    unsigned int l_h3 = (unsigned int)( 0xffff & i_val );
    unsigned int l_h2 = (unsigned int)( 0xffff & ( i_val >> 16 ) );
    unsigned int l_h1 = (unsigned int)( 0xffff & ( i_val >> 32 ) );
    unsigned int l_h0 = (unsigned int)( 0xffff & ( i_val >> 48 ) );

    if ( 0 != l_h0 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, i_dst, l_dst, l_h0 );
      l_dst = i_dst;
    }

    if ( 0 != l_h1 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, i_dst, l_dst, l_h1 );
      l_dst = i_dst;
    }

    if ( 0 != l_h1 || 0 != l_h0 ) {
      libxsmm_ppc64le_instr_set_shift_left( io_generated_code, i_dst, i_dst, 32 );
    }

    if ( 0 != l_h2 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, i_dst, l_dst, l_h2 );
      l_dst = i_dst;
    }

    if ( 0 != l_h3 ) {
      libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, i_dst, l_dst, l_h3 );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_set_shift_left( libxsmm_generated_code *io_generated_code,
                                           unsigned int            i_src,
                                           unsigned int            i_dst,
                                           unsigned char           i_n ) {
  unsigned char l_mask = 63 - i_n;
  unsigned char l_mask_in = ( ( 0x1f & l_mask ) << 1 ) + ( ( 0x20 & l_mask ) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_RLDICR, i_src, i_dst, i_n, l_mask_in, (0x20 & i_n) >> 5, 0 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_vec_merge_high( libxsmm_generated_code *io_generated_code,
                                           libxsmm_datatype const  i_datatype,
                                           unsigned int            i_src_0,
                                           unsigned int            i_src_1,
                                           unsigned int            i_dst ) {
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_ppc64le_instr_6( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                               i_dst,
                               i_src_0,
                               i_src_1,
                               (0x20 & i_src_0) >> 5,
                               (0x20 & i_src_1) >> 5,
                               (0x20 & i_dst) >> 5 );
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_ppc64le_instr_7( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                               i_dst,
                               i_src_0,
                               i_src_1,
                               0x00,
                               (0x20 & i_src_0) >> 5,
                               (0x20 & i_src_1) >> 5,
                               (0x20 & i_dst) >> 5 );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_vec_merge_low( libxsmm_generated_code *io_generated_code,
                                          libxsmm_datatype const  i_datatype,
                                          unsigned int            i_src_0,
                                          unsigned int            i_src_1,
                                          unsigned int            i_dst ) {
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_ppc64le_instr_6( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                               i_dst,
                               i_src_0,
                               i_src_1,
                               (0x20 & i_src_0) >> 5,
                               (0x20 & i_src_1) >> 5,
                               (0x20 & i_dst) >> 5 );
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_ppc64le_instr_7( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                               i_dst,
                               i_src_0,
                               i_src_1,
                               0x03,
                               (0x20 & i_src_0) >> 5,
                               (0x20 & i_src_1) >> 5,
                               (0x20 & i_dst) >> 5 );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_vec_splat( libxsmm_generated_code *io_generated_code,
                                      libxsmm_datatype const  i_datatype,
                                      unsigned int            i_src,
                                      unsigned int            i_pos,
                                      unsigned int            i_dst ) {
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      libxsmm_ppc64le_instr_5( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXSPLTW,
                               i_dst,
                               i_pos,
                               i_src,
                               ( 0x20 & i_src ) >> 5,
                               ( 0x20 & i_dst ) >> 5 );
    } break;
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_ppc64le_instr_7( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                               i_dst,
                               i_src,
                               i_src,
                               i_pos*3,
                               ( 0x20 & i_src ) >> 5,
                               ( 0x20 & i_src ) >> 5,
                               ( 0x20 & i_dst ) >> 5 );
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_vec_shift_left( libxsmm_generated_code *io_generated_code,
                                           libxsmm_datatype const  i_datatype,
                                           unsigned int            i_src,
                                           unsigned int            i_dst,
                                           unsigned char           i_n ) {
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      if ( i_n % 4 == 0 ){
        libxsmm_ppc64le_instr_6( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_XXLOR,
                                 i_dst,
                                 i_src,
                                 i_src,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_dst) >> 5 );
      } else {
        libxsmm_ppc64le_instr_7( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                                 i_dst,
                                 i_src,
                                 i_src,
                                 0x03 & i_n,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_dst) >> 5 );
      }
    } break;
    case LIBXSMM_DATATYPE_F64: {
      if ( i_n % 2 == 0 ) {
        libxsmm_ppc64le_instr_6( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_XXLOR,
                                 i_dst,
                                 i_src,
                                 i_src,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_dst) >> 5 );
      } else {
        libxsmm_ppc64le_instr_7( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                                 i_dst,
                                 i_src,
                                 i_src,
                                 0x02,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_src) >> 5,
                                 (0x20 & i_dst) >> 5 );
      }
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_load( libxsmm_generated_code *io_generated_code,
                                 unsigned int            i_a,
                                 long                    i_offset,
                                 unsigned int            i_t ) {
  if ( i_offset <= 0x7fff && i_offset >= -0x7fff && ( i_offset % 16 ) == 0 ) {
    unsigned int l_offset = (unsigned int)( 0xffff & i_offset ) >> 4;
    libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_LXV, i_t, i_a, l_offset, (0x20 & i_t) >> 5 );
  } else if ( i_offset <= 0x01ffffffff && i_offset >= -0x01ffffffff ) {
    if ( io_generated_code->arch >= LIBXSMM_PPC64LE_MMA ) {
      unsigned int l_offl = (unsigned int)( 0xffff & i_offset );
      unsigned int l_offh = (unsigned int)( 0x03ffff & ( i_offset >> 16 ) );
      libxsmm_ppc64le_instr_prefix_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_PLXV, 0, l_offh, (0x20 & i_t) >> 5, i_t, i_a, l_offl );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_load_part( libxsmm_generated_code *io_generated_code,
                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                      unsigned int            i_a,
                                      long                    i_offset,
                                      unsigned int            i_mask,
                                      unsigned int            i_t ) {
  unsigned int l_a;
  if ( i_offset != 0 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR);
    libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a, i_offset);
  } else {
    l_a = i_a;
  }
  libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_LXVL, i_t, l_a, i_mask, (0x20 & i_t) >> 5 );

  if ( i_offset != 0 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a);
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_load_pair( libxsmm_generated_code *io_generated_code,
                                      unsigned int            i_a,
                                      long                    i_offset,
                                      unsigned int            i_t ) {
  if ( i_t % 2 == 1) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_REGNUM );
    return;
  }

  if ( io_generated_code->arch >= LIBXSMM_PPC64LE_MMA ) {
    unsigned int l_tp = (0x1f & i_t) >> 1;
    unsigned int l_tx = (0x20 & i_t) >> 5 ;

    if ( i_offset <= 0x7fff && i_offset >= -0x7fff && ( i_offset % 16 ) == 0 ) {
      unsigned int l_offset = (unsigned int)( 0xffff & i_offset ) >> 4;

      libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_LXVP, l_tp, l_tx, i_a, l_offset );
    } else if ( i_offset <= 0x01ffffffff && i_offset >= -0x01ffffffff ) {
      unsigned int l_offl = (unsigned int)( 0xffff & i_offset );
      unsigned int l_offh = (unsigned int)( 0x03ffff & ( i_offset >> 16 ) );

      libxsmm_ppc64le_instr_prefix_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_PLXVP, 0, l_offh, l_tp, l_tx, i_a, l_offl );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  } else {
     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_store( libxsmm_generated_code *io_generated_code,
                                  unsigned int            i_a,
                                  long                    i_offset,
                                  unsigned int            i_t ) {
  if ( i_offset <= 0x7fff && i_offset >= -0x7fff && ( i_offset % 16 ) == 0 ) {
    unsigned int l_offset = (unsigned int)( 0xffff & i_offset ) >> 4;
    libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_STXV, i_t, i_a, l_offset, (0x20 & i_t) >> 5 );
  } else if ( i_offset <= 0x01ffffffff && i_offset >= -0x01ffffffff ) {
    if ( io_generated_code->arch == LIBXSMM_PPC64LE_MMA ) {
      unsigned int l_offl = (unsigned int)( 0xffff & i_offset );
      unsigned int l_offh = (unsigned int)( 0x03ffff & ( i_offset >> 16 ) );
      libxsmm_ppc64le_instr_prefix_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_PSTXV, 0, l_offh, (0x20 & i_t) >> 5, i_t, i_a, l_offl );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_store_part( libxsmm_generated_code *io_generated_code,
                                       libxsmm_ppc64le_reg    *io_reg_tracker,
                                       unsigned int            i_a,
                                       long                    i_offset,
                                       unsigned int            i_mask,
                                       unsigned int            i_t ) {
  unsigned int l_a;
  if ( i_offset != 0 ) {
    l_a = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR);
    libxsmm_ppc64le_instr_add_value( io_generated_code, io_reg_tracker, i_a, l_a, i_offset);
  } else {
    l_a = i_a;
  }
  libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_STXVL, i_t, l_a, i_mask, (0x20 & i_t) >> 5 );

  if ( i_offset != 0 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_a);
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_load_splat( libxsmm_generated_code *io_generated_code,
                                       libxsmm_ppc64le_reg    *io_reg_tracker,
                                       libxsmm_datatype const  i_datatype,
                                       unsigned int            i_a,
                                       long                    i_offset,
                                       unsigned int            i_t ) {
  unsigned int l_instr, l_offset;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F32: {
      l_instr = LIBXSMM_PPC64LE_INSTR_LXVWSX;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      l_instr = LIBXSMM_PPC64LE_INSTR_LXVDSX;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  if ( i_offset != 0 ) {
    l_offset = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_offset, i_offset );
  } else {
    l_offset = 0;
  }

  libxsmm_ppc64le_instr_4( io_generated_code, l_instr, i_t, l_offset, i_a, (0x20 & i_t) >> 5 );

  if ( i_offset != 0 ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_offset );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_open_stream( libxsmm_generated_code *io_generated_code,
                                        libxsmm_ppc64le_reg    *io_reg_tracker ) {
  /* From "64-Bit ELF V2 ABI Specification: Power Architecture" */
  unsigned int i_reg, i;
  unsigned int gpr_offset = LIBXSMM_PPC64LE_STACK_SIZE - 16 ;
  unsigned int fpr_offset = gpr_offset - (LIBXSMM_PPC64LE_GPR_NMAX - LIBXSMM_PPC64LE_GPR_IVOL)*8;
  unsigned int vsr_offset = fpr_offset - (LIBXSMM_PPC64LE_FPR_NMAX - LIBXSMM_PPC64LE_FPR_IVOL)*8;

  /* Save R31 for immediate use */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_GPR_SP,
                           ( -8 >> 2 ) );

  /* Decrease stack pointer */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STDU,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_GPR_SP,
                           (-LIBXSMM_PPC64LE_STACK_SIZE >> 2 ) );

  /* Get the LR and store it in the stackframe */
  libxsmm_ppc64le_instr_2( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MFSPR,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_SPR_LR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_GPR_SP,
                           ( 16 >> 2 ) );

  /* Get CR and store it in the stackframe */
  libxsmm_ppc64le_instr_1( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MFCR,
                           LIBXSMM_PPC64LE_GPR_R31 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_GPR_SP,
                           ( 24 >> 2 ) );

  /* Save non-volatile general purpose registers */

  for( i_reg = LIBXSMM_PPC64LE_GPR_IVOL; i_reg < LIBXSMM_PPC64LE_GPR_NMAX; ++i_reg ) {
    unsigned int l_offset = gpr_offset - (i_reg - LIBXSMM_PPC64LE_GPR_IVOL)*8;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STD,
                             i_reg,
                             LIBXSMM_PPC64LE_GPR_SP,
                             l_offset >> 2 );
  }

  /* Save non-volatile floating point registers */
  for( i_reg = LIBXSMM_PPC64LE_FPR_IVOL; i_reg < LIBXSMM_PPC64LE_FPR_NMAX; ++i_reg ) {
    unsigned int l_offset = fpr_offset -  (i_reg - LIBXSMM_PPC64LE_FPR_IVOL)*8;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STFD,
                             i_reg,
                             LIBXSMM_PPC64LE_GPR_SP,
                             l_offset );
  }

  /* Save non-volatile vector registers */
  for( i_reg = LIBXSMM_PPC64LE_VR_IVOL, i = 0; i_reg < LIBXSMM_PPC64LE_VR_NMAX; ++i_reg, ++i ) {
    unsigned int l_offset = vsr_offset - i*16;
    libxsmm_ppc64le_instr_4( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STXV,
                             i_reg,
                             LIBXSMM_PPC64LE_GPR_SP,
                             l_offset >> 4,
                             1 );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_unpack_args( libxsmm_generated_code *io_generated_code,
                                        libxsmm_ppc64le_reg    *io_reg_tracker ) {
  /* Set up input args */
  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));

  libxsmm_ppc64le_instr_copy_reg( io_generated_code, LIBXSMM_PPC64LE_GPR_ARG0, LIBXSMM_PPC64LE_GPR_R31 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG0,
                           LIBXSMM_PPC64LE_GPR_R31,
                           l_offset_ptr_a >> 2 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG1,
                           LIBXSMM_PPC64LE_GPR_R31,
                           l_offset_ptr_b >> 2 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG2,
                           LIBXSMM_PPC64LE_GPR_R31,
                           l_offset_ptr_c >> 2 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_unpack_brargs( libxsmm_generated_code        *io_generated_code,
                                          libxsmm_gemm_descriptor const *io_xgemm_desc,
                                          libxsmm_ppc64le_reg           *io_reg_tracker ) {
  /* Set up input args */
  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));

  libxsmm_ppc64le_instr_copy_reg( io_generated_code, LIBXSMM_PPC64LE_GPR_ARG0, LIBXSMM_PPC64LE_GPR_R31 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG0,
                           LIBXSMM_PPC64LE_GPR_R31,
                           l_offset_ptr_a >> 2 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG1,
                           LIBXSMM_PPC64LE_GPR_R31,
                           l_offset_ptr_b >> 2 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG2,
                           LIBXSMM_PPC64LE_GPR_R31,
                           l_offset_ptr_c >> 2 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_ARG3,
                           LIBXSMM_PPC64LE_GPR_R31,
                           16 >> 2 );

  if ( 0 < ( io_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET ) ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LD,
                             LIBXSMM_PPC64LE_GPR_ARG4,
                             LIBXSMM_PPC64LE_GPR_R31,
                             ( l_offset_ptr_a + 8 ) >> 2 );
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LD,
                             LIBXSMM_PPC64LE_GPR_ARG5,
                             LIBXSMM_PPC64LE_GPR_R31,
                             ( l_offset_ptr_b + 8 ) >> 2 );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_colapse_stack( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker ) {
  /* From "64-Bit ELF V2 ABI Specification: Power Architecture" */
  unsigned int i_reg, i;
  unsigned int gpr_offset = LIBXSMM_PPC64LE_STACK_SIZE - 16 ;
  unsigned int fpr_offset = gpr_offset - (LIBXSMM_PPC64LE_GPR_NMAX - LIBXSMM_PPC64LE_GPR_IVOL)*8;
  unsigned int vsr_offset = fpr_offset - (LIBXSMM_PPC64LE_FPR_NMAX - LIBXSMM_PPC64LE_FPR_IVOL)*8;

  /* Restore non-volatile general purpose registers */
  for( i_reg = LIBXSMM_PPC64LE_GPR_IVOL; i_reg < LIBXSMM_PPC64LE_GPR_NMAX; ++i_reg ) {
    unsigned int l_offset = gpr_offset - (i_reg - LIBXSMM_PPC64LE_GPR_IVOL)*8;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LD,
                             i_reg,
                             LIBXSMM_PPC64LE_GPR_SP,
                             l_offset >> 2 );
  }

  /* Restore non-volatile floating point registers */
  for( i_reg = LIBXSMM_PPC64LE_FPR_IVOL; i_reg < LIBXSMM_PPC64LE_FPR_NMAX; ++i_reg ) {
    unsigned int l_offset = fpr_offset -  (i_reg - LIBXSMM_PPC64LE_FPR_IVOL)*8;
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LFD,
                             i_reg,
                             LIBXSMM_PPC64LE_GPR_SP,
                             l_offset );
  }

  /* Restore non-volatile vector registers */
  for( i_reg = LIBXSMM_PPC64LE_VR_IVOL, i = 0; i_reg < LIBXSMM_PPC64LE_VR_NMAX; ++i_reg, ++i ) {
    unsigned int l_offset = vsr_offset - i*16;
    libxsmm_ppc64le_instr_4( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LXV,
                             i_reg,
                             LIBXSMM_PPC64LE_GPR_SP,
                             l_offset >> 4,
                             1 );
  }

  /* Get the LR and restore */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_GPR_SP,
                           ( 16 >> 2 ) );
  libxsmm_ppc64le_instr_2( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MTSPR,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_SPR_LR );

  /* Get CR and restore */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_GPR_SP,
                           ( 24 >> 2 ) );
  libxsmm_ppc64le_instr_2( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MTCRF,
                           0xff,
                           LIBXSMM_PPC64LE_GPR_R31 );

  /* Increase stack pointer */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_STACK_SIZE );

  /* Finally restore R31 */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R31,
                           LIBXSMM_PPC64LE_GPR_SP,
                           ( -8 >> 2 ) );

  /* Return statement */
  libxsmm_ppc64le_instr_blr( io_generated_code );
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_bytes( libxsmm_generated_code *io_generated_code,
                                          libxsmm_datatype const  i_datatype ) {
  unsigned int bytes = 0;

  switch ( i_datatype ) {
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
void libxsmm_ppc64le_instr_register_jump_back_label( libxsmm_generated_code     *io_generated_code,
                                                     libxsmm_loop_label_tracker *io_loop_label_tracker ) {
  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 512 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    int l_lab = io_loop_label_tracker->label_count;
    io_loop_label_tracker->label_count++;
    io_loop_label_tracker->label_address[l_lab] = io_generated_code->code_size;
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_cond_jump_back_to_label( libxsmm_generated_code     *io_generated_code,
                                                    unsigned int                i_gpr,
                                                    libxsmm_loop_label_tracker *io_loop_label_tracker ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_lab = --io_loop_label_tracker->label_count;
    unsigned int l_b_dst = (io_loop_label_tracker->label_address[l_lab]) / 4;
    unsigned int l_code_head = io_generated_code->code_size / 4;

    /* branch immediate */
    int l_b_imm = (int)l_b_dst - (int)l_code_head;

    /* compare GPR to 0 */
    libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_CMPI, 0, 0, i_gpr, 0 );

    /* branch if equal */
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_BC, 4, 2, l_b_imm - 1 );

  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_cond_jump_back_to_label_ctr( libxsmm_generated_code     *io_generated_code,
                                                        libxsmm_loop_label_tracker *io_loop_label_tracker ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_lab = --io_loop_label_tracker->label_count;
    unsigned int l_b_dst = (io_loop_label_tracker->label_address[l_lab]) / 4;
    unsigned int l_code_head = io_generated_code->code_size/4;

    /* branch immediate */
    int l_b_imm = (int)l_b_dst - (int)l_code_head;

    /* bdnz */
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_BC, 16, 0, l_b_imm );
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_jump_ctr_imm( libxsmm_generated_code *io_generated_code,
                                         libxsmm_ppc64le_reg    *io_reg_tracker,
                                         unsigned long           i_ptr ) {
  unsigned int l_reg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_reg, i_ptr );
  libxsmm_ppc64le_instr_jump_ctr( io_generated_code, l_reg );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_jump_ctr( libxsmm_generated_code *io_generated_code,
                                     unsigned int            i_reg ) {
  /* Load the address into the count register */
  libxsmm_ppc64le_instr_2( io_generated_code, LIBXSMM_PPC64LE_INSTR_MTSPR, i_reg, LIBXSMM_PPC64LE_SPR_CTR );

  /* Unconditional count register jump with return */
  libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_BCCTR, 0x14, 0, 0x00, 0x01 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_jump_lr_imm( libxsmm_generated_code *io_generated_code,
                                        libxsmm_ppc64le_reg    *io_reg_tracker,
                                        unsigned long           i_ptr ) {
  unsigned int l_reg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_set_imm64( io_generated_code, l_reg, i_ptr );
  libxsmm_ppc64le_instr_jump_lr( io_generated_code, l_reg );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_jump_lr( libxsmm_generated_code *io_generated_code,
                                    unsigned int            i_reg ) {
  /* Load the address into the count register */
  libxsmm_ppc64le_instr_2( io_generated_code, LIBXSMM_PPC64LE_INSTR_MTSPR, i_reg, LIBXSMM_PPC64LE_SPR_LR );

  /* Unconditional count register jump with return */
  libxsmm_ppc64le_instr_4( io_generated_code, LIBXSMM_PPC64LE_INSTR_BCLR, 0x14, 0, 0x00, 0x01 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefetch_stream_open( libxsmm_generated_code *io_generated_code,
                                                 libxsmm_ppc64le_reg    *io_reg_tracker,
                                                 char const              i_stream,
                                                 unsigned int const      i_a,
                                                 unsigned int const      i_lda,
                                                 unsigned int const      i_len ) {
  unsigned int l_stream, l_cfg, l_lda;

  l_stream = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  l_cfg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_stream, 0, (0x0f & i_stream) );

  /* TH=0x08
     0:56  EATRUNC
     57    Direction
     58    Unlimted
     59    RESERVED
     60:63 Stream
  */
  libxsmm_ppc64le_instr_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_RLDICR, l_cfg, l_cfg, 0, 56, 0, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0x08, l_cfg, l_stream );

  /* TH=0x0b
     0:31  RESERVED
     32:49 Stride
     50    RESERVED
     51:55 Offset
     56:59 RESERVED
     60:63 Stream
  */
  l_lda = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_lda, 0, i_lda / 2 );
  if ( i_lda % 2 == 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_lda, l_lda, 0x8000 );
  }
  libxsmm_ppc64le_instr_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_RLWINM, i_a, l_cfg, 6, 19, 23, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_cfg, l_cfg, l_lda );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0x0b, l_cfg, l_stream );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_lda );

  /* TH=0x0a
     0:31  RESERVED
     32    Go
     33:34 Stop
     35    RESERVED
     36:38 Depth
     39:46 RESERVED
     47:56 Count
     57    Transient
     58    Unlimited
     59    RESERVED
     60:63 Stream
  */
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_cfg, l_cfg, 0x8000 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_cfg, l_cfg, LIBXSMM_PPC64LE_TOUCH_DEPTH << 9 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_cfg, 0, ( 0x01ff & i_len ) << 7 );
  if ( i_len > 0x01ff ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_cfg, l_cfg, 1 );
  }
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_cfg, l_cfg, 0x40 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0x0a, l_cfg, l_stream );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_cfg );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_stream );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefetch_stream_close( libxsmm_generated_code *io_generated_code,
                                                  libxsmm_ppc64le_reg    *io_reg_tracker,
                                                  char const              i_stream ) {
  unsigned int l_cfg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* TH=0x0a
     0:31  RESERVED
     32    Go
     33:34 Stop
     35    RESERVED
     36:38 Depth
     39:46 RESERVED
     47:56 Count
     57    Transient
     58    Unlimited
     59    RESERVED
     60:63 Stream
  */
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_cfg, 0, 0x4000 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_cfg, l_cfg, (0x0f & i_stream) );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0x0a, 0, l_cfg );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_cfg );
}
