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
unsigned int libxsmm_ppc64le_get_reg( libxsmm_generated_code *io_generated_code,
                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                      unsigned int const      i_reg_type ) {
  if ( i_reg_type == LIBXSMM_PPC64LE_GPR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_GPR_NMAX - 1; i >= 0; --i ) {
      if ( io_reg_tracker->gpr[i] == LIBXSMM_PPC64LE_REG_FREE ) {
        io_reg_tracker->gpr[i] = LIBXSMM_PPC64LE_REG_USED;
        return i;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_FPR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_FPR_NMAX - 1; i >= 0; --i ) {
      if ( ( io_reg_tracker->fpr[i] == LIBXSMM_PPC64LE_REG_FREE ) &&
           ( io_reg_tracker->vsr[i] == LIBXSMM_PPC64LE_REG_FREE ) ) {
        io_reg_tracker->fpr[i] = LIBXSMM_PPC64LE_REG_USED;
        io_reg_tracker->vsr[i] = LIBXSMM_PPC64LE_REG_USED;
        return i;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_VR_NMAX - 1; i >= 0; --i ) {
      if ( ( io_reg_tracker->vr[i] == LIBXSMM_PPC64LE_REG_FREE ) &&
           ( io_reg_tracker->vsr[i + LIBXSMM_PPC64LE_FPR_NMAX] == LIBXSMM_PPC64LE_REG_FREE ) ) {
        io_reg_tracker->vr[i] = LIBXSMM_PPC64LE_REG_USED;
        io_reg_tracker->vsr[i + LIBXSMM_PPC64LE_FPR_NMAX] = LIBXSMM_PPC64LE_REG_USED;
        return i;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VSR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_VSR_NMAX - 1; i >= 0; --i ) {
      if ( i < LIBXSMM_PPC64LE_FPR_NMAX ) {
        if ( ( io_reg_tracker->fpr[i] == LIBXSMM_PPC64LE_REG_FREE ) &&
             ( io_reg_tracker->vsr[i] == LIBXSMM_PPC64LE_REG_FREE ) ) {
          io_reg_tracker->fpr[i] = LIBXSMM_PPC64LE_REG_USED;
          io_reg_tracker->vsr[i] = LIBXSMM_PPC64LE_REG_USED;
          return i;
        }
      } else if ( i < LIBXSMM_PPC64LE_FPR_NMAX + LIBXSMM_PPC64LE_VR_NMAX ) {
        if ( ( io_reg_tracker->vr[i - LIBXSMM_PPC64LE_FPR_NMAX] == LIBXSMM_PPC64LE_REG_FREE ) &&
             ( io_reg_tracker->vsr[i] == LIBXSMM_PPC64LE_REG_FREE ) ) {
          io_reg_tracker->vr[i - LIBXSMM_PPC64LE_FPR_NMAX] = LIBXSMM_PPC64LE_REG_USED;
          io_reg_tracker->vsr[i] = LIBXSMM_PPC64LE_REG_USED;
          return i;
        }
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_ACC ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_ACC_NMAX - 1; i >= 0; --i ) {
      if ( io_reg_tracker->acc[i] == LIBXSMM_PPC64LE_REG_FREE ) {
        unsigned char is_free = 1;
        for ( unsigned int j = i*4; j > (i + 1)*4; ++j ) {
          is_free &= ( io_reg_tracker->fpr[j] == LIBXSMM_PPC64LE_REG_FREE );
          is_free &= ( io_reg_tracker->vsr[j] == LIBXSMM_PPC64LE_REG_FREE );
        }

        if ( is_free ) {
          io_reg_tracker->acc[i] = LIBXSMM_PPC64LE_REG_USED;
          for ( unsigned int j = i*4; j > (i + 1)*4; ++j ) {
            io_reg_tracker->fpr[j] = LIBXSMM_PPC64LE_REG_USED;
            io_reg_tracker->vsr[j] = LIBXSMM_PPC64LE_REG_USED;
          }
          return i;
        }
      }
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return -1;
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_set_reg( libxsmm_generated_code *io_generated_code,
                              libxsmm_ppc64le_reg    *io_reg_tracker,
                              unsigned int const      i_reg_type,
                              unsigned int const      i_reg,
                              unsigned int const      i_value ) {
  if ( !(( i_value == LIBXSMM_PPC64LE_REG_RESV ) ||
         ( i_value == LIBXSMM_PPC64LE_REG_USED ) ||
         ( i_value == LIBXSMM_PPC64LE_REG_FREE )) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_reg_type == LIBXSMM_PPC64LE_GPR ) {
    io_reg_tracker->gpr[i_reg] = i_value;
  } else if ( i_reg_type == LIBXSMM_PPC64LE_FPR ) {
    io_reg_tracker->fpr[i_reg] = i_value;
    io_reg_tracker->vsr[i_reg] = i_value;
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VR) {
    io_reg_tracker->vr[i_reg] =  i_value;
    io_reg_tracker->vsr[i_reg + LIBXSMM_PPC64LE_FPR_NMAX] = i_value;
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VSR) {
    io_reg_tracker->vsr[i_reg] =  i_value;
    if ( i_reg < LIBXSMM_PPC64LE_FPR_NMAX ) {
      io_reg_tracker->fpr[i_reg] = i_value;
    } else if ( i_reg < LIBXSMM_PPC64LE_FPR_NMAX + LIBXSMM_PPC64LE_VR_NMAX ) {
      io_reg_tracker->vr[i_reg - LIBXSMM_PPC64LE_FPR_NMAX] = i_value;
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_ACC ) {
    io_reg_tracker->acc[i_reg] = i_value;
    for ( unsigned int i = i_reg*4; i > (i_reg + 1)*4; ++i ) {
      io_reg_tracker->fpr[i] = i_value;
      io_reg_tracker->vsr[i] = i_value;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_free_reg( libxsmm_generated_code *io_generated_code,
                               libxsmm_ppc64le_reg    *io_reg_tracker,
                               unsigned int const      i_reg_type,
                               unsigned int const      i_reg ) {
  libxsmm_ppc64le_set_reg( io_generated_code, io_reg_tracker, i_reg_type, i_reg, LIBXSMM_PPC64LE_REG_FREE );
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
  l_instr |= (unsigned int)( (0x00000fff & i_dq) << (31 - 16 - 11) );
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
  l_instr |= (unsigned int)( (0x000003ff & i_d) << (31 - 16 - 13) );

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
  /* Set RS */
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
unsigned int libxsmm_ppc64le_instr_xfx_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned int  i_r ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x07 & i_t) << (31 - 6 - 4) );
  /* Set R */
  l_instr |= (unsigned int)( (0x000003ff & i_r) << (31 - 11 - 9) );

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
unsigned int libxsmm_ppc64le_instr_0_wrapper( unsigned int i_instr ) {
  unsigned int op;

  switch( i_instr ) {
    case LIBXSMM_PPC64LE_INSTR_BLR:
    case LIBXSMM_PPC64LE_INSTR_NOP: {
      op = i_instr;
    } break;
    default: {
      return -1;
    }
  }

  return op;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_1_wrapper( unsigned int i_instr,
                                              unsigned int i_1 ) {
  unsigned int op;

  switch( i_instr ) {
    /* VX (vrb) form */
    case LIBXSMM_PPC64LE_INSTR_MTVSCR: {
      op = libxsmm_ppc64le_instr_vx_form_vrb( i_instr, (unsigned char)i_1 );
    } break;
    /* VX (vrt) form */
    case LIBXSMM_PPC64LE_INSTR_MFVSCR: {
    op = libxsmm_ppc64le_instr_vx_form_vrt( i_instr, (unsigned char)i_1 );
    } break;
    /* X (3) form */
    case LIBXSMM_PPC64LE_INSTR_XXMFACC:
    case LIBXSMM_PPC64LE_INSTR_XXMTACC:
    case LIBXSMM_PPC64LE_INSTR_XXSETACCZ: {
      op = libxsmm_ppc64le_instr_x_form_3( i_instr, (unsigned char)i_1 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_2_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2 ) {
  unsigned int op;

  switch( i_instr ) {
    /* X (33) form */
    case LIBXSMM_PPC64LE_INSTR_MCRFS: {
      op = libxsmm_ppc64le_instr_x_form_33( i_instr, (unsigned char)i_1, (unsigned char)i_2 );
    } break;
    /* XFX form */
    case LIBXSMM_PPC64LE_INSTR_MTSPR: {
      op = libxsmm_ppc64le_instr_xfx_form( i_instr, (unsigned char)i_1, (unsigned int)i_2 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_3_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3 ) {
  unsigned int op;

  switch( i_instr ) {
    /* B form */
    case LIBXSMM_PPC64LE_INSTR_BC: {
      op = libxsmm_ppc64le_instr_b_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
    } break;
    /* D form */
    case LIBXSMM_PPC64LE_INSTR_LFD:
    case LIBXSMM_PPC64LE_INSTR_STFD:
    case LIBXSMM_PPC64LE_INSTR_ORI:
    case LIBXSMM_PPC64LE_INSTR_ANDI:
    case LIBXSMM_PPC64LE_INSTR_ADDI: {
      op = libxsmm_ppc64le_instr_d_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
    } break;
    /* DS form */
    case LIBXSMM_PPC64LE_INSTR_STFDP:
    case LIBXSMM_PPC64LE_INSTR_STXSD:
    case LIBXSMM_PPC64LE_INSTR_STXSSP:
    case LIBXSMM_PPC64LE_INSTR_STD:
    case LIBXSMM_PPC64LE_INSTR_STDU:
    case LIBXSMM_PPC64LE_INSTR_STQ:
    case LIBXSMM_PPC64LE_INSTR_LD: {
      op = libxsmm_ppc64le_instr_ds_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
    } break;
    /* X (355) form */
    case LIBXSMM_PPC64LE_INSTR_FCMPU:
    case LIBXSMM_PPC64LE_INSTR_FCMPO:
    case LIBXSMM_PPC64LE_INSTR_LVEBX: {
      op = libxsmm_ppc64le_instr_x_form_355( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
    } break;
    /* X (555) form */
    case LIBXSMM_PPC64LE_INSTR_LVEHX:
    case LIBXSMM_PPC64LE_INSTR_LVEWX:
    case LIBXSMM_PPC64LE_INSTR_LVSL:
    case LIBXSMM_PPC64LE_INSTR_LVSR:
    case LIBXSMM_PPC64LE_INSTR_STVX:
    case LIBXSMM_PPC64LE_INSTR_STVXL:
    case LIBXSMM_PPC64LE_INSTR_STVEBX:
    case LIBXSMM_PPC64LE_INSTR_STVEHX:
    case LIBXSMM_PPC64LE_INSTR_STVEWX:
    case LIBXSMM_PPC64LE_INSTR_LVX:
    case LIBXSMM_PPC64LE_INSTR_LVXL:
    case LIBXSMM_PPC64LE_INSTR_AND:
    case LIBXSMM_PPC64LE_INSTR_NAND:
    case LIBXSMM_PPC64LE_INSTR_OR:
    case LIBXSMM_PPC64LE_INSTR_NOR:
    case LIBXSMM_PPC64LE_INSTR_ADD: {
      op = libxsmm_ppc64le_instr_x_form_555( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
    } break;
    /* X (581) form */
    case LIBXSMM_PPC64LE_INSTR_XXSPLTIB: {
        op = libxsmm_ppc64le_instr_x_form_581( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
    } break;
    /* XX2 (2) form */
    case LIBXSMM_PPC64LE_INSTR_XSXEXPDP:
    case LIBXSMM_PPC64LE_INSTR_XSXSIGDP: {
      op = libxsmm_ppc64le_instr_xx2_form_2( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3);
    } break;
    default: {
      return -1;
    }
  }

  return op;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_4_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4 ) {
  unsigned int op;

  switch( i_instr ) {
    /* D (bf) form */
    case LIBXSMM_PPC64LE_INSTR_CMPI: {
      op = libxsmm_ppc64le_instr_d_form_bf( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned int)i_4 );
    } break;
    /* DQ (x) form */
    case LIBXSMM_PPC64LE_INSTR_LXV:
    case LIBXSMM_PPC64LE_INSTR_STXV:{
      op = libxsmm_ppc64le_instr_dq_form_x( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4 );
    } break;
    /* X form */
    case LIBXSMM_PPC64LE_INSTR_XXGENPCVBM:
    case LIBXSMM_PPC64LE_INSTR_XXGENPCVWM:
    case LIBXSMM_PPC64LE_INSTR_XXGENPCVHM:
    case LIBXSMM_PPC64LE_INSTR_XXGENPCVDM:
    case LIBXSMM_PPC64LE_INSTR_XSIEXPDP:
    case LIBXSMM_PPC64LE_INSTR_LXSIWZX:
    case LIBXSMM_PPC64LE_INSTR_LXSIWAX:
    case LIBXSMM_PPC64LE_INSTR_STXSIWX:
    case LIBXSMM_PPC64LE_INSTR_LXVDSX:
    case LIBXSMM_PPC64LE_INSTR_LXVB16X:
    case LIBXSMM_PPC64LE_INSTR_LXVD2X:
    case LIBXSMM_PPC64LE_INSTR_LXVH8X:
    case LIBXSMM_PPC64LE_INSTR_LXVW4X:
    case LIBXSMM_PPC64LE_INSTR_LXVX:
    case LIBXSMM_PPC64LE_INSTR_STXVX:
    case LIBXSMM_PPC64LE_INSTR_LXSSPX:
    case LIBXSMM_PPC64LE_INSTR_LXSDX:
    case LIBXSMM_PPC64LE_INSTR_STXSDX:
    case LIBXSMM_PPC64LE_INSTR_LXSIBZX:
    case LIBXSMM_PPC64LE_INSTR_LXSIHZX:
    case LIBXSMM_PPC64LE_INSTR_STXSIBX:
    case LIBXSMM_PPC64LE_INSTR_STXSIHX:
    case LIBXSMM_PPC64LE_INSTR_STXVLL:
    case LIBXSMM_PPC64LE_INSTR_STXVL:
    case LIBXSMM_PPC64LE_INSTR_LXVLL:
    case LIBXSMM_PPC64LE_INSTR_LXVL:
    case LIBXSMM_PPC64LE_INSTR_LXVWSX:
    case LIBXSMM_PPC64LE_INSTR_LXVRBX:
    case LIBXSMM_PPC64LE_INSTR_LXVRDX:
    case LIBXSMM_PPC64LE_INSTR_LXVRHX:
    case LIBXSMM_PPC64LE_INSTR_LXVRWX:
    case LIBXSMM_PPC64LE_INSTR_STXVB16X:
    case LIBXSMM_PPC64LE_INSTR_STXVD2X:
    case LIBXSMM_PPC64LE_INSTR_STXVH8X:
    case LIBXSMM_PPC64LE_INSTR_STXVW4X:
    case LIBXSMM_PPC64LE_INSTR_STXVRBX:
    case LIBXSMM_PPC64LE_INSTR_STXVRDX: {
      op = libxsmm_ppc64le_instr_x_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
    } break;
      /* X (4155) form */
    case LIBXSMM_PPC64LE_INSTR_LXVPX:
    case LIBXSMM_PPC64LE_INSTR_STXVPX: {
      op = libxsmm_ppc64le_instr_x_form_4155( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
    } break;
    /* XX2 (3) form */
    case LIBXSMM_PPC64LE_INSTR_XSCVDPUXWS:
    case LIBXSMM_PPC64LE_INSTR_XSCVDPSXWS:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPUXWS:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPSXWS:
    case LIBXSMM_PPC64LE_INSTR_XVCVUXWSP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSXWSP:
    case LIBXSMM_PPC64LE_INSTR_XVCVDPUXWS:
    case LIBXSMM_PPC64LE_INSTR_XVCVDPSXWS:
    case LIBXSMM_PPC64LE_INSTR_XVCVUXWDP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSXWDP:
    case LIBXSMM_PPC64LE_INSTR_XSCVUXDSP:
    case LIBXSMM_PPC64LE_INSTR_XSCVSXDSP:
    case LIBXSMM_PPC64LE_INSTR_XSCVDPUXDS:
    case LIBXSMM_PPC64LE_INSTR_XSCVDPSXDS:
    case LIBXSMM_PPC64LE_INSTR_XSCVUXDDP:
    case LIBXSMM_PPC64LE_INSTR_XSCVSXDDP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPUXDS:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPSXDS:
    case LIBXSMM_PPC64LE_INSTR_XVCVUXDSP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSXDSP:
    case LIBXSMM_PPC64LE_INSTR_XVCVDPUXDS:
    case LIBXSMM_PPC64LE_INSTR_XVCVDPSXDS:
    case LIBXSMM_PPC64LE_INSTR_XVCVUXDDP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSXDDP:
    case LIBXSMM_PPC64LE_INSTR_XSRDPI:
    case LIBXSMM_PPC64LE_INSTR_XSRDPIZ:
    case LIBXSMM_PPC64LE_INSTR_XSRDPIP:
    case LIBXSMM_PPC64LE_INSTR_XSRDPIM:
    case LIBXSMM_PPC64LE_INSTR_XVRSPI:
    case LIBXSMM_PPC64LE_INSTR_XVRSPIZ:
    case LIBXSMM_PPC64LE_INSTR_XVRSPIP:
    case LIBXSMM_PPC64LE_INSTR_XVRSPIM:
    case LIBXSMM_PPC64LE_INSTR_XVRDPI:
    case LIBXSMM_PPC64LE_INSTR_XVRDPIZ:
    case LIBXSMM_PPC64LE_INSTR_XVRDPIP:
    case LIBXSMM_PPC64LE_INSTR_XVRDPIM:
    case LIBXSMM_PPC64LE_INSTR_XSCVDPSP:
    case LIBXSMM_PPC64LE_INSTR_XSRSP:
    case LIBXSMM_PPC64LE_INSTR_XSCVSPDP:
    case LIBXSMM_PPC64LE_INSTR_XSABSDP:
    case LIBXSMM_PPC64LE_INSTR_XSNABSDP:
    case LIBXSMM_PPC64LE_INSTR_XSNEGDP:
    case LIBXSMM_PPC64LE_INSTR_XVCVDPSP:
    case LIBXSMM_PPC64LE_INSTR_XVABSSP:
    case LIBXSMM_PPC64LE_INSTR_XVNABSSP:
    case LIBXSMM_PPC64LE_INSTR_XVNEGSP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPDP:
    case LIBXSMM_PPC64LE_INSTR_XVABSDP:
    case LIBXSMM_PPC64LE_INSTR_XVNABSDP:
    case LIBXSMM_PPC64LE_INSTR_XVNEGDP:
    case LIBXSMM_PPC64LE_INSTR_XSRSQRTESP:
    case LIBXSMM_PPC64LE_INSTR_XSRESP:
    case LIBXSMM_PPC64LE_INSTR_XSRSQRTEDP:
    case LIBXSMM_PPC64LE_INSTR_XSREDP:
    case LIBXSMM_PPC64LE_INSTR_XSSQRTSP:
    case LIBXSMM_PPC64LE_INSTR_XSSQRTDP:
    case LIBXSMM_PPC64LE_INSTR_XSRDPIC:
    case LIBXSMM_PPC64LE_INSTR_XVSQRTSP:
    case LIBXSMM_PPC64LE_INSTR_XVRSPIC:
    case LIBXSMM_PPC64LE_INSTR_XVSQRTDP:
    case LIBXSMM_PPC64LE_INSTR_XVRDPIC:
    case LIBXSMM_PPC64LE_INSTR_XSCVDPSPN:
    case LIBXSMM_PPC64LE_INSTR_XSCVSPDPN:
    case LIBXSMM_PPC64LE_INSTR_XSCVHPDP:
    case LIBXSMM_PPC64LE_INSTR_XSCVDPHP:
    case LIBXSMM_PPC64LE_INSTR_XVXEXPDP:
    case LIBXSMM_PPC64LE_INSTR_XVXEXPSP:
    case LIBXSMM_PPC64LE_INSTR_XVXSIGDP:
    case LIBXSMM_PPC64LE_INSTR_XVXSIGSP:
    case LIBXSMM_PPC64LE_INSTR_XXBRH:
    case LIBXSMM_PPC64LE_INSTR_XXBRW:
    case LIBXSMM_PPC64LE_INSTR_XXBRD:
    case LIBXSMM_PPC64LE_INSTR_XXBRQ:
    case LIBXSMM_PPC64LE_INSTR_XVCVBF16SP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPBF16:
    case LIBXSMM_PPC64LE_INSTR_XVCVHPSP:
    case LIBXSMM_PPC64LE_INSTR_XVCVSPHP: {
      op = libxsmm_ppc64le_instr_xx2_form_3( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}



LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_5_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5 ) {
  unsigned int op;

  switch( i_instr ) {
    /* B (al) form */
    case LIBXSMM_PPC64LE_INSTR_UNDEF: {
      op = libxsmm_ppc64le_instr_b_form_al( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5 );
    } break;
    /* XX2 (4) form */
    case LIBXSMM_PPC64LE_INSTR_XXEXTRACTUW:
    case LIBXSMM_PPC64LE_INSTR_XXINSERTW:
    case LIBXSMM_PPC64LE_INSTR_XXSPLTW: {
      op = libxsmm_ppc64le_instr_xx2_form_4( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
    } break;
    /* XX3 (0) form */
    case LIBXSMM_PPC64LE_INSTR_XSCMPUDP:
    case LIBXSMM_PPC64LE_INSTR_XSCMPODP:
    case LIBXSMM_PPC64LE_INSTR_XSCMPEXPDP:
    case LIBXSMM_PPC64LE_INSTR_XVI8GER4:
    case LIBXSMM_PPC64LE_INSTR_XVI8GER4PP:
    case LIBXSMM_PPC64LE_INSTR_XVI8GER4SPP:
    case LIBXSMM_PPC64LE_INSTR_XVI4GER8:
    case LIBXSMM_PPC64LE_INSTR_XVI4GER8PP:
    case LIBXSMM_PPC64LE_INSTR_XVI16GER2:
    case LIBXSMM_PPC64LE_INSTR_XVI16GER2PP:
    case LIBXSMM_PPC64LE_INSTR_XVBF16GER2:
    case LIBXSMM_PPC64LE_INSTR_XVI16GER2S:
    case LIBXSMM_PPC64LE_INSTR_XVI16GER2SPP:
    case LIBXSMM_PPC64LE_INSTR_XVBF16GER2NN:
    case LIBXSMM_PPC64LE_INSTR_XVBF16GER2NP:
    case LIBXSMM_PPC64LE_INSTR_XVBF16GER2PN:
    case LIBXSMM_PPC64LE_INSTR_XVBF16GER2PP:
    case LIBXSMM_PPC64LE_INSTR_XVF16GER2:
    case LIBXSMM_PPC64LE_INSTR_XVF16GER2NN:
    case LIBXSMM_PPC64LE_INSTR_XVF16GER2NP:
    case LIBXSMM_PPC64LE_INSTR_XVF16GER2PN:
    case LIBXSMM_PPC64LE_INSTR_XVF16GER2PP:
    case LIBXSMM_PPC64LE_INSTR_XVF32GER:
    case LIBXSMM_PPC64LE_INSTR_XVF32GERNN:
    case LIBXSMM_PPC64LE_INSTR_XVF32GERNP:
    case LIBXSMM_PPC64LE_INSTR_XVF32GERPN:
    case LIBXSMM_PPC64LE_INSTR_XVF32GERPP:
    case LIBXSMM_PPC64LE_INSTR_XVF64GER:
    case LIBXSMM_PPC64LE_INSTR_XVF64GERNN:
    case LIBXSMM_PPC64LE_INSTR_XVF64GERNP:
    case LIBXSMM_PPC64LE_INSTR_XVF64GERPN:
    case LIBXSMM_PPC64LE_INSTR_XVF64GERPP: {
      op = libxsmm_ppc64le_instr_xx3_form_0( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_6_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6 ) {
  unsigned int op;

  switch( i_instr ) {
    /* MD form */
    case LIBXSMM_PPC64LE_INSTR_RLDICR: {
      op = libxsmm_ppc64le_instr_md_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
    } break;
    /* XX3 (6) form */
    case LIBXSMM_PPC64LE_INSTR_XSMAXCDP:
    case LIBXSMM_PPC64LE_INSTR_XSMINCDP:
    case LIBXSMM_PPC64LE_INSTR_XSMAXJDP:
    case LIBXSMM_PPC64LE_INSTR_XSMINJDP:
    case LIBXSMM_PPC64LE_INSTR_XSMAXDP:
    case LIBXSMM_PPC64LE_INSTR_XSMINDP:
    case LIBXSMM_PPC64LE_INSTR_XSCPSGNDP:
    case LIBXSMM_PPC64LE_INSTR_XVMAXDP:
    case LIBXSMM_PPC64LE_INSTR_XVMAXSP:
    case LIBXSMM_PPC64LE_INSTR_XVMINDP:
    case LIBXSMM_PPC64LE_INSTR_XVMINSP:
    case LIBXSMM_PPC64LE_INSTR_XVIEXPDP:
    case LIBXSMM_PPC64LE_INSTR_XVIEXPSP:
    case LIBXSMM_PPC64LE_INSTR_XVCPSGNDP:
    case LIBXSMM_PPC64LE_INSTR_XVCPSGNSP:
    case LIBXSMM_PPC64LE_INSTR_XSMADDASP:
    case LIBXSMM_PPC64LE_INSTR_XSMADDMSP:
    case LIBXSMM_PPC64LE_INSTR_XSMADDADP:
    case LIBXSMM_PPC64LE_INSTR_XSMADDMDP:
    case LIBXSMM_PPC64LE_INSTR_XSMSUBASP:
    case LIBXSMM_PPC64LE_INSTR_XSMSUBMSP:
    case LIBXSMM_PPC64LE_INSTR_XSMSUBADP:
    case LIBXSMM_PPC64LE_INSTR_XSMSUBMDP:
    case LIBXSMM_PPC64LE_INSTR_XVMADDASP:
    case LIBXSMM_PPC64LE_INSTR_XVMADDMSP:
    case LIBXSMM_PPC64LE_INSTR_XVMADDADP:
    case LIBXSMM_PPC64LE_INSTR_XVMADDMDP:
    case LIBXSMM_PPC64LE_INSTR_XVMSUBASP:
    case LIBXSMM_PPC64LE_INSTR_XVMSUBMSP:
    case LIBXSMM_PPC64LE_INSTR_XVMSUBADP:
    case LIBXSMM_PPC64LE_INSTR_XVMSUBMDP:
    case LIBXSMM_PPC64LE_INSTR_XSNMADDASP:
    case LIBXSMM_PPC64LE_INSTR_XSNMADDMSP:
    case LIBXSMM_PPC64LE_INSTR_XSNMADDADP:
    case LIBXSMM_PPC64LE_INSTR_XSNMADDMDP:
    case LIBXSMM_PPC64LE_INSTR_XSNMSUBASP:
    case LIBXSMM_PPC64LE_INSTR_XSNMSUBMSP:
    case LIBXSMM_PPC64LE_INSTR_XSNMSUBADP:
    case LIBXSMM_PPC64LE_INSTR_XSNMSUBMDP:
    case LIBXSMM_PPC64LE_INSTR_XVNMADDASP:
    case LIBXSMM_PPC64LE_INSTR_XVNMADDMSP:
    case LIBXSMM_PPC64LE_INSTR_XVNMADDADP:
    case LIBXSMM_PPC64LE_INSTR_XVNMADDMDP:
    case LIBXSMM_PPC64LE_INSTR_XVNMSUBASP:
    case LIBXSMM_PPC64LE_INSTR_XVNMSUBMSP:
    case LIBXSMM_PPC64LE_INSTR_XVNMSUBADP:
    case LIBXSMM_PPC64LE_INSTR_XVNMSUBMDP:
    case LIBXSMM_PPC64LE_INSTR_XXMRGHW:
    case LIBXSMM_PPC64LE_INSTR_XXMRGLW:
    case LIBXSMM_PPC64LE_INSTR_XXPERM:
    case LIBXSMM_PPC64LE_INSTR_XXPERMR:
    case LIBXSMM_PPC64LE_INSTR_XXLAND:
    case LIBXSMM_PPC64LE_INSTR_XXLANDC:
    case LIBXSMM_PPC64LE_INSTR_XXLNAND:
    case LIBXSMM_PPC64LE_INSTR_XXLEQV:
    case LIBXSMM_PPC64LE_INSTR_XXLNOR:
    case LIBXSMM_PPC64LE_INSTR_XXLORC:
    case LIBXSMM_PPC64LE_INSTR_XXLOR:
    case LIBXSMM_PPC64LE_INSTR_XXLXOR:
    case LIBXSMM_PPC64LE_INSTR_XSCMPEQDP:
    case LIBXSMM_PPC64LE_INSTR_XSCMPGTDP:
    case LIBXSMM_PPC64LE_INSTR_XSCMPGEDP:
    case LIBXSMM_PPC64LE_INSTR_XVADDDP:
    case LIBXSMM_PPC64LE_INSTR_XVADDSP:
    case LIBXSMM_PPC64LE_INSTR_XVMULDP:
    case LIBXSMM_PPC64LE_INSTR_XVMULSP:
    case LIBXSMM_PPC64LE_INSTR_XVSUBDP:
    case LIBXSMM_PPC64LE_INSTR_XVSUBSP: {
      op = libxsmm_ppc64le_instr_xx3_form_6( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_7_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6,
                                              unsigned int i_7 ) {
  unsigned int op;

  switch( i_instr ) {
    /* XX3 (3) form */
    case LIBXSMM_PPC64LE_INSTR_XXPERMDI: {
      op = libxsmm_ppc64le_instr_xx3_form_3( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
    } break;
    /* XX3 (5) form */
    case LIBXSMM_PPC64LE_INSTR_XVCMPEQDP:
    case LIBXSMM_PPC64LE_INSTR_XVCMPEQSP:
    case LIBXSMM_PPC64LE_INSTR_XVCMPGTDP:
    case LIBXSMM_PPC64LE_INSTR_XVCMPGTSP:
    case LIBXSMM_PPC64LE_INSTR_XVCMPGEDP:
    case LIBXSMM_PPC64LE_INSTR_XVCMPGESP: {
      op = libxsmm_ppc64le_instr_xx3_form_5( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_8_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6,
                                              unsigned int i_7,
                                              unsigned int i_8 ) {
  unsigned int op;

  switch( i_instr ) {
    /* XX4 form */
    case LIBXSMM_PPC64LE_INSTR_XXSEL: {
      op = libxsmm_ppc64le_instr_xx4_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7, (unsigned char)i_8 );
    } break;
    default: {
      return -1;
    }
  }

  return op;
}



LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_4l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4 ) {
  unsigned long op;

  switch( i_instr ) {
    /* D-8RR (0, 3) form */
    case LIBXSMM_PPC64LE_INSTR_XXSPLTIDP:
    case LIBXSMM_PPC64LE_INSTR_XXSPLTIW: {
      op = libxsmm_ppc64le_instr_d_form_1_8rr3( i_instr, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned int)i_4);
    } break;
    default: {
      return -1;
    }
  }
  return op;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_5l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5 ) {
  unsigned long op;

  switch( i_instr ) {
    /* D-8RR (0, 3) form */
    case LIBXSMM_PPC64LE_INSTR_XXSPLTI32DX: {
      op = libxsmm_ppc64le_instr_d_form_0_8rr3( i_instr, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned int)i_5);
    } break;
    default: {
      return -1;
    }
  }
  return op;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_7l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5,
                                                unsigned int i_6,
                                                unsigned int i_7 ) {
  unsigned long op;

  switch( i_instr ) {
    /* XX3-MMIRR (0, 0) form */
    case LIBXSMM_PPC64LE_INSTR_PMXVF32GER:
    case LIBXSMM_PPC64LE_INSTR_PMXVF32GERNN:
    case LIBXSMM_PPC64LE_INSTR_PMXVF32GERNP:
    case LIBXSMM_PPC64LE_INSTR_PMXVF32GERPN:
    case LIBXSMM_PPC64LE_INSTR_PMXVF32GERPP: {
      op = libxsmm_ppc64le_instr_xx3_form_0_mmirr0( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
    } break;
    /* XX3-MMIRR (0, 1) form */
    case LIBXSMM_PPC64LE_INSTR_PMXVF64GER:
    case LIBXSMM_PPC64LE_INSTR_PMXVF64GERNN:
    case LIBXSMM_PPC64LE_INSTR_PMXVF64GERNP:
    case LIBXSMM_PPC64LE_INSTR_PMXVF64GERPN:
    case LIBXSMM_PPC64LE_INSTR_PMXVF64GERPP: {
      op = libxsmm_ppc64le_instr_xx3_form_0_mmirr1( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
    } break;
    default: {
      return -1;
    }
  }
  return op;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_8l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5,
                                                unsigned int i_6,
                                                unsigned int i_7,
                                                unsigned int i_8 ) {
  unsigned long op;

  switch( i_instr ) {
    /* XX3-MMIRR (0, 3) form */
    case LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2:
    case LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2NN:
    case LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2NP:
    case LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2PN:
    case LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2PP:
    case LIBXSMM_PPC64LE_INSTR_PMXVF16GER2:
    case LIBXSMM_PPC64LE_INSTR_PMXVF16GER2NN:
    case LIBXSMM_PPC64LE_INSTR_PMXVF16GER2NP:
    case LIBXSMM_PPC64LE_INSTR_PMXVF16GER2PN:
    case LIBXSMM_PPC64LE_INSTR_PMXVF16GER2PP:
    case LIBXSMM_PPC64LE_INSTR_PMXVI16GER2:
    case LIBXSMM_PPC64LE_INSTR_PMXVI16GER2S:
    case LIBXSMM_PPC64LE_INSTR_PMXVI16GER2SPP:
    case LIBXSMM_PPC64LE_INSTR_PMXVI16GER2PP: {
      op = libxsmm_ppc64le_instr_xx3_form_0_mmirr3( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7, (unsigned char)i_8 );
    } break;
    /* XX4-8RR (0) form */
    case LIBXSMM_PPC64LE_INSTR_XXBLENDVB:
    case LIBXSMM_PPC64LE_INSTR_XXBLENDVD:
    case LIBXSMM_PPC64LE_INSTR_XXBLENDVH:
    case LIBXSMM_PPC64LE_INSTR_XXBLENDVW: {
        op =libxsmm_ppc64le_instr_xx4_form_8rr0( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7, (unsigned char)i_8 );
    } break;
    default: {
      return -1;
    }
  }
  return op;
}


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_9l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5,
                                                unsigned int i_6,
                                                unsigned int i_7,
                                                unsigned int i_8,
                                                unsigned int i_9 ) {
  unsigned long op;

  switch( i_instr ) {
    /* XX4-8RR (2) form */
    case LIBXSMM_PPC64LE_INSTR_XXEVAL: {
      op =libxsmm_ppc64le_instr_xx4_form_8rr2( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7, (unsigned char)i_8, (unsigned char)i_9 );
    } break;
    default: {
      return -1;
    }
  }
  return op;
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr( libxsmm_generated_code * io_generated_code,
                              unsigned int           i_instr ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_0_wrapper( i_instr );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
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
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_1_wrapper( i_instr,
                                                         i_0 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_2( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_2_wrapper( i_instr,
                                                         i_0,
                                                         i_1 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_3( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_3_wrapper( i_instr,
                                                         i_0,
                                                         i_1,
                                                         i_2);
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_4( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_4_wrapper( i_instr,
                                                         i_0,
                                                         i_1,
                                                         i_2,
                                                         i_3 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
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
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_5_wrapper( i_instr,
                                                         i_0,
                                                         i_1,
                                                         i_2,
                                                         i_3,
                                                         i_4 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_6( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4,
                              unsigned int             i_5 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_6_wrapper( i_instr,
                                                         i_0,
                                                         i_1,
                                                         i_2,
                                                         i_3,
                                                         i_4,
                                                         i_5 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_7( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4,
                              unsigned int             i_5,
                              unsigned int             i_6 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_7_wrapper( i_instr,
                                                         i_0,
                                                         i_1,
                                                         i_2,
                                                         i_3,
                                                         i_4,
                                                         i_5,
                                                         i_6 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_8( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4,
                              unsigned int             i_5,
                              unsigned int             i_6,
                              unsigned int             i_7 ) {

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned int l_op = libxsmm_ppc64le_instr_8_wrapper( i_instr,
                                                     i_0,
                                                     i_1,
                                                     i_2,
                                                     i_3,
                                                     i_4,
                                                     i_5,
                                                     i_6,
                                                     i_7 );
    if ( l_op != -1 ) {
      l_code[l_code_head] = l_op;
      io_generated_code->code_size += 4;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_4( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned long l_op = libxsmm_ppc64le_instr_4l_wrapper( i_instr,
                                                           i_0,
                                                           i_1,
                                                           i_2,
                                                           i_3 );

    if ( l_op != -1 ) {
      unsigned int l_op_h = (unsigned int)( ( l_op >> 32 ) & 0x00000000ffffffff );
      unsigned int l_op_l = (unsigned int)( l_op & 0x00000000ffffffff );

      /* From ABI 8-byte 'prefix' ops cannot cross 64-byte boundaries */
      if ( ( l_code_head / 16 ) == ( ( l_code_head + 1 ) / 16 ) ) {
        l_code[l_code_head] = l_op_h;
        l_code[l_code_head + 1] = l_op_l;
        io_generated_code->code_size += 8;
      } else {
        unsigned int l_nop = libxsmm_ppc64le_instr_0_wrapper( LIBXSMM_PPC64LE_INSTR_NOP );
        l_code[l_code_head] = l_nop;
        l_code[l_code_head + 1] = l_op_h;
        l_code[l_code_head + 2] = l_op_l;
        io_generated_code->code_size += 12;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_5( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3,
                                     unsigned int             i_4 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned long l_op = libxsmm_ppc64le_instr_5l_wrapper( i_instr,
                                                           i_0,
                                                           i_1,
                                                           i_2,
                                                           i_3,
                                                           i_4 );

    if ( l_op != -1 ) {
      unsigned int l_op_h = (unsigned int)( ( l_op >> 32 ) & 0x00000000ffffffff );
      unsigned int l_op_l = (unsigned int)( l_op & 0x00000000ffffffff );

      /* From ABI 8-byte 'prefix' ops cannot cross 64-byte boundaries */
      if ( ( l_code_head / 16 ) == ( ( l_code_head + 1 ) / 16 ) ) {
        l_code[l_code_head] = l_op_h;
        l_code[l_code_head + 1] = l_op_l;
        io_generated_code->code_size += 8;
      } else {
        unsigned int l_nop = libxsmm_ppc64le_instr_0_wrapper( LIBXSMM_PPC64LE_INSTR_NOP );
        l_code[l_code_head] = l_nop;
        l_code[l_code_head + 1] = l_op_h;
        l_code[l_code_head + 2] = l_op_l;
        io_generated_code->code_size += 12;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_7( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3,
                                     unsigned int             i_4,
                                     unsigned int             i_5,
                                     unsigned int             i_6 ) {
 if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned long l_op = libxsmm_ppc64le_instr_7l_wrapper( i_instr,
                                                           i_0,
                                                           i_1,
                                                           i_2,
                                                           i_3,
                                                           i_4,
                                                           i_5,
                                                           i_6 );

    if ( l_op != -1 ) {
      unsigned int l_op_h = (unsigned int)( ( l_op >> 32 ) & 0x00000000ffffffff );
      unsigned int l_op_l = (unsigned int)( l_op & 0x00000000ffffffff );

      /* From ABI 8-byte 'prefix' ops cannot cross 64-byte boundaries */
      if ( ( l_code_head / 16 ) == ( ( l_code_head + 1 ) / 16 ) ) {
        l_code[l_code_head] = l_op_h;
        l_code[l_code_head + 1] = l_op_l;
        io_generated_code->code_size += 8;
      } else {
        unsigned int l_nop = libxsmm_ppc64le_instr_0_wrapper( LIBXSMM_PPC64LE_INSTR_NOP );
        l_code[l_code_head] = l_nop;
        l_code[l_code_head + 1] = l_op_h;
        l_code[l_code_head + 2] = l_op_l;
        io_generated_code->code_size += 12;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_8( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3,
                                     unsigned int             i_4,
                                     unsigned int             i_5,
                                     unsigned int             i_6,
                                     unsigned int             i_7 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned long l_op = libxsmm_ppc64le_instr_8l_wrapper( i_instr,
                                                           i_0,
                                                           i_1,
                                                           i_2,
                                                           i_3,
                                                           i_4,
                                                           i_5,
                                                           i_6,
                                                           i_7 );

    if ( l_op != -1 ) {
      unsigned int l_op_h = (unsigned int)( ( l_op >> 32 ) & 0x00000000ffffffff );
      unsigned int l_op_l = (unsigned int)( l_op & 0x00000000ffffffff );

      /* From ABI 8-byte 'prefix' ops cannot cross 64-byte boundaries */
      if ( ( l_code_head / 16 ) == ( ( l_code_head + 1 ) / 16 ) ) {
        l_code[l_code_head] = l_op_h;
        l_code[l_code_head + 1] = l_op_l;
        io_generated_code->code_size += 8;
      } else {
        unsigned int l_nop = libxsmm_ppc64le_instr_0_wrapper( LIBXSMM_PPC64LE_INSTR_NOP );
        l_code[l_code_head] = l_nop;
        l_code[l_code_head + 1] = l_op_h;
        l_code[l_code_head + 2] = l_op_l;
        io_generated_code->code_size += 12;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_9( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3,
                                     unsigned int             i_4,
                                     unsigned int             i_5,
                                     unsigned int             i_6,
                                     unsigned int             i_7,
                                     unsigned int             i_8 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_code_head = io_generated_code->code_size / 4;
    unsigned int *l_code = (unsigned int*) io_generated_code->generated_code;
    unsigned long l_op = libxsmm_ppc64le_instr_9l_wrapper( i_instr,
                                                           i_0,
                                                           i_1,
                                                           i_2,
                                                           i_3,
                                                           i_4,
                                                           i_5,
                                                           i_6,
                                                           i_7,
                                                           i_8 );

    if ( l_op != -1 ) {
      unsigned int l_op_h = (unsigned int)( ( l_op >> 32 ) & 0x00000000ffffffff );
      unsigned int l_op_l = (unsigned int)( l_op & 0x00000000ffffffff );

      /* From ABI 8-byte 'prefix' ops cannot cross 64-byte boundaries */
      if ( ( l_code_head / 16 ) == ( ( l_code_head + 1 ) / 16 ) ) {
        l_code[l_code_head] = l_op_h;
        l_code[l_code_head + 1] = l_op_l;
        io_generated_code->code_size += 8;
      } else {
        unsigned int l_nop = libxsmm_ppc64le_instr_0_wrapper( LIBXSMM_PPC64LE_INSTR_NOP );
        l_code[l_code_head] = l_nop;
        l_code[l_code_head + 1] = l_op_h;
        l_code[l_code_head + 2] = l_op_l;
        io_generated_code->code_size += 12;
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    }
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_open_stream( libxsmm_generated_code * io_generated_code,
                                        libxsmm_ppc64le_reg    * io_reg_tracker ) {
  /* From "64-Bit ELF V2 ABI Specification: Power Architecture"
   * GPR3 contains the pointer to the first arguement
   * The first arg to the gemm is a point to a libxsmm_gemm_param struct, which we
   * can then unpack, into the standard
   */

  unsigned int gpr_offset = 0;
  unsigned int fpr_offset = (LIBXSMM_PPC64LE_GPR_NMAX - LIBXSMM_PPC64LE_GPR_IVOL)*8;
  unsigned int vsr_offset = fpr_offset + (LIBXSMM_PPC64LE_FPR_NMAX - LIBXSMM_PPC64LE_FPR_IVOL)*8;

  /* decrease stack pointer */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_GPR_SP,
                           -512 );

  /* save non-volatile general purpose registers */
  for( unsigned int gpr = LIBXSMM_PPC64LE_GPR_IVOL; gpr < LIBXSMM_PPC64LE_GPR_NMAX; ++gpr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STD,
                             gpr,
                             LIBXSMM_PPC64LE_GPR_SP,
                             (gpr - LIBXSMM_PPC64LE_GPR_IVOL)*8 + gpr_offset);
  }

  /* save non-volatile floating point registers */
  for( unsigned int fpr = LIBXSMM_PPC64LE_FPR_IVOL; fpr < LIBXSMM_PPC64LE_FPR_NMAX; ++fpr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STFD,
                             fpr,
                             LIBXSMM_PPC64LE_GPR_SP,
                             (fpr - LIBXSMM_PPC64LE_FPR_IVOL)*8 + fpr_offset );
  }

  /* save non-volatile vector registers */
  unsigned int index = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           index,
                           LIBXSMM_PPC64LE_GPR_SP,
                           vsr_offset );


  for( unsigned int vr = LIBXSMM_PPC64LE_VR_IVOL; vr < LIBXSMM_PPC64LE_VR_NMAX; ++vr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STVX,
                             vr,
                             0,
                             index );
    /* increment index if not last */
    if( vr < LIBXSMM_PPC64LE_VR_NMAX - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               index,
                               index,
                               16 );
    }
  }

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, index);

  /* Set up input args */
  unsigned int struct_ptr = LIBXSMM_PPC64LE_GPR_R6;
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           LIBXSMM_PPC64LE_GPR_R3,
                           struct_ptr,
                           LIBXSMM_PPC64LE_GPR_R3 );

  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R3,
                           struct_ptr,
                           8 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R4,
                           struct_ptr,
                           16 );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R5,
                           struct_ptr,
                           24 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_close_stream( libxsmm_generated_code * io_generated_code,
                                         libxsmm_ppc64le_reg    * io_reg_tracker ) {
  /* From "64-Bit ELF V2 ABI Specification: Power Architecture" */

  unsigned int gpr_offset = 0;
  unsigned int fpr_offset = (LIBXSMM_PPC64LE_GPR_NMAX - LIBXSMM_PPC64LE_GPR_IVOL)*8;
  unsigned int vsr_offset = fpr_offset + (LIBXSMM_PPC64LE_FPR_NMAX - LIBXSMM_PPC64LE_FPR_IVOL)*8;

  /* restore non-volatile general purpose registers */
  for( unsigned int gpr = LIBXSMM_PPC64LE_GPR_IVOL; gpr < LIBXSMM_PPC64LE_GPR_NMAX; ++gpr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LD,
                             gpr,
                             LIBXSMM_PPC64LE_GPR_SP,
                             (gpr - LIBXSMM_PPC64LE_GPR_IVOL)*8 + gpr_offset );
  }

  /* restore non-volatile floating point registers */
  for( unsigned int fpr = LIBXSMM_PPC64LE_FPR_IVOL; fpr < LIBXSMM_PPC64LE_FPR_NMAX; ++fpr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LFD,
                             fpr,
                             LIBXSMM_PPC64LE_GPR_SP,
                             (fpr - LIBXSMM_PPC64LE_FPR_IVOL)*8 + fpr_offset );
  }

  /* save non-volatile vector registers */
  unsigned int index = LIBXSMM_PPC64LE_GPR_R10; /* as non-volatile GPR have already been restored */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           index,
                           LIBXSMM_PPC64LE_GPR_SP,
                           vsr_offset );


  for( unsigned int vr = LIBXSMM_PPC64LE_VR_IVOL; vr < LIBXSMM_PPC64LE_VR_NMAX; ++vr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LVX,
                             vr,
                             0,
                             index );
    /* increment index if not last */
    if( vr < LIBXSMM_PPC64LE_VR_NMAX - 1) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDI,
                               index,
                               index,
                               16 );
    }
  }

  /* increase stack pointer */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_GPR_SP,
                           512 );

  /* return statement */
  libxsmm_ppc64le_instr( io_generated_code, LIBXSMM_PPC64LE_INSTR_BLR );
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_bytes( libxsmm_generated_code * io_generated_code,
                                          libxsmm_datatype const   i_datatype ) {
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
void libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( libxsmm_generated_code * io_generated_code,
                                                      libxsmm_ppc64le_reg    * io_reg_tracker,
                                                      unsigned int           * io_v ) {

  unsigned int l_scratch[4];
  for ( unsigned int l_i = 0; l_i < 4 ; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[0],
                           io_v[0],
                           io_v[2],
                           (0x0020 & io_v[0]) >> 5,
                           (0x0020 & io_v[2]) >> 5,
                           (0x0020 & l_scratch[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[2],
                           io_v[1],
                           io_v[3],
                           (0x0020 & io_v[1]) >> 5,
                           (0x0020 & io_v[3]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[1],
                           io_v[0],
                           io_v[2],
                           (0x0020 & io_v[0]) >> 5,
                           (0x0020 & io_v[2]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[3],
                           io_v[1],
                           io_v[3],
                           (0x0020 & io_v[1]) >> 5,
                           (0x0020 & io_v[3]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           io_v[0],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & io_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           io_v[2],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & io_v[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           io_v[1],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & io_v[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           io_v[3],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & io_v[3]) >> 5 );

  for ( unsigned int l_i =0 ; l_i < 4 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x2_inplace( libxsmm_generated_code * io_generated_code,
                                                      libxsmm_ppc64le_reg    * io_reg_tracker,
                                                      unsigned int           * io_v ) {
  unsigned int l_scratch = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );

  /* high double-words */
  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                           l_scratch,
                           io_v[0],
                           io_v[1],
                           0x0000,
                           (0x0020 & io_v[0]) >> 5,
                           (0x0020 & io_v[1]) >> 5,
                           (0x0020 & l_scratch) >> 5 );
  /* low double-words inplace */
  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                           io_v[1],
                           io_v[0],
                           io_v[1],
                           0x0003,
                           (0x0020 & io_v[0]) >> 5,
                           (0x0020 & io_v[1]) >> 5,
                           (0x0020 & io_v[1]) >> 5 );
  /* */
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           io_v[0],
                           l_scratch,
                           l_scratch,
                           (0x0020 & l_scratch) >> 5,
                           (0x0020 & l_scratch) >> 5,
                           (0x0020 & io_v[0]) >> 5 );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_register_jump_back_label( libxsmm_generated_code     * io_generated_code,
                                                     libxsmm_loop_label_tracker * io_loop_label_tracker ) {
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
void libxsmm_ppc64le_instr_cond_jump_back_to_label( libxsmm_generated_code     * io_generated_code,
                                                    unsigned int                 i_gpr,
                                                    libxsmm_loop_label_tracker * io_loop_label_tracker ) {
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
void libxsmm_ppc64le_instr_cond_jump_back_to_label_ctr( libxsmm_generated_code     * io_generated_code,
                                                        libxsmm_loop_label_tracker * io_loop_label_tracker ) {
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
