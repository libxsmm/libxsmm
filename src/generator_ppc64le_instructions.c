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
int libxsmm_ppc64le_graph_depth( libxsmm_ppc64le_node *i_vertex ) {
  int l_depth = i_vertex->depth;
  if ( l_depth < 0 ) {
    for (int i = 0; i < i_vertex->ndep; i++) {
      int temp = libxsmm_ppc64le_graph_depth(i_vertex->dep[i]) + 1;
      l_depth = (temp > l_depth) ? temp : l_depth;
    }
    i_vertex->depth = l_depth;
  }
  return l_depth;
}


LIBXSMM_API_INTERN
int libxsmm_ppc64le_sched_graph( libxsmm_ppc64le_node *i_root, libxsmm_ppc64le_node **io_sched) {
  int l_nsched = 0;
  libxsmm_ppc64le_node *l_node = NULL;
  char l_undisc = 1;

  while ( l_undisc == 1 ) {
    int l_max_depth = -1;
    l_undisc = 0;

    for ( int i=0; i < i_root->ndep; ++i ) {
      libxsmm_ppc64le_node *l_dep = i_root->dep[i];
      if ( l_dep->depth > l_max_depth && l_dep->scheduled == 0) {
        l_undisc = 1;
        l_node = l_dep;
        l_max_depth = l_dep->depth;
      }
    }

    if ( l_undisc == 1 ) {
      l_nsched += libxsmm_ppc64le_sched_graph(l_node, &io_sched[l_nsched]);
    }
  }

  io_sched[l_nsched] = i_root;
  i_root->scheduled = 1;
  l_nsched += 1;

  return l_nsched;
}


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
      if ( io_reg_tracker->gpr[i] >= LIBXSMM_PPC64LE_REG_FREE ) {
        io_reg_tracker->gpr[i] = LIBXSMM_PPC64LE_REG_USED;
        return i;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_FPR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_FPR_NMAX - 1; i >= 0; --i ) {
      if ( ( io_reg_tracker->fpr[i] >= LIBXSMM_PPC64LE_REG_FREE ) &&
           ( io_reg_tracker->vsr[i] >= LIBXSMM_PPC64LE_REG_FREE ) ) {
        io_reg_tracker->fpr[i] = LIBXSMM_PPC64LE_REG_USED;
        io_reg_tracker->vsr[i] = LIBXSMM_PPC64LE_REG_USED;
        return i;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_VR_NMAX - 1; i >= 0; --i ) {
      if ( ( io_reg_tracker->vr[i] >= LIBXSMM_PPC64LE_REG_FREE ) &&
           ( io_reg_tracker->vsr[i + LIBXSMM_PPC64LE_FPR_NMAX] >= LIBXSMM_PPC64LE_REG_FREE ) ) {
        io_reg_tracker->vr[i] = LIBXSMM_PPC64LE_REG_USED;
        io_reg_tracker->vsr[i + LIBXSMM_PPC64LE_FPR_NMAX] = LIBXSMM_PPC64LE_REG_USED;
        return i;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VSR ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_VSR_NMAX - 1; i >= 0; --i ) {
      if ( i < LIBXSMM_PPC64LE_FPR_NMAX ) {
        if ( ( io_reg_tracker->fpr[i] >= LIBXSMM_PPC64LE_REG_FREE ) &&
             ( io_reg_tracker->vsr[i] >= LIBXSMM_PPC64LE_REG_FREE ) ) {
          io_reg_tracker->fpr[i] = LIBXSMM_PPC64LE_REG_USED;
          io_reg_tracker->vsr[i] = LIBXSMM_PPC64LE_REG_USED;
          return i;
        }
      } else if ( i < LIBXSMM_PPC64LE_FPR_NMAX + LIBXSMM_PPC64LE_VR_NMAX ) {
        if ( ( io_reg_tracker->vr[i - LIBXSMM_PPC64LE_FPR_NMAX] >= LIBXSMM_PPC64LE_REG_FREE ) &&
             ( io_reg_tracker->vsr[i] >= LIBXSMM_PPC64LE_REG_FREE ) ) {
          io_reg_tracker->vr[i - LIBXSMM_PPC64LE_FPR_NMAX] = LIBXSMM_PPC64LE_REG_USED;
          io_reg_tracker->vsr[i] = LIBXSMM_PPC64LE_REG_USED;
          return i;
        }
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_ACC ) {
    for ( unsigned int i = LIBXSMM_PPC64LE_ACC_NMAX - 1; i >= 0; --i ) {
      if ( io_reg_tracker->acc[i] >= LIBXSMM_PPC64LE_REG_FREE ) {
        unsigned char is_free = 1;
        for ( unsigned int j = i*4; j < (i + 1)*4; ++j ) {
          is_free &= ( io_reg_tracker->fpr[j] >= LIBXSMM_PPC64LE_REG_FREE );
          is_free &= ( io_reg_tracker->vsr[j] >= LIBXSMM_PPC64LE_REG_FREE );
        }

        if ( is_free ) {
          io_reg_tracker->acc[i] = LIBXSMM_PPC64LE_REG_USED;
          for ( unsigned int j = i*4; j < (i + 1)*4; ++j ) {
            io_reg_tracker->fpr[j] = LIBXSMM_PPC64LE_REG_USED;
            io_reg_tracker->vsr[j] = LIBXSMM_PPC64LE_REG_USED;
          }
          return i;
        }
      }
    }
  }

  LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  printf("Oh dear\n");
  return -1;

}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_get_sequential_reg( libxsmm_generated_code *io_generated_code,
                                         libxsmm_ppc64le_reg    *io_reg_tracker,
                                         unsigned int const      i_reg_type,
                                         unsigned int const      i_n,
                                         unsigned int           *o_reg ) {
  if ( i_reg_type == LIBXSMM_PPC64LE_GPR ) {
    int l_i = 0;
    while ( l_i < LIBXSMM_PPC64LE_GPR_NMAX ) {
      unsigned char l_avail = 1;
      for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
        if ( io_reg_tracker->gpr[l_i + l_j] < LIBXSMM_PPC64LE_REG_FREE ) {
          l_avail = 0;
          l_i += l_j + 1;
          break;
        }
      }
      if ( l_avail ) {
        for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
          o_reg[l_j] = l_i + l_j;
          io_reg_tracker->gpr[l_i + l_j] = LIBXSMM_PPC64LE_REG_USED;
        }
        return;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_FPR ) {
    int l_i = 0;
    while ( l_i < LIBXSMM_PPC64LE_FPR_NMAX ) {
      unsigned char l_avail = 1;
      for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
        if ( io_reg_tracker->fpr[l_i + l_j] < LIBXSMM_PPC64LE_REG_FREE ||
             io_reg_tracker->vsr[l_i + l_j] < LIBXSMM_PPC64LE_REG_FREE ) {
          l_avail = 0;
          l_i += l_j + 1;
          break;
        }
      }
      if ( l_avail ) {
        for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
          o_reg[l_j] = l_i + l_j;
          io_reg_tracker->fpr[l_i + l_j] = LIBXSMM_PPC64LE_REG_USED;
          io_reg_tracker->vsr[l_i + l_j] = LIBXSMM_PPC64LE_REG_USED;
        }
        return;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VR ) {
    int l_i = 0;
    while ( l_i < LIBXSMM_PPC64LE_VR_NMAX ) {
      unsigned char l_avail = 1;
      for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
        if ( io_reg_tracker->vr[l_i + l_j] < LIBXSMM_PPC64LE_REG_FREE ||
             io_reg_tracker->vsr[l_i + l_j + LIBXSMM_PPC64LE_FPR_NMAX] < LIBXSMM_PPC64LE_REG_FREE ) {
          l_avail = 0;
          l_i += l_j + 1;
          break;
        }
      }
      if ( l_avail ) {
        for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
          o_reg[l_j] = l_i + l_j;
          io_reg_tracker->vr[l_i + l_j] = LIBXSMM_PPC64LE_REG_USED;
          io_reg_tracker->vsr[l_i + l_j + LIBXSMM_PPC64LE_FPR_NMAX] = LIBXSMM_PPC64LE_REG_USED;
        }
        return;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VSR ) {
    int l_i = 0;
    while ( l_i < LIBXSMM_PPC64LE_VSR_NMAX ) {
      unsigned char l_avail = 1;
      for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
        int l_reg = l_i + l_j;
        if ( l_reg < LIBXSMM_PPC64LE_FPR_NMAX &&
             ( io_reg_tracker->fpr[l_reg] < LIBXSMM_PPC64LE_REG_FREE ||
               io_reg_tracker->vsr[l_reg] < LIBXSMM_PPC64LE_REG_FREE ) ) {
          l_avail = 0;
          l_i += l_j + 1;
          break;
        } else if ( io_reg_tracker->vsr[l_reg] < LIBXSMM_PPC64LE_REG_FREE ||
                    io_reg_tracker->vr[l_reg - LIBXSMM_PPC64LE_FPR_NMAX] < LIBXSMM_PPC64LE_REG_FREE ) {
          l_avail = 0;
          l_i += l_j + 1;
        }
      }
      if ( l_avail ) {
        for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
          int l_reg = l_i + l_j;
          o_reg[l_j] = l_reg;
          io_reg_tracker->vsr[l_reg] = LIBXSMM_PPC64LE_REG_USED;
          if ( l_reg < LIBXSMM_PPC64LE_FPR_NMAX ) {
            io_reg_tracker->fpr[l_reg] = LIBXSMM_PPC64LE_REG_USED;
          } else {
            io_reg_tracker->vr[l_reg - LIBXSMM_PPC64LE_FPR_NMAX] = LIBXSMM_PPC64LE_REG_USED;
          }
        }
        return;
      }
    }
  } else if ( i_reg_type == LIBXSMM_PPC64LE_ACC ) {
    int l_i = 0;
    while ( l_i < LIBXSMM_PPC64LE_ACC_NMAX ) {
      unsigned char l_avail = 1;
      for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
        int l_reg = l_i + l_j;
        if ( io_reg_tracker->acc[l_reg] < LIBXSMM_PPC64LE_REG_FREE ) {
          l_avail = 0;
          l_i += l_j + 1;
          break;
        } else {
          for ( int l_k = 0; l_k < 4; ++l_k ) {
            l_avail &= ( io_reg_tracker->fpr[l_reg*4 + l_k] >= LIBXSMM_PPC64LE_REG_FREE );
            l_avail &= ( io_reg_tracker->vsr[l_reg*4 + l_k] >= LIBXSMM_PPC64LE_REG_FREE );
          }
          if ( !l_avail ) {
            l_i += l_j + 1;
            break;
          }
        }
      }
      if ( l_avail ) {
        for ( int l_j = 0 ; l_j < i_n; ++l_j ) {
          int l_reg = l_i + l_j;
          o_reg[l_j] = l_reg;
          io_reg_tracker->acc[l_reg] = LIBXSMM_PPC64LE_REG_USED;
          for ( int l_k = 0; l_k < 4; ++l_k ) {
            io_reg_tracker->fpr[l_reg*4 + l_k] = LIBXSMM_PPC64LE_REG_USED;
            io_reg_tracker->vsr[l_reg*4 + l_k] = LIBXSMM_PPC64LE_REG_USED;
          }
        }
        return;
      }
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
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
         ( i_value == LIBXSMM_PPC64LE_REG_FREE ) ||
         ( i_value == LIBXSMM_PPC64LE_REG_ALTD )) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  if ( i_reg_type == LIBXSMM_PPC64LE_GPR ) {
    io_reg_tracker->gpr[i_reg] = i_value;
  } else if ( i_reg_type == LIBXSMM_PPC64LE_FPR ) {
    io_reg_tracker->fpr[i_reg] = i_value;
    io_reg_tracker->vsr[i_reg] = i_value;
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VR ) {
    io_reg_tracker->vr[i_reg] =  i_value;
    io_reg_tracker->vsr[i_reg + LIBXSMM_PPC64LE_FPR_NMAX] = i_value;
  } else if ( i_reg_type == LIBXSMM_PPC64LE_VSR ) {
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
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 8 - 2) );
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
  l_instr |= (unsigned int)( (0x07 & i_rs) << (31 - 6 - 4) );
  /* Set spr */
  l_instr |= (unsigned int)( (0x000003ff & i_spr) << (31 - 11 - 9) );

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
  l_instr |= (unsigned long)(0x01 & i_a) << (31 - 11 - 4);
  /* Set S10 */
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
      default: {
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
      /* X (355) form */
      case LIBXSMM_PPC64LE_FORM_X_355: {
        l_op = libxsmm_ppc64le_instr_x_form_355( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2 );
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
      /* XX2 (3) form */
      case LIBXSMM_PPC64LE_FORM_XX2_3: {
        l_op = libxsmm_ppc64le_instr_xx2_form_3( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
      } break;
      default: {
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
      /* MD form */
      case LIBXSMM_PPC64LE_FORM_MD: {
        l_op = libxsmm_ppc64le_instr_md_form( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
      } break;
      /* XX3 (6) form */
      case LIBXSMM_PPC64LE_FORM_XX3_6: {
        l_op = libxsmm_ppc64le_instr_xx3_form_6( l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
      } break;
      default: {
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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
        l_op = -1;
      }
    }

    if ( l_op != -1 ) {
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


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      unsigned int           *io_v ) {

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

  for ( unsigned int l_i = 0 ; l_i < 4 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  unsigned int l_scratch[4];
  for ( unsigned int l_i = 0; l_i < 4 ; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[2],
                           i_v[1],
                           i_v[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[3]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[1],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[3],
                           i_v[1],
                           i_v[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[3]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[2],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[1],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[3],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & o_v[3]) >> 5 );

  for ( unsigned int l_i = 0 ; l_i < 4 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  unsigned int l_scratch[4];
  for ( unsigned int l_i = 0; l_i < 4 ; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[1],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[2],
                           i_v[1],
                           i_v[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[3]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[3],
                           i_v[1],
                           i_v[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[3]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[1],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[2],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );
  for ( unsigned int l_i = 0 ; l_i < 4 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           i_v[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & i_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           i_v[1],
                           i_v[1],
                           i_v[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[3]) >> 5,
                           (0x0020 & i_v[1]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[1],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           i_v[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & i_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           i_v[1],
                           i_v[1],
                           i_v[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[3]) >> 5,
                           (0x0020 & i_v[1]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  unsigned int l_scratch[4];
  for ( unsigned int l_i = 0; l_i < 4 ; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[1],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[2],
                           i_v[1],
                           l_scratch[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[3],
                           i_v[1],
                           l_scratch[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[2],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[1],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[3],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & o_v[3]) >> 5 );

  for ( unsigned int l_i = 0 ; l_i < 4 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  unsigned int l_scratch[4];
  for ( unsigned int l_i = 0; l_i < 4 ; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[1],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[2],
                           i_v[1],
                           l_scratch[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           l_scratch[3],
                           i_v[1],
                           l_scratch[3],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[2],
                           l_scratch[1],
                           l_scratch[3],
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & l_scratch[3]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[1],
                           l_scratch[0],
                           l_scratch[2],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[2]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );

  for ( unsigned int l_i = 0 ; l_i < 4 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}



LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  unsigned int l_scratch[2];
  for ( unsigned int l_i = 0; l_i < 2 ; ++l_i ) {
    l_scratch[l_i] = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR );
  }

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & l_scratch[0]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           l_scratch[1],
                           i_v[1],
                           i_v[1],
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           l_scratch[0],
                           l_scratch[1],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[1],
                           l_scratch[0],
                           l_scratch[1],
                           (0x0020 & l_scratch[0]) >> 5,
                           (0x0020 & l_scratch[1]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );

  for ( unsigned int l_i = 0 ; l_i < 2 ; ++l_i ) {
    libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_VSR, l_scratch[l_i] );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[2],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[2]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           o_v[0],
                           i_v[1],
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[2],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[1],
                           o_v[0],
                           o_v[0],
                           2,
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[3],
                           o_v[2],
                           o_v[2],
                           2,
                           (0x0020 & o_v[2]) >> 5,
                           (0x0020 & o_v[2]) >> 5,
                           (0x0020 & o_v[3]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGLW,
                           o_v[2],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[1],
                           o_v[0],
                           o_v[0],
                           2,
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[1],
                           o_v[0],
                           o_v[0],
                           2,
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXMRGHW,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           o_v[0],
                           i_v[0],
                           i_v[0],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[1],
                           i_v[0],
                           i_v[0],
                           1,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[2],
                           i_v[0],
                           i_v[0],
                           2,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[3],
                           i_v[0],
                           i_v[0],
                           3,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[3]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           o_v[0],
                           i_v[0],
                           i_v[0],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[1],
                           i_v[0],
                           i_v[0],
                           1,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[2],
                           i_v[0],
                           i_v[0],
                           2,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[2]) >> 5 );
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {

  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           o_v[0],
                           i_v[0],
                           i_v[0],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXSLDWI,
                           o_v[1],
                           i_v[0],
                           i_v[0],
                           1,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v ) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           o_v[0],
                           i_v[0],
                           i_v[0],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
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
void libxsmm_ppc64le_instr_transpose_f64_2x2_inplace( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      unsigned int           *io_v ) {
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
void libxsmm_ppc64le_instr_transpose_f64_2x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           0x0000,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                           o_v[1],
                           i_v[0],
                           i_v[1],
                           0x0003,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_1x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                           o_v[0],
                           i_v[0],
                           i_v[1],
                           0x0000,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[1]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           o_v[0],
                           i_v[0],
                           i_v[0],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );

  libxsmm_ppc64le_instr_7( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXPERMDI,
                           o_v[1],
                           i_v[0],
                           i_v[0],
                           0x0002,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[1]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_1x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v) {
  libxsmm_ppc64le_instr_6( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_XXLOR,
                           o_v[0],
                           i_v[0],
                           i_v[0],
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & i_v[0]) >> 5,
                           (0x0020 & o_v[0]) >> 5 );
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_add_value( libxsmm_generated_code *io_generated_code,
                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                      unsigned int const      i_src,
                                      unsigned int const      i_dst,
                                      unsigned int const      i_val ) {

  if ( i_val >= (1 << 15) ) {
    if (io_generated_code->arch == LIBXSMM_PPC64LE_VSX) {
      unsigned int l_low = 0xffff & i_val;
      unsigned int l_high = i_val >> 16;
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_ADDIS,
                               i_dst,
                               i_src,
                               l_high );
      if ( l_low > 0 ) {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 i_dst,
                                 i_dst,
                                 l_low );
      }
    } else if (io_generated_code->arch == LIBXSMM_PPC64LE_MMA) {
      unsigned int l_low = 0xffff & i_val;
      unsigned int l_high = i_val >> 16;
      libxsmm_ppc64le_instr_prefix_5( io_generated_code,
                                      LIBXSMM_PPC64LE_INSTR_PADDI,
                                      0,
                                      l_high,
                                      i_dst,
                                      i_src,
                                      l_low );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_ADDI,
                             i_dst,
                             i_src,
                             i_val );
  }
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_open_stream( libxsmm_generated_code *io_generated_code,
                                        libxsmm_ppc64le_reg    *io_reg_tracker ) {
  /* From "64-Bit ELF V2 ABI Specification: Power Architecture"
   * GPR3 contains the pointer to the first arguement
   * The first arg to the gemm is a pointer to a libxsmm_gemm_param struct, which we
   * can then unpack, into the standard register locations
   */

  unsigned int gpr_offset = 0;
  unsigned int fpr_offset = (LIBXSMM_PPC64LE_GPR_NMAX - LIBXSMM_PPC64LE_GPR_IVOL)*8;
  unsigned int vsr_offset = fpr_offset + (LIBXSMM_PPC64LE_FPR_NMAX - LIBXSMM_PPC64LE_FPR_IVOL)*8;

  /* Get the LR and store it in the previous stackframe */
  libxsmm_ppc64le_instr_2( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MFSPR,
                           0,
                           LIBXSMM_PPC64LE_SPR_LR );
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           0,
                           LIBXSMM_PPC64LE_GPR_SP,
                           4 );

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
                             (gpr - LIBXSMM_PPC64LE_GPR_IVOL)*2 + gpr_offset);
  }

  /* save non-volatile floating point registers */
  for( unsigned int fpr = LIBXSMM_PPC64LE_FPR_IVOL; fpr < LIBXSMM_PPC64LE_FPR_NMAX; ++fpr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_STFD,
                             fpr,
                             LIBXSMM_PPC64LE_GPR_SP,
                             (fpr - LIBXSMM_PPC64LE_FPR_IVOL)*2 + fpr_offset );
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
void libxsmm_ppc64le_instr_epilogue_stack( libxsmm_generated_code *io_generated_code,
                                           libxsmm_ppc64le_reg    *io_reg_tracker ) {
  unsigned int l_frame_fixed = 32;
  libxsmm_generated_code l_generated_code;

  l_generated_code.generated_code = malloc(sizeof(char)*io_generated_code->buffer_size);
  l_generated_code.arch = io_generated_code->arch;
  l_generated_code.buffer_size = io_generated_code->buffer_size;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = io_generated_code->code_type;
  l_generated_code.data_size = 0;
  l_generated_code.last_error = 0;


  /* Calculate the stack size */
  unsigned int l_stack_size = l_frame_fixed;
  for( unsigned int l_reg = LIBXSMM_PPC64LE_GPR_IVOL; l_reg < LIBXSMM_PPC64LE_GPR_NMAX; ++l_reg ) {
    l_stack_size += ( io_reg_tracker->gpr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) ? 8 : 0;
  }
  for( unsigned int l_reg = LIBXSMM_PPC64LE_FPR_IVOL; l_reg < LIBXSMM_PPC64LE_FPR_NMAX; ++l_reg ) {
    l_stack_size += ( io_reg_tracker->fpr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) ? 8 : 0;
  }
  for( unsigned int l_reg = LIBXSMM_PPC64LE_VR_IVOL; l_reg < LIBXSMM_PPC64LE_VR_NMAX; ++l_reg ) {
    l_stack_size += ( io_reg_tracker->vr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) ? 16 : 0;

  }
  l_stack_size += (l_stack_size % 16);

  /* Decrease stack pointer */
  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_GPR_SP,
                           -l_stack_size );

  /* Get the LR and store it in the previous stackframe */
  libxsmm_ppc64le_instr_2( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MFSPR,
                           0,
                           LIBXSMM_PPC64LE_SPR_LR );
  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           0,
                           LIBXSMM_PPC64LE_GPR_SP,
                           4 );

  /* Save non-volatile general purpose registers */
  unsigned int l_woffset_st = l_frame_fixed/4;
  for( unsigned int l_reg = LIBXSMM_PPC64LE_GPR_IVOL; l_reg < LIBXSMM_PPC64LE_GPR_NMAX; ++l_reg ) {
    if ( io_reg_tracker->gpr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) {
      libxsmm_ppc64le_instr_3( &l_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STD,
                               l_reg,
                               LIBXSMM_PPC64LE_GPR_SP,
                               l_woffset_st );
      l_woffset_st += 2;
    }
  }

  /* Save non-volatile floating point registers */
  for( unsigned int l_reg = LIBXSMM_PPC64LE_FPR_IVOL; l_reg < LIBXSMM_PPC64LE_FPR_NMAX; ++l_reg ) {
    if ( io_reg_tracker->fpr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) {
      libxsmm_ppc64le_instr_3( &l_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STFD,
                               l_reg,
                               LIBXSMM_PPC64LE_GPR_SP,
                               l_woffset_st );
      l_woffset_st += 2;
    }
  }

  /* Align vector stack */
  l_woffset_st += (l_woffset_st % 4);

  /* Save non-volatile vector registers */
  unsigned int index = LIBXSMM_PPC64LE_GPR_R10;
  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           index,
                           LIBXSMM_PPC64LE_GPR_SP,
                           l_woffset_st*4 );

  char l_first = 1;
  for( unsigned int l_reg = LIBXSMM_PPC64LE_VR_IVOL; l_reg < LIBXSMM_PPC64LE_VR_NMAX; ++l_reg ) {
    if ( io_reg_tracker->vr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) {
      if ( l_first == 1 ) {
        l_first = 0;
      } else {
        libxsmm_ppc64le_instr_3( &l_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 index,
                                 index,
                                 16 );
      }
      libxsmm_ppc64le_instr_3( &l_generated_code,
                               LIBXSMM_PPC64LE_INSTR_STVX,
                               l_reg,
                               0,
                               index );
    }
  }
  libxsmm_ppc64le_free_reg( &l_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, index);

  /* Set up input args */
  unsigned int struct_ptr = LIBXSMM_PPC64LE_GPR_R6;
  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_OR,
                           LIBXSMM_PPC64LE_GPR_R3,
                           struct_ptr,
                           LIBXSMM_PPC64LE_GPR_R3 );

  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R3,
                           struct_ptr,
                           8 );
  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R4,
                           struct_ptr,
                           16 );
  libxsmm_ppc64le_instr_3( &l_generated_code,
                           LIBXSMM_PPC64LE_INSTR_LD,
                           LIBXSMM_PPC64LE_GPR_R5,
                           struct_ptr,
                           24 );

  /* Restore non-volatile general purpose registers */
  unsigned int l_woffset_ld = 0;
  for( unsigned int l_reg = LIBXSMM_PPC64LE_GPR_IVOL; l_reg < LIBXSMM_PPC64LE_GPR_NMAX; ++l_reg ) {
    if ( io_reg_tracker->gpr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LD,
                               l_reg,
                               LIBXSMM_PPC64LE_GPR_SP,
                               l_woffset_ld );
      l_woffset_ld += 2;
    }
  }

  /* Restore non-volatile floating point registers */
  for( unsigned int l_reg = LIBXSMM_PPC64LE_FPR_IVOL; l_reg < LIBXSMM_PPC64LE_FPR_NMAX; ++l_reg ) {
    if ( io_reg_tracker->fpr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) {
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LFD,
                               l_reg,
                               LIBXSMM_PPC64LE_GPR_SP,
                               l_woffset_ld );
      l_woffset_ld += 2;
    }
  }

  /* Align vector stack */
  l_woffset_ld += (l_woffset_ld % 4);

  /* Restore non-volatile vector registers */
  /* As non-volatile GPR have already been restored, use R10 */
  index = LIBXSMM_PPC64LE_GPR_R10;
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           index,
                           LIBXSMM_PPC64LE_GPR_SP,
                           l_woffset_ld*4 );

  l_first = 1;
  for( unsigned int l_reg = LIBXSMM_PPC64LE_VR_IVOL; l_reg < LIBXSMM_PPC64LE_VR_NMAX; ++l_reg ) {
    if ( io_reg_tracker->vr[l_reg] == LIBXSMM_PPC64LE_REG_ALTD ) {
      if ( l_first == 1 ) {
        l_first = 0;
      } else {
        libxsmm_ppc64le_instr_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 index,
                                 index,
                                 16 );
      }
      libxsmm_ppc64le_instr_3( io_generated_code,
                               LIBXSMM_PPC64LE_INSTR_LVX,
                               l_reg,
                               0,
                               index );
    }
  }

  /* Increase stack pointer */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_ADDI,
                           LIBXSMM_PPC64LE_GPR_SP,
                           LIBXSMM_PPC64LE_GPR_SP,
                           l_stack_size );

  /* Get the LR from previous stackframe and restore */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           0,
                           LIBXSMM_PPC64LE_GPR_SP,
                           4 );
  libxsmm_ppc64le_instr_2( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MTSPR,
                           0,
                           LIBXSMM_PPC64LE_SPR_LR );

  /* Return statement */
  libxsmm_ppc64le_instr_blr( io_generated_code );

  /* Build one code from the two fragments */
  char *l_old_code = (char *)io_generated_code->generated_code;
  char *l_new_code = (char *)l_generated_code.generated_code;
  unsigned int l_new_size = l_generated_code.code_size + io_generated_code->code_size;
  unsigned int l_old_size = io_generated_code->code_size;
  for ( int i = 0; i < io_generated_code->code_size; ++i ) {
    l_old_code[l_new_size - i - 1] = l_old_code[l_old_size - i - 1];
  }
  for ( int i = 0; i < l_generated_code.code_size; ++i ) {
    l_old_code[i] = l_new_code[i];
  }
  io_generated_code->code_size = l_new_size;

  /* Free the local code */
  free(l_generated_code.generated_code);
}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_colapse_stack( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker ) {
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
                             (gpr - LIBXSMM_PPC64LE_GPR_IVOL)*2 + gpr_offset );
  }

  /* restore non-volatile floating point registers */
  for( unsigned int fpr = LIBXSMM_PPC64LE_FPR_IVOL; fpr < LIBXSMM_PPC64LE_FPR_NMAX; ++fpr ) {
    libxsmm_ppc64le_instr_3( io_generated_code,
                             LIBXSMM_PPC64LE_INSTR_LFD,
                             fpr,
                             LIBXSMM_PPC64LE_GPR_SP,
                             (fpr - LIBXSMM_PPC64LE_FPR_IVOL)*2 + fpr_offset );
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

  /* Get the LR from previous stackframe and restore */
  libxsmm_ppc64le_instr_3( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_STD,
                           0,
                           LIBXSMM_PPC64LE_GPR_SP,
                           4 );
  libxsmm_ppc64le_instr_2( io_generated_code,
                           LIBXSMM_PPC64LE_INSTR_MTSPR,
                           0,
                           LIBXSMM_PPC64LE_SPR_LR );

  /* return statement */
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
void libxsmm_ppc64le_instr_prefetch_stream_open( libxsmm_gerenrated_code *io_genreated_code,
                                                 libxsmm_ppc64le_reg     *io_reg_tracker,
                                                 char const               i_stream,
                                                 unsigned int const       i_a,
                                                 unsigned int const       i_lda,
                                                 unsigned int const       i_len ) {

  unsigned int l_stream = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  unsigned int l_cfg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_stream, 0, (0x0f & i_stream) );

  /* TH=01000
     0:56  EATRUNC
     57    Direction
     58    Unlimted
     59    RESERVED
     60:63 Stream
  */
  libxsmm_ppc64le_instr_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_RLDICR, l_cfg, l_cfg, 0, 56, 0, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0b1000, l_cfg, l_stream );

  /* TH=01011
     0:31  RESERVED
     32:49 Stride
     50    RESERVED
     51:55 Offset
     56:59 RESERVED
     60:63 Stream
  */
  unsigned int l_lda = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_lda, 0, i_lda / 2 );
  if ( i_lda % 2 == 1 ) {
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_lda, l_lda, 0x8000 );
  }
  libxsmm_ppc64le_instr_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_RLWINM, i_a, l_cfg, 6, 19, 23, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADD, l_cfg, l_cfg, l_lda );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0b1011, l_cfg, l_stream );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_lda );

  /* TH=01010
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
    libxsmm_ppc64le_instr_6( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDIS, l_cfg, l_cfg, 1 );
  }
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_ADDI, l_cfg, l_cfg, 0x40 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0b1010, l_cfg, l_stream );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_cfg );
  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_stream );

}


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefetch_stream_close( libxsmm_gerenrated_code *io_genreated_code,
                                                  libxsmm_ppc64le_reg     *io_reg_tracker,
                                                  char const               i_stream ) {
  unsigned int l_cfg = libxsmm_ppc64le_get_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR );

  /* TH=01010
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
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_DCBT, 0b1010, 0, l_cfg );

  libxsmm_ppc64le_free_reg( io_generated_code, io_reg_tracker, LIBXSMM_PPC64LE_GPR, l_cfg );
}
