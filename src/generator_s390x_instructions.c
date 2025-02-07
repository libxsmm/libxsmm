/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_s390x_instructions.h"

/* Based on "z/Architecture: Principles of Operation".
   Below is a table showing Z model corresponding revision number:
   z14     SA22-7832-11 (12th edition)
   z15     SA22-7832-12 (13th edition)
   z16     SA22-7832-13 (14th edition)

   Also based on "z/Architecture: Reference Summary"
   Below is a table showing Z model corresponding revision number:
   z14     SA22-7871-9  (12th edition)
   z15     SA22-7871-10 (12th edition)
   z16     SA22-7871-11 (12th edition)
*/


LIBXSMM_API_INTERN
void libxsmm_s390x_instr_open_stack( libxsmm_generated_code *io_generated_code ) {
  /* Based on "ELF Application Binary Interface s390x Supplement: v1.6.1"
   */
  return;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_colapse_stack( libxsmm_generated_code *io_generated_code ) {
  /* Based on "ELF Application Binary Interface s390x Supplement: v1.6.1"
   */
  return;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_append( libxsmm_generated_code *io_generated_code,
                                 unsigned char          *i_op,
                                 char                    i_nbytes ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *l_code = (unsigned char*) io_generated_code->generated_code;
    unsigned int l_code_head = io_generated_code->code_size;
    for (char i = 0; i < i_nbytes ; ++i ) {
      l_code[l_code_head + i] = i_op[(int)i];
    }
    io_generated_code->code_size += i_nbytes;
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_0( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_e_form( (unsigned int)l_instr, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_1( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_I_FORM: {
        l_nbytes = libxsmm_s390x_instr_i_form( (unsigned int)l_instr, (unsigned char)i_0, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_2( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_IE_FORM: {
        l_nbytes = libxsmm_s390x_instr_ie_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RI_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_ri_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RI_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_ri_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RI_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_ri_c_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIL_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_ril_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIL_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_ril_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIL_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_ril_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RR_FORM: {
        l_nbytes = libxsmm_s390x_instr_rr_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRE_FORM: {
        l_nbytes = libxsmm_s390x_instr_rre_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_S_FORM: {
        l_nbytes = libxsmm_s390x_instr_s_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_G_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_g_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_3( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_MII_FORM: {
        l_nbytes = libxsmm_s390x_instr_mii_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_G_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_g_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRD_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrd_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSI_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsi_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSL_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsl_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSL_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsl_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SI_FORM: {
        l_nbytes = libxsmm_s390x_instr_si_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SIL_FORM: {
        l_nbytes = libxsmm_s390x_instr_sil_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned int)i_2, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_4( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_RIE_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_c_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_d_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_e_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RS_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rs_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RS_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rs_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RX_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rx_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RX_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rx_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SIY_FORM: {
        l_nbytes = libxsmm_s390x_instr_siy_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SMI_FORM: {
        l_nbytes = libxsmm_s390x_instr_smi_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SSE_FORM: {
        l_nbytes = libxsmm_s390x_instr_sse_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_H_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_h_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_H_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_h_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_K_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_k_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_5( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_RIE_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIS_FORM: {
        l_nbytes = libxsmm_s390x_instr_ris_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRS_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrs_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSY_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsy_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSY_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsy_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXE_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxe_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXF_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxf_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXY_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxy_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXY_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxy_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned int)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned int)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SSF_FORM: {
        l_nbytes = libxsmm_s390x_instr_ssf_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned int)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_I_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_i_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_I_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_i_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_J_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_j_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VSI_FORM: {
        l_nbytes = libxsmm_s390x_instr_vsi_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_6( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4,
                            unsigned int            i_5 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_SS_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_G_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_g_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRV_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrv_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRX_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrx_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_7( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4,
                            unsigned int            i_5,
                            unsigned int            i_6 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = (i_instr & ~LIBXSMM_S390X_FMASK);
    unsigned long l_instr = (i_instr & LIBXSMM_S390X_FMASK);

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_VRR_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }

    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


/* All code below here is auto-generated */

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_e_form(unsigned int instr, unsigned char *output) {
unsigned int opcode = (0xffff & instr);
output[0] = 0xff & (opcode >> 8);
output[1] = 0xff & (opcode >> 0);
return 2;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_i_form(unsigned int instr, unsigned char i, unsigned char *output) {
unsigned int opcode = (0xffff & instr);
opcode += (unsigned int)(0xff & i);
output[0] = 0xff & (opcode >> 8);
output[1] = 0xff & (opcode >> 0);
return 2;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ie_form(unsigned int instr, unsigned char i1,unsigned char i2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & i1) << 4;
opcode += (unsigned int)(0xf & i2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_mii_form(unsigned long instr, unsigned char m1,unsigned int ri2,unsigned int ri3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xfff & ri2) << 24;
opcode += (unsigned long)(0xffffff & ri3);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_a_form(unsigned int instr, unsigned char r1,unsigned int i2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xffff & i2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_b_form(unsigned int instr, unsigned char r1,unsigned int ri1, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xffff & ri1);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_c_form(unsigned int instr, unsigned char m1,unsigned int ri2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m1) << 20;
opcode += (unsigned int)(0xffff & ri2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_a_form(unsigned long instr, unsigned char r1,unsigned int i2,unsigned char m3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_b_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned int ri4,unsigned char m3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xffff & ri4) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_c_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned int ri4,unsigned char i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xffff & ri4) << 16;
opcode += (unsigned long)(0xff & i2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_d_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xffff & i2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_e_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xffff & ri2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_f_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned char i3,unsigned char i4,unsigned char i5, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xff & i3) << 24;
opcode += (unsigned long)(0xff & i4) << 16;
opcode += (unsigned long)(0xff & i5) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_g_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xffff & i2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_a_form(unsigned long instr, unsigned char r1,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xffffffff & i2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_b_form(unsigned long instr, unsigned char r1,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xffffffff & ri2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_c_form(unsigned long instr, unsigned char m1,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xffffffff & ri2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ris_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned char b4,unsigned int d4,unsigned char i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xf & b4) << 28;
opcode += (unsigned long)(0xfff & d4) << 16;
opcode += (unsigned long)(0xff & i2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rr_form(unsigned int instr, unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffff & instr);
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 8);
output[1] = 0xff & (opcode >> 0);
return 2;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrd_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 12;
opcode += (unsigned int)(0xf & r3) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rre_form(unsigned int instr, unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_a_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_b_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_c_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_d_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_e_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrs_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned char b4,unsigned int d4,unsigned char m4, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xf & b4) << 28;
opcode += (unsigned long)(0xfff & d4) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rs_a_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & r3) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rs_b_form(unsigned int instr, unsigned char r1,unsigned char m3,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & m3) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsi_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned int ri2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & r3) << 16;
opcode += (unsigned int)(0xffff & ri2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsl_a_form(unsigned long instr, unsigned char l1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & l1) << 36;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsl_b_form(unsigned long instr, unsigned char l1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & l1) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsy_a_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsy_b_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rx_a_form(unsigned int instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & x2) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rx_b_form(unsigned int instr, unsigned char m1,unsigned char x2,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m1) << 20;
opcode += (unsigned int)(0xf & x2) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxe_form(unsigned long instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char m3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxf_form(unsigned long instr, unsigned char r3,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char r1, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r3) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & r1) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxy_a_form(unsigned long instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxy_b_form(unsigned long instr, unsigned char m1,unsigned char x2,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_s_form(unsigned int instr, unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_si_form(unsigned int instr, unsigned char i2,unsigned char b1,unsigned int d1, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xff & i2) << 16;
opcode += (unsigned int)(0xf & b1) << 12;
opcode += (unsigned int)(0xfff & d1);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_sil_form(unsigned long instr, unsigned char b1,unsigned int d1,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xffff & i2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_siy_form(unsigned long instr, unsigned char i2,unsigned char b1,unsigned int dl1,unsigned char dh1, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & i2) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & dl1) << 16;
opcode += (unsigned long)(0xff & dh1) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_smi_form(unsigned long instr, unsigned char m1,unsigned char b3,unsigned int d3,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xf & b3) << 28;
opcode += (unsigned long)(0xfff & d3) << 16;
opcode += (unsigned long)(0xffff & ri2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_a_form(unsigned long instr, unsigned char l,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & l) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_b_form(unsigned long instr, unsigned char l1,unsigned char l2,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & l1) << 36;
opcode += (unsigned long)(0xf & l2) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_c_form(unsigned long instr, unsigned char l1,unsigned char i3,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & l1) << 36;
opcode += (unsigned long)(0xf & i3) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_d_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_e_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int d2,unsigned char b4,unsigned int d4, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & b4) << 12;
opcode += (unsigned long)(0xfff & d4);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_f_form(unsigned long instr, unsigned char l2,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & l2) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_sse_form(unsigned long instr, unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ssf_form(unsigned long instr, unsigned char r1,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_a_form(unsigned long instr, unsigned char v1,unsigned int i2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_b_form(unsigned long instr, unsigned char v1,unsigned char i2,unsigned char i3,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xff & i2) << 24;
opcode += (unsigned long)(0xff & i3) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_c_form(unsigned long instr, unsigned char v1,unsigned char v3,unsigned int i2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v3) << 32;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_d_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char i4,unsigned char m5,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xff & i4) << 16;
opcode += (unsigned long)(0xf & m5) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_e_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned int i3,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xfff & i3) << 20;
opcode += (unsigned long)(0xf & m5) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_f_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char i4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xff & i4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_g_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char i4,unsigned char m5,unsigned char i3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xff & i4) << 24;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xff & i3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_h_form(unsigned long instr, unsigned char v1,unsigned int i2,unsigned char i3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & i3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_i_form(unsigned long instr, unsigned char v1,unsigned char r2,unsigned char m4,unsigned char i3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xf & m4) << 20;
opcode += (unsigned long)(0xff & i3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_a_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m5,unsigned char m4,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xf & m4) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_b_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_c_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m6,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m6) << 20;
opcode += (unsigned long)(0xf & m5) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_d_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char m6,unsigned char v4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m5) << 24;
opcode += (unsigned long)(0xf & m6) << 20;
opcode += (unsigned long)(0xf & v4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_e_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m6,unsigned char m5,unsigned char v4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m6) << 24;
opcode += (unsigned long)(0xf & m5) << 16;
opcode += (unsigned long)(0xf & v4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_f_form(unsigned long instr, unsigned char v1,unsigned char r2,unsigned char r3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xf & r3) << 28;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_g_form(unsigned long instr, unsigned char v1,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 32;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_h_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 32;
opcode += (unsigned long)(0xf & v2) << 28;
opcode += (unsigned long)(0xf & m3) << 20;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_i_form(unsigned long instr, unsigned char r1,unsigned char v2,unsigned char m3,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & m3) << 20;
opcode += (unsigned long)(0xf & m4) << 16;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_j_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m4) << 20;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_k_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & m3) << 20;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_a_form(unsigned long instr, unsigned char v1,unsigned char v3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_b_form(unsigned long instr, unsigned char v1,unsigned char r3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_c_form(unsigned long instr, unsigned char r1,unsigned char v3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & v3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_d_form(unsigned long instr, unsigned char r3,unsigned char b2,unsigned int d2,unsigned char v1,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & v1) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrv_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char b2,unsigned int d2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrx_form(unsigned long instr, unsigned char v1,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vsi_form(unsigned long instr, unsigned char i3,unsigned char b2,unsigned int d2,unsigned char v1,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & i3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & v1) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}
