/******************************************************************************
* Copyright (c) 2024, IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_S390X_INSTRUCTIONS_H
#define GENERATOR_S390X_INSTRUCTIONS_H

#include "generator_common.h"

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_e_form(unsigned int instr, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_i_form(unsigned int instr, unsigned char i, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ie_form(unsigned int instr, unsigned char i1,unsigned char i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_mii_form(unsigned long instr, unsigned char m1,unsigned int ri2,unsigned int ri3, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_a_form(unsigned int instr, unsigned char r1,unsigned int i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_b_form(unsigned int instr, unsigned char r1,unsigned int ri1, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_c_form(unsigned int instr, unsigned char m1,unsigned int ri2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_a_form(unsigned long instr, unsigned char r1,unsigned int i2,unsigned char m3, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_b_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned int ri4,unsigned char m3, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_c_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned int ri4,unsigned char i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_d_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned int i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_e_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned int ri2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_f_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned char i3,unsigned char i4,unsigned char i5, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_g_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned int i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_a_form(unsigned long instr, unsigned char r1,unsigned int i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_b_form(unsigned long instr, unsigned char r1,unsigned int ri2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_c_form(unsigned long instr, unsigned char m1,unsigned int ri2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ris_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned char b4,unsigned int d4,unsigned char i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rr_form(unsigned int instr, unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrd_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rre_form(unsigned int instr, unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rre_ab_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rre_c_e_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrs_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned char b4,unsigned int d4,unsigned char m4, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rs_a_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rs_b_form(unsigned int instr, unsigned char r1,unsigned char m3,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsi_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned int ri2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsl_a_form(unsigned long instr, unsigned char l1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsl_b_form(unsigned long instr, unsigned char l1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsy_a_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsy_b_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rx_a_form(unsigned int instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rx_b_form(unsigned int instr, unsigned char m1,unsigned char x2,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxe_form(unsigned long instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char m3, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxf_form(unsigned long instr, unsigned char r3,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char r1, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxy_a_form(unsigned long instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxy_b_form(unsigned long instr, unsigned char m1,unsigned char x2,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_s_form(unsigned int instr, unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_si_form(unsigned int instr, unsigned char i2,unsigned char b1,unsigned int d1, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_sil_form(unsigned long instr, unsigned char b1,unsigned int d1,unsigned int i2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_siy_form(unsigned long instr, unsigned char i2,unsigned char b1,unsigned int dl1,unsigned char dh1, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_smi_form(unsigned long instr, unsigned char m1,unsigned char b3,unsigned int d3,unsigned int ri2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_a_form(unsigned long instr, unsigned char l,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_b_form(unsigned long instr, unsigned char l1,unsigned char l2,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_c_form(unsigned long instr, unsigned char l1,unsigned char i3,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_d_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_e_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int d2,unsigned char b4,unsigned int d4, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_f_form(unsigned long instr, unsigned char l2,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_sse_form(unsigned long instr, unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ssf_form(unsigned long instr, unsigned char r1,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_a_form(unsigned long instr, unsigned char v1,unsigned int i2,unsigned char m3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_b_form(unsigned long instr, unsigned char v1,unsigned char i2,unsigned char i3,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_c_form(unsigned long instr, unsigned char v1,unsigned char v3,unsigned int i2,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_d_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char i4,unsigned char m5,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_e_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned int i3,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_f_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char i4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_g_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char i4,unsigned char m5,unsigned char i3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_h_form(unsigned long instr, unsigned char v1,unsigned int i2,unsigned char i3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_i_form(unsigned long instr, unsigned char v1,unsigned char r2,unsigned char m4,unsigned char i3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_a_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m5,unsigned char m4,unsigned char m3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_b_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_c_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m6,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_d_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char m6,unsigned char v4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_e_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m6,unsigned char m5,unsigned char v4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_f_form(unsigned long instr, unsigned char v1,unsigned char r2,unsigned char r3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_g_form(unsigned long instr, unsigned char v1,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_h_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_i_form(unsigned long instr, unsigned char r1,unsigned char v2,unsigned char m3,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_j_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_k_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_a_form(unsigned long instr, unsigned char v1,unsigned char v3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_b_form(unsigned long instr, unsigned char v1,unsigned char r3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_c_form(unsigned long instr, unsigned char r1,unsigned char v3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_d_form(unsigned long instr, unsigned char r3,unsigned char b2,unsigned int d2,unsigned char v1,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrv_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char b2,unsigned int d2,unsigned char m3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrx_form(unsigned long instr, unsigned char v1,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char m3,unsigned char rxb, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vsi_form(unsigned long instr, unsigned char i3,unsigned char b2,unsigned int d2,unsigned char v1,unsigned char rxb, unsigned char *output);


#endif