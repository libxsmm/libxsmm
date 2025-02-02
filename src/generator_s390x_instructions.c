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

#include "generator_s390x_instructions.h"

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
char libxsmm_s390x_instr_rre_ab_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
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
char libxsmm_s390x_instr_rre_c_e_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
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

