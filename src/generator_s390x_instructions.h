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

#ifndef GENERATOR_S390X_INSTRUCTIONS_H
#define GENERATOR_S390X_INSTRUCTIONS_H

#include "generator_common.h"

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


#define LIBXSMM_S390X_GPR_RA 14
#define LIBXSMM_S390X_GPR_SP 15
#define LIBXSMM_S390X_GPR_ARG0 2
#define LIBXSMM_S390X_GPR_ARG1 3
#define LIBXSMM_S390X_GPR_ARG2 4
#define LIBXSMM_S390X_GPR_ARG3 5

#define LIBXSMM_S390X_F16 0x01
#define LIBXSMM_S390X_F32 0x02
#define LIBXSMM_S390X_F64 0x03
#define LIBXSMM_S390X_F128 0x04

#define LIBXSMM_S390X_FP_TINY LIBXSMM_S390X_F16
#define LIBXSMM_S390X_FP_SHORT LIBXSMM_S390X_F32
#define LIBXSMM_S390X_FP_LONG LIBXSMM_S390X_F64
#define LIBXSMM_S390X_FP_EXT LIBXSMM_S390X_F128

#define LIBXSMM_S390X_INSTR_RETURN 0x07fe
#define LIBXSMM_S390X_INSTR_NOP 0x47000000


LIBXSMM_API_INTERN
void libxsmm_s390x_instr_open_stack( libxsmm_generated_code *io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_colapse_stack( libxsmm_generated_code *io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_nop( libxsmm_generated_code *io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_return( libxsmm_generated_code *io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_append( libxsmm_generated_code *io_generated_code,
                                 unsigned char          *i_op,
                                 char                    i_nbytes );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_0( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_1( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0 );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_2( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1 );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_3( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2 );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_4( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3 );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_5( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4 );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_6( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4,
                            unsigned int            i_5 );

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_7( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4,
                            unsigned int            i_5,
                            unsigned int            i_6 );


/* All code below here is auto-generated */
#define LIBXSMM_S390X_FMASK 0xffff000000000000UL

#define LIBXSMM_S390X_FORM_E_FORM 0x0000000000000000UL /* 0 */
#define LIBXSMM_S390X_FORM_I_FORM 0x0001000000000000UL /* 1 */
#define LIBXSMM_S390X_FORM_IE_FORM 0x0002000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_MII_FORM 0x0003000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RI_A_FORM 0x0004000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RI_B_FORM 0x0005000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RI_C_FORM 0x0006000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RIE_A_FORM 0x0007000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RIE_B_FORM 0x0008000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RIE_C_FORM 0x0009000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RIE_D_FORM 0x000a000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RIE_E_FORM 0x000b000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RIE_F_FORM 0x000c000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RIE_G_FORM 0x000d000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RIL_A_FORM 0x000e000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RIL_B_FORM 0x000f000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RIL_C_FORM 0x0010000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RIS_FORM 0x0011000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RR_FORM 0x0012000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RRD_FORM 0x0013000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RRE_FORM 0x0014000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_RRF_A_FORM 0x0015000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RRF_B_FORM 0x0016000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RRF_C_FORM 0x0017000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RRF_D_FORM 0x0018000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RRF_E_FORM 0x0019000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RRS_FORM 0x001a000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RS_A_FORM 0x001b000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RS_B_FORM 0x001c000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RSI_FORM 0x001d000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RSL_A_FORM 0x001e000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RSL_B_FORM 0x001f000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_RSY_A_FORM 0x0020000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RSY_B_FORM 0x0021000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RX_A_FORM 0x0022000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RX_B_FORM 0x0023000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_RXE_FORM 0x0024000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RXF_FORM 0x0025000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RXY_A_FORM 0x0026000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_RXY_B_FORM 0x0027000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_S_FORM 0x0028000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_SI_FORM 0x0029000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_SIL_FORM 0x002a000000000000UL /* 3 */
#define LIBXSMM_S390X_FORM_SIY_FORM 0x002b000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_SMI_FORM 0x002c000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_SS_A_FORM 0x002d000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_SS_B_FORM 0x002e000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_SS_C_FORM 0x002f000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_SS_D_FORM 0x0030000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_SS_E_FORM 0x0031000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_SS_F_FORM 0x0032000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_SSE_FORM 0x0033000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_SSF_FORM 0x0034000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRI_A_FORM 0x0035000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_VRI_B_FORM 0x0036000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRI_C_FORM 0x0037000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRI_D_FORM 0x0038000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRI_E_FORM 0x0039000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRI_F_FORM 0x003a000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRI_G_FORM 0x003b000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRI_H_FORM 0x003c000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_VRI_I_FORM 0x003d000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRR_A_FORM 0x003e000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRR_B_FORM 0x003f000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRR_C_FORM 0x0040000000000000UL /* 7 */
#define LIBXSMM_S390X_FORM_VRR_D_FORM 0x0041000000000000UL /* 7 */
#define LIBXSMM_S390X_FORM_VRR_E_FORM 0x0042000000000000UL /* 7 */
#define LIBXSMM_S390X_FORM_VRR_F_FORM 0x0043000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_VRR_G_FORM 0x0044000000000000UL /* 2 */
#define LIBXSMM_S390X_FORM_VRR_H_FORM 0x0045000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_VRR_I_FORM 0x0046000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRR_J_FORM 0x0047000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRR_K_FORM 0x0048000000000000UL /* 4 */
#define LIBXSMM_S390X_FORM_VRS_A_FORM 0x0049000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRS_B_FORM 0x004a000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRS_C_FORM 0x004b000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRS_D_FORM 0x004c000000000000UL /* 5 */
#define LIBXSMM_S390X_FORM_VRV_FORM 0x004d000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VRX_FORM 0x004e000000000000UL /* 6 */
#define LIBXSMM_S390X_FORM_VSI_FORM 0x004f000000000000UL /* 5 */

#define LIBXSMM_S390X_INSTR_AGH 0x0026e30000000038UL /* Add Halfword (64<-16), form: RXY-a */
#define LIBXSMM_S390X_INSTR_AGHI 0x00040000a70b0000UL /* Add Halfword Immediate (64<-16), form: RI-a */
#define LIBXSMM_S390X_INSTR_AGHIK 0x000aec00000000d9UL /* Add Immediate (64<-16), form: RIE-d */
#define LIBXSMM_S390X_INSTR_ALG 0x0026e3000000000aUL /* Add Logical (64), form: RXY-a */
#define LIBXSMM_S390X_INSTR_ALGF 0x0026e3000000001aUL /* Add Logical (64<-32), form: RXY-a */
#define LIBXSMM_S390X_INSTR_ALGFI 0x000ec20a00000000UL /* Add Logical Immediate (64<-32), form: RIL-a */
#define LIBXSMM_S390X_INSTR_ALGFR 0x00140000b91a0000UL /* Add Logical (64<-32), form: RRE */
#define LIBXSMM_S390X_INSTR_ALGHSIK 0x000aec00000000dbUL /* Add Logical with Signed Immediate (64<-16), form: RIE-d */
#define LIBXSMM_S390X_INSTR_ALGR 0x00140000b90a0000UL /* Add Logical (64), form: RRE */
#define LIBXSMM_S390X_INSTR_ALGRK 0x00150000b9ea0000UL /* Add Logical (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_M 0x002200005c000000UL /* Multiply (64<-32), form: RX-a */
#define LIBXSMM_S390X_INSTR_MAD 0x0025ed000000003eUL /* Multiply and Add (LH), form: RXF */
#define LIBXSMM_S390X_INSTR_MADB 0x0025ed000000001eUL /* Multiply and Add (LB), form: RXF */
#define LIBXSMM_S390X_INSTR_MADBR 0x00130000b31e0000UL /* Multiply and Add (LB), form: RRD */
#define LIBXSMM_S390X_INSTR_MADR 0x00130000b33e0000UL /* Multiply and Add (LH), form: RRD */
#define LIBXSMM_S390X_INSTR_MAE 0x0025ed000000002eUL /* Multiply and Add (SH), form: RXF */
#define LIBXSMM_S390X_INSTR_MAEB 0x0025ed000000000eUL /* Multiply and Add (SB), form: RXF */
#define LIBXSMM_S390X_INSTR_MAEBR 0x00130000b30e0000UL /* Multiply and Add (SB), form: RRD */
#define LIBXSMM_S390X_INSTR_MAER 0x00130000b32e0000UL /* Multiply and Add (SH), form: RRD */
#define LIBXSMM_S390X_INSTR_MAY 0x0025ed000000003aUL /* Multiply and Add Unnormalized (EH<-LH), form: RXF */
#define LIBXSMM_S390X_INSTR_MAYH 0x0025ed000000003cUL /* Multiply and Add Unnormalized (EHH<-LH), form: RXF */
#define LIBXSMM_S390X_INSTR_MAYHR 0x00130000b33c0000UL /* Multiply and Add Unnormalized (EHH<-LH), form: RRD */
#define LIBXSMM_S390X_INSTR_MAYL 0x0025ed0000000038UL /* Multiply and Add Unnormalized (EHL<-LH), form: RXF */
#define LIBXSMM_S390X_INSTR_MAYLR 0x00130000b3380000UL /* Multiply and Add Unnormalized (EHL<-LH), form: RRD */
#define LIBXSMM_S390X_INSTR_MAYR 0x00130000b33a0000UL /* Multiply and Add Unnormalized (EH<-LH), form: RRD */
#define LIBXSMM_S390X_INSTR_MVC 0x002dd20000000000UL /* Move (character), form: SS-a */
#define LIBXSMM_S390X_INSTR_MVCDK 0x0033e50f00000000UL /* Move with Destination Key, form: SSE */
#define LIBXSMM_S390X_INSTR_MVCIN 0x002de80000000000UL /* Move Inverse, form: SS-a */
#define LIBXSMM_S390X_INSTR_MVCL 0x0012000000000e00UL /* Move Long, form: RR */
#define LIBXSMM_S390X_INSTR_MVCLE 0x001b0000a8000000UL /* Move Long Extended, form: RS-a */
#define LIBXSMM_S390X_INSTR_MVCP 0x0030da0000000000UL /* Move to Primary, form: SS-d */
#define LIBXSMM_S390X_INSTR_MVCRL 0x0033e50a00000000UL /* Move Right to Left, form: SSE */
#define LIBXSMM_S390X_INSTR_MVGHI 0x002ae54800000000UL /* Move (64<-16), form: SIL */
#define LIBXSMM_S390X_INSTR_MVHHI 0x002ae54400000000UL /* Move (16<-16), form: SIL */
#define LIBXSMM_S390X_INSTR_MVHI 0x002ae54c00000000UL /* Move (32<-16), form: SIL */
#define LIBXSMM_S390X_INSTR_MVI 0x0029000092000000UL /* Move Immediate, form: SI */
#define LIBXSMM_S390X_INSTR_MVIY 0x002beb0000000052UL /* Move Immediate, form: SIY */
#define LIBXSMM_S390X_INSTR_NCGRK 0x00150000b9e50000UL /* AND with Complement (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_NG 0x0026e30000000080UL /* AND (64), form: RXY-a */
#define LIBXSMM_S390X_INSTR_NGR 0x00140000b9800000UL /* AND (64), form: RRE */
#define LIBXSMM_S390X_INSTR_NGRK 0x00150000b9e40000UL /* AND (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_NI 0x0029000094000000UL /* AND Immediate, form: SI */
#define LIBXSMM_S390X_INSTR_NIAI 0x00020000b2fa0000UL /* Next Instruction Access Intent, form: IE */
#define LIBXSMM_S390X_INSTR_NIHF 0x000ec00a00000000UL /* AND Immediate (high), form: RIL-a */
#define LIBXSMM_S390X_INSTR_NIHH 0x00040000a5040000UL /* AND Immediate (high high), form: RI-a */
#define LIBXSMM_S390X_INSTR_NIHL 0x00040000a5050000UL /* AND Immediate (high low), form: RI-a */
#define LIBXSMM_S390X_INSTR_NILF 0x000ec00b00000000UL /* AND Immediate (low), form: RIL-a */
#define LIBXSMM_S390X_INSTR_NILH 0x00040000a5060000UL /* AND Immediate (low high), form: RI-a */
#define LIBXSMM_S390X_INSTR_NILL 0x00040000a5070000UL /* AND Immediate (low low), form: RI-a */
#define LIBXSMM_S390X_INSTR_NIY 0x002beb0000000054UL /* AND Immediate, form: SIY */
#define LIBXSMM_S390X_INSTR_NNGRK 0x00150000b9640000UL /* NAND (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_NOGRK 0x00150000b9660000UL /* NOR (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_NTSTG 0x0026e30000000025UL /* Nontransactional Store (64), form: RXY-a */
#define LIBXSMM_S390X_INSTR_NXGRK 0x00150000b9670000UL /* NOT Exclusive OR (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_OCGRK 0x00150000b9650000UL /* OR with Complement (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_OG 0x0026e30000000081UL /* OR (64), form: RXY-a */
#define LIBXSMM_S390X_INSTR_OGR 0x00140000b9810000UL /* OR (64), form: RRE */
#define LIBXSMM_S390X_INSTR_OGRK 0x00150000b9e60000UL /* OR (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_OI 0x0029000096000000UL /* OR Immediate, form: SI */
#define LIBXSMM_S390X_INSTR_OIHF 0x000ec00c00000000UL /* OR Immediate (high), form: RIL-a */
#define LIBXSMM_S390X_INSTR_OIHH 0x00040000a5080000UL /* OR Immediate (high high), form: RI-a */
#define LIBXSMM_S390X_INSTR_OIHL 0x00040000a5090000UL /* OR Immediate (high low), form: RI-a */
#define LIBXSMM_S390X_INSTR_OILF 0x000ec00d00000000UL /* OR Immediate (low), form: RIL-a */
#define LIBXSMM_S390X_INSTR_OILH 0x00040000a50a0000UL /* OR Immediate (low high), form: RI-a */
#define LIBXSMM_S390X_INSTR_OILL 0x00040000a50b0000UL /* OR Immediate (low low), form: RI-a */
#define LIBXSMM_S390X_INSTR_OIY 0x002beb0000000056UL /* OR Immediate, form: SIY */
#define LIBXSMM_S390X_INSTR_RISBG 0x000cec0000000055UL /* [,I5] Rotate then Insert Selected Bits (64), form: RIE-f */
#define LIBXSMM_S390X_INSTR_RISBGN 0x000cec0000000059UL /* [,I5] Rotate then Insert Selected Bits (64), form: RIE-f */
#define LIBXSMM_S390X_INSTR_RISBHG 0x000cec000000005dUL /* [,I5] Rotate then Insert Selected Bits High (32), form: RIE-f */
#define LIBXSMM_S390X_INSTR_RISBLG 0x000cec0000000051UL /* [,I5] Rotate then Insert Selected Bits Low (32), form: RIE-f */
#define LIBXSMM_S390X_INSTR_RLL 0x0020eb000000001dUL /* Rotate Left Single Logical (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_RLLG 0x0020eb000000001cUL /* Rotate Left Single Logical (64), form: RSY-a */
#define LIBXSMM_S390X_INSTR_RNSBG 0x000cec0000000054UL /* [,I5] Rotate then AND Selected Bits (64), form: RIE-f */
#define LIBXSMM_S390X_INSTR_ROSBG 0x000cec0000000056UL /* [,I5] Rotate then OR Selected Bits (64), form: RIE-f */
#define LIBXSMM_S390X_INSTR_SLA 0x001b00008b000000UL /* Shift Left Single (32), form: RS-a */
#define LIBXSMM_S390X_INSTR_SLAG 0x0020eb000000000bUL /* Shift Left Single (64), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SLAK 0x0020eb00000000ddUL /* Shift Left Single (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SLDA 0x001b00008f000000UL /* Shift Left Double (64), form: RS-a */
#define LIBXSMM_S390X_INSTR_SLDL 0x001b00008d000000UL /* Shift Left Double Logical (64), form: RS-a */
#define LIBXSMM_S390X_INSTR_SLDT 0x0025ed0000000040UL /* Shift Significand Left (LD), form: RXF */
#define LIBXSMM_S390X_INSTR_SLL 0x001b000089000000UL /* Shift Left Single Logical (32), form: RS-a */
#define LIBXSMM_S390X_INSTR_SLLG 0x0020eb000000000dUL /* Shift Left Single Logical (64), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SLLK 0x0020eb00000000dfUL /* Shift Left Single Logical (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SRA 0x001b00008a000000UL /* Shift Right Single (32), form: RS-a */
#define LIBXSMM_S390X_INSTR_SRAG 0x0020eb000000000aUL /* Shift Right Single (64), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SRAK 0x0020eb00000000dcUL /* Shift Right Single (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SRDA 0x001b00008e000000UL /* Shift Right Double (64), form: RS-a */
#define LIBXSMM_S390X_INSTR_SRDL 0x001b00008c000000UL /* Shift Right Double Logical (64), form: RS-a */
#define LIBXSMM_S390X_INSTR_SRDT 0x0025ed0000000041UL /* Shift Significand Right (LD), form: RXF */
#define LIBXSMM_S390X_INSTR_SRK 0x00150000b9f90000UL /* Subtract (32), form: RRF-a */
#define LIBXSMM_S390X_INSTR_SRL 0x001b000088000000UL /* Shift Right Single Logical (32), form: RS-a */
#define LIBXSMM_S390X_INSTR_SRLG 0x0020eb000000000cUL /* Shift Right Single Logical (64), form: RSY-a */
#define LIBXSMM_S390X_INSTR_SRLK 0x0020eb00000000deUL /* Shift Right Single Logical (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_ST 0x0022000050000000UL /* Store (32), form: RX-a */
#define LIBXSMM_S390X_INSTR_STAM 0x001b00009b000000UL /* Store Access Multiple, form: RS-a */
#define LIBXSMM_S390X_INSTR_STAMY 0x0020eb000000009bUL /* Store Access Multiple, form: RSY-a */
#define LIBXSMM_S390X_INSTR_STG 0x0026e30000000024UL /* Store (64), form: RXY-a */
#define LIBXSMM_S390X_INSTR_STH 0x0022000040000000UL /* Store Halfword (16), form: RX-a */
#define LIBXSMM_S390X_INSTR_STM 0x001b000090000000UL /* Store Multiple (32), form: RS-a */
#define LIBXSMM_S390X_INSTR_STMG 0x0020eb0000000024UL /* Store Multiple (64), form: RSY-a */
#define LIBXSMM_S390X_INSTR_STMH 0x0020eb0000000026UL /* Store Multiple High (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_STMY 0x0020eb0000000090UL /* Store Multiple (32), form: RSY-a */
#define LIBXSMM_S390X_INSTR_STY 0x0026e30000000050UL /* Store (32), form: RXY-a */
#define LIBXSMM_S390X_INSTR_TBDR 0x00190000b3510000UL /* Convert HFP to BFP (LB<-LH), form: RRF-e */
#define LIBXSMM_S390X_INSTR_TBEDR 0x00190000b3500000UL /* Convert HFP to BFP (SB<-LH), form: RRF-e */
#define LIBXSMM_S390X_INSTR_VA 0x0040e700000000f3UL /* Vector Add, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VAC 0x0041e700000000bbUL /* Vector Add With Carry, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VACC 0x0040e700000000f1UL /* Vector Add Compute Carry, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VACCC 0x0041e700000000b9UL /* Vector Add With Carry Compute Carry, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VAP 0x003ae60000000071UL /* Vector Add Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VAVG 0x0040e700000000f2UL /* Vector Average, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VAVGL 0x0040e700000000f0UL /* Vector Average Logical, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VBPERM 0x0040e70000000085UL /* Vector Bit Permute, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VCDG 0x003ee700000000c3UL /* Vector FP Convert from Fixed 64-bit, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCDLG 0x003ee700000000c1UL /* Vector FP Convert from Logical 64-bit, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCEQ 0x003fe700000000f8UL /* Vector Compare Equal, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VCFN 0x003ee6000000005dUL /* Vector FP Convert From NNP, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCFPL 0x003ee700000000c1UL /* Vector FP Convert from Logical, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCFPS 0x003ee700000000c3UL /* Vector FP Convert from Fixed, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCGD 0x003ee700000000c2UL /* Vector FP Convert to Fixed 64-bit, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCH 0x003fe700000000fbUL /* Vector Compare High, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VCHL 0x003fe700000000f9UL /* Vector Compare High Logical, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VCKSM 0x0040e70000000066UL /* Vector Checksum, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VCLFNH 0x003ee60000000056UL /* Vector FP Convert and Lengthen from NNP High, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCLFNL 0x003ee6000000005eUL /* Vector FP Convert and Lengthen from NNP Low, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCLFP 0x003ee700000000c0UL /* Vector FP Convert to Logical, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCLGD 0x003ee700000000c0UL /* Vector FP Convert to Logical 64-bit, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCLZ 0x003ee70000000053UL /* Vector Count Leading Zeros, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCLZDP 0x0048e60000000051UL /* Vector Count Leading Zero Digits, form: VRR-k */
#define LIBXSMM_S390X_INSTR_VCNF 0x003ee60000000055UL /* Vector FP Convert to NNP, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCP 0x0045e60000000077UL /* Vector Compare Decimal, form: VRR-h */
#define LIBXSMM_S390X_INSTR_VCRNF 0x0040e60000000075UL /* Vector FP Convert and Round to NNP, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VCSFP 0x003ee700000000c2UL /* Vector FP Convert to Fixed, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCSPH 0x0047e6000000007dUL /* Vector Convert HFP to Scaled Decimal, form: VRR-j */
#define LIBXSMM_S390X_INSTR_VCTZ 0x003ee70000000052UL /* Vector Count Trailing Zeros, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VCVB 0x0046e60000000050UL /* Vector Convert to Binary, form: VRR-i */
#define LIBXSMM_S390X_INSTR_VCVBG 0x0046e60000000052UL /* Vector Convert to Binary, form: VRR-i */
#define LIBXSMM_S390X_INSTR_VCVD 0x003de60000000058UL /* Vector Convert to Decimal, form: VRI-i */
#define LIBXSMM_S390X_INSTR_VCVDG 0x003de6000000005aUL /* Vector Convert to Decimal, form: VRI-i */
#define LIBXSMM_S390X_INSTR_VDP 0x003ae6000000007aUL /* Vector Divide Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VEC 0x003ee700000000dbUL /* Vector Element Compare, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VECL 0x003ee700000000d9UL /* Vector Element Compare Logical, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VERIM 0x0038e70000000072UL /* Vector Element Rotate and Insert Under Mask, form: VRI-d */
#define LIBXSMM_S390X_INSTR_VERLL 0x0049e70000000033UL /* Vector Element Rotate Left Logical, form: VRS-a */
#define LIBXSMM_S390X_INSTR_VERLLV 0x0040e70000000073UL /* Vector Element Rotate Left Logical, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VESL 0x0049e70000000030UL /* Vector Element Shift Left, form: VRS-a */
#define LIBXSMM_S390X_INSTR_VESLV 0x0040e70000000070UL /* Vector Element Shift Left, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VESRA 0x0049e7000000003aUL /* Vector Element Shift Right Arithmetic, form: VRS-a */
#define LIBXSMM_S390X_INSTR_VESRAV 0x0040e7000000007aUL /* Vector Element Shift Right Arithmetic, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VESRL 0x0049e70000000038UL /* Vector Element Shift Right Logical, form: VRS-a */
#define LIBXSMM_S390X_INSTR_VESRLV 0x0040e70000000078UL /* Vector Element Shift Right Logical, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFA 0x0040e700000000e3UL /* Vector FP Add, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFAE 0x003fe70000000082UL /* [,M5] Vector Find Any Element Equal, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VFCE 0x0040e700000000e8UL /* Vector FP Compare Equal, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFCH 0x0040e700000000ebUL /* Vector FP Compare High, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFCHE 0x0040e700000000eaUL /* Vector FP Compare High or Equal, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFD 0x0040e700000000e5UL /* Vector FP Divide, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFEE 0x003fe70000000080UL /* [,M5] Vector Find Element Equal, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VFENE 0x003fe70000000081UL /* [,M5] Vector Find Element Not Equal, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VFI 0x003ee700000000c7UL /* Vector Load FP Integer, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VFLL 0x003ee700000000c4UL /* Vector FP Load Lengthened, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VFLR 0x003ee700000000c5UL /* Vector FP Load Rounded, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VFM 0x0040e700000000e7UL /* Vector FP Multiply, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFMA 0x0042e7000000008fUL /* Vector FP Multiply and Add, form: VRR-e */
#define LIBXSMM_S390X_INSTR_VFMAX 0x0040e700000000efUL /* Vector FP Maximum, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFMIN 0x0040e700000000eeUL /* Vector FP Minimum, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFMS 0x0042e7000000008eUL /* Vector FP Multiply and Subtract, form: VRR-e */
#define LIBXSMM_S390X_INSTR_VFNMA 0x0042e7000000009fUL /* Vector FP Negative Multiply and Add, form: VRR-e */
#define LIBXSMM_S390X_INSTR_VFNMS 0x0042e7000000009eUL /* Vector FP Negative Multiply and Subtract, form: VRR-e */
#define LIBXSMM_S390X_INSTR_VFPSO 0x003ee700000000ccUL /* Vector FP Perform Sign Operation, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VFS 0x0040e700000000e2UL /* Vector FP Subtract, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VFSQ 0x003ee700000000ceUL /* Vector FP Square Root, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VFTCI 0x0039e7000000004aUL /* Vector FP Test Data Class Immediate, form: VRI-e */
#define LIBXSMM_S390X_INSTR_VGBM 0x0035e70000000044UL /* Vector Generate Byte Mask, form: VRI-a */
#define LIBXSMM_S390X_INSTR_VGEF 0x004de70000000013UL /* Vector Gather Element (32), form: VRV */
#define LIBXSMM_S390X_INSTR_VGEG 0x004de70000000012UL /* Vector Gather Element (64), form: VRV */
#define LIBXSMM_S390X_INSTR_VGFM 0x0040e700000000b4UL /* Vector Galois Field Multiply Sum, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VGFMA 0x0041e700000000bcUL /* Vector Galois Field Multiply Sum and Accumulate, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VGM 0x0036e70000000046UL /* Vector Generate Mask, form: VRI-b */
#define LIBXSMM_S390X_INSTR_VISTR 0x003ee7000000005cUL /* [,M5] Vector Isolate String, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VL 0x004ee70000000006UL /* Vector Load, form: VRX */
#define LIBXSMM_S390X_INSTR_VLBB 0x004ee70000000007UL /* Vector Load to Block Boundary, form: VRX */
#define LIBXSMM_S390X_INSTR_VLBR 0x004ee60000000006UL /* Vector Load Byte Reversed Elements, form: VRX */
#define LIBXSMM_S390X_INSTR_VLBRREP 0x004ee60000000005UL /* Vector Load Byte Reversed Element and Replicate, form: VRX */
#define LIBXSMM_S390X_INSTR_VLC 0x003ee700000000deUL /* Vector Load Complement, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VLEB 0x004ee70000000000UL /* Vector Load Element (8), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEBRF 0x004ee60000000003UL /* Vector Load Byte Reversed Element (32), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEBRG 0x004ee60000000002UL /* Vector Load Byte Reversed Element (64), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEBRH 0x004ee60000000001UL /* Vector Load Byte Reversed Element (16), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEF 0x004ee70000000003UL /* Vector Load Element (32), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEG 0x004ee70000000002UL /* Vector Load Element (64), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEH 0x004ee70000000001UL /* Vector Load Element (16), form: VRX */
#define LIBXSMM_S390X_INSTR_VLEIB 0x0035e70000000040UL /* Vector Load Element Immediate (8), form: VRI-a */
#define LIBXSMM_S390X_INSTR_VLEIF 0x0035e70000000043UL /* Vector Load Element Immediate (32), form: VRI-a */
#define LIBXSMM_S390X_INSTR_VLEIG 0x0035e70000000042UL /* Vector Load Element Immediate (64), form: VRI-a */
#define LIBXSMM_S390X_INSTR_VLEIH 0x0035e70000000041UL /* Vector Load Element Immediate (16), form: VRI-a */
#define LIBXSMM_S390X_INSTR_VLER 0x004ee60000000007UL /* Vector Load Elements Reversed, form: VRX */
#define LIBXSMM_S390X_INSTR_VLGV 0x004be70000000021UL /* Vector Load GR from VR Element, form: VRS-c */
#define LIBXSMM_S390X_INSTR_VLIP 0x003ce60000000049UL /* Vector Load Immediate Decimal, form: VRI-h */
#define LIBXSMM_S390X_INSTR_VLL 0x004ae70000000037UL /* Vector Load With Length, form: VRS-b */
#define LIBXSMM_S390X_INSTR_VLLEBRZ 0x004ee60000000004UL /* Vector Load Byte Reversed Element and Zero, form: VRX */
#define LIBXSMM_S390X_INSTR_VLLEZ 0x004ee70000000004UL /* Vector Load Logical Element and Zero, form: VRX */
#define LIBXSMM_S390X_INSTR_VLM 0x0049e70000000036UL /* [,M4] Vector Load Multiple, form: VRS-a */
#define LIBXSMM_S390X_INSTR_VLP 0x003ee700000000dfUL /* Vector Load Positive, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VLR 0x003ee70000000056UL /* Vector Load, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VLREP 0x004ee70000000005UL /* Vector Load and Replicate, form: VRX */
#define LIBXSMM_S390X_INSTR_VLRL 0x004fe60000000035UL /* Vector Load Rightmost with Length, form: VSI */
#define LIBXSMM_S390X_INSTR_VLRLR 0x004ce60000000037UL /* Vector Load Rightmost with Length, form: VRS-d */
#define LIBXSMM_S390X_INSTR_VLVG 0x004ae70000000022UL /* Vector Load VR Element from GR, form: VRS-b */
#define LIBXSMM_S390X_INSTR_VLVGP 0x0043e70000000062UL /* Vector Load VR from GRs Disjoint, form: VRR-f */
#define LIBXSMM_S390X_INSTR_VMAE 0x0041e700000000aeUL /* Vector Multiply and Add Even, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMAH 0x0041e700000000abUL /* Vector Multiply and Add High, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMAL 0x0041e700000000aaUL /* Vector Multiply and Add Low, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMALE 0x0041e700000000acUL /* Vector Multiply and Add Logical Even, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMALH 0x0041e700000000a9UL /* Vector Multiply and Add Logical High, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMALO 0x0041e700000000adUL /* Vector Multiply and Add Logical Odd, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMAO 0x0041e700000000afUL /* Vector Multiply and Add Odd, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VME 0x0040e700000000a6UL /* Vector Multiply Even, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMH 0x0040e700000000a3UL /* Vector Multiply High, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VML 0x0040e700000000a2UL /* Vector Multiply Low, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMLE 0x0040e700000000a4UL /* Vector Multiply Logical Even, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMLH 0x0040e700000000a1UL /* Vector Multiply Logical High, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMLO 0x0040e700000000a5UL /* Vector Multiply Logical Odd, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMN 0x0040e700000000feUL /* Vector Minimum, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMNL 0x0040e700000000fcUL /* Vector Minimum Logical, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMO 0x0040e700000000a7UL /* Vector Multiply Odd, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMP 0x003ae60000000078UL /* Vector Multiply Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VMRH 0x0040e70000000061UL /* Vector Merge High, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMRL 0x0040e70000000060UL /* Vector Merge Low, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMSL 0x0041e700000000b8UL /* Vector Multiply Sum Logical, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VMSP 0x003ae60000000079UL /* Vector Multiply and Shift Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VMX 0x0040e700000000ffUL /* Vector Maximum, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VMXL 0x0040e700000000fdUL /* Vector Maximum Logical, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VN 0x0040e70000000068UL /* Vector AND, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VNC 0x0040e70000000069UL /* Vector AND with Complement, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VNN 0x0040e7000000006eUL /* Vector NAND, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VNO 0x0040e7000000006bUL /* Vector NOR, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VNX 0x0040e7000000006cUL /* Vector Not Exclusive OR, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VO 0x0040e7000000006aUL /* Vector OR, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VOC 0x0040e7000000006fUL /* Vector OR with Complement, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VPDI 0x0040e70000000084UL /* Vector Permute Doubleword Immediate, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VPERM 0x0042e7000000008cUL /* Vector Permute, form: VRR-e */
#define LIBXSMM_S390X_INSTR_VPK 0x0040e70000000094UL /* Vector Pack, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VPKLS 0x003fe70000000095UL /* Vector Pack Logical Saturate, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VPKS 0x003fe70000000097UL /* Vector Pack Saturate, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VPKZ 0x004fe60000000034UL /* Vector Pack Zoned, form: VSI */
#define LIBXSMM_S390X_INSTR_VPKZR 0x003ae60000000070UL /* Vector Pack Zoned Register, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VPOPCT 0x003ee70000000050UL /* Vector Population Count, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VPSOP 0x003be6000000005bUL /* Vector Perform Sign Operation Decimal, form: VRI-g */
#define LIBXSMM_S390X_INSTR_VREP 0x0037e7000000004dUL /* Vector Replicate, form: VRI-c */
#define LIBXSMM_S390X_INSTR_VREPI 0x0035e70000000045UL /* Vector Replicate Immediate, form: VRI-a */
#define LIBXSMM_S390X_INSTR_VRP 0x003ae6000000007bUL /* Vector Remainder Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VS 0x0040e700000000f7UL /* Vector Subtract, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSBCBI 0x0041e700000000bdUL /* Vector Subtract With Borrow Compute Borrow Indication, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VSBI 0x0041e700000000bfUL /* Vector Subtract With Borrow Indication, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VSCBI 0x0040e700000000f5UL /* Vector Subtract Compute Borrow Indication, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSCEF 0x004de7000000001bUL /* Vector Scatter Element (32), form: VRV */
#define LIBXSMM_S390X_INSTR_VSCEG 0x004de7000000001aUL /* Vector Scatter Element (64), form: VRV */
#define LIBXSMM_S390X_INSTR_VSCHP 0x003fe60000000074UL /* Decimal Scale and Convert to HFP, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VSCSHP 0x003fe6000000007cUL /* Decimal Scale and Convert and Split to HFP, form: VRR-b */
#define LIBXSMM_S390X_INSTR_VSDP 0x003ae6000000007eUL /* Vector Shift and Divide Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VSEG 0x003ee7000000005fUL /* Vector Sign Extend to Doubleword, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VSEL 0x0042e7000000008dUL /* Vector Select, form: VRR-e */
#define LIBXSMM_S390X_INSTR_VSL 0x0040e70000000074UL /* Vector Shift Left, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSLB 0x0040e70000000075UL /* Vector Shift Left By Byte, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSLD 0x0038e70000000086UL /* Vector Shift Left Double by Bit, form: VRI-d */
#define LIBXSMM_S390X_INSTR_VSLDB 0x0038e70000000077UL /* Vector Shift Left Double By Byte, form: VRI-d */
#define LIBXSMM_S390X_INSTR_VSP 0x003ae60000000073UL /* Vector Subtract Decimal, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VSRA 0x0040e7000000007eUL /* Vector Shift Right Arithmetic, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSRAB 0x0040e7000000007fUL /* Vector Shift Right Arithmetic By Byte, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSRD 0x0038e70000000087UL /* Vector Shift Right Double by Bit, form: VRI-d */
#define LIBXSMM_S390X_INSTR_VSRL 0x0040e7000000007cUL /* Vector Shift Right Logical, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSRLB 0x0040e7000000007dUL /* Vector Shift Right Logical By Byte, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSRP 0x003be60000000059UL /* Vector Shift and Round Decimal, form: VRI-g */
#define LIBXSMM_S390X_INSTR_VSRPR 0x003ae60000000072UL /* Vector Shift and Round Decimal Register, form: VRI-f */
#define LIBXSMM_S390X_INSTR_VST 0x004ee7000000000eUL /* [,M3] Vector Store, form: VRX */
#define LIBXSMM_S390X_INSTR_VSTBR 0x004ee6000000000eUL /* Vector Store Byte Reversed Elements, form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEB 0x004ee70000000008UL /* Vector Store Element (8), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEBRF 0x004ee6000000000bUL /* Vector Store Byte Reversed Element (32), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEBRG 0x004ee6000000000aUL /* Vector Store Byte Reversed Element (64), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEBRH 0x004ee60000000009UL /* Vector Store Byte Reversed Element (16), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEF 0x004ee7000000000bUL /* Vector Store Element (32), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEG 0x004ee7000000000aUL /* Vector Store Element (64), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTEH 0x004ee70000000009UL /* Vector Store Element (16), form: VRX */
#define LIBXSMM_S390X_INSTR_VSTER 0x004ee6000000000fUL /* Vector Store Elements Reversed, form: VRX */
#define LIBXSMM_S390X_INSTR_VSTL 0x004ae7000000003fUL /* Vector Store With Length, form: VRS-b */
#define LIBXSMM_S390X_INSTR_VSTM 0x0049e7000000003eUL /* [,M4] Vector Store Multiple, form: VRS-a */
#define LIBXSMM_S390X_INSTR_VSTRC 0x0041e7000000008aUL /* [,M6] Vector String Range Compare, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VSTRL 0x004fe6000000003dUL /* Vector Store Rightmost with Length, form: VSI */
#define LIBXSMM_S390X_INSTR_VSTRLR 0x004ce6000000003fUL /* Vector Store Rightmost with Length, form: VRS-d */
#define LIBXSMM_S390X_INSTR_VSTRS 0x0041e7000000008bUL /* [,M6] Vector String Search, form: VRR-d */
#define LIBXSMM_S390X_INSTR_VSUM 0x0040e70000000064UL /* Vector Sum Across Word, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSUMG 0x0040e70000000065UL /* Vector Sum Across Doubleword, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VSUMQ 0x0040e70000000067UL /* Vector Sum Across Quadword, form: VRR-c */
#define LIBXSMM_S390X_INSTR_VTM 0x003ee700000000d8UL /* Vector Test Under Mask, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VTP 0x0044e6000000005fUL /* Vector Test Decimal, form: VRR-g */
#define LIBXSMM_S390X_INSTR_VUPH 0x003ee700000000d7UL /* Vector Unpack High, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VUPKZ 0x004fe6000000003cUL /* Vector Unpack Zoned, form: VSI */
#define LIBXSMM_S390X_INSTR_VUPKZH 0x0048e60000000054UL /* Vector Unpack Zoned High, form: VRR-k */
#define LIBXSMM_S390X_INSTR_VUPKZL 0x0048e6000000005cUL /* Vector Unpack Zoned Low, form: VRR-k */
#define LIBXSMM_S390X_INSTR_VUPL 0x003ee700000000d6UL /* Vector Unpack Low, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VUPLH 0x003ee700000000d5UL /* Vector Unpack Logical High, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VUPLL 0x003ee700000000d4UL /* Vector Unpack Logical Low, form: VRR-a */
#define LIBXSMM_S390X_INSTR_VX 0x0040e7000000006dUL /* Vector Exclusive OR, form: VRR-c */
#define LIBXSMM_S390X_INSTR_WFC 0x003ee700000000cbUL /* Vector FP Compare Scalar, form: VRR-a */
#define LIBXSMM_S390X_INSTR_WFK 0x003ee700000000caUL /* Vector FP Compare and Signal Scalar, form: VRR-a */
#define LIBXSMM_S390X_INSTR_X 0x0022000057000000UL /* Exclusive OR (32), form: RX-a */
#define LIBXSMM_S390X_INSTR_XC 0x002dd70000000000UL /* Exclusive OR (character), form: SS-a */
#define LIBXSMM_S390X_INSTR_XG 0x0026e30000000082UL /* Exclusive OR (64), form: RXY-a */
#define LIBXSMM_S390X_INSTR_XGR 0x00140000b9820000UL /* Exclusive OR (64), form: RRE */
#define LIBXSMM_S390X_INSTR_XGRK 0x00150000b9e70000UL /* Exclusive OR (64), form: RRF-a */
#define LIBXSMM_S390X_INSTR_XI 0x0029000097000000UL /* Exclusive OR Immediate, form: SI */
#define LIBXSMM_S390X_INSTR_XIHF 0x000ec00600000000UL /* Exclusive OR Immediate (high), form: RIL-a */
#define LIBXSMM_S390X_INSTR_XILF 0x000ec00700000000UL /* Exclusive OR Immediate (low), form: RIL-a */
#define LIBXSMM_S390X_INSTR_XIY 0x002beb0000000057UL /* Exclusive OR Immediate, form: SIY */
#define LIBXSMM_S390X_INSTR_XR 0x0012000000001700UL /* Exclusive OR (32), form: RR */
#define LIBXSMM_S390X_INSTR_XRK 0x00150000b9f70000UL /* Exclusive OR (32), form: RRF-a */
#define LIBXSMM_S390X_INSTR_XSCH 0x00280000b2760000UL /* ancel Subchannel, form: S */
#define LIBXSMM_S390X_INSTR_XY 0x0026e30000000057UL /* Exclusive Or (32), form: RXY-a */
#define LIBXSMM_S390X_INSTR_ZAP 0x002ef80000000000UL /* Zero and Add, form: SS-b */

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
char libxsmm_s390x_instr_rrf_a_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_b_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_c_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_d_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_e_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output);

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
