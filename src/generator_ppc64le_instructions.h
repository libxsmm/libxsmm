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

#ifndef GENERATOR_PPC64LE_INSTRUCTIONS_H
#define GENERATOR_PPC64LE_INSTRUCTIONS_H

#include "generator_common.h"

#define LIBXSMM_PPC64LE_GPR_NMAX 32
#define LIBXSMM_PPC64LE_FPR_NMAX 32
#define LIBXSMM_PPC64LE_VR_NMAX  32
#define LIBXSMM_PPC64LE_VSR_NMAX 64
#define LIBXSMM_PPC64LE_ACC_NMAX 8

/* number of volatile registers */
/* From "64-Bit ELF V2 ABI Specification: Power Architecture"
 * A number of registers are either volatile or non-volatile.
 * The numbers below are the number of the first non-volatile
 * register
 */
#define LIBXSMM_PPC64LE_GPR_IVOL 13
#define LIBXSMM_PPC64LE_FPR_IVOL 14
#define LIBXSMM_PPC64LE_VR_IVOL  20

/* general purpose registers */
#define LIBXSMM_PPC64LE_GPR_R0   0
#define LIBXSMM_PPC64LE_GPR_R1   1
#define LIBXSMM_PPC64LE_GPR_R2   2
#define LIBXSMM_PPC64LE_GPR_R3   3
#define LIBXSMM_PPC64LE_GPR_R4   4
#define LIBXSMM_PPC64LE_GPR_R5   5
#define LIBXSMM_PPC64LE_GPR_R6   6
#define LIBXSMM_PPC64LE_GPR_R7   7
#define LIBXSMM_PPC64LE_GPR_R8   8
#define LIBXSMM_PPC64LE_GPR_R9   9
#define LIBXSMM_PPC64LE_GPR_R10 10
#define LIBXSMM_PPC64LE_GPR_R11 11
#define LIBXSMM_PPC64LE_GPR_R12 12
#define LIBXSMM_PPC64LE_GPR_R13 13
#define LIBXSMM_PPC64LE_GPR_R14 14
#define LIBXSMM_PPC64LE_GPR_R15 15
#define LIBXSMM_PPC64LE_GPR_R16 16
#define LIBXSMM_PPC64LE_GPR_R17 17
#define LIBXSMM_PPC64LE_GPR_R18 18
#define LIBXSMM_PPC64LE_GPR_R19 19
#define LIBXSMM_PPC64LE_GPR_R20 20
#define LIBXSMM_PPC64LE_GPR_R21 21
#define LIBXSMM_PPC64LE_GPR_R22 22
#define LIBXSMM_PPC64LE_GPR_R23 23
#define LIBXSMM_PPC64LE_GPR_R24 24
#define LIBXSMM_PPC64LE_GPR_R25 25
#define LIBXSMM_PPC64LE_GPR_R26 26
#define LIBXSMM_PPC64LE_GPR_R27 27
#define LIBXSMM_PPC64LE_GPR_R28 28
#define LIBXSMM_PPC64LE_GPR_R29 29
#define LIBXSMM_PPC64LE_GPR_R30 30
#define LIBXSMM_PPC64LE_GPR_R31 31


/* special registers */
#define LIBXSMM_PPC64LE_GPR_SP 1 /* Stack pointer (GPR R1) */


/* 5-bit chunks reversed for SPR */
#define LIBXSMM_PPC64LE_SPR_XER 0x00000020 /* REG 1 64-bit */
#define LIBXSMM_PPC64LE_SPR_DSCR 0x00000060 /* REG 3 64-bit */
#define LIBXSMM_PPC64LE_SPR_LR 0x00000100 /* REG 8 64-bit */
#define LIBXSMM_PPC64LE_SPR_CTR 0x00000120 /* REG 9 64-bit */
#define LIBXSMM_PPC64LE_SPR_AMR 0x000001a0 /* REG 13 64-bit */
#define LIBXSMM_PPC64LE_SPR_CTRL 0x00000104 /* REG 136 32-bit */
#define LIBXSMM_PPC64LE_SPR_VRSAVE 0x00000008 /* REG 256 32-bit */
#define LIBXSMM_PPC64LE_SPR_SPRG3 0x00000068 /* REG 259 64-bit */
#define LIBXSMM_PPC64LE_SPR_TB 0x00000188 /* REG 268 64-bit */
#define LIBXSMM_PPC64LE_SPR_TBU 0x000001a8 /* REG 269 32-bit */
#define LIBXSMM_PPC64LE_SPR_HDEXCR 0x000000ee /* REG 455 32-bit */
#define LIBXSMM_PPC64LE_SPR_SIER2 0x00000017 /* REG 736 64-bit */
#define LIBXSMM_PPC64LE_SPR_SIER3 0x00000037 /* REG 737 64-bit */
#define LIBXSMM_PPC64LE_SPR_MMCR3 0x00000057 /* REG 738 64-bit */
#define LIBXSMM_PPC64LE_SPR_SIER 0x00000018 /* REG 768 64-bit */
#define LIBXSMM_PPC64LE_SPR_MMCR2 0x00000038 /* REG 769 64-bit */
#define LIBXSMM_PPC64LE_SPR_MMCRA 0x00000058 /* REG 770 64-bit */
#define LIBXSMM_PPC64LE_SPR_PMC1 0x00000078 /* REG 771 32-bit */
#define LIBXSMM_PPC64LE_SPR_PMC2 0x00000098 /* REG 772 32-bit */
#define LIBXSMM_PPC64LE_SPR_PMC3 0x000000b8 /* REG 773 32-bit */
#define LIBXSMM_PPC64LE_SPR_PMC4 0x000000d8 /* REG 774 32-bit */
#define LIBXSMM_PPC64LE_SPR_PMC5 0x000000f8 /* REG 775 32-bit */
#define LIBXSMM_PPC64LE_SPR_PMC6 0x00000118 /* REG 776 32-bit */
#define LIBXSMM_PPC64LE_SPR_MMCR0 0x00000178 /* REG 779 64-bit */
#define LIBXSMM_PPC64LE_SPR_SIAR 0x00000198 /* REG 780 64-bit */
#define LIBXSMM_PPC64LE_SPR_SDAR 0x000001b8 /* REG 781 64-bit */
#define LIBXSMM_PPC64LE_SPR_MMCR1 0x000001d8 /* REG 782 64-bit */
#define LIBXSMM_PPC64LE_SPR_BESCRS15 0x00000019 /* REG 800 64-bit */
#define LIBXSMM_PPC64LE_SPR_BESCRSU16 0x00000039 /* REG 801 32-bit */
#define LIBXSMM_PPC64LE_SPR_BESCRR15 0x00000059 /* REG 802 64-bit */
#define LIBXSMM_PPC64LE_SPR_BESCRRU16 0x00000079 /* REG 803 32-bit */
#define LIBXSMM_PPC64LE_SPR_EBBHR 0x00000099 /* REG 804 64-bit */
#define LIBXSMM_PPC64LE_SPR_EBBRR 0x000000b9 /* REG 805 64-bit */
#define LIBXSMM_PPC64LE_SPR_BESCR 0x000000d9 /* REG 806 64-bit */
#define LIBXSMM_PPC64LE_SPR_DEXCR 0x00000199 /* REG 812 32-bit */
#define LIBXSMM_PPC64LE_SPR_TAR 0x000001d9 /* REG 815 64-bit */
#define LIBXSMM_PPC64LE_SPR_PPR 0x0000001c /* REG 896 64-bit */
#define LIBXSMM_PPC64LE_SPR_PPR32 0x0000005c /* REG 898 32-bit */


/* floating-point registers */
#define LIBXSMM_PPC64LE_FPR_WIDTH 64
#define LIBXSMM_PPC64LE_FPR_F0    0
#define LIBXSMM_PPC64LE_FPR_F1    1
#define LIBXSMM_PPC64LE_FPR_F2    2
#define LIBXSMM_PPC64LE_FPR_F3    3
#define LIBXSMM_PPC64LE_FPR_F4    4
#define LIBXSMM_PPC64LE_FPR_F5    5
#define LIBXSMM_PPC64LE_FPR_F6    6
#define LIBXSMM_PPC64LE_FPR_F7    7
#define LIBXSMM_PPC64LE_FPR_F8    8
#define LIBXSMM_PPC64LE_FPR_F9    9
#define LIBXSMM_PPC64LE_FPR_F10  10
#define LIBXSMM_PPC64LE_FPR_F11  11
#define LIBXSMM_PPC64LE_FPR_F12  12
#define LIBXSMM_PPC64LE_FPR_F13  13
#define LIBXSMM_PPC64LE_FPR_F14  14
#define LIBXSMM_PPC64LE_FPR_F15  15
#define LIBXSMM_PPC64LE_FPR_F16  16
#define LIBXSMM_PPC64LE_FPR_F17  17
#define LIBXSMM_PPC64LE_FPR_F18  18
#define LIBXSMM_PPC64LE_FPR_F19  19
#define LIBXSMM_PPC64LE_FPR_F20  20
#define LIBXSMM_PPC64LE_FPR_F21  21
#define LIBXSMM_PPC64LE_FPR_F22  22
#define LIBXSMM_PPC64LE_FPR_F23  23
#define LIBXSMM_PPC64LE_FPR_F24  24
#define LIBXSMM_PPC64LE_FPR_F25  25
#define LIBXSMM_PPC64LE_FPR_F26  26
#define LIBXSMM_PPC64LE_FPR_F27  27
#define LIBXSMM_PPC64LE_FPR_F28  28
#define LIBXSMM_PPC64LE_FPR_F29  29
#define LIBXSMM_PPC64LE_FPR_F30  30
#define LIBXSMM_PPC64LE_FPR_F31  31


/* vector status and control register */
#define LIBXSMM_PPC64LE_VR_WIDTH 128
#define LIBXSMM_PPC64LE_VR_V0    0
#define LIBXSMM_PPC64LE_VR_V1    1
#define LIBXSMM_PPC64LE_VR_V2    2
#define LIBXSMM_PPC64LE_VR_V3    3
#define LIBXSMM_PPC64LE_VR_V4    4
#define LIBXSMM_PPC64LE_VR_V5    5
#define LIBXSMM_PPC64LE_VR_V6    6
#define LIBXSMM_PPC64LE_VR_V7    7
#define LIBXSMM_PPC64LE_VR_V8    8
#define LIBXSMM_PPC64LE_VR_V9    9
#define LIBXSMM_PPC64LE_VR_V10  10
#define LIBXSMM_PPC64LE_VR_V11  11
#define LIBXSMM_PPC64LE_VR_V12  12
#define LIBXSMM_PPC64LE_VR_V13  13
#define LIBXSMM_PPC64LE_VR_V14  14
#define LIBXSMM_PPC64LE_VR_V15  15
#define LIBXSMM_PPC64LE_VR_V16  16
#define LIBXSMM_PPC64LE_VR_V17  17
#define LIBXSMM_PPC64LE_VR_V18  18
#define LIBXSMM_PPC64LE_VR_V19  19
#define LIBXSMM_PPC64LE_VR_V20  20
#define LIBXSMM_PPC64LE_VR_V21  21
#define LIBXSMM_PPC64LE_VR_V22  22
#define LIBXSMM_PPC64LE_VR_V23  23
#define LIBXSMM_PPC64LE_VR_V24  24
#define LIBXSMM_PPC64LE_VR_V25  25
#define LIBXSMM_PPC64LE_VR_V26  26
#define LIBXSMM_PPC64LE_VR_V27  27
#define LIBXSMM_PPC64LE_VR_V28  28
#define LIBXSMM_PPC64LE_VR_V29  29
#define LIBXSMM_PPC64LE_VR_V30  30
#define LIBXSMM_PPC64LE_VR_V31  31

/* vector-scaler status and control register */
#define LIBXSMM_PPC64LE_VSR_WIDTH 128
#define LIBXSMM_PPC64LE_VSR_VS0   0
#define LIBXSMM_PPC64LE_VSR_VS1   1
#define LIBXSMM_PPC64LE_VSR_VS2   2
#define LIBXSMM_PPC64LE_VSR_VS3   3
#define LIBXSMM_PPC64LE_VSR_VS4   4
#define LIBXSMM_PPC64LE_VSR_VS5   5
#define LIBXSMM_PPC64LE_VSR_VS6   6
#define LIBXSMM_PPC64LE_VSR_VS7   7
#define LIBXSMM_PPC64LE_VSR_VS8   8
#define LIBXSMM_PPC64LE_VSR_VS9   9
#define LIBXSMM_PPC64LE_VSR_VS10 10
#define LIBXSMM_PPC64LE_VSR_VS11 11
#define LIBXSMM_PPC64LE_VSR_VS12 12
#define LIBXSMM_PPC64LE_VSR_VS13 13
#define LIBXSMM_PPC64LE_VSR_VS14 14
#define LIBXSMM_PPC64LE_VSR_VS15 15
#define LIBXSMM_PPC64LE_VSR_VS16 16
#define LIBXSMM_PPC64LE_VSR_VS17 17
#define LIBXSMM_PPC64LE_VSR_VS18 18
#define LIBXSMM_PPC64LE_VSR_VS19 19
#define LIBXSMM_PPC64LE_VSR_VS20 20
#define LIBXSMM_PPC64LE_VSR_VS21 21
#define LIBXSMM_PPC64LE_VSR_VS22 22
#define LIBXSMM_PPC64LE_VSR_VS23 23
#define LIBXSMM_PPC64LE_VSR_VS24 24
#define LIBXSMM_PPC64LE_VSR_VS25 25
#define LIBXSMM_PPC64LE_VSR_VS26 26
#define LIBXSMM_PPC64LE_VSR_VS27 27
#define LIBXSMM_PPC64LE_VSR_VS28 28
#define LIBXSMM_PPC64LE_VSR_VS29 29
#define LIBXSMM_PPC64LE_VSR_VS30 30
#define LIBXSMM_PPC64LE_VSR_VS31 31
#define LIBXSMM_PPC64LE_VSR_VS32 32
#define LIBXSMM_PPC64LE_VSR_VS33 33
#define LIBXSMM_PPC64LE_VSR_VS34 34
#define LIBXSMM_PPC64LE_VSR_VS35 35
#define LIBXSMM_PPC64LE_VSR_VS36 36
#define LIBXSMM_PPC64LE_VSR_VS37 37
#define LIBXSMM_PPC64LE_VSR_VS38 38
#define LIBXSMM_PPC64LE_VSR_VS39 39
#define LIBXSMM_PPC64LE_VSR_VS40 40
#define LIBXSMM_PPC64LE_VSR_VS41 41
#define LIBXSMM_PPC64LE_VSR_VS42 42
#define LIBXSMM_PPC64LE_VSR_VS43 43
#define LIBXSMM_PPC64LE_VSR_VS44 44
#define LIBXSMM_PPC64LE_VSR_VS45 45
#define LIBXSMM_PPC64LE_VSR_VS46 46
#define LIBXSMM_PPC64LE_VSR_VS47 47
#define LIBXSMM_PPC64LE_VSR_VS48 48
#define LIBXSMM_PPC64LE_VSR_VS49 49
#define LIBXSMM_PPC64LE_VSR_VS50 50
#define LIBXSMM_PPC64LE_VSR_VS51 51
#define LIBXSMM_PPC64LE_VSR_VS52 52
#define LIBXSMM_PPC64LE_VSR_VS53 53
#define LIBXSMM_PPC64LE_VSR_VS54 54
#define LIBXSMM_PPC64LE_VSR_VS55 55
#define LIBXSMM_PPC64LE_VSR_VS56 56
#define LIBXSMM_PPC64LE_VSR_VS57 57
#define LIBXSMM_PPC64LE_VSR_VS58 58
#define LIBXSMM_PPC64LE_VSR_VS59 59
#define LIBXSMM_PPC64LE_VSR_VS60 60
#define LIBXSMM_PPC64LE_VSR_VS61 61
#define LIBXSMM_PPC64LE_VSR_VS62 62
#define LIBXSMM_PPC64LE_VSR_VS63 63


/* accumulators */
#define LIBXSMM_PPC64LE_ACC_A0 0
#define LIBXSMM_PPC64LE_ACC_A1 1
#define LIBXSMM_PPC64LE_ACC_A2 2
#define LIBXSMM_PPC64LE_ACC_A3 3
#define LIBXSMM_PPC64LE_ACC_A4 4
#define LIBXSMM_PPC64LE_ACC_A5 5
#define LIBXSMM_PPC64LE_ACC_A6 6
#define LIBXSMM_PPC64LE_ACC_A7 7

#define LIBXSMM_PPC64LE_GPR 0
#define LIBXSMM_PPC64LE_FPR 1
#define LIBXSMM_PPC64LE_VR 2
#define LIBXSMM_PPC64LE_VSR 3
#define LIBXSMM_PPC64LE_ACC 3

#define LIBXSMM_PPC64LE_REG_RESV -2
#define LIBXSMM_PPC64LE_REG_USED -1
#define LIBXSMM_PPC64LE_REG_FREE 0

struct libxsmm_ppc64le_reg {
  unsigned int gpr[LIBXSMM_PPC64LE_GPR_NMAX];
  unsigned int fpr[LIBXSMM_PPC64LE_FPR_NMAX];
  unsigned int vr[LIBXSMM_PPC64LE_VR_NMAX];
  unsigned int vsr[LIBXSMM_PPC64LE_VSR_NMAX];
  unsigned int acc[LIBXSMM_PPC64LE_ACC_NMAX];
};

#define LIBXSMM_PPC64LE_REG_DEFAULT { { /* GPR */ \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_RESV, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE }, /* FPR */ { \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE }, { /* VR */ \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE }, { /* VSR */ \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE }, { /* ACC */ \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE, \
  LIBXSMM_PPC64LE_REG_FREE } \
};

typedef struct libxsmm_ppc64le_reg libxsmm_ppc64le_reg;

/* undefined instruction */
#define LIBXSMM_PPC64LE_INSTR_UNDEF 9999

/* nop */
#define LIBXSMM_PPC64LE_INSTR_NOP 0x60000000 /* NOP */

/* basic arithmetic opcodes */
#define LIBXSMM_PPC64LE_INSTR_ADDI 0x38000000 /* Add Immediate D-form */
#define LIBXSMM_PPC64LE_INSTR_RLDICR 0x78000004 /* Rotate Left Doubleword Immediate then Clear Right MD-form */
#define LIBXSMM_PPC64LE_INSTR_ADD 0x7c000214 /* Add XO-form */

/* logic opcodes */
#define LIBXSMM_PPC64LE_INSTR_BC 0x40000000 /* Branch Conditional B-form */
#define LIBXSMM_PPC64LE_INSTR_BLR  0x4e800020 /* Branch Unconditionally to LR */

#define LIBXSMM_PPC64LE_INSTR_ANDI 0x70000000 /* AND Immediate D-form */
#define LIBXSMM_PPC64LE_INSTR_ORI 0x60000000 /* OR Immediate D-form */
#define LIBXSMM_PPC64LE_INSTR_OR 0x7c000378 /* OR X-form */
#define LIBXSMM_PPC64LE_INSTR_NOR 0x7c0000f8 /* NOR X-form */
#define LIBXSMM_PPC64LE_INSTR_AND 0x7c000038 /* AND X-form */
#define LIBXSMM_PPC64LE_INSTR_NAND 0x7c0003b8 /* NAND X-form */

#define LIBXSMM_PPC64LE_INSTR_CMPI 0x2c000000 /* Compare Immediate D-form */

#define LIBXSMM_PPC64LE_INSTR_XXSLDWI 0xf0000010 /* VSX Vector Shift Left Double by Word Immediate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXPERMDI 0xf0000050 /* VSX Vector Permute Doubleword Immediate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXMRGHW 0xf0000090 /* VSX Vector Merge High Word XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXPERM 0xf00000d0 /* VSX Vector Permute XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXMRGLW 0xf0000190 /* VSX Vector Merge Low Word XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXPERMR 0xf00001d0 /* VSX Vector Permute Right-indexed XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLAND 0xf0000410 /* VSX Vector Logical AND XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLANDC 0xf0000450 /* VSX Vector Logical AND with Complement XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLOR 0xf0000490 /* VSX Vector Logical OR XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLXOR 0xf00004d0 /* VSX Vector Logical XOR XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLNOR 0xf0000510 /* VSX Vector Logical NOR XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLORC 0xf0000550 /* VSX Vector Logical OR with Complement XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLNAND 0xf0000590 /* VSX Vector Logical NAND XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXLEQV 0xf00005d0 /* VSX Vector Logical Equivalence XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTW 0xf0000290 /* VSX Vector Splat Word XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTIB 0xf00002d0 /* VSX Vector Splat Immediate Byte X-form */

#define LIBXSMM_PPC64LE_INSTR_XVCMPEQSP 0xf0000218 /* VSX Vector Compare Equal To Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCMPGTSP 0xf0000258 /* VSX Vector Compare Greater Than Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCMPGESP 0xf0000298 /* VSX Vector Compare Greater Than or Equal To Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCMPEQDP 0xf0000318 /* VSX Vector Compare Equal To Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCMPGTDP 0xf0000358 /* VSX Vector Compare Greater Than Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCMPGEDP 0xf0000398 /* VSX Vector Compare Greater Than or Equal To Double-Precision XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XSCMPEQDP 0xf0000018 /* VSX Scalar Compare Equal Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSCMPGTDP 0xf0000058 /* VSX Scalar Compare Greater Than Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSCMPGEDP 0xf0000098 /* VSX Scalar Compare Greater Than or Equal Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSCMPUDP 0xf0000118 /* VSX Scalar Compare Unordered Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSCMPODP 0xf0000158 /* VSX Scalar Compare Ordered Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSCMPEXPDP 0xf00001d8 /* VSX Scalar Compare Exponents Double-Precision XX3-form */


/* Load/Store opcodes */
#define LIBXSMM_PPC64LE_INSTR_LD 0xe8000000 /* Load Doubleword DS-form */
#define LIBXSMM_PPC64LE_INSTR_STD 0xf8000000 /* Store Doubleword DS-form  */
#define LIBXSMM_PPC64LE_INSTR_LFD 0xc8000000 /* Load Floating-Point Double D-form */
#define LIBXSMM_PPC64LE_INSTR_STFD 0xd8000000 /* Store Floating-Point Double D-form */
#define LIBXSMM_PPC64LE_INSTR_STFDP 0xf4000000 /* Store Floating-Point Double Pair DS-form */
#define LIBXSMM_PPC64LE_INSTR_STD 0xf8000000 /* Store Doubleword DS-form */
#define LIBXSMM_PPC64LE_INSTR_STDU 0xf8000001 /* Store Doubleword with Update DS-form */
#define LIBXSMM_PPC64LE_INSTR_STQ 0xf8000002 /* Store Quadword DS-form */

#define LIBXSMM_PPC64LE_INSTR_LVEBX 0x7c00000e /* Load Vector Element Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LVEHX 0x7c00004e /* Load Vector Element Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LVEWX 0x7c00008e /* Load Vector Element Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LVSL 0x7c00000c /* Load Vector for Shift Left Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LVSR 0x7c00004c /* Load Vector for Shift Right Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LVX 0x7c0000ce /* Load Vector Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LVXL 0x7c0002ce /* Load Vector Indexed Last X-form */
#define LIBXSMM_PPC64LE_INSTR_STVEBX 0x7c00010e /* Store Vector Element Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STVEHX 0x7c00014e /* Store Vector Element Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STVEWX 0x7c00018e /* Store Vector Element Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STVX 0x7c0001ce /* Store Vector Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STVXL 0x7c0003ce /* Store Vector Indexed Last X-form */

#define LIBXSMM_PPC64LE_INSTR_STXSD 0xf4000002 /* Store VSX Scalar Doubleword DS-form */
#define LIBXSMM_PPC64LE_INSTR_STXSSP 0xf4000003 /* Store VSX Scalar Single DS-form */
#define LIBXSMM_PPC64LE_INSTR_LXV 0xf4000001 /* Load VSX Vector DQ-form */
#define LIBXSMM_PPC64LE_INSTR_STXV 0xf4000005 /* Store VSX Vector DQ-form */

#define LIBXSMM_PPC64LE_INSTR_LXSIWZX 0x7c000018 /* Load VSX Scalar as Integer Word & Zero Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIWAX 0x7c000098 /* Load VSX Scalar as Integer Word Algebraic IndexedX-form */
#define LIBXSMM_PPC64LE_INSTR_STXSIWX 0x7c000118 /* Store VSX Scalar as Integer Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVX 0x7c000218 /* Load VSX Vector Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVDSX 0x7c000298 /* Load VSX Vector Doubleword & Splat Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVWSX 0x7c0002d8 /* Load VSX Vector Word & Splat Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVX 0x7c000318 /* Store VSX Vector Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSSPX 0x7c000418 /* Load VSX Scalar Single-Precision Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSDX 0x7c000498 /* Load VSX Scalar Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSSPX 0x7c000518 /* Store VSX Scalar Single-Precision Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSDX 0x7c000598 /* Store VSX Scalar Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVW4X 0x7c000618 /* Load VSX Vector Word*4 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVH8X 0x7c000658 /* Load VSX Vector Halfword*8 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVD2X 0x7c000698 /* Load VSX Vector Doubleword*2 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVB16X 0x7c0006d8 /* Load VSX Vector Byte*16 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVW4X 0x7c000718 /* Store VSX Vector Word*4 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVH8X 0x7c000758 /* Store VSX Vector Halfword*8 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVD2X 0x7c000798 /* Store VSX Vector Doubleword*2 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVB16X 0x7c0007d8 /* Store VSX Vector Byte*16 Indexed X-form */

#define LIBXSMM_PPC64LE_INSTR_LXVRBX 0x7c00001a /* Load VSX Vector Rightmost Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRHX 0x7c00005a /* Load VSX Vector Rightmost Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRWX 0x7c00009a /* Load VSX Vector Rightmost Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRDX 0x7c0000da /* Load VSX Vector Rightmost Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRBX 0x7c00011a /* Store VSX Vector Rightmost Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRHX 0x7c00015a /* Store VSX Vector Rightmost Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRWX 0x7c00019a /* Store VSX Vector Rightmost Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRDX 0x7c0001da /* Store VSX Vector Rightmost Doubleword Indexed X-form */

#define LIBXSMM_PPC64LE_INSTR_LXVL 0x7c00021a /* Load VSX Vector with Length X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVLL 0x7c00025a /* Load VSX Vector with Length Left-justified X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVPX 0x7c00029a /* Load VSX Vector Paired Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVL 0x7c00031a /* Store VSX Vector with Length X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVLL 0x7c00035a /* Store VSX Vector with Length Left-justified X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVPX 0x7c00039a /* Store VSX Vector Paired Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIBZX 0x7c00061a /* Load VSX Scalar as Integer Byte & Zero Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIHZX 0x7c00065a /* Load VSX Scalar as Integer Halfword & Zero Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSIBX 0x7c00071a /* Store VSX Scalar as Integer Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSIHX 0x7c00075a /* Store VSX Scalar as Integer Halfword Indexed X-form */

#define LIBXSMM_PPC64LE_INSTR_FCMPU 0xfc000000 /* Floating Compare Unordered X-form */
#define LIBXSMM_PPC64LE_INSTR_FCMPO 0xfc000040 /* Floating Compare Ordered X-form */
#define LIBXSMM_PPC64LE_INSTR_MCRFS 0xfc000080 /* Move to Condition Register from FPSCR X-form */
#define LIBXSMM_PPC64LE_INSTR_MFVSCR 0x10000604 /* Move From Vector Status and Control Register VX-form */
#define LIBXSMM_PPC64LE_INSTR_MTVSCR 0x10000644 /* Move To Vector Status and Control Register VX-form */
#define LIBXSMM_PPC64LE_INSTR_MTSPR 0x7c0003a6 /* Move To Special Purpose Register XFX-form */


/* Convertion opcode */
#define LIBXSMM_PPC64LE_INSTR_XSCVUXDSP 0xf00004a0 /* VSX Scalar Convert with round Unsigned Doubleword to Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSXDSP 0xf00004e0 /* VSX Scalar Convert with round Signed Doubleword to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVUXDDP 0xf00005a0 /* VSX Scalar Convert with round Unsigned Doubleword to Double-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSXDDP 0xf00005e0 /* VSX Scalar Convert with round Signed Doubleword to Double-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPUXDS 0xf0000520 /* VSX Scalar Convert with round to zero Double-Precision to Unsigned Doubleword format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSXDS 0xf0000560 /* VSX Scalar Convert with round to zero Double-Precision to Signed Doubleword format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPUXWS 0xf0000120 /* VSX Scalar Convert with round to zero Double-Precision to Unsigned Word format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSXWS 0xf0000160 /* VSX Scalar Convert with round to zero Double-Precision to Signed Word format XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSCVHPDP 0xf010056c /* VSX Scalar Convert Half-Precision to Double-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPHP 0xf011056c /* VSX Scalar Convert with round Double-Precision to Half-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSPN 0xf000042c /* VSX Scalar Convert Scalar Single-Precision to Vector Single-Precision format Non-signalling XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSPDPN 0xf000052c /* VSX Scalar Convert Single-Precision to Double-Precision format Non-signalling XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSP 0xf0000424 /* VSX Scalar Convert with round Double-Precision to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSPDP 0xf0000524 /* VSX Scalar Convert Single-Precision to Double-Precision format XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XVCVUXWDP 0xf00003a0 /* VSX Vector Convert Unsigned Word to Double-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXWDP 0xf00003e0 /* VSX Vector Convert Signed Word to Double-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVBF16SP 0xf010076c /* VSX Vector Convert bfloat16 to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPBF16 0xf011076c /* VSX Vector Convert with round Single-Precision to bfloat16 format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVHPSP 0xf018076c /* VSX Vector Convert Half-Precision to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPHP 0xf019076c /* VSX Vector Convert with round Single-Precision to Half-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPSP 0xf0000624 /* VSX Vector Convert with round Double-Precision to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPDP 0xf0000724 /* VSX Vector Convert Single-Precision to Double-Precision format XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XVCVSPUXWS 0xf0000220 /* VSX Vector Convert with round to zero Single-Precision to Unsigned Word format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPSXWS 0xf0000260 /* VSX Vector Convert with round to zero Single-Precision to Signed Word format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXWSP 0xf00002a0 /* VSX Vector Convert with round Unsigned Word to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXWSP 0xf00002e0 /* VSX Vector Convert with round Signed Word to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPUXWS 0xf0000320 /* VSX Vector Convert with round to zero Double-Precision to Unsigned Word format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPSXWS 0xf0000360 /* VSX Vector Convert with round to zero Double-Precision to Signed Word format XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XVCVSPUXDS 0xf0000620 /* VSX Vector Convert with round to zero Single-Precision to Unsigned Doubleword format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPSXDS 0xf0000660 /* VSX Vector Convert with round to zero Single-Precision to Signed Doubleword format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXDSP 0xf00006a0 /* VSX Vector Convert with round Unsigned Doubleword to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXDSP 0xf00006e0 /* VSX Vector Convert with round Signed Doubleword to Single-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPUXDS 0xf0000720 /* VSX Vector Convert with round to zero Double-Precision to Unsigned Doubleword format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPSXDS 0xf0000760 /* VSX Vector Convert with round to zero Double-Precision to Signed Doubleword format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXDDP 0xf00007a0 /* VSX Vector Convert with round Unsigned Doubleword to Double-Precision format XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXDDP 0xf00007e0 /* VSX Vector Convert with round Signed Doubleword to Double-Precision format XX2-form */


/* rounding opcodes */
#define LIBXSMM_PPC64LE_INSTR_XSRSP 0xf0000464 /* VSX Scalar Round to Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPI 0xf0000124 /* VSX Scalar Round to Double-Precision Integer using round to Nearest Away XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIZ 0xf0000164 /* VSX Scalar Round to Double-Precision Integer using round toward Zero XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIP 0xf00001a4 /* VSX Scalar Round to Double-Precision Integer using round toward +Infinity XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIM 0xf00001e4 /* VSX Scalar Round to Double-Precision Integer using round toward -Infinity XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XVRSPI 0xf0000224 /* VSX Vector Round to Single-Precision Integer using round to Nearest Away XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIZ 0xf0000264 /* VSX Vector Round to Single-Precision Integer using round toward Zero XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIP 0xf00002a4 /* VSX Vector Round to Single-Precision Integer using round toward +Infinity XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIM 0xf00002e4 /* VSX Vector Round to Single-Precision Integer using round toward -Infinity XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPI 0xf0000324 /* VSX Vector Round to Double-Precision Integer using round to Nearest Away XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIZ 0xf0000364 /* VSX Vector Round to Double-Precision Integer using round toward Zero XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIP 0xf00003a4 /* VSX Vector Round to Double-Precision Integer using round toward +Infinity XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIM 0xf00003e4 /* VSX Vector Round to Double-Precision Integer using round toward -Infinity XX2-form */


/* min/max opcodes */
#define LIBXSMM_PPC64LE_INSTR_XSMAXCDP 0xf0000400 /* VSX Scalar Maximum Type-C Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMAXJDP 0xf0000480 /* VSX Scalar Maximum Type-J Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMAXDP 0xf0000500 /* VSX Scalar Maximum Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMAXSP 0xf0000600 /* VSX Vector Maximum Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMAXDP 0xf0000700 /* VSX Vector Maximum Double-Precision XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XSMINCDP 0xf0000440 /* VSX Scalar Minimum Type-C Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMINJDP 0xf00004c0 /* VSX Scalar Minimum Type-J Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMINDP 0xf0000540 /* VSX Scalar Minimum Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMINSP 0xf0000640 /* VSX Vector Minimum Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMINDP 0xf0000740 /* VSX Vector Minimum Double-Precision XX3-form */


/* ABS and negate opcodes */
#define LIBXSMM_PPC64LE_INSTR_XSABSDP 0xf0000564 /* VSX Scalar Absolute Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVABSSP 0xf0000664 /* VSX Vector Absolute Value Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVABSDP 0xf0000764 /* VSX Vector Absolute Value Double-Precision XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSNEGDP 0xf00005e4 /* VSX Scalar Negate Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVNEGSP 0xf00006e4 /* VSX Vector Negate Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVNEGDP 0xf00007e4 /* VSX Vector Negate Double-Precision XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSNABSDP 0xf00005a4 /* VSX Scalar Negative Absolute Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVNABSSP 0xf00006a4 /* VSX Vector Negative Absolute Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVNABSDP 0xf00007a4 /* VSX Vector Negative Absolute Double-Precision XX2-form */


/* SQRT(X), 1/X, 1/SQRT(X) opcodes */
#define LIBXSMM_PPC64LE_INSTR_XSRSQRTESP 0xf0000028 /* VSX Scalar Reciprocal Square Root Estimate Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSRSQRTEDP 0xf0000128 /* VSX Scalar Reciprocal Square Root Estimate Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSQRTESP 0xf0000228 /* VSX Vector Reciprocal Square Root Estimate Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSQRTEDP 0xf0000328 /* VSX Vector Reciprocal Square Root Estimate Double-Precision XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSRESP 0xf0000068 /* VSX Scalar Reciprocal Estimate Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSREDP 0xf0000168 /* VSX Scalar Reciprocal Estimate Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRESP 0xf0000268 /* VSX Vector Reciprocal Estimate Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVREDP 0xf0000368 /* VSX Vector Reciprocal Estimate Double-Precision XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSSQRTSP 0xf000002c /* VSX Scalar Square Root Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSSQRTDP 0xf000012c /* VSX Scalar Square Root Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVSQRTSP 0xf000022c /* VSX Vector Square Root Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVSQRTDP 0xf000032c /* VSX Vector Square Root Double-Precision XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSRDPIC 0xf00001ac /* VSX Scalar Round to Double-Precision Integer Exact using Current rounding mode XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIC 0xf00003ac /* VSX Vector Round to Double-Precision Integer Exact using Current rounding mode XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIC 0xf00002ac /* VSX Vector Round to Single-Precision Integer Exact using Current rounding mode XX2-form */


/* Masking, insertion and extraction opcodes */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVBM 0xf0000728 /* VSX Vector Generate PCV from Byte Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVWM 0xf0000768 /* VSX Vector Generate PCV from Word Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVHM 0xf000072a /* VSX Vector Generate PCV from Halfword Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVDM 0xf000076a /* VSX Vector Generate PCV from Doubleword Mask X-form */

#define LIBXSMM_PPC64LE_INSTR_XSXEXPDP 0xf000056c /* VSX Scalar Extract Exponent Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVXEXPDP 0xf000076c /* VSX Vector Extract Exponent Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVXEXPSP 0xf008076c /* VSX Vector Extract Exponent Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSXSIGDP 0xf001056c /* VSX Scalar Extract Significand Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVXSIGDP 0xf001076c /* VSX Vector Extract Significand Double-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XVXSIGSP 0xf009076c /* VSX Vector Extract Significand Single-Precision XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XXEXTRACTUW 0xf0000294 /* VSX Vector Extract Unsigned Word XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XXBRH 0xf007076c /* VSX Vector Byte-Reverse Halfword XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRW 0xf00f076c /* VSX Vector Byte-Reverse Word XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRD 0xf017076c /* VSX Vector Byte-Reverse Doubleword XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRQ 0xf01f076c /* VSX Vector Byte-Reverse Quadword XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XSIEXPDP 0xf000072c /* VSX Scalar Insert Exponent Double-Precision X-form */
#define LIBXSMM_PPC64LE_INSTR_XVIEXPDP 0xf00007c0 /* VSX Vector Insert Exponent Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVIEXPSP 0xf00006c0 /* VSX Vector Insert Exponent Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXINSERTW 0xf00002d4 /* VSX Vector Insert Word XX2-form */

#define LIBXSMM_PPC64LE_INSTR_XXSEL 0xf0000030 /* VSX Vector Select XX4-form */

#define LIBXSMM_PPC64LE_INSTR_XSCPSGNDP 0xf0000580 /* VSX Scalar Copy Sign Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCPSGNDP 0xf0000780 /* VSX Vector Copy Sign Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVCPSGNSP 0xf0000680 /* VSX Vector Copy Sign Single-Precision XX3-form */


/* Prefixed masking, insertion and extraction opcodes */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVB 0x0500000084000000 /* VSX Vector Blend Variable Byte 8RR:XX4-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVD 0x0500000084000030 /* VSX Vector Blend Variable Doubleword 8RR:XX4-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVH 0x0500000084000010 /* VSX Vector Blend Variable Halfword 8RR:XX4-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVW 0x0500000084000020 /* VSX Vector Blend Variable Word 8RR:XX4-form */
#define LIBXSMM_PPC64LE_INSTR_XXEVAL 0x0500000088000010 /* VSX Vector Evaluate 8RR:XX4-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTI32DX 0x0500000080000000 /* VSX Vector Splat Immediate32 Doubleword Indexed 8RR:D-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTIDP 0x0500000080040000 /* VSX Vector Splat Immediate Double-Precision 8RR:D-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTIW 0x0500000080060000 /* VSX Vector Splat Immediate Word 8RR:D-form */


/* vector arithmetic opcodes */
#define LIBXSMM_PPC64LE_INSTR_XVADDDP 0xf0000300 /* VSX Vector Add Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVADDSP 0xf0000200 /* VSX Vector Add Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMULDP 0xf0000380 /* VSX Vector Multiply Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMULSP 0xf0000280 /* VSX Vector Multiply Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVSUBDP 0xf0000340 /* VSX Vector Subtract Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVSUBSP 0xf0000240 /* VSX Vector Subtract Single-Precision XX3-form */


/* FMA type opcodes */
#define LIBXSMM_PPC64LE_INSTR_XSMADDASP 0xf0000008 /* VSX Scalar Multiply-Add Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDMSP 0xf0000048 /* VSX Scalar Multiply-Add Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBASP 0xf0000088 /* VSX Scalar Multiply-Subtract Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBMSP 0xf00000c8 /* VSX Scalar Multiply-Subtract Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDADP 0xf0000108 /* VSX Scalar Multiply-Add Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDMDP 0xf0000148 /* VSX Scalar Multiply-Add Type-M Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBADP 0xf0000188 /* VSX Scalar Multiply-Subtract Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBMDP 0xf00001c8 /* VSX Scalar Multiply-Subtract Type-M Double-Precision XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVMADDASP 0xf0000208 /* VSX Vector Multiply-Add Type-A Single-Precision XX3-form T = T + A*B */
#define LIBXSMM_PPC64LE_INSTR_XVMADDMSP 0xf0000248 /* VSX Vector Multiply-Add Type-M Single-Precision XX3-form T = B + A*T */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBASP 0xf0000288 /* VSX Vector Multiply-Subtract Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBMSP 0xf00002c8 /* VSX Vector Multiply-Subtract Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMADDADP 0xf0000308 /* VSX Vector Multiply-Add Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMADDMDP 0xf0000348 /* VSX Vector Multiply-Add Type-M Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBADP 0xf0000388 /* VSX Vector Multiply-Subtract Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBMDP 0xf00003c8 /* VSX Vector Multiply-Subtract Type-M Double-Precision XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XSNMADDASP 0xf0000408 /* VSX Scalar Negative Multiply-Add Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDMSP 0xf0000448 /* VSX Scalar Negative Multiply-Add Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBASP 0xf0000488 /* VSX Scalar Negative Multiply-Subtract Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBMSP 0xf00004c8 /* VSX Scalar Negative Multiply-Subtract Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDADP 0xf0000508 /* VSX Scalar Negative Multiply-Add Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDMDP 0xf0000548 /* VSX Scalar Negative Multiply-Add Type-M Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBADP 0xf0000588 /* VSX Scalar Negative Multiply-Subtract Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBMDP 0xf00005c8 /* VSX Scalar Negative Multiply-Subtract Type-M Double-Precision XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVNMADDASP 0xf0000608 /* VSX Vector Negative Multiply-Add Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDMSP 0xf0000648 /* VSX Vector Negative Multiply-Add Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBASP 0xf0000688 /* VSX Vector Negative Multiply-Subtract Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBMSP 0xf00006c8 /* VSX Vector Negative Multiply-Subtract Type-M Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDADP 0xf0000708 /* VSX Vector Negative Multiply-Add Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDMDP 0xf0000748 /* VSX Vector Negative Multiply-Add Type-M Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBADP 0xf0000788 /* VSX Vector Negative Multiply-Subtract Type-A Double-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBMDP 0xf00007c8 /* VSX Vector Negative Multiply-Subtract Type-M Double-Precision XX3-form */


/* MMA opcodes */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2 0xec000198 /* VSX Vector bfloat16 GER (Rank-2 Update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2NN 0xec000790 /* VSX Vector bfloat16 GER (Rank-2 Update) Negative multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2NP 0xec000390 /* VSX Vector bfloat16 GER (Rank-2 Update) Negative multiply, Positive accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2PN 0xec000590 /* VSX Vector bfloat16 GER (Rank-2 Update) Positive multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2PP 0xec000190 /* VSX Vector bfloat16 GER (Rank-2 Update) Positive multiply, Positive accumulate XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVF16GER2 0xec000098 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2NN 0xec000690 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2NP 0xec000290 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Positive accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2PN 0xec000490 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2PP 0xec000090 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Positive accumulate XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVF32GER 0xec0000d8 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERNN 0xec0006d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERNP 0xec0002d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERPN 0xec0004d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERPP 0xec0000d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVF64GER 0xec0001d8 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERNN 0xec0007d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERNP 0xec0003d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERPN 0xec0005d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERPP 0xec0001d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVI16GER2 0xec000258 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2PP 0xec000358 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) Positive multiply, Positive accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2S 0xec000158 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2SPP 0xec000150 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation Positive multiply, Positive accumulate XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVI4GER8 0xec000118 /* VSX Vector 4-bit Signed Integer GER (rank-8 update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVI4GER8PP 0xec000110 /* VSX Vector 4-bit Signed Integer GER (rank-8 update) Positive multiply, Positive accumulate XX3-form */

#define LIBXSMM_PPC64LE_INSTR_XVI8GER4 0xec000018 /* VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVI8GER4PP 0xec000010 /* VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) Positive multiply, Positive accumulate XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVI8GER4SPP 0xec000318 /* VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) with Saturate Positive multiply, Positive accumulate XX3-form */

/* Prefixed MMA opcodes */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2 0x07900000ec000198 /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2NN 0x07900000ec000790 /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Negative multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2NP 0x07900000ec000390 /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Negative multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2PN 0x07900000ec000590 /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Positive multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2PP 0x07900000ec000190 /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2 0x07900000ec000098 /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2NN 0x07900000ec000690 /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2NP 0x07900000ec000290 /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2PN 0x07900000ec000490 /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2PP 0x07900000ec000090 /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GER 0x07900000ec0000d8 /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERNN 0x07900000ec0006d0 /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERNP 0x07900000ec0002d0 /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERPN 0x07900000ec0004d0 /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERPP 0x07900000ec0000d0 /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GER 0x07900000ec0001d8 /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERNN 0x07900000ec0007d0 /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERNP 0x07900000ec0003d0 /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERPN 0x07900000ec0005d0 /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERPP 0x07900000ec0001d0 /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2 0x07900000ec000258 /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2PP 0x07900000ec000358 /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2S 0x07900000ec000158 /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2SPP 0x07900000ec000150 /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI4GER8 0x07900000ec000118 /* Prefixed Masked VSX Vector 4-bit Signed Integer GER (rank-8 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI4GER8PP 0x07900000ec000110 /* Prefixed Masked VSX Vector 4-bit Signed Integer GER (rank-8 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI8GER4 0x07900000ec000018 /* Prefixed Masked VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI8GER4PP 0x07900000ec000010 /* Prefixed Masked VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI8GER4SPP 0x07900000ec000318 /* Prefixed Masked VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) with Saturation Positive multiply, Positive accumulate MMIRR:XX3-form */


/* MMA Register opcodes */
#define LIBXSMM_PPC64LE_INSTR_XXMFACC 0x7c000162 /* VSX Move From Accumulator X-form */
#define LIBXSMM_PPC64LE_INSTR_XXMTACC 0x7c010162 /* VSX Move To Accumulator X-form */
#define LIBXSMM_PPC64LE_INSTR_XXSETACCZ 0x7c030162 /* VSX Set Accumulator to Zero X-form */


LIBXSMM_API_INTERN
libxsmm_ppc64le_reg libxsmm_ppc64le_reg_init();



LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_get_reg( libxsmm_ppc64le_reg * io_reg_tracker,
                                      unsigned int const     i_reg_type );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_set_reg( libxsmm_ppc64le_reg * io_reg_tracker,
                              unsigned int const    io_reg_type,
                              unsigned int const    i_reg,
                              unsigned int const    i_value );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_free_reg( libxsmm_ppc64le_reg * io_reg_tracker,
                               unsigned int const    i_reg_type,
                               unsigned int const    i_reg );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_b_form( unsigned int  i_instr,
                                           unsigned char i_bo,
                                           unsigned char i_bi,
                                           unsigned int  i_bd );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_b_form_al( unsigned int  i_instr,
                                              unsigned char i_bo,
                                              unsigned char i_bi,
                                              unsigned int  i_bd,
                                              unsigned char i_aa,
                                              unsigned char i_lk );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_d_form( unsigned int  i_instr,
                                           unsigned char i_t,
                                           unsigned char i_a,
                                           unsigned int  i_d );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_d_form_bf( unsigned int  i_instr,
                                              unsigned char i_bf,
                                              unsigned char i_l,
                                              unsigned char i_a,
                                              unsigned int  i_d );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_dq_form_x( unsigned int  i_instr,
                                              unsigned char i_t,
                                              unsigned char i_ra,
                                              unsigned int  i_dq,
                                              unsigned char i_x );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_ds_form( unsigned int  i_instr,
                                            unsigned char i_s,
                                            unsigned char i_a,
                                            unsigned int  i_d );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_md_form( unsigned int  i_instr,
                                            unsigned char i_rs,
                                            unsigned char i_ra,
                                            unsigned char i_sh,
                                            unsigned char i_m,
                                            unsigned char i_sh2,
                                            unsigned char i_rc );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_vx_form_vrb( unsigned int  i_instr,
                                                unsigned char i_vrb );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_vx_form_vrt( unsigned int  i_instr,
                                                unsigned char i_vrt );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form( unsigned int  i_instr,
                                           unsigned char i_t,
                                           unsigned char i_a,
                                           unsigned char i_b,
                                           unsigned char i_x );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_3( unsigned int  i_instr,
                                             unsigned char i_a );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_33( unsigned int  i_instr,
                                              unsigned char i_bf,
                                              unsigned char i_bfa );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_355( unsigned int  i_instr,
                                               unsigned char i_bf,
                                               unsigned char i_a,
                                               unsigned char i_b );

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_4155( unsigned int  i_instr,
                                                unsigned char i_t,
                                                unsigned char i_x,
                                                unsigned char i_a,
                                                unsigned char i_b );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_555( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_581( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_imm,
                                               unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xfx_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned int  i_r );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_2( unsigned int  i_instr,
                                               unsigned char i_rt,
                                               unsigned char i_b,
                                               unsigned char i_bx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_3( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_b,
                                               unsigned char i_bx,
                                               unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_4( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_b,
                                               unsigned char i_uim,
                                               unsigned char i_bx,
                                               unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_0( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_ax,
                                               unsigned char i_bx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_3( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_w,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_5( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_rc,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_6( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx4_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned char i_a,
                                             unsigned char i_b,
                                             unsigned char i_c,
                                             unsigned char i_cx,
                                             unsigned char i_ax,
                                             unsigned char i_bx,
                                             unsigned char i_tx );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_0_wrapper( unsigned int i_instr );

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_1_wrapper( unsigned int i_instr,
                                              unsigned int i_1 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_2_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_3_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_4_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_5_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_6_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_7_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6,
                                              unsigned int i_7 );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_8_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6,
                                              unsigned int i_7,
                                              unsigned int i_8 );


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_4l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4 );


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_5l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5 );


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_7l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5,
                                                unsigned int i_6,
                                                unsigned int i_7 );


LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_8l_wrapper( unsigned long i_instr,
                                                unsigned int i_1,
                                                unsigned int i_2,
                                                unsigned int i_3,
                                                unsigned int i_4,
                                                unsigned int i_5,
                                                unsigned int i_6,
                                                unsigned int i_7,
                                                unsigned int i_8 );


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
                                                unsigned int i_9 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr( libxsmm_generated_code * io_generated_code,
                            unsigned int             i_instr);


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_1( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0  );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_2( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_3( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_4( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_5( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_6( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4,
                              unsigned int             i_5 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_7( libxsmm_generated_code * io_generated_code,
                              unsigned int             i_instr,
                              unsigned int             i_0,
                              unsigned int             i_1,
                              unsigned int             i_2,
                              unsigned int             i_3,
                              unsigned int             i_4,
                              unsigned int             i_5,
                              unsigned int             i_6 );


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
                              unsigned int             i_7 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_4( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_5( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3,
                                     unsigned int             i_4 );



LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_7( libxsmm_generated_code * io_generated_code,
                                     unsigned long            i_instr,
                                     unsigned int             i_0,
                                     unsigned int             i_1,
                                     unsigned int             i_2,
                                     unsigned int             i_3,
                                     unsigned int             i_4,
                                     unsigned int             i_5,
                                     unsigned int             i_6 );


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
                                     unsigned int             i_7 );


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
                                     unsigned int             i_8 );


/**
 * Opens the inline assembly section / jit stream.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_open_stream( libxsmm_generated_code * io_generated_code,
                                        libxsmm_ppc64le_reg    * io_reg_tracker );

/**
 * Closes the inline assembly section / jit stream.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_close_stream( libxsmm_generated_code * io_generated_code,
                                         libxsmm_ppc64le_reg    * io_reg_tracker );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_bytes( libxsmm_generated_code * io_generated_code,
                                          libxsmm_datatype const   i_datatype );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( libxsmm_generated_code * io_generated_code,
                                                      libxsmm_ppc64le_reg    * io_reg_tracker,
                                                      unsigned int           * io_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x2_inplace( libxsmm_generated_code * io_generated_code,
                                                      libxsmm_ppc64le_reg    * io_reg_tracker,
                                                      unsigned int           * io_v );


/**
 * Generates a label to which one can jump back and pushes it on the loop label stack.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param io_loop_label_tracker data structure to handle loop labels, nested loops are supported, but not overlapping loops.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_register_jump_back_label( libxsmm_generated_code     * io_generated_code,
                                                     libxsmm_loop_label_tracker * io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and jumps there based on the condition.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gprs  GPR which is compared to zero.
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_cond_jump_back_to_label( libxsmm_generated_code     * io_generated_code,
                                                    unsigned int                 i_gpr,
                                                    libxsmm_loop_label_tracker * io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and adds a jump based on the value of the count register, i.e., ctr==0.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_cond_jump_back_to_label_ctr( libxsmm_generated_code     * io_generated_code,
                                                        libxsmm_loop_label_tracker * io_loop_label_tracker );

#endif
