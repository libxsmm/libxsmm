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

LIBXSMM_EXTERN_C typedef struct libxsmm_ppc64le_blocking {
  unsigned int vector_len_a;
  unsigned int vector_len_b;
  unsigned int vector_len_c;
  unsigned int vector_len_comp;

  unsigned int comp_bytes;

  unsigned int block_m;
  unsigned int block_n;
  unsigned int block_k;

  unsigned int n_block_m_full;
  unsigned int n_block_n_full;
  unsigned int n_block_k_full;

  unsigned int n_reg_a;
  unsigned int n_reg_b;
  unsigned int n_reg_c;
  unsigned int n_acc_c;

  unsigned int reg_lda;
  unsigned int reg_ldb;
  unsigned int reg_ldc;
} libxsmm_ppc64le_blocking;


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
#define LIBXSMM_PPC64LE_ACC_WIDTH 512
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
#define LIBXSMM_PPC64LE_VR  2
#define LIBXSMM_PPC64LE_VSR 3
#define LIBXSMM_PPC64LE_ACC 4

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

/* Special instructions */
#define LIBXSMM_PPC64LE_INSTR_NOP 0x60000000 /* NOP */
#define LIBXSMM_PPC64LE_INSTR_BLR 0x4e800020 /* Branch Unconditionally to LR */

#define LIBXSMM_PPC64LE_32FMASK 0xfc1f07ff /* 32-bit opcode form index mask */
#define LIBXSMM_PPC64LE_64FMASK 0xfff00000ffffffffUL /* 64-bit opcode form index mask */

#define LIBXSMM_PPC64LE_FORM_VX_VRB 0x00000800
#define LIBXSMM_PPC64LE_FORM_X_4155 0x00001000
#define LIBXSMM_PPC64LE_FORM_DS 0x00001800
#define LIBXSMM_PPC64LE_FORM_XX4 0x00002000
#define LIBXSMM_PPC64LE_FORM_D 0x00002800
#define LIBXSMM_PPC64LE_FORM_X_555 0x00003000
#define LIBXSMM_PPC64LE_FORM_XX3 0x00003800
#define LIBXSMM_PPC64LE_FORM_XX2 0x00004000
#define LIBXSMM_PPC64LE_FORM_XX2_2 0x00004800
#define LIBXSMM_PPC64LE_FORM_B 0x00005000
#define LIBXSMM_PPC64LE_FORM_X_581 0x00005800
#define LIBXSMM_PPC64LE_FORM_X 0x00006000
#define LIBXSMM_PPC64LE_FORM_XX2_4 0x00006800
#define LIBXSMM_PPC64LE_FORM_X_3 0x00007000
#define LIBXSMM_PPC64LE_FORM_D_BF 0x00007800
#define LIBXSMM_PPC64LE_FORM_XX2_30 0x00008000
#define LIBXSMM_PPC64LE_FORM_XX3_3 0x00008800
#define LIBXSMM_PPC64LE_FORM_X_33 0x00009000
#define LIBXSMM_PPC64LE_FORM_XX3_6 0x00009800
#define LIBXSMM_PPC64LE_FORM_MD 0x0000a000
#define LIBXSMM_PPC64LE_FORM_XX2_3 0x0000a800
#define LIBXSMM_PPC64LE_FORM_VX_VRT 0x0000b000
#define LIBXSMM_PPC64LE_FORM_DQ_X 0x0000b800
#define LIBXSMM_PPC64LE_FORM_XX3_0 0x0000c000

#define LIBXSMM_PPC64LE_FORM_MMIRR_XX3_0_1 0x0000000100000000UL
#define LIBXSMM_PPC64LE_FORM_8LS_D 0x0000000200000000UL
#define LIBXSMM_PPC64LE_FORM_8RR_D_0_3 0x0000000300000000UL
#define LIBXSMM_PPC64LE_FORM_MMIRR_XX3_0_0 0x0000000400000000UL
#define LIBXSMM_PPC64LE_FORM_8RR_D_1_3 0x0000000500000000UL
#define LIBXSMM_PPC64LE_FORM_MMIRR_XX3 0x0000000600000000UL
#define LIBXSMM_PPC64LE_FORM_8RR_XX4_2 0x0000000700000000UL
#define LIBXSMM_PPC64LE_FORM_8RR_XX4_0 0x0000000800000000UL
#define LIBXSMM_PPC64LE_FORM_MMIRR_XX3_0_3 0x0000000900000000UL
#define LIBXSMM_PPC64LE_FORM_MLS_D 0x0000000a00000000UL


#define LIBXSMM_PPC64LE_INSTR_ADD 0x7c003214 /* Add X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_OR 0x7c003378 /* OR X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_NOR 0x7c0030f8 /* NOR X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_AND 0x7c003038 /* AND X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_NAND 0x7c0033b8 /* NAND X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_BC 0x40005000 /* Branch Conditional B-form */
#define LIBXSMM_PPC64LE_INSTR_LFD 0xc8002800 /* Load Floating-Point Double D-form */
#define LIBXSMM_PPC64LE_INSTR_STFD 0xd8002800 /* Store Floating-Point Double D-form */
#define LIBXSMM_PPC64LE_INSTR_ORI 0x60002800 /* OR Immediate D-form */
#define LIBXSMM_PPC64LE_INSTR_ADDI 0x38002800 /* Add Immediate D-form */
#define LIBXSMM_PPC64LE_INSTR_ANDI 0x70002800 /* AND Immediate D-form */
#define LIBXSMM_PPC64LE_INSTR_LD 0xe8001800 /* Load Doubleword DS-form */
#define LIBXSMM_PPC64LE_INSTR_CMPI 0x2c007800 /* Compare Immediate D(BF)-form */
#define LIBXSMM_PPC64LE_INSTR_RLDICR 0x7800a004 /* Rotate Left Doubleword Immediate then Clear Right MD-form */

#define LIBXSMM_PPC64LE_INSTR_XSMAXCDP 0xf0009c00 /* VSX Scalar Maximum Type-C Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMINCDP 0xf0009c40 /* VSX Scalar Minimum Type-C Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMAXDP 0xf0009d00 /* VSX Scalar Maximum Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMINDP 0xf0009d40 /* VSX Scalar Minimum Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCPSGNDP 0xf0009d80 /* VSX Scalar Copy Sign Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMAXSP 0xf0009e00 /* VSX Vector Maximum Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMINSP 0xf0009e40 /* VSX Vector Minimum Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCPSGNSP 0xf0009e80 /* VSX Vector Copy Sign Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVIEXPSP 0xf0009ec0 /* VSX Vector Insert Exponent Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMAXDP 0xf0009f00 /* VSX Vector Maximum Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMINDP 0xf0009f40 /* VSX Vector Minimum Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCPSGNDP 0xf0009f80 /* VSX Vector Copy Sign Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVIEXPDP 0xf0009fc0 /* VSX Vector Insert Exponent Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDASP 0xf0009808 /* VSX Scalar Multiply-Add Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDMSP 0xf0009848 /* VSX Scalar Multiply-Add Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBASP 0xf0009888 /* VSX Scalar Multiply-Subtract Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBMSP 0xf00098c8 /* VSX Scalar Multiply-Subtract Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDADP 0xf0009908 /* VSX Scalar Multiply-Add Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMADDMDP 0xf0009948 /* VSX Scalar Multiply-Add Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBADP 0xf0009988 /* VSX Scalar Multiply-Subtract Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSMSUBMDP 0xf00099c8 /* VSX Scalar Multiply-Subtract Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMADDASP 0xf0003a08 /* VSX Vector Multiply-Add Type-A Single-Precision XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XVMADDMSP 0xf0009a48 /* VSX Vector Multiply-Add Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBASP 0xf0009a88 /* VSX Vector Multiply-Subtract Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBMSP 0xf0009ac8 /* VSX Vector Multiply-Subtract Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMADDADP 0xf0009b08 /* VSX Vector Multiply-Add Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMADDMDP 0xf0009b48 /* VSX Vector Multiply-Add Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBADP 0xf0009b88 /* VSX Vector Multiply-Subtract Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMSUBMDP 0xf0009bc8 /* VSX Vector Multiply-Subtract Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDASP 0xf0009c08 /* VSX Scalar Negative Multiply-Add Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDMSP 0xf0009c48 /* VSX Scalar Negative Multiply-Add Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBASP 0xf0009c88 /* VSX Scalar Negative Multiply-Subtract Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBMSP 0xf0009cc8 /* VSX Scalar Negative Multiply-Subtract Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDADP 0xf0009d08 /* VSX Scalar Negative Multiply-Add Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMADDMDP 0xf0009d48 /* VSX Scalar Negative Multiply-Add Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBADP 0xf0009d88 /* VSX Scalar Negative Multiply-Subtract Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNMSUBMDP 0xf0009dc8 /* VSX Scalar Negative Multiply-Subtract Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDASP 0xf0009e08 /* VSX Vector Negative Multiply-Add Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDMSP 0xf0009e48 /* VSX Vector Negative Multiply-Add Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBASP 0xf0009e88 /* VSX Vector Negative Multiply-Subtract Type-A Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBMSP 0xf0009ec8 /* VSX Vector Negative Multiply-Subtract Type-M Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDADP 0xf0009f08 /* VSX Vector Negative Multiply-Add Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMADDMDP 0xf0009f48 /* VSX Vector Negative Multiply-Add Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBADP 0xf0009f88 /* VSX Vector Negative Multiply-Subtract Type-A Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNMSUBMDP 0xf0009fc8 /* VSX Vector Negative Multiply-Subtract Type-M Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXMRGHW 0xf0009890 /* VSX Vector Merge High Word XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXPERM 0xf00098d0 /* VSX Vector Permute XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXMRGLW 0xf0009990 /* VSX Vector Merge Low Word XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXPERMR 0xf00099d0 /* VSX Vector Permute Right-indexed XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLAND 0xf0009c10 /* VSX Vector Logical AND XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLANDC 0xf0009c50 /* VSX Vector Logical AND with Complement XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLOR 0xf0009c90 /* VSX Vector Logical OR XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLXOR 0xf0009cd0 /* VSX Vector Logical XOR XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLNOR 0xf0009d10 /* VSX Vector Logical NOR XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLORC 0xf0009d50 /* VSX Vector Logical OR with Complement XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLNAND 0xf0009d90 /* VSX Vector Logical NAND XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXLEQV 0xf0009dd0 /* VSX Vector Logical Equivalence XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSLDWI 0xf0008810 /* VSX Vector Shift Left Double by Word Immediate XX3(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXPERMDI 0xf0008850 /* VSX Vector Permute Doubleword Immediate XX3(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTW 0xf0006a90 /* VSX Vector Splat Word XX2(4)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTIB 0xf0005ad0 /* VSX Vector Splat Immediate Byte X(581)-form */
#define LIBXSMM_PPC64LE_INSTR_LXVKQ 0xf01f62d0 /* Load VSX Vector Special Value Quadword X-form */
#define LIBXSMM_PPC64LE_INSTR_XXINSERTW 0xf00042d4 /* VSX Vector Insert Word XX2-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPUXWS 0xf000a920 /* VSX Scalar Convert with round to zero Double-Precision to Unsigned Word format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSXWS 0xf000a960 /* VSX Scalar Convert with round to zero Double-Precision to Signed Word format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPUXWS 0xf000aa20 /* VSX Vector Convert with round to zero Single-Precision to Unsigned Word format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPSXWS 0xf000aa60 /* VSX Vector Convert with round to zero Single-Precision to Signed Word format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXWSP 0xf000aaa0 /* VSX Vector Convert with round Unsigned Word to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXWSP 0xf000aae0 /* VSX Vector Convert with round Signed Word to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPUXWS 0xf000ab20 /* VSX Vector Convert with round to zero Double-Precision to Unsigned Word format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPSXWS 0xf000ab60 /* VSX Vector Convert with round to zero Double-Precision to Signed Word format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXWDP 0xf000aba0 /* VSX Vector Convert Unsigned Word to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXWDP 0xf000abe0 /* VSX Vector Convert Signed Word to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVUXDSP 0xf000aca0 /* VSX Scalar Convert with round Unsigned Doubleword to Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSXDSP 0xf000ace0 /* VSX Scalar Convert with round Signed Doubleword to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPUXDS 0xf000ad20 /* VSX Scalar Convert with round to zero Double-Precision to Unsigned Doubleword format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSXDS 0xf000ad60 /* VSX Scalar Convert with round to zero Double-Precision to Signed Doubleword format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVUXDDP 0xf000ada0 /* VSX Scalar Convert with round Unsigned Doubleword to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSXDDP 0xf000ade0 /* VSX Scalar Convert with round Signed Doubleword to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPUXDS 0xf000ae20 /* VSX Vector Convert with round to zero Single-Precision to Unsigned Doubleword format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPSXDS 0xf000ae60 /* VSX Vector Convert with round to zero Single-Precision to Signed Doubleword format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXDSP 0xf000aea0 /* VSX Vector Convert with round Unsigned Doubleword to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXDSP 0xf000aee0 /* VSX Vector Convert with round Signed Doubleword to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPUXDS 0xf000af20 /* VSX Vector Convert with round to zero Double-Precision to Unsigned Doubleword format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPSXDS 0xf000af60 /* VSX Vector Convert with round to zero Double-Precision to Signed Doubleword format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVUXDDP 0xf000afa0 /* VSX Vector Convert with round Unsigned Doubleword to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSXDDP 0xf000afe0 /* VSX Vector Convert with round Signed Doubleword to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPI 0xf000a924 /* VSX Scalar Round to Double-Precision Integer using round to Nearest Away XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIZ 0xf000a964 /* VSX Scalar Round to Double-Precision Integer using round toward Zero XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIP 0xf000a9a4 /* VSX Scalar Round to Double-Precision Integer using round toward +Infinity XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIM 0xf000a9e4 /* VSX Scalar Round to Double-Precision Integer using round toward -Infinity XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPI 0xf000aa24 /* VSX Vector Round to Single-Precision Integer using round to Nearest Away XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIZ 0xf000aa64 /* VSX Vector Round to Single-Precision Integer using round toward Zero XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIP 0xf000aaa4 /* VSX Vector Round to Single-Precision Integer using round toward +Infinity XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIM 0xf000aae4 /* VSX Vector Round to Single-Precision Integer using round toward -Infinity XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPI 0xf000ab24 /* VSX Vector Round to Double-Precision Integer using round to Nearest Away XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIZ 0xf000ab64 /* VSX Vector Round to Double-Precision Integer using round toward Zero XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIP 0xf000aba4 /* VSX Vector Round to Double-Precision Integer using round toward +Infinity XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIM 0xf000abe4 /* VSX Vector Round to Double-Precision Integer using round toward -Infinity XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSP 0xf000ac24 /* VSX Scalar Convert with round Double-Precision to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRSP 0xf0008464 /* VSX Scalar Round to Single-Precision XX2(30-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSPDP 0xf000ad24 /* VSX Scalar Convert Single-Precision to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSABSDP 0xf000ad64 /* VSX Scalar Absolute Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNABSDP 0xf000ada4 /* VSX Scalar Negative Absolute Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSNEGDP 0xf000ade4 /* VSX Scalar Negate Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVDPSP 0xf000ae24 /* VSX Vector Convert with round Double-Prec   ision to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVABSSP 0xf000ae64 /* VSX Vector Absolute Value Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNABSSP 0xf000aea4 /* VSX Vector Negative Absolute Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNEGSP 0xf000aee4 /* VSX Vector Negate Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPDP 0xf000af24 /* VSX Vector Convert Single-Precision to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVABSDP 0xf000af64 /* VSX Vector Absolute Value Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNABSDP 0xf000afa4 /* VSX Vector Negative Absolute Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVNEGDP 0xf000afe4 /* VSX Vector Negate Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRSQRTESP 0xf000a828 /* VSX Scalar Reciprocal Square Root Estimate Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRESP 0xf000a868 /* VSX Scalar Reciprocal Estimate Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRSQRTEDP 0xf000a928 /* VSX Scalar Reciprocal Square Root Estimate Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSREDP 0xf000a968 /* VSX Scalar Reciprocal Estimate Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSQRTESP 0xf000aa28 /* VSX Vector Reciprocal Square Root Estimate Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRESP 0xf000aa68 /* VSX Vector Reciprocal Estimate Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSQRTEDP 0xf000ab28 /* VSX Vector Reciprocal Square Root Estimate Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVREDP 0xf000ab68 /* VSX Vector Reciprocal Estimate Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVBM 0xf0006728 /* VSX Vector Generate PCV from Byte Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVWM 0xf0006768 /* VSX Vector Generate PCV from Word Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVHM 0xf000672a /* VSX Vector Generate PCV from Halfword Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XXGENPCVDM 0xf000676a /* VSX Vector Generate PCV from Doubleword Mask X-form */
#define LIBXSMM_PPC64LE_INSTR_XSSQRTSP 0xf000a82c /* VSX Scalar Square Root Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSSQRTDP 0xf000a92c /* VSX Scalar Square Root Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSRDPIC 0xf000a9ac /* VSX Scalar Round to Double-Precision Integer exact using Current rounding mode XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVSQRTSP 0xf000aa2c /* VSX Vector Square Root Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRSPIC 0xf000aaac /* VSX Vector Round to Single-Precision Integer Exact using Current rounding mode XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVSQRTDP 0xf000ab2c /* VSX Vector Square Root Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVRDPIC 0xf000abac /* VSX Vector Round to Double-Precision Integer Exact using Current rounding mode XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPSPN 0xf000ac2c /* VSX Scalar Convert Scalar Single-Precision to Vector Single-Precision format Non-signalling XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVSPDPN 0xf000ad2c /* VSX Scalar Convert Single-Precision to Double-Precision format Non-signalling XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSXEXPDP 0xf0004d6c /* VSX Scalar Extract Exponent Double-Precision XX2(2)-form */
#define LIBXSMM_PPC64LE_INSTR_XSXSIGDP 0xf0014d6c /* VSX Scalar Extract Significand Double-Precision XX2(2)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVHPDP 0xf010ad6c /* VSX Scalar Convert Half-Precision to Double-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSCVDPHP 0xf011ad6c /* VSX Scalar Convert with round Double-Precision to Half-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVXEXPDP 0xf000af6c /* VSX Vector Extract Exponent Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVXSIGDP 0xf001af6c /* VSX Vector Extract Significand Double-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRH 0xf007af6c /* VSX Vector Byte-Reverse Halfword XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVXEXPSP 0xf008af6c /* VSX Vector Extract Exponent Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVXSIGSP 0xf009af6c /* VSX Vector Extract Significand Single-Precision XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRW 0xf00faf6c /* VSX Vector Byte-Reverse Word XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVBF16SP 0xf010af6c /* VSX Vector Convert bfloat16 to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPBF16 0xf011af6c /* VSX Vector Convert with round Single-Precision to bfloat16 format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRD 0xf017af6c /* VSX Vector Byte-Reverse Doubleword XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVHPSP 0xf018af6c /* VSX Vector Convert Half-Precision to Single-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XVCVSPHP 0xf019af6c /* VSX Vector Convert with round Single-Precision to Half-Precision format XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBRQ 0xf01faf6c /* VSX Vector Byte-Reverse Quadword XX2(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XSIEXPDP 0xf000672c /* VSX Scalar Insert Exponent Double-Precision X-form */
#define LIBXSMM_PPC64LE_INSTR_XXSEL 0xf0002030 /* VSX Vector Select XX4-form */
#define LIBXSMM_PPC64LE_INSTR_STFDP 0xf4001800 /* Store Floating-Point Double Pair DS-form */
#define LIBXSMM_PPC64LE_INSTR_STXSD 0xf4001802 /* Store VSX Scalar Doubleword DS-form */
#define LIBXSMM_PPC64LE_INSTR_STXSSP 0xf4001803 /* Store VSX Scalar Single DS-form */
#define LIBXSMM_PPC64LE_INSTR_LXV 0xf400b801 /* Load VSX Vector DQ(X)-form */
#define LIBXSMM_PPC64LE_INSTR_STXV 0xf400b805 /* Store VSX Vector DQ(X)-form */
#define LIBXSMM_PPC64LE_INSTR_STD 0xf8001800 /* Store Doubleword DS-form */
#define LIBXSMM_PPC64LE_INSTR_STDU 0xf8001801 /* Store Doubleword with Update DS-form */
#define LIBXSMM_PPC64LE_INSTR_STQ 0xf8001802 /* Store Quadword DS-form */
#define LIBXSMM_PPC64LE_INSTR_MCRFS 0xfc009080 /* Move to Condition Register from FPSCR X(33)-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2 0xec00c198 /* VSX Vector bfloat16 GER (Rank-2 Update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2NN 0xec00c790 /* VSX Vector bfloat16 GER (Rank-2 Update) Negative multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2NP 0xec00c390 /* VSX Vector bfloat16 GER (Rank-2 Update) Negative multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2PN 0xec00c590 /* VSX Vector bfloat16 GER (Rank-2 Update) Positive multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVBF16GER2PP 0xec00c190 /* VSX Vector bfloat16 GER (Rank-2 Update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2 0xec00c098 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2NN 0xec00c690 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2NP 0xec00c290 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2PN 0xec00c490 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF16GER2PP 0xec00c090 /* VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GER 0xec00c0d8 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERNN 0xec00c6d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERNP 0xec00c2d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERPN 0xec00c4d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF32GERPP 0xec00c0d0 /* VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GER 0xec00c1d8 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERNN 0xec00c7d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERNP 0xec00c3d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERPN 0xec00c5d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVF64GERPP 0xec00c1d0 /* VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2 0xec00c258 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2PP 0xec00c358 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2S 0xec00c158 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI16GER2SPP 0xec00c150 /* VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI4GER8 0xec00c118 /* VSX Vector 4-bit Signed Integer GER (rank-8 update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI4GER8PP 0xec00c110 /* VSX Vector 4-bit Signed Integer GER (rank-8 update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI8GER4 0xec00c018 /* VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI8GER4PP 0xec00c010 /* VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XVI8GER4SPP 0xec00c318 /* VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) with Saturate Positive multiply, Positive accumulate XX3(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XXMFACC 0x7c007162 /* VSX Move From Accumulator X(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXMTACC 0x7c017162 /* VSX Move To Accumulator X(3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSETACCZ 0x7c037162 /* VSX Set Accumulator to Zero X(3)-form */
#define LIBXSMM_PPC64LE_INSTR_LVEBX 0x7c00300e /* Load Vector Element Byte Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LVEHX 0x7c00304e /* Load Vector Element Halfword Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LVEWX 0x7c00308e /* Load Vector Element Word Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LVSL 0x7c00300c /* Load Vector for Shift Left Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LVSR 0x7c00304c /* Load Vector for Shift Right Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LVX 0x7c0030ce /* Load Vector Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LVXL 0x7c0032ce /* Load Vector Indexed Last X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_MFVSCR 0x1000b604 /* Move From Vector Status and Control Register VX(VRT)-form */
#define LIBXSMM_PPC64LE_INSTR_MTVSCR 0x10000e44 /* Move To Vector Status and Control Register VX(VRB)-form */
#define LIBXSMM_PPC64LE_INSTR_STVEBX 0x7c00310e /* Store Vector Element Byte Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_STVEHX 0x7c00314e /* Store Vector Element Halfword Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_STVEWX 0x7c00318e /* Store Vector Element Word Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_STVX 0x7c0031ce /* Store Vector Indexed X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_STVXL 0x7c0033ce /* Store Vector Indexed Last X(555)-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIWZX 0x7c006018 /* Load VSX Scalar as Integer Word & Zero Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIWAX 0x7c006098 /* Load VSX Scalar as Integer Word Algebraic Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSIWX 0x7c006118 /* Store VSX Scalar as Integer Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVX 0x7c006218 /* Load VSX Vector Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVDSX 0x7c006298 /* Load VSX Vector Doubleword & Splat Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVWSX 0x7c0062d8 /* Load VSX Vector Word & Splat Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVX 0x7c006318 /* Store VSX Vector Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSSPX 0x7c006418 /* Load VSX Scalar Single-Precision Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSDX 0x7c006498 /* Load VSX Scalar Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSSPX 0x7c006518 /* Store VSX Scalar Single-Precision Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSDX 0x7c006598 /* Store VSX Scalar Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVW4X 0x7c006618 /* Load VSX Vector Word*4 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVH8X 0x7c006658 /* Load VSX Vector Halfword*8 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVD2X 0x7c006698 /* Load VSX Vector Doubleword*2 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVB16X 0x7c0066d8 /* Load VSX Vector Byte*16 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVW4X 0x7c006718 /* Store VSX Vector Word*4 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVH8X 0x7c006758 /* Store VSX Vector Halfword*8 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVD2X 0x7c006798 /* Store VSX Vector Doubleword*2 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVB16X 0x7c0067d8 /* Store VSX Vector Byte*16 Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRBX 0x7c00601a /* Load VSX Vector Rightmost Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRHX 0x7c00605a /* Load VSX Vector Rightmost Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRWX 0x7c00609a /* Load VSX Vector Rightmost Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVRDX 0x7c0060da /* Load VSX Vector Rightmost Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRBX 0x7c00611a /* Store VSX Vector Rightmost Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRHX 0x7c00615a /* Store VSX Vector Rightmost Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRWX 0x7c00619a /* Store VSX Vector Rightmost Word Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVRDX 0x7c0061da /* Store VSX Vector Rightmost Doubleword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVL 0x7c00621a /* Load VSX Vector with Length X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVLL 0x7c00625a /* Load VSX Vector with Length Left-justified X-form */
#define LIBXSMM_PPC64LE_INSTR_LXVPX 0x7c00129a /* Load VSX Vector Paired Indexed X(4155)-form */
#define LIBXSMM_PPC64LE_INSTR_STXVL 0x7c00631a /* Store VSX Vector with Length X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVLL 0x7c00635a /* Store VSX Vector with Length Left-justified X-form */
#define LIBXSMM_PPC64LE_INSTR_STXVPX 0x7c00139a /* Store VSX Vector Paired Indexed X(4155)-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIBZX 0x7c00661a /* Load VSX Scalar as Integer Byte & Zero Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_LXSIHZX 0x7c00665a /* Load VSX Scalar as Integer Halfword & Zero Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSIBX 0x7c00671a /* Store VSX Scalar as Integer Byte Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_STXSIHX 0x7c00675a /* Store VSX Scalar as Integer Halfword Indexed X-form */
#define LIBXSMM_PPC64LE_INSTR_XVADDDP 0xf0009b00 /* VSX Vector Add Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVADDSP 0xf0009a00 /* VSX Vector Add Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMULDP 0xf0009b80 /* VSX Vector Multiply Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVMULSP 0xf0009a80 /* VSX Vector Multiply Single-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVSUBDP 0xf0009b40 /* VSX Vector Subtract Double-Precision XX3(6)-form */
#define LIBXSMM_PPC64LE_INSTR_XVSUBSP 0xf0009a40 /* VSX Vector Subtract Single-Precision XX3(6)-form */

#define LIBXSMM_PPC64LE_INSTR_PLXVP 0x4000002e8000000UL /* Prefixed Load VSX Vector Paired 8LS:D-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2 0x7900009ec000198UL /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2NN 0x7900009ec000790UL /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Negative multiply, Negative accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2NP 0x7900009ec000390UL /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Negative multiply, Positive accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2PN 0x7900009ec000590UL /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Positive multiply, Negative accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVBF16GER2PP 0x7900009ec000190UL /* Prefixed Masked VSX Vector bfloat16 GER (rank-2 update) Positive multiply, Positive accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2 0x7900009ec000098UL /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2NN 0x7900009ec000690UL /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Negative accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2NP 0x7900009ec000290UL /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Negative multiply, Positive accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2PN 0x7900009ec000490UL /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Negative accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF16GER2PP 0x7900009ec000090UL /* Prefixed Masked VSX Vector 16-bit Floating-Point GER (rank-2 update) Positive multiply, Positive accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GER 0x7900004ec0000d8UL /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) MMIRR:XX3(0,0)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERNN 0x7900004ec0006d0UL /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate MMIRR:XX3(0,0)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERNP 0x7900004ec0002d0UL /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate MMIRR:XX3(0,0)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERPN 0x7900004ec0004d0UL /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate MMIRR:XX3(0,0)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF32GERPP 0x7900004ec0000d0UL /* Prefixed Masked VSX Vector 32-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate MMIRR:XX3(0,0)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GER 0x7900001ec0001d8UL /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) MMIRR:XX3(0,1)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERNN 0x7900001ec0007d0UL /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Negative accumulate MMIRR:XX3(0,1)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERNP 0x7900001ec0003d0UL /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Negative multiply, Positive accumulate MMIRR:XX3(0,1)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERPN 0x7900001ec0005d0UL /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Negative accumulate MMIRR:XX3(0,1)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVF64GERPP 0x7900001ec0001d0UL /* Prefixed Masked VSX Vector 64-bit Floating-Point GER (rank-1 update) Positive multiply, Positive accumulate MMIRR:XX3(0,1)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2 0x7900009ec000258UL /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2PP 0x7900009ec000358UL /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) Positive multiply, Positive accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2S 0x7900009ec000158UL /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI16GER2SPP 0x7900009ec000150UL /* Prefixed Masked VSX Vector 16-bit Signed Integer GER (rank-2 update) with Saturation Positive multiply, Positive accumulate MMIRR:XX3(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI4GER8 0x7900006ec000118UL /* Prefixed Masked VSX Vector 4-bit Signed Integer GER (rank-8 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI4GER8PP 0x7900006ec000110UL /* Prefixed Masked VSX Vector 4-bit Signed Integer GER (rank-8 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI8GER4 0x7900006ec000018UL /* Prefixed Masked VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI8GER4PP 0x7900006ec000010UL /* Prefixed Masked VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_PMXVI8GER4SPP 0x7900006ec000318UL /* Prefixed Masked VSX Vector 8-bit Signed/Unsigned Integer GER (rank-4 update) with Saturation Positive multiply, Positive accumulate MMIRR:XX3-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVB 0x500000884000000UL /* VSX Vector Blend Variable Byte 8RR:XX4(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVD 0x500000884000030UL /* VSX Vector Blend Variable Doubleword 8RR:XX4(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVH 0x500000884000010UL /* VSX Vector Blend Variable Halfword 8RR:XX4(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XXBLENDVW 0x500000884000020UL /* VSX Vector Blend Variable Word 8RR:XX4(0)-form */
#define LIBXSMM_PPC64LE_INSTR_XXEVAL 0x500000788000010UL /* VSX Vector Evaluate 8RR:XX4(2)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTI32DX 0x500000380000000UL /* VSX Vector Splat Immediate32 Doubleword Indexed 8RR:D(0,3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTIDP 0x500000580040000UL /* VSX Vector Splat Immediate Double-Precision 8RR:D(1,3)-form */
#define LIBXSMM_PPC64LE_INSTR_XXSPLTIW 0x500000580060000UL /* VSX Vector Splat Immediate Word 8RR:D(1,3)-form */
#define LIBXSMM_PPC64LE_INSTR_PADDI 0x600000a38000000UL /* Prefixed Add Immediate MLS:D-form */


LIBXSMM_API_INTERN
libxsmm_ppc64le_reg libxsmm_ppc64le_reg_init();


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_get_reg( libxsmm_generated_code *io_generated_code,
                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                      unsigned int const      i_reg_type );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_get_sequential_reg( libxsmm_generated_code *io_generated_code,
                                         libxsmm_ppc64le_reg    *io_reg_tracker,
                                         unsigned int const      i_reg_type,
                                         unsigned int const      i_n,
                                         unsigned int           *o_reg );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_set_reg( libxsmm_generated_code *io_generated_code,
                              libxsmm_ppc64le_reg    *io_reg_tracker,
                              unsigned int const      i_reg_type,
                              unsigned int const      i_reg,
                              unsigned int const      i_value );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_free_reg( libxsmm_generated_code *io_generated_code,
                               libxsmm_ppc64le_reg    *io_reg_tracker,
                               unsigned int const      i_reg_type,
                               unsigned int const      i_reg );


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
unsigned long libxsmm_ppc64le_instr_d_form_mls( unsigned long i_instr,
                                                unsigned char i_r,
                                                unsigned int  i_si0,
                                                unsigned char i_t,
                                                unsigned char i_a,
                                                unsigned int  i_si1 );

LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_0_8rr3( unsigned long i_instr,
                                                   unsigned int  i_imm0,
                                                   unsigned char i_t,
                                                   unsigned char i_ix,
                                                   unsigned char i_tx,
                                                   unsigned int  i_imm1 );

LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_d_form_1_8rr3( unsigned long i_instr,
                                                   unsigned int  i_imm0,
                                                   unsigned char i_t,
                                                   unsigned char i_tx,
                                                   unsigned int  i_imm1 );

LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx3_form_0_mmirr0( unsigned long i_instr,
                                                       unsigned char i_xmsk,
                                                       unsigned char i_ymsk,
                                                       unsigned char i_at,
                                                       unsigned char i_a,
                                                       unsigned char i_b,
                                                       unsigned char i_ax,
                                                       unsigned char i_bx );

LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx3_form_0_mmirr1( unsigned long i_instr,
                                                       unsigned char i_xmsk,
                                                       unsigned char i_ymsk,
                                                       unsigned char i_at,
                                                       unsigned char i_a,
                                                       unsigned char i_b,
                                                       unsigned char i_ax,
                                                       unsigned char i_bx );

LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx3_form_0_mmirr3( unsigned long i_instr,
                                                       unsigned char i_pmsk,
                                                       unsigned char i_xmsk,
                                                       unsigned char i_ymsk,
                                                       unsigned char i_at,
                                                       unsigned char i_a,
                                                       unsigned char i_b,
                                                       unsigned char i_ax,
                                                       unsigned char i_bx );

LIBXSMM_API_INTERN
unsigned long libxsmm_ppc64le_instr_xx4_form_8rr0( unsigned long i_instr,
                                                   unsigned char i_t,
                                                   unsigned char i_a,
                                                   unsigned char i_b,
                                                   unsigned char i_c,
                                                   unsigned char i_cx,
                                                   unsigned char i_ax,
                                                   unsigned char i_bx,
                                                   unsigned char i_tx );

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
                                                   unsigned char i_tx );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_nop( libxsmm_generated_code *io_generated_code );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_blr( libxsmm_generated_code *io_generated_code );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_append( libxsmm_generated_code *io_generated_code,
                                   unsigned int            i_op );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_1( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0  );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_2( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_3( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_4( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_5( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3,
                              unsigned int            i_4 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_6( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3,
                              unsigned int            i_4,
                              unsigned int            i_5 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_7( libxsmm_generated_code *io_generated_code,
                              unsigned int            i_instr,
                              unsigned int            i_0,
                              unsigned int            i_1,
                              unsigned int            i_2,
                              unsigned int            i_3,
                              unsigned int            i_4,
                              unsigned int            i_5,
                              unsigned int            i_6 );


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
                              unsigned int            i_7 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_append( libxsmm_generated_code *io_generated_code,
                                          unsigned long           i_op );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_4( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3 );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_5( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4 );



LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_prefix_7( libxsmm_generated_code *io_generated_code,
                                     unsigned long           i_instr,
                                     unsigned int            i_0,
                                     unsigned int            i_1,
                                     unsigned int            i_2,
                                     unsigned int            i_3,
                                     unsigned int            i_4,
                                     unsigned int            i_5,
                                     unsigned int            i_6 );


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
                                     unsigned int            i_7 );


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
                                     unsigned int            i_8 );


/**
 * Opens the stream, sets up the stack frame according to the ABI and stores the
 * values of non-volatile register.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param io_reg_tracker pointer to register tracking structure.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_open_stream( libxsmm_generated_code *io_generated_code,
                                        libxsmm_ppc64le_reg    *io_reg_tracker );

/**
 * Colapses the stack frame, resetting non-volatile registers.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param io_reg_tracker pointer to register tracking structure.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_colapse_stack( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker );


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_bytes( libxsmm_generated_code *io_generated_code,
                                          libxsmm_datatype const  i_datatype );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x4_inplace( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      unsigned int           *io_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker,
                                          unsigned int           *i_v,
                                          unsigned int            i_n,
                                          unsigned int           *o_v,
                                          unsigned int            i_m );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x4( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x3( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_4x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_3x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_2x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f32_1x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x2_inplace( libxsmm_generated_code *io_generated_code,
                                                      libxsmm_ppc64le_reg    *io_reg_tracker,
                                                      unsigned int           *io_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64( libxsmm_generated_code *io_generated_code,
                                          libxsmm_ppc64le_reg    *io_reg_tracker,
                                          unsigned int           *i_v,
                                          unsigned int            i_n,
                                          unsigned int           *o_v,
                                          unsigned int            i_m );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_1x2( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_2x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_transpose_f64_1x1( libxsmm_generated_code *io_generated_code,
                                              libxsmm_ppc64le_reg    *io_reg_tracker,
                                              unsigned int           *i_v,
                                              unsigned int           *o_v );


/**
 * Generates a label to which one can jump back and pushes it on the loop label stack.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param io_loop_label_tracker data structure to handle loop labels, nested loops are supported, but not overlapping loops.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_register_jump_back_label( libxsmm_generated_code     *io_generated_code,
                                                     libxsmm_loop_label_tracker *io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and jumps there based on the condition.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gprs  GPR which is compared to zero.
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_cond_jump_back_to_label( libxsmm_generated_code     *io_generated_code,
                                                    unsigned int                i_gpr,
                                                    libxsmm_loop_label_tracker *io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and adds a jump based on the value of the count register, i.e., ctr==0.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label.
 **/
LIBXSMM_API_INTERN
void libxsmm_ppc64le_instr_cond_jump_back_to_label_ctr( libxsmm_generated_code     *io_generated_code,
                                                        libxsmm_loop_label_tracker *io_loop_label_tracker );

#endif
