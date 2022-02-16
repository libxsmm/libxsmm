/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_AARCH64_INSTRUCTIONS_H
#define GENERATOR_AARCH64_INSTRUCTIONS_H

#include "generator_common.h"
#include "../include/libxsmm_typedefs.h"

/* defining gp register mappings */
#define LIBXSMM_AARCH64_GP_REG_W0                         0
#define LIBXSMM_AARCH64_GP_REG_W1                         1
#define LIBXSMM_AARCH64_GP_REG_W2                         2
#define LIBXSMM_AARCH64_GP_REG_W3                         3
#define LIBXSMM_AARCH64_GP_REG_W4                         4
#define LIBXSMM_AARCH64_GP_REG_W5                         5
#define LIBXSMM_AARCH64_GP_REG_W6                         6
#define LIBXSMM_AARCH64_GP_REG_W7                         7
#define LIBXSMM_AARCH64_GP_REG_W8                         8
#define LIBXSMM_AARCH64_GP_REG_W9                         9
#define LIBXSMM_AARCH64_GP_REG_W10                       10
#define LIBXSMM_AARCH64_GP_REG_W11                       11
#define LIBXSMM_AARCH64_GP_REG_W12                       12
#define LIBXSMM_AARCH64_GP_REG_W13                       13
#define LIBXSMM_AARCH64_GP_REG_W14                       14
#define LIBXSMM_AARCH64_GP_REG_W15                       15
#define LIBXSMM_AARCH64_GP_REG_W16                       16
#define LIBXSMM_AARCH64_GP_REG_W17                       17
#define LIBXSMM_AARCH64_GP_REG_W18                       18
#define LIBXSMM_AARCH64_GP_REG_W19                       19
#define LIBXSMM_AARCH64_GP_REG_W20                       20
#define LIBXSMM_AARCH64_GP_REG_W21                       21
#define LIBXSMM_AARCH64_GP_REG_W22                       22
#define LIBXSMM_AARCH64_GP_REG_W23                       23
#define LIBXSMM_AARCH64_GP_REG_W24                       24
#define LIBXSMM_AARCH64_GP_REG_W25                       25
#define LIBXSMM_AARCH64_GP_REG_W26                       26
#define LIBXSMM_AARCH64_GP_REG_W27                       27
#define LIBXSMM_AARCH64_GP_REG_W28                       28
#define LIBXSMM_AARCH64_GP_REG_W29                       29
#define LIBXSMM_AARCH64_GP_REG_W30                       30
#define LIBXSMM_AARCH64_GP_REG_WSP                       31
#define LIBXSMM_AARCH64_GP_REG_WZR                       31
#define LIBXSMM_AARCH64_GP_REG_X0                        32
#define LIBXSMM_AARCH64_GP_REG_X1                        33
#define LIBXSMM_AARCH64_GP_REG_X2                        34
#define LIBXSMM_AARCH64_GP_REG_X3                        35
#define LIBXSMM_AARCH64_GP_REG_X4                        36
#define LIBXSMM_AARCH64_GP_REG_X5                        37
#define LIBXSMM_AARCH64_GP_REG_X6                        38
#define LIBXSMM_AARCH64_GP_REG_X7                        39
#define LIBXSMM_AARCH64_GP_REG_X8                        40
#define LIBXSMM_AARCH64_GP_REG_X9                        41
#define LIBXSMM_AARCH64_GP_REG_X10                       42
#define LIBXSMM_AARCH64_GP_REG_X11                       43
#define LIBXSMM_AARCH64_GP_REG_X12                       44
#define LIBXSMM_AARCH64_GP_REG_X13                       45
#define LIBXSMM_AARCH64_GP_REG_X14                       46
#define LIBXSMM_AARCH64_GP_REG_X15                       47
#define LIBXSMM_AARCH64_GP_REG_X16                       48
#define LIBXSMM_AARCH64_GP_REG_X17                       49
#define LIBXSMM_AARCH64_GP_REG_X18                       50
#define LIBXSMM_AARCH64_GP_REG_X19                       51
#define LIBXSMM_AARCH64_GP_REG_X20                       52
#define LIBXSMM_AARCH64_GP_REG_X21                       53
#define LIBXSMM_AARCH64_GP_REG_X22                       54
#define LIBXSMM_AARCH64_GP_REG_X23                       55
#define LIBXSMM_AARCH64_GP_REG_X24                       56
#define LIBXSMM_AARCH64_GP_REG_X25                       57
#define LIBXSMM_AARCH64_GP_REG_X26                       58
#define LIBXSMM_AARCH64_GP_REG_X27                       59
#define LIBXSMM_AARCH64_GP_REG_X28                       60
#define LIBXSMM_AARCH64_GP_REG_X29                       61
#define LIBXSMM_AARCH64_GP_REG_X30                       62
#define LIBXSMM_AARCH64_GP_REG_XSP                       63
#define LIBXSMM_AARCH64_GP_REG_XZR                       63
#define LIBXSMM_AARCH64_GP_REG_UNDEF                    127

/* defining asimd register mappings */
#define LIBXSMM_AARCH64_ASIMD_REG_V0                      0
#define LIBXSMM_AARCH64_ASIMD_REG_V1                      1
#define LIBXSMM_AARCH64_ASIMD_REG_V2                      2
#define LIBXSMM_AARCH64_ASIMD_REG_V3                      3
#define LIBXSMM_AARCH64_ASIMD_REG_V4                      4
#define LIBXSMM_AARCH64_ASIMD_REG_V5                      5
#define LIBXSMM_AARCH64_ASIMD_REG_V6                      6
#define LIBXSMM_AARCH64_ASIMD_REG_V7                      7
#define LIBXSMM_AARCH64_ASIMD_REG_V8                      8
#define LIBXSMM_AARCH64_ASIMD_REG_V9                      9
#define LIBXSMM_AARCH64_ASIMD_REG_V10                    10
#define LIBXSMM_AARCH64_ASIMD_REG_V11                    11
#define LIBXSMM_AARCH64_ASIMD_REG_V12                    12
#define LIBXSMM_AARCH64_ASIMD_REG_V13                    13
#define LIBXSMM_AARCH64_ASIMD_REG_V14                    14
#define LIBXSMM_AARCH64_ASIMD_REG_V15                    15
#define LIBXSMM_AARCH64_ASIMD_REG_V16                    16
#define LIBXSMM_AARCH64_ASIMD_REG_V17                    17
#define LIBXSMM_AARCH64_ASIMD_REG_V18                    18
#define LIBXSMM_AARCH64_ASIMD_REG_V19                    19
#define LIBXSMM_AARCH64_ASIMD_REG_V20                    20
#define LIBXSMM_AARCH64_ASIMD_REG_V21                    21
#define LIBXSMM_AARCH64_ASIMD_REG_V22                    22
#define LIBXSMM_AARCH64_ASIMD_REG_V23                    23
#define LIBXSMM_AARCH64_ASIMD_REG_V24                    24
#define LIBXSMM_AARCH64_ASIMD_REG_V25                    25
#define LIBXSMM_AARCH64_ASIMD_REG_V26                    26
#define LIBXSMM_AARCH64_ASIMD_REG_V27                    27
#define LIBXSMM_AARCH64_ASIMD_REG_V28                    28
#define LIBXSMM_AARCH64_ASIMD_REG_V29                    29
#define LIBXSMM_AARCH64_ASIMD_REG_V30                    30
#define LIBXSMM_AARCH64_ASIMD_REG_V31                    31
#define LIBXSMM_AARCH64_ASIMD_REG_UNDEF                 127

/* defining SVE register mappings */
#define LIBXSMM_AARCH64_SVE_REG_Z0                      0
#define LIBXSMM_AARCH64_SVE_REG_Z1                      1
#define LIBXSMM_AARCH64_SVE_REG_Z2                      2
#define LIBXSMM_AARCH64_SVE_REG_Z3                      3
#define LIBXSMM_AARCH64_SVE_REG_Z4                      4
#define LIBXSMM_AARCH64_SVE_REG_Z5                      5
#define LIBXSMM_AARCH64_SVE_REG_Z6                      6
#define LIBXSMM_AARCH64_SVE_REG_Z7                      7
#define LIBXSMM_AARCH64_SVE_REG_Z8                      8
#define LIBXSMM_AARCH64_SVE_REG_Z9                      9
#define LIBXSMM_AARCH64_SVE_REG_Z10                    10
#define LIBXSMM_AARCH64_SVE_REG_Z11                    11
#define LIBXSMM_AARCH64_SVE_REG_Z12                    12
#define LIBXSMM_AARCH64_SVE_REG_Z13                    13
#define LIBXSMM_AARCH64_SVE_REG_Z14                    14
#define LIBXSMM_AARCH64_SVE_REG_Z15                    15
#define LIBXSMM_AARCH64_SVE_REG_Z16                    16
#define LIBXSMM_AARCH64_SVE_REG_Z17                    17
#define LIBXSMM_AARCH64_SVE_REG_Z18                    18
#define LIBXSMM_AARCH64_SVE_REG_Z19                    19
#define LIBXSMM_AARCH64_SVE_REG_Z20                    20
#define LIBXSMM_AARCH64_SVE_REG_Z21                    21
#define LIBXSMM_AARCH64_SVE_REG_Z22                    22
#define LIBXSMM_AARCH64_SVE_REG_Z23                    23
#define LIBXSMM_AARCH64_SVE_REG_Z24                    24
#define LIBXSMM_AARCH64_SVE_REG_Z25                    25
#define LIBXSMM_AARCH64_SVE_REG_Z26                    26
#define LIBXSMM_AARCH64_SVE_REG_Z27                    27
#define LIBXSMM_AARCH64_SVE_REG_Z28                    28
#define LIBXSMM_AARCH64_SVE_REG_Z29                    29
#define LIBXSMM_AARCH64_SVE_REG_Z30                    30
#define LIBXSMM_AARCH64_SVE_REG_Z31                    31
#define LIBXSMM_AARCH64_SVE_REG_P0                      0
#define LIBXSMM_AARCH64_SVE_REG_P1                      1
#define LIBXSMM_AARCH64_SVE_REG_P2                      2
#define LIBXSMM_AARCH64_SVE_REG_P3                      3
#define LIBXSMM_AARCH64_SVE_REG_P4                      4
#define LIBXSMM_AARCH64_SVE_REG_P5                      5
#define LIBXSMM_AARCH64_SVE_REG_P6                      6
#define LIBXSMM_AARCH64_SVE_REG_P7                      7
#define LIBXSMM_AARCH64_SVE_REG_P8                      8
#define LIBXSMM_AARCH64_SVE_REG_P9                      9
#define LIBXSMM_AARCH64_SVE_REG_P10                    10
#define LIBXSMM_AARCH64_SVE_REG_P11                    11
#define LIBXSMM_AARCH64_SVE_REG_P12                    12
#define LIBXSMM_AARCH64_SVE_REG_P13                    13
#define LIBXSMM_AARCH64_SVE_REG_P14                    14
#define LIBXSMM_AARCH64_SVE_REG_P15                    15
#define LIBXSMM_AARCH64_SVE_REG_UNDEF                 127

/* special instruction */
#define LIBXSMM_AARCH64_INSTR_UNDEF                    9999

/* Descriptor
 * 4th byte
 *   --> from ISA manual
 * 3rd byte
 *   --> from ISA manual
 * 2nd byte
 *   --> from ISA manual
 * 1st byte
 *   7: SVE: predication required
 *   6-5: not used
 *   4:   tuple-type: ignore all sz bits, ignore shift bits for GP insturctions, if 4 & 3 is set for ASIMD -> immediate, e.g. shift is used
 *   3:   tuple-type: ignore second sz bit, vec register is destination (for UMOV/INS)
 *   2:   has immediate
 *   1-0: number of register operands
 */
/* define GP LD/ST instruction */
#define LIBXSMM_AARCH64_INSTR_GP_LDR_R           0xb8604803
#define LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF       0xb9400006
#define LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST      0xb8400406
#define LIBXSMM_AARCH64_INSTR_GP_LDR_I_PRE       0xb8400c06
#define LIBXSMM_AARCH64_INSTR_GP_STR_R           0xb8204803
#define LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF       0xb9000006
#define LIBXSMM_AARCH64_INSTR_GP_STR_I_POST      0xb8000406
#define LIBXSMM_AARCH64_INSTR_GP_STR_I_PRE       0xb8000c06
#define LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF       0x29400007
#define LIBXSMM_AARCH64_INSTR_GP_LDP_I_POST      0x28c00007
#define LIBXSMM_AARCH64_INSTR_GP_LDP_I_PRE       0x29c00007
#define LIBXSMM_AARCH64_INSTR_GP_LDNP_I_OFF      0x28400007
#define LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF       0x29000007
#define LIBXSMM_AARCH64_INSTR_GP_STP_I_POST      0x28800007
#define LIBXSMM_AARCH64_INSTR_GP_STP_I_PRE       0x29800007
#define LIBXSMM_AARCH64_INSTR_GP_STNP_I_OFF      0x28000007
/* define GP compute instructions */
/*#define LIBXSMM_AARCH64_INSTR_GP_ORR_I           0x22000006*/
#define LIBXSMM_AARCH64_INSTR_GP_ORR_SR          0x2a000007
/*#define LIBXSMM_AARCH64_INSTR_GP_AND_I           0x12000006*/
#define LIBXSMM_AARCH64_INSTR_GP_AND_SR          0x0a000007
/*#define LIBXSMM_AARCH64_INSTR_GP_EOR_I           0x52000006*/
#define LIBXSMM_AARCH64_INSTR_GP_EOR_SR          0x4a000007
/*#define LIBXSMM_AARCH64_INSTR_GP_LSL_I           0x53000006*/
#define LIBXSMM_AARCH64_INSTR_GP_LSL_SR          0x1ac02013
/*#define LIBXSMM_AARCH64_INSTR_GP_LSR_I           0x53007c06*/
#define LIBXSMM_AARCH64_INSTR_GP_LSR_SR          0x1ac02413
/*#define LIBXSMM_AARCH64_INSTR_GP_ASR_I           0x13000006*/
#define LIBXSMM_AARCH64_INSTR_GP_ASR_SR          0x1ac02813
#define LIBXSMM_AARCH64_INSTR_GP_ADD_I           0x11000006
#define LIBXSMM_AARCH64_INSTR_GP_ADD_SR          0x0b000007
#define LIBXSMM_AARCH64_INSTR_GP_SUB_I           0x51000006
#define LIBXSMM_AARCH64_INSTR_GP_SUB_SR          0x4b000007
#define LIBXSMM_AARCH64_INSTR_GP_MUL             0x1b007c13
#define LIBXSMM_AARCH64_INSTR_GP_UDIV            0x1ac00813
#define LIBXSMM_AARCH64_INSTR_GP_MOVZ            0x52800000
#define LIBXSMM_AARCH64_INSTR_GP_MOVK            0x72800000
#define LIBXSMM_AARCH64_INSTR_GP_MOVN            0x12800000
#define LIBXSMM_AARCH64_INSTR_GP_CBNZ            0x35000000
#define LIBXSMM_AARCH64_INSTR_GP_CBZ             0x34000000
/* define GP meta instructions which will to sequenes of aarch64 instructions */
#define LIBXSMM_AARCH64_INSTR_GP_META_ADD        0x00001000
#define LIBXSMM_AARCH64_INSTR_GP_META_SUB        0x00001001

/* define ASIMD LD/ST instructions */
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R        0x3c604803
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF    0x3d400006
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST   0x3c400406
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_PRE    0x3c400c06
#define LIBXSMM_AARCH64_INSTR_ASIMD_STR_R        0x3c204803
#define LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF    0x3d000006
#define LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST   0x3c000406
#define LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_PRE    0x3c000c06
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF    0x2d400007
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_POST   0x2cc00007
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_PRE    0x2dc00007
#define LIBXSMM_AARCH64_INSTR_ASIMD_LDNP_I_OFF   0x2c400007
#define LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF    0x2d000007
#define LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_POST   0x2c800007
#define LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_PRE    0x2d800007
#define LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF   0x2c000007
#define LIBXSMM_AARCH64_INSTR_ASIMD_LD1R         0x0d40c002
#define LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST  0x0dc0c003
#define LIBXSMM_AARCH64_INSTR_ASIMD_LD1_I_POST   0x0ddf8002
#define LIBXSMM_AARCH64_INSTR_ASIMD_LD1_R_POST   0x0dc08003

/* ASIMD <-> GPR moves */
#define LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V      0x4e001c1e
#define LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G     0x0e003c16
#define LIBXSMM_AARCH64_INSTR_ASIMD_DUP_HALF     0x0e000c1e
#define LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL     0x4e000c1e

/* define ASIMD compute instructions */
#define LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V        0x2e201c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V        0x0ea01c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_AND_V        0x0e201c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V        0x0e208403
#define LIBXSMM_AARCH64_INSTR_ASIMD_ADDV_V       0x0e31b802
#define LIBXSMM_AARCH64_INSTR_ASIMD_BIC_V        0x0e601c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V        0x2ee01c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V        0x2ea01c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_BSL_V        0x2e601c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_NEG_V        0x2e20b802
#define LIBXSMM_AARCH64_INSTR_ASIMD_NOT_V        0x2e205812
#define LIBXSMM_AARCH64_INSTR_ASIMD_ORN_V        0x0ee01c13
#define LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V      0x0f00541a
#define LIBXSMM_AARCH64_INSTR_ASIMD_SSHR_I_V     0x0f00041e
#define LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V     0x2f00041e
#define LIBXSMM_AARCH64_INSTR_ASIMD_SSHL_R_V     0x0e204403
#define LIBXSMM_AARCH64_INSTR_ASIMD_USHL_R_V     0x2e204403
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V     0x2e208c03
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_Z_V     0x0e209802
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMGE_R_V     0x0e203c03
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMGE_Z_V     0x2e208802
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMGT_R_V     0x0e203403
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMGT_Z_V     0x0e208802
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMLE_Z_V     0x2e209802
#define LIBXSMM_AARCH64_INSTR_ASIMD_CMLT_Z_V     0x0e20a802
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S     0x5f80100f
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V     0x0f80100f
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V       0x0e20cc0b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_S     0x5f80500f
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_V     0x0f80500f
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_V       0x0ea0cc0b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V       0x0e20d40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V       0x0ea0d40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V       0x2e20dc0b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_E_V     0x0f80900f
#define LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V       0x2e20fc0b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V       0x2ea0f80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FSQRT_V      0x2ea1f80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FRECPE_V     0x0ea1d80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FRECPS_V     0x0e20fc0b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTE_V    0x2ea1d80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTS_V    0x0ea0fc0b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V       0x0e20f40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V       0x0ea0f40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FADDP_V      0x2e20d40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMAXP_V      0x2e20f40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FMINP_V      0x2ea0f40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMEQ_R_V    0x0e20e40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMEQ_Z_V    0x0ea0d80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_R_V    0x2e20e40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_Z_V    0x2ea0c80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V    0x2ea0e40b
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V    0x0ea0c80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMLE_Z_V    0x2ea0d80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCMLT_Z_V    0x0ea0e80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FRINTM_V     0x0e21980a
#define LIBXSMM_AARCH64_INSTR_ASIMD_FCVTMS_V     0x0e21b80a
#define LIBXSMM_AARCH64_INSTR_ASIMD_TRN1         0x0e002803
#define LIBXSMM_AARCH64_INSTR_ASIMD_TRN2         0x0e006803
#define LIBXSMM_AARCH64_INSTR_ASIMD_ZIP1         0x0e003803
#define LIBXSMM_AARCH64_INSTR_ASIMD_ZIP2         0x0e007803
#define LIBXSMM_AARCH64_INSTR_ASIMD_UZP1         0x0e001803
#define LIBXSMM_AARCH64_INSTR_ASIMD_UZP2         0x0e005803
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBL_1        0x0e000013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBL_2        0x0e002013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBL_3        0x0e004013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4        0x0e006013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBX_1        0x0e001013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBX_2        0x0e003013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBX_3        0x0e005013
#define LIBXSMM_AARCH64_INSTR_ASIMD_TBX_4        0x0e007013

/* define SVE LD/ST instriction */
#define LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF    0x85804006
#define LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF    0x85800006
#define LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF    0xe5804006
#define LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF    0xe5800006
#define LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR        0xa5e04083
#define LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF     0xa5e0a086
#define LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR        0xa5404083
#define LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF     0xa540a086
#define LIBXSMM_AARCH64_INSTR_SVE_ST1D_SR        0xe5e04083
#define LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF     0xe5e0e086
#define LIBXSMM_AARCH64_INSTR_SVE_STNT1D_I_OFF   0xe590e086
#define LIBXSMM_AARCH64_INSTR_SVE_ST1W_SR        0xe5404083
#define LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF     0xe540e086
#define LIBXSMM_AARCH64_INSTR_SVE_STNT1W_I_OFF   0xe510e086
#define LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF    0x8540c086
#define LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF    0x85c0e086
#define LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF   0xa5802086
#define LIBXSMM_AARCH64_INSTR_SVE_PRFW_I_OFF     0x85c04085
#define LIBXSMM_AARCH64_INSTR_SVE_PRFD_I_OFF     0x85c06085

/* define SVE compute instructions */
#define LIBXSMM_AARCH64_INSTR_SVE_FMLA_V         0x65200083
#define LIBXSMM_AARCH64_INSTR_SVE_FMLS_V         0x65202083
#define LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_I       0x64200003
#define LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_I       0x64200403
#define LIBXSMM_AARCH64_INSTR_SVE_FMUL_V         0x65000803
#define LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_I       0x64202003
#define LIBXSMM_AARCH64_INSTR_SVE_FNEG_V         0x041da082
#define LIBXSMM_AARCH64_INSTR_SVE_EOR_V          0x04a03013
/* define SVE predicate instructions */
#define LIBXSMM_AARCH64_INSTR_SVE_PTRUE          0x2518e001
#define LIBXSMM_AARCH64_INSTR_SVE_WHILELT        0x25201403

/**
 * shift mode */
typedef enum libxsmm_aarch64_shiftmode {
  LIBXSMM_AARCH64_SHIFTMODE_LSL = 0x0,
  LIBXSMM_AARCH64_SHIFTMODE_LSR = 0x1,
  LIBXSMM_AARCH64_SHIFTMODE_ASR = 0x2
} libxsmm_aarch64_shiftmode;

/**
 * general purpose register width */
typedef enum libxsmm_aarch64_gp_width {
  LIBXSMM_AARCH64_GP_WIDTH_W = 0x0,
  LIBXSMM_AARCH64_GP_WIDTH_X = 0x1
} libxsmm_aarch64_gp_width;

/**
 * asimd vector width simd load and stores */
typedef enum libxsmm_aarch64_asimd_width {
  LIBXSMM_AARCH64_ASIMD_WIDTH_B = 0x0,
  LIBXSMM_AARCH64_ASIMD_WIDTH_H = 0x2,
  LIBXSMM_AARCH64_ASIMD_WIDTH_S = 0x4,
  LIBXSMM_AARCH64_ASIMD_WIDTH_D = 0x6,
  LIBXSMM_AARCH64_ASIMD_WIDTH_Q = 0x1
} libxsmm_aarch64_asimd_width;

/**
 * sz sz Q */
typedef enum libxsmm_aarch64_asimd_tupletype {
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B  = 0x0,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B = 0x1,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4H  = 0x2,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8H  = 0x3,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S  = 0x4,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S  = 0x5,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_1D  = 0x6,
  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D  = 0x7
} libxsmm_aarch64_asimd_tupletype;

typedef enum libxsmm_aarch64_sve_type {
  LIBXSMM_AARCH64_SVE_TYPE_B = 0x0,
  LIBXSMM_AARCH64_SVE_TYPE_H = 0x1,
  LIBXSMM_AARCH64_SVE_TYPE_S = 0x2,
  LIBXSMM_AARCH64_SVE_TYPE_D = 0x3
} libxsmm_aarch64_sve_type;

typedef enum libxsmm_aarch64_sve_pattern {
  LIBXSMM_AARCH64_SVE_PATTERN_POW2   = 0x00,
  LIBXSMM_AARCH64_SVE_PATTERN_VL1    = 0x01,
  LIBXSMM_AARCH64_SVE_PATTERN_VL2    = 0x02,
  LIBXSMM_AARCH64_SVE_PATTERN_VL3    = 0x03,
  LIBXSMM_AARCH64_SVE_PATTERN_VL4    = 0x04,
  LIBXSMM_AARCH64_SVE_PATTERN_VL5    = 0x05,
  LIBXSMM_AARCH64_SVE_PATTERN_VL6    = 0x06,
  LIBXSMM_AARCH64_SVE_PATTERN_VL7    = 0x07,
  LIBXSMM_AARCH64_SVE_PATTERN_VL8    = 0x08,
  LIBXSMM_AARCH64_SVE_PATTERN_VL16   = 0x09,
  LIBXSMM_AARCH64_SVE_PATTERN_VL32   = 0x0a,
  LIBXSMM_AARCH64_SVE_PATTERN_VL64   = 0x0b,
  LIBXSMM_AARCH64_SVE_PATTERN_VL128  = 0x0c,
  LIBXSMM_AARCH64_SVE_PATTERN_VL256  = 0x0d,
  LIBXSMM_AARCH64_SVE_PATTERN_MUL4   = 0x1d,
  LIBXSMM_AARCH64_SVE_PATTERN_MUL3   = 0x1e,
  LIBXSMM_AARCH64_SVE_PATTERN_ALL    = 0x1f
} libxsmm_aarch64_sve_pattern;

typedef enum libxsmm_aarch64_sve_prefetch {
  LIBXSMM_AARCH64_SVE_PREFETCH_LDL1KEEP = 0x0,
  LIBXSMM_AARCH64_SVE_PREFETCH_LDL1STRM = 0x1,
  LIBXSMM_AARCH64_SVE_PREFETCH_LDL2KEEP = 0x2,
  LIBXSMM_AARCH64_SVE_PREFETCH_LDL2STRM = 0x3,
  LIBXSMM_AARCH64_SVE_PREFETCH_LDL3KEEP = 0x4,
  LIBXSMM_AARCH64_SVE_PREFETCH_LDL3STRM = 0x5,
  LIBXSMM_AARCH64_SVE_PREFETCH_STL1KEEP = 0x8,
  LIBXSMM_AARCH64_SVE_PREFETCH_STL1STRM = 0x9,
  LIBXSMM_AARCH64_SVE_PREFETCH_STL2KEEP = 0xa,
  LIBXSMM_AARCH64_SVE_PREFETCH_STL2STRM = 0xb,
  LIBXSMM_AARCH64_SVE_PREFETCH_STL3KEEP = 0xc,
  LIBXSMM_AARCH64_SVE_PREFETCH_STL3STRM = 0xd
} libxsmm_aarch64_sve_prefetch;

/**
 * Opens the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_callee_save_bitmask lower 4 bits control d8-d15 in tuples, bits 4-11 control x16-x30 in tuples, e.g. 0xf saves d8-d15
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_open_stream( libxsmm_generated_code* io_generated_code,
                                              const unsigned short    i_callee_save_bitmask );

/**
 * Closes the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_callee_save_bitmask lower 4 bits control d8-d15 in tuples, bits 4-11 control x16-x30 in tuples, e.g. 0xf saves d8-d15
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_close_stream( libxsmm_generated_code* io_generated_code,
                                               const unsigned short    i_callee_save_bitmask );

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_close_data( libxsmm_generated_code*     io_generated_code,
                                             libxsmm_const_data_tracker* io_const_data );

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_adr_data( libxsmm_generated_code*     io_generated_code,
                                           unsigned int                i_reg,
                                           unsigned int                i_off,
                                           libxsmm_const_data_tracker* io_const_data );

LIBXSMM_API_INTERN
unsigned int libxsmm_aarch64_instruction_add_data( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned char*        i_data,
                                                   unsigned int                i_ndata_bytes,
                                                   unsigned int                i_alignment,
                                                   unsigned int                i_append_only,
                                                   libxsmm_const_data_tracker* io_const_data );

/**
 * Generates ldr, str, etc. instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_addr gp register containing the base address
 * @param i_gp_reg_offset gp register containing an offset
 * @param i_offset optinonal offset
 * @param i_vec_reg the simd register
 * @param i_asimdwidth widht of regiaters (1,2,4,8,16 byte)
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_move( libxsmm_generated_code*           io_generated_code,
                                             const unsigned int                i_vmove_instr,
                                             const unsigned int                i_gp_reg_addr,
                                             const unsigned int                i_gp_reg_offset,
                                             const int                         i_offset,
                                             const unsigned int                i_vec_reg,
                                             const libxsmm_aarch64_asimd_width i_asimdwidth );

/**
 * Generates ins, umov instructions for moving data between ASDIM and GP registers
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg gp register
 * @param i_vec_reg the simd register
 * @param i_index the index to address the vector element
 * @param i_asimdwidth widht of regiaters (1,2,4,8,16 byte)
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_gpr_move( libxsmm_generated_code*           io_generated_code,
                                                 const unsigned int                i_vmove_instr,
                                                 const unsigned int                i_gp_reg,
                                                 const unsigned int                i_vec_reg,
                                                 const short                       i_index,
                                                 const libxsmm_aarch64_asimd_width i_asimdwidth );

/**
 * Generates ldX, stX, etc. instructions for structs
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_addr gp register containing the base address
 * @param i_gp_reg_offset gp register containing an offset
 * @param i_vec_reg the simd register
 * @param i_tupletype tuple specifier
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_struct_r_move( libxsmm_generated_code*               io_generated_code,
                                                      const unsigned int                    i_vmove_instr,
                                                      const unsigned int                    i_gp_reg_addr,
                                                      const unsigned int                    i_gp_reg_offset,
                                                      const unsigned int                    i_vec_reg,
                                                      const libxsmm_aarch64_asimd_tupletype i_tupletype );

/**
 * Generates ldX, stX, etc. instructions for structs
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_addr gp register containing the base address
 * @param i_gp_reg_offset gp register containing an offset
 * @param i_vec_reg the simd register
 * @param i_index the index to address the vector element
 * @param i_asimdwidth widht of regiaters (1,2,4,8,16 byte)
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_struct_move( libxsmm_generated_code*           io_generated_code,
                                                    const unsigned int                i_vmove_instr,
                                                    const unsigned int                i_gp_reg_addr,
                                                    const unsigned int                i_gp_reg_offset,
                                                    const int                         i_offset,
                                                    const unsigned int                i_vec_reg,
                                                    const short                       i_index,
                                                    const libxsmm_aarch64_asimd_width i_asimdwidth );

/**
 * Generates ldp, stp, etc. instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_addr gp register containing the base address
 * @param i_offset optinonal offset
 * @param i_vec_reg_0 first simd register
 * @param i_vec_reg_1 second simd register
 * @param i_asimdwidth widht of regiaters (1,2,4,8,16 byte)
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_pair_move( libxsmm_generated_code*           io_generated_code,
                                                  const unsigned int                i_vmove_instr,
                                                  const unsigned int                i_gp_reg_addr,
                                                  const int                         i_offset,
                                                  const unsigned int                i_vec_reg_0,
                                                  const unsigned int                i_vec_reg_1,
                                                  const libxsmm_aarch64_asimd_width i_asimdwidth );

/**
 * Generates fmla and similar
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_instr actual operation variant
 * @param i_vec_reg_src_0 first source register
 * @param i_idx_shf index if non-negative this value is the scalar access to src0 or the shift immediaate
 * @param i_vec_reg_src_1 second source register
 * @param i_vec_reg_dst destination register
 * @param i_tupletype tuple type
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_compute( libxsmm_generated_code*               io_generated_code,
                                                const unsigned int                    i_vec_instr,
                                                const unsigned int                    i_vec_reg_src_0,
                                                const unsigned int                    i_vec_reg_src_1,
                                                const unsigned char                   i_idx_shf,
                                                const unsigned int                    i_vec_reg_dst,
                                                const libxsmm_aarch64_asimd_tupletype i_tupletype );

/**
 * Generates ldX, stX, etc. instructions for structs
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_addr gp register containing the base address
 * @param i_gp_reg_offset gp register containing an offset
 * @param i_offset imm offset
 * @param i_vec_reg the simd register
 * @param i_pred_reg pred specifier
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_move( libxsmm_generated_code*                io_generated_code,
                                           const unsigned int                     i_vmove_instr,
                                           const unsigned int                     i_gp_reg_addr,
                                           const unsigned int                     i_gp_reg_offset,
                                           const int                              i_offset,
                                           const unsigned int                     i_vec_reg,
                                           const unsigned int                     i_pred_reg );

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_prefetch( libxsmm_generated_code*            io_generated_code,
                                               const unsigned int                 i_prefetch_instr,
                                               const unsigned int                 i_gp_reg_addr,
                                               const unsigned int                 i_gp_reg_offset,
                                               const int                          i_offset,
                                               const unsigned int                 i_pred_reg,
                                               const libxsmm_aarch64_sve_prefetch i_prefetch );

/**
 * Generates fmla and similar
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_instr actual operation variant
 * @param i_vec_reg_src_0 first source register
 * @param i_index index if non-negative this value is the scalar access to src0
 * @param i_vec_reg_src_1 second source register
 * @param i_vec_reg_dst destination register
 * @param i_pred_reg pred register
 * @param i_type  type
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_compute( libxsmm_generated_code*        io_generated_code,
                                              const unsigned int             i_vec_instr,
                                              const unsigned int             i_vec_reg_src_0,
                                              const unsigned int             i_vec_reg_src_1,
                                              const unsigned char            i_index,
                                              const unsigned int             i_vec_reg_dst,
                                              const unsigned int             i_pred_reg,
                                              const libxsmm_aarch64_sve_type i_type );

/**
 * Generates ptrue and similar
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_pred_instr actual operation variant
 * @param i_pred_reg pred register
 * @param i_gp_reg_src_0 first source register
 * @param i_gp_width width of the GP-registers.
 * @param i_gp_reg_src_1 second source register
 * @param i_pattern type
 * @param i_type type
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_pcompute( libxsmm_generated_code*           io_generated_code,
                                               const unsigned int                i_pred_instr,
                                               const unsigned int                i_pred_reg,
                                               const unsigned int                i_gp_reg_src_0,
                                               libxsmm_aarch64_gp_width          i_gp_width,
                                               const unsigned int                i_gp_reg_src_1,
                                               const libxsmm_aarch64_sve_pattern i_pattern,
                                               const libxsmm_aarch64_sve_type    i_type );

/**
 * Generates alu memory movements like ldr, str,
 *
 * @param io_generated_code  pointer to the pointer of the generated code structure
 * @param i_move_instr actual ld/str instruction
 * @param i_gp_reg_addr base address register
 * @param i_gp_reg_off offset register
 * @param i_gp_reg_dst register
 * @param i_offset offset
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_move( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_move_instr,
                                           const unsigned int      i_gp_reg_addr,
                                           const unsigned int      i_gp_reg_off,
                                           const int               i_offset,
                                           const unsigned int      i_gp_reg_dst );

/**
 * Generates ldp, stp, etc. instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_move_instr actual move variant
 * @param i_gp_reg_addr gp register containing the base address
 * @param i_offset optinonal offset
 * @param i_gp_reg_0 first simd register
 * @param i_gp_reg_1 second simd register
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_pair_move( libxsmm_generated_code*           io_generated_code,
                                                const unsigned int                i_move_instr,
                                                const unsigned int                i_gp_reg_addr,
                                                const int                         i_offset,
                                                const unsigned int                i_gp_reg_0,
                                                const unsigned int                i_gp_reg_1 );

/**
 * Generates movk, movz instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual mov instr.
 * @param i_gp_reg_dst the destination register
 * @param i_shift the shift of the immediate
 * @param i_imm16 the 16bit immediate operand
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_move_imm16( libxsmm_generated_code* io_generated_code,
                                                 const unsigned int      i_alu_instr,
                                                 const unsigned int      i_gp_reg_dst,
                                                 const unsigned char     i_shift,
                                                 const unsigned int      i_imm16 );

/**
 * Generates a sequence of instructions to load a int64 into a GPR
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_dst the destination register
 * @param i_imm64 the 64bit immediate operand
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_set_imm64( libxsmm_generated_code*  io_generated_code,
                                                const unsigned int       i_gp_reg_dst,
                                                const unsigned long long i_imm64 );


/**
 * Generate compute with immediate
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_src soruce register
 * @param i_gp_reg_dst destination register
 * @param i_imm12 12bit immediate
 * @param i_imm12_lsl12 0/1 apply lsl 12 to the imm12
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm12( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_src,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned int      i_imm12,
                                                    const unsigned char     i_imm12_lsl12 );

/**
 * Generates a sequence of compute with intermediates
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_src soruce register
 * @param i_gp_reg_dst destination register
 * @param i_imm24 24bit immediate
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm24( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_src,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned int      i_imm24 );

/**
 * Generate compute with immediate
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_src_0 soruce register
 * @param i_gp_reg_src_1 soruce register
 * @param i_gp_reg_dst destination register
 * @param i_imm6 immediate
 * @param i_shift_dir shift mode
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_shifted_reg( libxsmm_generated_code*         io_generated_code,
                                                          const unsigned int              i_alu_instr,
                                                          const unsigned int              i_gp_reg_src_0,
                                                          const unsigned int              i_gp_reg_src_1,
                                                          const unsigned int              i_gp_reg_dst,
                                                          const unsigned int              i_imm6,
                                                          const libxsmm_aarch64_shiftmode i_shift_dir );

/**
 * Generates an optimal sequence of adding up to a 64bit imm to a GPR
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_src soruce register
 * @param i_gp_reg_dst destination register
 * @param i_gp_reg_tmp temp register which may be used
 * @param i_imm64 the 64 bit immediate
 */
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm64( libxsmm_generated_code*         io_generated_code,
                                                    const unsigned int              i_alu_meta_instr,
                                                    const unsigned int              i_gp_reg_src,
                                                    const unsigned int              i_gp_reg_tmp,
                                                    const unsigned int              i_gp_reg_dst,
                                                    const unsigned long long        i_imm64 );

/**
 * Generates a label to which one can jump back and pushes it on the loop label stack
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param io_loop_label_tracker data structure to handle loop labels, nested loops are supported, but not overlapping loops
*/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                           libxsmm_loop_label_tracker* io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and jumps there based on the condition
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_jmp_instr the particular jump instruction used (cbnz, cbz)
 * @param i_gp_reg_cmp the gp register which contains the comperitor
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label
*/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_cond_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                          const unsigned int          i_jmp_instr,
                                                          const unsigned int          i_gp_reg_cmp,
                                                          libxsmm_loop_label_tracker* io_loop_label_tracker );

/**
 * Generates a label to which one can jump back and pushes it on the loop label stack
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @parma i_labal_no position in the jump label tracker to set
 * @param io_jump_forward_label_tracker forward jump tracker structure for tracking the jump addresses/labels
*/
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                      const unsigned int          i_label_no,
                                                      libxsmm_jump_label_tracker* io_jump_label_tracker );

/**
 * Jumps to the address/label stored a specific position
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_jmp_instr the particular jump instruction used
 * @param i_gp_reg_cmp the register holding the condition result
 * @param i_label_no position in the jump label tracker to jump to
 * @param io_jump_label_tracker data structures that tracks arbitrary jump labels
*/

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_cond_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                                     const unsigned int          i_jmp_instr,
                                                     const unsigned int          i_gp_reg_cmp,
                                                     const unsigned int          i_label_no,
                                                     libxsmm_jump_label_tracker* io_jump_label_tracker );

#endif /* GENERATOR_AARCH64_INSTRUCTIONS_H */

