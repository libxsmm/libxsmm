/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_COMMON_H
#define GENERATOR_COMMON_H

#include <libxsmm_generator.h>
#include <libxsmm_cpuid.h>
#include "libxsmm_main.h"
#include "libxsmm_matrixeqn.h"

/*@TODO check if we want to use enums here? Has this implications in the encoder? */
/* defining register mappings */
#define LIBXSMM_X86_GP_REG_RAX               0
#define LIBXSMM_X86_GP_REG_RCX               1
#define LIBXSMM_X86_GP_REG_RDX               2
#define LIBXSMM_X86_GP_REG_RBX               3
#define LIBXSMM_X86_GP_REG_RSP               4
#define LIBXSMM_X86_GP_REG_RBP               5
#define LIBXSMM_X86_GP_REG_RSI               6
#define LIBXSMM_X86_GP_REG_RDI               7
#define LIBXSMM_X86_GP_REG_R8                8
#define LIBXSMM_X86_GP_REG_R9                9
#define LIBXSMM_X86_GP_REG_R10              10
#define LIBXSMM_X86_GP_REG_R11              11
#define LIBXSMM_X86_GP_REG_R12              12
#define LIBXSMM_X86_GP_REG_R13              13
#define LIBXSMM_X86_GP_REG_R14              14
#define LIBXSMM_X86_GP_REG_R15              15
#define LIBXSMM_X86_GP_REG_UNDEF           127

/* define a place holder to handle AVX and SSE with a single encoder function
   using this values as the third operand means SSE */
#define LIBXSMM_X86_VEC_REG_UNDEF          255
#define LIBXSMM_X86_MASK_REG_UNDEF         255
#define LIBXSMM_X86_AVX512_MASK              1  /* this specifies k1 */

/* special value for undefined immediate */
#define LIBXSMM_X86_IMM_UNDEF             1024

/* special instruction */
#define LIBXSMM_X86_INSTR_UNDEF           9999

/*
 * 4-byte Integer Instruction Header definition map:
 * 4th byte:
 * ---------
 * 31 encoding mode (11=EVEX only, 10=REX, 01=VEX, 00=VEX/EVEX Hybrid)
 * 30 encoding mode (11=EVEX only, 10=REX, 01=VEX, 00=VEX/EVEX Hybrid)
 * 29 #operands (2 bits=0-3)
 * 28 #operands (2 bits=0-3)
 * 27 Reversal load/store ordering. 0=regular, 1=reverse (open question: is one bit enough, or do I need a couple bits to show other orderings)
 * 26 Op code extension in ModRM Regfiles (extennsion is bits 20-22)
 * 25 gather/scatter instructions with VSIB / enforce SIB addressing (valid only), e.g. AMX
 * 24 unused - free
 * 3rd byte:
 * ---------
 * 23 W bit (single inputs=0 or double inputs=1)
 * 22 Op code extension shifts in Reg field in ModRM (Shifts like /2, /4, /7, etc.. Maps to values 0-7, corresponding to /0 to /7.)
 * 21 Op code extension shifts in Reg field in ModRM (Shifts like /2, /4, /7, etc.. Maps to values 0-7, corresponding to /0 to /7.)
 * 20 Op code extension shifts in Reg field in ModRM (Shifts like /2, /4, /7, etc.. Maps to values 0-7, corresponding to /0 to /7.)
 * 19 Immediate required by the instruction. 0=no, 1=yes.
 * 18 Reserved, must be 1 for now
 * 17 P (compressed prefix 2-bits and 4 values: None=0x4, 66=0x5, F3=0x6, F2=0x7 , values include the reserved bit)
 * 16 P (compressed prefix 2-bits and 4 values: None=0x4, 66=0x5, F3=0x6, F2=0x7 , values include the reserved bit)
 * 2nd byte:
 * ---------
 * 15 Reserved, must be 0 for now
 * 14 Reserved, must be 0 for now
 * 13 M (Map bit, 0F=0x1, 0F38=0x2, 0F3A=0x3)
 * 12 M (Map bit, 0F-0x1, 0F38=0x2, 0F3A=0x3)
 * EVEX/VEX:
 *   11 Disp8: N value constant for VL=128/256/512, 1=yes, 0=no (fullmem)
 *   10 Disp8: exp of width (0-6 values mapped to 1 to 64, 7 currently unused)
 *    9 Disp8: exp of width (0-6 values mapped to 1 to 64, 7 currently unused)
 *    8 Disp8: exp of width (0-6 values mapped to 1 to 64, 7 currently unused)
 * VEX-only
 *   11 free
 *   10 free
 *    9 L bit
 *    8 1: override L bit, 0 use L bit from user
 * 1st byte:
 * ---------
 *  7 - Op code byte
 *  6 - Op code byte
 *  5 - Op code byte
 *  4 - Op code byte
 *  3 - Op code byte
 *  2 - Op code byte
 *  1 - Op code byte
 *  0 - Op code byte
 */

/* Load/Store/Move instructions -  AVX1,AVX2,AVX512 - deprecated values */
#define LIBXSMM_X86_INSTR_VMOVAPD          0x20851628
#define LIBXSMM_X86_INSTR_VMOVUPD          0x20851610
#define LIBXSMM_X86_INSTR_VMOVAPS          0x20041628
#define LIBXSMM_X86_INSTR_VMOVUPS          0x20041610
#define LIBXSMM_X86_INSTR_VMOVSD           0x20871b10
#define LIBXSMM_X86_INSTR_VMOVSS           0x20061a10
#define LIBXSMM_X86_INSTR_VMASKMOVPD       0x7005202d
#define LIBXSMM_X86_INSTR_VMASKMOVPS       0x7005202c
#define LIBXSMM_X86_INSTR_VMOVDQA32        0xe005166f
#define LIBXSMM_X86_INSTR_VMOVDQA64        0xe085166f
#define LIBXSMM_X86_INSTR_VMOVDQU8         0xe007166f
#define LIBXSMM_X86_INSTR_VMOVDQU16        0xe087166f
#define LIBXSMM_X86_INSTR_VMOVDQU32        0xe006166f
#define LIBXSMM_X86_INSTR_VMOVDQU64        0xe086166f
/* Load instructions -  AVX,AVX2,AVX512 */
#define LIBXSMM_X86_INSTR_VMOVDDUP         0x20871612
#define LIBXSMM_X86_INSTR_VBROADCASTSD     0x20852b19
#define LIBXSMM_X86_INSTR_VBROADCASTSD_VEX 0x60052019
#define LIBXSMM_X86_INSTR_VBROADCASTSS     0x20052a18
#define LIBXSMM_X86_INSTR_VPBROADCASTB     0x20052878
#define LIBXSMM_X86_INSTR_VPBROADCASTW     0x20052979
#define LIBXSMM_X86_INSTR_VPBROADCASTD     0x20052a58
#define LIBXSMM_X86_INSTR_VPBROADCASTQ     0xe0852b59
#define LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX 0x60052059
#define LIBXSMM_X86_INSTR_VPBROADCASTB_GPR 0xe005287a
#define LIBXSMM_X86_INSTR_VPBROADCASTW_GPR 0xe005297b
#define LIBXSMM_X86_INSTR_VPBROADCASTD_GPR 0xe0052a7c
#define LIBXSMM_X86_INSTR_VPBROADCASTQ_GPR 0xe0852b7c
#define LIBXSMM_X86_INSTR_VMOVAPD_LD       0x20851628
#define LIBXSMM_X86_INSTR_VMOVUPD_LD       0x20851610
#define LIBXSMM_X86_INSTR_VMOVAPS_LD       0x20041628
#define LIBXSMM_X86_INSTR_VMOVUPS_LD       0x20041610
#define LIBXSMM_X86_INSTR_VMOVSD_LD        0x20871b10
#define LIBXSMM_X86_INSTR_VMOVSS_LD        0x20061a10
#define LIBXSMM_X86_INSTR_VMASKMOVPD_LD    0x7005202d
#define LIBXSMM_X86_INSTR_VMASKMOVPS_LD    0x7005202c
#define LIBXSMM_X86_INSTR_VMOVDQA32_LD     0xe005166f
#define LIBXSMM_X86_INSTR_VMOVDQA64_LD     0xe085166f
#define LIBXSMM_X86_INSTR_VMOVDQU8_LD      0xe007166f
#define LIBXSMM_X86_INSTR_VMOVDQU16_LD     0xe087166f
#define LIBXSMM_X86_INSTR_VMOVDQU32_LD     0xe006166f
#define LIBXSMM_X86_INSTR_VMOVDQU64_LD     0xe086166f
#define LIBXSMM_X86_INSTR_VBROADCASTI128   0x6005205a
#define LIBXSMM_X86_INSTR_VBROADCASTI32X2  0xe0052359
#define LIBXSMM_X86_INSTR_VBROADCASTI32X4  0xe005245a
#define LIBXSMM_X86_INSTR_VBROADCASTI64X2  0xe085245a
#define LIBXSMM_X86_INSTR_VBROADCASTI32X8  0xe005255b
#define LIBXSMM_X86_INSTR_VBROADCASTI64X4  0xe085255b
#define LIBXSMM_X86_INSTR_VMOVD_LD         0x20051a6e
#define LIBXSMM_X86_INSTR_VMOVQ_LD         0x20851b6e
/* Store instructions - AVX,AVX2,AVX512 */
#define LIBXSMM_X86_INSTR_VMOVNTPD         0x2085162b
#define LIBXSMM_X86_INSTR_VMOVNTPS         0x2004162b
#define LIBXSMM_X86_INSTR_VMOVNTDQ         0x200516e7
#define LIBXSMM_X86_INSTR_VMOVAPD_ST       0x20851629
#define LIBXSMM_X86_INSTR_VMOVUPD_ST       0x20851611
#define LIBXSMM_X86_INSTR_VMOVAPS_ST       0x20041629
#define LIBXSMM_X86_INSTR_VMOVUPS_ST       0x20041611
#define LIBXSMM_X86_INSTR_VMOVSD_ST        0x20871b11
#define LIBXSMM_X86_INSTR_VMOVSS_ST        0x20061a11
#define LIBXSMM_X86_INSTR_VMASKMOVPD_ST    0x7005202f
#define LIBXSMM_X86_INSTR_VMASKMOVPS_ST    0x7005202e
#define LIBXSMM_X86_INSTR_VMOVDQA32_ST     0xe005167f
#define LIBXSMM_X86_INSTR_VMOVDQA64_ST     0xe085167f
#define LIBXSMM_X86_INSTR_VMOVDQU8_ST      0xe007167f
#define LIBXSMM_X86_INSTR_VMOVDQU16_ST     0xe087167f
#define LIBXSMM_X86_INSTR_VMOVDQU32_ST     0xe006167f
#define LIBXSMM_X86_INSTR_VMOVDQU64_ST     0xe086167f
#define LIBXSMM_X86_INSTR_VMOVD_ST         0x20051a7e
#define LIBXSMM_X86_INSTR_VMOVQ_ST         0x20851b7e
/* Gather/Scatter instructions */
#define LIBXSMM_X86_INSTR_VGATHERDPS_VEX   0x72052092
#define LIBXSMM_X86_INSTR_VGATHERDPD_VEX   0x72852092
#define LIBXSMM_X86_INSTR_VGATHERQPS_VEX   0x72052093
#define LIBXSMM_X86_INSTR_VGATHERQPD_VEX   0x72852093
#define LIBXSMM_X86_INSTR_VPGATHERDD_VEX   0x72052090
#define LIBXSMM_X86_INSTR_VPGATHERDQ_VEX   0x72852090
#define LIBXSMM_X86_INSTR_VPGATHERQD_VEX   0x72052091
#define LIBXSMM_X86_INSTR_VPGATHERQQ_VEX   0x72852091
#define LIBXSMM_X86_INSTR_VGATHERDPS       0xe2052a92
#define LIBXSMM_X86_INSTR_VGATHERDPD       0xe2852b92
#define LIBXSMM_X86_INSTR_VGATHERQPS       0xe2052a93
#define LIBXSMM_X86_INSTR_VGATHERQPD       0xe2852b93
#define LIBXSMM_X86_INSTR_VPGATHERDD       0xe2052a90
#define LIBXSMM_X86_INSTR_VPGATHERDQ       0xe2852b90
#define LIBXSMM_X86_INSTR_VPGATHERQD       0xe2052a91
#define LIBXSMM_X86_INSTR_VPGATHERQQ       0xe2852b91
#define LIBXSMM_X86_INSTR_VSCATTERDPS      0xe2052aa2
#define LIBXSMM_X86_INSTR_VSCATTERDPD      0xe2852ba2
#define LIBXSMM_X86_INSTR_VSCATTERQPS      0xe2052aa3
#define LIBXSMM_X86_INSTR_VSCATTERQPD      0xe2852ba3
#define LIBXSMM_X86_INSTR_VPSCATTERDD      0xe2952aa0
#define LIBXSMM_X86_INSTR_VPSCATTERDQ      0xe2852ba0
#define LIBXSMM_X86_INSTR_VPSCATTERQD      0xe2052aa1
#define LIBXSMM_X86_INSTR_VPSCATTERQQ      0xe2852ba1

/* SSE */
#define LIBXSMM_X86_INSTR_MOVAPD           10009
#define LIBXSMM_X86_INSTR_MOVUPD           10010
#define LIBXSMM_X86_INSTR_MOVAPS           10011
#define LIBXSMM_X86_INSTR_MOVUPS           10012
#define LIBXSMM_X86_INSTR_MOVSD            10013
#define LIBXSMM_X86_INSTR_MOVSS            10014
#define LIBXSMM_X86_INSTR_MOVDDUP          10015
#define LIBXSMM_X86_INSTR_SHUFPS           10016
#define LIBXSMM_X86_INSTR_SHUFPD           10017

/* Shuffle/Permute/Blend instructions */
/* VEx and EVEX */
#define LIBXSMM_X86_INSTR_VSHUFPS          0x300c16c6
#define LIBXSMM_X86_INSTR_VSHUFPD          0x308d16c6
#define LIBXSMM_X86_INSTR_VPSHUFB          0x30052600
#define LIBXSMM_X86_INSTR_VPSHUFD          0x200d1670
#define LIBXSMM_X86_INSTR_VPSHUFHW         0x200e1670
#define LIBXSMM_X86_INSTR_VPSHUFLW         0x200f1670
#define LIBXSMM_X86_INSTR_VUNPCKLPD        0x30851614
#define LIBXSMM_X86_INSTR_VUNPCKLPS        0x30041614
#define LIBXSMM_X86_INSTR_VUNPCKHPD        0x30851615
#define LIBXSMM_X86_INSTR_VUNPCKHPS        0x30041615
#define LIBXSMM_X86_INSTR_VPUNPCKLWD       0x30051661
#define LIBXSMM_X86_INSTR_VPUNPCKHWD       0x30051669
#define LIBXSMM_X86_INSTR_VPUNPCKLDQ       0x30051662
#define LIBXSMM_X86_INSTR_VPUNPCKHDQ       0x3005166a
#define LIBXSMM_X86_INSTR_VPUNPCKLQDQ      0x3085166c
#define LIBXSMM_X86_INSTR_VPUNPCKHQDQ      0x3085166d
#define LIBXSMM_X86_INSTR_VPERMD           0x30052636
#define LIBXSMM_X86_INSTR_VPERMQ_I         0x208d3e00
#define LIBXSMM_X86_INSTR_VPERMPS          0x30052516
#define LIBXSMM_X86_INSTR_VPERMPD_I        0x208d3601
#define LIBXSMM_X86_INSTR_VPERMILPS        0x3005250c
#define LIBXSMM_X86_INSTR_VPERMILPS_I      0x200d3504
/* VEX only */
#define LIBXSMM_X86_INSTR_VPERM2F128       0x700d3006
#define LIBXSMM_X86_INSTR_VPERM2I128       0x700d3046
#define LIBXSMM_X86_INSTR_VEXTRACTF128     0x680d3019
#define LIBXSMM_X86_INSTR_VEXTRACTI128     0x680d3039
#define LIBXSMM_X86_INSTR_VPERMILPD_VEX    0x7005200d
#define LIBXSMM_X86_INSTR_VPERMILPD_VEX_I  0x600d3005
#define LIBXSMM_X86_INSTR_VBLENDPD         0x700d300d
#define LIBXSMM_X86_INSTR_VBLENDPS         0x700d300c
#define LIBXSMM_X86_INSTR_VBLENDVPD        0x700d304b
#define LIBXSMM_X86_INSTR_VBLENDVPS        0x700d304a
#define LIBXSMM_X86_INSTR_VPBLENDD         0x700d3002
#define LIBXSMM_X86_INSTR_VPBLENDW         0x700d300e
#define LIBXSMM_X86_INSTR_VPBLENDVB        0x700d304c
#define LIBXSMM_X86_INSTR_VMOVMSKPD        0x60051050
#define LIBXSMM_X86_INSTR_VMOVMSKPS        0x60041050
#define LIBXSMM_X86_INSTR_VPMOVMSKB        0x600510d7
/* EVEX only */
#define LIBXSMM_X86_INSTR_VSHUFF32X4       0xf00d3623
#define LIBXSMM_X86_INSTR_VSHUFF64X2       0xf08d3623
#define LIBXSMM_X86_INSTR_VSHUFI32X4       0xf00d3643
#define LIBXSMM_X86_INSTR_VSHUFI64X2       0xf08d3643
#define LIBXSMM_X86_INSTR_VEXTRACTF32X4    0xe80d3c19
#define LIBXSMM_X86_INSTR_VEXTRACTF64X2    0xe88d3c19
#define LIBXSMM_X86_INSTR_VEXTRACTF32X8    0xe80d3d1b
#define LIBXSMM_X86_INSTR_VEXTRACTF64X4    0xe88d3d1b
#define LIBXSMM_X86_INSTR_VEXTRACTI32X4    0xe80d3c39
#define LIBXSMM_X86_INSTR_VEXTRACTI64X2    0xe88d3c39
#define LIBXSMM_X86_INSTR_VEXTRACTI32X8    0xe80d3d3b
#define LIBXSMM_X86_INSTR_VEXTRACTI64X4    0xe88d3d3b
#define LIBXSMM_X86_INSTR_VINSERTI32X4     0xf80d3c38
#define LIBXSMM_X86_INSTR_VBLENDMPS        0xf0052665
#define LIBXSMM_X86_INSTR_VBLENDMPD        0xf0852665
#define LIBXSMM_X86_INSTR_VPBLENDMB        0xf0052666
#define LIBXSMM_X86_INSTR_VPBLENDMW        0xf0852666
#define LIBXSMM_X86_INSTR_VPBLENDMD        0xf0052664
#define LIBXSMM_X86_INSTR_VPBLENDMQ        0xf0852664
#define LIBXSMM_X86_INSTR_VEXPANDPD        0xe0852b88
#define LIBXSMM_X86_INSTR_VEXPANDPS        0xe0052a88
#define LIBXSMM_X86_INSTR_VPEXPANDQ        0xe0852b89
#define LIBXSMM_X86_INSTR_VPEXPANDD        0xe0052a89
#define LIBXSMM_X86_INSTR_VPEXPANDW        0xe0852962
#define LIBXSMM_X86_INSTR_VPEXPANDB        0xe0052862
#define LIBXSMM_X86_INSTR_VPERMW           0xf085268d
#define LIBXSMM_X86_INSTR_VPERMPD          0xf0852616
#define LIBXSMM_X86_INSTR_VPERMT2B         0xf005267d
#define LIBXSMM_X86_INSTR_VPERMT2W         0xf085267d
#define LIBXSMM_X86_INSTR_VPERMT2D         0xf005267e
#define LIBXSMM_X86_INSTR_VPERMT2Q         0xf085267e
#define LIBXSMM_X86_INSTR_VPERMILPD        0xf085260d
#define LIBXSMM_X86_INSTR_VPERMILPD_I      0xe08d3605

/* FMA instructions */
#define LIBXSMM_X86_INSTR_VFMADD132PS      0x30052698
#define LIBXSMM_X86_INSTR_VFMADD132PD      0x30852698
#define LIBXSMM_X86_INSTR_VFMADD213PS      0x300526a8
#define LIBXSMM_X86_INSTR_VFMADD213PD      0x308526a8
#define LIBXSMM_X86_INSTR_VFMADD231PS      0x300526b8
#define LIBXSMM_X86_INSTR_VFMADD231PD      0x308526b8
#define LIBXSMM_X86_INSTR_VFMSUB132PS      0x3005269a
#define LIBXSMM_X86_INSTR_VFMSUB132PD      0x3085269a
#define LIBXSMM_X86_INSTR_VFMSUB213PS      0x300526aa
#define LIBXSMM_X86_INSTR_VFMSUB213PD      0x308526aa
#define LIBXSMM_X86_INSTR_VFMSUB231PS      0x300526ba
#define LIBXSMM_X86_INSTR_VFMSUB231PD      0x308526ba
#define LIBXSMM_X86_INSTR_VFNMADD132PS     0x3005269c
#define LIBXSMM_X86_INSTR_VFNMADD132PD     0x3085269c
#define LIBXSMM_X86_INSTR_VFNMADD213PS     0x300526ac
#define LIBXSMM_X86_INSTR_VFNMADD213PD     0x308526ac
#define LIBXSMM_X86_INSTR_VFNMADD231PS     0x300526bc
#define LIBXSMM_X86_INSTR_VFNMADD231PD     0x308526bc
#define LIBXSMM_X86_INSTR_VFNMSUB132PS     0x3005269e
#define LIBXSMM_X86_INSTR_VFNMSUB132PD     0x3085269e
#define LIBXSMM_X86_INSTR_VFNMSUB213PS     0x300526ae
#define LIBXSMM_X86_INSTR_VFNMSUB213PD     0x308526ae
#define LIBXSMM_X86_INSTR_VFNMSUB231PS     0x300526be
#define LIBXSMM_X86_INSTR_VFNMSUB231PD     0x308526be
#define LIBXSMM_X86_INSTR_VFMADD132SD      0x30852b99
#define LIBXSMM_X86_INSTR_VFMADD213SD      0x30852ba9
#define LIBXSMM_X86_INSTR_VFMADD231SD      0x30852bb9
#define LIBXSMM_X86_INSTR_VFMADD132SS      0x30052a99
#define LIBXSMM_X86_INSTR_VFMADD213SS      0x30052aa9
#define LIBXSMM_X86_INSTR_VFMADD231SS      0x30052ab9
#define LIBXSMM_X86_INSTR_VFMSUB132SD      0x30852b9b
#define LIBXSMM_X86_INSTR_VFMSUB213SD      0x30852bab
#define LIBXSMM_X86_INSTR_VFMSUB231SD      0x30852bbb
#define LIBXSMM_X86_INSTR_VFMSUB132SS      0x30052a9b
#define LIBXSMM_X86_INSTR_VFMSUB213SS      0x30052aab
#define LIBXSMM_X86_INSTR_VFMSUB231SS      0x30052abb
#define LIBXSMM_X86_INSTR_VFNMADD132SD     0x30852b9d
#define LIBXSMM_X86_INSTR_VFNMADD213SD     0x30852bad
#define LIBXSMM_X86_INSTR_VFNMADD231SD     0x30852bbd
#define LIBXSMM_X86_INSTR_VFNMADD132SS     0x30052a9d
#define LIBXSMM_X86_INSTR_VFNMADD213SS     0x30052aad
#define LIBXSMM_X86_INSTR_VFNMADD231SS     0x30052abd
#define LIBXSMM_X86_INSTR_VFNMSUB132SD     0x30852b9f
#define LIBXSMM_X86_INSTR_VFNMSUB213SD     0x30852baf
#define LIBXSMM_X86_INSTR_VFNMSUB231SD     0x30852bbf
#define LIBXSMM_X86_INSTR_VFNMSUB132SS     0x30052a9f
#define LIBXSMM_X86_INSTR_VFNMSUB213SS     0x30052aaf
#define LIBXSMM_X86_INSTR_VFNMSUB231SS     0x30052abf

/* floating point helpers, VEX */
#define LIBXSMM_X86_INSTR_VROUNDPD         0x600d3009
#define LIBXSMM_X86_INSTR_VROUNDSD         0x700d300b
#define LIBXSMM_X86_INSTR_VROUNDPS         0x600d3008
#define LIBXSMM_X86_INSTR_VROUNDSS         0x700d300a
#define LIBXSMM_X86_INSTR_VRCPPS           0x60041053
#define LIBXSMM_X86_INSTR_VRCPSS           0x70061053
#define LIBXSMM_X86_INSTR_VRSQRTPS         0x60041052
#define LIBXSMM_X86_INSTR_VRSQRTSS         0x70061052

/* floating point helpers, EVEX */
#define LIBXSMM_X86_INSTR_VRANGEPS         0xf00d3650
#define LIBXSMM_X86_INSTR_VRANGEPD         0xf08d3650
#define LIBXSMM_X86_INSTR_VRANGESS         0xf00d3a51
#define LIBXSMM_X86_INSTR_VRANGESD         0xf08d3b51
#define LIBXSMM_X86_INSTR_VREDUCEPS        0xe00d3656
#define LIBXSMM_X86_INSTR_VREDUCEPD        0xe08d3656
#define LIBXSMM_X86_INSTR_VREDUCESS        0xf00d3a57
#define LIBXSMM_X86_INSTR_VREDUCESD        0xf08d3b57
#define LIBXSMM_X86_INSTR_VRCP14PS         0xe005264c
#define LIBXSMM_X86_INSTR_VRCP14PD         0xe085264c
#define LIBXSMM_X86_INSTR_VRCP14SS         0xf0052a4d
#define LIBXSMM_X86_INSTR_VRCP14SD         0xf0852b4d
#define LIBXSMM_X86_INSTR_VRNDSCALEPS      0xe00d3608
#define LIBXSMM_X86_INSTR_VRNDSCALEPD      0xe08d3609
#define LIBXSMM_X86_INSTR_VRNDSCALESS      0xf00d3a0a
#define LIBXSMM_X86_INSTR_VRNDSCALESD      0xf08d3b0b
#define LIBXSMM_X86_INSTR_VRSQRT14PS       0xe005264e
#define LIBXSMM_X86_INSTR_VRSQRT14PD       0xe085264e
#define LIBXSMM_X86_INSTR_VRSQRT14SS       0xf0052a4f
#define LIBXSMM_X86_INSTR_VRSQRT14SD       0xf0852b4f
#define LIBXSMM_X86_INSTR_VSCALEFPS        0xf005262c
#define LIBXSMM_X86_INSTR_VSCALEFPD        0xf085262c
#define LIBXSMM_X86_INSTR_VSCALEFSS        0xf0052a2d
#define LIBXSMM_X86_INSTR_VSCALEFSD        0xf0852b2d

/* compare instructions */
#define LIBXSMM_X86_INSTR_VCMPPS           0x300c16c2
#define LIBXSMM_X86_INSTR_VCMPSS           0x300e1ac2
#define LIBXSMM_X86_INSTR_VCMPPD           0x308d16c2
#define LIBXSMM_X86_INSTR_VCMPSD           0x308f1bc2
#define LIBXSMM_X86_INSTR_VPCMPB           0xf00d363f
#define LIBXSMM_X86_INSTR_VPCMPUB          0xf00d363e
#define LIBXSMM_X86_INSTR_VPCMPW           0xf08d363f
#define LIBXSMM_X86_INSTR_VPCMPUW          0xf08d363e
#define LIBXSMM_X86_INSTR_VPCMPD           0xf00d361f
#define LIBXSMM_X86_INSTR_VPCMPUD          0xf00d361e
#define LIBXSMM_X86_INSTR_VPCMPQ           0xf08d361f
#define LIBXSMM_X86_INSTR_VPCMPUQ          0xf08d361e
#define LIBXSMM_X86_INSTR_VPCMPEQB         0x30051674
#define LIBXSMM_X86_INSTR_VPCMPEQW         0x30051675
#define LIBXSMM_X86_INSTR_VPCMPEQD         0x30051676
#define LIBXSMM_X86_INSTR_VPCMPEQQ         0x30852629
#define LIBXSMM_X86_INSTR_VPCMPGTB         0x30051664
#define LIBXSMM_X86_INSTR_VPCMPGTW         0x30051665
#define LIBXSMM_X86_INSTR_VPCMPGTD         0x30051666
#define LIBXSMM_X86_INSTR_VPCMPGTQ         0x30852637
#define LIBXSMM_X86_INSTR_VPCMPESTRI       0x600d3061
#define LIBXSMM_X86_INSTR_VPCMPESTRM       0x600d3060
#define LIBXSMM_X86_INSTR_VPCMPISTRI       0x600d3063
#define LIBXSMM_X86_INSTR_VPCMPISTRM       0x600d3062

/* convert instructions */
#define LIBXSMM_X86_INSTR_VCVTPS2PD        0x2004155a
#define LIBXSMM_X86_INSTR_VCVTPH2PS        0x20052513
#define LIBXSMM_X86_INSTR_VCVTPS2PH        0x280d351d
#define LIBXSMM_X86_INSTR_VCVTDQ2PS        0x2004165b
#define LIBXSMM_X86_INSTR_VCVTPS2DQ        0x2005165b
#define LIBXSMM_X86_INSTR_VCVTPS2UDQ       0x20041679
#define LIBXSMM_X86_INSTR_VPMOVDW          0x28062533
#define LIBXSMM_X86_INSTR_VPMOVSXWD        0x20052523
#define LIBXSMM_X86_INSTR_VPMOVDB          0x28062431
#define LIBXSMM_X86_INSTR_VPMOVSDB         0x28062421
#define LIBXSMM_X86_INSTR_VPMOVUSDB        0x28062411
#define LIBXSMM_X86_INSTR_VPMOVZXWD        0x20052533
#define LIBXSMM_X86_INSTR_VPMOVSXBD        0x20052421
#define LIBXSMM_X86_INSTR_VPMOVZXBD        0x20052431
#define LIBXSMM_X86_INSTR_VPMOVUSWB        0xe0062510
#define LIBXSMM_X86_INSTR_VPMOVSWB         0xe0062520
#define LIBXSMM_X86_INSTR_VPMOVWB          0xe0062530

/* shift instructions */
#define LIBXSMM_X86_INSTR_VPSLLD_I         0x246d1672
#define LIBXSMM_X86_INSTR_VPSRAD_I         0x244d1672
#define LIBXSMM_X86_INSTR_VPSRLD_I         0x242d1672
#define LIBXSMM_X86_INSTR_VPSLLVW          0x30852612
#define LIBXSMM_X86_INSTR_VPSLLVD          0x30052647
#define LIBXSMM_X86_INSTR_VPSLLVQ          0x30852647
#define LIBXSMM_X86_INSTR_VPSRAVW          0x30852611
#define LIBXSMM_X86_INSTR_VPSRAVD          0x30052646
#define LIBXSMM_X86_INSTR_VPSRAVQ          0x30852646
#define LIBXSMM_X86_INSTR_VPSRLVW          0x30852610
#define LIBXSMM_X86_INSTR_VPSRLVD          0x30052645
#define LIBXSMM_X86_INSTR_VPSRLVQ          0x30852645

/* floating point compute */
#define LIBXSMM_X86_INSTR_VXORPD           0x30851657
#define LIBXSMM_X86_INSTR_VADDPD           0x30851658
#define LIBXSMM_X86_INSTR_VMULPD           0x30851659
#define LIBXSMM_X86_INSTR_VSUBPD           0x3085165c
#define LIBXSMM_X86_INSTR_VDIVPD           0x3085165e
#define LIBXSMM_X86_INSTR_VMINPD           0x3085165d
#define LIBXSMM_X86_INSTR_VMAXPD           0x3085165f
#define LIBXSMM_X86_INSTR_VSQRTPD          0x20851651
#define LIBXSMM_X86_INSTR_VADDSD           0x30871b58
#define LIBXSMM_X86_INSTR_VMULSD           0x30871b59
#define LIBXSMM_X86_INSTR_VSUBSD           0x30871b5c
#define LIBXSMM_X86_INSTR_VDIVSD           0x30871b5e
#define LIBXSMM_X86_INSTR_VMINSD           0x30871b5d
#define LIBXSMM_X86_INSTR_VMAXSD           0x30871b5f
#define LIBXSMM_X86_INSTR_VSQRTSD          0x30871b51

#define LIBXSMM_X86_INSTR_VXORPS           0x30041657
#define LIBXSMM_X86_INSTR_VADDPS           0x30041658
#define LIBXSMM_X86_INSTR_VMULPS           0x30041659
#define LIBXSMM_X86_INSTR_VSUBPS           0x3004165c
#define LIBXSMM_X86_INSTR_VDIVPS           0x3004165e
#define LIBXSMM_X86_INSTR_VMINPS           0x3004155d
#define LIBXSMM_X86_INSTR_VMAXPS           0x3004165f
#define LIBXSMM_X86_INSTR_VSQRTPS          0x20041551
#define LIBXSMM_X86_INSTR_VMULSS           0x30061a59
#define LIBXSMM_X86_INSTR_VADDSS           0x30061a58
#define LIBXSMM_X86_INSTR_VSUBSS           0x30061a5c
#define LIBXSMM_X86_INSTR_VDIVSS           0x30061a5e
#define LIBXSMM_X86_INSTR_VMINSS           0x30061a5d
#define LIBXSMM_X86_INSTR_VMAXSS           0x30061a5f
#define LIBXSMM_X86_INSTR_VSQRTSS          0x30061a51

/* integer compute */
#define LIBXSMM_X86_INSTR_VPXORD           0x300516ef
#define LIBXSMM_X86_INSTR_VPORD            0x300516eb
#define LIBXSMM_X86_INSTR_VPANDD           0x300516db
#define LIBXSMM_X86_INSTR_VPANDQ           0x308516db
#define LIBXSMM_X86_INSTR_VPADDQ           0x308516d4
#define LIBXSMM_X86_INSTR_VPADDB           0x300516fc
#define LIBXSMM_X86_INSTR_VPADDW           0x300516fd
#define LIBXSMM_X86_INSTR_VPADDD           0x300516fe
#define LIBXSMM_X86_INSTR_VPMADDWD         0x300516f5
#define LIBXSMM_X86_INSTR_VPMADDUBSW       0x30052604
#define LIBXSMM_X86_INSTR_VPADDSW          0x300516ed
#define LIBXSMM_X86_INSTR_VPADDSB          0x300516ec
#define LIBXSMM_X86_INSTR_VPSUBD           0x300516fa
#define LIBXSMM_X86_INSTR_VPMAXSD          0x3005263d
#define LIBXSMM_X86_INSTR_VPMINSD          0x30052639

/* QUAD MADD, QUAD VNNI and VNNI */
#define LIBXSMM_X86_INSTR_V4FMADDPS        0xf0072c9a
#define LIBXSMM_X86_INSTR_V4FNMADDPS       0xf0072caa
#define LIBXSMM_X86_INSTR_V4FMADDSS        0xf0072c9b
#define LIBXSMM_X86_INSTR_V4FNMADDSS       0xf0072cab
#define LIBXSMM_X86_INSTR_VP4DPWSSDS       0xf0072c53
#define LIBXSMM_X86_INSTR_VP4DPWSSD        0xf0072c52
#define LIBXSMM_X86_INSTR_VPDPBUSD         0x30052650
#define LIBXSMM_X86_INSTR_VPDPBUSDS        0x30052651
#define LIBXSMM_X86_INSTR_VPDPWSSD         0x30052652
#define LIBXSMM_X86_INSTR_VPDPWSSDS        0x30052653

/* AVX512 BF16 */
#define LIBXSMM_X86_INSTR_VDPBF16PS        0xf0062652
#define LIBXSMM_X86_INSTR_VCVTNEPS2BF16    0xe0062672
#define LIBXSMM_X86_INSTR_VCVTNE2PS2BF16   0xf0072672

/* AVX512 Mask compute instructions  */
#define LIBXSMM_X86_INSTR_KADDB            0xb005134a
#define LIBXSMM_X86_INSTR_KADDW            0xb004134a
#define LIBXSMM_X86_INSTR_KADDD            0xb085134a
#define LIBXSMM_X86_INSTR_KADDQ            0xb084134a
#define LIBXSMM_X86_INSTR_KANDB            0xb0051341
#define LIBXSMM_X86_INSTR_KANDW            0xb0041341
#define LIBXSMM_X86_INSTR_KANDD            0xb0851341
#define LIBXSMM_X86_INSTR_KANDQ            0xb0841341
#define LIBXSMM_X86_INSTR_KANDNB           0xb0051342
#define LIBXSMM_X86_INSTR_KANDNW           0xb0041342
#define LIBXSMM_X86_INSTR_KANDND           0xb0851342
#define LIBXSMM_X86_INSTR_KANDNQ           0xb0841342
#define LIBXSMM_X86_INSTR_KNOTB            0xa0051144
#define LIBXSMM_X86_INSTR_KNOTW            0xa0041144
#define LIBXSMM_X86_INSTR_KNOTD            0xa0851144
#define LIBXSMM_X86_INSTR_KNOTQ            0xa0841144
#define LIBXSMM_X86_INSTR_KORB             0xb0051345
#define LIBXSMM_X86_INSTR_KORW             0xb0041345
#define LIBXSMM_X86_INSTR_KORD             0xb0851345
#define LIBXSMM_X86_INSTR_KORQ             0xb0841345
#define LIBXSMM_X86_INSTR_KORTESTB         0xa0051198
#define LIBXSMM_X86_INSTR_KORTESTW         0xa0041198
#define LIBXSMM_X86_INSTR_KORTESTD         0xa0851198
#define LIBXSMM_X86_INSTR_KORTESTQ         0xa0841198
#define LIBXSMM_X86_INSTR_KSHIFTLB         0xa00d3132
#define LIBXSMM_X86_INSTR_KSHIFTLW         0xa08d3132
#define LIBXSMM_X86_INSTR_KSHIFTLD         0xa00d3133
#define LIBXSMM_X86_INSTR_KSHIFTLQ         0xa08d3133
#define LIBXSMM_X86_INSTR_KSHIFTRB         0xa00d3130
#define LIBXSMM_X86_INSTR_KSHIFTRW         0xa08d3130
#define LIBXSMM_X86_INSTR_KSHIFTRD         0xa00d3131
#define LIBXSMM_X86_INSTR_KSHIFTRQ         0xa08d3131
#define LIBXSMM_X86_INSTR_KTESTB           0xa0051199
#define LIBXSMM_X86_INSTR_KTESTW           0xa0041199
#define LIBXSMM_X86_INSTR_KTESTD           0xa0851199
#define LIBXSMM_X86_INSTR_KTESTQ           0xa0841199
#define LIBXSMM_X86_INSTR_KUNPCKBW         0xb005134b
#define LIBXSMM_X86_INSTR_KUNPCKWD         0xb004134b
#define LIBXSMM_X86_INSTR_KUNPCKDQ         0xb084134b
#define LIBXSMM_X86_INSTR_KXNORB           0xb0051346
#define LIBXSMM_X86_INSTR_KXNORW           0xb0041346
#define LIBXSMM_X86_INSTR_KXNORD           0xb0851346
#define LIBXSMM_X86_INSTR_KXNORQ           0xb0841346
#define LIBXSMM_X86_INSTR_KXORB            0xb0051347
#define LIBXSMM_X86_INSTR_KXORW            0xb0041347
#define LIBXSMM_X86_INSTR_KXORD            0xb0851347
#define LIBXSMM_X86_INSTR_KXORQ            0xb0841347

/* AVX512 Mask mov instructions */
#define LIBXSMM_X86_INSTR_KMOVB_GPR_LD     0xa0051192
#define LIBXSMM_X86_INSTR_KMOVW_GPR_LD     0xa0041192
#define LIBXSMM_X86_INSTR_KMOVD_GPR_LD     0xa0071192
#define LIBXSMM_X86_INSTR_KMOVQ_GPR_LD     0xa0871192
#define LIBXSMM_X86_INSTR_KMOVB_GPR_ST     0xa8051193
#define LIBXSMM_X86_INSTR_KMOVW_GPR_ST     0xa8041193
#define LIBXSMM_X86_INSTR_KMOVD_GPR_ST     0xa8071193
#define LIBXSMM_X86_INSTR_KMOVQ_GPR_ST     0xa8871193
#define LIBXSMM_X86_INSTR_KMOVB_LD         0xa0051190
#define LIBXSMM_X86_INSTR_KMOVW_LD         0xa0041190
#define LIBXSMM_X86_INSTR_KMOVD_LD         0xa0851190
#define LIBXSMM_X86_INSTR_KMOVQ_LD         0xa0841190
#define LIBXSMM_X86_INSTR_KMOVB_ST         0xa0051191
#define LIBXSMM_X86_INSTR_KMOVW_ST         0xa0041191
#define LIBXSMM_X86_INSTR_KMOVD_ST         0xa0851191
#define LIBXSMM_X86_INSTR_KMOVQ_ST         0xa0841191

/* SSE floating point compute */
#define LIBXSMM_X86_INSTR_XORPD          20063
#define LIBXSMM_X86_INSTR_MULPD          20064
#define LIBXSMM_X86_INSTR_ADDPD          20065
#define LIBXSMM_X86_INSTR_SUBPD          20066
#define LIBXSMM_X86_INSTR_MULSD          20067
#define LIBXSMM_X86_INSTR_ADDSD          20068
#define LIBXSMM_X86_INSTR_SUBSD          20069
#define LIBXSMM_X86_INSTR_XORPS          20070
#define LIBXSMM_X86_INSTR_MULPS          20071
#define LIBXSMM_X86_INSTR_ADDPS          20072
#define LIBXSMM_X86_INSTR_SUBPS          20073
#define LIBXSMM_X86_INSTR_MULSS          20074
#define LIBXSMM_X86_INSTR_ADDSS          20075
#define LIBXSMM_X86_INSTR_SUBSS          20076

/* GP instructions */
#define LIBXSMM_X86_INSTR_MOVB           30000
#define LIBXSMM_X86_INSTR_MOVW           30001
#define LIBXSMM_X86_INSTR_MOVL           30002
#define LIBXSMM_X86_INSTR_MOVQ           30003
#define LIBXSMM_X86_INSTR_ADDQ           30004
#define LIBXSMM_X86_INSTR_SUBQ           30005
#define LIBXSMM_X86_INSTR_CMPQ           30006
#define LIBXSMM_X86_INSTR_JL             30007
#define LIBXSMM_X86_INSTR_VPREFETCH0     30008
#define LIBXSMM_X86_INSTR_VPREFETCH1     30009
#define LIBXSMM_X86_INSTR_PREFETCHT0     30010
#define LIBXSMM_X86_INSTR_PREFETCHT1     30011
#define LIBXSMM_X86_INSTR_PREFETCHT2     30012
#define LIBXSMM_X86_INSTR_PREFETCHNTA    30013
#define LIBXSMM_X86_INSTR_MOVSLQ         30014
#define LIBXSMM_X86_INSTR_SALQ           30015
#define LIBXSMM_X86_INSTR_IMUL           30016
#define LIBXSMM_X86_INSTR_JE             30017
#define LIBXSMM_X86_INSTR_JZ             30018
#define LIBXSMM_X86_INSTR_JG             30019
#define LIBXSMM_X86_INSTR_JNE            30020
#define LIBXSMM_X86_INSTR_JNZ            30021
#define LIBXSMM_X86_INSTR_JGE            30022
#define LIBXSMM_X86_INSTR_JLE            30023
#define LIBXSMM_X86_INSTR_JMP            30024
#define LIBXSMM_X86_INSTR_POPCNT         30025
#define LIBXSMM_X86_INSTR_TZCNT          30026
#define LIBXSMM_X86_INSTR_LEAQ           30027
#define LIBXSMM_X86_INSTR_ANDQ           30028
#define LIBXSMM_X86_INSTR_CLDEMOTE       30029
#define LIBXSMM_X86_INSTR_SHLQ           30030
#define LIBXSMM_X86_INSTR_SARQ           30031
#define LIBXSMM_X86_INSTR_SHRQ           30032
#define LIBXSMM_X86_INSTR_CLFLUSHOPT     30033
#define LIBXSMM_X86_INSTR_CMOVA          30034
#define LIBXSMM_X86_INSTR_CMOVAE         30035
#define LIBXSMM_X86_INSTR_CMOVB          30036
#define LIBXSMM_X86_INSTR_CMOVBE         30037
#define LIBXSMM_X86_INSTR_CMOVC          30038
#define LIBXSMM_X86_INSTR_CMOVE          30039
#define LIBXSMM_X86_INSTR_CMOVG          30040
#define LIBXSMM_X86_INSTR_CMOVGE         30041
#define LIBXSMM_X86_INSTR_CMOVL          30042
#define LIBXSMM_X86_INSTR_CMOVLE         30043
#define LIBXSMM_X86_INSTR_CMOVNA         30044
#define LIBXSMM_X86_INSTR_CMOVNAE        30045
#define LIBXSMM_X86_INSTR_CMOVNB         30046
#define LIBXSMM_X86_INSTR_CMOVNBE        30047
#define LIBXSMM_X86_INSTR_CMOVNC         30048
#define LIBXSMM_X86_INSTR_CMOVNE         30049
#define LIBXSMM_X86_INSTR_CMOVNG         30050
#define LIBXSMM_X86_INSTR_CMOVNGE        30051
#define LIBXSMM_X86_INSTR_CMOVNL         30052
#define LIBXSMM_X86_INSTR_CMOVNLE        30053
#define LIBXSMM_X86_INSTR_CMOVNO         30054
#define LIBXSMM_X86_INSTR_CMOVNP         30055
#define LIBXSMM_X86_INSTR_CMOVNS         30056
#define LIBXSMM_X86_INSTR_CMOVNZ         30057
#define LIBXSMM_X86_INSTR_CMOVO          30058
#define LIBXSMM_X86_INSTR_CMOVP          30059
#define LIBXSMM_X86_INSTR_CMOVPE         30060
#define LIBXSMM_X86_INSTR_CMOVPO         30061
#define LIBXSMM_X86_INSTR_CMOVS          30062
#define LIBXSMM_X86_INSTR_CMOVZ          30063

/* Tile instructions */
/* CPUID: AMX-TILE INTERCEPT: SPR */
#define LIBXSMM_X86_INSTR_LDTILECFG          50001
#define LIBXSMM_X86_INSTR_STTILECFG          50002
#define LIBXSMM_X86_INSTR_TILERELEASE        50003
#define LIBXSMM_X86_INSTR_TILELOADD          0x6007204b
#define LIBXSMM_X86_INSTR_TILELOADDT1        0x6005204b
#define LIBXSMM_X86_INSTR_TILESTORED         0x6006204b
#define LIBXSMM_X86_INSTR_TILEZERO           0x50072049
/* CPUID: AMX-INT8 INTERCEPT: SPR */
#define LIBXSMM_X86_INSTR_TDPBSSD            0x7007205e
#define LIBXSMM_X86_INSTR_TDPBSUD            0x7006205e
#define LIBXSMM_X86_INSTR_TDPBUSD            0x7005205e
#define LIBXSMM_X86_INSTR_TDPBUUD            0x7004205e
/* CPUID: AMX-BF16 INTERCEPT: SPR */
#define LIBXSMM_X86_INSTR_TDPBF16PS          0x7006205c

/* define error codes */
#define LIBXSMM_ERR_GENERAL               90000
#define LIBXSMM_ERR_ALLOC                 90001
#define LIBXSMM_ERR_BUFFER_TOO_SMALL      90002
#define LIBXSMM_ERR_APPEND_STR            90003
#define LIBXSMM_ERR_ARCH_PREC             90004
#define LIBXSMM_ERR_ARCH                  90005
#define LIBXSMM_ERR_UNSUP_ARCH            90006
#define LIBXSMM_ERR_LDA                   90007
#define LIBXSMM_ERR_LDB                   90008
#define LIBXSMM_ERR_LDC                   90009
#define LIBXSMM_ERR_SPGEMM_GEN            90010
#define LIBXSMM_ERR_CSC_INPUT             90011
#define LIBXSMM_ERR_CSC_READ_LEN          90012
#define LIBXSMM_ERR_CSC_READ_DESC         90013
#define LIBXSMM_ERR_CSC_READ_ELEMS        90014
#define LIBXSMM_ERR_CSC_LEN               90015
#define LIBXSMM_ERR_N_BLOCK               90016
#define LIBXSMM_ERR_M_BLOCK               90017
#define LIBXSMM_ERR_K_BLOCK               90018
#define LIBXSMM_ERR_REG_BLOCK             90019
#define LIBXSMM_ERR_NO_AVX512_BCAST       90020
#define LIBXSMM_ERR_NO_AVX512_QFMA        90021
#define LIBXSMM_ERR_NO_INDEX_SCALE_ADDR   90022
#define LIBXSMM_ERR_UNSUPPORTED_JUMP      90023
#define LIBXSMM_ERR_NO_JMPLBL_AVAIL       90024
#define LIBXSMM_ERR_EXCEED_JMPLBL         90025
#define LIBXSMM_ERR_CSC_ALLOC_DATA        90026
#define LIBXSMM_ERR_CSR_ALLOC_DATA        90027
#define LIBXSMM_ERR_CSR_INPUT             90028
#define LIBXSMM_ERR_CSR_READ_LEN          90029
#define LIBXSMM_ERR_CSR_READ_DESC         90030
#define LIBXSMM_ERR_CSR_READ_ELEMS        90031
#define LIBXSMM_ERR_CSR_LEN               90032
#define LIBXSMM_ERR_UNSUP_DATATYPE        90033
#define LIBXSMM_ERR_UNSUP_DT_FORMAT       90034
#define LIBXSMM_ERR_INVALID_GEMM_CONFIG   90035
#define LIBXSMM_ERR_UNIQUE_VAL            90036
#define LIBXSMM_ERR_VEC_REG_MUST_BE_UNDEF 90037
#define LIBXSMM_ERR_JMPLBL_USED           90038
#define LIBXSMM_ERR_TRANS_B               90039
#define LIBXSMM_ERR_LDB_TRANS             90040
#define LIBXSMM_ERR_VNNI_A                90041
#define LIBXSMM_ERR_VNNI_B                90042
#define LIBXSMM_ERR_NO_AVX512VL           90043

#if defined(LIBXSMM_HANDLE_ERROR_QUIET)
# define LIBXSMM_HANDLE_ERROR(GENERATED_CODE, ERROR_CODE)
# define LIBXSMM_HANDLE_ERROR_VERBOSE(GENERATED_CODE, ERROR_CODE)
#else
# define LIBXSMM_HANDLE_ERROR(GENERATED_CODE, ERROR_CODE) libxsmm_handle_error( \
    GENERATED_CODE, ERROR_CODE, LIBXSMM_FUNCNAME, 1 < libxsmm_ninit ? libxsmm_verbosity : 1)
# define LIBXSMM_HANDLE_ERROR_VERBOSE(GENERATED_CODE, ERROR_CODE) libxsmm_handle_error( \
    GENERATED_CODE, ERROR_CODE, LIBXSMM_FUNCNAME, 1)
#endif

/* tile config structure */
typedef struct libxsmm_tile_config {
  unsigned char  palette_id;
  unsigned short tile0rowsb;
  unsigned char  tile0cols;
  unsigned short tile1rowsb;
  unsigned char  tile1cols;
  unsigned short tile2rowsb;
  unsigned char  tile2cols;
  unsigned short tile3rowsb;
  unsigned char  tile3cols;
  unsigned short tile4rowsb;
  unsigned char  tile4cols;
  unsigned short tile5rowsb;
  unsigned char  tile5cols;
  unsigned short tile6rowsb;
  unsigned char  tile6cols;
  unsigned short tile7rowsb;
  unsigned char  tile7cols;
} libxsmm_tile_config;

/* structure for tracking local labels in assembly we don't allow overlapping loops */
LIBXSMM_EXTERN_C typedef struct libxsmm_loop_label_tracker_struct {
  unsigned int label_address[512];
  unsigned int label_count;
} libxsmm_loop_label_tracker;

/* micro kernel configuration */
LIBXSMM_EXTERN_C typedef struct libxsmm_micro_kernel_config {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  unsigned int vector_length;
  unsigned int datatype_size_in;
  unsigned int datatype_size_out;
  unsigned int a_vmove_instruction;
  unsigned int b_vmove_instruction;
  unsigned int b_shuff_instruction;
  unsigned int c_vmove_instruction;
  unsigned int c_vmove_nts_instruction;
  unsigned int use_masking_a_c;
  unsigned int prefetch_instruction;
  unsigned int vxor_instruction;
  unsigned int vmul_instruction;
  unsigned int vadd_instruction;
  unsigned int alu_add_instruction;
  unsigned int alu_sub_instruction;
  unsigned int alu_cmp_instruction;
  unsigned int alu_jmp_instruction;
  unsigned int alu_mov_instruction;
  char vector_name;

  /* Auxiliary variables for GEMM fusion info  */
  unsigned int fused_eltwise;
  unsigned int m_loop_exists;
  unsigned int n_loop_exists;
  unsigned int fused_bcolbias;
  unsigned int fused_scolbias;
  unsigned int fused_relu;
  unsigned int fused_relu_bwd;
  unsigned int fused_sigmoid;
  unsigned int overwrite_C;
  unsigned int vnni_format_C;
  unsigned int sparsity_factor_A;
  unsigned int decompress_A;
  unsigned int vnni_cvt_output_ext_buf;
  unsigned int norm_to_normT_B_ext_buf;

  /* Register names/logistics for fusion boo-keeping  */
  unsigned int reserved_zmms;
  unsigned int reserved_mask_regs;
  unsigned int vnni_perm_reg;
  unsigned int zero_reg;
  unsigned int vec_x2;
  unsigned int vec_nom;
  unsigned int vec_denom;
  unsigned int vec_c0;
  unsigned int vec_c1;
  unsigned int vec_c2;
  unsigned int vec_c3;
  unsigned int vec_c1_d;
  unsigned int vec_c2_d;
  unsigned int vec_c3_d;
  unsigned int vec_hi_bound;
  unsigned int vec_lo_bound;
  unsigned int vec_ones;
  unsigned int vec_neg_ones;
  unsigned int vec_halves;
  unsigned int mask_hi;
  unsigned int mask_lo;
  unsigned int perm_table_vnni_lo;
  unsigned int perm_table_vnni_hi;
  unsigned int norm_to_normT_mask_reg_0;
  unsigned int norm_to_normT_mask_reg_1;

  /* Auxiliary arrays for micro-kernel iteration space traversal */
  int use_paired_tilestores;
  int _im[4];
  int _in[4];
  int _C_tile_id[4];
  int _C_tile_mate_id[4];
  int _im_offset_prefix_sums[4];
  int _in_offset_prefix_sums[4];

  /* Auxiliary data structure and fields when emulating AMX instructions */
  libxsmm_tile_config tile_config;
  unsigned int emulation_scratch_offset;
  unsigned int lda_emu;
  unsigned int ldb_emu;
  unsigned int ldc_emu;
  unsigned int emulate_cvt2bf16fp32;
  unsigned int emulate_cvt2bf16fp32_vperm;
  unsigned int emulate_cvt2bf16fp32_vaux;
  unsigned int emulate_cvt2bf16fp32_vaux0;
  unsigned int emulate_cvt2bf16fp32_vaux1;
  unsigned int mask_cvt_hi;
  unsigned int mask_cvt_lo;
  libxsmm_loop_label_tracker *io_loop_label_tracker;

  /* Auxiliary fields to propagate kernel info */
  unsigned int k_amx_microkernel;
  unsigned int B_offs_trans;
  unsigned int stride_b_trans;

} libxsmm_micro_kernel_config;

/* structure for storing the current gp reg mapping */
LIBXSMM_EXTERN_C typedef struct libxsmm_gp_reg_mapping_struct {
  unsigned int gp_reg_a;
  unsigned int gp_reg_a_base;
  unsigned int gp_reg_b;
  unsigned int gp_reg_b_base;
  unsigned int gp_reg_c;
  unsigned int gp_reg_a_prefetch;
  unsigned int gp_reg_a_offset;
  unsigned int gp_reg_b_prefetch;
  unsigned int gp_reg_b_offset;
/*  unsigned int gp_reg_c_prefetch;*/
  unsigned int gp_reg_mloop;
  unsigned int gp_reg_nloop;
  unsigned int gp_reg_kloop;
  unsigned int gp_reg_reduce_count;
  unsigned int gp_reg_reduce_loop;
  unsigned int gp_reg_a_ptrs;
  unsigned int gp_reg_b_ptrs;
  unsigned int gp_reg_lda;
  unsigned int gp_reg_ldb;
  unsigned int gp_reg_ldc;
  unsigned int gp_reg_scf;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
/* Auxiliary regs for sparsity in A support  */
  unsigned int gp_reg_bitmap_a;
  unsigned int gp_reg_decompressed_a;
} libxsmm_gp_reg_mapping;

/* structure for storing the current gp reg mapping for matcopy */
LIBXSMM_EXTERN_C typedef struct libxsmm_matcopy_gp_reg_mapping_struct {
  unsigned int gp_reg_a;
  unsigned int gp_reg_lda;
  unsigned int gp_reg_b;
  unsigned int gp_reg_ldb;
  unsigned int gp_reg_a_pf;
  unsigned int gp_reg_b_pf;
  unsigned int gp_reg_m_loop;
  unsigned int gp_reg_n_loop;
  unsigned int gp_reg_help_0;
} libxsmm_matcopy_gp_reg_mapping;

/* matcopy kernel configuration */
LIBXSMM_EXTERN_C typedef struct libxsmm_matcopy_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  unsigned int vector_length;
  unsigned int datatype_size;
  unsigned int prefetch_instruction;
  unsigned int vmove_instruction;
  unsigned int alu_add_instruction;
  unsigned int alu_cmp_instruction;
  unsigned int alu_jmp_instruction;
  unsigned int alu_mov_instruction;
  unsigned int vxor_instruction;
  char vector_name;
} libxsmm_matcopy_kernel_config;

/* structure for storing the current gp reg mapping for mateltwise */
LIBXSMM_EXTERN_C typedef struct libxsmm_mateltwise_gp_reg_mapping_struct {
  unsigned int gp_reg_param_struct;
  unsigned int gp_reg_in;
  unsigned int gp_reg_in2;
  unsigned int gp_reg_in_pf;
  unsigned int gp_reg_ldi;
  unsigned int gp_reg_out;
  unsigned int gp_reg_ldo;
  unsigned int gp_reg_relumask;
  unsigned int gp_reg_fam_lualpha;
  unsigned int gp_reg_offset;
  unsigned int gp_reg_dropoutmask;
  unsigned int gp_reg_dropoutprob;
  unsigned int gp_reg_prngstate;
  unsigned int gp_reg_reduced_elts;
  unsigned int gp_reg_reduced_elts_squared;
  unsigned int gp_reg_scale_vals;
  unsigned int gp_reg_shift_vals;
  unsigned int gp_reg_bias_vals;
  unsigned int gp_reg_scale_vals2;
  unsigned int gp_reg_shift_vals2;
  unsigned int gp_reg_bias_vals2;
  unsigned int gp_reg_m_loop;
  unsigned int gp_reg_n_loop;
  unsigned int gp_reg_n;
  unsigned int gp_reg_ind_base;
  unsigned int gp_reg_in_base;
  unsigned int gp_reg_invec;
  unsigned int gp_reg_ind_base2;
  unsigned int gp_reg_in_base2;
  unsigned int gp_reg_in_pf2;
  unsigned int gp_reg_scale_base;
} libxsmm_mateltwise_gp_reg_mapping;

/* mateltwise kernel configuration */
LIBXSMM_EXTERN_C typedef struct libxsmm_mateltwise_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  unsigned int vector_length_in;
  unsigned int vector_length_out;
  unsigned int datatype_size_in;
  unsigned int datatype_size_out;
  unsigned int vmove_instruction_in;
  unsigned int vmove_instruction_out;
  unsigned int alu_add_instruction;
  unsigned int alu_sub_instruction;
  unsigned int alu_cmp_instruction;
  unsigned int alu_jmp_instruction;
  unsigned int alu_mov_instruction;
  unsigned int vxor_instruction;

  /* Auxiliary varialiables for vreg management  */
  unsigned int reserved_zmms;
  unsigned int reserved_mask_regs;
  unsigned int use_fp32bf16_cvt_replacement;
  unsigned int dcvt_mask_aux0;
  unsigned int dcvt_mask_aux1;
  unsigned int dcvt_zmm_aux0;
  unsigned int dcvt_zmm_aux1;
  unsigned int inout_vreg_mask;
  unsigned int tmp_vreg;
  unsigned int tmp_vreg2;
  unsigned int tmp_vreg3;
  unsigned int zero_vreg;
  unsigned int vec_x2;
  unsigned int vec_nom;
  unsigned int vec_denom;
  unsigned int vec_c0;
  unsigned int vec_c1;
  unsigned int vec_c2;
  unsigned int vec_c3;
  unsigned int vec_c1_d;
  unsigned int vec_c2_d;
  unsigned int vec_c3_d;
  unsigned int vec_hi_bound;
  unsigned int vec_lo_bound;
  unsigned int vec_ones;
  unsigned int vec_neg_ones;
  unsigned int vec_halves;
  unsigned int mask_hi;
  unsigned int mask_lo;

  /* Additional aux variables for exp  */
  unsigned int vec_log2e;
  unsigned int vec_y;
  unsigned int vec_z;
  unsigned int vec_expmask;

  /* Additional aux variables for gelu */
  unsigned int vec_xr;
  unsigned int vec_xa;
  unsigned int vec_index;
  unsigned int vec_C0;
  unsigned int vec_C1;
  unsigned int vec_C2;
  unsigned int vec_thres;
  unsigned int vec_absmask;
  unsigned int vec_scale;
  unsigned int vec_shifter;

  /* Additional aux variables fir minimax approximations */
  unsigned int vec_c0_lo;
  unsigned int vec_c0_hi;
  unsigned int vec_c1_lo;
  unsigned int vec_c1_hi;
  unsigned int vec_c2_lo;
  unsigned int vec_c2_hi;
  unsigned int vec_tmp0;
  unsigned int vec_tmp1;
  unsigned int vec_tmp2;
  unsigned int vec_tmp3;
  unsigned int vec_tmp4;
  unsigned int vec_tmp5;
  unsigned int vec_tmp6;
  unsigned int vec_tmp7;
  int rbp_offs_thres;
  int rbp_offs_signmask;
  int rbp_offs_absmask;
  int rbp_offs_scale;
  int rbp_offs_shifter;
  int rbp_offs_half;

  /* Aux variables for relu variants */
  unsigned int fam_lu_vreg_alpha;

  /* Aux variable for dropout */
  unsigned int prng_state0_vreg;
  unsigned int prng_state1_vreg;
  unsigned int prng_state2_vreg;
  unsigned int prng_state3_vreg;
  unsigned int dropout_vreg_tmp0;
  unsigned int dropout_vreg_tmp1;
  unsigned int dropout_vreg_tmp2;
  unsigned int dropout_vreg_one;
  unsigned int dropout_vreg_zero;
  unsigned int dropout_prob_vreg;
  unsigned int dropout_invprob_vreg;
  unsigned int dropout_vreg_avxmask;

  /* Misc aux variables  */
  unsigned int neg_signs_vreg;

  /* Aux variables for kernel config  */
  unsigned int vlen_in;
  unsigned int vlen_out;
  unsigned int vlen_comp;
  unsigned int loop_order;
  unsigned int skip_pushpops_callee_gp_reg;
  unsigned int use_stack_vars;
  char vector_name;
} libxsmm_mateltwise_kernel_config;

/* structure for storing the current gp reg mapping for matequation */
LIBXSMM_EXTERN_C typedef struct libxsmm_matequation_gp_reg_mapping_struct {
  unsigned int                      gp_reg_param_struct;
  unsigned int gp_reg_in;
  unsigned int gp_reg_in2;
  unsigned int gp_reg_in_pf;
  unsigned int gp_reg_ldi;
  unsigned int gp_reg_out;
  unsigned int gp_reg_ldo;
  unsigned int gp_reg_relumask;
  unsigned int gp_reg_m_loop;
  unsigned int gp_reg_n_loop;
  unsigned int gp_reg_n;
  unsigned int gp_reg_offset;
  unsigned int temp_reg;
  unsigned int temp_reg2;
  libxsmm_mateltwise_gp_reg_mapping gp_reg_mapping_eltwise;
  libxsmm_gp_reg_mapping            gp_reg_mapping_gemm;
} libxsmm_matequation_gp_reg_mapping;

/* matequation kernel configuration */
LIBXSMM_EXTERN_C typedef struct libxsmm_matequation_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  unsigned int vector_length_in;
  unsigned int vector_length_out;
  unsigned int datatype_size_in;
  unsigned int datatype_size_out;
  unsigned int vmove_instruction_in;
  unsigned int vmove_instruction_out;
  unsigned int alu_add_instruction;
  unsigned int alu_sub_instruction;
  unsigned int alu_cmp_instruction;
  unsigned int alu_jmp_instruction;
  unsigned int alu_mov_instruction;
  unsigned int vxor_instruction;
  unsigned int skip_pushpops_callee_gp_reg;
  unsigned int n_args;
  unsigned int vlen_in;
  unsigned int vlen_comp;
  unsigned int vlen_out;
  char vector_name;
  unsigned int                      is_head_reduce_to_scalar;
  unsigned int                      inout_vreg_mask;
  unsigned int                      out_mask;
  unsigned int                      cvt_result_to_bf16;
  unsigned int                      use_fp32bf16_cvt_replacement;
  unsigned int                      dcvt_mask_aux0;
  unsigned int                      dcvt_mask_aux1;
  unsigned int                      dcvt_zmm_aux0;
  unsigned int                      dcvt_zmm_aux1;
  unsigned int                      reduce_vreg;
  unsigned int                      n_avail_gpr;
  unsigned int                      gpr_pool[16];
  unsigned int                      n_tmp_reg_blocks;
  unsigned int                      contains_binary_op;
  unsigned int                      contains_ternary_op;
  unsigned int                      tmp_size;
  libxsmm_matrix_eqn_arg            *arg_info;
  unsigned int                      reserved_zmms;
  unsigned int                      reserved_mask_regs;
  unsigned int                      register_block_size;
  unsigned int                      unary_ops_pool[64];
  unsigned int                      binary_ops_pool[64];
  libxsmm_mateltwise_kernel_config  meltw_kernel_config;
  libxsmm_micro_kernel_config       gemm_kernel_config;
} libxsmm_matequation_kernel_config;

/* structure for storing the current gp reg mapping for transpose */
LIBXSMM_EXTERN_C typedef struct libxsmm_transpose_gp_reg_mapping_struct {
  unsigned int gp_reg_a;
  unsigned int gp_reg_lda;
  unsigned int gp_reg_b;
  unsigned int gp_reg_ldb;
  unsigned int gp_reg_m_loop;
  unsigned int gp_reg_n_loop;
  unsigned int gp_reg_help_0;
  unsigned int gp_reg_help_1;
  unsigned int gp_reg_help_2;
  unsigned int gp_reg_help_3;
  unsigned int gp_reg_help_4;
  unsigned int gp_reg_help_5;
} libxsmm_transpose_gp_reg_mapping;

/* transpose kernel configuration */
LIBXSMM_EXTERN_C typedef struct libxsmm_transpose_kernel_config_struct {
  unsigned int instruction_set;
  unsigned int vector_reg_count;
  char vector_name;
} libxsmm_transpose_kernel_config;

/* structure to save jump properties to the same destination */
LIBXSMM_EXTERN_C typedef struct libxsmm_jump_source_struct {
  unsigned int instr_type[512];
  unsigned int instr_addr[512];
  unsigned int ref_count;
} libxsmm_jump_source;

/* structure for tracking arbitrary jump labels in assembly code */
LIBXSMM_EXTERN_C typedef struct libxsmm_jump_label_tracker_struct {
  unsigned int        label_address[512];
  libxsmm_jump_source label_source[512];
} libxsmm_jump_label_tracker;

LIBXSMM_EXTERN_C typedef struct libxsmm_blocking_info_t {
  unsigned int tiles;
  unsigned int sizes[4];
  unsigned int blocking;
  unsigned int block_size;
} libxsmm_blocking_info_t;

/* Auxiliary stack variable enumeration for kernels */
typedef enum libxsmm_meltw_stack_var {
  LIBXSMM_MELTW_STACK_VAR_NONE            =  0,
  LIBXSMM_MELTW_STACK_VAR_INP0_PTR0       =  1,
  LIBXSMM_MELTW_STACK_VAR_INP0_PTR1       =  2,
  LIBXSMM_MELTW_STACK_VAR_INP0_PTR2       =  3,
  LIBXSMM_MELTW_STACK_VAR_INP1_PTR0       =  4,
  LIBXSMM_MELTW_STACK_VAR_INP1_PTR1       =  5,
  LIBXSMM_MELTW_STACK_VAR_INP1_PTR2       =  6,
  LIBXSMM_MELTW_STACK_VAR_INP2_PTR0       =  7,
  LIBXSMM_MELTW_STACK_VAR_INP2_PTR1       =  8,
  LIBXSMM_MELTW_STACK_VAR_INP2_PTR2       =  9,
  LIBXSMM_MELTW_STACK_VAR_OUT_PTR0        =  10,
  LIBXSMM_MELTW_STACK_VAR_OUT_PTR1        =  11,
  LIBXSMM_MELTW_STACK_VAR_OUT_PTR2        =  12,
  LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR     =  13,
  LIBXSMM_MELTW_STACK_VAR_CONST_0         =  14,
  LIBXSMM_MELTW_STACK_VAR_CONST_1         =  15,
  LIBXSMM_MELTW_STACK_VAR_CONST_2         =  16,
  LIBXSMM_MELTW_STACK_VAR_CONST_3         =  17,
  LIBXSMM_MELTW_STACK_VAR_CONST_4         =  18,
  LIBXSMM_MELTW_STACK_VAR_CONST_5         =  19,
  LIBXSMM_MELTW_STACK_VAR_CONST_6         =  20,
  LIBXSMM_MELTW_STACK_VAR_CONST_7         =  21,
  LIBXSMM_MELTW_STACK_VAR_CONST_8         =  22,
  LIBXSMM_MELTW_STACK_VAR_CONST_9         =  23
} libxsmm_meltw_stack_var;

typedef enum libxsmm_meqn_stack_var {
  LIBXSMM_MEQN_STACK_VAR_NONE               =  0,
  LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR        =  1,
  LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR   =  2,
  LIBXSMM_MEQN_STACK_VAR_OUT_PTR            =  3,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR0  =  4,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR1  =  5,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR2  =  6,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR3  =  7,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4  =  8,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR5  =  9,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR6  =  10,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR7  =  11,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8  =  12,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9  =  13,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR10 =  14,
  LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR11 =  15,
  LIBXSMM_MEQN_STACK_VAR_CONST_0            =  16,
  LIBXSMM_MEQN_STACK_VAR_CONST_1            =  17,
  LIBXSMM_MEQN_STACK_VAR_CONST_2            =  18,
  LIBXSMM_MEQN_STACK_VAR_CONST_3            =  19,
  LIBXSMM_MEQN_STACK_VAR_CONST_4            =  20,
  LIBXSMM_MEQN_STACK_VAR_CONST_5            =  21,
  LIBXSMM_MEQN_STACK_VAR_CONST_6            =  22,
  LIBXSMM_MEQN_STACK_VAR_CONST_7            =  23,
  LIBXSMM_MEQN_STACK_VAR_CONST_8            =  24,
  LIBXSMM_MEQN_STACK_VAR_CONST_9            =  25
} libxsmm_meqn_stack_var;

/* Auxiliary stack variable enumeration in GEMM */
typedef enum libxsmm_gemm_stack_var {
  LIBXSMM_GEMM_STACK_VAR_NONE               =  0,
  LIBXSMM_GEMM_STACK_VAR_PFA_PTR            =  1,
  LIBXSMM_GEMM_STACK_VAR_PFB_PTR            =  2,
  LIBXSMM_GEMM_STACK_VAR_A_OFFS_BRGEMM_PTR  =  3,
  LIBXSMM_GEMM_STACK_VAR_B_OFFS_BRGEMM_PTR  =  4,
  LIBXSMM_GEMM_STACK_VAR_INT8_SCF           =  5,
  LIBXSMM_GEMM_STACK_VAR_GEMM_SCRATCH_PTR   =  6,
  LIBXSMM_GEMM_STACK_VAR_ELT_BIAS_PTR       =  7,
  LIBXSMM_GEMM_STACK_VAR_ELT_OUTPUT_PTR     =  8,
  LIBXSMM_GEMM_STACK_VAR_ARG_7              =  9,
  LIBXSMM_GEMM_STACK_VAR_ARG_8              = 10,
  LIBXSMM_GEMM_STACK_VAR_ARG_9              = 11,
  LIBXSMM_GEMM_STACK_VAR_ARG_10             = 12,
  LIBXSMM_GEMM_STACK_VAR_ELT_BUF1           = 13,
  LIBXSMM_GEMM_STACK_VAR_ELT_BUF2           = 14,
  LIBXSMM_GEMM_STACK_VAR_ELT_BITMAP_PTR     = 15,
  LIBXSMM_GEMM_STACK_VAR_ELT_DECOMPRESS_BUF = 16,
  LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_B    = 17,
  LIBXSMM_GEMM_STACK_VAR_TRANS_EXT_BUF_C    = 18,
  LIBXSMM_GEMM_STACK_VAR_ELT_RELU_BITMASK_PTR    = 19
} libxsmm_gemm_stack_var;

#if 0
/* compressed meltw reduce structure */
typedef enum libxsmm_meltw_comp_redu_flags {
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_NONE         = 0,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD       = 1,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_MAX       = 2,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_MUL       = 3,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_ROWS         = 4,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_COLS         = 5,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_ELTS         = 6,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_ELTS_SQUARED = 7,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD_ROWS  = 8,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD_COLS  = 9,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD_ROWS_ELTS_ELTS_SQUARED  = 10,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD_COLS_ELTS_ELTS_SQUARED  = 11,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD_ROWS_ELTS               = 12,
  LIBXSMM_MELTW_COMP_FLAG_REDUCE_OP_ADD_COLS_ELTS               = 13
} libxsmm_meltw_comp_redu_flags;

/* compressed meltw relu structure */
typedef enum libxsmm_meltw_comp_relu_flags {
  LIBXSMM_MELTW_COMP_FLAG_RELU_NONE         = 0,
  LIBXSMM_MELTW_COMP_FLAG_RELU_FWD          = 1,
  LIBXSMM_MELTW_COMP_FLAG_RELU_BWD          = 2
} libxsmm_meltw_comp_relu_flags;

/* compressed meltw scale structure */
typedef enum libxsmm_meltw_comp_scal_flags {
  LIBXSMM_MELTW_COMP_FLAG_SCALE_NONE                     = 0,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT                     = 1,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_SHIFT                    = 2,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS                 = 3,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ROWS                     = 4,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_COLS                     = 5,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_ROWS                = 6,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_SHIFT_ROWS               = 7,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS_ROWS            = 8,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_SHIFT_ROWS          = 9,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS_SHIFT_ROWS      = 10,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_ADD_BIAS_ROWS       = 11,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_SHIFT_ADD_BIAS_ROWS = 12,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_COLS                = 13,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_SHIFT_COLS               = 14,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS_COLS            = 15,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_SHIFT_COLS          = 16,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS_SHIFT_COLS      = 17,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_ADD_BIAS_COLS       = 18,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_SHIFT_ADD_BIAS_COLS = 19,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_ROWS_COLS                = 20,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_SHIFT_ROWS_COLS               = 21,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS_ROWS_COLS            = 22,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_SHIFT_ROWS_COLS          = 23,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_ADD_BIAS_SHIFT_ROWS_COLS      = 24,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_ADD_BIAS_ROWS_COLS       = 25,
  LIBXSMM_MELTW_COMP_FLAG_SCALE_MULT_SHIFT_ADD_BIAS_ROWS_COLS = 26
} libxsmm_meltw_comp_scal_flags;

/* compressed metlw cvta strcuture */
typedef enum libxsmm_meltw_comp_cvta_flags {
  LIBXSMM_MELTW_COMP_FLAG_CVTA_NONE           = 0,
  LIBXSMM_MELTW_COMP_FLAG_CVTA_FUSE_RELU      = 1,
  LIBXSMM_MELTW_COMP_FLAG_CVTA_FUSE_TANH      = 2,
  LIBXSMM_MELTW_COMP_FLAG_CVTA_FUSE_SIGM      = 3
} libxsmm_meltw_comp_cvta_flags;

/* compressed meltw acvt structure */
typedef enum libxsmm_meltw_comp_acvt_flags {
  LIBXSMM_MELTW_COMP_FLAG_ACVT_NONE           = 0,
  LIBXSMM_MELTW_COMP_FLAG_ACVT_FUSE_TANH      = 1,
  LIBXSMM_MELTW_COMP_FLAG_ACVT_FUSE_SIGM      = 2
} libxsmm_meltw_comp_acvt_flags;

/* compressed meltw cbiasact strcuture */
typedef enum libxsmm_meltw_comp_flags {
  LIBXSMM_MELTW_COMP_FLAG_NONE                         =  0,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS                      =  1,
  LIBXSMM_MELTW_COMP_FLAG_ACT_RELU                     =  2,
  LIBXSMM_MELTW_COMP_FLAG_ACT_TANH                     =  3,
  LIBXSMM_MELTW_COMP_FLAG_ACT_SIGM                     =  4,
  LIBXSMM_MELTW_COMP_FLAG_ACT_GELU                     =  5,
  LIBXSMM_MELTW_COMP_FLAG_OVERWRITE_C                  =  6,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_RELU             =  7,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_TANH             =  8,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_SIGM             =  9,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_GELU             = 10,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_RELU_OVERWRITE_C = 11,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_TANH_OVERWRITE_C = 12,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_SIGM_OVERWRITE_C = 13,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_ACT_GELU_OVERWRITE_C = 14,
  LIBXSMM_MELTW_COMP_FLAG_COLBIAS_OVERWRITE_C          = 15,
  LIBXSMM_MELTW_COMP_FLAG_ACT_RELU_OVERWRITE_C         = 16,
  LIBXSMM_MELTW_COMP_FLAG_ACT_TANH_OVERWRITE_C         = 17,
  LIBXSMM_MELTW_COMP_FLAG_ACT_SIGM_OVERWRITE_C         = 18,
  LIBXSMM_MELTW_COMP_FLAG_ACT_GELU_OVERWRITE_C         = 19
} libxsmm_meltw_comp_flags;
#endif

LIBXSMM_API_INTERN
void libxsmm_reset_loop_label_tracker( libxsmm_loop_label_tracker* io_loop_label_tracker );

LIBXSMM_API_INTERN
void libxsmm_reset_jump_label_tracker( libxsmm_jump_label_tracker* io_jump_lable_tracker );

LIBXSMM_API_INTERN
void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              o_gp_reg_name,
                                  const int          i_gp_reg_name_max_length );

LIBXSMM_API_INTERN
unsigned int libxsmm_check_x86_gp_reg_callee_save( const unsigned int i_gp_reg_number );

LIBXSMM_API_INTERN
void libxsmm_get_x86_instr_name( const unsigned int i_instr_number,
                                 char*              o_instr_name,
                                 const int          i_instr_name_max_length );

LIBXSMM_API_INTERN
void libxsmm_reset_x86_gp_reg_mapping( libxsmm_gp_reg_mapping* io_gp_reg_mapping );

LIBXSMM_API_INTERN
void libxsmm_reset_aarch64_gp_reg_mapping( libxsmm_gp_reg_mapping* io_gp_reg_mapping );

LIBXSMM_API_INTERN
unsigned int libxsmm_is_x86_vec_instr_single_precision( const unsigned int i_instr_number );

/* some string manipulation helper needed to generated code */
LIBXSMM_API_INTERN
void libxsmm_append_code_as_string( libxsmm_generated_code* io_generated_code,
                                    const char*             i_code_to_append,
                                    const int               i_append_length );

LIBXSMM_API_INTERN
void libxsmm_close_function( libxsmm_generated_code* io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_mmfunction_signature( libxsmm_generated_code*       io_generated_code,
                                  const char*                    i_routine_name,
                                  const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_isa_check_header( libxsmm_generated_code* io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_generator_isa_check_footer( libxsmm_generated_code* io_generated_code );

LIBXSMM_API_INTERN
void libxsmm_handle_error( libxsmm_generated_code* io_generated_code,
                           const unsigned int      i_error_code,
                           const char*             context,
                           int emit_message );

LIBXSMM_API_INTERN unsigned int libxsmm_compute_equalized_blocking(
  unsigned int i_size, unsigned int i_max_block,
  unsigned int* o_range_1, unsigned int* o_block_1,
  unsigned int* o_range_2, unsigned int* o_block_2 );

#endif /* GENERATOR_COMMON_H */

