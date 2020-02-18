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
#ifndef GENERATOR_PACKED_GEMMNN_H
#define GENERATOR_PACKED_GEMMNN_H

#include "generator_x86_instructions.h"
#include "generator_common.h"
#include <libxsmm_intrinsics_x86.h>


/*
   Does C(cm1:cm2,cn1:cn2) <- alpha*A(am1:am2,ak1:ak2)*B(bk1:bk2,bn1:bn2) +
                              beta *C(cm1:cm2,cn1:cn2)
   Obviously, the dimensions must conform
   Alpha and Beta are doubles even if the work should be single. Just convert
*/

LIBXSMM_API_INLINE void compact_gemmnn_ (
     unsigned int tra, /* 0 if non-transpose */
     unsigned int trb, /* 0 if non-transpose */
     unsigned int trc, /* 0 if non-transpose */
     unsigned int am1,
     unsigned int am2,
     unsigned int ak1,
     unsigned int ak2,
     unsigned int bk1,
     unsigned int bk2,
     unsigned int bn1,
     unsigned int bn2,
     unsigned int cm1,
     unsigned int cm2,
     unsigned int cn1,
     unsigned int cn2,
     double alpha,
     unsigned int areg,
     unsigned int lda,
     unsigned int breg,
     unsigned int ldb,
     double beta,
     unsigned int creg,
     unsigned int ldc,
     libxsmm_generated_code* io_code,
     unsigned int numb,
     char regset,
     unsigned int iunroll,
     unsigned int junroll,
     unsigned int loopi,
     unsigned int loopj )
{
     unsigned int i, j, l, datasz, nloopcnt=0, mloopcnt=0, nborder, mborder, mloopadj = 0;
     int aoffset, boffset, coffset; /* Address calcs for the loops */
     unsigned int iun=3, jun=3; /* Register blocking sizes */

     int a0 = -1, a1 = -1, a2 = -1, a3 = -1, a4 = -1, a5 = -1, a6 = -1, a7 = -1;
     int b0 = -1, b1 = -1, b2 = -1, b3 = -1, b4 = -1, b5 = -1, b6 = -1, b7 = -1;
     int c00 = -1, c01 = -1, c02 = -1, c03 = -1, c04 = -1, c05 = -1, c06 = -1, c07 = -1;
     int c10 = -1, c11 = -1, c12 = -1, c13 = -1, c14 = -1, c15 = -1, c16 = -1, c17 = -1;
     int c20 = -1, c21 = -1, c22 = -1, c23 = -1, c24 = -1, c25 = -1, c26 = -1, c27 = -1;
     int c30 = -1, c31 = -1, c32 = -1, c33 = -1, c34 = -1, c35 = -1, c36 = -1, c37 = -1;
     int c40 = -1, c41 = -1, c42 = -1, c43 = -1, c44 = -1, c45 = -1, c46 = -1, c47 = -1;
     int c50 = -1, c51 = -1, c52 = -1, c53 = -1, c54 = -1, c55 = -1, c56 = -1, c57 = -1;
     int c60 = -1, c61 = -1, c62 = -1, c63 = -1, c64 = -1, c65 = -1, c66 = -1, c67 = -1;
     int c70 = -1, c71 = -1, c72 = -1, c73 = -1, c74 = -1, c75 = -1, c76 = -1, c77 = -1;
     int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
     int j0 = 1, j1, j2, j3, j4, j5, j6, j7;
     int i0 = 1, i1, i2, i3, i4, i5, i6, i7;
     unsigned int maxregblocking = 8, maxreg = 16;

     libxsmm_loop_label_tracker l_loop_label_tracker;

     /* Test that the dimensions conform */
     if ( (am2-am1) != (cm2-cm1) ) {
        printf("compact_gemmnn m-dimensions don't conform: %u != %u\n",am2-am1+1,cm2-cm1+1);
        exit(-1);
     }
     if ( (ak2-ak1) != (bk2-bk1) ) {
        printf("compact_gemmnn k-dimensions don't conform: %u != %u\n",ak2-ak1+1,bk2-bk1+1);
        exit(-1);
     }
     if ( (bn2-bn1) != (cn2-cn1) ) {
        printf("compact_gemmnn n-dimensions don't conform: %u != %u\n",ak2-ak1+1,bk2-bk1+1);
        exit(-1);
     }

     /* See that all dimensions are at least 1 */
     if ( am2 < am1) {
        printf("compact_gemmnn m-dimension too small: %u\n",am2-am1+1);
        exit(-1);
     }
     if ( ak2 < ak1) {
        printf("compact_gemmnn k-dimension too small: %u\n",ak2-ak1+1);
        exit(-1);
     }
     if ( bn2 < bn1) {
        printf("compact_gemmnn n-dimension too small: %u\n",bn2-bn1+1);
        exit(-1);
     }

     /* Check that areg, breg, creg is valid */
     if ( /*(areg < 0) ||*/ (areg > 15) ) {
        printf("compact_gemmnn A gp register invalid: %u\n",areg);
        exit(-1);
     }
     if ( /*(breg < 0) ||*/ (breg > 15) ) {
        printf("compact_gemmnn B gp register invalid: %u\n",breg);
        exit(-1);
     }
     if ( /*(creg < 0) ||*/ (creg > 15) ) {
        printf("compact_gemmnn C gp register invalid: %u\n",creg);
        exit(-1);
     }

     if ( (numb == 8) && (regset=='z') ) { datasz = 8; }
     else if ( (numb == 16) && (regset=='z') ) { datasz = 4; }
     else if ( (numb == 8) && (regset=='y') ) { datasz = 4; }
     else if ( (numb == 4) && (regset=='y') ) { datasz = 8; }
     else {
        printf("compact_gemmnn Unknown number=%u or regset=%c\n",numb,regset);
        exit(-1);
     }

     if ( regset == 'y' ) { iun = 3; jun = 3; maxreg=16; maxregblocking= 7; }
     if ( regset == 'z' ) { iun = 5; jun = 4; maxreg=32; maxregblocking= 8; }

     if ( iunroll > 0 ) iun = iunroll;
     if ( junroll > 0 ) jun = junroll;

     /* Make sure values of register blocking are between 1 and maxregblocking */
     iun = LIBXSMM_MAX(LIBXSMM_MIN(iun,maxregblocking),1);
     jun = LIBXSMM_MAX(LIBXSMM_MIN(jun,maxregblocking),1);

     /* CHeck to see the register blocking parameters make sense: */
     if ( maxreg < 3 ) {
        printf("Sorry, not enough registers available in compact gemm nn\n");
        exit(-1);
     }
     while ( iun+jun+iun*jun > maxreg ) {
        if ( (iun >= jun) && (iun > 1) ) --iun;
        else if ( (jun >= iun) && (jun > 1) ) --jun;
        else {
           printf("Seems strange that we can't reduce the registers in compact gemm nn\n");
           exit(-1);
        }
     }

     /* Determine if the problem is too small for loops giving this register blocking */
     mloopcnt = (int)((am2-am1+1)/iun);
     nloopcnt = (int)((bn2-bn1+1)/jun);
     if ( mloopcnt < 2 ) loopi = 0;
     if ( nloopcnt < 2 ) loopj = 0;
     mborder = (am2-am1+1)-mloopcnt*iun;
     nborder = (bn2-bn1+1)-nloopcnt*jun;
     if ( loopj || loopi ) {
        libxsmm_reset_loop_label_tracker ( &l_loop_label_tracker );
     }

     /* DO register blocking */
     a0 = 0;
     if ( iun > 1 ) a1 = 1;
     if ( iun > 2 ) a2 = 2;
     if ( iun > 3 ) a3 = 3;
     if ( iun > 4 ) a4 = 4;
     if ( iun > 5 ) a5 = 5;
     if ( iun > 6 ) a6 = 6;
     if ( iun > 7 ) a7 = 7;
     b0 = iun;
     if ( jun > 1 ) b1 = b0+1;
     if ( jun > 2 ) b2 = b0+2;
     if ( jun > 3 ) b3 = b0+3;
     if ( jun > 4 ) b4 = b0+4;
     if ( jun > 5 ) b5 = b0+5;
     if ( jun > 6 ) b6 = b0+6;
     if ( jun > 7 ) b7 = b0+7;
     c00 = iun + jun;
     if ( jun > 1 ) c01 = c00 + 1;
     if ( jun > 2 ) c02 = c00 + 2;
     if ( jun > 3 ) c03 = c00 + 3;
     if ( jun > 4 ) c04 = c00 + 4;
     if ( jun > 5 ) c05 = c00 + 5;
     if ( jun > 6 ) c06 = c00 + 6;
     if ( jun > 7 ) c07 = c00 + 7;
     if ( iun > 1 ) c10 = c00 + jun;
     if ( (iun > 1) && (jun > 1) ) c11 = c10 + 1;
     if ( (iun > 1) && (jun > 2) ) c12 = c10 + 2;
     if ( (iun > 1) && (jun > 3) ) c13 = c10 + 3;
     if ( (iun > 1) && (jun > 4) ) c14 = c10 + 4;
     if ( (iun > 1) && (jun > 5) ) c15 = c10 + 5;
     if ( (iun > 1) && (jun > 6) ) c16 = c10 + 6;
     if ( (iun > 1) && (jun > 7) ) c17 = c10 + 7;
     if ( iun > 2 ) c20 = c10 + jun;
     if ( (iun > 2) && (jun > 1) ) c21 = c20 + 1;
     if ( (iun > 2) && (jun > 2) ) c22 = c20 + 2;
     if ( (iun > 2) && (jun > 3) ) c23 = c20 + 3;
     if ( (iun > 2) && (jun > 4) ) c24 = c20 + 4;
     if ( (iun > 2) && (jun > 5) ) c25 = c20 + 5;
     if ( (iun > 2) && (jun > 6) ) c26 = c20 + 6;
     if ( (iun > 2) && (jun > 7) ) c27 = c20 + 7;
     if ( iun > 3 ) c30 = c20 + jun;
     if ( (iun > 3) && (jun > 1) ) c31 = c30 + 1;
     if ( (iun > 3) && (jun > 2) ) c32 = c30 + 2;
     if ( (iun > 3) && (jun > 3) ) c33 = c30 + 3;
     if ( (iun > 3) && (jun > 4) ) c34 = c30 + 4;
     if ( (iun > 3) && (jun > 5) ) c35 = c30 + 5;
     if ( (iun > 3) && (jun > 6) ) c36 = c30 + 6;
     if ( (iun > 3) && (jun > 7) ) c37 = c30 + 7;
     if ( iun > 4 ) c40 = c30 + jun;
     if ( (iun > 4) && (jun > 1) ) c41 = c40 + 1;
     if ( (iun > 4) && (jun > 2) ) c42 = c40 + 2;
     if ( (iun > 4) && (jun > 3) ) c43 = c40 + 3;
     if ( (iun > 4) && (jun > 4) ) c44 = c40 + 4;
     if ( (iun > 4) && (jun > 5) ) c45 = c40 + 5;
     if ( (iun > 4) && (jun > 6) ) c46 = c40 + 6;
     if ( (iun > 4) && (jun > 7) ) c47 = c40 + 7;
     if ( iun > 5 ) c50 = c40 + jun;
     if ( (iun > 5) && (jun > 1) ) c51 = c50 + 1;
     if ( (iun > 5) && (jun > 2) ) c52 = c50 + 2;
     if ( (iun > 5) && (jun > 3) ) c53 = c50 + 3;
     if ( (iun > 5) && (jun > 4) ) c54 = c50 + 4;
     if ( (iun > 5) && (jun > 5) ) c55 = c50 + 5;
     if ( (iun > 5) && (jun > 6) ) c56 = c50 + 6;
     if ( (iun > 5) && (jun > 7) ) c57 = c50 + 7;
     if ( iun > 6 ) c60 = c50 + jun;
     if ( (iun > 6) && (jun > 1) ) c61 = c60 + 1;
     if ( (iun > 6) && (jun > 2) ) c62 = c60 + 2;
     if ( (iun > 6) && (jun > 3) ) c63 = c60 + 3;
     if ( (iun > 6) && (jun > 4) ) c64 = c60 + 4;
     if ( (iun > 6) && (jun > 5) ) c65 = c60 + 5;
     if ( (iun > 6) && (jun > 6) ) c66 = c60 + 6;
     if ( (iun > 6) && (jun > 7) ) c67 = c60 + 7;
     if ( iun > 7 ) c70 = c60 + jun;
     if ( (iun > 7) && (jun > 1) ) c71 = c70 + 1;
     if ( (iun > 7) && (jun > 2) ) c72 = c70 + 2;
     if ( (iun > 7) && (jun > 3) ) c73 = c70 + 3;
     if ( (iun > 7) && (jun > 4) ) c74 = c70 + 4;
     if ( (iun > 7) && (jun > 5) ) c75 = c70 + 5;
     if ( (iun > 7) && (jun > 6) ) c76 = c70 + 6;
     if ( (iun > 7) && (jun > 7) ) c77 = c70 + 7;

#if 0
#define COMPACT_GEMMNN_DEBUG
#endif

#ifdef COMPACT_GEMMNN_DEBUG
printf("iun=%d jun=%d loopi=%d loopj=%d\n",iun,jun,loopi,loopj);
printf("areg=%d breg=%d creg=%d mborder=%d nborder=%d\n",areg,breg,creg,mborder,nborder);
printf("a0:7=%d %d %d %d %d %d %d %d\n",a0,a1,a2,a3,a4,a5,a6,a7);
printf("b0:7=%d %d %d %d %d %d %d %d\n",b0,b1,b2,b3,b4,b5,b6,b7);
printf("c0,0:7=%d %d %d %d %d %d %d %d\n",c00,c01,c02,c03,c04,c05,c06,c07);
if (c10>0) printf("c1,0:7=%d %d %d %d %d %d %d %d\n",c10,c11,c12,c13,c14,c15,c16,c17);
if (c20>0) printf("c2,0:7=%d %d %d %d %d %d %d %d\n",c20,c21,c22,c23,c24,c25,c26,c27);
if (c30>0) printf("c3,0:7=%d %d %d %d %d %d %d %d\n",c30,c31,c32,c33,c34,c35,c36,c37);
if (c40>0) printf("c4,0:7=%d %d %d %d %d %d %d %d\n",c40,c41,c42,c43,c44,c45,c46,c47);
if (c50>0) printf("c5,0:7=%d %d %d %d %d %d %d %d\n",c50,c51,c52,c53,c54,c55,c56,c57);
if (c60>0) printf("c6,0:7=%d %d %d %d %d %d %d %d\n",c60,c61,c62,c63,c64,c65,c66,c67);
if (c70>0) printf("c7,0:7=%d %d %d %d %d %d %d %d\n",c70,c71,c72,c73,c74,c75,c76,c77);
#endif

     if ( loopj && (nloopcnt >=2) ) {
#ifdef COMPACT_GEMMNN_DEBUG
        printf("Setting up n-loop: loopj=%d nloopcnt=%d\n",loopj,nloopcnt);
#endif
        libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, nloopcnt );

        libxsmm_x86_instruction_register_jump_back_label( io_code, &l_loop_label_tracker );
     }

     for ( j = bn1; j <= bn2; j+=jun ) {
#ifdef COMPACT_GEMMNN_DEBUG
        printf("Doing j loop from %d to %d with blocksize %d\n",bn1,bn2,jun);
#endif
        if ( (  j  <= bn2 ) && ( jun >= 1 ) ) j0 = 1; else j0 = 0;
        if ( ( j+1 <= bn2 ) && ( jun >= 2 ) ) j1 = 1; else j1 = 0;
        if ( ( j+2 <= bn2 ) && ( jun >= 3 ) ) j2 = 1; else j2 = 0;
        if ( ( j+3 <= bn2 ) && ( jun >= 4 ) ) j3 = 1; else j3 = 0;
        if ( ( j+4 <= bn2 ) && ( jun >= 5 ) ) j4 = 1; else j4 = 0;
        if ( ( j+5 <= bn2 ) && ( jun >= 6 ) ) j5 = 1; else j5 = 0;
        if ( ( j+6 <= bn2 ) && ( jun >= 7 ) ) j6 = 1; else j6 = 0;
        if ( ( j+7 <= bn2 ) && ( jun >= 8 ) ) j7 = 1; else j7 = 0;
        if ( loopj && (j > bn1) && (j + jun -1 <= bn2) ) {
           /* Turn everything off, we're really supposed to be in a loop */
           j0=0; j1=0; j2=0; j3=0; j4=0; j5=0; j6=0; j7=0;
#ifdef COMPACT_GEMMNN_DEBUG
           printf("Emptying n-loop for j=%d\n",j);
#endif
        }
        if ( loopi && (mloopcnt >=2) && j0 ) {
#ifdef COMPACT_GEMMNN_DEBUG
           printf("Setting up m-loop: loopi=%d mloopcnt=%d\n",loopi,mloopcnt);
#endif
           libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RCX, mloopcnt );

           libxsmm_x86_instruction_register_jump_back_label( io_code, &l_loop_label_tracker );
           mloopadj = 1;
        }
        for ( i = am1; i <= am2; i+=iun ) {
           if ( (  i  <= am2 ) && ( iun >= 1 ) ) i0 = 1; else i0 = 0;
           if ( ( i+1 <= am2 ) && ( iun >= 2 ) ) i1 = 1; else i1 = 0;
           if ( ( i+2 <= am2 ) && ( iun >= 3 ) ) i2 = 1; else i2 = 0;
           if ( ( i+3 <= am2 ) && ( iun >= 4 ) ) i3 = 1; else i3 = 0;
           if ( ( i+4 <= am2 ) && ( iun >= 5 ) ) i4 = 1; else i4 = 0;
           if ( ( i+5 <= am2 ) && ( iun >= 6 ) ) i5 = 1; else i5 = 0;
           if ( ( i+6 <= am2 ) && ( iun >= 7 ) ) i6 = 1; else i6 = 0;
           if ( ( i+7 <= am2 ) && ( iun >= 8 ) ) i7 = 1; else i7 = 0;
#ifdef COMPACT_GEMMNN_DEBUG
           printf("Doing i loop from %d to %d with blocksize %d (%d,%d,%d,%d,%d,%d,%d,%d)\n",am1,am2,iun,i0,i1,i2,i3,i4,i5,i6,i7);
#endif
           if ( loopi && (i > am1) && (i + iun -1 <= am2) ) {
              /* Turn everything off, we're really supposed to be in a loop */
              i0=0; i1=0; i2=0; i3=0; i4=0; i5=0; i6=0; i7=0;
#ifdef COMPACT_GEMMNN_DEBUG
              printf("Emptying m-loop for i=%d j=%d i0=%d j0=%d\n",i,j,i0,j0);
#endif
           }
#ifdef COMPACT_GEMMNN_DEBUG
           if (i0 && j0 ) printf("Loading A into %d with tra=%d lda=%d i=%d j=%d numb=%d datasz=%d regset=%c areg=%d\n",a0,tra,lda,i,ak1,numb,datasz, regset,areg);
#endif
           if (i0 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i, ak1, a0, numb, datasz, regset, areg );
#ifdef COMPACT_GEMMNN_DEBUG
           if (i0 && j0 ) printf("Loaded A into %d with tra=%d lda=%d i=%d j=%d numb=%d datasz=%d regset=%c areg=%d\n",a0,tra,lda,i,ak1,numb,datasz, regset,areg);
#endif
           if (i0 && j0) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j, b0, numb, datasz, regset, breg );
           if (i0 && j0) compact_mult_two_nums_ ( io_code, a0, b0, c00, numb, regset );
           if (i1 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+1, ak1, a1, numb, datasz, regset, areg );
           if (i1 && j0) compact_mult_two_nums_ ( io_code, a1, b0, c10, numb, regset );
           if (i2 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+2, ak1, a2, numb, datasz, regset, areg );
           if (i2 && j0) compact_mult_two_nums_ ( io_code, a2, b0, c20, numb, regset );
           if (i3 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+3, ak1, a3, numb, datasz, regset, areg );
           if (i3 && j0) compact_mult_two_nums_ ( io_code, a3, b0, c30, numb, regset );
           if (i4 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+4, ak1, a4, numb, datasz, regset, areg );
           if (i4 && j0) compact_mult_two_nums_ ( io_code, a4, b0, c40, numb, regset );
           if (i5 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+5, ak1, a5, numb, datasz, regset, areg );
           if (i5 && j0) compact_mult_two_nums_ ( io_code, a5, b0, c50, numb, regset );
           if (i6 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+6, ak1, a6, numb, datasz, regset, areg );
           if (i6 && j0) compact_mult_two_nums_ ( io_code, a6, b0, c60, numb, regset );
           if (i7 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+7, ak1, a7, numb, datasz, regset, areg );
           if (i7 && j0) compact_mult_two_nums_ ( io_code, a7, b0, c70, numb, regset );
           if (i0 && j1) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+1, b1, numb, datasz, regset, breg );
           if (i0 && j1) compact_mult_two_nums_ ( io_code, a0, b1, c01, numb, regset );
           if (i0 && j2) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+2, b2, numb, datasz, regset, breg );
           if (i0 && j2) compact_mult_two_nums_ ( io_code, a0, b2, c02, numb, regset );
           if (i0 && j3) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+3, b3, numb, datasz, regset, breg );
           if (i0 && j3) compact_mult_two_nums_ ( io_code, a0, b3, c03, numb, regset );
           if (i0 && j4) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+4, b4, numb, datasz, regset, breg );
           if (i0 && j4) compact_mult_two_nums_ ( io_code, a0, b4, c04, numb, regset );
           if (i0 && j5) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+5, b5, numb, datasz, regset, breg );
           if (i0 && j5) compact_mult_two_nums_ ( io_code, a0, b5, c05, numb, regset );
           if (i0 && j6) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+6, b6, numb, datasz, regset, breg );
           if (i0 && j6) compact_mult_two_nums_ ( io_code, a0, b6, c06, numb, regset );
           if (i0 && j7) compact_load_matrix_gen_ ( io_code, trb, ldb, bk1, j+7, b7, numb, datasz, regset, breg );
           if (i0 && j7) compact_mult_two_nums_ ( io_code, a0, b7, c07, numb, regset );

           if (i1 && j1) compact_mult_two_nums_ ( io_code, a1, b1, c11, numb, regset );
           if (i1 && j2) compact_mult_two_nums_ ( io_code, a1, b2, c12, numb, regset );
           if (i1 && j3) compact_mult_two_nums_ ( io_code, a1, b3, c13, numb, regset );
           if (i1 && j4) compact_mult_two_nums_ ( io_code, a1, b4, c14, numb, regset );
           if (i1 && j5) compact_mult_two_nums_ ( io_code, a1, b5, c15, numb, regset );
           if (i1 && j6) compact_mult_two_nums_ ( io_code, a1, b6, c16, numb, regset );
           if (i1 && j7) compact_mult_two_nums_ ( io_code, a1, b7, c17, numb, regset );
           if (i2 && j1) compact_mult_two_nums_ ( io_code, a2, b1, c21, numb, regset );
           if (i2 && j2) compact_mult_two_nums_ ( io_code, a2, b2, c22, numb, regset );
           if (i2 && j3) compact_mult_two_nums_ ( io_code, a2, b3, c23, numb, regset );
           if (i2 && j4) compact_mult_two_nums_ ( io_code, a2, b4, c24, numb, regset );
           if (i2 && j5) compact_mult_two_nums_ ( io_code, a2, b5, c25, numb, regset );
           if (i2 && j6) compact_mult_two_nums_ ( io_code, a2, b6, c26, numb, regset );
           if (i2 && j7) compact_mult_two_nums_ ( io_code, a2, b7, c27, numb, regset );
           if (i3 && j1) compact_mult_two_nums_ ( io_code, a3, b1, c31, numb, regset );
           if (i3 && j2) compact_mult_two_nums_ ( io_code, a3, b2, c32, numb, regset );
           if (i3 && j3) compact_mult_two_nums_ ( io_code, a3, b3, c33, numb, regset );
           if (i3 && j4) compact_mult_two_nums_ ( io_code, a3, b4, c34, numb, regset );
           if (i3 && j5) compact_mult_two_nums_ ( io_code, a3, b5, c35, numb, regset );
           if (i3 && j6) compact_mult_two_nums_ ( io_code, a3, b6, c36, numb, regset );
           if (i3 && j7) compact_mult_two_nums_ ( io_code, a3, b7, c37, numb, regset );
           if (i4 && j1) compact_mult_two_nums_ ( io_code, a4, b1, c41, numb, regset );
           if (i4 && j2) compact_mult_two_nums_ ( io_code, a4, b2, c42, numb, regset );
           if (i4 && j3) compact_mult_two_nums_ ( io_code, a4, b3, c43, numb, regset );
           if (i4 && j4) compact_mult_two_nums_ ( io_code, a4, b4, c44, numb, regset );
           if (i4 && j5) compact_mult_two_nums_ ( io_code, a4, b5, c45, numb, regset );
           if (i4 && j6) compact_mult_two_nums_ ( io_code, a4, b6, c46, numb, regset );
           if (i4 && j7) compact_mult_two_nums_ ( io_code, a4, b7, c47, numb, regset );
           if (i5 && j1) compact_mult_two_nums_ ( io_code, a5, b1, c51, numb, regset );
           if (i5 && j2) compact_mult_two_nums_ ( io_code, a5, b2, c52, numb, regset );
           if (i5 && j3) compact_mult_two_nums_ ( io_code, a5, b3, c53, numb, regset );
           if (i5 && j4) compact_mult_two_nums_ ( io_code, a5, b4, c54, numb, regset );
           if (i5 && j5) compact_mult_two_nums_ ( io_code, a5, b5, c55, numb, regset );
           if (i5 && j6) compact_mult_two_nums_ ( io_code, a5, b6, c56, numb, regset );
           if (i5 && j7) compact_mult_two_nums_ ( io_code, a5, b7, c57, numb, regset );
           if (i6 && j1) compact_mult_two_nums_ ( io_code, a6, b1, c61, numb, regset );
           if (i6 && j2) compact_mult_two_nums_ ( io_code, a6, b2, c62, numb, regset );
           if (i6 && j3) compact_mult_two_nums_ ( io_code, a6, b3, c63, numb, regset );
           if (i6 && j4) compact_mult_two_nums_ ( io_code, a6, b4, c64, numb, regset );
           if (i6 && j5) compact_mult_two_nums_ ( io_code, a6, b5, c65, numb, regset );
           if (i6 && j6) compact_mult_two_nums_ ( io_code, a6, b6, c66, numb, regset );
           if (i6 && j7) compact_mult_two_nums_ ( io_code, a6, b7, c67, numb, regset );
           if (i7 && j1) compact_mult_two_nums_ ( io_code, a7, b1, c71, numb, regset );
           if (i7 && j2) compact_mult_two_nums_ ( io_code, a7, b2, c72, numb, regset );
           if (i7 && j3) compact_mult_two_nums_ ( io_code, a7, b3, c73, numb, regset );
           if (i7 && j4) compact_mult_two_nums_ ( io_code, a7, b4, c74, numb, regset );
           if (i7 && j5) compact_mult_two_nums_ ( io_code, a7, b5, c75, numb, regset );
           if (i7 && j6) compact_mult_two_nums_ ( io_code, a7, b6, c76, numb, regset );
           if (i7 && j7) compact_mult_two_nums_ ( io_code, a7, b7, c77, numb, regset );

           for ( l = ak1+1; l <= ak2; l++ ) {
#ifdef COMPACT_GEMMNN_DEBUG
              printf("Doing l loop from %d to %d\n",ak1+1,ak2);
#endif
              if (i0 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i, l, a0, numb, datasz, regset, areg );
              if (i0 && j0) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j, b0, numb, datasz, regset, breg);
              if (i0 && j0) compact_fma_cplusab_ ( io_code, c00, a0, b0, numb, regset );
              if (i1 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+1, l, a1, numb, datasz, regset, areg );
              if (i1 && j0) compact_fma_cplusab_ ( io_code, c10, a1, b0, numb, regset );
              if (i2 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+2, l, a2, numb, datasz, regset, areg );
              if (i2 && j0) compact_fma_cplusab_ ( io_code, c20, a2, b0, numb, regset );
              if (i3 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+3, l, a3, numb, datasz, regset, areg );
              if (i3 && j0) compact_fma_cplusab_ ( io_code, c30, a3, b0, numb, regset );
              if (i4 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+4, l, a4, numb, datasz, regset, areg );
              if (i4 && j0) compact_fma_cplusab_ ( io_code, c40, a4, b0, numb, regset );
              if (i5 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+5, l, a5, numb, datasz, regset, areg );
              if (i5 && j0) compact_fma_cplusab_ ( io_code, c50, a5, b0, numb, regset );
              if (i6 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+6, l, a6, numb, datasz, regset, areg );
              if (i6 && j0) compact_fma_cplusab_ ( io_code, c60, a6, b0, numb, regset );
              if (i7 && j0) compact_load_matrix_gen_ ( io_code, tra, lda, i+7, l, a7, numb, datasz, regset, areg );
              if (i7 && j0) compact_fma_cplusab_ ( io_code, c70, a7, b0, numb, regset );
              if (i0 && j1) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+1, b1, numb, datasz, regset, breg);
              if (i0 && j1) compact_fma_cplusab_ ( io_code, c01, a0, b1, numb, regset );
              if (i0 && j2) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+2, b2, numb, datasz, regset, breg);
              if (i0 && j2) compact_fma_cplusab_ ( io_code, c02, a0, b2, numb, regset );
              if (i0 && j3) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+3, b3, numb, datasz, regset, breg);
              if (i0 && j3) compact_fma_cplusab_ ( io_code, c03, a0, b3, numb, regset );
              if (i0 && j4) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+4, b4, numb, datasz, regset, breg);
              if (i0 && j4) compact_fma_cplusab_ ( io_code, c04, a0, b4, numb, regset );
              if (i0 && j5) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+5, b5, numb, datasz, regset, breg);
              if (i0 && j5) compact_fma_cplusab_ ( io_code, c05, a0, b5, numb, regset );
              if (i0 && j6) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+6, b6, numb, datasz, regset, breg);
              if (i0 && j6) compact_fma_cplusab_ ( io_code, c06, a0, b6, numb, regset );
              if (i0 && j7) compact_load_matrix_gen_ ( io_code, trb, ldb, l-ak1+bk1, j+7, b7, numb, datasz, regset, breg);
              if (i0 && j7) compact_fma_cplusab_ ( io_code, c07, a0, b7, numb, regset );

              if (i1 && j1) compact_fma_cplusab_ ( io_code, c11, a1, b1, numb, regset );
              if (i1 && j2) compact_fma_cplusab_ ( io_code, c12, a1, b2, numb, regset );
              if (i1 && j3) compact_fma_cplusab_ ( io_code, c13, a1, b3, numb, regset );
              if (i1 && j4) compact_fma_cplusab_ ( io_code, c14, a1, b4, numb, regset );
              if (i1 && j5) compact_fma_cplusab_ ( io_code, c15, a1, b5, numb, regset );
              if (i1 && j6) compact_fma_cplusab_ ( io_code, c16, a1, b6, numb, regset );
              if (i1 && j7) compact_fma_cplusab_ ( io_code, c17, a1, b7, numb, regset );
              if (i2 && j1) compact_fma_cplusab_ ( io_code, c21, a2, b1, numb, regset );
              if (i2 && j2) compact_fma_cplusab_ ( io_code, c22, a2, b2, numb, regset );
              if (i2 && j3) compact_fma_cplusab_ ( io_code, c23, a2, b3, numb, regset );
              if (i2 && j4) compact_fma_cplusab_ ( io_code, c24, a2, b4, numb, regset );
              if (i2 && j5) compact_fma_cplusab_ ( io_code, c25, a2, b5, numb, regset );
              if (i2 && j6) compact_fma_cplusab_ ( io_code, c26, a2, b6, numb, regset );
              if (i2 && j7) compact_fma_cplusab_ ( io_code, c27, a2, b7, numb, regset );
              if (i3 && j1) compact_fma_cplusab_ ( io_code, c31, a3, b1, numb, regset );
              if (i3 && j2) compact_fma_cplusab_ ( io_code, c32, a3, b2, numb, regset );
              if (i3 && j3) compact_fma_cplusab_ ( io_code, c33, a3, b3, numb, regset );
              if (i3 && j4) compact_fma_cplusab_ ( io_code, c34, a3, b4, numb, regset );
              if (i3 && j5) compact_fma_cplusab_ ( io_code, c35, a3, b5, numb, regset );
              if (i3 && j6) compact_fma_cplusab_ ( io_code, c36, a3, b6, numb, regset );
              if (i3 && j7) compact_fma_cplusab_ ( io_code, c37, a3, b7, numb, regset );
              if (i4 && j1) compact_fma_cplusab_ ( io_code, c41, a4, b1, numb, regset );
              if (i4 && j2) compact_fma_cplusab_ ( io_code, c42, a4, b2, numb, regset );
              if (i4 && j3) compact_fma_cplusab_ ( io_code, c43, a4, b3, numb, regset );
              if (i4 && j4) compact_fma_cplusab_ ( io_code, c44, a4, b4, numb, regset );
              if (i4 && j5) compact_fma_cplusab_ ( io_code, c45, a4, b5, numb, regset );
              if (i4 && j6) compact_fma_cplusab_ ( io_code, c46, a4, b6, numb, regset );
              if (i4 && j7) compact_fma_cplusab_ ( io_code, c47, a4, b7, numb, regset );
              if (i5 && j1) compact_fma_cplusab_ ( io_code, c51, a5, b1, numb, regset );
              if (i5 && j2) compact_fma_cplusab_ ( io_code, c52, a5, b2, numb, regset );
              if (i5 && j3) compact_fma_cplusab_ ( io_code, c53, a5, b3, numb, regset );
              if (i5 && j4) compact_fma_cplusab_ ( io_code, c54, a5, b4, numb, regset );
              if (i5 && j5) compact_fma_cplusab_ ( io_code, c55, a5, b5, numb, regset );
              if (i5 && j6) compact_fma_cplusab_ ( io_code, c56, a5, b6, numb, regset );
              if (i5 && j7) compact_fma_cplusab_ ( io_code, c57, a5, b7, numb, regset );
              if (i6 && j1) compact_fma_cplusab_ ( io_code, c61, a6, b1, numb, regset );
              if (i6 && j2) compact_fma_cplusab_ ( io_code, c62, a6, b2, numb, regset );
              if (i6 && j3) compact_fma_cplusab_ ( io_code, c63, a6, b3, numb, regset );
              if (i6 && j4) compact_fma_cplusab_ ( io_code, c64, a6, b4, numb, regset );
              if (i6 && j5) compact_fma_cplusab_ ( io_code, c65, a6, b5, numb, regset );
              if (i6 && j6) compact_fma_cplusab_ ( io_code, c66, a6, b6, numb, regset );
              if (i6 && j7) compact_fma_cplusab_ ( io_code, c67, a6, b7, numb, regset );
              if (i7 && j1) compact_fma_cplusab_ ( io_code, c71, a7, b1, numb, regset );
              if (i7 && j2) compact_fma_cplusab_ ( io_code, c72, a7, b2, numb, regset );
              if (i7 && j3) compact_fma_cplusab_ ( io_code, c73, a7, b3, numb, regset );
              if (i7 && j4) compact_fma_cplusab_ ( io_code, c74, a7, b4, numb, regset );
              if (i7 && j5) compact_fma_cplusab_ ( io_code, c75, a7, b5, numb, regset );
              if (i7 && j6) compact_fma_cplusab_ ( io_code, c76, a7, b6, numb, regset );
              if (i7 && j7) compact_fma_cplusab_ ( io_code, c77, a7, b7, numb, regset );

           } /* Inner loop */
           /* Storing into C, do it one column at a time and reuse some regs */
           for ( l = j; l <= LIBXSMM_MIN(j+jun-1,bn2); l++ ) {
#ifdef COMPACT_GEMMNN_DEBUG
              printf("Doing j wrap-up storage from %d to %d\n",j,LIBXSMM_MIN(j+jun-1,bn2));
#endif
              if (l== j ) { c0=c00; c1=c10; c2=c20; c3=c30; c4=c40; c5=c50; c6=c60; c7=c70; }
              if (l==j+1) { c0=c01; c1=c11; c2=c21; c3=c31; c4=c41; c5=c51; c6=c61; c7=c71; }
              if (l==j+2) { c0=c02; c1=c12; c2=c22; c3=c32; c4=c42; c5=c52; c6=c62; c7=c72; }
              if (l==j+3) { c0=c03; c1=c13; c2=c23; c3=c33; c4=c43; c5=c53; c6=c63; c7=c73; }
              if (l==j+4) { c0=c04; c1=c14; c2=c24; c3=c34; c4=c44; c5=c54; c6=c64; c7=c74; }
              if (l==j+5) { c0=c05; c1=c15; c2=c25; c3=c35; c4=c45; c5=c55; c6=c65; c7=c75; }
              if (l==j+6) { c0=c06; c1=c16; c2=c26; c3=c36; c4=c46; c5=c56; c6=c66; c7=c76; }
              if (l==j+7) { c0=c07; c1=c17; c2=c27; c3=c37; c4=c47; c5=c57; c6=c67; c7=c77; }
              if ( beta == 1.0 ) {
                 if (i0 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1, l-bn1+cn1, a0, numb, datasz, regset, creg );
                 if (i1 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+1, l-bn1+cn1, a1, numb, datasz, regset, creg );
                 if (i2 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+2, l-bn1+cn1, a2, numb, datasz, regset, creg );
                 if (i3 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+3, l-bn1+cn1, a3, numb, datasz, regset, creg );
                 if (i4 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+4, l-bn1+cn1, a4, numb, datasz, regset, creg );
                 if (i5 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+5, l-bn1+cn1, a5, numb, datasz, regset, creg );
                 if (i6 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+6, l-bn1+cn1, a6, numb, datasz, regset, creg );
                 if (i7 && j0) compact_load_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+7, l-bn1+cn1, a7, numb, datasz, regset, creg );
              } else if ( (beta == 0.0) && (alpha != 1.0) ) {
                 if (i0 && j0) compact_set_zero_( io_code, a0, numb, datasz, regset );
                 if (i1 && j0) compact_set_zero_( io_code, a1, numb, datasz, regset );
                 if (i2 && j0) compact_set_zero_( io_code, a2, numb, datasz, regset );
                 if (i3 && j0) compact_set_zero_( io_code, a3, numb, datasz, regset );
                 if (i4 && j0) compact_set_zero_( io_code, a4, numb, datasz, regset );
                 if (i5 && j0) compact_set_zero_( io_code, a5, numb, datasz, regset );
                 if (i6 && j0) compact_set_zero_( io_code, a6, numb, datasz, regset );
                 if (i7 && j0) compact_set_zero_( io_code, a7, numb, datasz, regset );
              }
              if ( alpha == -1.0 ) {
                 if (i0 && j0) compact_sub_two_nums_ ( io_code, a0, c0, c0, numb, regset );
                 if (i1 && j0) compact_sub_two_nums_ ( io_code, a1, c1, c1, numb, regset );
                 if (i2 && j0) compact_sub_two_nums_ ( io_code, a2, c2, c2, numb, regset );
                 if (i3 && j0) compact_sub_two_nums_ ( io_code, a3, c3, c3, numb, regset );
                 if (i4 && j0) compact_sub_two_nums_ ( io_code, a4, c4, c4, numb, regset );
                 if (i5 && j0) compact_sub_two_nums_ ( io_code, a5, c5, c5, numb, regset );
                 if (i6 && j0) compact_sub_two_nums_ ( io_code, a6, c6, c6, numb, regset );
                 if (i7 && j0) compact_sub_two_nums_ ( io_code, a7, c7, c7, numb, regset );
              } else if ( (beta != 0.0) && (alpha==1.0) ) {
                 if (i0 && j0) compact_add_two_nums_ ( io_code, a0, c0, c0, numb, regset );
                 if (i1 && j0) compact_add_two_nums_ ( io_code, a1, c1, c1, numb, regset );
                 if (i2 && j0) compact_add_two_nums_ ( io_code, a2, c2, c2, numb, regset );
                 if (i3 && j0) compact_add_two_nums_ ( io_code, a3, c3, c3, numb, regset );
                 if (i4 && j0) compact_add_two_nums_ ( io_code, a4, c4, c4, numb, regset );
                 if (i5 && j0) compact_add_two_nums_ ( io_code, a5, c5, c5, numb, regset );
                 if (i6 && j0) compact_add_two_nums_ ( io_code, a6, c6, c6, numb, regset );
                 if (i7 && j0) compact_add_two_nums_ ( io_code, a7, c7, c7, numb, regset );
              }
              if (i0 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1, l-bn1+cn1, c0, numb, datasz, regset, creg );
              if (i1 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+1, l-bn1+cn1, c1, numb, datasz, regset, creg );
              if (i2 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+2, l-bn1+cn1, c2, numb, datasz, regset, creg );
              if (i3 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+3, l-bn1+cn1, c3, numb, datasz, regset, creg );
              if (i4 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+4, l-bn1+cn1, c4, numb, datasz, regset, creg );
              if (i5 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+5, l-bn1+cn1, c5, numb, datasz, regset, creg );
              if (i6 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+6, l-bn1+cn1, c6, numb, datasz, regset, creg );
              if (i7 && j0) compact_store_matrix_gen_ ( io_code, trc, ldc, i-am1+cm1+7, l-bn1+cn1, c7, numb, datasz, regset, creg );
           } /* Store the results */
           if ( loopi && j0 ) {
              aoffset = datasz*iun*numb;
              coffset = datasz*iun*numb;
              if ( i == am1 ) {
#ifdef COMPACT_GEMMNN_DEBUG
                 printf("Should be putting in a m-jump soon: i=%d j=%d i0=%d j0=%d am1=%d am2=%d\n",i,j,i0,j0,am1,am2);
#endif
                 libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_ADDQ, areg, aoffset );
                 libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_ADDQ, creg, coffset );
                 libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RCX, 1 );
                 libxsmm_x86_instruction_jump_back_to_label( io_code, LIBXSMM_X86_INSTR_JG, &l_loop_label_tracker );
              }
              if ( (am2-i+1 < 2*iun) && ((mborder > 0) || (j + jun - 1 < bn2)) && (mloopadj==1) ) {
#ifdef COMPACT_GEMMNN_DEBUG
                 printf("Finished with m-loop, doing clean-up: i=%d i0=%d j0=%d mborder=%d j=%d jun=%d bn2=%d\n",i,i0,j0,mborder,j,jun,bn2);
#endif
                 aoffset = datasz*iun*numb*mloopcnt;
                 coffset = datasz*iun*numb*mloopcnt;
                 libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_SUBQ, areg, aoffset );
                 libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_SUBQ, creg, coffset );
                 mloopadj = 0;
              }
              i0 = 1; /* Turn everything back on again */
           }
        } /* M-loop */
        if ( loopj ) {
           coffset = ldc*datasz*jun*numb;
           boffset = ldb*datasz*jun*numb;
           if ( j == bn1 ) {
#ifdef COMPACT_GEMMNN_DEBUG
              printf("Should be putting in a n-jump soon: j=%d bn1=%d bn2=%d\n",j,bn1,bn2);
#endif
              libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_ADDQ, creg, coffset );
              libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_ADDQ, breg, boffset );
              libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RAX, 1 );
              libxsmm_x86_instruction_jump_back_to_label( io_code, LIBXSMM_X86_INSTR_JG, &l_loop_label_tracker );
           }
           if ( (bn2-j+1 < 2*jun) && (nborder > 0) ) {
#ifdef COMPACT_GEMMNN_DEBUG
              printf("Finished with n-loop, doing clean-up, j=%d\n",j);
#endif
              coffset = ldc*datasz*jun*numb*nloopcnt;
              boffset = ldb*datasz*jun*numb*nloopcnt;
              libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_SUBQ, creg, coffset );
              libxsmm_x86_instruction_alu_imm( io_code, LIBXSMM_X86_INSTR_SUBQ, breg, boffset );
           }
           j0 = 1; /* Turn everything back on again */
        }
     } /* N-loop */
#ifdef COMPACT_GEMMNN_DEBUG
     printf("Inlined Compact GEMM code pointer ends at: %u\n",io_code->code_size);
#endif
}

#endif /*GENERATOR_PACKED_GEMMNN_H*/

