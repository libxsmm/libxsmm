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
#ifndef GENERATOR_PACKED_AUX_H
#define GENERATOR_PACKED_AUX_H

#include "generator_x86_instructions.h"
#include "generator_common.h"
#include <libxsmm_intrinsics_x86.h>


LIBXSMM_API_INLINE void compact_load_parameter_ (
     libxsmm_generated_code* io_code,
     double alpha,
     unsigned int reg,
     unsigned int number,
     char regset )
{
     int datasize;
     int i;

     if ( (number == 2) && (regset=='x') ) {
        datasize = 8;
     } else if ( (number == 4) && (regset=='x') ) {
        datasize = 4;
     } else if ( (number == 4) && (regset=='y') ) {
        datasize = 8;
     } else if ( (number == 8) && (regset=='y') ) {
        datasize = 4;
     } else if ( (number == 8) && (regset=='z') ) {
        datasize = 8;
     } else if ( (number == 16) && (regset=='z') ) {
        datasize = 4;
     } else {
        fprintf(stderr,"Unknown number=%u regset=%c combo for compact_load_parameter\n",number,regset);
        exit(-1);
     }

     if ( datasize == 8 ) {
        double vector[16];
        for ( i = 0; i < (int)number; i++ ) vector[i]=alpha;

        libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) vector, "loadconst", regset, reg );
     } else {
        float vector[16];
        for ( i = 0; i < (int)number; i++ ) vector[i]=(float)alpha;

        libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) vector, "loadconst", regset, reg );
     }
}

LIBXSMM_API_INLINE void compact_set_zero_ (
     libxsmm_generated_code* io_code,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{

     LIBXSMM_UNUSED(datasize);

     if ( (number == 8) && (regset=='z') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX512, LIBXSMM_X86_INSTR_VXORPD, regset, reg, reg, reg );
     } else if ( (number == 16) && (regset=='z') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX512, LIBXSMM_X86_INSTR_VXORPS, regset, reg, reg, reg );
     } else if ( (number == 8) && (regset=='y') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VXORPS, regset, reg, reg, reg );
     } else if ( (number == 4) && (regset=='y') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VXORPD, regset, reg, reg, reg );
     }
}

LIBXSMM_API_INLINE void compact_set_one_ (
     libxsmm_generated_code* io_code,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     double dvector[16];
     float  svector[16];
     int i;

     if ( number > 16 )
     {
        fprintf(stderr,"loading too large a parameter for compact_set_one_\n");
        exit(-1);
     }
     for ( i = 0; i < (int)number; i++ ) { dvector[i]=1.0; svector[i]=1.0; }

     if ( datasize == 4 )
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) svector, "loadone", regset, reg );
     else if ( datasize == 8 )
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) dvector, "loadone", regset, reg );
     else
        printf("Unknown datasize in compact_set_one_ error\n");
}

LIBXSMM_API_INLINE void compact_store_matrix_gen_ (
     libxsmm_generated_code* io_code,
     unsigned int trans,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset,
     unsigned int matrix_gpreg )
{
     int element;
     int offset;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( !trans ) element = number*(j-1)*lda + number*(i-1);
     else          element = number*(i-1)*lda + number*(j-1);
     offset = element * datasize;
     if ( /*(reg < 0) ||*/ (reg >=32) ) {
        printf("compact_store_matrix_gen trying to store from an invalid register: %u\n",reg);
        exit(-1);
     }
     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_store_matrix_gen has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_store_matrix1\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, matrix_gpreg, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 1 );
}

LIBXSMM_API_INLINE void compact_store_matrix1_ (
     libxsmm_generated_code* io_code,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     int element = number*(j-1)*lda + number*(i-1);
     int offset = element * datasize;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_store_matrix1 has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_store_matrix1\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, LIBXSMM_X86_GP_REG_RDI, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 1 );
}

LIBXSMM_API_INLINE void compact_store_matrix2_ (
     libxsmm_generated_code* io_code,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     int element = number*(j-1)*lda + number*(i-1);
     int offset = element * datasize;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_store_matrix2 has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_store_matrix2\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 1 );
}

LIBXSMM_API_INLINE void compact_store_matrix3_ (
     libxsmm_generated_code* io_code,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     int element = number*(j-1)*lda + number*(i-1);
     int offset = element * datasize;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_store_matrix3 has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_store_matrix3\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, LIBXSMM_X86_GP_REG_RDX, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 1 );
}

LIBXSMM_API_INLINE void compact_load_matrix1_ (
     libxsmm_generated_code* io_code,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     int element = number*(j-1)*lda + number*(i-1);
     int offset = element * datasize;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_load_matrix1 has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_load_matrix1\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, LIBXSMM_X86_GP_REG_RDI, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 0 );
}

LIBXSMM_API_INLINE void compact_load_matrix_gen_ (
     libxsmm_generated_code* io_code,
     unsigned int trans,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset,
     unsigned int matrix_gpreg )
{
     int element;
     int offset;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( /*(reg < 0) ||*/ (reg >=32) ) {
        printf("compact_load_matrix_gen trying to load to an invalid register: %u\n",reg);
        printf("lda=%u i=%u j=%u reg=%u number=%u datasize=%u regset=%c matrix_gpreg=%u\n",lda,i,j,reg,number,datasize,regset,matrix_gpreg);
        exit(-1);
     }
     if ( !trans ) element = number*(j-1)*lda + number*(i-1);
     else          element = number*(i-1)*lda + number*(j-1);
     offset = element * datasize;
     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_load_matrix_gen has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_load_matrix_gen\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, matrix_gpreg, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 0 );
}

LIBXSMM_API_INLINE void compact_load_matrix2_ (
     libxsmm_generated_code* io_code,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     int element = number*(j-1)*lda + number*(i-1);
     int offset = element * datasize;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_load_matrix2 has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_load_matrix2\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 0 );
}

LIBXSMM_API_INLINE void compact_load_matrix3_ (
     libxsmm_generated_code* io_code,
     unsigned int lda,
     unsigned int i,
     unsigned int j,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     int element = number*(j-1)*lda + number*(i-1);
     int offset = element * datasize;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     if ( datasize == 8 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPD;
     } else if ( datasize == 4 )
     {
        i_vmove_instr = LIBXSMM_X86_INSTR_VMOVUPS;
     } else {
        fprintf(stderr,"compact_load_matrix3 has strange datasize=%u\n",datasize);
        exit(-1);
     }
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_load_matrix3\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_move ( io_code, i_instruction_set, i_vmove_instr, LIBXSMM_X86_GP_REG_RDX, LIBXSMM_X86_GP_REG_UNDEF, 1, offset, regset, reg, 0, 0, 0 );
}

LIBXSMM_API_INLINE void compact_mult_two_nums_ (
     libxsmm_generated_code* io_code,
     unsigned int reg0,
     unsigned int reg1,
     unsigned int reg2,
     unsigned int number,
     char regset )
{
     unsigned int i_vmove_instr;
     int i_instruction_set;
#if 0
     int datasize = 0;
     LIBXSMM_UNUSED(datasize);
#endif
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_mult_two_nums\n");
        exit(-1);
     }

     if ( (number==4) && (regset=='y') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPD;
     } else if ( (number==8) && (regset=='z') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPD;
     } else if ( (number==8) && (regset=='y') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPS;
     } else if ( (number==16) && (regset=='z') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_mult_two_nums\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg0, reg2 );
}

LIBXSMM_API_INLINE void compact_add_two_nums_ (
     libxsmm_generated_code* io_code,
     unsigned int reg0,
     unsigned int reg1,
     unsigned int reg2,
     unsigned int number,
     char regset )
{
     unsigned int i_vmove_instr;
     int i_instruction_set;
#if 0
     int datasize = 0;
     LIBXSMM_UNUSED(datasize);
#endif
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_mult_two_nums\n");
        exit(-1);
     }

     if ( (number==4) && (regset=='y') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VADDPD;
     } else if ( (number==8) && (regset=='z') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VADDPD;
     } else if ( (number==8) && (regset=='y') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VADDPS;
     } else if ( (number==16) && (regset=='z') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VADDPS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_mult_two_nums\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg0, reg2 );
}

LIBXSMM_API_INLINE void compact_sub_two_nums_ (
     libxsmm_generated_code* io_code,
     unsigned int reg0,
     unsigned int reg1,
     unsigned int reg2,
     unsigned int number,
     char regset )
{
     unsigned int i_vmove_instr;
     int i_instruction_set;
#if 0
     int datasize = 0;
     LIBXSMM_UNUSED(datasize);
#endif
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_mult_two_nums\n");
        exit(-1);
     }

     if ( (number==4) && (regset=='y') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VSUBPD;
     } else if ( (number==8) && (regset=='z') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VSUBPD;
     } else if ( (number==8) && (regset=='y') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VSUBPS;
     } else if ( (number==16) && (regset=='z') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VSUBPS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_mult_two_nums\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg0, reg2 );
}

LIBXSMM_API_INLINE void compact_fms_cminusab_(
     libxsmm_generated_code* io_code,
     unsigned int reg0,
     unsigned int reg1,
     unsigned int reg2,
     unsigned int number,
     char regset )
{
     unsigned int i_vmove_instr;
     int i_instruction_set;
#if 0
     int datasize = 0;
     LIBXSMM_UNUSED(datasize);
#endif
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_fms_cminusab\n");
        exit(-1);
     }

     if ( (number==4) && (regset=='y') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PD;
     } else if ( (number==8) && (regset=='z') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PD;
     } else if ( (number==8) && (regset=='y') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PS;
     } else if ( (number==16) && (regset=='z') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_fms_cminusab\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg2, reg0 );
}

LIBXSMM_API_INLINE void compact_fma_cplusab_(
     libxsmm_generated_code* io_code,
     unsigned int reg0,
     unsigned int reg1,
     unsigned int reg2,
     unsigned int number,
     char regset )
{
     unsigned int i_vmove_instr;
     int i_instruction_set;
#if 0
     int datasize = 0;
     LIBXSMM_UNUSED(datasize);
#endif
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_fms_cminusab\n");
        exit(-1);
     }

     if ( (number==4) && (regset=='y') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFMADD231PD;
     } else if ( (number==8) && (regset=='z') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFMADD231PD;
     } else if ( (number==8) && (regset=='y') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFMADD231PS;
     } else if ( (number==16) && (regset=='z') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VFMADD231PS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_fms_cminusab\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg2, reg0 );
}

LIBXSMM_API_INLINE void compact_divide_two_nums_ (
     libxsmm_generated_code* io_code,
     unsigned int reg0,
     unsigned int reg1,
     unsigned int reg2,
     unsigned int number,
     char regset )
{
     unsigned int i_vmove_instr;
     int i_instruction_set;
#if 0
     int datasize = 0;
     LIBXSMM_UNUSED(datasize);
#endif
     if ( regset == 'z' )
     {
        i_instruction_set = LIBXSMM_X86_AVX512;
     } else if ( regset == 'y' ) {
        i_instruction_set = LIBXSMM_X86_AVX2;
     } else {
        fprintf(stderr,"Unsupported instruction set in compact_divide_two_nums\n");
        exit(-1);
     }

     if ( (number==4) && (regset=='y') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPD;
     } else if ( (number==8) && (regset=='z') )
     {
#if 0
        datasize = 8;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPD;
     } else if ( (number==8) && (regset=='y') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPS;
     } else if ( (number==16) && (regset=='z') )
     {
#if 0
        datasize = 4;
#endif
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_divide_two_nums\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg0, reg2 );
}

#endif /*GENERATOR_PACKED_AUX_H*/

