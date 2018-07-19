/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/
#include "generator_packed_trsm_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INLINE void compact_load_parameter_ (
     libxsmm_generated_code* io_code,
     double alpha,
     unsigned int reg,
     unsigned int number,
     char regset )
{
     double vector[8];
     int i;

     if ( number > 8 )
     {
        fprintf(stderr,"loading too large a parameter for compact_load_parameter\n");
        exit(-1);
     }
     for ( i = 0 ; i < (int)number ; i++ ) vector[i]=alpha;

     libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) vector, "loadconst", regset, reg );
}

LIBXSMM_API_INLINE void compact_set_zero_ (
     libxsmm_generated_code* io_code,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     LIBXSMM_UNUSED(number);

     if ( (datasize == 8) && (regset=='z') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX512, LIBXSMM_X86_INSTR_VXORPD, regset, reg, reg, reg );
     } else if ( (datasize == 4) && (regset=='z') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX512, LIBXSMM_X86_INSTR_VXORPS, regset, reg, reg, reg );
     } else if ( (datasize == 8) && (regset=='y') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VXORPD, regset, reg, reg, reg );
     } else if ( (datasize == 4) && (regset=='y') )
     {
        libxsmm_x86_instruction_vec_compute_reg ( io_code, LIBXSMM_X86_AVX2, LIBXSMM_X86_INSTR_VXORPS, regset, reg, reg, reg );
     }
}

LIBXSMM_API_INLINE void compact_set_one_ (
     libxsmm_generated_code* io_code,
     unsigned int reg,
     unsigned int number,
     unsigned int datasize,
     char regset )
{
     double vector[8];
     int i;

     LIBXSMM_UNUSED(datasize);

     if ( number > 8 )
     {
        fprintf(stderr,"loading too large a parameter for compact_set_one_\n");
        exit(-1);
     }
     for ( i = 0 ; i < (int)number ; i++ ) vector[i]=1.0;

     libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) vector, "loadone", regset, reg );
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
     int datasize = 0;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     LIBXSMM_UNUSED(datasize);

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
        datasize = 8;
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPD;
     } else if ( (number==8) && (regset=='z') )
     {
        datasize = 8;
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPD;
     } else if ( (number==8) && (regset=='y') )
     {
        datasize = 4;
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPS;
     } else if ( (number==16) && (regset=='z') )
     {
        datasize = 4;
        i_vmove_instr = LIBXSMM_X86_INSTR_VMULPS;
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
     int datasize = 0;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     LIBXSMM_UNUSED(datasize);

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
        datasize = 8;
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PD;
     } else if ( (number==8) && (regset=='z') )
     {
        datasize = 8;
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PD;
     } else if ( (number==8) && (regset=='y') )
     {
        datasize = 4;
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PS;
     } else if ( (number==16) && (regset=='z') )
     {
        datasize = 4;
        i_vmove_instr = LIBXSMM_X86_INSTR_VFNMADD231PS;
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
     int datasize = 0;
     unsigned int i_vmove_instr;
     int i_instruction_set;

     LIBXSMM_UNUSED(datasize);

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
        datasize = 8;
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPD;
     } else if ( (number==8) && (regset=='z') )
     {
        datasize = 8;
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPD;
     } else if ( (number==8) && (regset=='y') )
     {
        datasize = 4;
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPS;
     } else if ( (number==16) && (regset=='z') )
     {
        datasize = 4;
        i_vmove_instr = LIBXSMM_X86_INSTR_VDIVPS;
     } else {
        fprintf(stderr,"Unsupported combo of number and regset in compact_divide_two_nums\n");
        exit(-1);
     }

     libxsmm_x86_instruction_vec_compute_reg ( io_code, i_instruction_set, i_vmove_instr, regset, reg1, reg0, reg2 );
}

