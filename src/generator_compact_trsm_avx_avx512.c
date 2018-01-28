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

#include "generator_compact_trsm_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_intrinsics_x86.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

/* #define GENERATOR_COMPACT_TRSM_DEBUG */

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_compact_trsm_avx_avx512_kernel(
                libxsmm_generated_code*                 io_generated_code,
                const libxsmm_compact_trsm_descriptor2* i_compact_trsm_desc,
                const char*                             i_arch )
{
  /* Just reuse transpose gp mapping */
  libxsmm_transpose_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
  libxsmm_loop_label_tracker l_loop_label_tracker /*= { 0 }*/;

  /* avx512 just represents whether we want to use zmm registers or not     *
   *      A value of 0 says not, a value of 1 targets AVX512_CORE, a value  *
   *      of 2 targets AVX512_MIC                                           */
  int avx512;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R9;
#endif
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  /* Actually, the logic is this: we need a, lda, and b. We don't need ldb  *
   * If n>=6, we need rbx                                                   *
   * If n>=8, we need rbp                                                   *
   * If LIBXSMM_MIN(n,REGSIZE)>=5 and m%REGSIZE==1, we need r12             *
   * If LIBXSMM_MIN(n,REGSIZE)>=6 and m%REGSIZE==1, we need r13             *
   * If LIBXSMM_MIN(n,REGSIZE)>=7 and m%REGSIZE==1, we need r14             *
   * If LIBXSMM_MIN(n,REGSIZE)>=8 and m%REGSIZE==1, we need r15             *
   * Otherwise, we get by with registers that don't require pushing/popping */

  /* define transposition kernel config */
  if (strcmp(i_arch, "skx") == 0) {
    avx512 = 1;
  } else if (strcmp(i_arch, "knl") == 0 || strcmp(i_arch, "knm") == 0) {
    avx512 = 2;
  } else if (strcmp(i_arch, "snb") == 0 || strcmp(i_arch, "hsw") == 0) {
    avx512 = 0;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  /* @Greg add more fields here */

  /* @Greg add generator code here, please use functions defined in generator_x86_instructions.h */
  /* Todo-> I first want this code to work, and verify it works, then I can
   *        convert one instruction at a time to those in
   *        generator_x86_instructions.h. Or add to the existing instructions */

  if ( io_generated_code->code_type > 1 )
  {
     unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
     int i = io_generated_code->code_size;
     unsigned int m = i_compact_trsm_desc->gemm->m;
     unsigned int n = i_compact_trsm_desc->gemm->n;
     unsigned int lda = i_compact_trsm_desc->gemm->lda;
     unsigned int ldb = i_compact_trsm_desc->gemm->ldb;
     const unsigned int *datasize_ptr = i_compact_trsm_desc->typesize;
     const unsigned int *layout = i_compact_trsm_desc->layout;
     unsigned int datasize = *datasize_ptr;
     const char *side = i_compact_trsm_desc->side;
     const char *diag = i_compact_trsm_desc->diag;
     const char *transa = i_compact_trsm_desc->transa;
     const char *uplo = i_compact_trsm_desc->uplo;
#if 0
     unsigned int lda = i_compact_trsm_desc->lda;
     unsigned int ldb = i_compact_trsm_desc->ldb;
#endif
     int imask = 0;
     int offsetA, offsetB, oldB;
     int j, k, m0, n0, shiftvalue, shiftmult;
     int REGSIZE;
     int maskvar = 0;

#if 0
     if ( *layout != 102 )
     {
        printf("layout problem\n");
        exit(-1);
     }
#endif
     if ( ( datasize !=4 ) && (datasize != 8) )
     {
        fprintf(stderr,"Expecting a datasize of 4 or 8, but got %d\n",datasize);
        exit(-1);
     }

    /* Early exit for now on misc cases */
    if ( (*side!='L') || (*uplo!='L') || (*transa!='N') || (*diag!='N') || (*layout!=102) || (avx512!=0) )
    {
       fprintf(stderr,"This case not yet ported in the prototype. Once the prototype is solidified, we can add it.\n");
       exit(-1);
    }

#include "compact_trsm_dmacros.h"

    auto int a_ptr = LIBXSMM_X86_GP_REG_RDI;
    auto int b_ptr = LIBXSMM_X86_GP_REG_RSI;
    char i_vector_name = 'y';
    int nonunit;

    if ( (*diag == 'N') || (*diag == 'n') ) nonunit = 1; else nonunit = 0; 
printf("Inside %c%c%c%c trsm generator\n",*side,*uplo,*transa,*diag);
    int n_in, m_in;
    int _is_row_;
    int ii;
    if (*layout == 101) {
        n_in = m;
        m_in = n;
        _is_row_ = 1;
    }
    else {
        m_in = m;
        n_in = n;
        _is_row_ = 0;
    }
    if ( nonunit && (datasize==8) )
    {
        double one_vector[8] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
        i = io_generated_code->code_size;

#ifdef DEBUG_COMPACT_TRSM
        printf("calling full_vec_load_of_constants: i=%d\n",i);
#endif

        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (unsigned char*) one_vector, "one_vec", i_vector_name, 15 );
        i = io_generated_code->code_size;

#ifdef DEBUG_COMPACT_TRSM
        printf("done calling full_vec_load_of_constants: i=%d\n",i);
        for ( i = 0 ; i < 44 ; i+= 4 )
        {
            printf("#bytes %d-%d:\n",i,i+3);
            printf(".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",buf[i],buf[i+1],buf[i+2],buf[i+3]);
        }
#endif
    }

    if ( nonunit && (datasize==4) )
    {
        float one_vector[16] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
        i = io_generated_code->code_size;

        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (unsigned char*) one_vector, "one_vec", i_vector_name, 15 );
        i = io_generated_code->code_size;
    }    


    int ymm0  = 0;
    int ymm1  = 1;
    int ymm2  = 2;
    int ymm3  = 3;
    int ymm4  = 4;
    int ymm5  = 5;
    int ymm6  = 6;
    int ymm7  = 7;
    int ymm8  = 8;
    int ymm9  = 9;
    int ymm10 = 10;
    int ymm11 = 11;
    int ymm12 = 12;
    int ymm13 = 13;
    int ymm14 = 14;
    int ymm15 = 15;
    int P_UNROLL_AVX2;

    if ( datasize == 8 ) P_UNROLL_AVX2 = P_UNROLL_AVX2_F64;
    else                 P_UNROLL_AVX2 = P_UNROLL_AVX2_F32;

    /* zero accumulation registers */
    i = io_generated_code->code_size;
#ifdef DEBUG_COMPACT_TRSM
    printf("Entering libxsmm_generator_transpose_avx_avx512_kernel with i loc=%d m=%d n=%d datasize=%d\n",i,m,n,datasize);
#endif
    SET_ZERO_PACKED(ymm0, ymm0, ymm0);                          /* T11*/
    i = io_generated_code->code_size;
#ifdef DEBUG_COMPACT_TRSM
    printf("After zero packed: i loc=%d m=%d n=%d datasize=%d\n",i,m,n,datasize);
#endif

    if (m_in > 1) SET_ZERO_PACKED(ymm1, ymm1, ymm1);            /* T21 */
    if (n_in > 1) {
        SET_ZERO_PACKED(ymm2, ymm2, ymm2);                      /* T12 */
        if (m_in > 1) SET_ZERO_PACKED(ymm3, ymm3, ymm3);        /* T22 */
    }

    for (j=0; j<(n_in/N_UNROLL_AVX2)*N_UNROLL_AVX2; j+=N_UNROLL_AVX2) {
        for (i=0; i<(m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+=M_UNROLL_AVX2) {
            /* gemm update */
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    /* A1 */
                VMOVU_PACKED(ymm6, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    /* A2 */
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);
                VFMADD231_PACKED(ymm1, ymm4, ymm6);

                VMOVU_PACKED(ymm4, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm2, ymm4, ymm5);
                VFMADD231_PACKED(ymm3, ymm4, ymm6);
            }
            /* update the 4x4 B matrix */
            /* 1st */
            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        /* B1 */

#ifdef DEBUG_COMPACT_TRSM
            printf("VMOVU_PACKED bug offset=%ld P_UNROLL_AVX2=%ld\n",sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))),P_UNROLL_AVX2);  
#endif

            VMOVU_PACKED(ymm9, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);        /* B2 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

            VSUB_PACKED(ymm2, ymm9, ymm2);                                                                                  /* T12 = B2-T12 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm2, ymm2, ymm4);                                                                                  /* T12 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm2, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);        /* Store T12 -> B2 */

            /* 2nd */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);        /* A2 */

            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);        /* B1 */
            VMOVU_PACKED(ymm9, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);        /* B2 */

            VFMADD231_PACKED(ymm1, ymm5, ymm0);                                                                             /* T21 += A2*T11 */
            VSUB_PACKED(ymm1, ymm8, ymm1);                                                                                  /* T21 = B1 - T21 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm1, ymm1, ymm4);                                                                                  /* T21 *= ONE/A1 */
         }
#endif
            VMOVU_PACKED(ymm1, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);        /* Store T21 -> B1 */

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              /* ZERO T11 */
            SET_ZERO_PACKED(ymm1, ymm1, ymm1);                                                                              /* ZERO T21 */

            VFMADD231_PACKED(ymm3, ymm5, ymm2);                                                                             /* T22 += A2*T12 */
            VSUB_PACKED(ymm3, ymm9, ymm3);                                                                                  /* T22 = B2 - T22 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm3, ymm3, ymm4);                                                                                  /* T22 *= ONE/A1 */
         }
#endif
            VMOVU_PACKED(ymm3, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 1);        /* Store T22 -> B2 */

            SET_ZERO_PACKED(ymm2, ymm2, ymm2);                                                                              /* ZERO T12 */
            SET_ZERO_PACKED(ymm3, ymm3, ymm3);                                                                              /* ZERO T22 */
        }
        if (m_in & 1) {
           /* gemm update */
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);   /* A1 */
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);

                VMOVU_PACKED(ymm4, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,1+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm2, ymm4, ymm5);
            }
            /* update the 4x4 B matrix */
            /* 1st */
            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        /* B1 */
            VMOVU_PACKED(ymm9, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);        /* B2 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              /* ZERO T11 */

            VSUB_PACKED(ymm2, ymm9, ymm2);                                                                                  /* T12 = B2-T12 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm2, ymm2, ymm4);                                                                                  /* T12 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm2, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);        /* Store T12 -> B2 */

            SET_ZERO_PACKED(ymm2, ymm2, ymm2);                                                                              /* ZERO T12 */
        }
    }
    if (n_in & 1) {
        for (i=0; i<(m_in/M_UNROLL_AVX2)*M_UNROLL_AVX2; i+=M_UNROLL_AVX2) {
            /* gemm update */
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);    /* A1 */
                VMOVU_PACKED(ymm6, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+ii,lda,_is_row_))), 0);    /* A2 */
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);
                VFMADD231_PACKED(ymm1, ymm4, ymm6);

            }
            /* update the 4x4 B matrix */
            /* 1st */
            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        /* B1 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif
            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

            /* 2nd */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);        /* A2 */

            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);        /* B1 */

            VFMADD231_PACKED(ymm1, ymm5, ymm0);                                                                             /* T21 += A2*T11 */
            VSUB_PACKED(ymm1, ymm8, ymm1);                                                                                  /* T21 = B1 - T21 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm1, ymm1, ymm4);                                                                                  /* T21 *= ONE/A1 */
         }
#endif
            VMOVU_PACKED(ymm1, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);        /* Store T21 -> B1 */

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              /* ZERO T11 */
            SET_ZERO_PACKED(ymm1, ymm1, ymm1);                                                                              /* ZERO T21 */

        }
        if (m_in & 1) {
           /* gemm update */
            for (ii=0; ii<i; ii++) {
                VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+ii,lda,_is_row_))), 0);   /* A1 */
                    
                VMOVU_PACKED(ymm4, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+ii,0+j,ldb,_is_row_))), 0);
                VFMADD231_PACKED(ymm0, ymm4, ymm5);

            }
            /* update the 4x4 B matrix */
            /* 1st */
            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 0);        /* B1 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nonunit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

        }

    }

  }

  int i = io_generated_code->code_size;
  unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
  buf[i++] = 0xc3; /* retq */
  io_generated_code->code_size = i;

  /* close asm: note that we really didn't need to push everything */
/*
  libxsmm_x86_instruction_close_stream_transpose( io_generated_code, i_arch );
*/
#ifdef GENERATOR_TRANSPOSE_DEBUG
  printf("done with m=%d n=%d i=%d\n",i_trans_desc->m,i_trans_desc->n,io_generated_code->code_size);
#endif

#ifdef DEBUG_GIVE_BYTE_CODE_OUTPUT
   buf = (unsigned char *) io_generated_code->generated_code;
   printf("#Final Routine: \n");
   for ( i = 0 ; i < io_generated_code->code_size ; i+=8 )
   {
      printf("#\tBytes %d-%d\n",i,i+7);
      printf(".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]);
   }
#endif

}
