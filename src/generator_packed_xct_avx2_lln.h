#if 0
# define GENERATOR_PACKED_XCT_DEBUG
#endif
#ifdef GENERATOR_PACKED_XCT_DEBUG
printf("******* HI *********\n");
#endif
#include "packed_trsm_dmacros.h"
    auto int a_ptr = LIBXSMM_X86_GP_REG_RDI;
    auto int b_ptr = LIBXSMM_X86_GP_REG_RSI;
    char i_vector_name = 'y';
#ifdef GENERATOR_PACKED_XCT_DEBUG
printf("Inside %c%c%c%c trsm generator\n",*side_ptr,*uplo_ptr,*transa_ptr,*diag_ptr);
#endif
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
    if ( nounit && (datasz==8) )
    {
        double one_vector[8] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
        i = io_code->code_size;
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) one_vector, "one_vec", i_vector_name, 15 );
        i = io_code->code_size;
    }
    if ( nounit && (datasz==4) )
    {
        float one_vector[16] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
        i = io_code->code_size;

        libxsmm_x86_instruction_full_vec_load_of_constants ( io_code, (unsigned char*) one_vector, "one_vec", i_vector_name, 15 );
        i = io_code->code_size;
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

    if ( datasz == 8 ) P_UNROLL_AVX2 = P_UNROLL_AVX2_F64;
    else                 P_UNROLL_AVX2 = P_UNROLL_AVX2_F32;

    /* zero accumulation registers */
    i = io_code->code_size;
    SET_ZERO_PACKED(ymm0, ymm0, ymm0);                          /* T11*/
    i = io_code->code_size;
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

            VMOVU_PACKED(ymm9, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 0);        /* B2 */

         if ( nounit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */

         if ( nounit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }

            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

            VSUB_PACKED(ymm2, ymm9, ymm2);                                                                                  /* T12 = B2-T12 */

         if ( nounit ) {
            VMUL_PACKED(ymm2, ymm2, ymm4);                                                                                  /* T12 *= ONE/A1 */
         }

            VMOVU_PACKED(ymm2, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,1+j,ldb,_is_row_))), 1);        /* Store T12 -> B2 */

            /* 2nd */
         if ( nounit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }

            VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);        /* A2 */

            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);        /* B1 */
            VMOVU_PACKED(ymm9, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,1+j,ldb,_is_row_))), 0);        /* B2 */

            VFMADD231_PACKED(ymm1, ymm5, ymm0);                                                                             /* T21 += A2*T11 */
            VSUB_PACKED(ymm1, ymm8, ymm1);                                                                                  /* T21 = B1 - T21 */

         if ( nounit ) {
            VMUL_PACKED(ymm1, ymm1, ymm4);                                                                                  /* T21 *= ONE/A1 */
         }
            VMOVU_PACKED(ymm1, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 1);        /* Store T21 -> B1 */

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              /* ZERO T11 */
            SET_ZERO_PACKED(ymm1, ymm1, ymm1);                                                                              /* ZERO T21 */

            VFMADD231_PACKED(ymm3, ymm5, ymm2);                                                                             /* T22 += A2*T12 */
            VSUB_PACKED(ymm3, ymm9, ymm3);                                                                                  /* T22 = B2 - T22 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
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
         if ( nounit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

            SET_ZERO_PACKED(ymm0, ymm0, ymm0);                                                                              /* ZERO T11 */

            VSUB_PACKED(ymm2, ymm9, ymm2);                                                                                  /* T12 = B2-T12 */

#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
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
         if ( nounit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif
            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

            /* 2nd */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,1+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VMOVU_PACKED(ymm5, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(1+i,0+i,lda,_is_row_))), 0);        /* A2 */

            VMOVU_PACKED(ymm8, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(1+i,0+j,ldb,_is_row_))), 0);        /* B1 */

            VFMADD231_PACKED(ymm1, ymm5, ymm0);                                                                             /* T21 += A2*T11 */
            VSUB_PACKED(ymm1, ymm8, ymm1);                                                                                  /* T21 = B1 - T21 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
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
         if ( nounit ) {
            VMOVU_PACKED(ymm4, a_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_ap(0+i,0+i,lda,_is_row_))), 0);        /* A1 */
            VDIV_PACKED(ymm4, ymm15, ymm4);                                                                                 /* A1 /= ONE */
         }
#endif

            VSUB_PACKED(ymm0, ymm8, ymm0);                                                                                  /* T11 = B1-T11 */
#ifdef _XCT_NOUNIT_DIAG_
         if ( nounit ) {
            VMUL_PACKED(ymm0, ymm0, ymm4);                                                                                  /* T11 *= ONE/A1 */
         }
#endif

            VMOVU_PACKED(ymm0, b_ptr, sizeof(xct_ftype)*(P_UNROLL_AVX2*(trsm_ll_bp(0+i,0+j,ldb,_is_row_))), 1);        /* Store T11 -> B1 */

        }

    }
