/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Anand Venkat (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include <libxsmm_macros.h>

extern "C" int  batch_reduce_kernel_update(const float *weight, const float *input, float *output, int blocks, int ofmblock, int ifmblock, int ofw, int stride_w, int r, int s, int ifh, int ifw){
    int ld_b = stride_w*ifmblock;
    libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr(ofmblock,ofw, ifmblock,NULL,&ld_b,NULL,NULL,NULL, NULL, NULL);
    const unsigned long long cblocks = blocks;
    const float * A[cblocks];
    const float * B[cblocks];
    int weight_stride = ofmblock*ifmblock*r*s;
    int input_stride = ifw*ifh*ifmblock;
    if(r == 1 && s == 1){
        for (int icb = 0; icb < cblocks; icb ++) {
            A[icb] = &weight[icb*weight_stride];
            B[icb] = &input[icb*input_stride];
        }
    }else{/*Eg.if( r == 3 &&  s == 3){*/
         for( int k = 0 ; k < blocks/(r*s); k++){
            for(int i=0; i < r; i++){
                for(int j =0; j < s; j++){
                    A[k*r*s + i*s + j] = &weight[k*r*s*ofmblock*ifmblock +  (i*s + j)*ofmblock*ifmblock];
                    B[k*r*s + i*s + j] = &input[k*ifw*ifh*ifmblock  +  i*ifw*ifmblock + j*ifmblock];
                }
            }
        }
    }

    /* Reduce batch gemm call  */
    batchreduce_kernela(A, B, output, &cblocks);

    return 0;
}

extern "C" int  batch_reduce_kernel_init_update(const float *weight, const float *input, float *output, int blocks, int ofmblock, int ifmblock,int r, int s, int ifh, int ifw,int ofw, int stride_w ){
    float beta = 0.0;
    int lda = ofmblock;
    int ldx = ofmblock;
    int ld_b = stride_w*ifmblock;
    int l_flags = ( LIBXSMM_GEMM_FLAGS('N', 'N') );
    libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr(ofmblock,ofw, ifmblock,&lda,&ld_b,&ldx,NULL,&beta, &l_flags, NULL);

    const unsigned long long cblocks = blocks;
    const float * A[cblocks];
    const float * B[cblocks];
    int weight_stride = ofmblock*ifmblock*r*s;
    int input_stride = ifw*ifh*ifmblock;
    if(r == 1 && s == 1){
    for (int icb = 0; icb < cblocks; icb ++) {
            A[icb] = &weight[icb*weight_stride];
            B[icb] = &input[icb*input_stride];
    }
    }else{ /*if( r == 3 &&  s == 3){*/
      for( int k = 0 ; k < blocks/(r*s); k++)
       for(int i=0; i < r; i++)
         for(int j =0; j < s; j++){
              A[k*r*s + i*s + j] = &weight[k*r*s*ofmblock*ifmblock +  (i*s + j)*ofmblock*ifmblock];
              B[k*r*s + i*s + j] = &input[k*ifw*ifh*ifmblock  +  i*ifw*ifmblock + j*ifmblock];
         }

    }
    /* Reduce batch gemm call  */
    batchreduce_kernela(A, B, output, &cblocks);


    return 0;
}

extern "C" int  batch_reduce_kernel_init(float *output, int ofmblock, int ofw){
    int num_elements = ofw*ofmblock;

    LIBXSMM_PRAGMA_SIMD
    for(int i=0; i < num_elements; i++)
          output[i] = 0.0;

    return 0;
}


