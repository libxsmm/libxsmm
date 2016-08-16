/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MAIN_H
#define LIBXSMM_MAIN_H

#include <libxsmm_conv.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXSMM_REGSIZE) /* must be POT */
# define LIBXSMM_REGSIZE 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif

#define LIBXSMM_DEFINE_SIZE1(TYPE, NAME, NDIMS, SHAPE, INIT) \
  TYPE NAME = LIBXSMM_MAX(INIT, 1); { unsigned int libxsmm_define_size1_i_ = 0; \
  LIBXSMM_REPEAT(NDIMS, \
    NAME *= (TYPE)((SHAPE)[libxsmm_define_size1_i_]); \
    ++libxsmm_define_size1_i_); \
  }
/* TODO: LIBXSMM_DEFINE_INDEX1 plus PITCH */
#define LIBXSMM_DEFINE_INDEX1(TYPE, NAME, NDIMS, INDEXN, SHAPE) \
  TYPE NAME = 0; { TYPE libxsmm_define_index1_size_ = 1; unsigned int libxsmm_define_index1_i_ = 0; \
  LIBXSMM_REPEAT(NDIMS, \
    NAME += libxsmm_define_index1_size_ * ((TYPE)(INDEXN)[libxsmm_define_index1_i_]); \
    libxsmm_define_index1_size_ *= (TYPE)((SHAPE)[libxsmm_define_index1_i_]); \
    ++libxsmm_define_index1_i_); \
  }
#define LIBXSMM_DEFINE_SOMETYPE(NAME, TYPESIZE) \
  typedef struct LIBXSMM_RETARGETABLE NAME { char byte[TYPESIZE]; } NAME


typedef union LIBXSMM_RETARGETABLE libxsmm_code_pointer {
#if defined(LIBXSMM_BUILD) || defined(LIBXSMM_CONV_INTERNAL_API)
  libxsmm_sconvfunction sconv;
#endif
  libxsmm_xmmfunction xmm;
  /*const*/void* pmm;
  uintptr_t imm;
} libxsmm_code_pointer;

typedef struct LIBXSMM_RETARGETABLE libxsmm_csr_soa_descriptor {
  const libxsmm_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxsmm_csr_soa_descriptor;

/** struct which holds description of a layer */
struct LIBXSMM_RETARGETABLE libxsmm_conv_layer {
  int N;                            /* number of images in minibatch */
  int splits;                       /* number of splits */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int H;                            /* height of image */
  int W;                            /* width of image */
  libxsmm_conv_datatype datatype;   /* data type */
  void* data;                       /* pointer to data */  
};

/** struct which holds description of a bias */
struct LIBXSMM_RETARGETABLE libxsmm_conv_bias {
  int splits;                       /* number of splits */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  libxsmm_conv_datatype datatype;   /* data type */
  void* data;                       /* pointer to data */  
};

/** struct which holds description of a filter */
struct LIBXSMM_RETARGETABLE libxsmm_conv_filter {
  int splits;                       /* number of splits */
  int ifmb;                         /* number of feature map blocks */
  int bifm;                         /* sized of blocked feature maps, in a block */
  int ofmb;                         /* number of feature map blocks */
  int bofm;                         /* sized of blocked feature maps, in a block */  
  int R;                            /* height of filter kernel */
  int S;                            /* width of filter kernel */
  libxsmm_conv_datatype datatype;   /* data type */
  void* data;                       /* pointer to data */ 
};

struct LIBXSMM_RETARGETABLE libxsmm_conv_handle {
  libxsmm_conv_datatype datatype;
  libxsmm_conv_desc desc;
  libxsmm_conv_algo algo;

  /* additional size for iternal data types */
  int ifhp;
  int ifwp;
  int ofh;
  int ofw;
  int ofhp;
  int ofwp;
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int fwd_ofw_rb;
  int fwd_ofh_rb;
  
  /* internal data representation */
  libxsmm_conv_layer* input;
  libxsmm_conv_layer* output;
  libxsmm_conv_layer* input_relu;
  libxsmm_conv_filter* filter;
  libxsmm_conv_bias* bias;
  void* scratch;

  /* JIT-generated convolution code */
  /*
  libxsmm_convolution_forward_descriptor       fwd_desc;
  libxsmm_convolution_forward_descriptor       bwd_desc;
  libxsmm_convolution_weight_update_descriptor wu_desc;
  */
  libxsmm_code_pointer code_fwd[4];
  libxsmm_code_pointer code_bwd[8];
  libxsmm_code_pointer code_upd[4];
};

typedef enum libxsmm_build_kind {
  LIBXSMM_BUILD_KIND_GEMM,
  LIBXSMM_BUILD_KIND_SSOA,
  LIBXSMM_BUILD_KIND_CFWD,
  LIBXSMM_BUILD_KIND_CBWD,
  LIBXSMM_BUILD_KIND_CUPD
} libxsmm_build_kind;

typedef struct LIBXSMM_RETARGETABLE libxsmm_build_request {
  union LIBXSMM_RETARGETABLE {
    const libxsmm_gemm_descriptor* gemm;
    const libxsmm_csr_soa_descriptor* ssoa;
    const libxsmm_convolution_forward_descriptor* cfwd;
    const libxsmm_convolution_backward_descriptor* cbwd;
    const libxsmm_convolution_weight_update_descriptor* cupd;
  } descriptor;
  libxsmm_build_kind kind;
} libxsmm_build_request;


LIBXSMM_API void libxsmm_build(const libxsmm_build_request* request, unsigned regindex, libxsmm_code_pointer* code);

#endif /*LIBXSMM_MAIN_H*/

