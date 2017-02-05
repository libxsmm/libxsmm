/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

LIBXSMM_INLINE void print_help(void) {
  printf("\nUsage:\n");
  printf("    inlineasm/plainasm\n");
  printf("    filename to append\n");
  printf("    routine name\n");
  printf("    kh\n");
  printf("    kw\n");
  printf("    unroll_kh\n");
  printf("    unroll_kw\n");
  printf("    ofm_block\n");
  printf("    ifm_block\n");
  printf("    ofh_padded\n");
  printf("    ofw_padded\n");
  printf("    ofh_rb\n");
  printf("    ofw_rb\n");
  printf("    ifh_padded\n");
  printf("    ifw_padded\n");
  printf("    stride_h\n");
  printf("    stride_w\n");
  printf("    precision (0=FP32,1=INT16)\n");
  printf("    ARCH: knl, knm, skx\n");
  printf("\n\n");
}

int main(int argc, char* argv []) {
  libxsmm_convolution_forward_descriptor l_conv_desc;
  char* l_type;
  char* l_file_out;
  char* l_routine_name;
  char* l_arch;
  int l_kw = 0;         /* kernel width */
  int l_kh = 0;         /* kernel height */
  int l_unroll_kw = 0;  /* kernel width, unrolled? */
  int l_unroll_kh = 0;  /* kernel height, unrolled? */
  int l_ofm_block = 0;  /* should be VLEN */
  int l_ifm_block = 0;  /* should be VLEN */
  int l_ofh_padded = 0; /* this we need for 2D register block */
  int l_ofw_padded = 0; /* this we use for 1D and 2D register block */
  int l_ifh_padded = 0;
  int l_ifw_padded = 0;
  int l_stride_h = 0;   /* this we use for offsets in the input */
  int l_stride_w = 0;   /* this we use for offsets in the input */
  int l_ofw_rb = 0;     /* UR, register block of ofw */
  int l_ofh_rb = 0;     /* register block of ofh */
  int l_prec = 0;

  /* check argument count for a valid range */
  if ( argc != 20 ) {
    print_help();
    return -1;
  }

  /* names of files and routines */
  l_type = argv[1];
  l_file_out = argv[2];
  l_routine_name = argv[3];

  /* convolution sizes */
  l_kh = atoi(argv[4]);
  l_kw = atoi(argv[5]);
  l_unroll_kh = atoi(argv[6]);
  l_unroll_kw = atoi(argv[7]);
  l_ofm_block = atoi(argv[8]);
  l_ifm_block = atoi(argv[9]);
  l_ofh_padded = atoi(argv[10]);
  l_ofw_padded = atoi(argv[11]);
  l_ofh_rb = atoi(argv[12]);
  l_ofw_rb = atoi(argv[13]);
  l_ifh_padded = atoi(argv[14]);
  l_ifw_padded = atoi(argv[15]);
  l_stride_h = atoi(argv[16]);
  l_stride_w = atoi(argv[17]);
  l_prec = atoi(argv[18]);

  /* arch specific stuff */
  l_arch = argv[19];

  /* some intial parameters checks */
  /* check for sparse / dense only */
  if ( (strcmp(l_type, "inlineasm") != 0) &&
       (strcmp(l_type, "plainasm")  != 0) ) {
    print_help();
    return -1;
  }

  /* check value of arch flag */
  if ( (strcmp(l_arch, "knl") != 0)    &&
       (strcmp(l_arch, "knm") != 0)    &&
       (strcmp(l_arch, "skx") != 0) ) {
    print_help();
    return -1;
  }

  /* set up convultion descriptor */
  l_conv_desc.kh = l_kh;
  l_conv_desc.kw = l_kw;
  l_conv_desc.unroll_kh = l_unroll_kh;
  l_conv_desc.unroll_kw = l_unroll_kw;
  l_conv_desc.ofm_block = l_ofm_block;
  l_conv_desc.ifm_block = l_ifm_block;
  l_conv_desc.ofh_padded = l_ofh_padded;
  l_conv_desc.ofw_padded = l_ofw_padded;
  l_conv_desc.ofh_rb = l_ofh_rb;
  l_conv_desc.ofw_rb = l_ofw_rb;
  l_conv_desc.ifh_padded = l_ifh_padded;
  l_conv_desc.ifw_padded = l_ifw_padded;
  l_conv_desc.stride_h = l_stride_h;
  l_conv_desc.stride_w = l_stride_w;
  switch (l_prec)
  {
    case 0:
      l_conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;
      l_conv_desc.datatype_itm = LIBXSMM_DNN_DATATYPE_F32;
      break;
    case 1:
      l_conv_desc.datatype = LIBXSMM_DNN_DATATYPE_I16;
      l_conv_desc.datatype_itm = LIBXSMM_DNN_DATATYPE_I32;
      break;
    default:
      l_conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;
      l_conv_desc.datatype_itm = LIBXSMM_DNN_DATATYPE_F32;
      break;
  }
  l_conv_desc.prefetch = LIBXSMM_CONVOLUTION_PREFETCH_NONE;
  l_conv_desc.format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;

  /* generate code */
  if ( strcmp(l_type, "inlineasm")  == 0 ) {
    libxsmm_generator_convolution_forward_inlineasm( l_file_out, l_routine_name, &l_conv_desc, l_arch );
  } else {
    libxsmm_generator_convolution_forward_directasm( l_file_out, l_routine_name, &l_conv_desc, l_arch );
  }

  return 0;
}

