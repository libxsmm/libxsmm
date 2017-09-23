/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Rajkishore Barik, Alexander Heinecke, Ankush Mandal, Jason Sewall (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_weight_update.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"
#include "stdio.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 1
void gather_transpose_ps_16_56_56_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x00FF;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n;
    #pragma unroll(3)
    for(n = 0; n < 3; ++n) {
      const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
      _mm512_store_ps((void*)(dst+m*56+n*16),tmp);
    }
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*56+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_58_60_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x03FF;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n;
    #pragma unroll(3)
    for(n = 0; n < 3; ++n) {
      const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
      _mm512_store_ps((void*)(dst+m*60+n*16),tmp);
    }
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*60+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_28_28_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x0FFF;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*28+n*16),tmp);
    n = 1;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*28+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_30_32_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*32+n*16),tmp);
    n = 1;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*32+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_16_16_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*16+n*16),tmp);
  }
}

void gather_transpose_ps_16_14_16_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*16+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_7_8_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(0, 97, 81, 65, 49, 33, 17,  1,
                                          0, 96, 80, 64, 48, 32, 16,  0);
  const __mmask16 Nremmask = 0x7F7F;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 8; ++m) {
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m*2, 4);
    _mm512_mask_store_ps((void*)(dst+m*8*2),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_9_12_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x01FF;
  int m;
  #pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*12+n*16),Nremmask,tmprem);
  }
}

void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex_base = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  const __m512i vindex = _mm512_mullo_epi32(_mm512_set1_epi32(ldS), vindex_base);

  const int whole16s = N/16;
  const int remainder = N-whole16s*16;
  const __mmask16 Nmask = (1<<remainder)-1;
  int i;
  #pragma unroll_and_jam(2)
  for(i = 0; i < M; ++i) {
    int j;
    #pragma unroll(4)
    for(j = 0; j < whole16s; ++j) {
      const __m512 res = _mm512_i32gather_ps(vindex, src+i+j*16*ldS, 4);
      _mm512_store_ps(dst + ldD*i+j*16, res);
    }
    if(remainder) {
      const __m512 res = _mm512_mask_i32gather_ps(_mm512_undefined(), Nmask, vindex, src+i+j*16*ldS, 4);
      _mm512_mask_store_ps(dst + ldD*i+j*16, Nmask, res);
    }
  }
}
#else
void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  int n, m;
  for (n = 0; n < N; ++n) {
    for (m = 0; m < M; ++m) {
      dst[m*ldD + n] = src[n*ldS + m];
    }
  }
}
#endif //defined(__AVX512F__)

typedef void (*transposer)(int M, int N, float *dst, int ldD, const float *src, int ldS);

transposer get_transposer(int M, int N, int ldD, int ldS) {
  if(M == 16 && N == 7 && ldD == 8 && ldS == 16) {
    return gather_transpose_ps_16_7_8_16;
  }
  if(M == 16 && N == 9 && ldD == 12 && ldS == 16) {
    return gather_transpose_ps_16_9_12_16;
  }
  if(M == 16 && N == 14 && ldD == 16 && ldS == 16) {
    return gather_transpose_ps_16_14_16_16;
  }
  if(M == 16 && N == 16 && ldD == 16 && ldS == 16) {
    return gather_transpose_ps_16_16_16_16;
  }
  if(M == 16 && N == 28 && ldD == 28 && ldS == 16) {
    return gather_transpose_ps_16_28_28_16;
  }
  if(M == 16 && N == 30 && ldD == 32 && ldS == 16) {
    return gather_transpose_ps_16_30_32_16;
  }
  if(M == 16 && N == 56 && ldD == 56 && ldS == 16) {
    return gather_transpose_ps_16_56_56_16;
  }
  if(M == 16 && N == 58 && ldD == 60 && ldS == 16) {
    return gather_transpose_ps_16_58_60_16;
  }

  return transpose_fallback;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch3 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we scratch for MB parallel execution */
  if ( (handle->upd_use_thread_fil == 1) && (handle->scratch4 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_upd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
        #define INPUT_PADDING
        # include "template/libxsmm_dnn_convolve_st_upd_custom_custom_fallback.tpl.c"
        #undef INPUT_PADDING
      } else {
        # include "template/libxsmm_dnn_convolve_st_upd_custom_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        typedef libxsmm_smmfunction libxsmm_mmfunction;
        if (handle->padding_flag == 1) {
          #define INPUT_PADDING
          #define LIBXSMM_WU_PER_THREAD_ALLOCATION
          # include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
          #undef LIBXSMM_WU_PER_THREAD_ALLOCATION
          #undef INPUT_PADDING
        } else {
          #define LIBXSMM_WU_PER_THREAD_ALLOCATION
          # include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
          #undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
      }
      #if 1
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        typedef libxsmm_smmfunction libxsmm_mmfunction;
        if (handle->padding_flag == 1) {
          #define INPUT_PADDING
          if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM)
               && (handle->desc.v == 1) && (handle->upd_ofw_rb%4 == 0) )
          {
            #define LIBXSMM_WU_TRANSPOSE_OFW_IFM
            # include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
            #undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
            # include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
          }
          #undef INPUT_PADDING
        } else {
          if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM)
               && (handle->desc.v == 1) && (handle->upd_ofw_rb%4 == 0) )
          {
            #define LIBXSMM_WU_TRANSPOSE_OFW_IFM
            # include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
            #undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
            # include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
          }
        }
      }
      #endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch3 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we scratch for MB parallel execution */
  if ( (handle->upd_use_thread_fil == 1) && (handle->scratch4 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_upd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef INPUT_PADDING
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        } else {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
          }
#undef INPUT_PADDING
        } else {
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch3 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we scratch for MB parallel execution */
  if ( (handle->upd_use_thread_fil == 1) && (handle->scratch4 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_upd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
#undef INPUT_PADDING
        } else {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
          }
#undef INPUT_PADDING
        } else {
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}
