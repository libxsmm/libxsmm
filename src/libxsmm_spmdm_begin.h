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
/* Nadathur Satish, Hans Pabst (Intel Corp.)
******************************************************************************/

#define SIMD_WIDTH_FP32 (1)
#define SIMDTYPE_FP32 float
#define SIMDTYPE_INT32 int
#define SIMDMASKTYPE_FP32 int
#define _MM_SETZERO_FP32() (0)
#define _MM_SETZERO_INT32() (0)
#define _MM_SET1_FP32(x) (x)
#define _MM_SET1_INT32(x) (x)
#define _MM_SET1_INT16 (x)
#define _MM_LOAD_FP32(x) (*(x))
#define _MM_LOADU_FP32(x) (*(x))
#define _MM_LOAD_INT32(x) (*(x))
#define _MM_STORE_INT32(x,y) ((*(x)) = (y))
#define _MM_LOADU_INT32(x) (*(x))
#define _MM_GATHER_FP32(Addr, idx, scale) (*(Addr + (idx)))
#define _MM_CMPNEQ_FP32(v1,v2) (LIBXSMM_FEQ(v1, v2) ? 0 : 1)
#define _MM_STORE_FP32(x,y) ((*(x)) = (y))
#define _MM_STOREU_FP32(x,y) ((*(x)) = (y))
#define _MM_ADD_FP32(x,y) ((x) + (y))
#define _MM_FMADD_FP32(x,y,z) (((x)*(y))+(z))
#define _MM_MUL_FP32(x,y) ((x)*(y))
#define _MM_PREFETCH(x, y)
#define TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_A, ldA, ptr_B, ldB) ((*(ptr_B)) = (*(ptr_A)))
#define TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_A, ldA, ptr_B, ldB) \
{\
            uint16_t restmp = (*(ptr_A));\
            union { int i; float f; } res;\
            res.i = restmp;\
            res.i <<= 16;\
            (*(ptr_B)) = res.f;\
}

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  if (m) \
  { \
    values_ptr[cnt] = v; \
    colidx_ptr[cnt] = (uint16_t)(k); \
    cnt++; \
  } \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
    union { int i; float f; } vlo_tmp, vhi_tmp; \
    vlo_tmp.i = (v) & 0xFFFF; vlo_tmp.i <<= 16; \
    vlo_final = vlo_tmp.f; \
    vhi_tmp.i = (v) & 0x0000FFFF; \
    vhi_final = vhi_tmp.f; \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
    union { int i; float f; } vlo_tmp, vhi_tmp; \
    vlo_tmp.f = vlo; \
    v = (vlo_tmp.i >> 16); \
    vhi_tmp.f = vhi; \
    v = v | (vhi_tmp.i & 0xFFFF0000); \
  }

