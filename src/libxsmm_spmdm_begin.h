/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
#define TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_A, ldA, ptr_B, ldB) do { \
  uint16_t restmp = (*(ptr_A)); \
  union { int i; float f; } res; \
  res.i = restmp; \
  res.i <<= 16; \
  (*(ptr_B)) = res.f; \
} while(0)

#define COMPRESS_FP32(v, k, m, cnt) if (m) do { \
  values_ptr[cnt] = v; \
  colidx_ptr[cnt] = (uint16_t)(k); \
  cnt++; \
} while(0)

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) do { \
  union { int i; float f; } vlo_tmp, vhi_tmp; \
  vlo_tmp.i = (v) & 0xFFFF; vlo_tmp.i <<= 16; \
  vlo_final = vlo_tmp.f; \
  vhi_tmp.i = (v) & 0x0000FFFF; \
  vhi_final = vhi_tmp.f; \
} while(0)

#define COMPRESS_BFLOAT16(vlo, vhi, v) do { \
  union { int i; float f; } vlo_tmp, vhi_tmp; \
  vlo_tmp.f = vlo; \
  v = (vlo_tmp.i >> 16); \
  vhi_tmp.f = vhi; \
  v = v | (vhi_tmp.i & 0xFFFF0000); \
} while(0)

