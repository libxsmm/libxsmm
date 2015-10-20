/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#ifndef LIBXSMM_H
#define LIBXSMM_H

/** Parameters the library was built for. */
#define LIBXSMM_ALIGNMENT $ALIGNMENT
#define LIBXSMM_ALIGNED_STORES $ALIGNED_STORES
#define LIBXSMM_ALIGNED_LOADS $ALIGNED_LOADS
#define LIBXSMM_ALIGNED_MAX $ALIGNED_MAX
#define LIBXSMM_PREFETCH $PREFETCH
#define LIBXSMM_ROW_MAJOR $ROW_MAJOR
#define LIBXSMM_COL_MAJOR $COL_MAJOR
#define LIBXSMM_MAX_MNK $MAX_MNK
#define LIBXSMM_MAX_M $MAX_M
#define LIBXSMM_MAX_N $MAX_N
#define LIBXSMM_MAX_K $MAX_K
#define LIBXSMM_AVG_M $AVG_M
#define LIBXSMM_AVG_N $AVG_N
#define LIBXSMM_AVG_K $AVG_K
#define LIBXSMM_BETA $BETA
#define LIBXSMM_JIT $JIT

#include "libxsmm_macros.h"
#include "libxsmm_prefetch.h"
#include "libxsmm_fallback.h"


/** Explictly initializes the library; can be used to pay for setup cost at a specific point. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_build_static();

/** Generic type of a function. */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_function)(LIBXSMM_VARIADIC);

/** JIT-generate a function and make it available for later dispatch. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_function libxsmm_build_jit(int single_precision, int m, int n, int k);

/** Type of a function generated for a specific M, N, and K. */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_smm_function)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
                                    LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pa)
                                    LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pb)
                                    LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pc));
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmm_function)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
                                    LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pa)
                                    LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pb)
                                    LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pc));

/** Query the pointer of a generated function; zero if it does not exist. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smm_function libxsmm_smm_dispatch(int m, int n, int k);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmm_function libxsmm_dmm_dispatch(int m, int n, int k);

/** Dispatched matrix-matrix multiplication; single-precision. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_smm(int m, int n, int k, const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  LIBXSMM_MM(float, m, n, k, a, b, c, LIBXSMM_PREFETCH_ARG_pa, LIBXSMM_PREFETCH_ARG_pb, LIBXSMM_PREFETCH_ARG_pc);
}

/** Dispatched matrix-matrix multiplication; double-precision. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dmm(int m, int n, int k, const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  LIBXSMM_MM(double, m, n, k, a, b, c, LIBXSMM_PREFETCH_ARG_pa, LIBXSMM_PREFETCH_ARG_pb, LIBXSMM_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using inline code; single-precision. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_simm(int m, int n, int k, const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  LIBXSMM_IMM(float, int, m, n, k, a, b, c, LIBXSMM_PREFETCH_ARG_pa, LIBXSMM_PREFETCH_ARG_pb, LIBXSMM_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using inline code; double-precision. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dimm(int m, int n, int k, const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  LIBXSMM_IMM(double, int, m, n, k, a, b, c, LIBXSMM_PREFETCH_ARG_pa, LIBXSMM_PREFETCH_ARG_pb, LIBXSMM_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; single-precision. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_sblasmm(int m, int n, int k, const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c) {
  LIBXSMM_BLASMM(float, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; double-precision. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dblasmm(int m, int n, int k, const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c) {
  LIBXSMM_BLASMM(double, m, n, k, a, b, c);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Dispatched matrix-matrix multiplication. */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k, const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  libxsmm_smm(m, n, k, a, b, c LIBXSMM_PREFETCH_ARGA(pa) LIBXSMM_PREFETCH_ARGB(pb) LIBXSMM_PREFETCH_ARGC(pc));
}

/** Dispatched matrix-matrix multiplication. */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k, const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  libxsmm_dmm(m, n, k, a, b, c LIBXSMM_PREFETCH_ARGA(pa) LIBXSMM_PREFETCH_ARGB(pb) LIBXSMM_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXSMM_RETARGETABLE inline void libxsmm_imm(int m, int n, int k, const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const float *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  libxsmm_simm(m, n, k, a, b, c LIBXSMM_PREFETCH_ARGA(pa) LIBXSMM_PREFETCH_ARGB(pb) LIBXSMM_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXSMM_RETARGETABLE inline void libxsmm_imm(int m, int n, int k, const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pa) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pb) LIBXSMM_PREFETCH_DECL(const double *LIBXSMM_RESTRICT, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  libxsmm_dimm(m, n, k, a, b, c LIBXSMM_PREFETCH_ARGA(pa) LIBXSMM_PREFETCH_ARGB(pb) LIBXSMM_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k, const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_sblasmm(m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k, const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dblasmm(m, n, k, a, b, c);
}

/** Call libxsmm_smm_dispatch, or libxsmm_dmm_dispatch depending on T. */
template<typename T> class LIBXSMM_RETARGETABLE libxsmm_mm_dispatch { typedef void function_type; };

template<> class LIBXSMM_RETARGETABLE libxsmm_mm_dispatch<float> {
  typedef libxsmm_smm_function function_type;
  mutable/*retargetable*/ function_type m_function;
public:
  libxsmm_mm_dispatch(): m_function(0) {}
  libxsmm_mm_dispatch(int m, int n, int k): m_function(libxsmm_smm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

template<> class LIBXSMM_RETARGETABLE libxsmm_mm_dispatch<double> {
  typedef libxsmm_dmm_function function_type;
  mutable/*retargetable*/ function_type m_function;
public:
  libxsmm_mm_dispatch(): m_function(0) {}
  libxsmm_mm_dispatch(int m, int n, int k): m_function(libxsmm_dmm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

#endif /*__cplusplus*/

#endif /*LIBXSMM_H*/
