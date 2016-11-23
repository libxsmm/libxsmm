/******************************************************************************
** Copyright (c) 2013-2016, Intel Corporation                                **
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
#ifndef LIBXSMM_MACROS_H
#define LIBXSMM_MACROS_H

#include "libxsmm_config.h"

#define LIBXSMM_STRINGIFY2(SYMBOL) #SYMBOL
#define LIBXSMM_STRINGIFY(SYMBOL) LIBXSMM_STRINGIFY2(SYMBOL)
#define LIBXSMM_TOSTRING(SYMBOL) LIBXSMM_STRINGIFY(SYMBOL)
#define LIBXSMM_CONCATENATE2(A, B) A##B
#define LIBXSMM_CONCATENATE(A, B) LIBXSMM_CONCATENATE2(A, B)
#define LIBXSMM_FSYMBOL(SYMBOL) LIBXSMM_CONCATENATE2(SYMBOL, _)
#define LIBXSMM_UNIQUE(NAME) LIBXSMM_CONCATENATE(NAME, __LINE__)
#define LIBXSMM_EXPAND(A) A

#define LIBXSMM_VERSION2(MAJOR, MINOR) ((MAJOR) * 10000 + (MINOR) * 100)
#define LIBXSMM_VERSION3(MAJOR, MINOR, UPDATE) (LIBXSMM_VERSION2(MAJOR, MINOR) + (UPDATE))
#define LIBXSMM_VERSION4(MAJOR, MINOR, UPDATE, PATCH) ((MAJOR) * 100000000 + (MINOR) * 1000000 + (UPDATE) * 10000 + (PATCH))

#if defined(__cplusplus)
# define LIBXSMM_VARIADIC ...
# define LIBXSMM_EXTERN extern "C"
# define LIBXSMM_INLINE_KEYWORD inline
# define LIBXSMM_INLINE LIBXSMM_INLINE_KEYWORD
# define LIBXSMM_CALLER __FUNCTION__
#else
# define LIBXSMM_VARIADIC
# define LIBXSMM_EXTERN extern
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define LIBXSMM_PRAGMA(DIRECTIVE) _Pragma(LIBXSMM_STRINGIFY(DIRECTIVE))
#   define LIBXSMM_CALLER __func__
#   define LIBXSMM_RESTRICT restrict
#   define LIBXSMM_INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define LIBXSMM_CALLER __FUNCTION__
#   define LIBXSMM_INLINE_KEYWORD __inline
#   define LIBXSMM_INLINE_FIXUP
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
#   define LIBXSMM_CALLER __FUNCTION__
# endif
# if !defined(LIBXSMM_INLINE_KEYWORD)
#   define LIBXSMM_INLINE_KEYWORD
#   define LIBXSMM_INLINE_FIXUP
# endif
# if !defined(LIBXSMM_CALLER)
#   define LIBXSMM_CALLER 0
# endif
# define LIBXSMM_INLINE static LIBXSMM_INLINE_KEYWORD
#endif /*__cplusplus*/

#if !defined(LIBXSMM_INTERNAL_API)
# define LIBXSMM_INTERNAL_API LIBXSMM_EXTERN
#endif
#if !defined(LIBXSMM_INTERNAL_API_DEFINITION)
# define LIBXSMM_INTERNAL_API_DEFINITION LIBXSMM_INTERNAL_API
#endif
#if defined(LIBXSMM_BUILD)
# define LIBXSMM_INTERNAL_API_INLINE LIBXSMM_INTERNAL_API
#else
# define LIBXSMM_INTERNAL_API_INLINE LIBXSMM_INLINE
#endif

#define LIBXSMM_API LIBXSMM_INTERNAL_API LIBXSMM_RETARGETABLE
#define LIBXSMM_API_DEFINITION LIBXSMM_INTERNAL_API_DEFINITION LIBXSMM_RETARGETABLE
#define LIBXSMM_API_INLINE LIBXSMM_INTERNAL_API_INLINE LIBXSMM_RETARGETABLE

/* Some definitions kept for compatibility with earlier versions */
#if !defined(LIBXSMM_EXTERN_C) && defined(__cplusplus)
# define LIBXSMM_EXTERN_C LIBXSMM_EXTERN
#else
# define LIBXSMM_EXTERN_C
#endif

#if !defined(LIBXSMM_RESTRICT)
# if ((defined(__GNUC__) && !defined(__CYGWIN32__)) || defined(__INTEL_COMPILER)) && !defined(_WIN32)
#   define LIBXSMM_RESTRICT __restrict__
# elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#   define LIBXSMM_RESTRICT __restrict
# else
#   define LIBXSMM_RESTRICT
# endif
#endif /*LIBXSMM_RESTRICT*/

#if !defined(LIBXSMM_PRAGMA)
# if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#   define LIBXSMM_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
# else
#   define LIBXSMM_PRAGMA(DIRECTIVE)
# endif
#endif /*LIBXSMM_PRAGMA*/

#if defined(_MSC_VER)
# define LIBXSMM_MESSAGE(MSG) LIBXSMM_PRAGMA(message(MSG))
#elif LIBXSMM_VERSION3(4, 4, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__) \
   && LIBXSMM_VERSION3(5, 0, 0) >  LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
# define LIBXSMM_MESSAGE(MSG) LIBXSMM_PRAGMA(message MSG)
#else
# define LIBXSMM_MESSAGE(MSG)
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSMM_PRAGMA(simd reduction(EXPRESSION))
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N) LIBXSMM_PRAGMA(simd collapse(N))
# define LIBXSMM_PRAGMA_SIMD_PRIVATE(...) LIBXSMM_PRAGMA(simd private(__VA_ARGS__))
# define LIBXSMM_PRAGMA_SIMD LIBXSMM_PRAGMA(simd)
# define LIBXSMM_PRAGMA_NOVECTOR LIBXSMM_PRAGMA(novector)
#elif defined(_OPENMP) && (201307 <= _OPENMP) /*OpenMP 4.0*/
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXSMM_PRAGMA(omp simd reduction(EXPRESSION))
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N) LIBXSMM_PRAGMA(omp simd collapse(N))
# define LIBXSMM_PRAGMA_SIMD_PRIVATE(...) LIBXSMM_PRAGMA(omp simd private(__VA_ARGS__))
# define LIBXSMM_PRAGMA_SIMD LIBXSMM_PRAGMA(omp simd)
# define LIBXSMM_PRAGMA_NOVECTOR
#else
# define LIBXSMM_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXSMM_PRAGMA_SIMD_COLLAPSE(N)
# define LIBXSMM_PRAGMA_SIMD_PRIVATE(...)
# define LIBXSMM_PRAGMA_SIMD
# define LIBXSMM_PRAGMA_NOVECTOR
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSMM_PRAGMA_NONTEMPORAL_VARS(...) LIBXSMM_PRAGMA(vector nontemporal(__VA_ARGS__))
# define LIBXSMM_PRAGMA_NONTEMPORAL LIBXSMM_PRAGMA(vector nontemporal)
# define LIBXSMM_PRAGMA_VALIGNED_VARS(...) LIBXSMM_PRAGMA(vector aligned(__VA_ARGS__))
# define LIBXSMM_PRAGMA_VALIGNED LIBXSMM_PRAGMA(vector aligned)
# define LIBXSMM_PRAGMA_FORCEINLINE LIBXSMM_PRAGMA(forceinline)
# define LIBXSMM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXSMM_PRAGMA(loop_count min(MIN) max(MAX) avg(AVG))
# define LIBXSMM_PRAGMA_UNROLL_N(N) LIBXSMM_PRAGMA(unroll(N))
# define LIBXSMM_PRAGMA_UNROLL LIBXSMM_PRAGMA(unroll)
/*# define LIBXSMM_UNUSED(VARIABLE) LIBXSMM_PRAGMA(unused(VARIABLE))*/
#else
# define LIBXSMM_PRAGMA_NONTEMPORAL_VARS(...)
# define LIBXSMM_PRAGMA_NONTEMPORAL
# define LIBXSMM_PRAGMA_VALIGNED_VARS(...)
# define LIBXSMM_PRAGMA_VALIGNED
# define LIBXSMM_PRAGMA_FORCEINLINE
# define LIBXSMM_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXSMM_PRAGMA_UNROLL_N(N)
# define LIBXSMM_PRAGMA_UNROLL
#endif

#if defined(__clang__)
# define LIBXSMM_PRAGMA_OPTIMIZE_OFF LIBXSMM_PRAGMA(clang optimize off)
# define LIBXSMM_PRAGMA_OPTIMIZE_ON  LIBXSMM_PRAGMA(clang optimize on)
#else
# define LIBXSMM_PRAGMA_OPTIMIZE_OFF
# define LIBXSMM_PRAGMA_OPTIMIZE_ON
#endif

#if defined(_OPENMP) && (200805 <= _OPENMP) /*OpenMP 3.0*/
# define LIBXSMM_OPENMP_COLLAPSE(N) collapse(N)
#else
# define LIBXSMM_OPENMP_COLLAPSE(N)
#endif

#define LIBXSMM_REPEAT_1(A) A
#define LIBXSMM_REPEAT_2(A) LIBXSMM_REPEAT_1(A); A
#define LIBXSMM_REPEAT_3(A) LIBXSMM_REPEAT_2(A); A
#define LIBXSMM_REPEAT_4(A) LIBXSMM_REPEAT_3(A); A
#define LIBXSMM_REPEAT_5(A) LIBXSMM_REPEAT_4(A); A
#define LIBXSMM_REPEAT_6(A) LIBXSMM_REPEAT_5(A); A
#define LIBXSMM_REPEAT_7(A) LIBXSMM_REPEAT_6(A); A
#define LIBXSMM_REPEAT_8(A) LIBXSMM_REPEAT_7(A); A
#define LIBXSMM_REPEAT(N, A) LIBXSMM_CONCATENATE(LIBXSMM_REPEAT_, N)(A)

/*Based on Stackoverflow's NBITSx macro.*/
#define LIBXSMM_LOG2_02(N) (0 != ((N) & 0x2/*0b10*/) ? 1 : 0)
#define LIBXSMM_LOG2_04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 | LIBXSMM_LOG2_02((N) >> 2)) : LIBXSMM_LOG2_02(N))
#define LIBXSMM_LOG2_08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 | LIBXSMM_LOG2_04((N) >> 4)) : LIBXSMM_LOG2_04(N))
#define LIBXSMM_LOG2_16(N) (0 != ((N) & 0xFF00) ? (8 | LIBXSMM_LOG2_08((N) >> 8)) : LIBXSMM_LOG2_08(N))
#define LIBXSMM_LOG2_32(N) (0 != ((N) & 0xFFFF0000) ? (16 | LIBXSMM_LOG2_16((N) >> 16)) : LIBXSMM_LOG2_16(N))
#define LIBXSMM_LOG2_64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 | LIBXSMM_LOG2_32((N) >> 32)) : LIBXSMM_LOG2_32(N))
#define LIBXSMM_LOG2(N) LIBXSMM_MAX(LIBXSMM_LOG2_64((unsigned long long)(N)), 1)

#define LIBXSMM_DEFAULT(DEFAULT, VALUE) (0 < (VALUE) ? (VALUE) : (DEFAULT))
#define LIBXSMM_ABS(A) (0 <= (A) ? (A) : -(A))
#define LIBXSMM_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXSMM_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXSMM_CLMP(VALUE, LO, HI) ((LO) < (VALUE) ? ((HI) > (VALUE) ? (VALUE) : LIBXSMM_MAX(HI, VALUE)) : LIBXSMM_MIN(LO, VALUE))
#define LIBXSMM_MOD2(N, NPOT) ((N) & ((NPOT) - 1))
#define LIBXSMM_MUL2(N, NPOT) ((N) << LIBXSMM_LOG2(NPOT))
#define LIBXSMM_DIV2(N, NPOT) ((N) >> LIBXSMM_LOG2(NPOT))
#define LIBXSMM_SQRT2(N) (1 << (LIBXSMM_LOG2((N << 1) - 1) >> 1))
#define LIBXSMM_UP2(N, NPOT) LIBXSMM_MUL2(LIBXSMM_DIV2((N) + (NPOT) - 1, NPOT), NPOT)
#define LIBXSMM_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))
/* compares floating point values but avoids warning about unreliable comparison */
#define LIBXSMM_FEQ(A, B) (!((A) < (B) || (A) > (B)))

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXSMM_ATTRIBUTE(...) __declspec(__VA_ARGS__)
# if defined(__cplusplus)
#   define LIBXSMM_INLINE_ALWAYS __forceinline
# else
#   define LIBXSMM_INLINE_ALWAYS static __forceinline
# endif
# define LIBXSMM_ALIGNED(DECL, N) LIBXSMM_ATTRIBUTE(align(N)) DECL
# define LIBXSMM_CDECL __cdecl
#elif defined(__GNUC__)
# define LIBXSMM_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
# define LIBXSMM_INLINE_ALWAYS LIBXSMM_ATTRIBUTE(always_inline) LIBXSMM_INLINE
# define LIBXSMM_ALIGNED(DECL, N) DECL LIBXSMM_ATTRIBUTE(aligned(N))
# define LIBXSMM_CDECL LIBXSMM_ATTRIBUTE(cdecl)
#else
# define LIBXSMM_ATTRIBUTE(...)
# define LIBXSMM_INLINE_ALWAYS LIBXSMM_INLINE
# define LIBXSMM_ALIGNED(DECL, N)
# define LIBXSMM_CDECL
#endif

#if defined(__INTEL_COMPILER)
# define LIBXSMM_ASSUME_ALIGNED(A, N) __assume_aligned(A, N);
# define LIBXSMM_ASSUME(EXPRESSION) __assume(EXPRESSION);
#else
# define LIBXSMM_ASSUME_ALIGNED(A, N)
# if defined(_MSC_VER)
#   define LIBXSMM_ASSUME(EXPRESSION) __assume(EXPRESSION);
# elif (LIBXSMM_VERSION3(4, 5, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#   define LIBXSMM_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0);
# else
#   define LIBXSMM_ASSUME(EXPRESSION)
# endif
#endif
#define LIBXSMM_ALIGN_VALUE(N, TYPESIZE, ALIGNMENT/*POT*/) (LIBXSMM_UP2((N) * (TYPESIZE), ALIGNMENT) / (TYPESIZE))
#define LIBXSMM_ALIGN_VALUE2(N, POTSIZE, ALIGNMENT/*POT*/) LIBXSMM_DIV2(LIBXSMM_UP2(LIBXSMM_MUL2(N, POTSIZE), ALIGNMENT), POTSIZE)
#define LIBXSMM_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXSMM_ALIGN_VALUE((uintptr_t)(POINTER), 1, ALIGNMENT) - ((uintptr_t)(POINTER))) / sizeof(*(POINTER)))
#define LIBXSMM_ALIGN2(POINTPOT, ALIGNMENT/*POT*/) ((POINTPOT) + LIBXSMM_DIV2(LIBXSMM_ALIGN_VALUE2((uintptr_t)(POINTPOT), 1, ALIGNMENT) - ((uintptr_t)(POINTPOT)), sizeof(*(POINTPOT))))

#define LIBXSMM_HASH_VALUE(N) ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) ^ ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) >> 27))
#define LIBXSMM_HASH2(POINTER, ALIGNMENT/*POT*/, NPOT) LIBXSMM_MOD2(LIBXSMM_HASH_VALUE(LIBXSMM_DIV2((uintptr_t)(POINTER), ALIGNMENT)), NPOT)

#if defined(_MSC_VER) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXSMM_SELECT_ELEMENT(INDEX1/*one-based*/, .../*elements*/) LIBXSMM_CONCATENATE(LIBXSMM_SELECT_ELEMENT_, INDEX1)LIBXSMM_EXPAND((__VA_ARGS__))
#else
# define LIBXSMM_SELECT_ELEMENT(INDEX1/*one-based*/, .../*elements*/) LIBXSMM_CONCATENATE(LIBXSMM_SELECT_ELEMENT_, INDEX1)(__VA_ARGS__)
#endif
#define LIBXSMM_SELECT_ELEMENT_1(E0, E1, E2, E3, E4, E5, E6, E7) E0
#define LIBXSMM_SELECT_ELEMENT_2(E0, E1, E2, E3, E4, E5, E6, E7) E1
#define LIBXSMM_SELECT_ELEMENT_3(E0, E1, E2, E3, E4, E5, E6, E7) E2
#define LIBXSMM_SELECT_ELEMENT_4(E0, E1, E2, E3, E4, E5, E6, E7) E3
#define LIBXSMM_SELECT_ELEMENT_5(E0, E1, E2, E3, E4, E5, E6, E7) E4
#define LIBXSMM_SELECT_ELEMENT_6(E0, E1, E2, E3, E4, E5, E6, E7) E5
#define LIBXSMM_SELECT_ELEMENT_7(E0, E1, E2, E3, E4, E5, E6, E7) E6
#define LIBXSMM_SELECT_ELEMENT_8(E0, E1, E2, E3, E4, E5, E6, E7) E7

/**
 * For VLAs, check EXACTLY for C99 since a C11-conforming compiler may not provide VLAs.
 * However, some compilers (Intel) may signal support for VLA even with strict ANSI (C89).
 * To ultimately disable VLA-support, define LIBXSMM_NO_VLA (make VLA=0).
 * VLA-support is signaled by LIBXSMM_VLA.
 */
#if !defined(LIBXSMM_VLA) && !defined(LIBXSMM_NO_VLA) && ((defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || \
   (!defined(__STDC_NO_VLA__)&& 199901L/*C99*/ < __STDC_VERSION__))) || (defined(__INTEL_COMPILER) && !defined(_WIN32)) || \
    (defined(__GNUC__) && !defined(__STRICT_ANSI__) && !defined(__cplusplus))/*depends on above C99-check*/)
# define LIBXSMM_VLA
#endif

/**
 * LIBXSMM_INDEX1 calculates the linear address for a given set of (multiple) indexes/bounds.
 * Syntax: LIBXSMM_INDEX1(<ndims>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the leading dimension (s0) is omitted in the above syntax!
 * TODO: support leading dimension (pitch/stride).
 */
#if defined(_MSC_VER) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXSMM_INDEX1(NDIMS, ...) LIBXSMM_CONCATENATE(LIBXSMM_INDEX1_, NDIMS)LIBXSMM_EXPAND((__VA_ARGS__))
#else
# define LIBXSMM_INDEX1(NDIMS, ...) LIBXSMM_CONCATENATE(LIBXSMM_INDEX1_, NDIMS)(__VA_ARGS__)
#endif
#define LIBXSMM_INDEX1_1(I0) (1ULL * (I0))
#define LIBXSMM_INDEX1_2(I0, I1, S1) (LIBXSMM_INDEX1_1(I0) * (S1) + I1)
#define LIBXSMM_INDEX1_3(I0, I1, I2, S1, S2) (LIBXSMM_INDEX1_2(I0, I1, S1) * (S2) + (I2))
#define LIBXSMM_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) (LIBXSMM_INDEX1_3(I0, I1, I2, S1, S2) * (S3) + (I3))
#define LIBXSMM_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) (LIBXSMM_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) * (S4) + (I4))
#define LIBXSMM_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) (LIBXSMM_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) * (S5) + (I5))
#define LIBXSMM_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) (LIBXSMM_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) * (S6) + (I6))
#define LIBXSMM_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) (LIBXSMM_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) * (S7) + (I7))

 /**
 * LIBXSMM_VLA_DECL declares an array according to the given set of (multiple) bounds.
 * Syntax: LIBXSMM_VLA_DECL(<ndims>, <elem-type>, <var-name>, <init>, <s1>, ..., <s(ndims-1)>).
 * The element type can be "const" or otherwise qualified; initial value must be (const)element-type*.
 * Please note that the syntax is similar to LIBXSMM_INDEX1, and the leading dimension (s0) is omitted!
 *
 * LIBXSMM_VLA_ACCESS gives the array element according to the given set of (multiple) indexes/bounds.
 * Syntax: LIBXSMM_VLA_ACCESS(<ndims>, <array>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the syntax is similar to LIBXSMM_INDEX1, and the leading dimension (s0) is omitted!
 */
#if defined(LIBXSMM_VLA)
# define LIBXSMM_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXSMM_CONCATENATE(LIBXSMM_VLA_ACCESS_, NDIMS)(ARRAY, __VA_ARGS__)
# define LIBXSMM_VLA_ACCESS_0(ARRAY, ...) (ARRAY)
# define LIBXSMM_VLA_ACCESS_1(ARRAY, I0, ...) ((ARRAY)[I0])
# define LIBXSMM_VLA_ACCESS_2(ARRAY, I0, I1, ...) ((ARRAY)[I0][I1])
# define LIBXSMM_VLA_ACCESS_3(ARRAY, I0, I1, I2, ...) ((ARRAY)[I0][I1][I2])
# define LIBXSMM_VLA_ACCESS_4(ARRAY, I0, I1, I2, I3, ...) ((ARRAY)[I0][I1][I2][I3])
# define LIBXSMM_VLA_ACCESS_5(ARRAY, I0, I1, I2, I3, I4, ...) ((ARRAY)[I0][I1][I2][I3][I4])
# define LIBXSMM_VLA_ACCESS_6(ARRAY, I0, I1, I2, I3, I4, I5, ...) ((ARRAY)[I0][I1][I2][I3][I4][I5])
# define LIBXSMM_VLA_ACCESS_7(ARRAY, I0, I1, I2, I3, I4, I5, I6, ...) ((ARRAY)[I0][I1][I2][I3][I4][I5][I6])
# define LIBXSMM_VLA_ACCESS_8(ARRAY, I0, I1, I2, I3, I4, I5, I6, I7, ...) ((ARRAY)[I0][I1][I2][I3][I4][I5][I6][I7])
# define LIBXSMM_VLA_DECL(NDIMS, ELEMENT_TYPE, VARIABLE_NAME, INIT_VALUE, .../*bounds*/) \
    ELEMENT_TYPE LIBXSMM_VLA_ACCESS(LIBXSMM_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7), *LIBXSMM_RESTRICT VARIABLE_NAME, __VA_ARGS__/*bounds*/, __VA_ARGS__/*dummy*/) = \
   (ELEMENT_TYPE LIBXSMM_VLA_ACCESS(LIBXSMM_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7), *, __VA_ARGS__/*bounds*/, __VA_ARGS__/*dummy*/))(INIT_VALUE)
#else /* calculate linear index */
# define LIBXSMM_VLA_ACCESS(NDIMS, ARRAY, ...) ((ARRAY)[LIBXSMM_INDEX1(NDIMS, __VA_ARGS__)])
# define LIBXSMM_VLA_DECL(NDIMS, ELEMENT_TYPE, VARIABLE_NAME, INIT_VALUE, .../*bounds*/) \
    ELEMENT_TYPE *LIBXSMM_RESTRICT VARIABLE_NAME = /*(ELEMENT_TYPE*)*/(INIT_VALUE)
#endif

#if !defined(LIBXSMM_UNUSED)
# if 0 /*defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)*/
#   define LIBXSMM_UNUSED(VARIABLE) LIBXSMM_PRAGMA(unused(VARIABLE))
# else
#   define LIBXSMM_UNUSED(VARIABLE) (void)(VARIABLE)
# endif
#endif

#if defined(__GNUC__) || (defined(__INTEL_COMPILER) && !defined(_WIN32))
# define LIBXSMM_UNUSED_ARG LIBXSMM_ATTRIBUTE(unused)
#else
# define LIBXSMM_UNUSED_ARG
#endif

#if defined(__GNUC__) && defined(LIBXSMM_BUILD)
# define LIBXSMM_VISIBILITY_HIDDEN LIBXSMM_ATTRIBUTE(visibility("hidden"))
# define LIBXSMM_VISIBILITY_INTERNAL LIBXSMM_ATTRIBUTE(visibility("internal"))
#else
# define LIBXSMM_VISIBILITY_HIDDEN
# define LIBXSMM_VISIBILITY_INTERNAL
#endif

#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CYGWIN__)
# define LIBXSMM_ATTRIBUTE_WEAK_IMPORT LIBXSMM_ATTRIBUTE(weak_import)
# define LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE(weak)
#else
# define LIBXSMM_ATTRIBUTE_WEAK
# define LIBXSMM_ATTRIBUTE_WEAK_IMPORT
#endif

#if defined(__GNUC__)
# define LIBXSMM_MAY_ALIAS LIBXSMM_ATTRIBUTE(__may_alias__)
#else
# define LIBXSMM_MAY_ALIAS
#endif

#if defined(NDEBUG)
# define LIBXSMM_NDEBUG NDEBUG
# define LIBXSMM_DEBUG(...)
#else
# define LIBXSMM_DEBUG(...) __VA_ARGS__
#endif

#if defined(_WIN32)
# define LIBXSMM_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
# define LIBXSMM_FLOCK(FILE) _lock_file(FILE)
# define LIBXSMM_FUNLOCK(FILE) _unlock_file(FILE)
#else
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__)
#   define LIBXSMM_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
# else
#   define LIBXSMM_SNPRINTF(S, N, ...) sprintf(S, __VA_ARGS__); LIBXSMM_UNUSED(N)
# endif
# if !defined(__CYGWIN__)
#   define LIBXSMM_FLOCK(FILE) flockfile(FILE)
#   define LIBXSMM_FUNLOCK(FILE) funlockfile(FILE)
# else /* Only available with __CYGWIN__ *and* C++0x. */
#   define LIBXSMM_FLOCK(FILE)
#   define LIBXSMM_FUNLOCK(FILE)
# endif
#endif

/** Below group is to fixup some platform/compiler specifics. */
#if defined(_WIN32)
# if !defined(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES)
#   define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
# endif
# if !defined(_CRT_SECURE_NO_DEPRECATE)
#   define _CRT_SECURE_NO_DEPRECATE 1
# endif
# if !defined(_USE_MATH_DEFINES)
#   define _USE_MATH_DEFINES 1
# endif
# if !defined(WIN32_LEAN_AND_MEAN)
#   define WIN32_LEAN_AND_MEAN 1
# endif
# if !defined(NOMINMAX)
#   define NOMINMAX 1
# endif
# if defined(__INTEL_COMPILER) && (190023506 <= _MSC_FULL_VER)
#   define __builtin_huge_val() HUGE_VAL
#   define __builtin_huge_valf() HUGE_VALF
#   define __builtin_nan nan
#   define __builtin_nanf nanf
#   define __builtin_nans nan
#   define __builtin_nansf nanf
# endif
#endif
#if defined(__GNUC__)
# if !defined(_GNU_SOURCE)
#   define _GNU_SOURCE
# endif
#endif
#if defined(__clang__) && !defined(__extern_always_inline)
# define __extern_always_inline LIBXSMM_INLINE
#endif
#if defined(LIBXSMM_INLINE_FIXUP) && !defined(inline)
# define inline LIBXSMM_INLINE_KEYWORD
#endif

#if defined(LIBXSMM_OFFLOAD_BUILD) && \
  defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= __INTEL_COMPILER))
# define LIBXSMM_OFFLOAD(A) LIBXSMM_ATTRIBUTE(target(A))
# define LIBXSMM_NO_OFFLOAD(RTYPE, FN, ...) ((RTYPE (*)(LIBXSMM_VARIADIC))(FN))(__VA_ARGS__)
# if !defined(LIBXSMM_OFFLOAD_TARGET)
#   define LIBXSMM_OFFLOAD_TARGET mic
# endif
#else
# define LIBXSMM_OFFLOAD(A)
# define LIBXSMM_NO_OFFLOAD(RTYPE, FN, ...) (FN)(__VA_ARGS__)
#endif
#define LIBXSMM_RETARGETABLE LIBXSMM_OFFLOAD(LIBXSMM_OFFLOAD_TARGET)

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXSMM_MACROS_H*/

