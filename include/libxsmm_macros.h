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

#define LIBXSMM_STRINGIFY2(SYMBOL) #SYMBOL
#define LIBXSMM_STRINGIFY(SYMBOL) LIBXSMM_STRINGIFY2(SYMBOL)
#define LIBXSMM_TOSTRING(SYMBOL) LIBXSMM_STRINGIFY(SYMBOL)
#define LIBXSMM_CONCATENATE2(A, B) A##B
#define LIBXSMM_CONCATENATE(A, B) LIBXSMM_CONCATENATE2(A, B)
#define LIBXSMM_FSYMBOL(SYMBOL) LIBXSMM_CONCATENATE2(SYMBOL, _)
#define LIBXSMM_UNIQUE(NAME) LIBXSMM_CONCATENATE(NAME, __LINE__)

#define LIBXSMM_VERSION3(MAJOR, MINOR, UPDATE) ((MAJOR) * 10000 + (MINOR) * 100 + (UPDATE))
#define LIBXSMM_VERSION4(MAJOR, MINOR, UPDATE, PATCH) ((MAJOR) * 100000000 + (MINOR) * 1000000 + (UPDATE) * 10000 + (PATCH))

#if defined(__cplusplus)
# define LIBXSMM_VARIADIC ...
# define LIBXSMM_EXTERN extern "C"
# define LIBXSMM_INLINE_KEYWORD inline
# define LIBXSMM_INLINE LIBXSMM_INLINE_KEYWORD
#else
# define LIBXSMM_VARIADIC
# define LIBXSMM_EXTERN extern
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define LIBXSMM_PRAGMA(DIRECTIVE) _Pragma(LIBXSMM_STRINGIFY(DIRECTIVE))
#   define LIBXSMM_RESTRICT restrict
#   define LIBXSMM_INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define LIBXSMM_INLINE_KEYWORD __inline
# endif
# if !defined(LIBXSMM_INLINE_KEYWORD)
#   define LIBXSMM_INLINE_KEYWORD
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

/* For VLAs, check EXACTLY for C99 since a C11-conformant compiler may not provide VLAs */
#if !defined(LIBXSMM_VLA) && ((defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || \
   (!defined(__STDC_NO_VLA__)&& 199901L/*C99*/ < __STDC_VERSION__))) || \
     defined(__INTEL_COMPILER))
# define LIBXSMM_VLA
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
#define LIBXSMM_LOG2_02(N) (0 != ((N) & 2/*0b10*/) ? 1 : 0)
#define LIBXSMM_LOG2_04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 + LIBXSMM_LOG2_02((N) >> 2)) : LIBXSMM_LOG2_02(N))
#define LIBXSMM_LOG2_08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 + LIBXSMM_LOG2_04((N) >> 4)) : LIBXSMM_LOG2_04(N))
#define LIBXSMM_LOG2_16(N) (0 != ((N) & 0xFF00) ? (8 + LIBXSMM_LOG2_08((N) >> 8)) : LIBXSMM_LOG2_08(N))
#define LIBXSMM_LOG2_32(N) (0 != ((N) & 0xFFFF0000) ? (16 + LIBXSMM_LOG2_16((N) >> 16)) : LIBXSMM_LOG2_16(N))
#define LIBXSMM_LOG2_64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 + LIBXSMM_LOG2_32((N) >> 32)) : LIBXSMM_LOG2_32(N))
#define LIBXSMM_LOG2(N) LIBXSMM_MAX(LIBXSMM_LOG2_64((unsigned long long)(N)), 1)

#define LIBXSMM_DEFAULT(DEFAULT, VALUE) (0 < (VALUE) ? (VALUE) : (DEFAULT))
#define LIBXSMM_ABS(A) (0 <= (A) ? (A) : -(A))
#define LIBXSMM_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXSMM_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXSMM_MOD2(N, NPOT) ((N) & ((NPOT) - 1))
#define LIBXSMM_MUL2(N, NPOT) ((N) << LIBXSMM_LOG2(NPOT))
#define LIBXSMM_DIV2(N, NPOT) ((N) >> LIBXSMM_LOG2(NPOT))
#define LIBXSMM_UP2(N, NPOT) LIBXSMM_MUL2(LIBXSMM_DIV2((N) + (NPOT) - 1, NPOT), NPOT)
#define LIBXSMM_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))

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
#define LIBXSMM_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXSMM_ALIGN_VALUE((unsigned long long)(POINTER), 1, ALIGNMENT) - ((unsigned long long)(POINTER))) / sizeof(*(POINTER)))
#define LIBXSMM_ALIGN2(POINTPOT, ALIGNMENT/*POT*/) ((POINTPOT) + LIBXSMM_DIV2(LIBXSMM_ALIGN_VALUE2((unsigned long long)(POINTPOT), 1, ALIGNMENT) - ((unsigned long long)(POINTPOT)), sizeof(*(POINTPOT))))

#define LIBXSMM_HASH_VALUE(N) ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) ^ ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) >> 27))
#define LIBXSMM_HASH2(POINTER, ALIGNMENT/*POT*/, NPOT) LIBXSMM_MOD2(LIBXSMM_HASH_VALUE(LIBXSMM_DIV2((unsigned long long)(POINTER), ALIGNMENT)), NPOT)

#define LIBXSMM_CALC_SIZE1(TYPE, VARIABLE, NDIMS, SHAPE, INIT) { \
  unsigned int libxsmm_calc_size1_i_ = 0; \
  VARIABLE = LIBXSMM_MAX(INIT, 1); \
  LIBXSMM_REPEAT(NDIMS, \
    VARIABLE *= (TYPE)((SHAPE)[libxsmm_calc_size1_i_]); \
    ++libxsmm_calc_size1_i_;) \
}
/* TODO: LIBXSMM_CALC_INDEX1 plus PITCH */
#define LIBXSMM_CALC_INDEX1(TYPE, VARIABLE, NDIMS, INDEXN, SHAPE) { \
  unsigned int libxsmm_calc_index1_i_ = 0; \
  TYPE libxsmm_calc_index1_size_ = 1; \
  VARIABLE = 0; \
  LIBXSMM_REPEAT(NDIMS, \
    VARIABLE += libxsmm_calc_index1_size_ * ((TYPE)(INDEXN)[libxsmm_calc_index1_i_]); \
    libxsmm_calc_index1_size_ *= (TYPE)((SHAPE)[libxsmm_calc_index1_i_]); \
    ++libxsmm_calc_index1_i_;) \
}

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

#if (defined(__GNUC__) || defined(__clang__))
# define LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE(weak)
# define LIBXSMM_ATTRIBUTE_WEAK_IMPORT LIBXSMM_ATTRIBUTE(weak_import)
#else
# define LIBXSMM_ATTRIBUTE_WEAK
# define LIBXSMM_ATTRIBUTE_WEAK_IMPORT
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
#if defined(__clang__)
# if !defined(__extern_always_inline)
#   define __extern_always_inline LIBXSMM_INLINE_KEYWORD
# endif
# if !defined(__cplusplus) && (199901L > __STDC_VERSION__)
#   define inline LIBXSMM_INLINE_KEYWORD
# endif
#endif

#endif /*LIBXSMM_MACROS_H*/

