/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_main.h"
#include <sys/types.h>
#include <sys/stat.h>

#if defined(LIBXSMM_DEFAULT_CONFIG) || (defined(LIBXSMM_SOURCE_H) && !defined(LIBXSMM_CONFIGURED))
# if !defined(LIBXSMM_MATHDIFF_MHD)
#   include <utils/libxsmm_mhd.h>
#   define LIBXSMM_MATHDIFF_MHD
# endif
#endif
#if !defined(LIBXSMM_MATH_DELIMS)
# define LIBXSMM_MATH_DELIMS " \t;,:"
#endif
#if !defined(LIBXSMM_MATH_ISDIR)
# if defined(S_IFDIR)
#   define LIBXSMM_MATH_ISDIR(MODE) 0 != ((MODE) & (S_IFDIR))
# else
#   define LIBXSMM_MATH_ISDIR(MODE) S_ISDIR(MODE)
# endif
#endif

/**
 * LIBXSMM_MATDIFF_DIV devises the nominator by the reference-denominator
 * unless the latter is zero in which case the fallback is returned.
 */
#define LIBXSMM_MATDIFF_DIV_DEN(A) (0 < (A) ? (A) : 1)   /* Clang: WA for div-by-zero */
#define LIBXSMM_MATDIFF_DIV(NOMINATOR, DENREF, FALLBACK) /* Clang: >= instead of < */ \
  (0 >= (DENREF) ? (FALLBACK) : ((NOMINATOR) / LIBXSMM_MATDIFF_DIV_DEN(DENREF)))


LIBXSMM_API int libxsmm_matdiff(libxsmm_matdiff_info* info,
  libxsmm_datatype datatype, libxsmm_blasint m, libxsmm_blasint n, const void* ref, const void* tst,
  const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst)
{
  int result = EXIT_SUCCESS, result_swap = 0, result_nan = 0;
  libxsmm_blasint ldr = (NULL == ldref ? m : *ldref), ldt = (NULL == ldtst ? m : *ldtst);
  if (NULL == ref && NULL != tst) { ref = tst; tst = NULL; result_swap = 1; }
  if (NULL != ref && NULL != info && m <= ldr && m <= ldt) {
    const char *const matdiff_shuffle_env = getenv("LIBXSMM_MATDIFF_SHUFFLE");
    const int matdiff_shuffle = (NULL == matdiff_shuffle_env ? 0
      : ('\0' != *matdiff_shuffle_env ? atoi(matdiff_shuffle_env) : 1));
    const size_t ntotal = (size_t)m * n;
    libxsmm_blasint mm = m, nn = n;
    double inf;
    if (1 == n) { mm = ldr = ldt = 1; nn = m; } /* ensure row-vector shape to standardize results */
    libxsmm_matdiff_clear(info);
    inf = info->min_ref;
    switch ((int)datatype) {
      case LIBXSMM_DATATYPE_I64: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE long long
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_I32: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE int
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_U32: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE unsigned int
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_I16: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE short
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_U16: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE unsigned short
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_I8: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE signed char
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_F64: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE double
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_F32: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE float
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_F16: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) libxsmm_convert_f16_to_f32(VALUE)
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE libxsmm_float16
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_BF16: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) libxsmm_convert_bf16_to_f32(VALUE)
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE libxsmm_bfloat16
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_BF8: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) libxsmm_convert_bf8_to_f32(VALUE)
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE libxsmm_bfloat8
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXSMM_DATATYPE_HF8: {
#       define LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) libxsmm_convert_hf8_to_f32(VALUE)
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE libxsmm_hfloat8
        if (0 == matdiff_shuffle) {
#         include "libxsmm_matdiff.h"
        }
        else {
#         define LIBXSMM_MATDIFF_SHUFFLE
#         include "libxsmm_matdiff.h"
#         undef LIBXSMM_MATDIFF_SHUFFLE
        }
#       undef LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXSMM_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      default: {
        static int error_once = 0;
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
        }
        result = EXIT_FAILURE;
      }
    }
    LIBXSMM_ASSERT((0 <= info->m && 0 <= info->n) || (0 > info->m && 0 > info->n));
    LIBXSMM_ASSERT(info->m < mm && info->n < nn);
    if (EXIT_SUCCESS == result) {
      const char *const env = getenv("LIBXSMM_DUMP");
      LIBXSMM_INIT
      if (NULL != env && 0 != *env && '0' != *env) {
        if ('-' != *env || (0 <= info->m && 0 <= info->n)) {
#if defined(LIBXSMM_MATHDIFF_MHD)
          const char *const defaultname = ((('0' < *env && '9' >= *env) || '-' == *env) ? "libxsmm_dump" : env);
          const libxsmm_mhd_elemtype type_src = (libxsmm_mhd_elemtype)datatype;
          const libxsmm_mhd_elemtype type_dst = LIBXSMM_MIN(LIBXSMM_MHD_ELEMTYPE_F32, type_src);
          char filename[256] = "";
          const int envi = atoi(env), reshape = (1 < envi || -1 > envi);
          size_t shape[2] = { 0 }, size[2] = { 0 };
          if (0 == reshape) {
            shape[0] = (size_t)mm; shape[1] = (size_t)nn;
            size[0] = (size_t)ldr; size[1] = (size_t)nn;
          }
          else { /* reshape */
            const size_t y = (size_t)libxsmm_isqrt2_u32((unsigned int)ntotal);
            shape[0] = ntotal / y; shape[1] = y;
            size[0] = shape[0];
            size[1] = shape[1];
          }
          LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s-%p-ref.mhd", defaultname, ref);
          libxsmm_mhd_write(filename, NULL/*offset*/, shape, size, 2/*ndims*/, 1/*ncomponents*/,
            type_src, &type_dst, ref, NULL/*header_size*/, NULL/*extension_header*/,
            NULL/*extension*/, 0/*extension_size*/);
#endif
          if (NULL != tst) {
#if defined(LIBXSMM_MATHDIFF_MHD)
            if (0 == reshape) {
              size[0] = (size_t)ldt;
              size[1] = (size_t)nn;
            }
            LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s-%p-tst.mhd", defaultname, ref/*adopt ref-ptr*/);
            libxsmm_mhd_write(filename, NULL/*offset*/, shape, size, 2/*ndims*/, 1/*ncomponents*/,
              type_src, &type_dst, tst, NULL/*header_size*/, NULL/*extension_header*/,
              NULL/*extension*/, 0/*extension_size*/);
#endif
            if ('-' == *env && '1' < env[1]) {
              printf("LIBXSMM MATDIFF (%s): m=%" PRIuPTR " n=%" PRIuPTR " ldi=%" PRIuPTR " ldo=%" PRIuPTR " failed.\n",
                libxsmm_get_typename(datatype), (uintptr_t)m, (uintptr_t)n, (uintptr_t)ldr, (uintptr_t)ldt);
            }
          }
        }
        else if ('-' == *env && '1' < env[1] && NULL != tst) {
          printf("LIBXSMM MATDIFF (%s): m=%" PRIuPTR " n=%" PRIuPTR " ldi=%" PRIuPTR " ldo=%" PRIuPTR " passed.\n",
            libxsmm_get_typename(datatype), (uintptr_t)m, (uintptr_t)n, (uintptr_t)ldr, (uintptr_t)ldt);
        }
      }
      if (0 == result_nan) {
        const double resrel = LIBXSMM_MATDIFF_DIV(info->l2_abs, info->var_ref, info->l2_abs);
        info->rsq = LIBXSMM_MAX(0.0, 1.0 - resrel);
        if (0 != ntotal) { /* final variance */
          info->var_ref /= ntotal;
          info->var_tst /= ntotal;
        }
        info->normf_rel = libxsmm_dsqrt(info->normf_rel);
        info->l2_abs = libxsmm_dsqrt(info->l2_abs);
        info->l2_rel = libxsmm_dsqrt(info->l2_rel);
      }
      else if (0 != result_nan) {
        /* in case of NaN (in test-set), initialize statistics to either Infinity or NaN */
        info->norm1_abs = info->norm1_rel = info->normi_abs = info->normi_rel = info->normf_rel
                        = info->linf_abs = info->linf_rel = info->l2_abs = info->l2_rel
                        = inf;
        if (1 == result_nan) {
          info->l1_tst = info->var_tst = inf;
          info->avg_tst = /*NaN*/info->v_tst;
          info->min_tst = +inf;
          info->max_tst = -inf;
        }
        else {
          info->l1_ref = info->var_ref = inf;
          info->avg_ref = /*NaN*/info->v_ref;
          info->min_ref = +inf;
          info->max_ref = -inf;
        }
      }
      if (1 == n) LIBXSMM_ISWAP(info->m, info->n);
      if (0 != result_swap) {
        info->min_tst = info->min_ref;
        info->min_ref = 0;
        info->max_tst = info->max_ref;
        info->max_ref = 0;
        info->avg_tst = info->avg_ref;
        info->avg_ref = 0;
        info->var_tst = info->var_ref;
        info->var_ref = 0;
        info->l1_tst = info->l1_ref;
        info->l1_ref = 0;
        info->v_tst = info->v_ref;
        info->v_ref = 0;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API double libxsmm_matdiff_epsilon(const libxsmm_matdiff_info* input)
{
  double result;
  if (NULL != input) {
    const char *const matdiff_env = getenv("LIBXSMM_MATDIFF");
    if (0 < input->rsq) {
      result = LIBXSMM_MIN(input->normf_rel, input->linf_abs) / input->rsq;
    }
    else {
      const double a = LIBXSMM_MIN(input->norm1_abs, input->normi_abs);
      const double b = LIBXSMM_MAX(input->linf_abs, input->l2_abs);
      result = LIBXSMM_MAX(a, b);
    }
    if (NULL != matdiff_env && '\0' != *matdiff_env) {
      char buffer[4096];
      struct stat stat_info;
      size_t offset = strlen(matdiff_env) + 1;
      char *const env = strncpy(buffer, matdiff_env, sizeof(buffer) - 1);
      const char *arg = strtok(env, LIBXSMM_MATH_DELIMS), *filename = NULL;
      if (0 == stat(arg, &stat_info) && LIBXSMM_MATH_ISDIR(stat_info.st_mode)) {
        const int nchars = LIBXSMM_SNPRINTF(buffer + offset, sizeof(buffer) - offset,
          "%s/libxsmm_matdiff.log", arg);
        if (0 < nchars && (offset + nchars + 1) < sizeof(buffer)) {
          filename = buffer + offset;
          offset += nchars + 1;
        }
      }
      else filename = arg; /* assume file */
      if (NULL != filename) { /* bufferize output before file I/O */
        const size_t begin = offset;
        int nchars = ((2 * offset) < sizeof(buffer)
          ? LIBXSMM_SNPRINTF(buffer + offset, sizeof(buffer) - offset, "%.17g", result)
          : 0);
        if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) {
          offset += nchars;
          arg = strtok(NULL, LIBXSMM_MATH_DELIMS);
          while (NULL != arg) {
            nchars = LIBXSMM_SNPRINTF(buffer + offset, sizeof(buffer) - offset, " %s", arg);
            if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) offset += nchars;
            else break;
            arg = strtok(NULL, LIBXSMM_MATH_DELIMS);
          }
          if (NULL == arg) { /* all args consumed */
            nchars = libxsmm_print_cmdline(buffer + offset, sizeof(buffer) - offset, " [", "]");
            if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) {
              FILE *const file = fopen(filename, "a");
              if (NULL != file) {
                buffer[offset + nchars] = '\n'; /* replace terminator */
                fwrite(buffer + begin, 1, offset + nchars - begin + 1, file);
                fclose(file);
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE) || \
   (defined(_XOPEN_SOURCE) && (500 <= _XOPEN_SOURCE))
                sync(); /* attempt to flush FS */
#endif
              }

            }
          }
        }
      }
    }
  }
  else result = 0;
  return result;
}


LIBXSMM_API void libxsmm_matdiff_reduce(libxsmm_matdiff_info* output, const libxsmm_matdiff_info* input)
{
  if (NULL != output && NULL != input) {
    /* epsilon is determined before updating the output */
    const double epsinp = libxsmm_matdiff_epsilon(input);
    const double epsout = libxsmm_matdiff_epsilon(output);
    if (output->linf_abs <= input->linf_abs) {
      output->linf_abs = input->linf_abs;
      output->linf_rel = input->linf_rel;
    }
    if (output->norm1_abs <= input->norm1_abs) {
      output->norm1_abs = input->norm1_abs;
      output->norm1_rel = input->norm1_rel;
    }
    if (output->normi_abs <= input->normi_abs) {
      output->normi_abs = input->normi_abs;
      output->normi_rel = input->normi_rel;
    }
    if (output->l2_abs <= input->l2_abs) {
      output->l2_abs = input->l2_abs;
      output->l2_rel = input->l2_rel;
    }
    if (output->normf_rel <= input->normf_rel) {
      output->normf_rel = input->normf_rel;
    }
    if (output->var_ref <= input->var_ref) {
      output->var_ref = input->var_ref;
    }
    if (output->var_tst <= input->var_tst) {
      output->var_tst = input->var_tst;
    }
    if (output->max_ref <= input->max_ref) {
      output->max_ref = input->max_ref;
    }
    if (output->max_tst <= input->max_tst) {
      output->max_tst = input->max_tst;
    }
    if (output->min_ref >= input->min_ref) {
      output->min_ref = input->min_ref;
    }
    if (output->min_tst >= input->min_tst) {
      output->min_tst = input->min_tst;
    }
    if (epsout < epsinp) {
      output->rsq = input->rsq;
      output->v_ref = input->v_ref;
      output->v_tst = input->v_tst;
      output->m = input->m;
      output->n = input->n;
      output->i = input->r;
    }
    output->avg_ref = 0.5 * (output->avg_ref + input->avg_ref);
    output->avg_tst = 0.5 * (output->avg_tst + input->avg_tst);
    output->l1_ref += input->l1_ref;
    output->l1_tst += input->l1_tst;
    ++output->r;
  }
  else {
    libxsmm_matdiff_clear(output);
  }
}


LIBXSMM_API void libxsmm_matdiff_clear(libxsmm_matdiff_info* info)
{
  if (NULL != info) {
    union { int raw; float value; } inf = { 0 };
#if defined(INFINITY) && /*overflow warning*/!defined(_CRAYC)
    inf.value = (float)(INFINITY);
#else
    inf.raw = 0x7F800000;
#endif
    memset(info, 0, sizeof(*info)); /* nullify */
    /* no location discovered yet with a difference */
    info->m = info->n = info->i = -1;
    /* initial minimum/maximum of reference/test */
    info->min_ref = info->min_tst = +inf.value;
    info->max_ref = info->max_tst = -inf.value;
    /* invalid rather than 1.0 */
    info->rsq = inf.value;
  }
}


LIBXSMM_API size_t libxsmm_coprime(size_t n, size_t minco)
{
  const size_t s = (0 != (n & 1) ? ((LIBXSMM_MAX(minco, 1) - 1) | 1) : (minco & ~1));
  const size_t d = (0 != (n & 1) ? 1 : 2);
  size_t result = (1 < n ? 1 : 0), i;
  for (i = (d < n ? (n - 1) : 0); d < i; i -= d) {
    const size_t c = LIBXSMM_DELTA(s, i);
    size_t a = n, b = c;
    assert(i != s);
    do { /* GCD of initial A and initial B (result is in A) */
      const size_t r = a % b;
      a = b; b = r;
    } while (0 != b);
    if (1 == a) {
      result = c;
      if (c <= minco) i = d; /* break */
    }
  }
  if (minco < result) result = 1;
  assert((0 == result && 1 >= n) || (result < n && 1 == libxsmm_gcd(result, n)));
  assert(0 == minco || (result <= minco));
  return result;
}


LIBXSMM_API size_t libxsmm_coprime2(size_t n)
{
  return libxsmm_coprime(n, n / 2);
}


LIBXSMM_API unsigned int libxsmm_isqrt_u64(unsigned long long x)
{
  unsigned long long b; unsigned int y = 0, s;
  for (s = 0x80000000/*2^31*/; 0 < s; s >>= 1) {
    b = y | s; y |= (b * b <= x ? s : 0);
  }
  return y;
}


LIBXSMM_API unsigned int libxsmm_isqrt_u32(unsigned int x)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 0x40000000/*2^30*/; 0 < s; s >>= 2) {
    b = y | s; y >>= 1;
    if (b <= x) { x -= b; y |= s; }
  }
  return y;
}


LIBXSMM_API unsigned int libxsmm_isqrt2_u32(unsigned int x)
{
  return libxsmm_product_limit(x, libxsmm_isqrt_u32(x), 0/*is_lower*/);
}


LIBXSMM_API double libxsmm_kahan_sum(double value, double* accumulator, double* compensation)
{
  double r, c;
  LIBXSMM_ASSERT(NULL != accumulator && NULL != compensation);
  c = value - *compensation; r = *accumulator + c;
  *compensation = (r - *accumulator) - c;
  *accumulator = r;
  return r;
}


LIBXSMM_API float libxsmm_convert_bf8_to_f32(libxsmm_bfloat8 in)
{
  const unsigned short inus = (unsigned short)in;
  const unsigned short tmp = (unsigned short)(inus << 8);
  return libxsmm_convert_f16_to_f32(tmp);
}


LIBXSMM_API float libxsmm_convert_hf8_to_f32(libxsmm_hfloat8 in)
{
  const unsigned int f32_bias = 127, f8_bias = 7;
  const unsigned int s = (in & 0x80 ) << 24;
  const unsigned int e = (in & 0x78 ) >> 3;
  unsigned int m = (in & 0x07 );
  unsigned int e_norm = e + (f32_bias - f8_bias);
  libxsmm_float_uint res;
  /* convert denormal fp8 number into a normal fp32 number */
  if ( (e == 0) && (m != 0) ) {
    unsigned int lz_cnt = 2;
    lz_cnt = (m > 0x1 ? 1 : lz_cnt);
    lz_cnt = (m > 0x3 ? 0 : lz_cnt);
    LIBXSMM_ASSERT(e_norm >= lz_cnt);
    e_norm -= lz_cnt;
    m = (m << (lz_cnt+1)) & 0x07;
  } else if (e == 0 && m == 0) {
    e_norm = 0;
  } else if (e == 0xf && m == 0x7) {
    e_norm = 0xff;
    m = 0x4; /* making first mantissa bit 1 */
  }
  /* set result to 0 */
  res.u = 0x0;
  /* set exponent and mantissa */
  res.u |= (e_norm << 23);
  res.u |= (m << 20);
  /* sign it */
  res.u |= s;
  return res.f;
}


LIBXSMM_API float libxsmm_convert_bf16_to_f32(libxsmm_bfloat16 in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  hybrid_in.u = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f80) == 0x0
    ? (unsigned short)(hybrid_in.u & 0x8000)
    : hybrid_in.u);
  hybrid_in.u = hybrid_in.u << 16;
  return hybrid_in.f;
}


LIBXSMM_API float libxsmm_convert_f16_to_f32(libxsmm_float16 in)
{
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  unsigned int s = (in & 0x8000) << 16;
  unsigned int e = (in & 0x7c00) >> 10;
  unsigned int m = (in & 0x03ff);
  unsigned int e_norm = e + (f32_bias - f16_bias);
  libxsmm_float_uint res = { 0 };

  /* convert denormal fp16 number into a normal fp32 number */
  if ((e == 0) && (m != 0)) {
    unsigned int lz_cnt = 9;
    lz_cnt = (m > 0x1 ? 8 : lz_cnt);
    lz_cnt = (m > 0x3 ? 7 : lz_cnt);
    lz_cnt = (m > 0x7 ? 6 : lz_cnt);
    lz_cnt = (m > 0xf ? 5 : lz_cnt);
    lz_cnt = (m > 0x1f ? 4 : lz_cnt);
    lz_cnt = (m > 0x3f ? 3 : lz_cnt);
    lz_cnt = (m > 0x7f ? 2 : lz_cnt);
    lz_cnt = (m > 0xff ? 1 : lz_cnt);
    lz_cnt = (m > 0x1ff ? 0 : lz_cnt);
    LIBXSMM_ASSERT(e_norm >= lz_cnt);
    e_norm -= lz_cnt;
    m = (m << (lz_cnt + 1)) & 0x03ff;
  }
  else if ((e == 0) && (m == 0)) {
    e_norm = 0;
  }
  else if (e == 0x1f) {
    e_norm = 0xff;
    m |= (m == 0 ? 0 : 0x0200); /* making first mantissa bit 1 */
  }

  /* set result to 0 */
  res.u = 0x0;
  /* set exponent and mantissa */
  res.u |= (e_norm << 23);
  res.u |= (m << 13);
  /* sign it */
  res.u |= s;

  return res.f;
}


LIBXSMM_API libxsmm_bfloat16 libxsmm_convert_f32_to_bf16_truncate(float in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_bfloat16 res;
  hybrid_in.f = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : hybrid_in.u);
  /* we do not round inf and NaN */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x7f800000
    ? ((hybrid_in.u & 0x007fffff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x00400000))
    : hybrid_in.u);
  /* shift right */
  res = (unsigned short)(hybrid_in.u >> 16);
  return res;
}


LIBXSMM_API libxsmm_bfloat16 libxsmm_convert_f32_to_bf16_rnaz(float in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_bfloat16 res;
  hybrid_in.f = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : hybrid_in.u);
  /* we do not round inf and NaN */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x7f800000
    ? ((hybrid_in.u & 0x007fffff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x00400000))
    : (hybrid_in.u + 0x00008000));
  /* shift right */
  res = (unsigned short)(hybrid_in.u >> 16);
  return res;
}


LIBXSMM_API libxsmm_bfloat16 libxsmm_convert_f32_to_bf16_rne(float in)
{
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_bfloat16 res;
  unsigned int fixup;
  hybrid_in.f = in;
  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : hybrid_in.u);
  /* RNE round */
  fixup = (hybrid_in.u >> 16) & 1;
  /* we do not round inf and NaN */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x7f800000
    ? ((hybrid_in.u & 0x007fffff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x00400000))
    : (hybrid_in.u + 0x00007fff + fixup));
  /* shift right */
  res = (unsigned short)(hybrid_in.u >> 16);
  return res;
}


LIBXSMM_API libxsmm_bfloat8 libxsmm_convert_f32_to_bf8_stochastic(float in, unsigned int seed)
{
  /* initial downcast */
  libxsmm_float16 f16 = libxsmm_convert_f32_to_f16(in);
  /* do not round inf and NaN */
  if ((f16 & 0x7c00) == 0x7c00) {
    f16 = (unsigned short)(((f16 & 0x03ff) == 0x0) ? f16 : (f16 | 0x0200));
  }
  else if ((f16 & 0x7c00) != 0x0000) { /* only round normal numbers */
#if 1
    const unsigned short stochastic = (unsigned short)LIBXSMM_MOD2(seed, 0xff + 1);
#else
    const unsigned short stochastic = (unsigned short)((seed >> 24) & 0xff);
#endif
    f16 = (unsigned short)(f16 + stochastic);
  }
  else { /* RNE for subnormal */
    const unsigned short fixup = (unsigned short)((f16 >> 8) & 1);
    f16 = (unsigned short)(f16 + 0x007f + fixup);
  }
  /* create the bf8 value by shifting out the lower 8 bits */
  return (unsigned char)(f16 >> 8);
}


LIBXSMM_API libxsmm_bfloat8 libxsmm_convert_f32_to_bf8_rne(float in)
{
  libxsmm_float16_ushort hybrid_in = { 0 };
  libxsmm_bfloat8 res;
  unsigned int fixup;
  hybrid_in.f = libxsmm_convert_f32_to_f16(in);
  /* RNE round */
  fixup = (hybrid_in.u >> 8) & 1;
  /* we do not round inf and NaN */
  hybrid_in.u = (unsigned short)(((hybrid_in.u & 0x7c00) == 0x7c00)
    ? ((hybrid_in.u & 0x03ff) == 0x0 ? hybrid_in.u : (hybrid_in.u | 0x0200))
    : (hybrid_in.u + 0x007f + fixup));
  /* shift right */
  res = (libxsmm_bfloat8)(hybrid_in.u >> 8);
  return res;
}


LIBXSMM_API libxsmm_hfloat8 libxsmm_convert_f16_to_hf8_rne(libxsmm_float16 in)
{
  unsigned int f16_bias = 15;
  unsigned int f8_bias = 7;
  libxsmm_hfloat8 res = 0;
  unsigned short s, e, m, e_f16, m_f16;
  unsigned int fixup;

  s = (in & 0x8000) >> 8;
  e_f16 = (in & 0x7c00) >> 10;
  m_f16 = (in & 0x03ff);

  /* special value --> make it NaN */
  if (e_f16 == 0x1f) {
    e = 0xf;
    m = 0x7;
    /* overflow --> make it NaN */
  }
  else if ((e_f16 >  (f16_bias - f8_bias + 15)) ||
          ((e_f16 == (f16_bias - f8_bias + 15)) && (m_f16 > 0x0340)))
  {
    e = 0xf;
    m = 0x7;
    /* smaller than denormal f8 + eps */
  }
  else if (e_f16 < f16_bias - f8_bias - 3) {
    e = 0x0;
    m = 0x0;
    /* denormal */
  }
  else if (e_f16 <= f16_bias - f8_bias) {
    /* RNE */
    /* denormalized mantissa */
    m = m_f16 | 0x0400;
    /* additionally subnormal shift */
    m = m >> ((f16_bias - f8_bias) + 1 - e_f16);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f16 & 0x007f) + 0x007f) >> 7);
    /* RNE Round */
    fixup = (m >> 7) & 0x1;
    m = m + LIBXSMM_CAST_USHORT(0x003f + fixup);
    m = m >> 7;
    e = 0x0;
    /* normal */
  }
  else {
    /* RNE round */
    fixup = (m_f16 >> 7) & 0x1;
    in = in + LIBXSMM_CAST_USHORT(0x003f + fixup);
    e = (in & 0x7c00) >> 10;
    m = (in & 0x03ff);
    LIBXSMM_ASSERT(e >= LIBXSMM_CAST_USHORT(f16_bias - f8_bias));
    e -= LIBXSMM_CAST_USHORT(f16_bias - f8_bias);
    m = m >> 7;
  }

  /* set result to 0 */
  res = 0x0;
  /* set exponent and mantissa */
  res |= e << 3;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}


LIBXSMM_API libxsmm_hfloat8 libxsmm_convert_f32_to_hf8_rne(float in)
{
  const libxsmm_float16 itm = libxsmm_convert_f32_to_f16(in);
  return libxsmm_convert_f16_to_hf8_rne(itm);
}


LIBXSMM_API libxsmm_float16 libxsmm_convert_f32_to_f16(float in)
{
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  libxsmm_float_uint hybrid_in = { 0 };
  libxsmm_float16 res = 0;
  unsigned int s, e, m, e_f32, m_f32;
  unsigned int fixup;
  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ((hybrid_in.u & 0x7f800000) == 0x0
    ? (hybrid_in.u & 0x80000000)
    : (hybrid_in.u & 0xffffffff));

  s = (hybrid_in.u & 0x80000000) >> 16;
  e_f32 = (hybrid_in.u & 0x7f800000) >> 23;
  m_f32 = (hybrid_in.u & 0x007fffff);

  /* special value */
  if (e_f32 == 0xff) {
    e = 0x1f;
    m = (m_f32 == 0 ? 0 : ((m_f32 >> 13) | 0x200));
    /* overflow */
  }
  else if (e_f32 > (f32_bias + f16_bias)) {
    e = 0x1f;
    m = 0x0;
    /* smaller than denormal f16 */
  }
  else if (e_f32 < f32_bias - f16_bias - 10) {
    e = 0x0;
    m = 0x0;
    /* denormal */
  }
  else if (e_f32 <= f32_bias - f16_bias) {
    /* RNE */
#if 1
    /* denormalized mantissa */
    m = m_f32 | 0x00800000;
    /* additionally subnormal shift */
    m = m >> ((f32_bias - f16_bias) + 1 - e_f32);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f32 & 0x1fff) + 0x1fff) >> 13);
    /* RNE Round */
    fixup = (m >> 13) & 0x1;
    m = m + 0x000000fff + fixup;
    m = m >> 13;
    e = 0x0;
#else
    /* RAZ */
    m = (m_f32 | 0x00800000) >> 12;
    m = (m >> ((f32_bias - f16_bias) + 2 - e_f32)) + ((m >> ((f32_bias - f16_bias) + 1 - e_f32)) & 1);
    e = 0x0;
#endif
    /* normal */
  }
  else {
#if 1
    /* RNE round */
    fixup = (m_f32 >> 13) & 0x1;
    hybrid_in.u = hybrid_in.u + 0x000000fff + fixup;
    e = (hybrid_in.u & 0x7f800000) >> 23;
    m = (hybrid_in.u & 0x007fffff);
    LIBXSMM_ASSERT(e >= (f32_bias - f16_bias));
    e -= (f32_bias - f16_bias);
    m = m >> 13;
#else
    /* RAZ */
    hybrid_in.u = hybrid_in.u + 0x00001000;
    e = (hybrid_in.u & 0x7f800000) >> 23;
    m = (hybrid_in.u & 0x007fffff);
    LIBXSMM_ASSERT(e >= (f32_bias - f16_bias));
    e -= (f32_bias - f16_bias);
    m = m >> 13;
#endif
  }

  /* set result to 0 */
  res = 0x0;
  /* set exponent and mantissa */
  res |= e << 10;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}


LIBXSMM_API LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC) double libxsmm_dsqrt(double x)
{
#if defined(LIBXSMM_INTRINSICS_X86) && !defined(__PGI)
  const __m128d a = LIBXSMM_INTRINSICS_MM_UNDEFINED_PD();
  const double result = _mm_cvtsd_f64(_mm_sqrt_sd(a, _mm_set_sd(x)));
#elif !defined(LIBXSMM_NO_LIBM)
  const double result = sqrt(x);
#else /* fallback */
  double result, y = x;
  if (LIBXSMM_NEQ(0, x)) {
    do {
      result = y;
      y = 0.5 * (y + x / y);
    } while (LIBXSMM_NEQ(result, y));
  }
  result = y;
#endif
  return result;
}


LIBXSMM_API LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC) float libxsmm_ssqrt(float x)
{
#if defined(LIBXSMM_INTRINSICS_X86)
  const float result = _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#elif !defined(LIBXSMM_NO_LIBM)
  const float result = LIBXSMM_SQRTF(x);
#else /* fallback */
  float result, y = x;
  if (LIBXSMM_NEQ(0, x)) {
    do {
      result = y;
      y = 0.5f * (y + x / y);
    } while (LIBXSMM_NEQ(result, y));
  }
  result = y;
#endif
  return result;
}


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matdiff)(libxsmm_matdiff_info* /*info*/,
  const int* /*datatype*/, const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const void* /*ref*/, const void* /*tst*/,
  const libxsmm_blasint* /*ldref*/, const libxsmm_blasint* /*ldtst*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matdiff)(libxsmm_matdiff_info* info,
  const int* datatype, const libxsmm_blasint* m, const libxsmm_blasint* n, const void* ref, const void* tst,
  const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst)
{
  static int error_once = 0;
  if ((NULL == datatype || LIBXSMM_DATATYPE_UNSUPPORTED <= *datatype || 0 > *datatype || NULL == m
    || EXIT_SUCCESS != libxsmm_matdiff(info, (libxsmm_datatype)*datatype, *m, *(NULL != n ? n : m), ref, tst, ldref, ldtst))
    && 0 != libxsmm_verbosity && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_matdiff specified!\n");
  }
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matdiff_reduce)(libxsmm_matdiff_info* /*output*/, const libxsmm_matdiff_info* /*input*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matdiff_reduce)(libxsmm_matdiff_info* output, const libxsmm_matdiff_info* input)
{
  libxsmm_matdiff_reduce(output, input);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matdiff_clear)(libxsmm_matdiff_info* /*info*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matdiff_clear)(libxsmm_matdiff_info* info)
{
  libxsmm_matdiff_clear(info);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_coprime2)(long long* /*coprime*/, const int* /*n*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_coprime2)(long long* coprime, const int* n)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != coprime && NULL != n && 0 <= *n)
#endif
  {
    *coprime = (long long)(libxsmm_coprime2((size_t)(*n)) & 0x7FFFFFFF);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_coprime2 specified!\n");
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/
