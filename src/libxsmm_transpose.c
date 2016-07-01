#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_trans.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_TRANSPOSE_CACHESIZE)
# define LIBXSMM_TRANSPOSE_CACHESIZE 32768
#endif
#if !defined(LIBXSMM_TRANSPOSE_N)
# define LIBXSMM_TRANSPOSE_N 32
#endif


/* Based on cache-oblivious scheme as published by Frigo et.al. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void inernal_transpose_oop(void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in,
  unsigned int typesize, libxsmm_blasint m0, libxsmm_blasint m1, libxsmm_blasint n0, libxsmm_blasint n1,
  libxsmm_blasint ld, libxsmm_blasint ldo)
{
  const libxsmm_blasint m = m1 - m0, n = n1 - n0;
  libxsmm_blasint i, j;

  if (m * n * typesize <= (LIBXSMM_TRANSPOSE_CACHESIZE / 2)) {
    switch(typesize) {
      case 1: {
        const char *const a = (const char*)in;
        char *const b = (char*)out;
        for (i = n0; i < n1; ++i) {
#if (0 < LIBXSMM_TRANSPOSE_N)
          LIBXSMM_PRAGMA_NONTEMPORAL
          LIBXSMM_PRAGMA_LOOP_COUNT(LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxsmm_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      case 2: {
        const short *const a = (const short*)in;
        short *const b = (short*)out;
        for (i = n0; i < n1; ++i) {
#if (0 < LIBXSMM_TRANSPOSE_N)
          LIBXSMM_PRAGMA_NONTEMPORAL
          LIBXSMM_PRAGMA_LOOP_COUNT(LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxsmm_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      case 4: {
        const float *const a = (const float*)in;
        float *const b = (float*)out;
        for (i = n0; i < n1; ++i) {
#if (0 < LIBXSMM_TRANSPOSE_N)
          LIBXSMM_PRAGMA_NONTEMPORAL
          LIBXSMM_PRAGMA_LOOP_COUNT(LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxsmm_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      case 8: {
        const double *const a = (const double*)in;
        double *const b = (double*)out;
        for (i = n0; i < n1; ++i) {
#if (0 < LIBXSMM_TRANSPOSE_N)
          LIBXSMM_PRAGMA_NONTEMPORAL
          LIBXSMM_PRAGMA_LOOP_COUNT(LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N, LIBXSMM_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxsmm_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      default: assert(0);
    }
  }
  else if (n >= m) {
    const libxsmm_blasint ni = (n0 + n1) / 2;
    inernal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
    inernal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
  }
  else {
#if (0 < LIBXSMM_TRANSPOSE_N)
    if (LIBXSMM_TRANSPOSE_N < m) {
      const libxsmm_blasint mi = m0 + LIBXSMM_TRANSPOSE_N;
      inernal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
      inernal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
    }
    else
#endif
    {
      const libxsmm_blasint mi = (m0 + m1) / 2;
      inernal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
      inernal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
    }
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ldo)
{
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (ld < m && ldo < n) {
    fprintf(stderr, "LIBXSMM: the leading dimensions of the transpose are too small!\n");
  }
  else if (ld < m) {
    fprintf(stderr, "LIBXSMM: the leading dimension of the transpose input is too small!\n");
  }
  else if (ldo < n) {
    fprintf(stderr, "LIBXSMM: the leading dimension of the transpose output is too small!\n");
  }
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
  if (8 == typesize) {
    mkl_domatcopy('C', 'T', m, n, 1, (const double*)in, ld, (double*)out, ldo);
  }
  else if (4 == typesize) {
    mkl_somatcopy('C', 'T', m, n, 1, (const float*)in, ld, (float*)out, ldo);
  }
  else
#endif
  {
    inernal_transpose_oop(out, in, typesize, 0, n, 0, m, ld, ldo);
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_stranspose_oop(float* out, const float* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ldo)
{
  libxsmm_transpose_oop(out, in, sizeof(float), m, n, ld, ldo);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dtranspose_oop(double* out, const double* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ldo)
{
  libxsmm_transpose_oop(out, in, sizeof(double), m, n, ld, ldo);
}

