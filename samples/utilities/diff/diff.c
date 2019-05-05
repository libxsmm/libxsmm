/******************************************************************************
** Copyright (c) 2019, Intel Corporation                                     **
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
#include <libxsmm.h>
#include <string.h>
#include <stdio.h>

#if !defined(USE_HASH) && 0
# define USE_HASH
#endif


int main(int argc, char* argv[])
{
  const int insize = (1 < argc ? atoi(argv[1]) : 0);
  const int incrmt = (2 < argc ? atoi(argv[2]) : 0);
  const int nelems = (3 < argc ? atoi(argv[3]) : 0);
  const int niters = (4 < argc ? atoi(argv[4]) : 1);
  const int elsize = (0 >= insize ? LIBXSMM_DESCRIPTOR_SIGSIZE : insize);
  const int stride = (0 >= incrmt ? LIBXSMM_MAX(LIBXSMM_DESCRIPTOR_MAXSIZE, elsize) : LIBXSMM_MAX(incrmt, elsize));
  const size_t n = (0 >= nelems ? (((size_t)2 << 30/*2 GB*/) / stride) : ((size_t)nelems));
  unsigned char *input, *icopy = NULL, *ilast = NULL;
  int result = EXIT_SUCCESS;
  size_t nbytes, size, nrpt;

  if (0 < niters) {
    size = n;
    nrpt = niters;
  }
  else {
    size = LIBXSMM_MAX(LIBXSMM_ABS(niters), 1);
    nrpt = n;
  }
  nbytes = size * stride;
  input = (unsigned char*)(0 != nbytes ? malloc(nbytes) : NULL);

  if (NULL != input) {
    unsigned char *const ref = input + (size - 1) * stride; /* last item */
    libxsmm_timer_tickint start;
    size_t i, j = 0;

    /* initialize the input data */
    for (i = 0; i < nbytes; ++i) input[i] = LIBXSMM_MOD2(i, 128);
    for (i = 0; i < (size_t)elsize; ++i) ref[i] = 255;

    { /* benchmark libxsmm_diff_n */
#if defined(USE_HASH)
      const unsigned int hashref = libxsmm_hash(ref, elsize, 0/*seed*/);
#endif
      start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
#if !defined(USE_HASH)
        j = libxsmm_diff_n(ref, input, (unsigned char)elsize, (unsigned char)stride,
          (unsigned int)LIBXSMM_MIN(i, size)/*hint*/, (unsigned int)size);
#else
        const unsigned char* tst = input;
        for (j = 0; j < size; ++j) {
          const unsigned int hashtst = libxsmm_hash(tst, elsize, 0/*seed*/);
          if (hashref == hashtst && 0 == libxsmm_diff(ref, tst, (unsigned char)elsize)) {
            break;
          }
          tst += stride;
        }
#endif
      }
      printf("libxsmm_diff_n:\t\t%.8f s\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
    }

    if (size == (j + 1) && 0 == memcmp(ref, input + j * stride, elsize)) { /* benchmark libxsmm_memcmp */
      icopy = (unsigned char*)(elsize == stride ? malloc(nbytes) : NULL);
      if (NULL != icopy) {
        ilast = icopy + (size - 1) * stride; /* last item */
        memcpy(icopy, input, nbytes);
        start = libxsmm_timer_tick();
        for (i = 0; i < nrpt; ++i) {
          j += libxsmm_memcmp(input, icopy, nbytes); /* take result of every execution */
          /* memcmp may be pure and without touching input it is not repeated (nrpt) */
          ilast[i%elsize] = 255;
        }
        printf("libxsmm_memcmp:\t\t%.8f s\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
        result += (int)j * ((int)stride / ((int)stride + 1)); /* ignore result */
      }
    }
    else {
      result = EXIT_FAILURE;
    }

    if (NULL != icopy) { /* benchmark stdlib's memcmp */
      LIBXSMM_ASSERT(NULL != ilast);
      start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
        j += memcmp(input, icopy, nbytes); /* take result of every execution */
        /* memcmp is likely pure and without touching input it is not repeated (nrpt) */
        ilast[i%elsize] = 255;
      }
      printf("stdlib memcmp:\t\t%.8f s\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      result += (int)j * ((int)stride / ((int)stride + 1)); /* ignore result */
      free(icopy);
    }

    free(input);
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}

