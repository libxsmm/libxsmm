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
  const int incrmt = (2 < argc ? atoi(argv[2]) : insize);
  const int nelems = (3 < argc ? atoi(argv[3]) : 0);
  const int niters = (4 < argc ? atoi(argv[4]) : 1);
  const int size = (0 >= insize ? LIBXSMM_DESCRIPTOR_MAXSIZE : insize);
  const int stride = LIBXSMM_MAX(incrmt, size);
  const size_t n = (0 >= nelems ? (((size_t)2 << 30/*2 GB*/) / stride) : ((size_t)nelems));
  unsigned char* input;
  size_t npot, nrpt;
  int result;

  if (0 < niters) {
    npot = LIBXSMM_UP2POT(n);
    nrpt = niters;
  }
  else {
    npot = LIBXSMM_UP2POT(LIBXSMM_MAX(LIBXSMM_ABS(niters), 1));
    nrpt = n;
  }
  input = (unsigned char*)(malloc(npot * stride));

  if (NULL != input) {
    unsigned char *const ref = input + (npot - 1) * stride; /* last item */
    libxsmm_timer_tickint start;
    size_t i, j = 0;

    /* initialize the input data */
    for (i = 0; i < npot * stride; ++i) input[i] = LIBXSMM_MOD2(i, 128);
    for (i = 0; i < (size_t)size; ++i) ref[i] = 255;

    { /* benchmark libxsmm_diff_n */
      start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
        j = libxsmm_diff_n(ref, input, (unsigned char)size, (unsigned char)stride, 0/*hint*/, (unsigned int)npot);
      }
      printf("libxsmm_diff_n:\t\t%.3f s\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      result = ((npot == (j + 1) && 0 == memcmp(ref, input + j * stride, size)) ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    if (EXIT_SUCCESS == result) { /* benchmark libxsmm_diff_npot */
#if defined(USE_HASH)
      const unsigned int hashref = libxsmm_hash(ref, size, 0/*seed*/);
#endif
      start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
#if !defined(USE_HASH)
        j = libxsmm_diff_npot(ref, input, (unsigned char)size, (unsigned char)stride, 0/*hint*/, (unsigned int)npot);
#else
        const unsigned char* tst = input;
        for (j = 0; j < npot; ++j) {
          const unsigned int hashtst = libxsmm_hash(tst, size, 0/*seed*/);
          if (hashref == hashtst && 0 == libxsmm_diff(ref, tst, (unsigned char)size)) {
            break;
          }
          tst += stride;
        }
#endif
      }
      printf("libxsmm_diff_npot:\t%.3f s\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      result = ((npot == (j + 1) && 0 == memcmp(ref, input + j * stride, size)) ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    free(input);
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}

