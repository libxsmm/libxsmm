/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
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
#include <libxsmm_source.h>
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main(void)
{
  const unsigned int seed = 1975;
  unsigned int size = 2507, i, h1, h2;
  int result = EXIT_SUCCESS;
  const int* value;

  int *const data = (int*)libxsmm_malloc(sizeof(int) * size);
  if (NULL == data) size = 0;
  for (i = 0; i < size; ++i) data[i] = (rand() - ((RAND_MAX) >> 1));

  h1 = libxsmm_crc32(data, sizeof(int) * size, seed);
  h2 = libxsmm_crc32_sw(data, sizeof(int) * size, seed);
  if (h1 != h2) {
#if defined(_DEBUG)
    fprintf(stderr, "(crc32=%u) != (crc32_sw=%u)\n", h1, h2);
#endif
    result = EXIT_FAILURE;
  }

  size >>= 4;
  value = data;
  h1 = h2 = seed;
  for (i = 0; i < size; ++i) {
    h1 = libxsmm_crc32_u512(value, h1);
    h2 = libxsmm_crc32_u512_sw(value, h2);
    value += 16;
  }
  if (h1 != h2 || h1 != libxsmm_crc32(data, sizeof(int) * 16 * size, seed)) {
#if defined(_DEBUG)
    fprintf(stderr, "(crc32=%u) != (crc32_sw=%u)\n", h1, h2);
#endif
    result = EXIT_FAILURE;
  }

  libxsmm_free(data);

  return result;
}

