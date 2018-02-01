/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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


int main(void)
{
  LIBXSMM_ATOMIC_LOCKTYPE lock = 0/*unlocked*/;
  const int kind = LIBXSMM_ATOMIC_RELAXED;
  int result = EXIT_SUCCESS;
  int mh = 1051981, hp, tmp;

  LIBXSMM_NONATOMIC_STORE(&hp, 25071975, kind);
  if (LIBXSMM_NONATOMIC_LOAD(&hp, kind) != LIBXSMM_ATOMIC_LOAD(&hp, kind)) {
    result = EXIT_FAILURE;
  }
  if (mh != LIBXSMM_NONATOMIC_SUB_FETCH(&hp, 24019994, kind)) {
    result = EXIT_FAILURE;
  }
  if (mh != LIBXSMM_ATOMIC_FETCH_ADD(&hp, 24019994, kind)) {
    result = EXIT_FAILURE;
  }
  LIBXSMM_ATOMIC_STORE(&tmp, mh, kind);
  if (25071975 != LIBXSMM_NONATOMIC_FETCH_OR(&hp, tmp, kind)) {
    result = EXIT_FAILURE;
  }
  if ((25071975 | mh) != hp) {
    result = EXIT_FAILURE;
  }
  /* check if non-atomic and atomic are compatible */
  if (LIBXSMM_NONATOMIC_TRYLOCK(&lock, kind)) {
    if (LIBXSMM_ATOMIC_TRYLOCK(&lock, kind)) {
      result = EXIT_FAILURE;
    }
    LIBXSMM_NONATOMIC_RELEASE(&lock, kind);
  }
  else {
    result = EXIT_FAILURE;
  }

  return result;
}

