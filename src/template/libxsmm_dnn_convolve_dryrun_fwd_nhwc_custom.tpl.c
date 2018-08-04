/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

int block_j = 14;
#if !defined(_OPENMP)
int ltid;
#endif
handle->block_fwd_ofm = 8;
handle->block_fwd_ifm = 8;

if ( (handle->ofh == 14 && handle->desc.R != 3 ) ||  handle->ofh == 27 || (handle->ofh == 28 && handle->desc.R == 1) || handle->ofh == 48 || handle->ofh == 54 || handle->ofh == 56 || handle->ofh == 112 ) {
  block_j = 4;
}
while ( block_j % handle->fwd_ofh_rb != 0 ) {
  block_j--;
}

handle->block_fwd_oj = block_j;

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ofm1, ifm1, oj, oi, ij, ii, local_entries = 0, ojb, ifmb, ofmb;
  int cur_wt, next_wt, cur_out, next_out;

  /* Threading related variables */
  int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  int threads_per_image = handle->desc.threads / handle->desc.N;
  int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
  int my_ofm_start = 0;
  int my_ofm_end = handle->blocksofm;
  int myOfmId;
  int nOfmBlocks;
  int total_calls;

  /* Arrays of stream indices */
  int *compute_indices;
  char *kernel_variant;

  if ( imgpt <= 1 ) {
    my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
    my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
    myOfmId = ltid % threads_per_image;
    nOfmBlocks = (handle->blocksofm + threads_per_image -1) / threads_per_image;
    my_ofm_start = LIBXSMM_MIN(myOfmId * nOfmBlocks, handle->blocksofm);
    my_ofm_end = LIBXSMM_MIN((myOfmId+1) * nOfmBlocks, handle->blocksofm);
  }

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
    for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
      for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
        for (img = my_img_start; img < my_img_end; img++) {
          for ( ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ++ifm1) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                  local_entries += 3;
                }
              }
            }
          }
        }
      }
    }
  }

  handle->n_entries_fwd[ltid] = local_entries/3;

  /* Allocate auxiliary data structures for index jitting  */
  compute_indices = (int*)libxsmm_aligned_malloc(((size_t)local_entries+3) * sizeof(int), 64);
  handle->compute_fwd_indices_ptrs[ltid] = compute_indices;
  kernel_variant = (char*)(3 <= local_entries ? libxsmm_aligned_malloc((local_entries / 3) * sizeof(char), 64) : NULL);
  handle->kernel_fwd_variant_ptrs[ltid] = kernel_variant;
  local_entries = 0;

  /* Second run to compute actual indices */
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for ( ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ++ifm1) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {

                  ij = oj * handle->desc.u;
                  ii = oi * handle->desc.v;
                  compute_indices[local_entries] =  ( ( ( ( ( (img *  handle->ifhp) +  ij) *  handle->ifwp )  +  ii) * handle->blocksifm)  +  ifm1  ) *  handle->ifmblock * handle->fm_lp_block;
                  compute_indices[local_entries+1] = ( (ofm1 *  handle->blocksifm )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                  compute_indices[local_entries+2] = ( ( ( ( ( (img *  handle->ofhp ) +  oj) *  handle->ofwp )  +  oi) * handle->blocksofm)  +  ofm1) *  handle->ofmblock;

                  /* Initialize  kernel variant the one that prefetches everything */
                  kernel_variant[local_entries/3] = 2;
                  local_entries += 3;
                }
              }
            }
          }
        }
      }
    }
  }

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;

  total_calls = local_entries/3;

  for (ii = 0; ii < total_calls-1; ii++) {
    cur_wt = compute_indices[ii*3+1];
    next_wt = compute_indices[(ii+1)*3+1];
    cur_out = compute_indices[ii*3+2];
    next_out = compute_indices[(ii+1)*3+2];
    if ( cur_wt == next_wt ) {
      kernel_variant[ii] = 1;
    } else if ( cur_out == next_out ) {
      kernel_variant[ii] = 3;
    }
  }

}

