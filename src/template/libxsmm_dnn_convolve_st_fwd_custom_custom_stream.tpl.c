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
#define IMG_LOOP_INIT 0
#define OFM_LOOP_INIT 1
#define OFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3
#define IFM_LOOP_CLOSE_S 4

#define FP64_BN_STATS

int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm;

const int ltid = tid-start_thread;
int gs = 72; /*atoi(getenv("GSIZE"));*/
const int tile_id = ltid/gs;
/* Pointer variables  */
const element_input_type *input_base, *input_ptr;
const element_filter_type *weight_base;
element_input_type *input_zero;
element_output_type *output_base;
element_input_type *copy_ptr, *prefetch_ptr;
element_output_type *out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_output_type, output, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
/* LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);*/
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data + tile_id * BLOCKSIFM * BLOCKSOFM * handle->ifmblock * handle->ofmblock * handle->fm_lp_block *  handle->desc.R * handle->desc.S, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

/* Auxiliary integer variables   */
int instr, n_segments, offset_bn, offset_i, offset_o, offset_w, pi, po, pw, pc, i, ih, n_convs, conv_i, ifm1, ofm1, ofm2, oj, img, input_h_start, input_h_end, my_h_out, oi;
/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_fwd_indices_ptrs[ltid];
int *bn_stream = handle->bn_indices_ptrs[ltid];

/* Padding related variables */
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * BLOCKSIFM * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
/* Kernel related variables  */
libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_fwd[0].xmatcopy;
libxsmm_xmcopyfunction jitted_zero_overwrite = handle->matcopy_fwd[1].xmatcopy;
libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
libxsmm_convfunction kernel2 = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
libxsmm_convfunction kernel_pool[2];
kernel_pool[0] = kernel;
kernel_pool[1] = kernel2;
char *variant = handle->kernel_fwd_variant_ptrs[ltid];
int pool_index = 0;

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  input_base = &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
  input_zero = &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
  /* we need to set the scratch to zero */
  /* @TODO: we need to find a better/faster code here */
  memset( input_zero, 0, BLOCKSIFM * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block * sizeof(element_input_type) );
} else {
  input_base = &LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, 0, 0, 0,
      BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
}
weight_base = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
    BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0,
    BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);

instr = handle->n_entries_fwd[ltid];
n_segments = handle->n_fwd_code_segments[ltid];

LIBXSMM_ALIGNED(float scale_factor, 64);
if (handle->use_lp_kernel == 1) {
  scale_factor = libxsmm_sexp2(-1.f*((float)(handle->reg_filter->scf + handle->reg_input->scf)));
}

LIBXSMM_ALIGNED(float *max_vals, 64);
#ifdef __AVX512F__
__m512 max_abs;
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif
if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, element_output_type, maxstats, (element_output_type*)handle->maxstats_fwd->data, handle->ofmblock);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, handle->ofmblock);
#ifdef __AVX512F__
  max_abs = _mm512_setzero_ps();
  _mm512_store_ps(max_vals, max_abs);
#else
/* won't happen as this code only runs on AVX512 platforms */
#endif
}

i = 0;
/* Stream for BN offsets */
int bn_i = 0;
#ifdef FP32_BN_STATS
element_output_type *bn_sum_base;
element_output_type *bn_sum_base2;
#endif
#ifdef FP64_BN_STATS
double *bn_sum_base;
double *bn_sum_base2;
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if (n_segments) {
  /* We have segmented the stream of convolutions since we need to inject different functionalities...  */
  code_stream = handle->fwd_code_segments[ltid];
  /* If we are in the img_par execution then avoid fine-grained copy in case of padding...  */
  if (handle->desc.N*BLOCKSOFM >= handle->desc.threads) {
    if (handle->compute_batch_stats_in_kernel == 1) { /* We  do BN stuff in the kernel  */
#ifdef FP32_BN_STATS
      LIBXSMM_VLA_DECL(4, element_output_type, kernel_stats, handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#endif
#ifdef FP64_BN_STATS
      LIBXSMM_VLA_DECL(4, double, kernel_stats, (double*)handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#endif
      bn_sum_base =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 0, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);
      bn_sum_base2 =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 1, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);

      if (handle->n_variants == 2) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            const int vi = variant[pool_index]; /* avoid warning about char used as array index */
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            offset_bn = bn_stream[bn_i];
            kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
            ++pool_index;
            i += 3;
            ++bn_i;
          }
        }
      } else {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;
          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            offset_bn = bn_stream[bn_i];
            kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
            i += 3;
            ++bn_i;
          }
        }
      }
    } else { /* We don't do BN stuff in the kernel  */
      if (handle->n_variants == 2) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if (instr == OFM_LOOP_CLOSE) {
            /* Compute batch norm statistics... */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) {
#ifdef __AVX512F__
#ifdef FP32_BN_STATS
              ofm1 =  code_stream[pc].aux_index;
              LIBXSMM_VLA_DECL(4, element_output_type, stats, handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
              element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              __m512 bsum  = _mm512_setzero_ps();
              __m512 bsum2 = _mm512_setzero_ps();

              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                  __m512 btmp = _mm512_load_ps( red+oi );
                  bsum = _mm512_add_ps( bsum, btmp );
                  bsum2 = _mm512_add_ps( bsum2, _mm512_mul_ps( btmp, btmp ) );
                }
                red += handle->ofwp*handle->ofmblock;
              }

              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N,  handle->ofmblock), bsum );
              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2 );
#endif
#ifdef FP64_BN_STATS
              ofm1 =  code_stream[pc].aux_index;
              LIBXSMM_VLA_DECL(4, double, stats, (double*)handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
              element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              __m512d bsum1a = _mm512_setzero_pd();
              __m512d bsum1b = _mm512_setzero_pd();
              __m512d bsum2a = _mm512_setzero_pd();
              __m512d bsum2b = _mm512_setzero_pd();

              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                  __m512d btmpa = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+0)) );
                  __m512d btmpb = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+8)) );
                  bsum1a = _mm512_add_pd( bsum1a, btmpa);
                  bsum1b = _mm512_add_pd( bsum1b, btmpb);
                  bsum2a = _mm512_add_pd( bsum2a, _mm512_mul_pd( btmpa, btmpa ) );
                  bsum2b = _mm512_add_pd( bsum2b, _mm512_mul_pd( btmpb, btmpb ) );
                }
                red += handle->ofwp*handle->ofmblock;
              }

              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1a );
              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 8,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1b );
              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2a );
              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 8,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2b );
#endif
#else
              /* won't happen as this code only runs on AVX512 platforms */
#endif
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              ofm1 =  code_stream[pc].aux_index;
              element_output_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(_mm512_load_ps(cur_vec+oi)));
#else
                  /* Won't happen as this code only runs on AVX512 systems */
#endif
                }
                cur_vec += handle->ofwp*handle->ofmblock;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            const int vi = variant[pool_index]; /* avoid warning about char used as array index */
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            ++pool_index;
            i += 3;
          }
        }
      } else {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;
          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == OFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == OFM_LOOP_CLOSE ) {
            /* Compute batch norm statistics... */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) {
#ifdef __AVX512F__
#ifdef FP32_BN_STATS
              ofm1 =  code_stream[pc].aux_index;
              LIBXSMM_VLA_DECL(4, element_output_type, stats, handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
              element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              __m512 bsum  = _mm512_setzero_ps();
              __m512 bsum2 = _mm512_setzero_ps();

              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                  __m512 btmp = _mm512_load_ps( red+oi );
                  bsum = _mm512_add_ps( bsum, btmp );
                  bsum2 = _mm512_add_ps( bsum2, _mm512_mul_ps( btmp, btmp ) );
                }
                red += handle->ofwp*handle->ofmblock;
              }

              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N,  handle->ofmblock), bsum );
              _mm512_store_ps( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2 );
#endif
#ifdef FP64_BN_STATS
              ofm1 =  code_stream[pc].aux_index;
              LIBXSMM_VLA_DECL(4, double, stats, (double*)handle->batch_stats->data,  BLOCKSOFM, handle->desc.N, handle->ofmblock);
              element_output_type* red = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              __m512d bsum1a = _mm512_setzero_pd();
              __m512d bsum1b = _mm512_setzero_pd();
              __m512d bsum2a = _mm512_setzero_pd();
              __m512d bsum2b = _mm512_setzero_pd();

              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
                  __m512d btmpa = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+0)) );
                  __m512d btmpb = _mm512_cvtps_pd ( _mm256_load_ps((const float*) (red+oi+8)) );
                  bsum1a = _mm512_add_pd( bsum1a, btmpa);
                  bsum1b = _mm512_add_pd( bsum1b, btmpb);
                  bsum2a = _mm512_add_pd( bsum2a, _mm512_mul_pd( btmpa, btmpa ) );
                  bsum2b = _mm512_add_pd( bsum2b, _mm512_mul_pd( btmpb, btmpb ) );
                }
                red += handle->ofwp*handle->ofmblock;
              }

              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1a );
              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 0, ofm1, img, 8,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum1b );
              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 0,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2a );
              _mm512_store_pd( &LIBXSMM_VLA_ACCESS(4, stats, 1, ofm1, img, 8,
                    BLOCKSOFM, handle->desc.N, handle->ofmblock), bsum2b );
#endif
#else
              /* won't happen as this code only runs on AVX512 platforms */
#endif
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
              ofm1 =  code_stream[pc].aux_index;
              element_output_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
              for ( oj = 0; oj < handle->ofh; oj++ ) {
                for ( oi = 0; oi < handle->ofw*handle->ofmblock; oi+=16 ) {
#ifdef __AVX512F__
                  max_abs = _mm512_max_ps(max_abs, LIBXSMM_INTRINSICS_MM512_ABS_PS(_mm512_load_ps(cur_vec+oi)));
#else
                  /* won't happen as this code only runs on AVX512 platforms */
#endif
                }
                cur_vec += handle->ofwp*handle->ofmblock;
              }
            }
          }

          /* Run the stream of convolutions for this segment */
          for (conv_i = 0; conv_i < n_convs; conv_i++) {
            offset_i = stream[i];
            offset_w = stream[i+1];
            offset_o = stream[i+2];
            pi = stream[i+3];
            pw = stream[i+4];
            po = stream[i+5];
            kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            i += 3;
          }
        }
      }
    }
  } else {
    /* Use fine-grained operations since we are in the img_par path, so update relevant kernel pointers... */
    jitted_matcopy = handle->matcopy_fwd[2].xmatcopy;
    jitted_zero_overwrite = handle->matcopy_fwd[3].xmatcopy;
    input_h_start = LIBXSMM_MAX(0,  handle->ofh_fwd_start[ltid] - handle->desc.R + 1);
    input_h_end = LIBXSMM_MIN( handle->ifhp, (handle->ofh_fwd_end[ltid] + handle->desc.R -1) * handle->desc.u );
    my_h_out = handle->ofh_fwd_end[ltid]-handle->ofh_fwd_start[ltid];
    for (pc = 0; pc < n_segments; pc++) {
      instr = code_stream[pc].segment_type;
      n_convs = code_stream[pc].n_convs;
      if (instr == IMG_LOOP_INIT) {
        /* Padding code via jitted matcopy kernel */
#include "libxsmm_dnn_fwd_custom_custom_padding_img_par.tpl.c"
      }

      if ( instr == OFM_LOOP_INIT ) {
        /* Apply bias if requested  */
        if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias_img_par.tpl.c"
        }
        /* Overwrite output with zeros if requested */
        if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
          for ( ih = 0; ih < my_h_out * handle->ofmblock * handle->ofwp; ih += handle->ofmblock * handle->ofwp) {
            jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2] + ih, NULL, NULL);
          }
        }
      }

      /* Run the stream of convolutions for this segment */
      for (conv_i = 0; conv_i < n_convs; conv_i++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
        i += 3;
      }
    }
  }
} else {
  /* Run the stream of convolutions, no extra operations are required... */
  if ( handle->compute_batch_stats_in_kernel == 1 ) { /* We  do BN stuff in the kernel  */
#ifdef FP32_BN_STATS
    LIBXSMM_VLA_DECL(4, element_output_type, kernel_stats, handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#endif
#ifdef FP64_BN_STATS
    LIBXSMM_VLA_DECL(4, double, kernel_stats, (double*)handle->batch_stats->data, BLOCKSOFM, handle->desc.N, handle->ofmblock);
#endif
    bn_sum_base =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 0, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);
    bn_sum_base2 =  &LIBXSMM_VLA_ACCESS(4, kernel_stats, 1, 0, 0, 0, BLOCKSOFM, handle->desc.N, handle->ofmblock);
    if (handle->n_variants  == 2) {
      for (pc = 0; pc < instr; pc += 1) {
        const int vi = variant[pc]; /* avoid warning about char used as array index */
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        offset_bn = bn_stream[bn_i];
        kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
        i += 3;
        ++bn_i;
      }
    } else {
      for (pc = 0; pc < instr; pc++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        offset_bn = bn_stream[bn_i];
        kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po,  bn_sum_base + offset_bn, bn_sum_base2 + offset_bn, &scale_factor, max_vals);
        i += 3;
        ++bn_i;
      }
    }
  } else { /* We do not  do BN stuff in the kernel  */
    if (handle->n_variants == 2) {
      for (pc = 0; pc < instr; pc += 1) {
        const int vi = variant[pc]; /* avoid warning about char used as array index */
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel_pool[vi]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
        i += 3;
      }
    } else {
      for (pc = 0; pc < instr; pc++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
        i += 3;
      }
    }
  }
}

if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_lp_kernel == 1) && (handle->compute_max_in_kernel_fwd == 0) ) {
#ifdef __AVX512F__
  _mm512_store_ps(max_vals, max_abs);
#else
  /* won't happen as this code only runs on AVX512 platforms */
#endif
}

libxsmm_barrier_wait(handle->barrier, ltid);

#undef IMG_LOOP_INIT
#undef OFM_LOOP_INIT
#undef OFM_LOOP_CLOSE
#undef CONVOLUTION_KERNEL
#undef IFM_LOOP_CLOSE_S
#undef FP64_BN_STATS

