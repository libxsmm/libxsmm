/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

void shuffle_array(unsigned int *array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n - 1; i++)
    {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

unsigned int random_mask_half_full(int sparsity_factor ) {
  int __i;
  unsigned int id_array[32];
  unsigned int cur_bitmap = 0;

  for (__i = 0; __i < 32; __i++) {
    id_array[__i] = __i;
  }
  shuffle_array(id_array, 32);

  for (__i = 0; __i < 32/sparsity_factor; __i++) {
    unsigned int cur_bit = (1 << id_array[__i]);
    cur_bitmap = cur_bitmap | cur_bit;
  }

  return cur_bitmap;
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_filter, *naive_delinput, *naive_deloutput, *naive_delfilter, *naive_bias, *naive_delbias;
  libxsmm_bfloat16 *naive_input_bf16, *naive_filter_bf16, *naive_output_bf16, *naive_delinput_bf16, *naive_delfilter_bf16, *naive_deloutput_bf16, *naive_bias_bf16, *naive_delbias_bf16;
  float *naive_libxsmm_output_f32, *naive_libxsmm_delinput_f32, *naive_libxsmm_delfilter_f32, *naive_libxsmm_delbias_f32;
  libxsmm_bfloat16 *naive_libxsmm_output_bf16, *naive_libxsmm_delinput_bf16, *naive_libxsmm_delfilter_bf16, *naive_libxsmm_delbias_bf16;
  libxsmm_bfloat16 *input_libxsmm, *filter_libxsmm, *filter_libxsmm_sparse, *delinput_libxsmm, *delfilter_libxsmm, *output_libxsmm, *deloutput_libxsmm, *bias_libxsmm, *delbias_libxsmm;
  unsigned char *relumask_libxsmm;
  unsigned int  *sparse_idx_libxsmm;
  int __i, __j = 0, __k = 0;

  naive_fullyconnected_t naive_param;
  void* scratch;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 100;         /* repetitions of benchmark */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIFm = 256;          /* number of input feature maps, "C" */
  int nOFm = 256;          /* number of input feature maps, "C" */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'B';
  int bn = 32;
  int bk = 32;
  int bc = 32;
  int sparsity_factor = 2;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gflop = 0.0;
  int i;

  libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
  libxsmm_dnn_fullyconnected* libxsmm_handle;
  libxsmm_dnn_tensor*  libxsmm_input;
  libxsmm_dnn_tensor*  libxsmm_delinput;
  libxsmm_dnn_tensor*  libxsmm_output;
  libxsmm_dnn_tensor*  libxsmm_deloutput;
  libxsmm_dnn_tensor*  libxsmm_filter;
  libxsmm_dnn_tensor*  libxsmm_delfilter;
  libxsmm_dnn_tensor*  libxsmm_bias;
  libxsmm_dnn_tensor*  libxsmm_delbias;
  libxsmm_dnn_tensor*  libxsmm_relumask;
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters nImg nIFm nOFm fuse_type type format\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIFm       = atoi(argv[i++]);
  if (argc > i) nOFm       = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) format     = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);
  if (argc > i) sparsity_factor        = atoi(argv[i++]);

  /* These are tuning parameters to be attached to the perfdump string  */
#if 0
  int fwd_bf = atoi(getenv("FWD_BF"));
  int bwd_bf = atoi(getenv("BWD_BF"));
  int upd_bf = atoi(getenv("UPD_BF"));
  int fwd_2d_blocking = atoi(getenv("FWD_2D_BLOCKING"));
  int bwd_2d_blocking = atoi(getenv("BWD_2D_BLOCKING"));
  int upd_2d_blocking = atoi(getenv("UPD_2D_BLOCKING"));
  int fwd_row_teams = atoi(getenv("FWD_ROW_TEAMS"));
  int fwd_column_teams = atoi(getenv("FWD_COLUMN_TEAMS"));
  int bwd_row_teams = atoi(getenv("BWD_ROW_TEAMS"));
  int bwd_column_teams = atoi(getenv("BWD_COLUMN_TEAMS"));
  int upd_row_teams = atoi(getenv("UPD_ROW_TEAMS"));
  int upd_column_teams = atoi(getenv("UPD_COLUMN_TEAMS"));
  int ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
  int ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
#endif
  int fwd_bf = 1;
  int bwd_bf = 1;
  int upd_bf = 1;
  int fwd_2d_blocking = 1;
  int bwd_2d_blocking = 1;
  int upd_2d_blocking = 1;
  int fwd_row_teams = 1;
  int fwd_column_teams = 1;
  int bwd_row_teams = 1;
  int bwd_column_teams = 1;
  int upd_row_teams = 1;
  int upd_column_teams = 1;
  int ifm_subtasks = 1;
  int ofm_subtasks = 1;

  if ( nImg % bn != 0 ) {
    bn = nImg;
  }
  if ( nIFm % bc != 0 ) {
    bc = nIFm;
  }
  if ( nOFm % bk != 0 ) {
    bk = nOFm;
  }

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U' && type != 'M') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (UP only). 'M' (BPUP-fused only)\n");
    return -1;
  }
  if ( (fuse_type < 0) || (fuse_type > 5) ) {
    printf("fuse type needs to be 0 (None), 1 (Bias), 2 (ReLU), 3 (Sigmoid), 4 (Bias+ReLU), 5 (Bias+Sigmoid)\n");
    return -1;
  }
  if (format != 'L' && format != 'B') {
    printf("format needs to be 'L' (libxsmm) or 'B' (for locked NCNC KCCK)\n");
    return -1;
  }

  /* set struct for naive convolution */
  naive_param.N = nImg;
  naive_param.C = nIFm;
  naive_param.K = nOFm;
  naive_param.fuse_type = fuse_type;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d  C:%d  K:%d\n", nImg, nIFm, nOFm);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOFm*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIFm*   sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOFm*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Filter     : %10.2f MiB\n", (double)(nIFm*nOFm*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );

  /* allocate data */
  naive_input                 = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_delinput              = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_output                = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_deloutput             = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_filter                = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_delfilter             = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_bias                  = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);
  naive_delbias               = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  naive_input_bf16            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_delinput_bf16         = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_output_bf16           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_deloutput_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_filter_bf16           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_delfilter_bf16        = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_bias_bf16             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  naive_delbias_bf16          = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);

  naive_libxsmm_output_bf16   = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delinput_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delfilter_bf16= (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_delbias_bf16  = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  naive_libxsmm_output_f32    = (float*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delinput_f32  = (float*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(float), 2097152);
  naive_libxsmm_delfilter_f32 = (float*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(float), 2097152);
  naive_libxsmm_delbias_f32   = (float*)libxsmm_aligned_malloc( nOFm     *sizeof(float), 2097152);

  input_libxsmm               = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  delinput_libxsmm            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIFm*sizeof(libxsmm_bfloat16), 2097152);
  output_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  deloutput_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  filter_libxsmm              = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  filter_libxsmm_sparse       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  sparse_idx_libxsmm          = (unsigned int*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(unsigned int), 2097152);
  delfilter_libxsmm           = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nIFm*nOFm*sizeof(libxsmm_bfloat16), 2097152);
  bias_libxsmm                =  (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  delbias_libxsmm             =  (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOFm     *sizeof(libxsmm_bfloat16), 2097152);
  relumask_libxsmm            =  (unsigned char*)libxsmm_aligned_malloc( nImg*nOFm*sizeof(unsigned char), 2097152);

  /* initialize data */
  init_buf( naive_input,     nImg*nIFm, 0, 0 );
  init_buf( naive_delinput,  nImg*nIFm, 0, 0 );
  init_buf( naive_output,    nImg*nOFm, 0, 0 );
  init_buf( naive_deloutput, nImg*nOFm, 0, 0 );
  init_buf( naive_filter,    nIFm*nOFm, 0, 0 );
  init_buf( naive_delfilter, nIFm*nOFm, 0, 0 );
  init_buf( naive_bias,      nOFm,      0, 0 );
  init_buf( naive_delbias,   nOFm,      0, 0 );

  libxsmm_rne_convert_fp32_bf16( naive_input,     naive_input_bf16,     nImg*nIFm );
  libxsmm_rne_convert_fp32_bf16( naive_delinput,  naive_delinput_bf16,  nImg*nIFm );
  libxsmm_rne_convert_fp32_bf16( naive_output,    naive_output_bf16,    nImg*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_deloutput, naive_deloutput_bf16, nImg*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_filter,    naive_filter_bf16,    nIFm*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_delfilter, naive_delfilter_bf16, nIFm*nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_bias,      naive_bias_bf16,      nOFm );
  libxsmm_rne_convert_fp32_bf16( naive_delbias,   naive_delbias_bf16,   nOFm );

  if (format == 'A' || format == 'B') {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    fullyconnected_desc.N = nImg;
    fullyconnected_desc.C = nIFm;
    fullyconnected_desc.K = nOFm;
    fullyconnected_desc.bn = bn;
    fullyconnected_desc.bk = bk;
    fullyconnected_desc.bc = bc;
    fullyconnected_desc.threads = nThreads;
    fullyconnected_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    fullyconnected_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
    fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
    fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
    fullyconnected_desc.compressed_A = 1;
    fullyconnected_desc.sparsity_factor_A = sparsity_factor;
    if ( fuse_type == 0 ) {
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;
    } else if ( fuse_type == 1 ) {
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS;
    } else if ( fuse_type == 2 ) {
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU;
    } else if ( fuse_type == 3 ) {
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID;
    } else if ( fuse_type == 4 ) {
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU;
    } else if ( fuse_type == 5 ) {
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID;
    } else {
      /* cannot happen */
    }

    libxsmm_handle = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers */
    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    printf("inner activation blocking: %i\n", libxsmm_layout->dim_size[0] );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout, filter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delfilter  = libxsmm_dnn_link_tensor( libxsmm_layout, delfilter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias = libxsmm_dnn_link_tensor( libxsmm_layout, bias_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbias  = libxsmm_dnn_link_tensor( libxsmm_layout, delbias_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RELU_MASK, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_relumask  = libxsmm_dnn_link_tensor( libxsmm_layout, relumask_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    /* we can also use the layout functions and set the data on our
       own external to the library */
    matrix_copy_NC_to_NCNC_bf16( naive_input_bf16,     input_libxsmm,     1, nImg, nIFm, bn, bc );
    matrix_copy_NC_to_NCNC_bf16( naive_delinput_bf16,  delinput_libxsmm,  1, nImg, nIFm, bn, bc );
    matrix_copy_NC_to_NCNC_bf16( naive_output_bf16,    output_libxsmm,    1, nImg, nOFm, bn, bk );
    matrix_copy_NC_to_NCNC_bf16( naive_deloutput_bf16, deloutput_libxsmm, 1, nImg, nOFm, bn, bk );
    matrix_copy_KC_to_KCCK_bf16( naive_filter_bf16,    filter_libxsmm      , nIFm, nOFm, bc, bk );
    matrix_copy_KC_to_KCCK_bf16( naive_delfilter_bf16, delfilter_libxsmm   , nIFm, nOFm, bc, bk );
    matrix_copy_NC_to_NCNC_bf16( naive_bias_bf16,    bias_libxsmm,    1, 1, nOFm, 1, nOFm );
    matrix_copy_NC_to_NCNC_bf16( naive_delbias_bf16, delbias_libxsmm, 1, 1, nOFm, 1, nOFm );

    /* Sparsify filters to requested level */
    memset(filter_libxsmm_sparse, 0, nIFm * nOFm * sizeof(libxsmm_bfloat16));
#if defined(__AVX512VBMI2__) || (defined(__AVX512BW__) && defined(LIBXSMM_INTEL_COMPILER))
    for (__i = 0; __i < nIFm * nOFm; __i+= 32 ) {
      unsigned int  cur_mask_int    = random_mask_half_full(sparsity_factor);
      __mmask32     cur_mask        = _cvtu32_mask32(cur_mask_int);
      __m512i       vreg_dense      = _mm512_loadu_si512 ((libxsmm_bfloat16*)filter_libxsmm+__i);
      sparse_idx_libxsmm[__k] = cur_mask_int;
      _mm512_mask_storeu_epi16 ((libxsmm_bfloat16*)filter_libxsmm_sparse+__i, cur_mask, vreg_dense);
      _mm512_mask_compressstoreu_epi16((libxsmm_bfloat16*)filter_libxsmm+__j, cur_mask, vreg_dense);
      __k += 1;
      __j += 32/sparsity_factor;
    }
#else
    fprintf(stderr,
      "ERROR: support for at least "
# if defined(LIBXSMM_INTEL_COMPILER)
      "AVX512BW"
# else
      "AVX512VBMI2"
# endif
      " required for this kernel!\n");
    exit(EXIT_FAILURE);
#endif
    /* Copyover the sparse idx tensor after the compressed filter buffer  */
    memcpy((libxsmm_bfloat16*)filter_libxsmm + (nIFm * nOFm)/sparsity_factor, sparse_idx_libxsmm, ((nIFm * nOFm)/32)*sizeof(unsigned int));
    /* Copyover the sparse filter ot the reference filter for correctness checks */
    matrix_copy_KCCK_to_KC_bf16( filter_libxsmm_sparse, naive_filter_bf16, nIFm, nOFm, bc, bk );
    libxsmm_convert_bf16_f32( naive_filter_bf16, naive_filter, nIFm*nOFm );

    if (LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#         Computing Reference ...        #\n");
      printf("##########################################\n");
      if (type == 'A' || type == 'F') {
        naive_fullyconnected_fused_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
      }
      if (type == 'A' || type == 'B' || type == 'M') {
        naive_fullyconnected_fused_bp(&naive_param, naive_delinput, naive_deloutput, naive_filter, naive_delbias, naive_output);
      }
      if (type == 'A' || type == 'U' || type == 'M') {
        naive_fullyconnected_wu(&naive_param, naive_input, naive_deloutput, naive_delfilter);
      }
      printf("##########################################\n");
      printf("#      Computing Reference ... done      #\n");
      printf("##########################################\n");
    }

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_input,        LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_delinput,     LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_output,       LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_deloutput,    LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_filter,       LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_delfilter,    LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_bias,         LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_delbias,      LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_relumask,     LIBXSMM_DNN_RELU_MASK ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");

#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }

      /* copy out data */
      matrix_copy_NCNC_to_NC_bf16( output_libxsmm, naive_libxsmm_output_bf16, 1, nImg, nOFm, bn, bk );
      libxsmm_convert_bf16_f32( naive_libxsmm_output_bf16, naive_libxsmm_output_f32, nImg*nOFm );

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, nImg*nOFm, 1, naive_output, naive_libxsmm_output_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    }
    if ( (type == 'A' || type == 'B') && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");

#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      matrix_copy_NCNC_to_NC_bf16( delinput_libxsmm, naive_libxsmm_delinput_bf16, 1, nImg, nIFm, bn, bc );
      libxsmm_convert_bf16_f32( naive_libxsmm_delinput_bf16, naive_libxsmm_delinput_f32, nImg*nIFm );

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIFm, 1, naive_delinput, naive_libxsmm_delinput_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      if ( (fuse_type == 1) || (fuse_type == 4) || (fuse_type == 5) ) {
        /* copy out data */
        matrix_copy_NCNC_to_NC_bf16( delbias_libxsmm, naive_libxsmm_delbias_bf16, 1, 1, nOFm, 1, nOFm );
        libxsmm_convert_bf16_f32( naive_libxsmm_delbias_bf16, naive_libxsmm_delbias_f32, nOFm );
        libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nOFm, 1, naive_delbias, naive_libxsmm_delbias_f32, 0, 0);
        printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
        printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
        printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
        libxsmm_matdiff_reduce(&diff, &norms_bwd);
      }
    }

    if ( (type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - UPD (custom-Storage)   #\n");
      printf("##########################################\n");

#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      /* copy out data */
      matrix_copy_KCCK_to_KC_bf16( delfilter_libxsmm, naive_libxsmm_delfilter_bf16, nIFm, nOFm, bc, bk );
      libxsmm_convert_bf16_f32( naive_libxsmm_delfilter_bf16, naive_libxsmm_delfilter_f32, nIFm*nOFm );

      /* compare */
      libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nIFm*nOFm, 1, naive_delfilter, naive_libxsmm_delfilter_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd);
    }

    if ( (type == 'A' || type == 'M') && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("# Correctness - BWDUPD (custom-Storage)  #\n");
      printf("##########################################\n");

#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid ) );
      }

      /* copy out data */
      matrix_copy_NCNC_to_NC_bf16( delinput_libxsmm, naive_libxsmm_delinput_bf16, 1, nImg, nIFm, bn, bc );
      libxsmm_convert_bf16_f32( naive_libxsmm_delinput_bf16, naive_libxsmm_delinput_f32, nImg*nIFm );

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nImg*nIFm, 1, naive_delinput, naive_libxsmm_delinput_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      if ( (fuse_type == 1) || (fuse_type == 4) || (fuse_type == 5) ) {
        /* copy out data */
        matrix_copy_NCNC_to_NC_bf16( delbias_libxsmm, naive_libxsmm_delbias_bf16, 1, 1, nOFm, 1, nOFm );
        libxsmm_convert_bf16_f32( naive_libxsmm_delbias_bf16, naive_libxsmm_delbias_f32, nOFm );
        libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, nOFm, 1, naive_delbias, naive_libxsmm_delbias_f32, 0, 0);
        printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
        printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
        printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
        libxsmm_matdiff_reduce(&diff, &norms_bwd);
      }

      /* copy out data */
      matrix_copy_KCCK_to_KC_bf16( delfilter_libxsmm, naive_libxsmm_delfilter_bf16, nIFm, nOFm, bc, bk );
      libxsmm_convert_bf16_f32( naive_libxsmm_delfilter_bf16, naive_libxsmm_delfilter_f32, nIFm*nOFm );

      /* compare */
      libxsmm_matdiff(&norms_upd, LIBXSMM_DATATYPE_F32, nIFm*nOFm, 1, naive_delfilter, naive_libxsmm_delfilter_f32, 0, 0);
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd);
    }

    if (type == 'A' || type == 'F') {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gflop = (2.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);

      char tune_string_fwd[1000];
      sprintf(tune_string_fwd,"threads=%d_2D=%d_rows=%d_cols=%d_BN=%d_BK=%d_BC=%d_BFACCUM=%d",nThreads, fwd_2d_blocking, fwd_row_teams, fwd_column_teams, bn, bk, bc, fwd_bf);

      printf("PERFDUMP,%s,FP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n",tune_string_fwd, LIBXSMM_VERSION, nThreads, nImg, nIFm,
          nOFm, ((double)(l_total/iters)), gflop/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
          norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }
    if (type == 'A' || type == 'B') {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gflop = (2.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);

      char tune_string_bwd[1000];
      sprintf(tune_string_bwd,"threads=%d_2D=%d_rows=%d_cols=%d_BN=%d_BK=%d_BC=%d_BFACCUM=%d",nThreads, bwd_2d_blocking, bwd_row_teams, bwd_column_teams, bn, bk, bc, bwd_bf);
      printf("PERFDUMP,%s,BP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", tune_string_bwd , LIBXSMM_VERSION, nThreads, nImg, nIFm,
          nOFm, ((double)(l_total/iters)), gflop/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
          norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }
    if (type == 'A' || type == 'U') {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gflop = (2.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);

      char tune_string_upd[1000];
      sprintf(tune_string_upd,"threads=%d_2D=%d_rows=%d_cols=%d_BN=%d_BK=%d_BC=%d_BFACCUM=%d_IFMSUBTASK=%d_OFMSUBTASK=%d",nThreads, upd_2d_blocking, upd_row_teams, upd_column_teams, bn, bk, bc, upd_bf, ifm_subtasks, ofm_subtasks);
      printf("PERFDUMP,%s,UP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", tune_string_upd , LIBXSMM_VERSION, nThreads, nImg, nIFm,
          nOFm, ((double)(l_total/iters)), gflop/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
          norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    if (type == 'A' || type == 'M') {
      printf("##########################################\n");
      printf("# Performance - BWDUPD (custom-Storage)  #\n");
      printf("##########################################\n");
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      gflop = (4.0*(double)nImg*(double)nIFm*(double)nOFm*(double)iters) / (1000*1000*1000);

      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);

      printf("PERFDUMP,UP,%s,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIFm,
          nOFm, ((double)(l_total/iters)), gflop/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
          norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_handle ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_handle, LIBXSMM_DNN_RELU_MASK ) );


    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_deloutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delfilter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_relumask ) );

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_delinput);
  libxsmm_free(naive_deloutput);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_delfilter);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_delinput_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_deloutput_bf16);
  libxsmm_free(naive_filter_bf16);
  libxsmm_free(naive_delfilter_bf16);
  libxsmm_free(naive_libxsmm_output_bf16);
  libxsmm_free(naive_libxsmm_delinput_bf16);
  libxsmm_free(naive_libxsmm_delfilter_bf16);
  libxsmm_free(naive_libxsmm_output_f32);
  libxsmm_free(naive_libxsmm_delinput_f32);
  libxsmm_free(naive_libxsmm_delfilter_f32);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(delinput_libxsmm);
  libxsmm_free(deloutput_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(delfilter_libxsmm);
  libxsmm_free(naive_bias);
  libxsmm_free(naive_delbias);
  libxsmm_free(naive_bias_bf16);
  libxsmm_free(naive_delbias_bf16);
  libxsmm_free(naive_libxsmm_delbias_bf16);
  libxsmm_free(naive_libxsmm_delbias_f32);
  libxsmm_free(relumask_libxsmm);
  libxsmm_free(bias_libxsmm);
  libxsmm_free(delbias_libxsmm);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return global_status;
}

