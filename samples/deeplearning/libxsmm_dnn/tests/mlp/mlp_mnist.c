/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include <libxsmm_dnn.h>
#include <dnn_common.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define TEST_ACCURACY

/* include c-based dnn library */
#include "../../datasets/mnist/mnist.h"

LIBXSMM_INLINE void my_init_buf_mlp(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

LIBXSMM_INLINE void my_init_buf_mlp_bf16(libxsmm_bfloat16* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_bf16(buf, size);
  for (i = 0; i < (int)size; ++i) {
    libxsmm_bfloat16_hp tmp;
    tmp.f = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
    buf[i] = tmp.i[1];
  }
}

int main(int argc, char* argv[])
{
  float **act_libxsmm, **fil_libxsmm, **delact_libxsmm, **delfil_libxsmm;
  float **bias_libxsmm, **delbias_libxsmm;
  libxsmm_bfloat16 **act_libxsmm_bf16, **fil_libxsmm_bf16, **delact_libxsmm_bf16, **delfil_libxsmm_bf16;
  libxsmm_bfloat16 **bias_libxsmm_bf16, **delbias_libxsmm_bf16;
  unsigned char **relumask_libxsmm;
  int *label_libxsmm;
  libxsmm_datatype in_dt, out_dt, comp_dt;
  libxsmm_dnn_fc_eltw_fuse my_fuse;
  libxsmm_dnn_fc_fwd_config* libxsmm_dnn_fc_fwd;
  libxsmm_dnn_fc_bwd_config* libxsmm_dnn_fc_bwd;
  libxsmm_dnn_opt_config* libxsmm_dnn_opt;
  libxsmm_dnn_smax_fwd_config libxsmm_dnn_smax_fwd;
  libxsmm_dnn_smax_bwd_config libxsmm_dnn_smax_bwd;
  void* scratch = NULL;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int MB = 256;          /* mini-batch size, "N" */
  int bn = 32;
  int bk = 32;
  int bc = 32;
  int *C;               /* number of input feature maps, "C" */
  int num_layers = 0;
  int prec_bf16 = 0;

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gflop = 0.0;
  int i, j;
  double fil_size = 0.0;
  double act_size = 0.0;
  float lr = 0.1f;
  float loss_weight = 1.0f;
  float loss = 0.0;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  act_libxsmm = NULL;
  fil_libxsmm = NULL;
  delact_libxsmm = NULL;
  delfil_libxsmm = NULL;
  bias_libxsmm = NULL;
  delbias_libxsmm = NULL;
  act_libxsmm_bf16 = NULL;
  fil_libxsmm_bf16 = NULL;
  delact_libxsmm_bf16 = NULL;
  delfil_libxsmm_bf16 = NULL;
  bias_libxsmm_bf16 = NULL;
  delbias_libxsmm_bf16 = NULL;
  relumask_libxsmm = NULL;
  label_libxsmm = NULL;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters MB bn bk bc prec_bf16 [C1 C2 ... CN-1]\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  num_layers = ( argc > 7 ) ? (argc - 8) + 2 : 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) MB         = atoi(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);
  if (argc > i) prec_bf16  = atoi(argv[i++]);
  /* allocate the number of channles buffer */
  C = (int*)malloc((num_layers+2)*sizeof(int));
  C[0] = 784;
  if ( argc > 6 ) {
    for (j = 1 ; i < argc; ++i, ++j ) {
      C[j] = atoi(argv[i]);
    }
  } else {
    j=1;
  }
  C[j] = 10;

  /* handle softmax config */
  C[num_layers+1] = C[num_layers];

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  if ( prec_bf16 > 0 ) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
    comp_dt = LIBXSMM_DATATYPE_BF16;
  } else {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
    comp_dt = LIBXSMM_DATATYPE_F32;
  }

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d\n", MB);
  printf("PARAMS: Layers: %d\n", num_layers);
  printf("PARAMS: ITERS:%d", iters); printf("  Threads:%d\n", nThreads);
  for (i = 0; i < num_layers; ++i ) {
    if (i == 0) {
      act_size += (double)(MB*C[i]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0);
      printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i, MB, C[i], (double)(MB*C[i]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0) );
    }
    act_size += (double)(MB*C[i+1]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0);
    fil_size += (double)(C[i]*C[i+1]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0);
    printf("SIZE Filter       %i (%dx%d): %10.2f MiB\n", i, C[i], C[i+1], (double)(C[i]*C[i+1]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0) );
    printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i+1, MB, C[i+1], (double)(MB*C[i+1]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0) );
  }
  act_size += (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0);
  printf("SIZE Activations softmax (%dx%d): %10.2f MiB\n", MB, C[num_layers+1], (double)(MB*C[num_layers+1]*LIBXSMM_TYPESIZE(in_dt))/(1024.0*1024.0) );
  printf("\nTOTAL SIZE Activations:    %10.2f MiB\n", act_size );
  printf("TOTAL SIZE Filter:         %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE delActivations: %10.2f MiB\n", act_size );
  printf("TOTAL SIZE delFilter:      %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE MLP:            %10.2f MiB\n", (2.0*fil_size) + (2.0*act_size) );

  /* allocate data */
  /* +2 because of the softwax layer */
  if ( prec_bf16 == 0 ) {
    act_libxsmm    = (float**)malloc( (num_layers+2)*sizeof(float*) );
    delact_libxsmm = (float**)malloc( (num_layers+1)*sizeof(float*) );
    for ( i = 0 ; i < num_layers+2; ++i ) {
      act_libxsmm[i]                = (float*)libxsmm_aligned_malloc( MB*C[i]*sizeof(float), 2097152);
      /* softmax has no incoming gradients */
      if ( i < num_layers+1 ) {
        delact_libxsmm[i]             = (float*)libxsmm_aligned_malloc( MB*C[i]*sizeof(float), 2097152);
      }
    }
    fil_libxsmm    = (float**)malloc( num_layers*sizeof(float*) );
    delfil_libxsmm = (float**)malloc( num_layers*sizeof(float*) );
    for ( i = 0 ; i < num_layers; ++i ) {
      fil_libxsmm[i]                = (float*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
      delfil_libxsmm[i]             = (float*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
    }
    bias_libxsmm    = (float**)malloc( num_layers*sizeof(float*) );
    delbias_libxsmm = (float**)malloc( num_layers*sizeof(float*) );
    for ( i = 0 ; i < num_layers; ++i ) {
      bias_libxsmm[i]               = (float*)libxsmm_aligned_malloc( C[i+1]*sizeof(float), 2097152);
      delbias_libxsmm[i]            = (float*)libxsmm_aligned_malloc( C[i+1]*sizeof(float), 2097152);
    }
    relumask_libxsmm = (unsigned char**)malloc( num_layers*sizeof(unsigned char*) );
    for ( i = 0 ; i < num_layers; ++i ) {
      relumask_libxsmm[i]           = (unsigned char*)libxsmm_aligned_malloc( MB*C[i+1]*sizeof(unsigned char), 2097152);
    }
    label_libxsmm = (int*)libxsmm_aligned_malloc( MB*sizeof(int), 2097152);

    /* init data */
    for ( i = 0 ; i < num_layers+2; ++i ) {
      my_init_buf_mlp( act_libxsmm[i], MB*C[i], 0, 0 );
    }
    for ( i = 0 ; i < num_layers+1; ++i ) {
      my_init_buf_mlp( delact_libxsmm[i], MB*C[i], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp( fil_libxsmm[i], C[i]*C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp( delfil_libxsmm[i], C[i]*C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp( bias_libxsmm[i], C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp( delbias_libxsmm[i], C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      zero_buf_uint8( relumask_libxsmm[i], MB*C[i+1] );
    }
    zero_buf_int32( label_libxsmm, MB );
  } else {
    /* allocate data */
    act_libxsmm_bf16    = (libxsmm_bfloat16**)malloc( (num_layers+2)*sizeof(libxsmm_bfloat16*) );
    delact_libxsmm_bf16 = (libxsmm_bfloat16**)malloc( (num_layers+1)*sizeof(libxsmm_bfloat16*) );
    for ( i = 0 ; i < num_layers+2; ++i ) {
      act_libxsmm_bf16[i]                = (libxsmm_bfloat16*)libxsmm_aligned_malloc( MB*C[i]*sizeof(libxsmm_bfloat16), 2097152);
      /* softmax has no incoming gradients */
      if ( i < num_layers+1 ) {
        delact_libxsmm_bf16[i]             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( MB*C[i]*sizeof(libxsmm_bfloat16), 2097152);
      }
    }
    fil_libxsmm         = (float**)           malloc( num_layers*sizeof(float*) );
    fil_libxsmm_bf16    = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
    delfil_libxsmm_bf16 = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
    for ( i = 0 ; i < num_layers; ++i ) {
      fil_libxsmm[i]         = (float*)           libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
      fil_libxsmm_bf16[i]    = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
      delfil_libxsmm_bf16[i] = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
    }
    bias_libxsmm_bf16    = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
    delbias_libxsmm_bf16 = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
    for ( i = 0 ; i < num_layers; ++i ) {
      bias_libxsmm_bf16[i]    = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
      delbias_libxsmm_bf16[i] = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
    }
    relumask_libxsmm = (unsigned char**)malloc( num_layers*sizeof(unsigned char*) );
    for ( i = 0 ; i < num_layers; ++i ) {
      relumask_libxsmm[i]           = (unsigned char*)libxsmm_aligned_malloc( MB*C[i+1]*sizeof(unsigned char), 2097152);
    }
    label_libxsmm = (int*)libxsmm_aligned_malloc( MB*sizeof(int), 2097152);

    /* init data */
    for ( i = 0 ; i < num_layers+2; ++i ) {
      my_init_buf_mlp_bf16( act_libxsmm_bf16[i], MB*C[i], 0, 0 );
    }
    for ( i = 0 ; i < num_layers+1; ++i ) {
      my_init_buf_mlp_bf16( delact_libxsmm_bf16[i], MB*C[i], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
#if 0
      {
        float *cur_fil = (float*) malloc(C[i]*C[i+1]*sizeof(float));
        my_init_buf( cur_fil, C[i]*C[i+1], 0, 0 );
        my_matrix_copy_KCCK_to_KCCK_vnni(cur_fil, fil_libxsmm[i], C[i], C[i+1], bc, bk);
        libxsmm_rne_convert_fp32_bf16( fil_libxsmm[i], fil_libxsmm_bf16[i], C[i]*C[i+1] );
        free(cur_fil);
       }
#else
       my_init_buf_mlp( fil_libxsmm[i], C[i]*C[i+1], 0, 0 );
       libxsmm_rne_convert_fp32_bf16( fil_libxsmm[i], fil_libxsmm_bf16[i], C[i]*C[i+1] );
#endif
    }
    for ( i = 0 ; i < num_layers; ++i ) {
#if 0
      float *cur_fil = (float*) malloc(C[i]*C[i+1]*sizeof(float));
      float *cur_fil_vnni = (float*) malloc(C[i]*C[i+1]*sizeof(float));
      my_init_buf( cur_fil, C[i]*C[i+1], 0, 0 );
      my_matrix_copy_KCCK_to_KCCK_vnni(cur_fil, cur_fil_vnni, C[i], C[i+1], bc, bk);
      libxsmm_rne_convert_fp32_bf16( cur_fil_vnni, delfil_libxsmm_bf16[i], C[i]*C[i+1] );
      free(cur_fil);
      free(cur_fil_vnni);
#else
      my_init_buf_mlp_bf16( delfil_libxsmm_bf16[i], C[i]*C[i+1], 0, 0 );
#endif
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp_bf16( bias_libxsmm_bf16[i], C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp_bf16( delbias_libxsmm_bf16[i], C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      zero_buf_uint8( relumask_libxsmm[i], MB*C[i+1] );
    }
    zero_buf_int32( label_libxsmm, MB );
  }

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  /* allocating handles */
  libxsmm_dnn_fc_fwd = (libxsmm_dnn_fc_fwd_config*) malloc( num_layers*sizeof(libxsmm_dnn_fc_fwd_config) );
  libxsmm_dnn_fc_bwd = (libxsmm_dnn_fc_bwd_config*) malloc( num_layers*sizeof(libxsmm_dnn_fc_bwd_config) );
  libxsmm_dnn_opt    = (libxsmm_dnn_opt_config*)    malloc( num_layers*sizeof(libxsmm_dnn_opt_config)    );

  /* setting up handles + scratch */
  for ( i = 0; i < num_layers; ++i ) {
    /* MNIST Specific where everywhere we use relu act except the last layer */
    if ( i < num_layers -1) {
      my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK;
    } else {
      my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_NONE;
    }
    libxsmm_dnn_fc_fwd[i] = setup_libxsmm_dnn_fc_fwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                             nThreads, my_fuse, in_dt, out_dt, comp_dt );

    libxsmm_dnn_fc_bwd[i] = setup_libxsmm_dnn_fc_bwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                             nThreads, my_fuse, in_dt, out_dt, comp_dt );

    libxsmm_dnn_opt[i] = setup_libxsmm_dnn_opt( C[i], C[i+1], (C[i  ] % bc == 0) ? bc : C[i  ],
                                            (C[i+1] % bk == 0) ? bk : C[i+1],
                                            nThreads, lr, in_dt, out_dt, comp_dt );

    /* let's allocate and bind scratch */
    if ( libxsmm_dnn_fc_fwd[i].scratch_size > 0 || libxsmm_dnn_fc_bwd[i].scratch_size > 0 || libxsmm_dnn_opt[i].scratch_size > 0 ) {
      size_t alloc_size = LIBXSMM_MAX( LIBXSMM_MAX( libxsmm_dnn_fc_fwd[i].scratch_size, libxsmm_dnn_fc_bwd[i].scratch_size), libxsmm_dnn_opt[i].scratch_size );
      if ( alloc_size > scratch_size ) {
        if ( scratch != NULL ) libxsmm_free( scratch );
        scratch_size = alloc_size;
        scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
        my_init_buf_mlp( (float*)(scratch), (scratch_size)/4, 0, 0 );
      }
    }
  }

  /* softmax+loss is treated as N+1 layer */
  libxsmm_dnn_smax_fwd = setup_libxsmm_dnn_smax_fwd( MB, C[num_layers+1], (MB % bn == 0) ? bn : MB,
                                       (C[num_layers+1] % bk == 0) ? bk : C[num_layers+1],
                                       nThreads, in_dt, out_dt, comp_dt );

  libxsmm_dnn_smax_bwd = setup_libxsmm_dnn_smax_bwd( MB, C[num_layers+1], (MB % bn == 0) ? bn : MB,
                                       (C[num_layers+1] % bk == 0) ? bk : C[num_layers+1],
                                       nThreads, loss_weight, in_dt, out_dt, comp_dt );

  if ( libxsmm_dnn_smax_fwd.scratch_size > 0 || libxsmm_dnn_smax_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( libxsmm_dnn_smax_fwd.scratch_size, libxsmm_dnn_smax_bwd.scratch_size );
    if ( alloc_size > scratch_size ) {
      if ( scratch != NULL ) libxsmm_free( scratch );
      scratch_size = alloc_size;
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      my_init_buf_mlp( (float*)(scratch), (scratch_size)/4, 0, 0 );
    }
  }

  /* Reading in the MNIST dataset */
  int n_batches  = NUM_TRAIN/MB, batch_id = 0;
  int n_epochs = iters, epoch_id = 0;
  float *input_acts = NULL;
  libxsmm_bfloat16 *input_acts_bf16 = NULL;

  if ( prec_bf16 == 0 ) {
    input_acts = (float*)libxsmm_aligned_malloc( NUM_TRAIN * C[0] * sizeof(float), 2097152);
  } else {
    input_acts_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( NUM_TRAIN * C[0] * sizeof(libxsmm_bfloat16), 2097152);
  }

  /* Read in input data */
  char *train_image_path = "../../datasets/mnist/train-images.idx3-ubyte";
  char *train_label_path = "../../datasets/mnist/train-labels.idx1-ubyte";
  char *test_image_path = "../../datasets/mnist/t10k-images.idx3-ubyte";
  char *test_label_path = "../../datasets/mnist/t10k-labels.idx1-ubyte";
  load_mnist(train_image_path, train_label_path, test_image_path, test_label_path);

  /* Format the input layer in NCNC blocked format  */
  int  _i, _j;
  for (_i = 0; _i < n_batches*MB; _i++) {
    for (_j = 0; _j < C[0]; _j++) {
      float val = (float) train_image[_i][_j];
      int batchid = _i/MB;
      int mb = _i % MB;
      int _bn = (MB % bn == 0) ? bn : MB;
      int _bc = (C[0] % bc == 0) ? bc : C[0];
      if ( prec_bf16 == 0 ) {
        float *cur_pos = input_acts + batchid * MB *C[0] + (mb / _bn) * C[0] * _bn + (_j / _bc) * _bn * _bc + (mb % _bn) * _bc + (_j % _bc);
        *cur_pos = val;
      } else {
        libxsmm_bfloat16 *cur_pos = input_acts_bf16 + batchid * MB *C[0] + (mb / _bn) * C[0] * _bn + (_j / _bc) * _bn * _bc + (mb % _bn) * _bc + (_j % _bc);
        libxsmm_rne_convert_fp32_bf16( &val,     cur_pos,    1 );
      }
    }
  }

  printf("###########################################\n");
  printf("# Training MNIST with %d training samples #\n", n_batches*MB);
  printf("###########################################\n");
  l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j,epoch_id,batch_id)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for (epoch_id = 0; epoch_id < n_epochs; epoch_id++) {
      for (batch_id = 0; batch_id < n_batches; batch_id++) {
        if ( prec_bf16 == 0 ) {
          for ( i = 0; i < num_layers; ++i) {
            float *input_act_ptr = (i == 0) ? input_acts + batch_id * MB * C[0] : act_libxsmm[i];
            libxsmm_dnn_fc_fwd_exec_f32( libxsmm_dnn_fc_fwd[i], fil_libxsmm[i], input_act_ptr, act_libxsmm[i+1],
                                bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch );
          }
          libxsmm_dnn_smax_fwd_exec_f32( libxsmm_dnn_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], train_label + batch_id * MB, &loss,
                                0, tid, scratch );

          if ((tid == 0) && (batch_id == 0) && (epoch_id % 10 == 0 || epoch_id == n_epochs - 1 )) {
            printf("Loss for epoch %d batch_id %d is %f\n", epoch_id, batch_id, loss);
          }

          libxsmm_dnn_smax_bwd_exec_f32( libxsmm_dnn_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], train_label + batch_id * MB,
                                0, tid, scratch );
          for ( i = num_layers-1; i > 0; --i) {
            libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd[i], fil_libxsmm[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                                act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], LIBXSMM_DNN_FC_PASS_BWD, 0, tid, scratch );
            libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[i], fil_libxsmm[i], delfil_libxsmm[i], 0, tid, scratch );
          }
          libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd[0], fil_libxsmm[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                          input_acts + batch_id * MB * C[0], delbias_libxsmm[0], relumask_libxsmm[0], LIBXSMM_DNN_FC_PASS_BWD_W, 0, tid, scratch );
          libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[0], fil_libxsmm[0], delfil_libxsmm[0], 0, tid, scratch );
        } else {
          for ( i = 0; i < num_layers; ++i) {
            libxsmm_bfloat16 *input_act_ptr = (i == 0) ? input_acts_bf16 + batch_id * MB * C[0] : act_libxsmm_bf16[i];
            libxsmm_dnn_fc_fwd_exec_bf16( libxsmm_dnn_fc_fwd[i], fil_libxsmm_bf16[i], input_act_ptr, act_libxsmm_bf16[i+1],
                                 bias_libxsmm_bf16[i], relumask_libxsmm[i], 0, tid, scratch );
          }
          libxsmm_dnn_smax_fwd_exec_bf16( libxsmm_dnn_smax_fwd, act_libxsmm_bf16[num_layers], act_libxsmm_bf16[num_layers+1], train_label + batch_id * MB, &loss,
                                 0, tid, scratch );

          if ((tid == 0) && (batch_id == 0) && (epoch_id % 10 == 0 || epoch_id == n_epochs - 1 )) {
            printf("Loss for epoch %d batch_id %d is %f\n", epoch_id, batch_id, loss);
          }

          libxsmm_dnn_smax_bwd_exec_bf16( libxsmm_dnn_smax_bwd, delact_libxsmm_bf16[num_layers], act_libxsmm_bf16[num_layers+1], train_label + batch_id * MB,
                                 0, tid, scratch );
          for ( i = num_layers-1; i > 0; --i) {
            libxsmm_dnn_fc_bwd_exec_bf16( libxsmm_dnn_fc_bwd[i], fil_libxsmm_bf16[i], delact_libxsmm_bf16[i], delact_libxsmm_bf16[i+1], delfil_libxsmm_bf16[i],
                                 act_libxsmm_bf16[i], delbias_libxsmm_bf16[i], relumask_libxsmm[i], LIBXSMM_DNN_FC_PASS_BWD, 0, tid, scratch );
            libxsmm_dnn_opt_exec_bf16( libxsmm_dnn_opt[i], fil_libxsmm_bf16[i], fil_libxsmm[i], delfil_libxsmm_bf16[i], 0, tid, scratch );
          }
          libxsmm_dnn_fc_bwd_exec_bf16( libxsmm_dnn_fc_bwd[0], fil_libxsmm_bf16[0], delact_libxsmm_bf16[0], delact_libxsmm_bf16[0+1], delfil_libxsmm_bf16[0],
                               input_acts_bf16 + batch_id * MB * C[0], delbias_libxsmm_bf16[0], relumask_libxsmm[0], LIBXSMM_DNN_FC_PASS_BWD_W, 0, tid, scratch );
          libxsmm_dnn_opt_exec_bf16( libxsmm_dnn_opt[0], fil_libxsmm_bf16[0], fil_libxsmm[0], delfil_libxsmm_bf16[0], 0, tid, scratch );
        }
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);

  gflop = 0.0;

  for ( i = num_layers-1; i > 0; --i) {
    gflop += (6.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)((double)n_epochs *(double)n_batches)) / (1000.0*1000.0*1000.0);
  }
  gflop += (4.0*(double)MB*(double)C[0]*(double)C[1]*(double)((double)n_epochs *(double)n_batches)) / (1000.0*1000.0*1000.0);
  printf("GFLOP  = %.5g\n", gflop/(double)((double)n_epochs *(double)n_batches));
  printf("fp time = %.5g\n", ((double)(l_total/((double)n_epochs *(double)n_batches))));
  printf("GFLOPS  = %.5g\n", gflop/l_total);
  printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
  for ( i = 0; i < num_layers; ++i ) {
    printf("%i,", C[i] );
  }
  printf("%f,%f\n", ((double)(l_total/((double)n_epochs *(double)n_batches))), gflop/l_total);

#ifdef TEST_ACCURACY
  /* Test accuracy */
  n_batches  = NUM_TEST/MB;

  for (_i = 0; _i < n_batches * MB; _i++) {
    for (_j = 0; _j < C[0]; _j++) {
      float val = (float) test_image[_i][_j];
      int batchid = _i/MB;
      int mb = _i % MB;
      int _bn = (MB % bn == 0) ? bn : MB;
      int _bc = (C[0] % bc == 0) ? bc : C[0];
      if ( prec_bf16 == 0 ) {
        float *cur_pos = input_acts + batchid * MB *C[0] + (mb / _bn) * C[0] * _bn + (_j / _bc) * _bn * _bc + (mb % _bn) * _bc + (_j % _bc);
        *cur_pos = val;
      } else {
        libxsmm_bfloat16 *cur_pos = input_acts_bf16 + batchid * MB *C[0] + (mb / _bn) * C[0] * _bn + (_j / _bc) * _bn * _bc + (mb % _bn) * _bc + (_j % _bc);
        libxsmm_rne_convert_fp32_bf16( &val,     cur_pos,    1 );
      }
    }
  }

  n_batches  = NUM_TEST/MB;
  unsigned int hits = 0;
  unsigned int samples = 0;

#if defined(_OPENMP)
#   pragma omp parallel private(i,j,batch_id)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for (batch_id = 0; batch_id < n_batches; batch_id++) {
      if ( prec_bf16 == 0 ) {
        for ( i = 0; i < num_layers; ++i) {
          float *input_act_ptr = (i == 0) ? input_acts + batch_id * MB * C[0] : act_libxsmm[i];
          libxsmm_dnn_fc_fwd_exec_f32( libxsmm_dnn_fc_fwd[i], fil_libxsmm[i], input_act_ptr, act_libxsmm[i+1],
                              bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch );
        }
        libxsmm_dnn_smax_fwd_exec_f32( libxsmm_dnn_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], test_label + batch_id * MB, &loss,
                              0, tid, scratch );

        if (tid == 0) {
          for (_i = 0; _i < MB; _i++) {
            int label =  *(test_label + batch_id * MB + _i);
            int max_id = 0;
            float max_val = 0.0;
            max_val = *(act_libxsmm[num_layers+1] +  _i * 10);
            float sum = max_val;
            /* Find predicted label */
            for (_j = 1; _j < 10; _j++) {
              float val = *(act_libxsmm[num_layers+1] + _i * 10  + _j);
              sum += val;
              if (val > max_val) {
                max_id = _j;
                max_val = val;
              }
            }
            /* Compare with true label */
            if (max_id == label) {
              hits++;
            }
            samples++;
          }
        }
        #pragma omp barrier
      } else {
        for ( i = 0; i < num_layers; ++i) {
          libxsmm_bfloat16 *input_act_ptr = (i == 0) ? input_acts_bf16 + batch_id * MB * C[0] : act_libxsmm_bf16[i];
          libxsmm_dnn_fc_fwd_exec_bf16( libxsmm_dnn_fc_fwd[i], fil_libxsmm_bf16[i], input_act_ptr, act_libxsmm_bf16[i+1],
                               bias_libxsmm_bf16[i], relumask_libxsmm[i], 0, tid, scratch );
        }
        libxsmm_dnn_smax_fwd_exec_bf16( libxsmm_dnn_smax_fwd, act_libxsmm_bf16[num_layers], act_libxsmm_bf16[num_layers+1], test_label + batch_id * MB, &loss,
                               0, tid, scratch );

        if (tid == 0) {
          for (_i = 0; _i < MB; _i++) {
            int label =  *(test_label + batch_id * MB + _i);
            int max_id = 0;
            float max_val = 0.0;
            libxsmm_convert_bf16_f32( act_libxsmm_bf16[num_layers+1] +  _i * 10, &max_val, 1 );
            /* Find predicted label */
            for (_j = 1; _j < 10; _j++) {
              libxsmm_bfloat16 val = *(act_libxsmm_bf16[num_layers+1] + _i * 10  + _j);
              float f32_val;
              libxsmm_convert_bf16_f32( &val, &f32_val, 1 );
              if (f32_val > max_val) {
                max_id = _j;
                max_val = f32_val;
              }
            }
            /* Compare with true label */
            if (max_id == label) {
              hits++;
            }
            samples++;
          }
        }
        #pragma omp barrier
      }
    }
  }

  printf("Accuracy is %f %% (%d test samples)\n", (1.0*hits)/(1.0*samples)*100.0, samples);
#endif

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
  if ( prec_bf16 > 0 ) {
    for ( i = 0; i < num_layers; ++i ) {
      if ( i == 0 ) {
        libxsmm_free(act_libxsmm_bf16[i]);
        libxsmm_free(delact_libxsmm_bf16[i]);
      }
      libxsmm_free(act_libxsmm_bf16[i+1]);
      libxsmm_free(delact_libxsmm_bf16[i+1]);

      libxsmm_free(fil_libxsmm_bf16[i]);
      libxsmm_free(fil_libxsmm[i]);
      libxsmm_free(delfil_libxsmm_bf16[i]);
      libxsmm_free(bias_libxsmm_bf16[i]);
      libxsmm_free(delbias_libxsmm_bf16[i]);
      libxsmm_free(relumask_libxsmm[i]);
    }
    libxsmm_free(act_libxsmm_bf16[num_layers+1]);
    libxsmm_free(label_libxsmm);
    libxsmm_free(input_acts_bf16);

    free( act_libxsmm_bf16 );
    free( delact_libxsmm_bf16 );
    free( fil_libxsmm_bf16 );
    free( fil_libxsmm );
    free( delfil_libxsmm_bf16 );
    free( bias_libxsmm_bf16 );
    free( delbias_libxsmm_bf16 );
    free( relumask_libxsmm );
  } else {
    for ( i = 0; i < num_layers; ++i ) {
      if ( i == 0 ) {
        libxsmm_free(act_libxsmm[i]);
        libxsmm_free(delact_libxsmm[i]);
      }
      libxsmm_free(act_libxsmm[i+1]);
      libxsmm_free(delact_libxsmm[i+1]);

      libxsmm_free(fil_libxsmm[i]);
      libxsmm_free(delfil_libxsmm[i]);
      libxsmm_free(bias_libxsmm[i]);
      libxsmm_free(delbias_libxsmm[i]);
      libxsmm_free(relumask_libxsmm[i]);
    }
    libxsmm_free(act_libxsmm[num_layers+1]);
    libxsmm_free(label_libxsmm);
    libxsmm_free(input_acts);

    free( act_libxsmm );
    free( delact_libxsmm );
    free( fil_libxsmm );
    free( delfil_libxsmm );
    free( bias_libxsmm );
    free( delbias_libxsmm );
    free( relumask_libxsmm );
  }

  free( libxsmm_dnn_opt );
  free( libxsmm_dnn_fc_fwd );
  free( libxsmm_dnn_fc_bwd );

  free( C );

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

