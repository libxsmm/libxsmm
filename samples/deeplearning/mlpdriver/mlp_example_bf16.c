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

int main(int argc, char* argv[])
{
  libxsmm_bfloat16 **act_libxsmm, **fil_libxsmm, **delact_libxsmm, **delfil_libxsmm;
  libxsmm_bfloat16 **bias_libxsmm, **delbias_libxsmm;
  float **fil_master;
  unsigned char **relumask_libxsmm;
  void* scratch = NULL;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;       /* repetitions of benchmark */
  int MB = 32;          /* mini-batch size, "N" */
  int fuse_type = 0;    /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';      /* 'A': ALL, 'F': FP, 'B': BP */
  int bn = 64;
  int bk = 64;
  int bc = 64;
  int *C;               /* number of input feature maps, "C" */
  int num_layers = 0;

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
  int i, j;
  double act_size = 0.0;
  double fil_size = 0.0;

  libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
  libxsmm_dnn_fullyconnected**    libxsmm_fc_layer;
  libxsmm_dnn_optimizer_desc      optimizer_desc;
  libxsmm_dnn_optimizer**         libxsmm_opt;
  libxsmm_dnn_softmaxloss_desc    softmaxloss_desc;
  libxsmm_dnn_softmaxloss*        libxsmm_softmax;
  libxsmm_dnn_tensor**            libxsmm_act;
  libxsmm_dnn_tensor**            libxsmm_delact;
  libxsmm_dnn_tensor**            libxsmm_fil;
  libxsmm_dnn_tensor**            libxsmm_delfil;
  libxsmm_dnn_tensor**            libxsmm_bias;
  libxsmm_dnn_tensor**            libxsmm_delbias;
  libxsmm_dnn_tensor**            libxsmm_relumask;
  libxsmm_dnn_tensor**            libxsmm_mafil;
  libxsmm_dnn_tensor_datalayout*  libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  num_layers = argc - 9;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) MB         = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);
  /* allocate the number of channles buffer */
  if ( num_layers < 1 ) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  C = (int*)malloc((num_layers+2)*sizeof(int));
  for (j = 0 ; i < argc; ++i, ++j ) {
    C[j] = atoi(argv[i]);
  }
  /* handle softmax config */
  C[num_layers+1] = C[num_layers];

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return -1;
  }
  if ( (fuse_type < 0) || (fuse_type > 5) ) {
    printf("fuse type needs to be 0 (None), 1 (Bias), 2 (ReLU), 3 (Sigmoid), 4 (Bias+ReLU), 5 (Bias+Sigmoid)\n");
    return -1;
  }

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d\n", MB);
  printf("PARAMS: Layers: %d\n", num_layers);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  for (i = 0; i < num_layers; ++i ) {
    if (i == 0) {
      act_size += (double)(MB*C[i]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0);
      printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i, MB, C[i], (double)(MB*C[i]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
    }
    act_size += (double)(MB*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0);
    fil_size += (double)(C[i]*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0);
    printf("SIZE Filter       %i (%dx%d): %10.2f MiB\n", i, C[i], C[i+1], (double)(C[i]*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
    printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i+1, MB, C[i+1], (double)(MB*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  }
  act_size += (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0);
  printf("SIZE Activations softmax (%dx%d): %10.2f MiB\n", MB, C[num_layers+1], (double)(MB*C[num_layers+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("\nTOTAL SIZE Activations:            %10.2f MiB\n", act_size );
  printf("TOTAL SIZE Filter (incl. master):  %10.2f MiB\n", 3.0*fil_size );
  printf("TOTAL SIZE delActivations:         %10.2f MiB\n", act_size );
  printf("TOTAL SIZE delFilter:              %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE MLP:                    %10.2f MiB\n", (4.0*fil_size) + (2.0*act_size) );

  /* allocate data */
  act_libxsmm    = (libxsmm_bfloat16**)malloc( (num_layers+1)*sizeof(libxsmm_bfloat16*) );
  delact_libxsmm = (libxsmm_bfloat16**)malloc( (num_layers+1)*sizeof(libxsmm_bfloat16*) );
  for ( i = 0 ; i < num_layers+2; ++i ) {
    act_libxsmm[i]                = (libxsmm_bfloat16*)libxsmm_aligned_malloc( MB*C[i]*sizeof(libxsmm_bfloat16), 2097152);
    /* softmax has no incoming gradients */
    if ( i < num_layers+1 ) {
      delact_libxsmm[i]             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( MB*C[i]*sizeof(libxsmm_bfloat16), 2097152);
    }
  }
  fil_master     = (float**)           malloc( num_layers*sizeof(float*) );
  fil_libxsmm    = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  delfil_libxsmm = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    fil_master[i]                 = (float*)           libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
    fil_libxsmm[i]                = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
    delfil_libxsmm[i]             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
  }
  bias_libxsmm    = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  delbias_libxsmm = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    bias_libxsmm[i]               = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
    delbias_libxsmm[i]            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
  }
  relumask_libxsmm = (unsigned char**)malloc( num_layers*sizeof(unsigned char*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    relumask_libxsmm[i]           = (unsigned char*)libxsmm_aligned_malloc( MB*C[i+1]*sizeof(unsigned char), 2097152);
  }

  /* init data */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    init_buf_bf16( act_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers+1; ++i ) {
    init_buf_bf16( delact_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    init_buf( fil_master[i], C[i]*C[i+1], 0, 0 );
    libxsmm_rne_convert_fp32_bf16( fil_master[i], fil_libxsmm[i], C[i]*C[i+1] );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    init_buf_bf16( delfil_libxsmm[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    init_buf_bf16( bias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    init_buf_bf16( delbias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    zero_buf_uint8( relumask_libxsmm[i], MB*C[i+1] );
  }

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  libxsmm_fc_layer   = (libxsmm_dnn_fullyconnected**) malloc( num_layers*sizeof(libxsmm_dnn_fullyconnected*) );
  libxsmm_opt      = (libxsmm_dnn_optimizer**) malloc( num_layers*sizeof(libxsmm_dnn_optimizer*) );
  libxsmm_act      = (libxsmm_dnn_tensor**) malloc( (num_layers+2)*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_delact   = (libxsmm_dnn_tensor**) malloc( (num_layers+1)*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_fil      = (libxsmm_dnn_tensor**) malloc( num_layers*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_delfil   = (libxsmm_dnn_tensor**) malloc( num_layers*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_bias     = (libxsmm_dnn_tensor**) malloc( num_layers*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_delbias  = (libxsmm_dnn_tensor**) malloc( num_layers*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_relumask = (libxsmm_dnn_tensor**) malloc( num_layers*sizeof(libxsmm_dnn_tensor*) );
  libxsmm_mafil    = (libxsmm_dnn_tensor**) malloc( num_layers*sizeof(libxsmm_dnn_tensor*) );

  for ( i = 0; i < num_layers; ++i ) {
    fullyconnected_desc.N = MB;
    fullyconnected_desc.C = C[i];
    fullyconnected_desc.K = C[i+1];
    fullyconnected_desc.bn = (MB % bn == 0) ? bn : MB;
    fullyconnected_desc.bc = (C[i  ] % bc == 0) ? bc : C[i  ];
    fullyconnected_desc.bk = (C[i+1] % bk == 0) ? bk : C[i+1];
    fullyconnected_desc.threads = nThreads;
    fullyconnected_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    fullyconnected_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
    fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
    fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;

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

    libxsmm_fc_layer[i] = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    optimizer_desc.C = C[i];
    optimizer_desc.K = C[i+1];
    optimizer_desc.bc = (C[i  ] % bc == 0) ? bc : C[i  ];
    optimizer_desc.bk = (C[i+1] % bk == 0) ? bk : C[i+1];
    optimizer_desc.learning_rate = 0.1f;
    optimizer_desc.threads = nThreads;
    optimizer_desc.opt_type = LIBXSMM_DNN_OPTIMIZER_SGD;
    optimizer_desc.datatype = LIBXSMM_DNN_DATATYPE_BF16;
    optimizer_desc.datatype_master = LIBXSMM_DNN_DATATYPE_F32;
    optimizer_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
    libxsmm_opt[i] = libxsmm_dnn_create_optimizer( optimizer_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers */
    if ( i == 0 ) {
      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_act[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, act_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_delact[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, delact_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    }

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_act[i+1]  = libxsmm_dnn_link_tensor( libxsmm_layout, act_libxsmm[i+1], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delact[i+1]  = libxsmm_dnn_link_tensor( libxsmm_layout, delact_libxsmm[i+1], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_fil[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, fil_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delfil[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, delfil_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_CHANNEL_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias[i] = libxsmm_dnn_link_tensor( libxsmm_layout, bias_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbias[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, delbias_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_fc_layer[i], LIBXSMM_DNN_RELU_MASK, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_relumask[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, relumask_libxsmm[i], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_act[  i],      LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_delact[i  ],   LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_act[i+1],      LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_delact[i+1],   LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_fil[i],        LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_delfil[i],     LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_bias[i],       LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_delbias[i],    LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_fc_layer[i], libxsmm_relumask[i],   LIBXSMM_DNN_RELU_MASK ) );

    libxsmm_layout = libxsmm_dnn_optimizer_create_tensor_datalayout( libxsmm_opt[i], LIBXSMM_DNN_MASTER_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_mafil[i]  = libxsmm_dnn_link_tensor( libxsmm_layout, fil_master[i], &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* bind filters to optimizer */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_bind_tensor( libxsmm_opt[i], libxsmm_fil[i],        LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_bind_tensor( libxsmm_opt[i], libxsmm_mafil[i],      LIBXSMM_DNN_MASTER_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_bind_tensor( libxsmm_opt[i], libxsmm_delfil[i],     LIBXSMM_DNN_GRADIENT_FILTER ) );

    /* let's allocate and bind scratch */
    if ( libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_fc_layer[i], &status ) > scratch_size ) {
      scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_fc_layer[i], &status );
      CHKERR_LIBXSMM_DNN( status );
      if ( scratch != NULL ) {
        libxsmm_free( scratch );
      }
      scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
      init_buf( (float*)scratch, scratch_size/4, 0, 0 );
    }
    if ( libxsmm_dnn_optimizer_get_scratch_size( libxsmm_opt[i], &status ) > scratch_size ) {
      scratch_size = libxsmm_dnn_optimizer_get_scratch_size( libxsmm_opt[i], &status );
      CHKERR_LIBXSMM_DNN( status );
      if ( scratch != NULL ) {
        libxsmm_free( scratch );
      }
      scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
      init_buf( (float*)scratch, scratch_size/4, 0, 0 );
    }
  }

  /* create softmax layer */
  softmaxloss_desc.N = MB;
  softmaxloss_desc.C = C[num_layers];
  softmaxloss_desc.bn = (MB % bn == 0) ? bn : MB;
  softmaxloss_desc.bc = (C[num_layers] % bc == 0) ? bc : C[num_layers];
  softmaxloss_desc.loss_weight = 1.0;
  softmaxloss_desc.threads = nThreads;
  softmaxloss_desc.datatype = LIBXSMM_DNN_DATATYPE_BF16;
  softmaxloss_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
  libxsmm_softmax = libxsmm_dnn_create_softmaxloss( softmaxloss_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  libxsmm_layout = libxsmm_dnn_softmaxloss_create_tensor_datalayout( libxsmm_softmax, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_act[num_layers+1]  = libxsmm_dnn_link_tensor( libxsmm_layout, act_libxsmm[num_layers+1], &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_bind_tensor( libxsmm_softmax, libxsmm_act[num_layers],      LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_bind_tensor( libxsmm_softmax, libxsmm_delact[num_layers],   LIBXSMM_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_bind_tensor( libxsmm_softmax, libxsmm_act[num_layers+1],      LIBXSMM_DNN_REGULAR_OUTPUT ) );

  if ( libxsmm_dnn_softmaxloss_get_scratch_size( libxsmm_softmax, &status ) > scratch_size ) {
    scratch_size = libxsmm_dnn_softmaxloss_get_scratch_size( libxsmm_softmax, &status );
    CHKERR_LIBXSMM_DNN( status );
    if ( scratch != NULL ) {
      libxsmm_free( scratch );
    }
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );
  }

  /* bind scratch to all layers */
  for ( i = 0; i < num_layers; ++i ) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_fc_layer[i], scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_bind_scratch(      libxsmm_opt[i],      scratch ) );
  }
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_bind_scratch(    libxsmm_softmax,     scratch ) );

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
        for ( i = 0; i < num_layers; ++i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_fc_layer[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
        libxsmm_dnn_softmaxloss_execute_st( libxsmm_softmax, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = 0; i < num_layers; ++i) {
      gflop += (2.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,FP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  if (type == 'A' || type == 'B') {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
        libxsmm_dnn_softmaxloss_execute_st( libxsmm_softmax, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        for ( i = num_layers-1; i > 0; --i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_fc_layer[i], LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid );
          libxsmm_dnn_optimizer_execute_st( libxsmm_opt[i], 0, tid );
        }
        libxsmm_dnn_fullyconnected_execute_st( libxsmm_fc_layer[0], LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        libxsmm_dnn_optimizer_execute_st( libxsmm_opt[i], 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (4.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (2.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  if (type == 'A') {
    printf("##########################################\n");
    printf("# Performance - FWD-BWD (custom-Storage) #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
        for ( i = 0; i < num_layers; ++i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_fc_layer[i], LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
        libxsmm_dnn_softmaxloss_execute_st( libxsmm_softmax, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        libxsmm_dnn_softmaxloss_execute_st( libxsmm_softmax, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        for ( i = (num_layers-1); i > 0; --i) {
          libxsmm_dnn_fullyconnected_execute_st( libxsmm_fc_layer[i], LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid );
          libxsmm_dnn_optimizer_execute_st( libxsmm_opt[i], 0, tid );
        }
        libxsmm_dnn_fullyconnected_execute_st( libxsmm_fc_layer[0], LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        libxsmm_dnn_optimizer_execute_st( libxsmm_opt[i], 0, tid );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (6.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (4.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  for ( i = 0; i < num_layers; ++i ) {
    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_fc_layer[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( libxsmm_fc_layer[i], LIBXSMM_DNN_RELU_MASK ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_fc_layer[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_release_scratch( libxsmm_opt[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_release_tensor( libxsmm_opt[i], LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_release_tensor( libxsmm_opt[i], LIBXSMM_DNN_MASTER_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_optimizer_release_tensor( libxsmm_opt[i], LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_optimizer( libxsmm_opt[i] ) );
  }
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_release_scratch( libxsmm_softmax ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_release_tensor( libxsmm_softmax, LIBXSMM_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_release_tensor( libxsmm_softmax, LIBXSMM_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_softmaxloss_release_tensor( libxsmm_softmax, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_softmaxloss( libxsmm_softmax ) );

  for ( i = 0; i < num_layers; ++i ) {
    if ( i == 0 ) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_act[i] ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delact[i] ) );
    }
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_act[i+1] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delact[i+1] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_fil[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delfil[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_delbias[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_relumask[i] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_mafil[i] ) );
  }
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_act[num_layers+1] ) );

  /* deallocate data */
  libxsmm_free(scratch);
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
    libxsmm_free(fil_master[i]);
  }
  libxsmm_free(act_libxsmm[num_layers+1]);

  free( libxsmm_act );
  free( libxsmm_delact );
  free( libxsmm_fil );
  free( libxsmm_delfil );
  free( libxsmm_bias );
  free( libxsmm_delbias );
  free( libxsmm_relumask );
  free( libxsmm_mafil );

  free( act_libxsmm );
  free( delact_libxsmm );
  free( fil_master );
  free( fil_libxsmm );
  free( delfil_libxsmm );
  free( bias_libxsmm );
  free( delbias_libxsmm );
  free( relumask_libxsmm );

  free( C );

  /* some empty lines at the end */
  printf("\n\n\n");

  return global_status;
}

