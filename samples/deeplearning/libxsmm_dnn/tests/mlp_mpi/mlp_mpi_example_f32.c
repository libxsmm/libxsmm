/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm_dnn.h>
#include <dnn_common.h>

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define DETAILED_PROFILE
#define N_PROF_THREADS 128

LIBXSMM_INLINE void my_init_buf_mlp(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

int main(int argc, char* argv[])
{
  /* Initialize the MPI environment */
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if(provided < MPI_THREAD_MULTIPLE) {
    printf("The threading support level is lesser than that demanded.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  float **act_libxsmm, **ref_act_libxsmm, **fil_libxsmm, **delact_libxsmm, **ref_delact_libxsmm, **delfil_libxsmm;
  float **bias_libxsmm, **delbias_libxsmm;
  unsigned char **relumask_libxsmm;
  int *label_libxsmm;
  void* scratch = NULL;
  size_t scratch_size = 0;
  libxsmm_matdiff_info norms;
  libxsmm_matdiff_clear(&norms);
  MPI_Request request[2];

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int n_procs = 4;
  int iters = 10;       /* repetitions of benchmark */
  int MB = 32;          /* mini-batch size, "N" */
  int global_MB = 32;
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
  double l_fwd_fc[N_PROF_THREADS];
  double l_bwdupd_fc[N_PROF_THREADS];
  double l_allreduce[N_PROF_THREADS];
  double l_optimizer[N_PROF_THREADS];
  double l_fwd_loss[N_PROF_THREADS];
  double l_bwd_loss[N_PROF_THREADS];
  double first_bwdupd_compute = 0.0;
  double gflop = 0.0;
  int i, j, rank;
  double fil_size = 0.0;
  double act_size = 0.0;
  float lr = 0.2f;
  float loss_weight = 0.1f;

  libxsmm_datatype in_dt, out_dt, comp_dt;
  libxsmm_dnn_fc_eltw_fuse my_fuse;
  libxsmm_dnn_fc_fwd_config* libxsmm_dnn_fc_fwd;
  libxsmm_dnn_fc_bwd_config* libxsmm_dnn_fc_bwd;
  libxsmm_dnn_opt_config* libxsmm_dnn_opt;
  libxsmm_dnn_smax_fwd_config libxsmm_dnn_smax_fwd;
  libxsmm_dnn_smax_bwd_config libxsmm_dnn_smax_bwd;

  for (i = 0; i < N_PROF_THREADS; i++) {
    l_fwd_fc[i] = 0.0;
    l_bwdupd_fc[i] = 0.0;
    l_allreduce[i] = 0.0;
    l_optimizer[i] = 0.0;
    l_fwd_loss[i] = 0.0;
    l_bwd_loss[i] = 0.0;
  }

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  act_libxsmm = NULL;
  fil_libxsmm = NULL;
  delact_libxsmm = NULL;
  delfil_libxsmm = NULL;
  bias_libxsmm = NULL;
  delbias_libxsmm = NULL;
  relumask_libxsmm = NULL;
  label_libxsmm = NULL;

  /* reading new values from cli */
  i = 1;
  num_layers = argc - 9;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) global_MB  = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);

  /* Get the rank of the process */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MB = global_MB / n_procs;

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

  in_dt = LIBXSMM_DATATYPE_F32;
  out_dt = LIBXSMM_DATATYPE_F32;
  comp_dt = LIBXSMM_DATATYPE_F32;

  /* print some summary */
  if (rank == 0 ) {
    printf("##########################################\n");
    printf("#          Setting Up (Common)           #\n");
    printf("##########################################\n");
    printf("PARAMS: N:%d\n", global_MB);
    printf("PARAMS: Layers: %d\n", num_layers);
    printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
    for (i = 0; i < num_layers; ++i ) {
      if (i == 0) {
        act_size += (double)(global_MB*C[i]*sizeof(float))/(1024.0*1024.0);
        printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i, global_MB, C[i], (double)(global_MB*C[i]*sizeof(float))/(1024.0*1024.0) );
      }
      act_size += (double)(global_MB*C[i+1]*sizeof(float))/(1024.0*1024.0);
      fil_size += (double)(C[i]*C[i+1]*sizeof(float))/(1024.0*1024.0);
      printf("SIZE Filter       %i (%dx%d): %10.2f MiB\n", i, C[i], C[i+1], (double)(C[i]*C[i+1]*sizeof(float))/(1024.0*1024.0) );
      printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i+1, global_MB, C[i+1], (double)(global_MB*C[i+1]*sizeof(float))/(1024.0*1024.0) );
    }
    act_size += (double)(global_MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0);
    printf("SIZE Activations softmax (%dx%d): %10.2f MiB\n", global_MB, C[num_layers+1], (double)(global_MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0) );
    printf("\nTOTAL SIZE Activations:    %10.2f MiB\n", act_size );
    printf("TOTAL SIZE Filter:         %10.2f MiB\n", fil_size );
    printf("TOTAL SIZE delActivations: %10.2f MiB\n", act_size );
    printf("TOTAL SIZE delFilter:      %10.2f MiB\n", fil_size );
    printf("TOTAL SIZE MLP:            %10.2f MiB\n", (2.0*fil_size) + (2.0*act_size) );
  }

  /* allocate data */
  /* +2 because of the softwax layer */
  act_libxsmm    = (float**)malloc( (num_layers+2)*sizeof(float*) );
  ref_act_libxsmm = (float**)malloc( (num_layers+2)*sizeof(float*) );
  delact_libxsmm = (float**)malloc( (num_layers+1)*sizeof(float*) );
  ref_delact_libxsmm = (float**)malloc( (num_layers+1)*sizeof(float*) );
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

  /* init data on every node for numa awarness */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    my_init_buf_mlp( act_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf_mlp( fil_libxsmm[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf_mlp( bias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers+1; ++i ) {
    my_init_buf_mlp( delact_libxsmm[i], MB*C[i], 0, 0 );
  }

  /* Serial initialization of data on proc 0 */
  if (rank == 0) {
    for ( i = 0 ; i < num_layers+2; ++i ) {
      ref_act_libxsmm[i]                = (float*)libxsmm_aligned_malloc( global_MB*C[i]*sizeof(float), 2097152);
      /* softmax has no incoming gradients */
      if ( i < num_layers+1 ) {
        ref_delact_libxsmm[i]             = (float*)libxsmm_aligned_malloc( global_MB*C[i]*sizeof(float), 2097152);
      }
    }
    /* init data */
    for ( i = 0 ; i < num_layers+2; ++i ) {
      my_init_buf_mlp( ref_act_libxsmm[i], global_MB*C[i], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp( fil_libxsmm[i], C[i]*C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers; ++i ) {
      my_init_buf_mlp( bias_libxsmm[i], C[i+1], 0, 0 );
    }
    for ( i = 0 ; i < num_layers+1; ++i ) {
      my_init_buf_mlp( ref_delact_libxsmm[i], global_MB*C[i], 0, 0 );
    }
  }

  /* Scatter the activations to all processes */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    MPI_Scatter(ref_act_libxsmm[i], MB * C[i], MPI_FLOAT, act_libxsmm[i], MB * C[i], MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  /* Scatter the del_activations to all processes */
  for ( i = 0 ; i < num_layers+1; ++i ) {
    MPI_Scatter(ref_delact_libxsmm[i], MB * C[i], MPI_FLOAT, delact_libxsmm[i], MB * C[i], MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  /* Now broadcast weights tensors */
  for ( i = 0 ; i < num_layers; ++i ) {
    MPI_Bcast(fil_libxsmm[i], C[i]*C[i+1], MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  /* Now broadcast bias tensors */
  for ( i = 0 ; i < num_layers; ++i ) {
    MPI_Bcast(bias_libxsmm[i], C[i], MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");
  }

  if ( fuse_type == 0 ) {
    my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK;
  } else if ( fuse_type == 3 ) {
    my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK;
  } else {
    my_fuse = LIBXSMM_DNN_FC_ELTW_FUSE_NONE;
  }

  /* allocating handles */
  libxsmm_dnn_fc_fwd = (libxsmm_dnn_fc_fwd_config*) malloc( num_layers*sizeof(libxsmm_dnn_fc_fwd_config) );
  libxsmm_dnn_fc_bwd = (libxsmm_dnn_fc_bwd_config*) malloc( num_layers*sizeof(libxsmm_dnn_fc_bwd_config) );
  libxsmm_dnn_opt    = (libxsmm_dnn_opt_config*)    malloc( num_layers*sizeof(libxsmm_dnn_opt_config)    );

  /* setting up handles + scratch */
  for ( i = 0; i < num_layers; ++i ) {
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

  /* softmax+loss is treated as N+! layer */
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

  if (type == 'F') {
    if (rank == 0) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
          libxsmm_dnn_fc_fwd_exec_f32( libxsmm_dnn_fc_fwd[i], fil_libxsmm[i], act_libxsmm[i], act_libxsmm[i+1],
                          bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch );
        }
        libxsmm_dnn_smax_fwd_exec_f32( libxsmm_dnn_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm, &loss_weight,
                            0, tid, scratch );
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = 0; i < num_layers; ++i) {
      gflop += (2.0*(double)global_MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    if (rank == 0) {
      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);
      printf("PERFDUMP,FP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
      for ( i = 0; i < num_layers; ++i ) {
        printf("%i,", C[i] );
      }
      printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
    }
  }

  if (type == 'B') {
    if (rank == 0) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
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
        libxsmm_dnn_smax_bwd_exec_f32( libxsmm_dnn_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm,
                          0, tid, scratch );
        for ( i = num_layers-1; i > 0; --i) {
          libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd[i], fil_libxsmm[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                          act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], LIBXSMM_DNN_FC_PASS_BWD, 0, tid, scratch );
          /* Thread 0 issues asynchronous all reduce */
          if (tid == 0) {
            MPI_Iallreduce(MPI_IN_PLACE, delfil_libxsmm[i], C[i]*C[i+1], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &request[i%2]);
          }
          if (i < num_layers-1) {
            /* Wait for the MPI_Iallreduce issued in the previous iteration to complete */
            if (tid == 0) {
              MPI_Wait(&request[(i+1)%2], MPI_STATUS_IGNORE);
            }
            /* All threads wait for the all-reduce to complete in order to execute the optimizer... */
            #pragma omp barrier
            libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[i+1], fil_libxsmm[i+1], delfil_libxsmm[i+1], 0, tid, scratch );
          }
        }

        /* Only UPD pass for first layer */
        libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd[0], fil_libxsmm[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                        act_libxsmm[0], delbias_libxsmm[0], relumask_libxsmm[0], LIBXSMM_DNN_FC_PASS_BWD_W, 0, tid, scratch );

        if (tid == 0) {
          MPI_Iallreduce(MPI_IN_PLACE, delfil_libxsmm[0], C[0]*C[1], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &request[0]);
        }
        if (tid == 0) {
          MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        }
        /* All threads wait for the all-reduce to complete in order to execute the optimizer... */
        #pragma omp barrier
        libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[1], fil_libxsmm[1], delfil_libxsmm[1], 0, tid, scratch );

        if (tid == 0) {
          MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        }
        /* All threads wait for the all-reduce to complete in order to execute the optimizer... */
        #pragma omp barrier
        libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[0], fil_libxsmm[0], delfil_libxsmm[0], 0, tid, scratch );
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (4.0*(double)global_MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (2.0*(double)global_MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);

    if (rank == 0) {
      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);
      printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
      for ( i = 0; i < num_layers; ++i ) {
        printf("%i,", C[i] );
      }
      printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#if 1
    if (rank == n_procs - 1) {
      for ( i = 0 ; i < num_layers; ++i ) {
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, C[i]*C[i+1], 1, delfil_libxsmm[i], delfil_libxsmm[i], 0, 0);
        printf("L1 of layer's %d dweights after training : %.25g\n", i, norms.l1_ref);
        libxsmm_matdiff_clear(&norms);
      }
    }
#endif
  }

  if (type == 'A') {
    if (rank == 0) {
      printf("##########################################\n");
      printf("# Performance - FWD-BWD (custom-Storage) #\n");
      printf("##########################################\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
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
      unsigned long long t0, t1;
      for (j = 0; j < iters; ++j) {
#ifdef DETAILED_PROFILE
        if (tid == 0) {
          t0 = libxsmm_timer_tick();
        }
#endif
        for ( i = 0; i < num_layers; ++i) {
          libxsmm_dnn_fc_fwd_exec_f32( libxsmm_dnn_fc_fwd[i], fil_libxsmm[i], act_libxsmm[i], act_libxsmm[i+1],
                          bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch );
        }
 #ifdef DETAILED_PROFILE
        if (tid == 0) {
          t1 = libxsmm_timer_tick();
          l_fwd_fc[0] += libxsmm_timer_duration(t0, t1);
          t0 = libxsmm_timer_tick();
        }
#endif
        libxsmm_dnn_smax_fwd_exec_f32( libxsmm_dnn_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm, &loss_weight,
                            0, tid, scratch );
#ifdef DETAILED_PROFILE
        if (tid == 0) {
          t1 = libxsmm_timer_tick();
          l_fwd_loss[0] += libxsmm_timer_duration(t0, t1);
          t0 = libxsmm_timer_tick();
        }
#endif
        libxsmm_dnn_smax_bwd_exec_f32( libxsmm_dnn_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm,
                          0, tid, scratch );
#ifdef DETAILED_PROFILE
        if (tid == 0) {
          t1 = libxsmm_timer_tick();
          l_bwd_loss[0] += libxsmm_timer_duration(t0, t1);
        }
#endif
        for ( i = num_layers-1; i > 0; --i) {
#ifdef DETAILED_PROFILE
          if (tid == 0) {
            t0 = libxsmm_timer_tick();
          }
#endif
          libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd[i], fil_libxsmm[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                          act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], LIBXSMM_DNN_FC_PASS_BWD, 0, tid, scratch );
#ifdef DETAILED_PROFILE
          if (tid == 0) {
            t1 = libxsmm_timer_tick();
            l_bwdupd_fc[0] += libxsmm_timer_duration(t0, t1);
            if (i == num_layers-1) {
              first_bwdupd_compute += libxsmm_timer_duration(t0, t1);
            }
          }
#endif
          /* Thread 0 issues asynchronous all reduce */
          if (tid == 0) {
            MPI_Iallreduce(MPI_IN_PLACE, delfil_libxsmm[i], C[i]*C[i+1], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &request[i%2]);
          }
          if (i < num_layers-1) {
            /* Wait for the MPI_Iallreduce issued in the previous iteration to complete */
            if (tid == 0) {
              MPI_Wait(&request[(i+1)%2], MPI_STATUS_IGNORE);
            }
            /* All threads wait for the all-reduce to complete in order to execute the optimizer... */
            #pragma omp barrier
            libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[i+1], fil_libxsmm[i+1], delfil_libxsmm[i+1], 0, tid, scratch );
          }
        }
        /* Only UPD pass for first layer */
        libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd[0], fil_libxsmm[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                        act_libxsmm[0], delbias_libxsmm[0], relumask_libxsmm[0], LIBXSMM_DNN_FC_PASS_BWD_W, 0, tid, scratch );
        if (tid == 0) {
          MPI_Iallreduce(MPI_IN_PLACE, delfil_libxsmm[0], C[0]*C[1], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &request[0]);
        }
        if (tid == 0) {
          MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        }
        /* All threads wait for the all-reduce to complete in order to execute the optimizer... */
        #pragma omp barrier
        libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[1], fil_libxsmm[1], delfil_libxsmm[1], 0, tid, scratch );

        if (tid == 0) {
          MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        }
        /* All threads wait for the all-reduce to complete in order to execute the optimizer... */
        #pragma omp barrier
        libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt[0], fil_libxsmm[0], delfil_libxsmm[0], 0, tid, scratch );
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (6.0*(double)global_MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (4.0*(double)global_MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);

    if (rank == 0) {
      printf("GFLOP  = %.5g\n", gflop/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", gflop/l_total);
      printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
      for ( i = 0; i < num_layers; ++i ) {
        printf("%i,", C[i] );
      }
      printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
#ifdef DETAILED_PROFILE
      double tot = /*l_allreduce[0] + l_optimizer[0] +*/ l_fwd_fc[0] + l_bwdupd_fc[0] + l_fwd_loss[0] + l_bwd_loss[0];
      printf("FC time compute/loss = %.5g\n", ((double)(tot/iters)));
      printf("Bwdupd compute FIRST time overlaped = %.5g\n", ((double)((first_bwdupd_compute)/iters)));
      printf("Bwdupd compute time overlaped = %.5g\n", ((double)((l_bwdupd_fc[0]-first_bwdupd_compute)/iters)));

#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
#if 0
    if (rank == n_procs - 1) {
      for ( i = 0 ; i < num_layers; ++i ) {
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, C[i]*C[i+1], 1, delfil_libxsmm[i], delfil_libxsmm[i], 0, 0);
        printf("L1 of layer's %d dweights after training : %.25g\n", i, norms.l1_ref);
        libxsmm_matdiff_clear(&norms);
      }
    }
#endif
  }

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
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

  free( act_libxsmm );
  free( delact_libxsmm );
  free( fil_libxsmm );
  free( delfil_libxsmm );
  free( bias_libxsmm );
  free( delbias_libxsmm );
  free( relumask_libxsmm );

  free( libxsmm_dnn_opt );
  free( libxsmm_dnn_fc_fwd );
  free( libxsmm_dnn_fc_bwd );

  free( C );

  if (rank == 0) {
    for ( i = 0 ; i < num_layers+2; ++i ) {
      libxsmm_free(ref_act_libxsmm[i]);
    }
    free(ref_act_libxsmm);
    for ( i = 0 ; i < num_layers+1; ++i ) {
      libxsmm_free(ref_delact_libxsmm[i]);
    }
    free(ref_delact_libxsmm);
  }

  /* Finalize the MPI environment */
  MPI_Finalize();

  return 0;
}

