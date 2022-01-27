/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kirill Voronin (Intel Corp.)
******************************************************************************/

void my_bn_fwd_exec( my_bn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout, unsigned char *prelumask, float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(4, const float,         inp,      pinp, CP, HW, bc);            /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       float,         out,      pout, CP, HW, bc);            /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                  /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                   /* [CP, bc] */

  LIBXSMM_VLA_DECL(4, const float,         inp_add,  pinp_add, CP, HW, bc);        /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  const float scale = 1.0f /((float)N * HW);

  LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, bc);  /* [2, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
  const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sum_N, ((float*)scratch) + sum_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
  const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sumsq_N, ((float*)scratch) + sumsq_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);

  { /* stupid block to keep indentation */
    LIBXSMM_ALIGNED(float s[bc], 64);
    LIBXSMM_ALIGNED(float b[bc], 64);
    int n, cp;

    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      int hwb;

      float *sum_ncp_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, n, 0, N, bc);
      float *sumsq_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, n, 0, N, bc);

      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = sum_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = sumsq_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd  */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_ncp_ptr[cb] = 0.0f;    */
      /*   sumsq_ncp_ptr[cb] = 0.0f;  */
      /* } */

      libxsmm_meltw_binary_param add_param;

      libxsmm_meltw_unary_param reduce_HW_params;       /*Private params and tmp array */
      LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bc], 64);
      reduce_HW_params.out.primary   = lcl_sum_X_X2;                                                         /* [2*bc]  */
      for(hwb=0; hwb < num_HW_blocks; hwb++){

        reduce_HW_params.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
        cfg.reduce_HW_kernel(&reduce_HW_params);                                                       /* [HW, bc] -----> [2 * bc] */

        add_param.in0.primary = sum_ncp_ptr;
        add_param.in1.primary = lcl_sum_X_X2;
        add_param.out.primary = sum_ncp_ptr;
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = sumsq_ncp_ptr;
        add_param.in1.primary = &lcl_sum_X_X2[bc];
        add_param.out.primary = sumsq_ncp_ptr;
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) {  */
        /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
        /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[bc + cb];  */
        /* }  */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {

      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_X_X2[cp*bc + cb] = 0.0f;   */
      /*   sum_X_X2[CP*bc + (cp*bc + cb)] = 0.0f;  */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int cb, ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   sum_X_X2[cp*bc + cb] += sum_N[cp*N*bc + n*bc + cb]; */
        /*   sum_X_X2[CP*bc + (cp*bc + cb)] += sumsq_N[cp*N*bc + n*bc + cb]; */
        /* } */
      }

      for(cb = 0; cb < bc; cb++){
        mean[cp*bc + cb] = (LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, cb, CP, bc)) * scale;                 /* E[X] */
        var[cp*bc + cb] = ((LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, cb, CP, bc)) * scale) - (mean[cp*bc + cb]*mean[cp*bc + cb]);
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      libxsmm_matrix_arg arg_array[5];                                                         /* private eqn args and params*/
      libxsmm_matrix_eqn_param eqn_param;
      int hwb, cb;

      for(cb = 0; cb < bc; cb++){
        s[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps));                                 /* s = 1/sqrt(var(X) + eps)     [bc] */
        b[cb] = -1 * mean[cp*bc + cb] * s[cb];                                               /* b = -E[X]/sqrt(var(X) + eps) [bc] */
      }
      arg_array[1].primary = s;                                                              /* [bc] */
      arg_array[2].primary = b;                                                              /* [bc] */
      arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);                       /* [bc] */
      arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta,  cp, 0, bc);                       /* [bc] */

      for(hwb=0; hwb < num_HW_blocks; hwb++){

        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);           /* [HW, bc] */
        eqn_param.inputs = arg_array;
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);       /* [HW,bc] */
        cfg.func10(&eqn_param);                                                                    /* Normalization equation -> y = ((s*x + b)*gamma + beta) */

        /* Eltwise add */
        if (cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
          libxsmm_meltw_binary_param add_param;
          add_param.in0.primary = (void*)&LIBXSMM_VLA_ACCESS(4, out,     n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          add_param.in1.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          add_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, out,     n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          cfg.ewise_add_kernel(&add_param);
        }

        /* ReLU */
        if (cfg.fuse_type == MY_NORMALIZE_FUSE_RELU || cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
          libxsmm_meltw_unary_param all_relu_param;

          all_relu_param.op.primary   = (void*)(&alpha);
          all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
          all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
          all_relu_param.out.secondary = ((cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                            (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc/BITS_PER_CHAR) : NULL );
          cfg.relu_kernel(&all_relu_param);
        } /* ReLU */
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);

}

