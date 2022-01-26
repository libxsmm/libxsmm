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

void my_bn_bwd_exec( my_bn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                     float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                     int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = N * CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = ( ltid      * chunksize_dN < work_dN) ? ( ltid      * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN   = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = ( ltid      * chunksize_C < work_C) ? ( ltid      * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C   = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  const float scale = 1.0f / ((float)N*HW);                   /* Scaling parameter*/

  LIBXSMM_VLA_DECL(4,       float, din, pdin, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4, const float, inp, pinp, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       float, dout, pdout, CP, HW, bc);        /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float, gamma, pgamma, bc);              /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dgamma, pdgamma, bc);            /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dbeta, pdbeta, bc);              /* [CP, bc] */

  LIBXSMM_VLA_DECL(4,       float, din_add, pdin_add, CP, HW, bc);          /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  { /* stupid block to keep indentation */
    LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
    LIBXSMM_ALIGNED(float b[bc], 64);
    LIBXSMM_ALIGNED(float c[bc], 64);
    int n, cp;

    int cpxnt;

    /* ReLU/Mask/Eltwise */
    if (cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE ||
        cfg.fuse_type == MY_NORMALIZE_FUSE_RELU || cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {

      for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
        { /* stupid block to keep indentation */
          n  = cpxnt%N;
          cp = cpxnt/N;

          int hwb, cb;

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            if (cfg.fuse_type == MY_NORMALIZE_FUSE_RELU || cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
              libxsmm_meltw_unary_param all_relu_param;

              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
              cfg.relu_kernel(&all_relu_param);
            } /* ReLU/mask */
//#if 0
            if (cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
#if 0
              int i;
              for (i = 0; i < bc * (HW/num_HW_blocks); ++i) {
                int index;
                index = n * (CP * HW * bc) + cp * (HW * bc) + (hwb*(HW/num_HW_blocks)) * (bc) + i;
                pdin_add[index] = pdout[index];
              }
#endif
//#if 0
              libxsmm_meltw_unary_param ewise_copy_param;
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
//#endif
            } /* Eltwise */
//#endif
          }
        }
      } /* loop over the 1d parallel blocking */
    }  /* ReLU/Mask/Eltwise */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        n  = cpxnt%N;
        cp = cpxnt/N;

        int hwb, cb;
        libxsmm_matrix_arg arg_array[10];                                                           /* Private values of args and params */
        libxsmm_matrix_eqn_param eqn_param;

        LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
        LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

        float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, bc);
        float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, bc);

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = lcl_dgamma_ptr;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = lcl_dbeta_ptr;
        cfg.all_zero_kernel(&all_zero_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   lcl_dgamma_ptr[cb] = 0.0f; */
        /*   lcl_dbeta_ptr[cb] = 0.0f; */
        /* } */

        for(cb = 0; cb < bc; cb++){
          a[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps));
          b[cb] = -a[cb]*mean[cp*bc + cb];
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[4].primary = lcl_dgamma_ptr;
        arg_array[5].primary = lcl_dbeta_ptr;
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);

        for(hwb=0; hwb < num_HW_blocks; hwb++){

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = lcl_dgamma_ptr;
          cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        }

        libxsmm_meltw_unary_param copy_param;
        copy_param.in.primary = lcl_dgamma_ptr;
        copy_param.out.primary = dgamma_ncp_ptr;
        cfg.copy_kernel(&copy_param);

        copy_param.in.primary = lcl_dbeta_ptr;
        copy_param.out.primary = dbeta_ncp_ptr;
        cfg.copy_kernel(&copy_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
        /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
        /* } */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   pdgamma[cp*bc + cb] = 0.0f; */
      /*   pdbeta[cp*bc + cb] = 0.0f; */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        cfg.add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
        /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
        /* } */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        n  = cpxnt%N;
        cp = cpxnt/N;

        libxsmm_matrix_arg arg_array[8];                                                               /* Private eqn args and params */
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;

        /* FIXME: Replace expressions for pgamma, pdgamma etc. with ACCESS? */
        for(cb = 0; cb < bc; cb++){
          a[cb] = pgamma[cp*bc + cb] / ((float)sqrt(var[cp*bc + cb] + eps));                            /* a = gamma_ptr[bc] * brstd_ptr[bc] */
          b[cb] = -a[cb] * scale * pdgamma[cp*bc + cb] / ((float)sqrt(var[cp*bc + cb] + eps));          /* b = gamma_ptr[bc] * brstd_ptr[bc] * del_gamma_ptr[v] * brstd_ptr[bc] * recp_nhw */
          c[cb] = -b[cb] * mean[cp*bc + cb] - a[cb] * scale * pdbeta[cp*bc + cb] ;                      /* c = -gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * del_beta_ptr[bc] + gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * bmean_ptr[bc] * del_gamma_ptr[bc] * brstd_ptr[bc]) */
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
        arg_array[7].primary = c;

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          cfg.din_func(&eqn_param);                                                                        /* din = dout * a + b * inp + c */

        }
      }
    }
  } /* simple code block or parallel section for old code */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

