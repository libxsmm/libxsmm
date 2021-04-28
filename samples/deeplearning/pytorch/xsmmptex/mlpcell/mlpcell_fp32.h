#ifndef MLPCELL_F32
#define MLPCELL_F32

#include "mc_funcs.h"

#define PCL_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)
#define DECL_VLA_PTR(type, name, dims, ptr) type (*name)dims = (type (*)dims)ptr
#define DECL_VLA_PTR_CHECK_VAR(var, type, name, dims, ptr) type (*name)dims = (var > 0) ? (type (*)dims)ptr : NULL
#define DECL_VLA_PTR_CHECK_COND(cond, type, name, dims, ptr) type (*name)dims = cond ? (type (*)dims)ptr : NULL
#define DECL_VLA_PTR_CHECK_COND_VAR(cond, var, type, name, dims, ptr) type (*name)dims = (cond && var > 0) ? (type (*)dims)ptr : NULL
#define DECL_VLA_PTR_PT(type, name, dims, t) type (*name)dims = (type (*)dims)(t.data_ptr<type>())
#define DECL_VLA_PTR_PT_CHECK_COND(cond, type, name, dims, t) type (*name)dims = cond ? (type (*)dims)(t.data_ptr<type>()) : NULL
#define DECL_VLA_PTR_NPT(newtype, type, name, dims, t) newtype (*name)dims = (newtype (*)dims)(t.data_ptr<type>())
#define DECL_VLA_PTR_NPT_CHECK_COND(cond, newtype, type, name, dims, t) newtype (*name)dims = cond ? (newtype (*)dims)(t.data_ptr<type>()) : NULL
#define LIBXSMM_ALIGNDOWN(N, A) ((N) & ~((A)-1))


//--------------------------------------norm_to_normT-----------------------------------------------------
//
void norm_to_normT_32b(float* in, float* out, int N, int M)
{
  libxsmm_meltw_unary_param trans_param;

  trans_param.in.primary  = (void*)in;
  trans_param.out.primary = (void*)out;

  libxsmm_meltwfunction_unary trans_kernel = libxsmm_dispatch_meltw_unary(M, N, &M, &N, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
  if ( trans_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_NORMT TPP. Bailing...!\n");
    exit(-1);
  }
  trans_kernel( &trans_param );
}

// ----------------------------------------------------------------------------------------------------------------

inline void colbcast_f32_copy(int N, int M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
  libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);

  if ( kernel == NULL )
  {
    fprintf( stderr, "JIT for bf16 to b16 broadcast copy failed. Bailing...!\n");
    exit(-1);
  }
  kernel(params);
}

inline void dropout_f32(long N, long M, libxsmm_meltw_unary_param *params, libxsmm_meltw_unary_flags flags)
{
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void dropout_bwd_f32(long N, long M, libxsmm_meltw_unary_param *params, libxsmm_meltw_unary_flags flags)
{
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void brgemm_f32_f32(long n, long m, long k, long stride_b, long stride_a, float *B_, float *A_, float *C, long count, const float beta = 1.0, const char b_trans='n', const char a_trans='n')
{
  const float alpha = 1.0;
  float *A = A_;
  float *B = B_;
  unsigned long long l_br = count;
  int flags = LIBXSMM_GEMM_FLAGS('n', b_trans);
  // Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). *
  libxsmm_smmfunction_reducebatch_strd kernel = libxsmm_smmdispatch_reducebatch_strd(m, n, k, stride_a*sizeof(float), stride_b*sizeof(float), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
  PCL_ASSERT(kernel, "Null brgemm bf16 kernel\n");
  kernel(A, B, C, &l_br);
}

inline void delbias_f32(int N, int M, int LD_N, int LD_M, libxsmm_meltw_unary_param *delbias_params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT;
  libxsmm_meltwfunction_unary delbias_kernel = libxsmm_dispatch_meltw_unary(M, N, (libxsmm_blasint*)&LD_M, (libxsmm_blasint*)&LD_N, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);

  if (delbias_kernel == NULL ) {
    printf("Could not create f32 delbias kernel.. bailing\n");
    exit(-1);
  }

  delbias_kernel(delbias_params);
}

inline void add_f32_f32(int N, int M, libxsmm_meltw_binary_param *binary_param)
{
  libxsmm_meltw_binary_type binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  libxsmm_meltw_binary_flags binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  libxsmm_meltwfunction_binary add_kernel = libxsmm_dispatch_meltw_binary(M, N, NULL, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, binary_flags, binary_type);
  if ( add_kernel == NULL ){
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  add_kernel(binary_param);
}

inline void relu_fwd_f32(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

  libxsmm_meltwfunction_unary relu_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  if ( relu_kernel == NULL ) {
    fprintf( stderr, "JIT for ReLU TPP. Bailing...!\n");
    exit(-1);
  }
  relu_kernel( params );
}

inline void relu_bwd_f32(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

  libxsmm_meltwfunction_unary relu_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  if ( relu_kernel == NULL ) {
    fprintf( stderr, "JIT for ReLU TPP. Bailing...!\n");
    exit(-1);
  }
  relu_kernel( params );
}

class MLPCell_F32
{
  public:
    MLPCell_F32(int N, int C, int K, int bn, int bc, int bk, bool bias, bool skip, int act, bool norm, float p, bool train)
    {
      pN = N;
      pC = C;
      pK = K;
      pbn = bn;
      pbc = bc;
      pbk = bk;
      pbias = bias;
      pskip = skip;
      pact = act;
      pnorm = norm;
      pp = p;
      ptrain = train;
      //printf("MLPCell: N = %d, C = %d, K = %d, bf = %d, bias = %d, skip = %d, act = %d, norm = %d, dropout prob = %.2f train = %d\n", N, C, K, bias, skip, act, norm, p, train);
    }


  std::vector<at::Tensor> fwd(std::vector<at::Tensor> inputs)
  {
    long bn = pbn;
    long bc = pbc;
    long bk = pbk;

    long nn = pN/bn;
    long nc = pC;
    long nk = pK;
    long rn = pN % bn;

    long in_off = nn*nc*bn*bc;
    long out_off = nn*nk*bn*bk;
    long C = nc*bc;
    long K = nk*bk;

    // std::cout << "F32--------------> "  << std::endl;

    libxsmm_meltw_unary_param copy_params;
    libxsmm_meltw_unary_param cvt_params;
    libxsmm_meltw_unary_param relu_params;
    libxsmm_meltw_unary_param dropout_params;
    libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;
    libxsmm_meltw_binary_param add_params;
    libxsmm_meltw_binary_flags binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
    libxsmm_meltw_binary_type binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;

    int i=0;
    at::Tensor t_input_l = inputs[i++];
    at::Tensor t_input_r = inputs[i++];
    at::Tensor t_weights_l = inputs[i++];
    at::Tensor t_weights_r = inputs[i++];
    at::Tensor t_bias_l = inputs[i++];
    at::Tensor t_bias_r = inputs[i++];

    at::Tensor t_output = t_input_l.new_empty({pN, K});

    int dd = (bk % 32 == 0) ? bk/32 : bk/32 + 1;

    at::Tensor t_dropout_mask_bN, t_dropout_mask_rN;
    if(ptrain && pp > 0)
    {
      int size = nn*nk*bn*bk;
      t_dropout_mask_bN = at::empty(size, torch::TensorOptions().dtype(torch::kByte));
      if(rn > 0)
      {
        size = nk*rn*bk;
        t_dropout_mask_rN = at::empty(size, torch::TensorOptions().dtype(torch::kByte));
      }
    }
    __mmask32 (*dropout_mask_bN)[nk][bn][dd] = (ptrain && pp > 0) ? (__mmask32 (*)[nk][bn][dd])(t_dropout_mask_bN.data_ptr()) : NULL;
    __mmask32 (*dropout_mask_rN)[nk][rn][dd] = (ptrain && pp > 0 && rn > 0) ? (__mmask32 (*)[nk][rn][dd])(t_dropout_mask_rN.data_ptr()) : NULL;

    int rd = (bk % 32 == 0) ? bk/32 : bk/32 + 1;
    at::Tensor t_relumask_bN, t_relumask_rN;
    if(pact==1)
    {
      int size = nn*nk*bn*bk;
      t_relumask_bN = at::empty(size, torch::TensorOptions().dtype(torch::kByte));
      if(rn > 0)
      {
        size = nk*rn*bk;
        t_relumask_rN = at::empty(size, torch::TensorOptions().dtype(torch::kByte));
      }
    }
    __mmask32 (*relumask_bN)[nk][bn][rd] = pact==1 ? (__mmask32 (*)[nk][bn][rd])(t_relumask_bN.data_ptr()) : NULL;
    __mmask32 (*relumask_rN)[nk][rn][rd] = (pact==1 && rn > 0) ? (__mmask32 (*)[nk][rn][rd])(t_relumask_rN.data_ptr()) : NULL;

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    long wts = nk*nc*bk*bc;
    long in_bn = threads*nc*bn*bc;
    long in_rn = nc*rn*bc;
    long out_bn = threads*nk*bn*bk;
    long out_rn = nk*rn*bk;
    long scratch_size;

    if(pskip)
      scratch_size = (wts*4 + in_bn*2 + in_rn*2 + out_bn*3 + out_rn*3)*sizeof(float);
    else
      scratch_size = (wts*2 + in_bn + in_rn + out_bn + out_rn)*sizeof(float);

    void *scratch = libxsmm_aligned_malloc(scratch_size, 2097152);

    float *t_wt_l          = (float*)scratch;
    float *t_tr_wt_l       = t_wt_l + wts;
    float *t_input_bN_l    = t_tr_wt_l + wts;
    float *t_output_bN_l   = t_input_bN_l + in_bn;
    float *t_output_bN     = t_output_bN_l + out_bn;

    float *t_input_rN_l=NULL, *t_output_rN_l=NULL, *t_output_rN=NULL;
    if(rn > 0)
    {
      t_input_rN_l  = t_output_bN + out_bn;
      t_output_rN_l = t_input_rN_l + in_rn;
      t_output_rN   = t_output_rN_l + out_rn;
    }

    float *t_wt_r=NULL, *t_tr_wt_r=NULL, *t_input_bN_r=NULL, *t_output_bN_r=NULL;
    float *t_input_rN_r=NULL, *t_output_rN_r=NULL;
    if(pskip)
    {
      if(rn > 0)
        t_wt_r = t_output_rN + out_rn;
      else
        t_wt_r = t_output_bN + out_bn;

      t_tr_wt_r     = t_wt_r + wts;
      t_input_bN_r  = t_tr_wt_r + wts;
      t_output_bN_r = t_input_bN_r + in_bn;

      if(rn > 0)
      {
        t_input_rN_r = t_output_bN_r + out_bn;
        t_output_rN_r = t_input_rN_r + in_rn;
      }
    }

    DECL_VLA_PTR_PT(float, wt_f32_l, [C], t_weights_l);
    float *bias_l = t_bias_l.data_ptr<float>();
    float (*wt_f32_r)[C] = pskip ? (float (*)[C])t_weights_r.data_ptr<float>() : NULL;
    float *bias_r = pskip ? t_bias_r.data_ptr<float>() : NULL;

    DECL_VLA_PTR_PT(float, input_l, [C], t_input_l);
    DECL_VLA_PTR_PT_CHECK_COND(pskip, float,   input_r, [C], t_input_r);
    DECL_VLA_PTR_PT(float,  output, [K], t_output);

    DECL_VLA_PTR(float, wt_l, [nc][bk][bc], t_wt_l);
    DECL_VLA_PTR(float, tr_wt_l, [nc][bc][bk], t_tr_wt_l);
    DECL_VLA_PTR(float, input_bN_l, [nc][bn][bc], t_input_bN_l);
    DECL_VLA_PTR_CHECK_VAR(rn, float, input_rN_l, [nc][rn][bc], t_input_rN_l);
    DECL_VLA_PTR(float, output_bN, [nk][bn][bk], t_output_bN);
    DECL_VLA_PTR_CHECK_VAR(rn, float, output_rN, [nk][rn][bk], t_output_rN);

    DECL_VLA_PTR_CHECK_COND(pskip, float, wt_r, [nc][bk][bc], t_wt_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, tr_wt_r, [nc][bc][bk], t_tr_wt_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, input_bN_r, [nc][bn][bc], t_input_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, input_rN_r, [nc][rn][bc], t_input_rN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, output_bN_l, [nk][bn][bk], t_output_bN_l);
    DECL_VLA_PTR_CHECK_COND(pskip, float, output_bN_r, [nk][bn][bk], t_output_bN_r);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, float, output_rN_l, [nk][rn][bk], t_output_rN_l);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, float, output_rN_r, [nk][rn][bk], t_output_rN_r);

    for(int k=0; k<nk; k++) {
      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &wt_f32_l[k*bk][c*bc];
        copy_params.out.primary = &wt_l[k][c];
        f32_copy(bk, bc, bc, bc, &copy_params);
      }
    }
    // Wt: NORM to NORM_T
    norm_to_normT_32b(wt_l[0][0][0], tr_wt_l[0][0][0], bk, bc);

    if(pskip)
    {
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &wt_f32_r[k*bk][c*bc];
          copy_params.out.primary = &wt_r[k][c];
          f32_copy(bk, bc, bc, bc, &copy_params);
        }
      }
      norm_to_normT_32b(wt_r[0][0][0], tr_wt_r[0][0][0], bk, bc);
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_max_threads();
      int jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
      int tb = (tid*jobs < nn) ? tid*jobs : nn;
      int te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;
      int count = nc;

      libxsmm_meltw_unary_param copy_params;
      libxsmm_meltw_binary_param add_params;

      libxsmm_meltw_unary_param relu_params;
      libxsmm_meltw_unary_param dropout_params;
      libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

      for(int m=tb; m<te; m++) {

        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &input_l[m*bn][c*bc];
          copy_params.out.primary = &input_bN_l[tid][c];
          f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
        }

        if(pskip)
        {
          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &input_r[m*bn][c*bc];
            copy_params.out.primary = &input_bN_r[tid][c];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          for(int k=0; k<nk; k++) {
            copy_params.in.primary = bias_l;
            copy_params.out.primary = &output_bN_l[tid][k];
            colbcast_f32_copy(bn, bk, &copy_params);
          }

          for(int k=0; k<nk; k++) {
            copy_params.in.primary = bias_r;
            copy_params.out.primary = &output_bN_r[tid][k];
            colbcast_f32_copy(bn, bk, &copy_params);
          }
        }
        else
        {
          for(int k=0; k<nk; k++) {
            copy_params.in.primary = bias_l;
            copy_params.out.primary = &output_bN[tid][k];
            colbcast_f32_copy(bn, bk, &copy_params);
          }
        }

        if(pskip)
        {
          brgemm_f32_f32(bn, bk, bc, bn*bk, 0, input_bN_l[tid][0][0], tr_wt_l[0][0][0], output_bN_l[tid][0][0], count);
          brgemm_f32_f32(bn, bk, bc, bn*bk, 0, input_bN_r[tid][0][0], tr_wt_r[0][0][0], output_bN_r[tid][0][0], count);
          add_params.in0.primary = (void*)&output_bN_l[tid][0];
          add_params.in1.primary = (void*)&output_bN_r[tid][0];
          add_params.out.primary = (void*)&output_bN[tid][0];
          add_f32_f32(bn, bk, &add_params);
        }
        else
          brgemm_f32_f32(bn, bk, bc, bn*bk, 0, input_bN_l[tid][0][0], tr_wt_l[0][0][0], output_bN[tid][0][0], count);

        if(pact == 1)
        {
          for(int k=0; k<nk; k++) {
            relu_params.in.primary = &output_bN[tid][k];
            relu_params.out.primary = &output_bN[tid][k];
            relu_params.out.secondary = &relumask_bN[m][k];
            relu_fwd_f32(bn, bk, &relu_params);
          }
        }

        if(ptrain && pp > 0)
        {
          for(int k=0; k<nk; k++)
          {
            dropout_params.in.primary = &output_bN[tid][k];
            dropout_params.in.secondary = rnd_state;
            dropout_params.in.tertiary = &pp;
            dropout_params.out.primary = &output_bN[tid][k];
            dropout_params.out.secondary = &dropout_mask_bN[m][k];

            dropout_f32(bn, bk, &dropout_params, dropout_flags);
          }
        }

        for(int k=0; k<nk; k++) {
          copy_params.in.primary = &output_bN[tid][k];
          copy_params.out.primary = &output[m*bn][k*bk];
          f32_copy(bn, bk, nk*bk, nk*bk, &copy_params);
        }
      }
    }

    if(rn > 0)
    {
      // Single-threaded part of compute
      //
      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &input_l[nn*bn][c*bc];
        copy_params.out.primary = &input_rN_l[0][c];
        f32_copy(rn, bc, nc*bc, nc*bc, &copy_params);
      }

      if(pskip)
      {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &input_r[nn*bn][c*bc];
          copy_params.out.primary = &input_rN_r[0][c];
          f32_copy(rn, bc, nc*bc, nc*bc, &copy_params);
        }

        for(int k=0; k<nk; k++) {
          copy_params.in.primary = bias_l;
          copy_params.out.primary = &output_rN_l[0][k];
          colbcast_f32_copy(rn, bk, &copy_params);

          copy_params.in.primary = bias_r;
          copy_params.out.primary = &output_rN_r[0][k];
          colbcast_f32_copy(rn, bk, &copy_params);
        }
      }
      else
      {
        for(int k=0; k<nk; k++) {
          copy_params.in.primary = bias_l;
          copy_params.out.primary = &output_rN[0][k];
          colbcast_f32_copy(rn, bk, &copy_params);
        }
      }

      int count = nc;

      if(pskip)
      {
        brgemm_f32_f32(rn, bk, bc, rn*bk, 0, input_rN_l[0][0][0], tr_wt_l[0][0][0], output_rN_l[0][0][0], count);
        brgemm_f32_f32(rn, bk, bc, rn*bk, 0, input_rN_r[0][0][0], tr_wt_r[0][0][0], output_rN_r[0][0][0], count);
        add_params.in0.primary = (void*)&output_rN_l[0][0];
        add_params.in1.primary = (void*)&output_rN_r[0][0];
        add_params.out.primary = (void*)&output_rN[0][0];
        add_f32_f32(rn, bk, &add_params);
      }
      else
        brgemm_f32_f32(rn, bk, bc, rn*bk, 0, input_rN_l[0][0][0], tr_wt_l[0][0][0], output_rN[0][0][0], count);

      if(pact == 1)
      {
        for(int k=0; k<nk; k++) {
          relu_params.in.primary = &output_rN[0][k];
          relu_params.out.primary = &output_rN[0][k];
          relu_params.out.secondary = &relumask_rN[0][k];
          relu_fwd_f32(rn, bk, &relu_params);
        }
      }

      if(ptrain && pp > 0)
      {
        for(int k=0; k<nk; k++)
        {
          dropout_params.in.primary = &output_rN[0][k];
          dropout_params.in.secondary = rnd_state;
          dropout_params.in.tertiary = &pp;
          dropout_params.out.primary = &output_rN[0][k];
          dropout_params.out.secondary = &dropout_mask_rN[0][k];

          dropout_f32(rn, bk, &dropout_params, dropout_flags);
        }
      }

      for(int k=0; k<nk; k++) {
        copy_params.in.primary = &output_rN[0][k];
        copy_params.out.primary = &output[nn*bn][k*bk];
        f32_copy(rn, bk, nk*bk, nk*bk, &copy_params);
      }
    }

    libxsmm_free((void*)scratch);

    return {t_output, t_relumask_bN, t_relumask_rN, t_dropout_mask_bN, t_dropout_mask_rN};
  }

////=======================================================
//// ====================== BWD ===========================
////=======================================================

std::vector<at::Tensor> bwd(std::vector<at::Tensor> inputs)
  {
    long bn = pbn;
    long bc = pbc;
    long bk = pbk;

    long nn = pN/bn;
    long nc = pC;
    long nk = pK;
    long rn = pN % bn;

    long K = nk*bk;
    long C = nc*bc;

    libxsmm_meltw_unary_param copy_params;
    libxsmm_meltw_unary_param relu_params;
    libxsmm_meltw_unary_param dropout_params;
    libxsmm_meltw_unary_param delbias_params;
    libxsmm_meltw_unary_param cvt_params;
    libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    int i=0;
    at::Tensor t_grad_output = inputs[i++];
    at::Tensor t_input_l = inputs[i++];
    at::Tensor t_input_r = inputs[i++];
    at::Tensor t_weights_l = inputs[i++];
    at::Tensor t_weights_r = inputs[i++];
    at::Tensor t_relumask_bN = inputs[i++];
    at::Tensor t_relumask_rN = inputs[i++];
    at::Tensor t_dropout_mask_bN = inputs[i++];
    at::Tensor t_dropout_mask_rN = inputs[i++];

    at::Tensor t_grad_weights_l = t_weights_l.new_empty({nk, nc, bk, bc});
    at::Tensor t_grad_bias_l = t_weights_l.new_empty(K);
    at::Tensor t_grad_input_l = t_input_l.new_empty({pN, C});

    at::Tensor t_grad_weights_r, t_grad_bias_r, t_grad_input_r;
    if(pskip)
    {
      t_grad_weights_r = t_weights_r.new_empty({nk, nc, bk, bc});
      t_grad_bias_r =  t_weights_r.new_empty(K);
      t_grad_input_r = t_input_r.new_empty({pN, C});
    }

    long wts = nk*nc*bk*bc;
    long go_bn = threads*nk*bn*bk;
    long go_rn = nk*rn*bk;
    long gi_bn = threads*nc*bn*bc;
    long gi_rn = nc*rn*bc;
    long in_bn = threads*nc*bn*bc;
    long in_rn = nc*rn*bc;

    long scratch_size;
    if(pskip)
      scratch_size = (wts*4 + go_bn + go_rn + gi_bn*2 + gi_rn*2 + in_bn*2 + in_rn*2)*sizeof(float);
    else
      scratch_size = (wts*2 + go_bn + go_rn + gi_bn + gi_rn + in_bn + in_rn)*sizeof(float);

    void *scratch = libxsmm_aligned_malloc(scratch_size, 2097152);

    float* t_grad_output_bN    = (float*)scratch;
    float* t_grad_input_bN_l   = t_grad_output_bN + go_bn;
    float* t_input_bN_l        = t_grad_input_bN_l + gi_bn;
    float* t_f32_grad_wt_l     = t_input_bN_l + in_bn;
    float* t_f32_wt_l          = t_f32_grad_wt_l + wts;

    float *t_grad_output_rN=NULL, *t_grad_input_rN_l=NULL, *t_input_rN_l=NULL;
    if(rn > 0)
    {
      t_grad_output_rN      = t_f32_wt_l + wts;
      t_grad_input_rN_l     = t_grad_output_rN + go_rn;
      t_input_rN_l          = t_grad_input_rN_l + gi_rn;
    }

    float *t_grad_input_bN_r=NULL, *t_input_bN_r=NULL;
    float *t_grad_input_rN_r=NULL, *t_input_rN_r=NULL;
    float *t_f32_grad_wt_r=NULL, *t_f32_wt_r=NULL;

    if(pskip)
    {
      if(rn > 0)
        t_grad_input_bN_r = t_input_rN_l + in_rn;
      else
        t_grad_input_bN_r = t_f32_wt_l + wts;

      t_input_bN_r        = t_grad_input_bN_r + gi_bn;
      t_f32_grad_wt_r     = t_input_bN_r + in_bn;
      t_f32_wt_r          = t_f32_grad_wt_r + wts;

      if(rn > 0)
      {
        t_grad_input_rN_r = t_f32_wt_r + wts;
        t_input_rN_r      = t_grad_input_rN_r + gi_rn;
      }
    }

    DECL_VLA_PTR_PT(float, wt_l, [C], t_weights_l);
    DECL_VLA_PTR_PT(float, grad_wt_l, [C], t_grad_weights_l);
    float (*wt_r)[C] = pskip ? (float (*)[C])t_weights_r.data_ptr<float>() : NULL;
    float (*grad_wt_r)[C] = pskip ? (float (*)[C])t_grad_weights_r.data_ptr<float>() : NULL;

    DECL_VLA_PTR_PT(float, grad_output, [K], t_grad_output);
    DECL_VLA_PTR_PT(float, input_l, [C], t_input_l);
    DECL_VLA_PTR_PT(float, grad_input_l, [C], t_grad_input_l);
    DECL_VLA_PTR_PT_CHECK_COND(pskip, float, input_r, [C], t_input_r);
    DECL_VLA_PTR_PT_CHECK_COND(pskip, float, grad_input_r, [C], t_grad_input_r);

    DECL_VLA_PTR(float, grad_output_bN, [nk][bn][bk], t_grad_output_bN);
    DECL_VLA_PTR(float, grad_input_bN_l, [nc][bn][bc], t_grad_input_bN_l);
    DECL_VLA_PTR(float, input_bN_l, [nc][bn][bc], t_input_bN_l);
    DECL_VLA_PTR(float, wt_f32_l, [nc][bk][bc], t_f32_wt_l);
    DECL_VLA_PTR(float, grad_wt_f32_l, [nc][bk][bc], t_f32_grad_wt_l);
    float *grad_bias_l = t_grad_bias_l.data_ptr<float>();

    DECL_VLA_PTR_CHECK_COND(pskip, float, grad_input_bN_r, [nc][bn][bc], t_grad_input_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, input_bN_r, [nc][bn][bc], t_input_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, wt_f32_r, [nc][bk][bc], t_f32_wt_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, grad_wt_f32_r, [nc][bk][bc], t_f32_grad_wt_r);
    float *grad_bias_r = pskip ? t_grad_bias_r.data_ptr<float>() : NULL;

    DECL_VLA_PTR_CHECK_VAR(rn, float, grad_output_rN, [nk][rn][bk], t_grad_output_rN);
    DECL_VLA_PTR_CHECK_VAR(rn, float, grad_input_rN_l, [nc][rn][bc], t_grad_input_rN_l);
    DECL_VLA_PTR_CHECK_VAR(rn, float, input_rN_l, [nc][rn][bc], t_input_rN_l);

    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, float, grad_input_rN_r, [nc][rn][bc], t_grad_input_rN_r);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, float, input_rN_r, [nc][rn][bc], t_input_rN_r);

    int dd = (bk % 32 == 0) ? bk/32 :  bk/32 + 1;
    int rd = (bk % 32 == 0) ? bk/32 :  bk/32 + 1;

    __mmask32 (*dropout_mask_bN)[nk][bn][dd] = (ptrain && pp > 0) ? (__mmask32 (*)[nk][bn][dd])(t_dropout_mask_bN.data_ptr()) : NULL;
    __mmask32 (*dropout_mask_rN)[nk][rn][dd] = (ptrain && pp > 0 && rn > 0) ? (__mmask32 (*)[nk][rn][dd])(t_dropout_mask_rN.data_ptr()) : NULL;
    __mmask32 (*relumask_bN)[nk][bn][rd] = pact==1 ? (__mmask32 (*)[nk][bn][rd])(t_relumask_bN.data_ptr()) : NULL;
    __mmask32 (*relumask_rN)[nk][rn][rd] = (pact==1 && rn > 0) ? (__mmask32 (*)[nk][rn][rd])(t_relumask_rN.data_ptr()) : NULL;

    copy_params.out.primary = t_f32_grad_wt_l;
    zero(K*C, &copy_params);

    copy_params.out.primary = t_grad_weights_l.data_ptr<float>();
    zero(K*C, &copy_params);

    copy_params.out.primary = t_grad_bias_l.data_ptr<float>();
    zero(K, &copy_params);

    if(pskip)
    {
      copy_params.out.primary = t_f32_grad_wt_r;
      zero(K*C, &copy_params);
    }

    if(pskip)
    {
      copy_params.out.primary = t_grad_weights_r.data_ptr<float>();
      zero(K*C, &copy_params);
      copy_params.out.primary = t_grad_bias_r.data_ptr<float>();
      zero(K, &copy_params);
    }

    // Get F32 copy of weights
    for(int k=0; k<nk; k++) {
      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &wt_l[k*bk][c*bc];
        copy_params.out.primary = &wt_f32_l[k][c];
        f32_copy(bk, bc, bc, bc, &copy_params);
      }
    }

    if(pskip)
    {
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &wt_r[k*bk][c*bc];
          copy_params.out.primary = &wt_f32_r[k][c];
          f32_copy(bk, bc, bc, bc, &copy_params);
        }
      }
    }

    if(pskip)
    {
#ifdef _OPENMP
#pragma omp parallel reduction(+: grad_wt_f32_l[:nk][:nc][:bk][:bc], grad_bias_l[:K], grad_wt_f32_r[:nk][:nc][:bk][:bc], grad_bias_r[:K])
#endif
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
        int tb = (tid*jobs < nn) ? tid*jobs : nn;
        int te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;

        libxsmm_meltw_unary_param relu_params;
        libxsmm_meltw_unary_param dropout_params;
        libxsmm_meltw_unary_param copy_params;
        libxsmm_meltw_unary_param delbias_params;

        for(int m=tb; m<te; m++) {
          for(int k=0; k<nk; k++) {

            if(ptrain && pp > 0) {
              dropout_params.in.primary = &grad_output[m*bn][k*bk];
              dropout_params.in.secondary = &dropout_mask_bN[m][k][0][0];
              dropout_params.in.tertiary = &pp;
              dropout_params.out.primary = &grad_output[m*bn][k*bk];
              dropout_bwd_f32(bn, bk, &dropout_params, dropout_flags);
            }

            if(pact == 1) {
              relu_params.in.primary = &grad_output[m*bn][k*bk];
              relu_params.in.secondary = &relumask_bN[m][k][0][0];
              relu_params.out.primary = &grad_output[m*bn][k*bk];
              relu_bwd_f32(bn, bk, &relu_params);
            }

            copy_params.in.primary = &grad_output[m*bn][k*bk];
            copy_params.out.primary = &grad_output_bN[tid][k];
            f32_copy(bn, bk, nk*bk, nk*bk, &copy_params);
          }

          int count=1;
          brgemm_f32_f32(bn, bc, bk, bn*bk, 0, grad_output_bN[tid][0][0], wt_f32_l[0][0][0], grad_input_bN_l[tid][0][0], count, 0.0);

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &grad_input_bN_l[tid][c];
            copy_params.out.primary = &grad_input_l[m*bn][c*bc];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          brgemm_f32_f32(bn, bc, bk, bn*bk, 0, grad_output_bN[tid][0][0], wt_f32_r[0][0][0], grad_input_bN_r[tid][0][0], count, 0.0);

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &grad_input_bN_r[tid][c];
            copy_params.out.primary = &grad_input_r[m*bn][c*bc];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &input_l[m*bn][c*bc];
            copy_params.out.primary = &input_bN_l[tid][c];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }
          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &input_r[m*bn][c*bc];
            copy_params.out.primary = &input_bN_r[tid][c];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          count = 1;
          brgemm_f32_f32(bk, bc, bn, bn*bk, bn*bc, grad_output_bN[tid][0][0], input_bN_l[tid][0][0], grad_wt_f32_l[0][0][0], count, 1.0, 't');
          brgemm_f32_f32(bk, bc, bn, bn*bk, bn*bc, grad_output_bN[tid][0][0], input_bN_r[tid][0][0], grad_wt_f32_r[0][0][0], count, 1.0, 't');

          for(int k=0; k<nk; k++) {
            delbias_params.in.primary = &grad_output_bN[tid][k];
            delbias_params.out.primary = grad_bias_l;
            delbias_f32(bn, bk, bn, bk, &delbias_params);
          }

          copy_params.in.primary = grad_bias_l;
          copy_params.out.primary = grad_bias_r;
          f32_copy(1, K, K, K, &copy_params);
        }
      }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel reduction(+: grad_wt_f32_l[:nk][:nc][:bk][:bc], grad_bias_l[:K])
#endif
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
        int tb = (tid*jobs < nn) ? tid*jobs : nn;
        int te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;

        libxsmm_meltw_unary_param relu_params;
        libxsmm_meltw_unary_param dropout_params;
        libxsmm_meltw_unary_param copy_params;
        libxsmm_meltw_unary_param  delbias_params;

        for(int m=tb; m<te; m++) {

          for(int k=0; k<nk; k++) {
            if(ptrain && pp > 0) {
              dropout_params.in.primary = &grad_output[m*bn][k*bk];
              dropout_params.in.secondary = &dropout_mask_bN[m][k][0][0];
              dropout_params.in.tertiary = &pp;
              dropout_params.out.primary = &grad_output[m*bn][k*bk];
              dropout_bwd_f32(bn, bk, &dropout_params, dropout_flags);
            }

            if(pact == 1) {
              relu_params.in.primary = &grad_output[m*bn][k*bk];
              relu_params.in.secondary = &relumask_bN[m][k][0][0];
              relu_params.out.primary = &grad_output[m*bn][k*bk];
              relu_bwd_f32(bn, bk, &relu_params);
            }

            copy_params.in.primary = &grad_output[m*bn][k*bk];
            copy_params.out.primary = &grad_output_bN[tid][k];
            f32_copy(bn, bk, nk*bk, nk*bk, &copy_params);
          }

          int count = 1;
          brgemm_f32_f32(bn, bc, bk, bn*bk, 0, grad_output_bN[tid][0][0], wt_f32_l[0][0][0], grad_input_bN_l[tid][0][0], count, 0.0);

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &grad_input_bN_l[tid][c];
            copy_params.out.primary = &grad_input_l[m*bn][c*bc];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &input_l[m*bn][c*bc];
            copy_params.out.primary = &input_bN_l[tid][c];
            f32_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          count = 1;
          brgemm_f32_f32(bk, bc, bn, bn*bk, bn*bc, grad_output_bN[tid][0][0], input_bN_l[tid][0][0], grad_wt_f32_l[0][0][0], count, 1.0, 't');

          for(int k=0; k<nk; k++) {
            delbias_params.in.primary = &grad_output_bN[tid][k];
            delbias_params.out.primary = grad_bias_l;
            delbias_f32(bn, bk, bn, bk, &delbias_params);
          }
        }
      }
    }


    if(rn > 0)
    {
      //Single-thread portion of code--------------------------

      // Dropout
      if(ptrain && pp > 0)
      {
        for(int k=0; k<nk; k++) {
          dropout_params.in.primary = &grad_output[nn*bn][k*bk];
          dropout_params.in.secondary = &dropout_mask_rN[0][k][0][0];
          dropout_params.in.tertiary = &pp;
          dropout_params.out.primary = &grad_output[nn*bn][k*bk];
          dropout_bwd_f32(rn, bk, &dropout_params, dropout_flags);
        }
      }

      // ReLU
      if(pact == 1)
      {
        for(int k=0; k<nk; k++) {
          relu_params.in.primary = &grad_output[nn*bn][k*bk];
          relu_params.in.secondary = &relumask_rN[0][k][0][0];
          relu_params.out.primary = &grad_output[nn*bn][k*bk];
          relu_bwd_f32(rn, bk, &relu_params);
        }
      }

      int count=1;

      //grad-input
      for(int k=0; k<nk; k++) {
        copy_params.in.primary = &grad_output[nn*bn][k*bk];
        copy_params.out.primary = &grad_output_rN[0][k];
        f32_copy(rn, bk, nk*bk, nk*bk, &copy_params);
      }
      brgemm_f32_f32(rn, bc, bk, rn*bk, 0, grad_output_rN[0][0][0], wt_f32_l[0][0][0], grad_input_rN_l[0][0][0], count, 0.0);

      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &grad_input_rN_l[0][c];
        copy_params.out.primary = &grad_input_l[nn*bn][c*bc];
        f32_copy(rn, bc, nc*bc, nc*bc, &copy_params);
      }

      if(pskip)
      {
        brgemm_f32_f32(rn, bc, bk, rn*bk, 0, grad_output_rN[0][0][0], wt_f32_r[0][0][0], grad_input_rN_r[0][0][0], count, 0.0);

        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &grad_input_rN_r[0][c];
          copy_params.out.primary = &grad_input_r[nn*bn][c*bc];
          f32_copy(rn, bc, nc*bc, nc*bc, &copy_params);
        }
      }

      //grad-weights
      count = 1;
      brgemm_f32_f32(bk, bc, rn, rn*bk, rn*bc, grad_output_rN[0][0][0], input_rN_l[0][0][0], grad_wt_f32_l[0][0][0], count, 1.0, 't');

      if(pskip)
      {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &input_r[nn*bn][c*bc];
          copy_params.out.primary = &input_rN_r[0][c];
          f32_copy(rn, bc, nc*bc, nc*bc, &copy_params);
        }

        count = 1;
        brgemm_f32_f32(bk, bc, rn, rn*bk, rn*bc, grad_output_rN[0][0][0], input_rN_r[0][0][0], grad_wt_f32_r[0][0][0], count, 1.0, 't');
      }

      for(int k=0; k<nk; k++) {
        delbias_params.in.primary = &grad_output_rN[0][k];
        delbias_params.out.primary = grad_bias_l;
        delbias_f32(rn, bk, rn, bk, &delbias_params);
      }

      if(pskip)
      {
        for(int k=0; k<nk; k++) {
          delbias_params.in.primary = &grad_output_rN[0][k];
          delbias_params.out.primary = grad_bias_r;
          delbias_f32(rn, bk, rn, bk, &delbias_params);
        }
      }
    }

    for(int k=0; k<nk; k++) {
      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &grad_wt_f32_l[k][c];
        copy_params.out.primary = &grad_wt_l[k*bk][c*bc];
        f32_copy(bk, bc, nc*bc, nc*bc, &copy_params);
      }
    }

    if(pskip)
    {
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &grad_wt_f32_r[k][c];
          copy_params.out.primary = &grad_wt_r[k*bk][c*bc];
          f32_copy(bk, bc, nc*bc, nc*bc, &copy_params);
        }
      }
    }
    libxsmm_free(scratch);

    return {t_grad_input_l, t_grad_input_r, t_grad_weights_l, t_grad_weights_r, t_grad_bias_l, t_grad_bias_r};
  }

  bool has_bias() {return pbias;}
  bool has_skip() {return pskip;}
  bool has_norm() {return pnorm;}

  private:
    long pN;
    long pC;
    long pK;
    long pbn;
    long pbc;
    long pbk;
    bool pbias;
    bool pskip;
    int pact;
    bool pnorm;
    float pp;
    bool ptrain;
};


#endif

