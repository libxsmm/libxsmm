#ifndef MLPCELL_BF16
#define MLPCELL_BF16

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

//--------------------------------------norm_to_vnni-----------------------------------------------------
//
void norm_to_vnni_16b(libxsmm_bfloat16* in, libxsmm_bfloat16* out, int N, int M)
{
  libxsmm_meltw_unary_param trans_param;
  libxsmm_meltw_unary_type trans_type;

  trans_param.in.primary  = (void*)in;
  trans_param.out.primary = (void*)out;

  if ( N % 2 == 1 ) {
    trans_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD;
  } else {
    trans_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI;
  }
  libxsmm_meltwfunction_unary trans_kernel = libxsmm_dispatch_meltw_unary(M, N, (libxsmm_blasint*)&M, (libxsmm_blasint*)&M, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, trans_type);
  if ( trans_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_VNNI TPP. Bailing...!\n");
    exit(-1);
  }
  trans_kernel( &trans_param );
}

//--------------------------------------norm_to_normT-----------------------------------------------------
//
void norm_to_normT_16b(libxsmm_bfloat16* in, libxsmm_bfloat16* out, int N, int M)
{
  libxsmm_meltw_unary_param trans_param;
  libxsmm_meltw_unary_type trans_type;

  trans_param.in.primary  = (void*)in;
  trans_param.out.primary = (void*)out;

  trans_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;

  libxsmm_meltwfunction_unary trans_kernel = libxsmm_dispatch_meltw_unary(M, N, (libxsmm_blasint*)&M, (libxsmm_blasint*)&N, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, trans_type);
  if ( trans_kernel == NULL ) {
    fprintf( stderr, "JIT for NORM_TO_NORMT TPP. Bailing...!\n");
    exit(-1);
  }
  trans_kernel( &trans_param );
}

//--------------------------------------------------convert f32 to bf16 TPP-------------------------------------
inline void cvt_f32_bf16(int N, int M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_meltwfunction_unary cvt_f32_bf16_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type );

  PCL_ASSERT(cvt_f32_bf16_kernel, "Null cvt_f32_bf16 kernel");

  cvt_f32_bf16_kernel(params);
}

inline void bf16_copy(int N, int M, int LDO, int LDI, libxsmm_meltw_unary_param  *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_datatype compute_dtype = LIBXSMM_DATATYPE_BF16;
  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(M, N, (libxsmm_blasint*)&LDI, (libxsmm_blasint*)&LDO, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, compute_dtype, unary_flags, unary_type);

  if ( kernel == NULL )
  {
    fprintf( stderr, "JIT for bf16 to b16 copy failed. Bailing...!\n");
    exit(-1);
  }
  kernel(params);
}

inline void colbcast_bf16_copy(int N, int M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
  libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_datatype compute_dtype = LIBXSMM_DATATYPE_BF16;

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, compute_dtype, unary_flags, unary_type);

  if ( kernel == NULL )
  {
    fprintf( stderr, "JIT for bf16 to b16 broadcast copy failed. Bailing...!\n");
    exit(-1);
  }
  kernel(params);
}

inline void bf16_f32_copy(int N, int M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_datatype compute_dtype = LIBXSMM_DATATYPE_F32;
  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, compute_dtype, unary_flags, unary_type);

  if ( kernel == NULL )
  {
    fprintf( stderr, "JIT for bf16 to f32 copy failed. Bailing...!\n");
    exit(-1);
  }
  kernel(params);
}

inline void add_bf16_bf16(int N, int M, libxsmm_meltw_binary_param *binary_param)
{
  libxsmm_meltw_binary_flags binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  libxsmm_meltw_binary_type binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  libxsmm_meltwfunction_binary add_kernel = libxsmm_dispatch_meltw_binary(M, N, (libxsmm_blasint*)&M, (libxsmm_blasint*)&M, (libxsmm_blasint*)&N, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, binary_flags, binary_type);

  if ( add_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  add_kernel(binary_param);
}

inline void relu_fwd_bf16(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

  libxsmm_meltwfunction_unary relu_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  if ( relu_kernel == NULL ) {
    fprintf( stderr, "JIT for ReLU TPP. Bailing...!\n");
    exit(-1);
  }
  relu_kernel( params );
}

inline void relu_bwd_bf16(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

  libxsmm_meltwfunction_unary relu_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  if ( relu_kernel == NULL ) {
    fprintf( stderr, "JIT for ReLU TPP. Bailing...!\n");
    exit(-1);
  }
  relu_kernel( params );
}

inline void dropout_bf16(long N, long M, libxsmm_meltw_unary_param *params, libxsmm_meltw_unary_flags flags)
{
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void dropout_bwd_bf16(long N, long M, libxsmm_meltw_unary_param *params, libxsmm_meltw_unary_flags flags)
{
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void brgemm_bf16_f32(long n, long m, long k, long stride_b, long stride_a, libxsmm_bfloat16 *B_, libxsmm_bfloat16 *A_, float *C,
  long count, const float beta = 1.0, const char b_trans='n', const char a_trans='n', const char b_vnni='n', const char a_vnni='n')
{
  const float alpha = 1.0;
  libxsmm_bfloat16 *A = A_;
  libxsmm_bfloat16 *B = B_;
  unsigned long long l_br = count;
  int flags = LIBXSMM_GEMM_VNNI_FLAGS('n', 'n', 'v', 'n');
  // Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). *
  libxsmm_bsmmfunction_reducebatch_strd kernel = libxsmm_bsmmdispatch_reducebatch_strd(m, n, k, stride_a*sizeof(libxsmm_bfloat16), stride_b*sizeof(libxsmm_bfloat16), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
  PCL_ASSERT(kernel, "Null brgemm bf16 kernel\n");
  kernel(A, B, C, &l_br);
}

inline void brgemm_bf16_bf16(long n, long m, long k, long stride_b, long stride_a, libxsmm_bfloat16 *B_, libxsmm_bfloat16 *A_, libxsmm_bfloat16 *C,
  long count, const float beta = 1.0, const char b_trans='n', const char a_trans='n', const char b_vnni='n', const char a_vnni='n')
{
  const float alpha = 1.0;
  libxsmm_bfloat16 *A = A_;
  libxsmm_bfloat16 *B = B_;
  unsigned long long l_br = count;
  int flags = LIBXSMM_GEMM_VNNI_FLAGS('n', 'n', 'v', 'n');
  // Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). *
  libxsmm_bmmfunction_reducebatch_strd kernel = libxsmm_bmmdispatch_reducebatch_strd(m, n, k, stride_a*sizeof(libxsmm_bfloat16), stride_b*sizeof(libxsmm_bfloat16), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
  PCL_ASSERT(kernel, "Null brgemm bf16 kernel\n");
  kernel(A, B, C, &l_br);
}

inline void delbias_bf16_f32(int N, int M, int ldo, int ldi, libxsmm_meltw_unary_param *delbias_params)
{
  libxsmm_meltwfunction_unary delbias_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT);

  if (delbias_kernel == NULL ) {
    printf("Could not create bf16 delbias kernel.. bailing\n");
    exit(-1);
  }

  delbias_kernel(delbias_params);
}

class MLPCell_BF16
{
  public:
    MLPCell_BF16(int N, int C, int K, int bn, int bc, int bk, bool bias, bool skip, int act, bool norm, float p, bool train)
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

    // std::cout << "BF16--------------> "  << std::endl;

    long bcp = (bc % 2 != 0) ? (bc + 1): bc;

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

    long wts = nk*nc*bk*bcp;
    long in_bn = threads*nc*bn*bc;
    long in_rn = nc*rn*bc;
    long out_bn = threads*nk*bn*bk;
    long out_rn = nk*rn*bk;
    long scratch_size;

    if(pskip)
      scratch_size = (wts*6 + in_bn*2 + in_rn*2 + out_bn*3 + out_rn*3 + K*2)*sizeof(libxsmm_bfloat16);
    else
      scratch_size = (wts*3 + in_bn + in_rn + out_bn + out_rn + K)*sizeof(libxsmm_bfloat16);

    void *scratch = libxsmm_aligned_malloc(scratch_size, 2097152);

    libxsmm_bfloat16 *t_bf16_weights_l = (libxsmm_bfloat16*)scratch;
    libxsmm_bfloat16 *t_tr_weights_l   = t_bf16_weights_l + wts;
    libxsmm_bfloat16 *t_vnni_weights_l = t_tr_weights_l + wts;
    libxsmm_bfloat16 *t_input_bN_l     = t_vnni_weights_l + wts;
    libxsmm_bfloat16 *t_output_bN_l    = t_input_bN_l + in_bn;
    libxsmm_bfloat16 *t_output_bN      = t_output_bN_l + out_bn;
    libxsmm_bfloat16 *t_bf16_bias_l    = t_output_bN + out_bn;

    libxsmm_bfloat16 *t_input_rN_l=NULL, *t_output_rN_l=NULL, *t_output_rN=NULL;
    if(rn > 0)
    {
      t_input_rN_l  = t_bf16_bias_l + K;
      t_output_rN_l = t_input_rN_l + in_rn;
      t_output_rN   = t_output_rN_l + out_rn;
    }

    libxsmm_bfloat16 *t_bf16_weights_r=NULL, *t_tr_weights_r=NULL, *t_vnni_weights_r=NULL, *t_input_bN_r=NULL, *t_output_bN_r=NULL, *t_bf16_bias_r=NULL;
    libxsmm_bfloat16 *t_input_rN_r=NULL, *t_output_rN_r=NULL;
    if(pskip)
    {
      if(rn > 0)
        t_bf16_weights_r = t_output_rN + out_rn;
      else
        t_bf16_weights_r = t_bf16_bias_l + K;

      t_tr_weights_r   = t_bf16_weights_r + wts;
      t_vnni_weights_r = t_tr_weights_r + wts;
      t_input_bN_r     = t_vnni_weights_r + wts;
      t_output_bN_r    = t_input_bN_r + in_bn;
      t_bf16_bias_r    = t_output_bN_r + out_bn;

      if(rn > 0)
      {
        t_input_rN_r = t_bf16_bias_r + K;
        t_output_rN_r = t_input_rN_r + in_rn;
      }
    }

    DECL_VLA_PTR_PT(float, wt_f32_l, [C], t_weights_l);
    float *bias_f32_l = t_bias_l.data_ptr<float>();
    float (*wt_f32_r)[C] = pskip ? (float (*)[C])t_weights_r.data_ptr<float>() : NULL;
    float *bias_f32_r = pskip ? t_bias_r.data_ptr<float>() : NULL;

    DECL_VLA_PTR_NPT(libxsmm_bfloat16, at::BFloat16, input_l, [C], t_input_l);
    DECL_VLA_PTR_NPT_CHECK_COND(pskip, libxsmm_bfloat16, at::BFloat16,  input_r, [C], t_input_r);
    DECL_VLA_PTR_NPT(libxsmm_bfloat16, at::BFloat16, output, [K], t_output);

    DECL_VLA_PTR(libxsmm_bfloat16, wt_l, [nc][bk][bc], t_bf16_weights_l);
    DECL_VLA_PTR(libxsmm_bfloat16, tr_wt_l, [nc][bcp][bk], t_tr_weights_l);
    DECL_VLA_PTR(libxsmm_bfloat16, vnni_wt_l, [nc][bcp/2][bk][2], t_vnni_weights_l);
    DECL_VLA_PTR(libxsmm_bfloat16, input_bN_l, [nc][bn][bc], t_input_bN_l);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, input_rN_l, [nc][rn][bc], t_input_rN_l);
    DECL_VLA_PTR(libxsmm_bfloat16, output_bN, [nk][bn][bk], t_output_bN);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, output_rN, [nk][rn][bk], t_output_rN);
    DECL_VLA_PTR(libxsmm_bfloat16, bias_l, [bk], t_bf16_bias_l);

    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, wt_r, [nc][bk][bc], t_bf16_weights_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, tr_wt_r, [nc][bcp][bk], t_tr_weights_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, vnni_wt_r, [nc][bcp/2][bk][2], t_vnni_weights_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, input_bN_r, [nc][bn][bc], t_input_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, input_rN_r, [nc][rn][bc], t_input_rN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, output_bN_l, [nk][bn][bk], t_output_bN_l);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, output_bN_r, [nk][bn][bk], t_output_bN_r);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, libxsmm_bfloat16, output_rN_l, [nk][rn][bk], t_output_rN_l);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, libxsmm_bfloat16, output_rN_r, [nk][rn][bk], t_output_rN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, bias_r, [bk], t_bf16_bias_r);

    // Get BF16 copy of weights
    for(int k=0; k<nk; k++) {
      for(int c=0; c<nc; c++) {
        cvt_params.in.primary = &wt_f32_l[k*bk][c*bc];
        cvt_params.out.primary = &wt_l[k][c];
        cvt_f32_bf16(bk, bc, &cvt_params);
      }
    }

    if(pskip)
    {
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
          cvt_params.in.primary = &wt_f32_r[k*bk][c*bc];
          cvt_params.out.primary = &wt_r[k][c];
          cvt_f32_bf16(bk, bc, &cvt_params);
        }
      }
    }

    for(int k=0; k<nk; k++) {
      cvt_params.in.primary = bias_f32_l;
      cvt_params.out.primary = &bias_l[k];
      cvt_f32_bf16(nk, bk, &cvt_params);
    }

    if(pskip)
    {
      for(int k=0; k<nk; k++) {
        cvt_params.in.primary = bias_f32_r;
        cvt_params.out.primary = &bias_r[k];
        cvt_f32_bf16(nk, bk, &cvt_params);
      }
    }

    // Wt: NORM layout to VNNI
    norm_to_normT_16b(wt_l[0][0][0], tr_wt_l[0][0][0], bk, bcp);
    norm_to_vnni_16b(tr_wt_l[0][0][0], vnni_wt_l[0][0][0][0], bcp, bk);

    if(pskip)
    {
      norm_to_normT_16b(wt_r[0][0][0], tr_wt_r[0][0][0], bk, bcp);
      norm_to_vnni_16b(tr_wt_r[0][0][0], vnni_wt_r[0][0][0][0], bcp, bk);
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
          bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
        }

        if(pskip)
        {
          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &input_r[m*bn][c*bc];
            copy_params.out.primary = &input_bN_r[tid][c];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          for(int k=0; k<nk; k++) {
            copy_params.in.primary = &bias_l[k][0];
            copy_params.out.primary = &output_bN_l[tid][k];
            colbcast_bf16_copy(bn, bk, &copy_params);
          }

          for(int k=0; k<nk; k++) {
            copy_params.in.primary = &bias_r[k][0];
            copy_params.out.primary = &output_bN_r[tid][k];
            colbcast_bf16_copy(bn, bk, &copy_params);
          }
        }
        else
        {
          for(int k=0; k<nk; k++) {
            copy_params.in.primary = &bias_l[k][0];
            copy_params.out.primary = &output_bN[tid][k];
            colbcast_bf16_copy(bn, bk, &copy_params);
          }
        }

        if(pskip)
        {
          brgemm_bf16_bf16(bn, bk, bcp, bn*bk, 0, input_bN_l[tid][0][0], vnni_wt_l[0][0][0][0], output_bN_l[tid][0][0], count);
          brgemm_bf16_bf16(bn, bk, bcp, bn*bk, 0, input_bN_r[tid][0][0], vnni_wt_r[0][0][0][0], output_bN_r[tid][0][0], count);
          add_params.in0.primary = (void*)&output_bN_l[tid][0];
          add_params.in1.primary = (void*)&output_bN_r[tid][0];
          add_params.out.primary = (void*)&output_bN[tid][0];
          add_bf16_bf16(bn, bk, &add_params);
        }
        else
          brgemm_bf16_bf16(bn, bk, bcp, bn*bk, 0, input_bN_l[tid][0][0], vnni_wt_l[0][0][0][0], output_bN[tid][0][0], count);

        if(pact == 1)
        {
          for(int k=0; k<nk; k++) {
            relu_params.in.primary = &output_bN[tid][k];
            relu_params.out.primary = &output_bN[tid][k];
            relu_params.out.secondary = &relumask_bN[m][k];
            relu_fwd_bf16(bn, bk, &relu_params);
          }
        }

        if(ptrain && pp > 0)
        {
          for(int k=0; k<nk; k++)
          {
            dropout_params.in.primary = &output_bN[tid][k];
            dropout_params.in.tertiary = &pp;
            dropout_params.in.secondary = rnd_state;
            dropout_params.out.primary = &output_bN[tid][k];
            dropout_params.out.secondary = &dropout_mask_bN[m][k];

            dropout_bf16(bn, bk, &dropout_params, dropout_flags);
          }
        }

        for(int k=0; k<nk; k++) {
          copy_params.in.primary = &output_bN[tid][k];
          copy_params.out.primary = &output[m*bn][k*bk];
          bf16_copy(bn, bk, nk*bk, nk*bk, &copy_params);
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
        bf16_copy(rn, bc, nc*bc, nc*bc, &copy_params);
      }

      if(pskip)
      {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &input_r[nn*bn][c*bc];
          copy_params.out.primary = &input_rN_r[0][c];
          bf16_copy(rn, bc, nc*bc, nc*bc, &copy_params);
        }

        for(int k=0; k<nk; k++) {
          copy_params.in.primary = bias_l;
          copy_params.out.primary = &output_rN_l[0][k];
          colbcast_bf16_copy(rn, bk, &copy_params);

          copy_params.in.primary = bias_r;
          copy_params.out.primary = &output_rN_r[0][k];
          colbcast_bf16_copy(rn, bk, &copy_params);
        }
      }
      else
      {
        for(int k=0; k<nk; k++) {
          copy_params.in.primary = bias_l;
          copy_params.out.primary = &output_rN[0][k];
          colbcast_bf16_copy(rn, bk, &copy_params);
        }
      }

      int count = nc;

      if(pskip)
      {
        brgemm_bf16_bf16(rn, bk, bcp, rn*bk, 0, input_rN_l[0][0][0], vnni_wt_l[0][0][0][0], output_rN_l[0][0][0], count);
        brgemm_bf16_bf16(rn, bk, bcp, rn*bk, 0, input_rN_r[0][0][0], vnni_wt_r[0][0][0][0], output_rN_r[0][0][0], count);
        add_params.in0.primary = (void*)&output_rN_l[0][0];
        add_params.in1.primary = (void*)&output_rN_r[0][0];
        add_params.out.primary = (void*)&output_rN[0][0];
        add_bf16_bf16(rn, bk, &add_params);
      }
      else
        brgemm_bf16_bf16(rn, bk, bcp, rn*bk, 0, input_rN_l[0][0][0], vnni_wt_l[0][0][0][0], output_rN[0][0][0], count);

      if(pact == 1)
      {
        for(int k=0; k<nk; k++) {
          relu_params.in.primary = &output_rN[0][k];
          relu_params.out.primary = &output_rN[0][k];
          relu_params.out.secondary = &relumask_rN[0][k];
          relu_fwd_bf16(rn, bk, &relu_params);
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

          dropout_bf16(rn, bk, &dropout_params, dropout_flags);
        }
      }

      for(int k=0; k<nk; k++) {
        copy_params.in.primary = &output_rN[0][k];
        copy_params.out.primary = &output[nn*bn][k*bk];
        bf16_copy(rn, bk, nk*bk, nk*bk, &copy_params);
      }
    }

    libxsmm_free((void*)scratch);

    return {t_output, t_relumask_bN, t_relumask_rN, t_dropout_mask_bN, t_dropout_mask_rN};
  }

////=====================================================================================================================================================
//// ====================== BackPass Function ===========================
////=====================================================================================================================================================

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

// ---------------- zero Padding to handle brgemm reduction -------------
    long bnp = (bn % 2 != 0) ? (bn + 1): bn;
    long rnp = (rn % 2 != 0) ? (rn + 1): rn;
    long bkp = (bk % 2 != 0) ? (bk + 1): bk;
// ----------------------------------------------------------------------

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

    long wts = nk*nc*bkp*bc;
    long go_bn_k = threads*nk*bn*bkp;
    long go_rn_k = nk*rn*bkp;
    long go_bn_n = threads*nk*bnp*bk;
    long go_rn_n = nk*rnp*bk;
    long gi_bn = threads*nc*bn*bc;
    long gi_rn = nc*rn*bc;
    long tr_go_bn = threads*nk*bnp*bk;
    long tr_go_rn = nk*rnp*bk;
    long in_v_bn = threads*nc*bnp*bc;
    long in_v_rn = nc*rnp*bc;
    long in_bn = threads*nc*bn*bc;
    long in_rn = nc*rn*bc;

    long scratch_size;
    if(pskip)
      scratch_size = (wts*4 + go_bn_k + go_rn_k + go_bn_n + go_rn_n + gi_bn*2 + gi_rn*2 + tr_go_bn + tr_go_rn + in_v_bn*2 + in_v_rn*2 + in_bn*2 + in_rn*2)*sizeof(libxsmm_bfloat16) + (nk*nc*bk*bc*2)*sizeof(float);
    else
      scratch_size = (wts*2 + go_bn_k + go_rn_k + go_bn_n + go_rn_n + gi_bn + gi_rn + tr_go_bn + tr_go_rn + in_v_bn + in_v_rn + in_bn + in_rn)*sizeof(libxsmm_bfloat16) + (nk*nc*bk*bc)*sizeof(float);
    void *scratch = libxsmm_aligned_malloc(scratch_size, 2097152);

    libxsmm_bfloat16* t_grad_output_bN_K    = (libxsmm_bfloat16*)scratch;
    libxsmm_bfloat16* t_grad_output_bN_N    = t_grad_output_bN_K + go_bn_k;
    libxsmm_bfloat16* t_tr_grad_output_bN   = t_grad_output_bN_N + go_bn_n;
    libxsmm_bfloat16* t_input_vnni_bN_l     = t_tr_grad_output_bN + tr_go_bn;
    libxsmm_bfloat16* t_grad_input_bN_l     = t_input_vnni_bN_l + in_v_bn;
    libxsmm_bfloat16* t_input_bN_l          = t_grad_input_bN_l + gi_bn;
    libxsmm_bfloat16* t_vnni_weights_l      = t_input_bN_l + in_bn;
    libxsmm_bfloat16* t_bf16_weights_l      = t_vnni_weights_l + wts;
               float* t_f32_grad_wt_l       = (float*)(t_bf16_weights_l + wts);

    libxsmm_bfloat16 *t_grad_output_rN_K=NULL, *t_grad_output_rN_N=NULL, *t_tr_grad_output_rN=NULL, *t_input_vnni_rN_l=NULL, *t_grad_input_rN_l=NULL;
    libxsmm_bfloat16 *t_input_rN_l=NULL;
    if(rn > 0)
    {
      t_grad_output_rN_K    = (libxsmm_bfloat16*)(t_f32_grad_wt_l + wts);
      t_grad_output_rN_N    = t_grad_output_rN_K + go_rn_k;
      t_tr_grad_output_rN   = t_grad_output_rN_N + go_rn_n;
      t_input_vnni_rN_l     = t_tr_grad_output_rN + tr_go_rn;
      t_grad_input_rN_l     = t_input_vnni_rN_l + in_v_rn;
      t_input_rN_l          = t_grad_input_rN_l + gi_rn;
    }

    libxsmm_bfloat16* t_input_vnni_bN_r=NULL, *t_grad_input_bN_r=NULL, *t_input_bN_r=NULL;
    libxsmm_bfloat16* t_vnni_weights_r=NULL, *t_bf16_weights_r=NULL, *t_input_vnni_rN_r=NULL, *t_grad_input_rN_r=NULL, *t_input_rN_r=NULL;
    float *t_f32_grad_wt_r=NULL;

    if(pskip)
    {
      if(rn > 0)
        t_input_vnni_bN_r = t_input_rN_l + in_rn;
      else
        t_input_vnni_bN_r = (libxsmm_bfloat16*)(t_f32_grad_wt_l + wts);

      t_grad_input_bN_r   = t_input_vnni_bN_r + in_v_bn;
      t_input_bN_r        = t_grad_input_bN_r + gi_bn;
      t_vnni_weights_r    = t_input_bN_r + in_bn;
      t_bf16_weights_r    = t_vnni_weights_r + wts;
      t_f32_grad_wt_r     = (float*)(t_bf16_weights_r + wts);

      if(rn > 0)
      {
        t_input_vnni_rN_r = (libxsmm_bfloat16*)(t_f32_grad_wt_r + wts);
        t_grad_input_rN_r = t_input_vnni_rN_r + in_v_rn;
        t_input_rN_r      = t_grad_input_rN_r + gi_rn;
      }
    }

    DECL_VLA_PTR_PT(float, wt_f32_l, [C], t_weights_l);
    DECL_VLA_PTR_PT(float, grad_wt_l, [C], t_grad_weights_l);
    float (*wt_f32_r)[C] = pskip ? (float (*)[C])t_weights_r.data_ptr<float>() : NULL;
    float (*grad_wt_r)[C] = pskip ? (float (*)[C])t_grad_weights_r.data_ptr<float>() : NULL;

    DECL_VLA_PTR_NPT(libxsmm_bfloat16, at::BFloat16, grad_output, [K], t_grad_output);
    DECL_VLA_PTR_NPT(libxsmm_bfloat16, at::BFloat16, input_l, [C], t_input_l);
    DECL_VLA_PTR_NPT(libxsmm_bfloat16, at::BFloat16, grad_input_l, [C], t_grad_input_l);
    DECL_VLA_PTR_NPT_CHECK_COND(pskip, libxsmm_bfloat16, at::BFloat16, input_r, [C], t_input_r);
    DECL_VLA_PTR_NPT_CHECK_COND(pskip, libxsmm_bfloat16, at::BFloat16, grad_input_r, [C], t_grad_input_r);

    DECL_VLA_PTR(libxsmm_bfloat16, grad_output_bN_K, [nk][bn][bkp], t_grad_output_bN_K);
    DECL_VLA_PTR(libxsmm_bfloat16, grad_output_bN_N, [nk][bnp][bk], t_grad_output_bN_N);
    DECL_VLA_PTR(libxsmm_bfloat16, tr_grad_output_bN, [nk][bk][bnp], t_tr_grad_output_bN);
    DECL_VLA_PTR(libxsmm_bfloat16, input_vnni_bN_l, [nc][bnp/2][bc][2], t_input_vnni_bN_l);
    DECL_VLA_PTR(libxsmm_bfloat16, grad_input_bN_l, [nc][bn][bc], t_grad_input_bN_l);
    DECL_VLA_PTR(libxsmm_bfloat16, input_bN_l, [nc][bn][bc], t_input_bN_l);
    DECL_VLA_PTR(libxsmm_bfloat16, vnni_wt_l, [nc][bkp/2][bc][2], t_vnni_weights_l);
    DECL_VLA_PTR(libxsmm_bfloat16, wt_l, [nc][bk][bc], t_bf16_weights_l);
    DECL_VLA_PTR(float, grad_wt_f32_l, [nc][bk][bc], t_f32_grad_wt_l);
    float *grad_bias_l = t_grad_bias_l.data_ptr<float>();

    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, input_vnni_bN_r, [nc][bnp/2][bc][2], t_input_vnni_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, grad_input_bN_r, [nc][bn][bc], t_grad_input_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, input_bN_r, [nc][bn][bc], t_input_bN_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, vnni_wt_r, [nc][bkp/2][bc][2], t_vnni_weights_r);
    DECL_VLA_PTR_CHECK_COND(pskip, libxsmm_bfloat16, wt_r, [nc][bk][bc], t_bf16_weights_r);
    DECL_VLA_PTR_CHECK_COND(pskip, float, grad_wt_f32_r, [nc][bk][bc], t_f32_grad_wt_r);
    float *grad_bias_r = pskip ? t_grad_bias_r.data_ptr<float>() : NULL;

    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, grad_output_rN_K, [nk][rn][bkp], t_grad_output_rN_K);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, grad_output_rN_N, [nk][rnp][bk], t_grad_output_rN_N);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, tr_grad_output_rN, [nk][bk][rnp], t_tr_grad_output_rN);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, input_vnni_rN_l, [nc][rnp/2][bc][2], t_input_vnni_rN_l);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, grad_input_rN_l, [nc][rn][bc], t_grad_input_rN_l);
    DECL_VLA_PTR_CHECK_VAR(rn, libxsmm_bfloat16, input_rN_l, [nc][rn][bc], t_input_rN_l);

    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, libxsmm_bfloat16, input_vnni_rN_r, [nc][rnp/2][bc][2], t_input_vnni_rN_r);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, libxsmm_bfloat16, grad_input_rN_r, [nc][rn][bc], t_grad_input_rN_r);
    DECL_VLA_PTR_CHECK_COND_VAR(pskip, rn, libxsmm_bfloat16, input_rN_r, [nc][rn][bc], t_input_rN_r);

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

    // Get BF16 copy of weights
    for(int k=0; k<nk; k++) {
      for(int c=0; c<nc; c++) {
        cvt_params.in.primary = &wt_f32_l[k*bk][c*bc];
        cvt_params.out.primary = &wt_l[k][c];
        cvt_f32_bf16(bk, bc, &cvt_params);
      }
    }
    int count = nk;
    norm_to_vnni_16b(wt_l[0][0][0], vnni_wt_l[0][0][0][0], bkp, bc); //bk x bc --> bkp/2 x bc x 2

    if(pskip)
    {
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
          cvt_params.in.primary = &wt_f32_r[k*bk][c*bc];
          cvt_params.out.primary = &wt_r[k][c];
          cvt_f32_bf16(bk, bc, &cvt_params);
        }
      }
      int count = nk;
      norm_to_vnni_16b(wt_r[0][0][0], vnni_wt_r[0][0][0][0], bkp, bc); //bk x bc --> bkp/2 x bc x 2
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
        libxsmm_meltw_unary_param  delbias_params;

        for(int m=tb; m<te; m++) {

          for(int k=0; k<nk; k++) {
            if(ptrain && pp > 0) {
              dropout_params.in.primary = &grad_output[m*bn][k*bk];
              dropout_params.in.secondary = &dropout_mask_bN[m][k][0][0];
              dropout_params.in.tertiary = &pp;
              dropout_params.out.primary = &grad_output[m*bn][k*bk];
              dropout_bwd_bf16(bn, bk, &dropout_params, dropout_flags);
            }

            if(pact == 1) {
              relu_params.in.primary = &grad_output[m*bn][k*bk];
              relu_params.in.secondary = &relumask_bN[m][k][0][0];
              relu_params.out.primary = &grad_output[m*bn][k*bk];
              relu_bwd_bf16(bn, bk, &relu_params);
            }

            copy_params.in.primary = &grad_output[m*bn][k*bk];
            copy_params.out.primary = &grad_output_bN_K[tid][k];
            bf16_copy(bn, bk, nk*bkp, nk*bk, &copy_params);
          }

          brgemm_bf16_bf16(bn, bc, bkp, bn*bkp, 0, grad_output_bN_K[tid][0][0], vnni_wt_l[0][0][0][0], grad_input_bN_l[tid][0][0], count, 0.0);

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &grad_input_bN_l[tid][c];
            copy_params.out.primary = &grad_input_l[m*bn][c*bc];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          brgemm_bf16_bf16(bn, bc, bkp, bn*bkp, 0, grad_output_bN_K[tid][0][0], vnni_wt_r[0][0][0][0], grad_input_bN_r[tid][0][0], count, 0.0);

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &grad_input_bN_r[tid][c];
            copy_params.out.primary = &grad_input_r[m*bn][c*bc];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          for(int k=0; k<nk; k++)
          {
            copy_params.in.primary = &grad_output[m*bn][k*bk];
            copy_params.out.primary = &grad_output_bN_N[tid][k];
            bf16_copy(bnp, bk, nk*bk, nk*bk, &copy_params);

            norm_to_normT_16b(grad_output_bN_N[tid][k][0], tr_grad_output_bN[tid][k][0], bnp, bk);
          }

          for(int c=0; c<nc; c++)
          {
            copy_params.in.primary = &input_l[m*bn][c*bc];
            copy_params.out.primary = &input_bN_l[tid][c];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
            norm_to_vnni_16b(input_bN_l[tid][c][0], input_vnni_bN_l[tid][c][0][0], bnp, bc);
          }

          count = 1;
          brgemm_bf16_f32(bk, bc, bnp, bnp*bk, bnp*bc, tr_grad_output_bN[tid][0][0], input_vnni_bN_l[tid][0][0][0], grad_wt_f32_l[0][0][0], count, 1.0);

          for(int c=0; c<nc; c++)
          {
            copy_params.in.primary = &input_r[m*bn][c*bc];
            copy_params.out.primary = &input_bN_r[tid][c];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
            norm_to_vnni_16b(input_bN_r[tid][c][0], input_vnni_bN_r[tid][c][0][0], bnp, bc);
          }

          count = 1;
          brgemm_bf16_f32(bk, bc, bnp, bnp*bk, bnp*bc, tr_grad_output_bN[tid][0][0], input_vnni_bN_r[tid][0][0][0], grad_wt_f32_r[0][0][0], count, 1.0);

          for(int k=0; k<nk; k++) {
            delbias_params.in.primary = &grad_output_bN_N[tid][k];
            delbias_params.out.primary = grad_bias_l;
            delbias_bf16_f32(bn, bk, bn, bk, &delbias_params);
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
              dropout_bwd_bf16(bn, bk, &dropout_params, dropout_flags);
            }

            if(pact == 1) {
              relu_params.in.primary = &grad_output[m*bn][k*bk];
              relu_params.in.secondary = &relumask_bN[m][k][0][0];
              relu_params.out.primary = &grad_output[m*bn][k*bk];
              relu_bwd_bf16(bn, bk, &relu_params);
            }

            copy_params.in.primary = &grad_output[m*bn][k*bk];
            copy_params.out.primary = &grad_output_bN_K[tid][k];
            bf16_copy(bn, bk, nk*bkp, nk*bk, &copy_params);
          }

          brgemm_bf16_bf16(bn, bc, bkp, bn*bkp, 0, grad_output_bN_K[tid][0][0], vnni_wt_l[0][0][0][0], grad_input_bN_l[tid][0][0], count, 0.0);

          for(int c=0; c<nc; c++) {
            copy_params.in.primary = &grad_input_bN_l[tid][c];
            copy_params.out.primary = &grad_input_l[m*bn][c*bc];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
          }

          for(int k=0; k<nk; k++)
          {
            copy_params.in.primary = &grad_output[m*bn][k*bk];
            copy_params.out.primary = &grad_output_bN_N[tid][k];
            bf16_copy(bnp, bk, nk*bk, nk*bk, &copy_params);

            norm_to_normT_16b(grad_output_bN_N[tid][k][0], tr_grad_output_bN[tid][k][0], bnp, bk);
          }

          for(int c=0; c<nc; c++)
          {
            copy_params.in.primary = &input_l[m*bn][c*bc];
            copy_params.out.primary = &input_bN_l[tid][c];
            bf16_copy(bn, bc, nc*bc, nc*bc, &copy_params);
            norm_to_vnni_16b(input_bN_l[tid][c][0], input_vnni_bN_l[tid][c][0][0], bnp, bc);
          }

          count = 1;
          brgemm_bf16_f32(bk, bc, bnp, bnp*bk, bnp*bc, tr_grad_output_bN[tid][0][0], input_vnni_bN_l[tid][0][0][0], grad_wt_f32_l[0][0][0], count, 1.0);

          for(int k=0; k<nk; k++) {
            delbias_params.in.primary = &grad_output_bN_N[tid][k];
            delbias_params.out.primary = grad_bias_l;
            delbias_bf16_f32(bn, bk, bn, bk, &delbias_params);
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
          dropout_bwd_bf16(rn, bk, &dropout_params, dropout_flags);
        }
      }

      // ReLU
      if(pact == 1)
      {
        for(int k=0; k<nk; k++) {
          relu_params.in.primary = &grad_output[nn*bn][k*bk];
          relu_params.in.secondary = &relumask_rN[0][k][0][0];
          relu_params.out.primary = &grad_output[nn*bn][k*bk];
          relu_bwd_bf16(rn, bk, &relu_params);
        }
      }

      //grad-input
      for(int k=0; k<nk; k++) {
        copy_params.in.primary = &grad_output[nn*bn][k*bk];
        copy_params.out.primary = &grad_output_rN_K[0][k];
        bf16_copy(rn, bk, nk*bkp, nk*bk, &copy_params);
      }
      brgemm_bf16_bf16(rn, bc, bkp, rn*bkp, 0, grad_output_rN_K[0][0][0], vnni_wt_l[0][0][0][0], grad_input_rN_l[0][0][0], count, 0.0);

      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &grad_input_rN_l[0][c];
        copy_params.out.primary = &grad_input_l[nn*bn][c*bc];
        bf16_copy(rn, bc, nc*bc, nc*bc, &copy_params);
      }

      if(pskip)
      {
        brgemm_bf16_bf16(rn, bc, bkp, rn*bkp, 0, grad_output_rN_K[0][0][0], vnni_wt_r[0][0][0][0], grad_input_rN_r[0][0][0], count, 0.0);

        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &grad_input_rN_r[0][c];
          copy_params.out.primary = &grad_input_r[nn*bn][c*bc];
          bf16_copy(rn, bc, nc*bc, nc*bc, &copy_params);
        }
      }

      //grad-weights
      for(int k=0; k<nk; k++) {
        copy_params.in.primary = &grad_output[nn*bn][k*bk];
        copy_params.out.primary = &grad_output_rN_N[0][k];
        bf16_copy(rn, bk, nk*bk, nk*bk, &copy_params);
        norm_to_normT_16b(grad_output_rN_N[0][k][0], tr_grad_output_rN[0][k][0], rnp, bk);
      }
      for(int c=0; c<nc; c++) {
        copy_params.in.primary = &input_l[nn*bn][c*bc];
        copy_params.out.primary = &input_rN_l[0][c];
        bf16_copy(rn, bc, nc*bc, nc*bc, &copy_params);
      }
      for(int c=0; c<nc; c++)
        norm_to_vnni_16b(input_rN_l[0][c][0], input_vnni_rN_l[0][c][0][0], rnp, bc);

      count = 1;
      brgemm_bf16_f32(bk, bc, rnp, rnp*bk, rnp*bc, tr_grad_output_rN[0][0][0], input_vnni_rN_l[0][0][0][0], grad_wt_f32_l[0][0][0], count, 1.0);

      if(pskip)
      {
        for(int c=0; c<nc; c++) {
          copy_params.in.primary = &input_r[nn*bn][c*bc];
          copy_params.out.primary = &input_rN_r[0][c];
          bf16_copy(rn, bc, nc*bc, nc*bc, &copy_params);
        }
        for(int c=0; c<nc; c++)
          norm_to_vnni_16b(input_rN_r[0][c][0], input_vnni_rN_r[0][c][0][0], rnp, bc);

        count = 1;
        brgemm_bf16_f32(bk, bc, rnp, rnp*bk, rnp*bc, tr_grad_output_rN[0][0][0], input_vnni_rN_r[0][0][0][0], grad_wt_f32_r[0][0][0], count, 1.0);
      }

      for(int k=0; k<nk; k++) {
        delbias_params.in.primary = &grad_output_rN_N[0][k];
        delbias_params.out.primary = grad_bias_l;
        delbias_bf16_f32(rn, bk, rn, bk, &delbias_params);
      }

      if(pskip)
      {
        for(int k=0; k<nk; k++) {
          delbias_params.in.primary = &grad_output_rN_N[0][k];
          delbias_params.out.primary = grad_bias_r;
          delbias_bf16_f32(rn, bk, rn, bk, &delbias_params);
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

