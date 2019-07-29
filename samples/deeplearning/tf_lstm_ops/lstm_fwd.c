#include <libxsmm.h>

#if defined(__linux__)
# include <sys/syscall.h>
# define gettid() syscall(SYS_gettid)
#else
# define gettid() libxsmm_get_tid()
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include "lstm_fwd.h"
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );
#if 0
# define PRINT_LAYOUT(DESC, LAYOUT) print_layout(DESC, LAYOUT)
#else
# define PRINT_LAYOUT(DESC, LAYOUT)
#endif

void print_layout(char *desc, libxsmm_dnn_tensor_datalayout *layout) {
  char *dim_name[] = {"N", "H", "W", "C", "K", "R", "S", "X", "RLM", "RLK", "RLN"};
  int i;
  printf("%s: F:%d TT: %d [", desc, layout->format, layout->tensor_type);
  for(i = layout->num_dims - 1; i >= 0; i--) {
    printf("%s:%d%s", dim_name[layout->dim_type[i]], layout->dim_size[i], i == 0 ? "" : ", ");
  }
  printf("]\n");
}

static void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

void* lstm_fwd_create( int N, /* minibatch size */
                       int C, /* input size     */
                       int K, /* output size    */
                       int t, /* timesteps = 1  */
                       int nThreads, /* number of threads */
                       const float forget_bias,
                       const int w_in_kcck,
                       const float *xt,
                       const float *csp,
                       const float *hp,
                       const float *w,
                       const float *r,
                       const float *b,
                       float *cst,
                       float *ht,
                       float *it,
                       float *ft,
                       float *ot,
                       float *cit,
                       float *cot )
{
  libxsmm_dnn_rnncell_desc lstmcell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_cs_prev;
  libxsmm_dnn_tensor* libxsmm_hidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  libxsmm_dnn_tensor* libxsmm_bias;
  libxsmm_dnn_tensor* libxsmm_cs;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_i;
  libxsmm_dnn_tensor* libxsmm_f;
  libxsmm_dnn_tensor* libxsmm_o;
  libxsmm_dnn_tensor* libxsmm_ci;
  libxsmm_dnn_tensor* libxsmm_co;
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;

  if (N <= 0) {
    printf("N: %d should be > 0\n", N);
  }
  if (C <= 0) {
    printf("C: %d should be > 0\n", C);
  }
  if (K <= 0) {
    printf("K: %d should be > 0\n", K);
  }
  if (xt == 0  || csp == 0 || hp == 0 || w == 0  || r == 0  || b == 0   ||
      cst == 0 || ht == 0  || it == 0 || ft == 0 || ot == 0 || cit == 0 || cot == 0) {
    printf("None of the pointers should be NULL::\n");
    printf("x:%p\n", xt);
    printf("cs_prev:%p\n", csp);
    printf("h_prev:%p\n", hp);
    printf("w:%p\n", w);
    printf("r:%p\n", r);
    printf("b:%p\n", b);
    printf("cs:%p\n", cst);
    printf("h:%p\n", ht);
    printf("i:%p\n", it);
    printf("f:%p\n", ft);
    printf("o:%p\n", ot);
    printf("ci:%p\n", cit);
    printf("co:%p\n", cot);
  }

  /* setup LIBXSMM handle */
  lstmcell_desc.threads = nThreads;
  lstmcell_desc.N = N;
  lstmcell_desc.C = C;
  lstmcell_desc.K = K;
  lstmcell_desc.max_T = t;
  lstmcell_desc.bn = 24;
  if(N % 24 == 0) lstmcell_desc.bn = 24;
  else if(N % 16 == 0) lstmcell_desc.bn = 16;
  else if(N % 12 == 0) lstmcell_desc.bn = 12;
  else if(N % 8 == 0) lstmcell_desc.bn = 8;
  else if(N % 6 == 0) lstmcell_desc.bn = 6;
  lstmcell_desc.bc = 64;
  lstmcell_desc.bk = 64;
  lstmcell_desc.cell_type = LIBXSMM_DNN_RNNCELL_LSTM;
  lstmcell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  lstmcell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  lstmcell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NC;
  lstmcell_desc.filter_format = (w_in_kcck ? LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED : LIBXSMM_DNN_TENSOR_FORMAT_CK);

  libxsmm_handle = libxsmm_dnn_create_rnncell( lstmcell_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_allocate_forget_bias(libxsmm_handle, forget_bias) );

  /* setup LIBXSMM buffers and filter */
  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("Xt", libxsmm_layout);
  libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_CS_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("CSP", libxsmm_layout);
  libxsmm_cs_prev = libxsmm_dnn_link_tensor( libxsmm_layout, csp, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("HP", libxsmm_layout);
  libxsmm_hidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, hp, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("W", libxsmm_layout);
  libxsmm_weight = libxsmm_dnn_link_tensor( libxsmm_layout, w, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("R", libxsmm_layout);
  libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, r, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("B", libxsmm_layout);
  libxsmm_bias = libxsmm_dnn_link_tensor( libxsmm_layout, b, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_CS, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("CSt", libxsmm_layout);
  libxsmm_cs = libxsmm_dnn_link_tensor( libxsmm_layout, cst, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("Ht", libxsmm_layout);
  libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("It", libxsmm_layout);
  libxsmm_i = libxsmm_dnn_link_tensor( libxsmm_layout, it, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("Ft", libxsmm_layout);
  libxsmm_f = libxsmm_dnn_link_tensor( libxsmm_layout, ft, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("Ot", libxsmm_layout);
  libxsmm_o = libxsmm_dnn_link_tensor( libxsmm_layout, ot, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("CIt", libxsmm_layout);
  libxsmm_ci = libxsmm_dnn_link_tensor( libxsmm_layout, cit, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CO, &status ); CHKERR_LIBXSMM_DNN( status );
  PRINT_LAYOUT("COt", libxsmm_layout);
  libxsmm_co = libxsmm_dnn_link_tensor( libxsmm_layout, cot, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

  /* bind buffers and filter to handle */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_cs_prev, LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state_prev, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_cs, LIBXSMM_DNN_RNN_REGULAR_CS ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_i, LIBXSMM_DNN_RNN_INTERNAL_I ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_f, LIBXSMM_DNN_RNN_INTERNAL_F ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_o, LIBXSMM_DNN_RNN_INTERNAL_O ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_ci, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_co, LIBXSMM_DNN_RNN_INTERNAL_CO ) );

  size_t scratch_size = libxsmm_dnn_rnncell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
  CHKERR_LIBXSMM_DNN( status );
  if(scratch_size > 0) {
    void *scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) );
    zero_buf( (float*)scratch, scratch_size/4 );
  }
  return (void*)libxsmm_handle;
}

void lstm_fwd_set_ptr( void* libxsmm_handle_,
                       const float forget_bias,
                       const int t,
                       const float *xt,
                       const float *csp,
                       const float *hp,
                       const float *w,
                       const float *r,
                       const float *b,
                       float *cst,
                       float *ht,
                       float *it,
                       float *ft,
                       float *ot,
                       float *cit,
                       float *cot )
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_dnn_rnncell* handle = (libxsmm_dnn_rnncell*) libxsmm_handle_;
  if (xt == 0  || csp == 0 || hp == 0 || w == 0  || r == 0  || b == 0   ||
      cst == 0 || ht == 0  || it == 0 || ft == 0 || ot == 0 || cit == 0 || cot == 0) {
    printf("None of the pointers should be NULL::\n");
    printf("x:%p\n", xt);
    printf("cs_prev:%p\n", csp);
    printf("h_prev:%p\n", hp);
    printf("w:%p\n", w);
    printf("r:%p\n", r);
    printf("b:%p\n", b);
    printf("cs:%p\n", cst);
    printf("h:%p\n", ht);
    printf("i:%p\n", it);
    printf("f:%p\n", ft);
    printf("o:%p\n", ot);
    printf("ci:%p\n", cit);
    printf("co:%p\n", cot);
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_allocate_forget_bias(handle, forget_bias) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_set_sequence_length( handle, t) );

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status), xt) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_CS_PREV, &status), csp) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV, &status), hp) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status), w) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status), r) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_BIAS, &status), b) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_CS, &status), cst) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status), ht) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_I, &status), it) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_F, &status), ft) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_O, &status), ot) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_CI, &status), cit) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_CO, &status), cot) );
}

void lstm_fwd_execute_omp( void* libxsmm_handle_)
{
#ifdef _OPENMP
  libxsmm_dnn_rnncell* handle = (libxsmm_dnn_rnncell*) libxsmm_handle_;
  /* run LIBXSMM LSTM FWD */
#pragma omp parallel
{
  int tid = omp_get_thread_num();
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
}
#else
  printf("%s:%d Shouldn't come here... exiting\n", __FILE__, __LINE__);
  exit(1);
#endif
}

void lstm_fwd_execute_st( void* libxsmm_handle_, int tid )
{
  libxsmm_dnn_rnncell* handle = (libxsmm_dnn_rnncell*) libxsmm_handle_;
  /* run LIBXSMM LSTM FWD */
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
}

void lstm_fwd_destroy( void* libxsmm_handle_ )
{
  libxsmm_dnn_rnncell* handle = (libxsmm_dnn_rnncell*) libxsmm_handle_;
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_CS_PREV, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_BIAS, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_CS, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_I, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_F, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_O, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_CI, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dnn_rnncell_get_tensor(handle, LIBXSMM_DNN_RNN_INTERNAL_CO, &status) ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_CS ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_INTERNAL_I ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_INTERNAL_F ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_INTERNAL_O ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( handle, LIBXSMM_DNN_RNN_INTERNAL_CO ) );

  size_t scratch_size = libxsmm_dnn_rnncell_get_scratch_size( handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
  if(scratch_size > 0) {
    void *scratch = libxsmm_dnn_rnncell_get_scratch_ptr( handle, /*LIBXSMM_DNN_COMPUTE_KIND_FWD,*/ &status );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_scratch( handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
    if(scratch) libxsmm_free(scratch);
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_rnncell( handle ) );
}

