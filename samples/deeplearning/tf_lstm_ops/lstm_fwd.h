#ifndef _LSTM_FWD_H_
#define _LSTM_FWD_H_

#ifdef __cplusplus
extern "C" {
#endif

void *lstm_fwd_create( int N, /* minibatch size */
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
                      float *cot );

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
                       float *cot );

void lstm_fwd_execute_omp( void* libxsmm_handle_);
void lstm_fwd_execute_st( void* libxsmm_handle_, int tid );
void lstm_fwd_destroy( void* libxsmm_handle_ );

#ifdef __cplusplus
}
#endif

#endif /*_LSTM_FWD_H_*/
