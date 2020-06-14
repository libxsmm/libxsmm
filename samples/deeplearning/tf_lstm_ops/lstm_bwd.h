#ifndef _LSTM_BWD_H_
#define _LSTM_BWD_H_

#ifdef __cplusplus
extern "C" {
#endif

void* lstm_bwd_create( int N, /* minibatch size */
                       int C, /* input size     */
                       int K, /* output size    */
                       int t, /* timesteps = 1  */
                       int nThreads, /* number of threads */
                       const int w_in_kcck,
                       const int w_in_trans,
                       const float *xt,
                       const float *csp,
                       const float *hp,
                       const float *ht,
                       const float *w,
                       const float *r,
                       const float *cst,
                       const float *it,
                       const float *ft,
                       const float *ot,
                       const float *cit,
                       const float *cot,
                       const float *dcs,
                       const float *dht,
                       float *dxt,
                       float *dcspt,
                       float *dhpt,
                       float *dw,
                       float *dr,
                       float *db );

void lstm_bwd_set_ptr( void* libxsmm_handle_, int w_in_trans,
                       const int t,
                       const float *xt,
                       const float *csp,
                       const float *hp,
                       const float *ht,
                       const float *w,
                       const float *r,
                       const float *cst,
                       const float *it,
                       const float *ft,
                       const float *ot,
                       const float *cit,
                       const float *cot,
                       const float *dcs,
                       const float *dht,
                       float *dxt,
                       float *dcspt,
                       float *dhpt,
                       float *dw,
                       float *dr,
                       float *db );

void lstm_bwd_execute_omp( void* libxsmm_handle_ );
void lstm_bwd_execute_st( void* libxsmm_handle_, int tid );
void lstm_bwd_destroy( void* libxsmm_handle_ );

#ifdef __cplusplus
}
#endif

#endif /*_LSTM_BWD_H_*/
