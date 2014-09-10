#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_13_20_1(const double* a, const double* b, double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xb0;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
for(i=0;i<13;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*20+0],255);
    xa0=_mm512_set1_pd(a[i*1+0]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*20+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],255);
for(i=0;i<13;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*20+8],255);
    xa0=_mm512_set1_pd(a[i*1+0]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*20+8],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+16],15);
for(i=0;i<13;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*20+16],15);
    xa0=_mm512_set1_pd(a[i*1+0]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,15);
    _MM512_MASK_STOREU_PD(&c[i*20+16],xc0,15);
}
#else
printf("cppgemm_2_13_1_20\n");
for(int m=0;m<13;m++){
   for(int n=0;n<20;n++){
      for(int k=0;k<1;k++){
         c[m*20+n]+=a[m*1+k]*b[k*20+n];
      }
   }
}
#endif
}
 
