#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec( target (mic))
void smm_dnn_10_10_1(double* a,double* b,double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xb0;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
for(i=0;i<10;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*10+0],255);
    xa0=_mm512_set1_pd(a[i*1+0]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*10+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],3);
for(i=0;i<10;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*10+8],3);
    xa0=_mm512_set1_pd(a[i*1+0]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,3);
    _MM512_MASK_STOREU_PD(&c[i*10+8],xc0,3);
}
#else
printf("cppgemm_2_10_1_10\n");
for(int m=0;m<10;m++){
   for(int n=0;n<10;n++){
      for(int k=0;k<1;k++){
         c[m*10+n]+=a[m*1+k]*b[k*10+n];
      }
   }
}
#endif
}
 
