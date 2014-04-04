#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_2_10_3(const double* a, const double* b, double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xa2;
__m512d xb0;
__m512d xb1;
__m512d xb2;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[10+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[20+0],255);
for(i=0;i<2;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*10+0],255);
    xa0=_mm512_set1_pd(a[i*3+0]);
    xa1=_mm512_set1_pd(a[i*3+1]);
    xa2=_mm512_set1_pd(a[i*3+2]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*10+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],3);
 xb1 = _MM512_MASK_LOADU_PD(&b[10+8],3);
 xb2 = _MM512_MASK_LOADU_PD(&b[20+8],3);
for(i=0;i<2;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*10+8],3);
    xa0=_mm512_set1_pd(a[i*3+0]);
    xa1=_mm512_set1_pd(a[i*3+1]);
    xa2=_mm512_set1_pd(a[i*3+2]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,3);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,3);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,3);
    _MM512_MASK_STOREU_PD(&c[i*10+8],xc0,3);
}
#else
printf("cppgemm_2_2_3_10\n");
for(int m=0;m<2;m++){
   for(int n=0;n<10;n++){
      for(int k=0;k<3;k++){
         c[m*10+n]+=a[m*3+k]*b[k*10+n];
      }
   }
}
#endif
}
 
