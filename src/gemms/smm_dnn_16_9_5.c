#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec( target (mic))
void smm_dnn_16_9_5(double* a,double* b,double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xa2;
__m512d xa3;
__m512d xa4;
__m512d xb0;
__m512d xb1;
__m512d xb2;
__m512d xb3;
__m512d xb4;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[9+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[18+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[27+0],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[36+0],255);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*9+0],255);
    xa0=_mm512_set1_pd(a[i*5+0]);
    xa1=_mm512_set1_pd(a[i*5+1]);
    xa2=_mm512_set1_pd(a[i*5+2]);
    xa3=_mm512_set1_pd(a[i*5+3]);
    xa4=_mm512_set1_pd(a[i*5+4]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*9+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],1);
 xb1 = _MM512_MASK_LOADU_PD(&b[9+8],1);
 xb2 = _MM512_MASK_LOADU_PD(&b[18+8],1);
 xb3 = _MM512_MASK_LOADU_PD(&b[27+8],1);
 xb4 = _MM512_MASK_LOADU_PD(&b[36+8],1);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*9+8],1);
    xa0=_mm512_set1_pd(a[i*5+0]);
    xa1=_mm512_set1_pd(a[i*5+1]);
    xa2=_mm512_set1_pd(a[i*5+2]);
    xa3=_mm512_set1_pd(a[i*5+3]);
    xa4=_mm512_set1_pd(a[i*5+4]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,1);
    _MM512_MASK_STOREU_PD(&c[i*9+8],xc0,1);
}
#else
printf("cppgemm_2_16_5_9\n");
for(int m=0;m<16;m++){
   for(int n=0;n<9;n++){
      for(int k=0;k<5;k++){
         c[m*9+n]+=a[m*5+k]*b[k*9+n];
      }
   }
}
#endif
}
 
