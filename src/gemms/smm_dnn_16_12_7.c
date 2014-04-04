#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec( target (mic))
void smm_dnn_16_12_7(double* a,double* b,double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xa2;
__m512d xa3;
__m512d xa4;
__m512d xa5;
__m512d xa6;
__m512d xb0;
__m512d xb1;
__m512d xb2;
__m512d xb3;
__m512d xb4;
__m512d xb5;
__m512d xb6;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[12+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[24+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[36+0],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[48+0],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[60+0],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[72+0],255);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*12+0],255);
    xa0=_mm512_set1_pd(a[i*7+0]);
    xa1=_mm512_set1_pd(a[i*7+1]);
    xa2=_mm512_set1_pd(a[i*7+2]);
    xa3=_mm512_set1_pd(a[i*7+3]);
    xa4=_mm512_set1_pd(a[i*7+4]);
    xa5=_mm512_set1_pd(a[i*7+5]);
    xa6=_mm512_set1_pd(a[i*7+6]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*12+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],15);
 xb1 = _MM512_MASK_LOADU_PD(&b[12+8],15);
 xb2 = _MM512_MASK_LOADU_PD(&b[24+8],15);
 xb3 = _MM512_MASK_LOADU_PD(&b[36+8],15);
 xb4 = _MM512_MASK_LOADU_PD(&b[48+8],15);
 xb5 = _MM512_MASK_LOADU_PD(&b[60+8],15);
 xb6 = _MM512_MASK_LOADU_PD(&b[72+8],15);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*12+8],15);
    xa0=_mm512_set1_pd(a[i*7+0]);
    xa1=_mm512_set1_pd(a[i*7+1]);
    xa2=_mm512_set1_pd(a[i*7+2]);
    xa3=_mm512_set1_pd(a[i*7+3]);
    xa4=_mm512_set1_pd(a[i*7+4]);
    xa5=_mm512_set1_pd(a[i*7+5]);
    xa6=_mm512_set1_pd(a[i*7+6]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,15);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,15);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,15);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,15);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,15);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,15);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,15);
    _MM512_MASK_STOREU_PD(&c[i*12+8],xc0,15);
}
#else
printf("cppgemm_2_16_7_12\n");
for(int m=0;m<16;m++){
   for(int n=0;n<12;n++){
      for(int k=0;k<7;k++){
         c[m*12+n]+=a[m*7+k]*b[k*12+n];
      }
   }
}
#endif
}
 
