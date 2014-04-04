#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_2_16_9(const double* a, const double* b, double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xa2;
__m512d xa3;
__m512d xa4;
__m512d xa5;
__m512d xa6;
__m512d xa7;
__m512d xa8;
__m512d xb0;
__m512d xb1;
__m512d xb2;
__m512d xb3;
__m512d xb4;
__m512d xb5;
__m512d xb6;
__m512d xb7;
__m512d xb8;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[16+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[32+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[48+0],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[64+0],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[80+0],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[96+0],255);
 xb7 = _MM512_MASK_LOADU_PD(&b[112+0],255);
 xb8 = _MM512_MASK_LOADU_PD(&b[128+0],255);
for(i=0;i<2;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*16+0],255);
    xa0=_mm512_set1_pd(a[i*9+0]);
    xa1=_mm512_set1_pd(a[i*9+1]);
    xa2=_mm512_set1_pd(a[i*9+2]);
    xa3=_mm512_set1_pd(a[i*9+3]);
    xa4=_mm512_set1_pd(a[i*9+4]);
    xa5=_mm512_set1_pd(a[i*9+5]);
    xa6=_mm512_set1_pd(a[i*9+6]);
    xa7=_mm512_set1_pd(a[i*9+7]);
    xa8=_mm512_set1_pd(a[i*9+8]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa7,xb7,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa8,xb8,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*16+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[16+8],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[32+8],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[48+8],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[64+8],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[80+8],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[96+8],255);
 xb7 = _MM512_MASK_LOADU_PD(&b[112+8],255);
 xb8 = _MM512_MASK_LOADU_PD(&b[128+8],255);
for(i=0;i<2;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*16+8],255);
    xa0=_mm512_set1_pd(a[i*9+0]);
    xa1=_mm512_set1_pd(a[i*9+1]);
    xa2=_mm512_set1_pd(a[i*9+2]);
    xa3=_mm512_set1_pd(a[i*9+3]);
    xa4=_mm512_set1_pd(a[i*9+4]);
    xa5=_mm512_set1_pd(a[i*9+5]);
    xa6=_mm512_set1_pd(a[i*9+6]);
    xa7=_mm512_set1_pd(a[i*9+7]);
    xa8=_mm512_set1_pd(a[i*9+8]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa7,xb7,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa8,xb8,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*16+8],xc0,255);
}
#else
printf("cppgemm_2_2_9_16\n");
for(int m=0;m<2;m++){
   for(int n=0;n<16;n++){
      for(int k=0;k<9;k++){
         c[m*16+n]+=a[m*9+k]*b[k*16+n];
      }
   }
}
#endif
}
 
