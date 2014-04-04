#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_1_11_11(const double* a, const double* b, double* c){
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
__m512d xa9;
__m512d xa10;
__m512d xb0;
__m512d xb1;
__m512d xb2;
__m512d xb3;
__m512d xb4;
__m512d xb5;
__m512d xb6;
__m512d xb7;
__m512d xb8;
__m512d xb9;
__m512d xb10;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[11+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[22+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[33+0],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[44+0],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[55+0],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[66+0],255);
 xb7 = _MM512_MASK_LOADU_PD(&b[77+0],255);
 xb8 = _MM512_MASK_LOADU_PD(&b[88+0],255);
 xb9 = _MM512_MASK_LOADU_PD(&b[99+0],255);
 xb10 = _MM512_MASK_LOADU_PD(&b[110+0],255);
for(i=0;i<1;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*11+0],255);
    xa0=_mm512_set1_pd(a[i*11+0]);
    xa1=_mm512_set1_pd(a[i*11+1]);
    xa2=_mm512_set1_pd(a[i*11+2]);
    xa3=_mm512_set1_pd(a[i*11+3]);
    xa4=_mm512_set1_pd(a[i*11+4]);
    xa5=_mm512_set1_pd(a[i*11+5]);
    xa6=_mm512_set1_pd(a[i*11+6]);
    xa7=_mm512_set1_pd(a[i*11+7]);
    xa8=_mm512_set1_pd(a[i*11+8]);
    xa9=_mm512_set1_pd(a[i*11+9]);
    xa10=_mm512_set1_pd(a[i*11+10]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa7,xb7,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa8,xb8,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa9,xb9,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa10,xb10,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*11+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],7);
 xb1 = _MM512_MASK_LOADU_PD(&b[11+8],7);
 xb2 = _MM512_MASK_LOADU_PD(&b[22+8],7);
 xb3 = _MM512_MASK_LOADU_PD(&b[33+8],7);
 xb4 = _MM512_MASK_LOADU_PD(&b[44+8],7);
 xb5 = _MM512_MASK_LOADU_PD(&b[55+8],7);
 xb6 = _MM512_MASK_LOADU_PD(&b[66+8],7);
 xb7 = _MM512_MASK_LOADU_PD(&b[77+8],7);
 xb8 = _MM512_MASK_LOADU_PD(&b[88+8],7);
 xb9 = _MM512_MASK_LOADU_PD(&b[99+8],7);
 xb10 = _MM512_MASK_LOADU_PD(&b[110+8],7);
for(i=0;i<1;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*11+8],7);
    xa0=_mm512_set1_pd(a[i*11+0]);
    xa1=_mm512_set1_pd(a[i*11+1]);
    xa2=_mm512_set1_pd(a[i*11+2]);
    xa3=_mm512_set1_pd(a[i*11+3]);
    xa4=_mm512_set1_pd(a[i*11+4]);
    xa5=_mm512_set1_pd(a[i*11+5]);
    xa6=_mm512_set1_pd(a[i*11+6]);
    xa7=_mm512_set1_pd(a[i*11+7]);
    xa8=_mm512_set1_pd(a[i*11+8]);
    xa9=_mm512_set1_pd(a[i*11+9]);
    xa10=_mm512_set1_pd(a[i*11+10]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa7,xb7,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa8,xb8,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa9,xb9,xc0,7);
    xc0=_mm512_mask3_fmadd_pd(xa10,xb10,xc0,7);
    _MM512_MASK_STOREU_PD(&c[i*11+8],xc0,7);
}
#else
printf("cppgemm_2_1_11_11\n");
for(int m=0;m<1;m++){
   for(int n=0;n<11;n++){
      for(int k=0;k<11;k++){
         c[m*11+n]+=a[m*11+k]*b[k*11+n];
      }
   }
}
#endif
}
 
