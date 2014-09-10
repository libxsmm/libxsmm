#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_6_23_4(const double* a, const double* b, double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xa2;
__m512d xa3;
__m512d xb0;
__m512d xb1;
__m512d xb2;
__m512d xb3;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[23+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[46+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[69+0],255);
for(i=0;i<6;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*23+0],255);
    xa0=_mm512_set1_pd(a[i*4+0]);
    xa1=_mm512_set1_pd(a[i*4+1]);
    xa2=_mm512_set1_pd(a[i*4+2]);
    xa3=_mm512_set1_pd(a[i*4+3]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*23+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[23+8],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[46+8],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[69+8],255);
for(i=0;i<6;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*23+8],255);
    xa0=_mm512_set1_pd(a[i*4+0]);
    xa1=_mm512_set1_pd(a[i*4+1]);
    xa2=_mm512_set1_pd(a[i*4+2]);
    xa3=_mm512_set1_pd(a[i*4+3]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*23+8],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+16],127);
 xb1 = _MM512_MASK_LOADU_PD(&b[23+16],127);
 xb2 = _MM512_MASK_LOADU_PD(&b[46+16],127);
 xb3 = _MM512_MASK_LOADU_PD(&b[69+16],127);
for(i=0;i<6;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*23+16],127);
    xa0=_mm512_set1_pd(a[i*4+0]);
    xa1=_mm512_set1_pd(a[i*4+1]);
    xa2=_mm512_set1_pd(a[i*4+2]);
    xa3=_mm512_set1_pd(a[i*4+3]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,127);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,127);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,127);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,127);
    _MM512_MASK_STOREU_PD(&c[i*23+16],xc0,127);
}
#else
printf("cppgemm_2_6_4_23\n");
for(int m=0;m<6;m++){
   for(int n=0;n<23;n++){
      for(int k=0;k<4;k++){
         c[m*23+n]+=a[m*4+k]*b[k*23+n];
      }
   }
}
#endif
}
 
