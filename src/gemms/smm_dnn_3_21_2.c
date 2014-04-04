#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec( target (mic))
void smm_dnn_3_21_2(double* a,double* b,double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xb0;
__m512d xb1;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[21+0],255);
for(i=0;i<3;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*21+0],255);
    xa0=_mm512_set1_pd(a[i*2+0]);
    xa1=_mm512_set1_pd(a[i*2+1]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*21+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[21+8],255);
for(i=0;i<3;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*21+8],255);
    xa0=_mm512_set1_pd(a[i*2+0]);
    xa1=_mm512_set1_pd(a[i*2+1]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*21+8],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+16],31);
 xb1 = _MM512_MASK_LOADU_PD(&b[21+16],31);
for(i=0;i<3;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*21+16],31);
    xa0=_mm512_set1_pd(a[i*2+0]);
    xa1=_mm512_set1_pd(a[i*2+1]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,31);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,31);
    _MM512_MASK_STOREU_PD(&c[i*21+16],xc0,31);
}
#else
printf("cppgemm_2_3_2_21\n");
for(int m=0;m<3;m++){
   for(int n=0;n<21;n++){
      for(int k=0;k<2;k++){
         c[m*21+n]+=a[m*2+k]*b[k*21+n];
      }
   }
}
#endif
}
 
