#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_7_9_18(const double* a, const double* b, double* c){
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
__m512d xa11;
__m512d xa12;
__m512d xa13;
__m512d xa14;
__m512d xa15;
__m512d xa16;
__m512d xa17;
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
__m512d xb11;
__m512d xb12;
__m512d xb13;
__m512d xb14;
__m512d xb15;
__m512d xb16;
__m512d xb17;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[9+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[18+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[27+0],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[36+0],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[45+0],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[54+0],255);
 xb7 = _MM512_MASK_LOADU_PD(&b[63+0],255);
 xb8 = _MM512_MASK_LOADU_PD(&b[72+0],255);
 xb9 = _MM512_MASK_LOADU_PD(&b[81+0],255);
 xb10 = _MM512_MASK_LOADU_PD(&b[90+0],255);
 xb11 = _MM512_MASK_LOADU_PD(&b[99+0],255);
 xb12 = _MM512_MASK_LOADU_PD(&b[108+0],255);
 xb13 = _MM512_MASK_LOADU_PD(&b[117+0],255);
 xb14 = _MM512_MASK_LOADU_PD(&b[126+0],255);
 xb15 = _MM512_MASK_LOADU_PD(&b[135+0],255);
 xb16 = _MM512_MASK_LOADU_PD(&b[144+0],255);
 xb17 = _MM512_MASK_LOADU_PD(&b[153+0],255);
for(i=0;i<7;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*9+0],255);
    xa0=_mm512_set1_pd(a[i*18+0]);
    xa1=_mm512_set1_pd(a[i*18+1]);
    xa2=_mm512_set1_pd(a[i*18+2]);
    xa3=_mm512_set1_pd(a[i*18+3]);
    xa4=_mm512_set1_pd(a[i*18+4]);
    xa5=_mm512_set1_pd(a[i*18+5]);
    xa6=_mm512_set1_pd(a[i*18+6]);
    xa7=_mm512_set1_pd(a[i*18+7]);
    xa8=_mm512_set1_pd(a[i*18+8]);
    xa9=_mm512_set1_pd(a[i*18+9]);
    xa10=_mm512_set1_pd(a[i*18+10]);
    xa11=_mm512_set1_pd(a[i*18+11]);
    xa12=_mm512_set1_pd(a[i*18+12]);
    xa13=_mm512_set1_pd(a[i*18+13]);
    xa14=_mm512_set1_pd(a[i*18+14]);
    xa15=_mm512_set1_pd(a[i*18+15]);
    xa16=_mm512_set1_pd(a[i*18+16]);
    xa17=_mm512_set1_pd(a[i*18+17]);
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
    xc0=_mm512_mask3_fmadd_pd(xa11,xb11,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa12,xb12,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa13,xb13,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa14,xb14,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa15,xb15,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa16,xb16,xc0,255);
    xc0=_mm512_mask3_fmadd_pd(xa17,xb17,xc0,255);
    _MM512_MASK_STOREU_PD(&c[i*9+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],1);
 xb1 = _MM512_MASK_LOADU_PD(&b[9+8],1);
 xb2 = _MM512_MASK_LOADU_PD(&b[18+8],1);
 xb3 = _MM512_MASK_LOADU_PD(&b[27+8],1);
 xb4 = _MM512_MASK_LOADU_PD(&b[36+8],1);
 xb5 = _MM512_MASK_LOADU_PD(&b[45+8],1);
 xb6 = _MM512_MASK_LOADU_PD(&b[54+8],1);
 xb7 = _MM512_MASK_LOADU_PD(&b[63+8],1);
 xb8 = _MM512_MASK_LOADU_PD(&b[72+8],1);
 xb9 = _MM512_MASK_LOADU_PD(&b[81+8],1);
 xb10 = _MM512_MASK_LOADU_PD(&b[90+8],1);
 xb11 = _MM512_MASK_LOADU_PD(&b[99+8],1);
 xb12 = _MM512_MASK_LOADU_PD(&b[108+8],1);
 xb13 = _MM512_MASK_LOADU_PD(&b[117+8],1);
 xb14 = _MM512_MASK_LOADU_PD(&b[126+8],1);
 xb15 = _MM512_MASK_LOADU_PD(&b[135+8],1);
 xb16 = _MM512_MASK_LOADU_PD(&b[144+8],1);
 xb17 = _MM512_MASK_LOADU_PD(&b[153+8],1);
for(i=0;i<7;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*9+8],1);
    xa0=_mm512_set1_pd(a[i*18+0]);
    xa1=_mm512_set1_pd(a[i*18+1]);
    xa2=_mm512_set1_pd(a[i*18+2]);
    xa3=_mm512_set1_pd(a[i*18+3]);
    xa4=_mm512_set1_pd(a[i*18+4]);
    xa5=_mm512_set1_pd(a[i*18+5]);
    xa6=_mm512_set1_pd(a[i*18+6]);
    xa7=_mm512_set1_pd(a[i*18+7]);
    xa8=_mm512_set1_pd(a[i*18+8]);
    xa9=_mm512_set1_pd(a[i*18+9]);
    xa10=_mm512_set1_pd(a[i*18+10]);
    xa11=_mm512_set1_pd(a[i*18+11]);
    xa12=_mm512_set1_pd(a[i*18+12]);
    xa13=_mm512_set1_pd(a[i*18+13]);
    xa14=_mm512_set1_pd(a[i*18+14]);
    xa15=_mm512_set1_pd(a[i*18+15]);
    xa16=_mm512_set1_pd(a[i*18+16]);
    xa17=_mm512_set1_pd(a[i*18+17]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa2,xb2,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa3,xb3,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa4,xb4,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa5,xb5,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa6,xb6,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa7,xb7,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa8,xb8,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa9,xb9,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa10,xb10,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa11,xb11,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa12,xb12,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa13,xb13,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa14,xb14,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa15,xb15,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa16,xb16,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa17,xb17,xc0,1);
    _MM512_MASK_STOREU_PD(&c[i*9+8],xc0,1);
}
#else
printf("cppgemm_2_7_18_9\n");
for(int m=0;m<7;m++){
   for(int n=0;n<9;n++){
      for(int k=0;k<18;k++){
         c[m*9+n]+=a[m*18+k]*b[k*9+n];
      }
   }
}
#endif
}
 
