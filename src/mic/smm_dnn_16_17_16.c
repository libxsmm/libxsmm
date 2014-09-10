#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_16_17_16(const double* a, const double* b, double* c){
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
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[17+0],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[34+0],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[51+0],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[68+0],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[85+0],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[102+0],255);
 xb7 = _MM512_MASK_LOADU_PD(&b[119+0],255);
 xb8 = _MM512_MASK_LOADU_PD(&b[136+0],255);
 xb9 = _MM512_MASK_LOADU_PD(&b[153+0],255);
 xb10 = _MM512_MASK_LOADU_PD(&b[170+0],255);
 xb11 = _MM512_MASK_LOADU_PD(&b[187+0],255);
 xb12 = _MM512_MASK_LOADU_PD(&b[204+0],255);
 xb13 = _MM512_MASK_LOADU_PD(&b[221+0],255);
 xb14 = _MM512_MASK_LOADU_PD(&b[238+0],255);
 xb15 = _MM512_MASK_LOADU_PD(&b[255+0],255);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*17+0],255);
    xa0=_mm512_set1_pd(a[i*16+0]);
    xa1=_mm512_set1_pd(a[i*16+1]);
    xa2=_mm512_set1_pd(a[i*16+2]);
    xa3=_mm512_set1_pd(a[i*16+3]);
    xa4=_mm512_set1_pd(a[i*16+4]);
    xa5=_mm512_set1_pd(a[i*16+5]);
    xa6=_mm512_set1_pd(a[i*16+6]);
    xa7=_mm512_set1_pd(a[i*16+7]);
    xa8=_mm512_set1_pd(a[i*16+8]);
    xa9=_mm512_set1_pd(a[i*16+9]);
    xa10=_mm512_set1_pd(a[i*16+10]);
    xa11=_mm512_set1_pd(a[i*16+11]);
    xa12=_mm512_set1_pd(a[i*16+12]);
    xa13=_mm512_set1_pd(a[i*16+13]);
    xa14=_mm512_set1_pd(a[i*16+14]);
    xa15=_mm512_set1_pd(a[i*16+15]);
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
    _MM512_MASK_STOREU_PD(&c[i*17+0],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+8],255);
 xb1 = _MM512_MASK_LOADU_PD(&b[17+8],255);
 xb2 = _MM512_MASK_LOADU_PD(&b[34+8],255);
 xb3 = _MM512_MASK_LOADU_PD(&b[51+8],255);
 xb4 = _MM512_MASK_LOADU_PD(&b[68+8],255);
 xb5 = _MM512_MASK_LOADU_PD(&b[85+8],255);
 xb6 = _MM512_MASK_LOADU_PD(&b[102+8],255);
 xb7 = _MM512_MASK_LOADU_PD(&b[119+8],255);
 xb8 = _MM512_MASK_LOADU_PD(&b[136+8],255);
 xb9 = _MM512_MASK_LOADU_PD(&b[153+8],255);
 xb10 = _MM512_MASK_LOADU_PD(&b[170+8],255);
 xb11 = _MM512_MASK_LOADU_PD(&b[187+8],255);
 xb12 = _MM512_MASK_LOADU_PD(&b[204+8],255);
 xb13 = _MM512_MASK_LOADU_PD(&b[221+8],255);
 xb14 = _MM512_MASK_LOADU_PD(&b[238+8],255);
 xb15 = _MM512_MASK_LOADU_PD(&b[255+8],255);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*17+8],255);
    xa0=_mm512_set1_pd(a[i*16+0]);
    xa1=_mm512_set1_pd(a[i*16+1]);
    xa2=_mm512_set1_pd(a[i*16+2]);
    xa3=_mm512_set1_pd(a[i*16+3]);
    xa4=_mm512_set1_pd(a[i*16+4]);
    xa5=_mm512_set1_pd(a[i*16+5]);
    xa6=_mm512_set1_pd(a[i*16+6]);
    xa7=_mm512_set1_pd(a[i*16+7]);
    xa8=_mm512_set1_pd(a[i*16+8]);
    xa9=_mm512_set1_pd(a[i*16+9]);
    xa10=_mm512_set1_pd(a[i*16+10]);
    xa11=_mm512_set1_pd(a[i*16+11]);
    xa12=_mm512_set1_pd(a[i*16+12]);
    xa13=_mm512_set1_pd(a[i*16+13]);
    xa14=_mm512_set1_pd(a[i*16+14]);
    xa15=_mm512_set1_pd(a[i*16+15]);
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
    _MM512_MASK_STOREU_PD(&c[i*17+8],xc0,255);
}
 xb0 = _MM512_MASK_LOADU_PD(&b[0+16],1);
 xb1 = _MM512_MASK_LOADU_PD(&b[17+16],1);
 xb2 = _MM512_MASK_LOADU_PD(&b[34+16],1);
 xb3 = _MM512_MASK_LOADU_PD(&b[51+16],1);
 xb4 = _MM512_MASK_LOADU_PD(&b[68+16],1);
 xb5 = _MM512_MASK_LOADU_PD(&b[85+16],1);
 xb6 = _MM512_MASK_LOADU_PD(&b[102+16],1);
 xb7 = _MM512_MASK_LOADU_PD(&b[119+16],1);
 xb8 = _MM512_MASK_LOADU_PD(&b[136+16],1);
 xb9 = _MM512_MASK_LOADU_PD(&b[153+16],1);
 xb10 = _MM512_MASK_LOADU_PD(&b[170+16],1);
 xb11 = _MM512_MASK_LOADU_PD(&b[187+16],1);
 xb12 = _MM512_MASK_LOADU_PD(&b[204+16],1);
 xb13 = _MM512_MASK_LOADU_PD(&b[221+16],1);
 xb14 = _MM512_MASK_LOADU_PD(&b[238+16],1);
 xb15 = _MM512_MASK_LOADU_PD(&b[255+16],1);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*17+16],1);
    xa0=_mm512_set1_pd(a[i*16+0]);
    xa1=_mm512_set1_pd(a[i*16+1]);
    xa2=_mm512_set1_pd(a[i*16+2]);
    xa3=_mm512_set1_pd(a[i*16+3]);
    xa4=_mm512_set1_pd(a[i*16+4]);
    xa5=_mm512_set1_pd(a[i*16+5]);
    xa6=_mm512_set1_pd(a[i*16+6]);
    xa7=_mm512_set1_pd(a[i*16+7]);
    xa8=_mm512_set1_pd(a[i*16+8]);
    xa9=_mm512_set1_pd(a[i*16+9]);
    xa10=_mm512_set1_pd(a[i*16+10]);
    xa11=_mm512_set1_pd(a[i*16+11]);
    xa12=_mm512_set1_pd(a[i*16+12]);
    xa13=_mm512_set1_pd(a[i*16+13]);
    xa14=_mm512_set1_pd(a[i*16+14]);
    xa15=_mm512_set1_pd(a[i*16+15]);
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
    _MM512_MASK_STOREU_PD(&c[i*17+16],xc0,1);
}
#else
printf("cppgemm_2_16_16_17\n");
for(int m=0;m<16;m++){
   for(int n=0;n<17;n++){
      for(int k=0;k<16;k++){
         c[m*17+n]+=a[m*16+k]*b[k*17+n];
      }
   }
}
#endif
}
 
