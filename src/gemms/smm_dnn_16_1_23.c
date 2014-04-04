#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec( target (mic))
void smm_dnn_16_1_23(double* a,double* b,double* c){
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
__m512d xa18;
__m512d xa19;
__m512d xa20;
__m512d xa21;
__m512d xa22;
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
__m512d xb18;
__m512d xb19;
__m512d xb20;
__m512d xb21;
__m512d xb22;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],1);
 xb1 = _MM512_MASK_LOADU_PD(&b[1+0],1);
 xb2 = _MM512_MASK_LOADU_PD(&b[2+0],1);
 xb3 = _MM512_MASK_LOADU_PD(&b[3+0],1);
 xb4 = _MM512_MASK_LOADU_PD(&b[4+0],1);
 xb5 = _MM512_MASK_LOADU_PD(&b[5+0],1);
 xb6 = _MM512_MASK_LOADU_PD(&b[6+0],1);
 xb7 = _MM512_MASK_LOADU_PD(&b[7+0],1);
 xb8 = _MM512_MASK_LOADU_PD(&b[8+0],1);
 xb9 = _MM512_MASK_LOADU_PD(&b[9+0],1);
 xb10 = _MM512_MASK_LOADU_PD(&b[10+0],1);
 xb11 = _MM512_MASK_LOADU_PD(&b[11+0],1);
 xb12 = _MM512_MASK_LOADU_PD(&b[12+0],1);
 xb13 = _MM512_MASK_LOADU_PD(&b[13+0],1);
 xb14 = _MM512_MASK_LOADU_PD(&b[14+0],1);
 xb15 = _MM512_MASK_LOADU_PD(&b[15+0],1);
 xb16 = _MM512_MASK_LOADU_PD(&b[16+0],1);
 xb17 = _MM512_MASK_LOADU_PD(&b[17+0],1);
 xb18 = _MM512_MASK_LOADU_PD(&b[18+0],1);
 xb19 = _MM512_MASK_LOADU_PD(&b[19+0],1);
 xb20 = _MM512_MASK_LOADU_PD(&b[20+0],1);
 xb21 = _MM512_MASK_LOADU_PD(&b[21+0],1);
 xb22 = _MM512_MASK_LOADU_PD(&b[22+0],1);
for(i=0;i<16;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*1+0],1);
    xa0=_mm512_set1_pd(a[i*23+0]);
    xa1=_mm512_set1_pd(a[i*23+1]);
    xa2=_mm512_set1_pd(a[i*23+2]);
    xa3=_mm512_set1_pd(a[i*23+3]);
    xa4=_mm512_set1_pd(a[i*23+4]);
    xa5=_mm512_set1_pd(a[i*23+5]);
    xa6=_mm512_set1_pd(a[i*23+6]);
    xa7=_mm512_set1_pd(a[i*23+7]);
    xa8=_mm512_set1_pd(a[i*23+8]);
    xa9=_mm512_set1_pd(a[i*23+9]);
    xa10=_mm512_set1_pd(a[i*23+10]);
    xa11=_mm512_set1_pd(a[i*23+11]);
    xa12=_mm512_set1_pd(a[i*23+12]);
    xa13=_mm512_set1_pd(a[i*23+13]);
    xa14=_mm512_set1_pd(a[i*23+14]);
    xa15=_mm512_set1_pd(a[i*23+15]);
    xa16=_mm512_set1_pd(a[i*23+16]);
    xa17=_mm512_set1_pd(a[i*23+17]);
    xa18=_mm512_set1_pd(a[i*23+18]);
    xa19=_mm512_set1_pd(a[i*23+19]);
    xa20=_mm512_set1_pd(a[i*23+20]);
    xa21=_mm512_set1_pd(a[i*23+21]);
    xa22=_mm512_set1_pd(a[i*23+22]);
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
    xc0=_mm512_mask3_fmadd_pd(xa18,xb18,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa19,xb19,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa20,xb20,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa21,xb21,xc0,1);
    xc0=_mm512_mask3_fmadd_pd(xa22,xb22,xc0,1);
    _MM512_MASK_STOREU_PD(&c[i*1+0],xc0,1);
}
#else
printf("cppgemm_2_16_23_1\n");
for(int m=0;m<16;m++){
   for(int n=0;n<1;n++){
      for(int k=0;k<23;k++){
         c[m*1+n]+=a[m*23+k]*b[k*1+n];
      }
   }
}
#endif
}
 
