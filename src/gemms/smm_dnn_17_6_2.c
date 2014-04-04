#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec( target (mic))
void smm_dnn_17_6_2(double* a,double* b,double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xa1;
__m512d xb0;
__m512d xb1;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],63);
 xb1 = _MM512_MASK_LOADU_PD(&b[6+0],63);
for(i=0;i<17;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*6+0],63);
    xa0=_mm512_set1_pd(a[i*2+0]);
    xa1=_mm512_set1_pd(a[i*2+1]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,63);
    xc0=_mm512_mask3_fmadd_pd(xa1,xb1,xc0,63);
    _MM512_MASK_STOREU_PD(&c[i*6+0],xc0,63);
}
#else
printf("cppgemm_2_17_2_6\n");
for(int m=0;m<17;m++){
   for(int n=0;n<6;n++){
      for(int k=0;k<2;k++){
         c[m*6+n]+=a[m*2+k]*b[k*6+n];
      }
   }
}
#endif
}
 
