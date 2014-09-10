#include <immintrin.h>
#include <micsmmmisc.h>
#include <mkl.h>
__declspec(target(mic))
void smm_dnn_12_3_1(const double* a, const double* b, double* c){
#ifdef __MIC__
int i;
__m512d xa0;
__m512d xb0;
__m512d xc0;
 xb0 = _MM512_MASK_LOADU_PD(&b[0+0],7);
for(i=0;i<12;i+=1){
    xc0 = _MM512_MASK_LOADU_PD(&c[i*3+0],7);
    xa0=_mm512_set1_pd(a[i*1+0]);
    xc0=_mm512_mask3_fmadd_pd(xa0,xb0,xc0,7);
    _MM512_MASK_STOREU_PD(&c[i*3+0],xc0,7);
}
#else
printf("cppgemm_2_12_1_3\n");
for(int m=0;m<12;m++){
   for(int n=0;n<3;n++){
      for(int k=0;k<1;k++){
         c[m*3+n]+=a[m*1+k]*b[k*3+n];
      }
   }
}
#endif
}
 
