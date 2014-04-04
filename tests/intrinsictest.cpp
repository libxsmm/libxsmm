#include <immintrin.h>
#include <micsmmmisc.h>

int main(void){
#pragma offload target(mic:0)
  {
#ifdef __MIC__
    int size=100;
    double* a = new double[size];
    double* b = new double[size];
    double* c = new double[size];
    __m512d xa;
    __m512d xb;
    __m512d xc;
    for (int i=0;i<size;i++){
      a[i]=i+1;
      b[i]=i+1+8;
      c[i]=i+1+16;
    }
    xa= _MM512_MASK_LOADU_PD(&a[0],0);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],1);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],3);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],7);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],15);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],31);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],63);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],127);
    print512d(xa);
    xa= _MM512_MASK_LOADU_PD(&a[0],255);
    print512d(xa);
    printf("\n");
    xa= _MM512_MASK_LOADU_PD(&a[0],255);
    xb= _MM512_MASK_LOADU_PD(&b[0],255);
    xb=_mm512_mask3_fmadd_pd(xb,xa,xa,85);
    print512d(xb);




#else
    printf("not offloading \n");
#endif 
  }
  return 0;
}
