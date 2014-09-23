#include <stdio.h>
#include <immintrin.h>
#include <math.h>


#ifdef __MIC__
__declspec(target(mic))
inline __m512d   _MM512_LOADU_PD(const double* a) {
    __m512d va= _mm512_setzero_pd();
    /* va=_mm512_extloadunpacklo_pd(va, &a[0],_MM_UPCONV_PD_NONE, _MM_HINT_NONE); */
    /* va=_mm512_extloadunpackhi_pd(va, &a[8],_MM_UPCONV_PD_NONE, _MM_HINT_NONE); */
    va=_mm512_loadunpacklo_pd(va, &a[0]);
    va=_mm512_loadunpackhi_pd(va, &a[8]);
    return va;
}


__declspec(target(mic))
inline void _MM512_STOREU_PD(const double* a,__m512d v) {
//  __m512i vindex= _mm512_set_epi32(0,0,0,0,0,0,0,0,7,6,5,4,3,2,1,0);
//  _mm512_i32loscatter_pd(a,vindex, v, 8);
    _mm512_packstorelo_pd(&a[0], v);
    _mm512_packstorehi_pd(&a[8], v);
}



__declspec(target(mic))
inline __m512d _MM512_MASK_LOADU_PD(const double* a, char mask) {
    __m512d va= _mm512_setzero_pd();
    /* va=_mm512_mask_extloadunpacklo_pd(va, mask,&a[0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE); */
    /* va=_mm512_mask_extloadunpackhi_pd(va, mask,&a[8], _MM_UPCONV_PD_NONE, _MM_HINT_NONE); */
    va=_mm512_mask_loadunpacklo_pd(va, mask,&a[0]);
    va=_mm512_mask_loadunpackhi_pd(va, mask,&a[8]);
    return va;
}

__declspec(target(mic))
inline void _MM512_MASK_STOREU_PD(const double* a,__m512d v, char mask) {
//  __m512i vindex= _mm512_set_epi32(0,0,0,0,0,0,0,0,7,6,5,4,3,2,1,0);
//  _mm512_mask_i32loscatter_pd(a,mask,vindex, v, 8);
    _mm512_mask_packstorelo_pd(&a[0], mask,v);
    _mm512_mask_packstorehi_pd(&a[8],mask, v);
}


__declspec(target(mic))
void print512d(__m512d a){
    double* f = (double *) _mm_malloc(sizeof(double)*8,64);
    _mm512_store_pd(f,a);
    printf("%f %f %f %f %f %f %f %f\n",f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7]);
    _mm_free(f);
}

#endif

__declspec(target(mic))
void xsmm_knc_c_23_(double* c, const double* a, const double* b){
#ifdef __MIC__
    int i;

    /* printf("%ld %ld %ld\n",c,a,b); */
    /* return ; */

    __m512d xc0;

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

    //  __m512d* xa = new __m512d[23];
    //  __m512d* xb = new __m512d[23];
    __m512d xa0;
    __m512d xa1;
    __m512d xa2;
    __m512d xa3;
    __m512d xa4;
    __m512d xa5;
    __m512d xa6;
    __m512d xa7;

    xb0 = _MM512_LOADU_PD(&b[23*0]);
    xb1 = _MM512_LOADU_PD(&b[23*1]);
    xb2 = _MM512_LOADU_PD(&b[23*2]);
    xb3 = _MM512_LOADU_PD(&b[23*3]);
    xb4 = _MM512_LOADU_PD(&b[23*4]);
    xb5 = _MM512_LOADU_PD(&b[23*5]);
    xb6 = _MM512_LOADU_PD(&b[23*6]);
    xb7 = _MM512_LOADU_PD(&b[23*7]);
    xb8 = _MM512_LOADU_PD(&b[23*8]);
    xb9 = _MM512_LOADU_PD(&b[23*9]);
    xb10 = _MM512_LOADU_PD(&b[23*10]);
    xb11 = _MM512_LOADU_PD(&b[23*11]);
    xb12 = _MM512_LOADU_PD(&b[23*12]);
    xb13 = _MM512_LOADU_PD(&b[23*13]);
    xb14 = _MM512_LOADU_PD(&b[23*14]);
    xb15 = _MM512_LOADU_PD(&b[23*15]);
    xb16 = _MM512_LOADU_PD(&b[23*16]);
    xb17 = _MM512_LOADU_PD(&b[23*17]);
    xb18 = _MM512_LOADU_PD(&b[23*18]);
    xb19 = _MM512_LOADU_PD(&b[23*19]);
    xb20 = _MM512_LOADU_PD(&b[23*20]);
    xb21 = _MM512_LOADU_PD(&b[23*21]);
    xb22 = _MM512_LOADU_PD(&b[23*22]);


    for(i=0;i<23*23;i+=23)
    {

	xc0 = _MM512_LOADU_PD(&c[i]);

	xa0=_mm512_set1_pd(a[i]);
	xa1=_mm512_set1_pd(a[1+i]);
	xa2=_mm512_set1_pd(a[2+i]);
	xa3=_mm512_set1_pd(a[3+i]);
	xa4=_mm512_set1_pd(a[4+i]);
	xa5=_mm512_set1_pd(a[5+i]);
	xa6=_mm512_set1_pd(a[6+i]);
	xa7=_mm512_set1_pd(a[7+i]);

	xc0=_mm512_fmadd_pd(xa0,xb0,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb1,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb2,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb3,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb4,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb5,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb6,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb7,xc0);

	xa0=_mm512_set1_pd(a[8+i]);
	xa1=_mm512_set1_pd(a[9+i]);
	xa2=_mm512_set1_pd(a[10+i]);
	xa3=_mm512_set1_pd(a[11+i]);
	xa4=_mm512_set1_pd(a[12+i]);
	xa5=_mm512_set1_pd(a[13+i]);
	xa6=_mm512_set1_pd(a[14+i]);
	xa7=_mm512_set1_pd(a[15+i]);

	xc0=_mm512_fmadd_pd(xa0,xb8,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb9,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb10,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb11,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb12,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb13,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb14,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb15,xc0);

	xa0=_mm512_set1_pd(a[16+i]);
	xa1=_mm512_set1_pd(a[17+i]);
	xa2=_mm512_set1_pd(a[18+i]);
	xa3=_mm512_set1_pd(a[19+i]);
	xa4=_mm512_set1_pd(a[20+i]);
	xa5=_mm512_set1_pd(a[21+i]);
	xa6=_mm512_set1_pd(a[22+i]);


	xc0=_mm512_fmadd_pd(xa0,xb16,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb17,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb18,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb19,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb20,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb21,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb22,xc0);

	_MM512_STOREU_PD(&c[i], xc0);

    }

    xb0 = _MM512_LOADU_PD(&b[23*0+8]);
    xb1 = _MM512_LOADU_PD(&b[23*1+8]);
    xb2 = _MM512_LOADU_PD(&b[23*2+8]);
    xb3 = _MM512_LOADU_PD(&b[23*3+8]);
    xb4 = _MM512_LOADU_PD(&b[23*4+8]);
    xb5 = _MM512_LOADU_PD(&b[23*5+8]);
    xb6 = _MM512_LOADU_PD(&b[23*6+8]);
    xb7 = _MM512_LOADU_PD(&b[23*7+8]);
    xb8 = _MM512_LOADU_PD(&b[23*8+8]);
    xb9 = _MM512_LOADU_PD(&b[23*9+8]);
    xb10 = _MM512_LOADU_PD(&b[23*10+8]);
    xb11 = _MM512_LOADU_PD(&b[23*11+8]);
    xb12 = _MM512_LOADU_PD(&b[23*12+8]);
    xb13 = _MM512_LOADU_PD(&b[23*13+8]);
    xb14 = _MM512_LOADU_PD(&b[23*14+8]);
    xb15 = _MM512_LOADU_PD(&b[23*15+8]);
    xb16 = _MM512_LOADU_PD(&b[23*16+8]);
    xb17 = _MM512_LOADU_PD(&b[23*17+8]);
    xb18 = _MM512_LOADU_PD(&b[23*18+8]);
    xb19 = _MM512_LOADU_PD(&b[23*19+8]);
    xb20 = _MM512_LOADU_PD(&b[23*20+8]);
    xb21 = _MM512_LOADU_PD(&b[23*21+8]);
    xb22 = _MM512_LOADU_PD(&b[23*22+8]);


    for(i=0;i<23*23;i+=23)
    {

	xc0 = _MM512_LOADU_PD(&c[i+8]);

	xa0=_mm512_set1_pd(a[0+i]);
	xa1=_mm512_set1_pd(a[1+i]);
	xa2=_mm512_set1_pd(a[2+i]);
	xa3=_mm512_set1_pd(a[3+i]);
	xa4=_mm512_set1_pd(a[4+i]);
	xa5=_mm512_set1_pd(a[5+i]);
	xa6=_mm512_set1_pd(a[6+i]);
	xa7=_mm512_set1_pd(a[7+i]);

	xc0=_mm512_fmadd_pd(xa0,xb0,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb1,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb2,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb3,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb4,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb5,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb6,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb7,xc0);

	xa0=_mm512_set1_pd(a[8+i]);
	xa1=_mm512_set1_pd(a[9+i]);
	xa2=_mm512_set1_pd(a[10+i]);
	xa3=_mm512_set1_pd(a[11+i]);
	xa4=_mm512_set1_pd(a[12+i]);
	xa5=_mm512_set1_pd(a[13+i]);
	xa6=_mm512_set1_pd(a[14+i]);
	xa7=_mm512_set1_pd(a[15+i]);

	xc0=_mm512_fmadd_pd(xa0,xb8,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb9,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb10,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb11,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb12,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb13,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb14,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb15,xc0);

	xa0=_mm512_set1_pd(a[16+i]);
	xa1=_mm512_set1_pd(a[17+i]);
	xa2=_mm512_set1_pd(a[18+i]);
	xa3=_mm512_set1_pd(a[19+i]);
	xa4=_mm512_set1_pd(a[20+i]);
	xa5=_mm512_set1_pd(a[21+i]);
	xa6=_mm512_set1_pd(a[22+i]);


	xc0=_mm512_fmadd_pd(xa0,xb16,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb17,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb18,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb19,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb20,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb21,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb22,xc0);

	_MM512_STOREU_PD(&c[i+8], xc0);
    }

    xb0 = _MM512_MASK_LOADU_PD(&b[23*0+16],127);
    xb1 = _MM512_MASK_LOADU_PD(&b[23*1+16],127);
    xb2 = _MM512_MASK_LOADU_PD(&b[23*2+16],127);
    xb3 = _MM512_MASK_LOADU_PD(&b[23*3+16],127);
    xb4 = _MM512_MASK_LOADU_PD(&b[23*4+16],127);
    xb5 = _MM512_MASK_LOADU_PD(&b[23*5+16],127);
    xb6 = _MM512_MASK_LOADU_PD(&b[23*6+16],127);
    xb7 = _MM512_MASK_LOADU_PD(&b[23*7+16],127);
    xb8 = _MM512_MASK_LOADU_PD(&b[23*8+16],127);
    xb9 = _MM512_MASK_LOADU_PD(&b[23*9+16],127);
    xb10 = _MM512_MASK_LOADU_PD(&b[23*10+16],127);
    xb11 = _MM512_MASK_LOADU_PD(&b[23*11+16],127);
    xb12 = _MM512_MASK_LOADU_PD(&b[23*12+16],127);
    xb13 = _MM512_MASK_LOADU_PD(&b[23*13+16],127);
    xb14 = _MM512_MASK_LOADU_PD(&b[23*14+16],127);
    xb15 = _MM512_MASK_LOADU_PD(&b[23*15+16],127);
    xb16 = _MM512_MASK_LOADU_PD(&b[23*16+16],127);
    xb17 = _MM512_MASK_LOADU_PD(&b[23*17+16],127);
    xb18 = _MM512_MASK_LOADU_PD(&b[23*18+16],127);
    xb19 = _MM512_MASK_LOADU_PD(&b[23*19+16],127);
    xb20 = _MM512_MASK_LOADU_PD(&b[23*20+16],127);
    xb21 = _MM512_MASK_LOADU_PD(&b[23*21+16],127);
    xb22 = _MM512_MASK_LOADU_PD(&b[23*22+16],127);

    for(i=0;i<23*23;i+=23)
    {

	xc0 = _MM512_MASK_LOADU_PD(&c[i+16],127);

	xa0=_mm512_set1_pd(a[0+i]);
	xa1=_mm512_set1_pd(a[1+i]);
	xa2=_mm512_set1_pd(a[2+i]);
	xa3=_mm512_set1_pd(a[3+i]);
	xa4=_mm512_set1_pd(a[4+i]);
	xa5=_mm512_set1_pd(a[5+i]);
	xa6=_mm512_set1_pd(a[6+i]);
	xa7=_mm512_set1_pd(a[7+i]);

	xc0=_mm512_fmadd_pd(xa0,xb0,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb1,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb2,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb3,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb4,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb5,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb6,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb7,xc0);

	xa0=_mm512_set1_pd(a[8+i]);
	xa1=_mm512_set1_pd(a[9+i]);
	xa2=_mm512_set1_pd(a[10+i]);
	xa3=_mm512_set1_pd(a[11+i]);
	xa4=_mm512_set1_pd(a[12+i]);
	xa5=_mm512_set1_pd(a[13+i]);
	xa6=_mm512_set1_pd(a[14+i]);
	xa7=_mm512_set1_pd(a[15+i]);

	xc0=_mm512_fmadd_pd(xa0,xb8,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb9,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb10,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb11,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb12,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb13,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb14,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb15,xc0);

	xa0=_mm512_set1_pd(a[16+i]);
	xa1=_mm512_set1_pd(a[17+i]);
	xa2=_mm512_set1_pd(a[18+i]);
	xa3=_mm512_set1_pd(a[19+i]);
	xa4=_mm512_set1_pd(a[20+i]);
	xa5=_mm512_set1_pd(a[21+i]);
	xa6=_mm512_set1_pd(a[22+i]);


	xc0=_mm512_fmadd_pd(xa0,xb16,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb17,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb18,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb19,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb20,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb21,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb22,xc0);

	_MM512_MASK_STOREU_PD(&c[i+16], xc0,127);

    }
//    print512d(xc0);

#else
    int i,j,n;

    /* printf("%ld %ld %ld\n",c,a,b); */
    /* return ; */

#pragma simd
    for (i=0;i<23;i++)
    {
	for (j=0;j<23;j++)
	{
	    int n=i*23;
	    c[n+j]=c[n+j]+a[n]*b[j]
		+a[n+1]*b[23+j]
		+a[n+2]*b[2*23+j]
		+a[n+3]*b[3*23+j]
		+a[n+4]*b[4*23+j]
		+a[n+5]*b[5*23+j]
		+a[n+6]*b[6*23+j]
		+a[n+7]*b[7*23+j]
		+a[n+8]*b[8*23+j]
		+a[n+9]*b[9*23+j]
		+a[n+10]*b[10*23+j]
		+a[n+11]*b[11*23+j]
		+a[n+12]*b[12*23+j]
		+a[n+13]*b[13*23+j]
		+a[n+14]*b[14*23+j]
		+a[n+15]*b[15*23+j]
		+a[n+16]*b[16*23+j]
		+a[n+17]*b[17*23+j]
		+a[n+18]*b[18*23+j]
		+a[n+19]*b[19*23+j]
		+a[n+20]*b[20*23+j]
		+a[n+21]*b[21*23+j]
		+a[n+22]*b[22*23+j];
	}
    }

#endif
}

__declspec(target(mic))
void xsmm_knc_c_23_(double* c, const double* a, const double* b){
#ifdef __MIC__
    int i;

    /* printf("%ld %ld %ld\n",c,a,b); */
    /* return ; */

    __m512d xc0;
    __m512d xc8;
    __m512d xc16;


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

    //  __m512d* xa = new __m512d[23];
    //  __m512d* xb = new __m512d[23];
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

#if 1
    xb0 = _MM512_LOADU_PD(&b[23*0]);
    xb1 = _MM512_LOADU_PD(&b[23*1]);
    xb2 = _MM512_LOADU_PD(&b[23*2]);
    xb3 = _MM512_LOADU_PD(&b[23*3]);
    xb4 = _MM512_LOADU_PD(&b[23*4]);
    xb5 = _MM512_LOADU_PD(&b[23*5]);
    xb6 = _MM512_LOADU_PD(&b[23*6]);
    xb7 = _MM512_LOADU_PD(&b[23*7]);
    xb8 = _MM512_LOADU_PD(&b[23*8]);
    xb9 = _MM512_LOADU_PD(&b[23*9]);
    xb10 = _MM512_LOADU_PD(&b[23*10]);
    xb11 = _MM512_LOADU_PD(&b[23*11]);
    xb12 = _MM512_LOADU_PD(&b[23*12]);
    xb13 = _MM512_LOADU_PD(&b[23*13]);
    xb14 = _MM512_LOADU_PD(&b[23*14]);
    xb15 = _MM512_LOADU_PD(&b[23*15]);
    xb16 = _MM512_LOADU_PD(&b[23*16]);
    xb17 = _MM512_LOADU_PD(&b[23*17]);
    xb18 = _MM512_LOADU_PD(&b[23*18]);
    xb19 = _MM512_LOADU_PD(&b[23*19]);
    xb20 = _MM512_LOADU_PD(&b[23*20]);
    xb21 = _MM512_LOADU_PD(&b[23*21]);
    xb22 = _MM512_LOADU_PD(&b[23*22]);


    for(i=0;i<23*23;i+=23)
    {

	xc0 = _MM512_LOADU_PD(&c[i]);

	xa0=_mm512_set1_pd(a[i]);
	xa1=_mm512_set1_pd(a[1+i]);
	xa2=_mm512_set1_pd(a[2+i]);
	xa3=_mm512_set1_pd(a[3+i]);
	xa4=_mm512_set1_pd(a[4+i]);
	xa5=_mm512_set1_pd(a[5+i]);
	xa6=_mm512_set1_pd(a[6+i]);
	xa7=_mm512_set1_pd(a[7+i]);

	xc0=_mm512_fmadd_pd(xa0,xb0,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb1,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb2,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb3,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb4,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb5,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb6,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb7,xc0);

	xa8=_mm512_set1_pd(a[8+i]);
	xa9=_mm512_set1_pd(a[9+i]);
	xa10=_mm512_set1_pd(a[10+i]);
	xa11=_mm512_set1_pd(a[11+i]);
	xa12=_mm512_set1_pd(a[12+i]);
	xa13=_mm512_set1_pd(a[13+i]);
	xa14=_mm512_set1_pd(a[14+i]);
	xa15=_mm512_set1_pd(a[15+i]);

	xc0=_mm512_fmadd_pd(xa8,xb8,xc0);
	xc0=_mm512_fmadd_pd(xa9,xb9,xc0);
	xc0=_mm512_fmadd_pd(xa10,xb10,xc0);
	xc0=_mm512_fmadd_pd(xa11,xb11,xc0);
	xc0=_mm512_fmadd_pd(xa12,xb12,xc0);
	xc0=_mm512_fmadd_pd(xa13,xb13,xc0);
	xc0=_mm512_fmadd_pd(xa14,xb14,xc0);
	xc0=_mm512_fmadd_pd(xa15,xb15,xc0);
	xc0=_mm512_fmadd_pd(xa16,xb16,xc0);

	xa16=_mm512_set1_pd(a[16+i]);
	xa17=_mm512_set1_pd(a[17+i]);
	xa18=_mm512_set1_pd(a[18+i]);
	xa19=_mm512_set1_pd(a[19+i]);
	xa20=_mm512_set1_pd(a[20+i]);
	xa21=_mm512_set1_pd(a[21+i]);
	xa22=_mm512_set1_pd(a[22+i]);


	xc0=_mm512_fmadd_pd(xa17,xb17,xc0);
	xc0=_mm512_fmadd_pd(xa18,xb18,xc0);
	xc0=_mm512_fmadd_pd(xa19,xb19,xc0);
	xc0=_mm512_fmadd_pd(xa20,xb20,xc0);
	xc0=_mm512_fmadd_pd(xa21,xb21,xc0);
	xc0=_mm512_fmadd_pd(xa22,xb22,xc0);

	_MM512_STOREU_PD(&c[i], xc0);

    }
//  return;
    xb0 = _MM512_LOADU_PD(&b[23*0+8]);
    xb1 = _MM512_LOADU_PD(&b[23*1+8]);
    xb2 = _MM512_LOADU_PD(&b[23*2+8]);
    xb3 = _MM512_LOADU_PD(&b[23*3+8]);
    xb4 = _MM512_LOADU_PD(&b[23*4+8]);
    xb5 = _MM512_LOADU_PD(&b[23*5+8]);
    xb6 = _MM512_LOADU_PD(&b[23*6+8]);
    xb7 = _MM512_LOADU_PD(&b[23*7+8]);
    xb8 = _MM512_LOADU_PD(&b[23*8+8]);
    xb9 = _MM512_LOADU_PD(&b[23*9+8]);
    xb10 = _MM512_LOADU_PD(&b[23*10+8]);
    xb11 = _MM512_LOADU_PD(&b[23*11+8]);
    xb12 = _MM512_LOADU_PD(&b[23*12+8]);
    xb13 = _MM512_LOADU_PD(&b[23*13+8]);
    xb14 = _MM512_LOADU_PD(&b[23*14+8]);
    xb15 = _MM512_LOADU_PD(&b[23*15+8]);
    xb16 = _MM512_LOADU_PD(&b[23*16+8]);
    xb17 = _MM512_LOADU_PD(&b[23*17+8]);
    xb18 = _MM512_LOADU_PD(&b[23*18+8]);
    xb19 = _MM512_LOADU_PD(&b[23*19+8]);
    xb20 = _MM512_LOADU_PD(&b[23*20+8]);
    xb21 = _MM512_LOADU_PD(&b[23*21+8]);
    xb22 = _MM512_LOADU_PD(&b[23*22+8]);


    for(i=0;i<23*23;i+=23)
    {

	xc0 = _MM512_LOADU_PD(&c[i+8]);

	xa0=_mm512_set1_pd(a[0+i]);
	xa1=_mm512_set1_pd(a[1+i]);
	xa2=_mm512_set1_pd(a[2+i]);
	xa3=_mm512_set1_pd(a[3+i]);
	xa4=_mm512_set1_pd(a[4+i]);
	xa5=_mm512_set1_pd(a[5+i]);
	xa6=_mm512_set1_pd(a[6+i]);
	xa7=_mm512_set1_pd(a[7+i]);

	xc0=_mm512_fmadd_pd(xa0,xb0,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb1,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb2,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb3,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb4,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb5,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb6,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb7,xc0);

	xa8=_mm512_set1_pd(a[8+i]);
	xa9=_mm512_set1_pd(a[9+i]);
	xa10=_mm512_set1_pd(a[10+i]);
	xa11=_mm512_set1_pd(a[11+i]);
	xa12=_mm512_set1_pd(a[12+i]);
	xa13=_mm512_set1_pd(a[13+i]);
	xa14=_mm512_set1_pd(a[14+i]);
	xa15=_mm512_set1_pd(a[15+i]);

	xc0=_mm512_fmadd_pd(xa8,xb8,xc0);
	xc0=_mm512_fmadd_pd(xa9,xb9,xc0);
	xc0=_mm512_fmadd_pd(xa10,xb10,xc0);
	xc0=_mm512_fmadd_pd(xa11,xb11,xc0);
	xc0=_mm512_fmadd_pd(xa12,xb12,xc0);
	xc0=_mm512_fmadd_pd(xa13,xb13,xc0);
	xc0=_mm512_fmadd_pd(xa14,xb14,xc0);
	xc0=_mm512_fmadd_pd(xa15,xb15,xc0);
	xc0=_mm512_fmadd_pd(xa16,xb16,xc0);

	xa16=_mm512_set1_pd(a[16+i]);
	xa17=_mm512_set1_pd(a[17+i]);
	xa18=_mm512_set1_pd(a[18+i]);
	xa19=_mm512_set1_pd(a[19+i]);
	xa20=_mm512_set1_pd(a[20+i]);
	xa21=_mm512_set1_pd(a[21+i]);
	xa22=_mm512_set1_pd(a[22+i]);


	xc0=_mm512_fmadd_pd(xa17,xb17,xc0);
	xc0=_mm512_fmadd_pd(xa18,xb18,xc0);
	xc0=_mm512_fmadd_pd(xa19,xb19,xc0);
	xc0=_mm512_fmadd_pd(xa20,xb20,xc0);
	xc0=_mm512_fmadd_pd(xa21,xb21,xc0);
	xc0=_mm512_fmadd_pd(xa22,xb22,xc0);

	_MM512_STOREU_PD(&c[i+8], xc0);
    }

    xb0 = _MM512_MASK_LOADU_PD(&b[23*0+16],127);
    xb1 = _MM512_MASK_LOADU_PD(&b[23*1+16],127);
    xb2 = _MM512_MASK_LOADU_PD(&b[23*2+16],127);
    xb3 = _MM512_MASK_LOADU_PD(&b[23*3+16],127);
    xb4 = _MM512_MASK_LOADU_PD(&b[23*4+16],127);
    xb5 = _MM512_MASK_LOADU_PD(&b[23*5+16],127);
    xb6 = _MM512_MASK_LOADU_PD(&b[23*6+16],127);
    xb7 = _MM512_MASK_LOADU_PD(&b[23*7+16],127);
    xb8 = _MM512_MASK_LOADU_PD(&b[23*8+16],127);
    xb9 = _MM512_MASK_LOADU_PD(&b[23*9+16],127);
    xb10 = _MM512_MASK_LOADU_PD(&b[23*10+16],127);
    xb11 = _MM512_MASK_LOADU_PD(&b[23*11+16],127);
    xb12 = _MM512_MASK_LOADU_PD(&b[23*12+16],127);
    xb13 = _MM512_MASK_LOADU_PD(&b[23*13+16],127);
    xb14 = _MM512_MASK_LOADU_PD(&b[23*14+16],127);
    xb15 = _MM512_MASK_LOADU_PD(&b[23*15+16],127);
    xb16 = _MM512_MASK_LOADU_PD(&b[23*16+16],127);
    xb17 = _MM512_MASK_LOADU_PD(&b[23*17+16],127);
    xb18 = _MM512_MASK_LOADU_PD(&b[23*18+16],127);
    xb19 = _MM512_MASK_LOADU_PD(&b[23*19+16],127);
    xb20 = _MM512_MASK_LOADU_PD(&b[23*20+16],127);
    xb21 = _MM512_MASK_LOADU_PD(&b[23*21+16],127);
    xb22 = _MM512_MASK_LOADU_PD(&b[23*22+16],127);

    for(i=0;i<23*23;i+=23)
    {

	xc0 = _MM512_MASK_LOADU_PD(&c[i+16],127);

	xa0=_mm512_set1_pd(a[0+i]);
	xa1=_mm512_set1_pd(a[1+i]);
	xa2=_mm512_set1_pd(a[2+i]);
	xa3=_mm512_set1_pd(a[3+i]);
	xa4=_mm512_set1_pd(a[4+i]);
	xa5=_mm512_set1_pd(a[5+i]);
	xa6=_mm512_set1_pd(a[6+i]);
	xa7=_mm512_set1_pd(a[7+i]);

	xc0=_mm512_fmadd_pd(xa0,xb0,xc0);
	xc0=_mm512_fmadd_pd(xa1,xb1,xc0);
	xc0=_mm512_fmadd_pd(xa2,xb2,xc0);
	xc0=_mm512_fmadd_pd(xa3,xb3,xc0);
	xc0=_mm512_fmadd_pd(xa4,xb4,xc0);
	xc0=_mm512_fmadd_pd(xa5,xb5,xc0);
	xc0=_mm512_fmadd_pd(xa6,xb6,xc0);
	xc0=_mm512_fmadd_pd(xa7,xb7,xc0);

	xa8=_mm512_set1_pd(a[8+i]);
	xa9=_mm512_set1_pd(a[9+i]);
	xa10=_mm512_set1_pd(a[10+i]);
	xa11=_mm512_set1_pd(a[11+i]);
	xa12=_mm512_set1_pd(a[12+i]);
	xa13=_mm512_set1_pd(a[13+i]);
	xa14=_mm512_set1_pd(a[14+i]);
	xa15=_mm512_set1_pd(a[15+i]);

	xc0=_mm512_fmadd_pd(xa8,xb8,xc0);
	xc0=_mm512_fmadd_pd(xa9,xb9,xc0);
	xc0=_mm512_fmadd_pd(xa10,xb10,xc0);
	xc0=_mm512_fmadd_pd(xa11,xb11,xc0);
	xc0=_mm512_fmadd_pd(xa12,xb12,xc0);
	xc0=_mm512_fmadd_pd(xa13,xb13,xc0);
	xc0=_mm512_fmadd_pd(xa14,xb14,xc0);
	xc0=_mm512_fmadd_pd(xa15,xb15,xc0);
	xc0=_mm512_fmadd_pd(xa16,xb16,xc0);

	xa16=_mm512_set1_pd(a[16+i]);
	xa17=_mm512_set1_pd(a[17+i]);
	xa18=_mm512_set1_pd(a[18+i]);
	xa19=_mm512_set1_pd(a[19+i]);
	xa20=_mm512_set1_pd(a[20+i]);
	xa21=_mm512_set1_pd(a[21+i]);
	xa22=_mm512_set1_pd(a[22+i]);


	xc0=_mm512_fmadd_pd(xa17,xb17,xc0);
	xc0=_mm512_fmadd_pd(xa18,xb18,xc0);
	xc0=_mm512_fmadd_pd(xa19,xb19,xc0);
	xc0=_mm512_fmadd_pd(xa20,xb20,xc0);
	xc0=_mm512_fmadd_pd(xa21,xb21,xc0);
	xc0=_mm512_fmadd_pd(xa22,xb22,xc0);

	_MM512_MASK_STOREU_PD(&c[i+16], xc0,127);

    }
//    print512d(xc0);

#endif
#else
    int i,j,n;

    /* printf("%ld %ld %ld\n",c,a,b); */
    /* return ; */

#pragma simd
    for (i=0;i<23;i++)
    {
	for (j=0;j<23;j++)
	{
	    int n=i*23;
	    c[n+j]=c[n+j]+a[n]*b[j]
		+a[n+1]*b[23+j]
		+a[n+2]*b[2*23+j]
		+a[n+3]*b[3*23+j]
		+a[n+4]*b[4*23+j]
		+a[n+5]*b[5*23+j]
		+a[n+6]*b[6*23+j]
		+a[n+7]*b[7*23+j]
		+a[n+8]*b[8*23+j]
		+a[n+9]*b[9*23+j]
		+a[n+10]*b[10*23+j]
		+a[n+11]*b[11*23+j]
		+a[n+12]*b[12*23+j]
		+a[n+13]*b[13*23+j]
		+a[n+14]*b[14*23+j]
		+a[n+15]*b[15*23+j]
		+a[n+16]*b[16*23+j]
		+a[n+17]*b[17*23+j]
		+a[n+18]*b[18*23+j]
		+a[n+19]*b[19*23+j]
		+a[n+20]*b[20*23+j]
		+a[n+21]*b[21*23+j]
		+a[n+22]*b[22*23+j];
	}
    }

#endif
}
