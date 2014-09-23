#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <micsmmmisc.h>
#include <immintrin.h>
#include <mkl.h>


extern "C" {
  __declspec( target (mic)) 
  void smm_dnn(int m, int n, int k,double* a, double* b, double* c);
}
extern "C" {
  __declspec( target (mic)) 
  void micgemm_2_1_1_12(double* a,double* b,double* c);
}


__declspec( target (mic))
double mytime() {
  timeval a;
  gettimeofday(&a, 0);
  return (double)(a.tv_sec*1000 + a.tv_usec/1000.0);
}


__declspec( target (mic))
void fillrandom(double* p,int m, int n){
  for(int i=0;i<m*n;i++)
    p[i]=(double(rand()) / double(RAND_MAX));
}


void copymatrix(double* a,double* b, int m, int n)
{
  for(int i=0;i<m*n;i++)
    b[i]=a[i];
}


double compmatrices(double* a,double* b, int m, int n){
  double s=0;
  for(int i=0;i<m*n;i++)
    s+=(b[i]-a[i])*(b[i]-a[i]);
  return s;
}


void corrsmm(double* a,double* b,double* c,int M,int K,int N){
  int m,k,n;
  for(m=0;m<M;m++){
    for(n=0;n<N;n++){
      for(k=0;k<K;k++){
        //c[m*M+n]+=a[m*M+k]*b[k*K+n];
        c[m*N+n]+=a[m*K+k]*b[k*N+n];
      }
    }
  }
  // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, double(1),
  //   a, K, b,N, double(1), c, N);
}


void corrsmm2(double* a,double* b,double* c,int M,int K,int N){
  int m,k,n;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, double(1),
    a, K, b,N, double(1), c, N);
}


void printmatrix(double* a, int M, int N){
  for(int m=0;m<M;m++){
    for(int n=0;n<N;n++){
      printf("%f ", a[m*N+n]);
    }
    printf("\n");
  }
}


void mmtest(void){
  double ONEGFLOP=1000.0*1000.0*1000.0;
  long flopspermm;
  long iterations;
  int m,k,n,i;
  int maxsize=13;
  int minsize=1;
  double* a;
  double* b;
  double* c;
  double* aa;
  double* bb;
  double* cc;
  double* aaa;
  double* bbb;
  double* ccc;
  double t1,t2;
  double perf;

  a=new double[maxsize*maxsize];
  b=new double[maxsize*maxsize];
  c=new double[maxsize*maxsize];
  aa=new double[maxsize*maxsize];
  bb=new double[maxsize*maxsize];
  cc=new double[maxsize*maxsize];
  aaa=new double[maxsize*maxsize];
  bbb=new double[maxsize*maxsize];
  ccc=new double[maxsize*maxsize];
  m=2;
  k=2;
  n=1;
  fillrandom(a,m,k);
  copymatrix(a,aa,m,k);
  copymatrix(a,aaa,m,k);
  fillrandom(b,k,n);
  copymatrix(b,bb,k,n);
  copymatrix(b,bbb,k,n);
  fillrandom(c,m,n);
  copymatrix(c,cc,m,n);
  copymatrix(c,ccc,m,n);
  
#pragma offload target(mic:0) in(m,k,n) in(a:length(maxsize*maxsize)) in(b:length(maxsize*maxsize)) inout(c:length(maxsize*maxsize))
  {
    smm_dnn(m,n,k,a,b,c);
  }
  corrsmm(aa,bb,cc,m,k,n);
  corrsmm2(aaa,bbb,ccc,m,k,n);
  printf("instrinsics:\n");
  printmatrix(c,m,n);
  printf("C:\n");
  printmatrix(cc,m,n);
  printf("BLAS:\n");
  printmatrix(ccc,m,n);
  printf("\n");
  double s = compmatrices(c,ccc,m,n);
  double ss = compmatrices(cc,ccc,m,n);
  printf("%d %d %d = %f %f\n",m,k,n,s,ss); 
  printf("\n");
  free(a);
  free(b);
  free(c);
  free(aa);
  free(bb);
  free(cc);
  free(aaa);
  free(bbb);
  free(ccc);
}


void corr(void){
  double ONEGFLOP=1000.0*1000.0*1000.0;
  long flopspermm;
  long iterations;
  int m,k,n,i;
  int maxsize=12;
  int minsize=1;
  double* a;
  double* b;
  double* c;
  double* aa;
  double* bb;
  double* cc;
  double* aaa;
  double* bbb;
  double* ccc;
  double t1,t2;
  double perf;

  a=new double[maxsize*maxsize];
  b=new double[maxsize*maxsize];
  c=new double[maxsize*maxsize];
  aa=new double[maxsize*maxsize];
  bb=new double[maxsize*maxsize];
  cc=new double[maxsize*maxsize];
  aaa=new double[maxsize*maxsize];
  bbb=new double[maxsize*maxsize];
  ccc=new double[maxsize*maxsize];
  for(m=minsize;m<=maxsize;m++)
  {
    for(k=minsize;k<=maxsize;k++)
    {
      for(n=minsize;n<=maxsize;n++)
      {
        fillrandom(a,m,k);
        copymatrix(a,aa,m,k);
        copymatrix(a,aaa,m,k);
        fillrandom(b,k,n);
        copymatrix(b,bb,k,n);
        copymatrix(b,bbb,k,n);
        fillrandom(c,m,n);
        copymatrix(c,cc,m,n);
        copymatrix(c,ccc,m,n);
        
#       pragma offload target(mic:0) in(m,k,n) in(a:length(maxsize*maxsize)) in(b:length(maxsize*maxsize)) inout(c:length(maxsize*maxsize))
        {
          smm_dnn(m,n,k,a,b,c);
        }
        corrsmm(aa,bb,cc,m,k,n);
        corrsmm2(aaa,bbb,ccc,m,k,n);
        double s = compmatrices(c,ccc,m,n);
        double ss = compmatrices(cc,ccc,m,n);
        printf("%d %d %d = %f %f\n",m,k,n,s,ss); 
      }
    }
  }
  free(a);
  free(b);
  free(c);
  free(aa);
  free(bb);
  free(cc);
  free(aaa);
  free(bbb);
  free(ccc);
}


void bench(void){
  double ONEGFLOP=1000.0*1000.0*1000.0;
  long flopspermm;
  long iterations;
  int m,k,n,i;
  int maxsize=32;
  int minsize=2;
  double* a;
  double* b;
  double* c;
  double t1,t2;
  double perf;

  a=new double[maxsize*maxsize];
  b=new double[maxsize*maxsize];
  c=new double[maxsize*maxsize];
  for(m=minsize;m<=maxsize;m++) {
    n=k=m;
    fillrandom(a,m,k);
    fillrandom(b,k,n);
    fillrandom(c,m,n);
    // compute the iterations
    flopspermm=2*m*k*n;
    iterations=long(ONEGFLOP/double(flopspermm));
#   pragma offload target(mic:0) in(m,k,n) in(a:length(m*k)) in(b:length(k*n)) inout(c:length(m*n)) in(iterations) inout(t1,t2)
    {
      t1=mytime();
      for(i=0;i<iterations;i++){
        smm_dnn(m,n,k,a,b,c);
      }
      t2=mytime()-t1;
    }
    perf=double(iterations)*double(flopspermm)/t2/double(1000)/double(1000);
    printf("%d %d %d = %f\n",m,k,n,perf); //fflush(stdout);
  }

  free(a);
  free(b);
  free(c);
}

int main(void){
  //mmtest();
  //corr();
  bench();
}
