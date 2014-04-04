# N=8
# M=8
# K=8

# for n in range(0,N):
#     for m in range(0,M):
#         for k in range(0,K):
#             print "c["+str(n*N+m)+"]=c["+str(n*N+m)+"]+a["+str(n*N+k)+"]*b["+str(k*K+m)+"];"

import math
import sys

def createsgemm(M,K,N):
    print "__declspec( target (mic))"
    print "void micgemm_0_"+str(M)+"_"+str(K)+"_"+str(N)+"(double* a,double* b,double* c){"
    print "int n,m,k;"
    print "#pragma simd"
    print "for (m=0;m<"+str(M)+";m++)"
    print "{"
    print "for (n=0;n<"+str(N)+";n++)"
    print "{"
    print "for (k=0;k<"+str(K)+";k++)"
    print "{"
    print "c[m*"+str(M)+"+n]=c[m*"+str(M)+"+n]+a[m*"+str(M)+"+k]*b[k*"+str(K)+"+n];"
    print "}"
    print "}"
    print "}"    
    print "}"
    print " "


def createvgemm(M,K,N):
    print "__declspec( target (mic))"
    print "void micgemm_1_"+str(M)+"_"+str(K)+"_"+str(N)+"(double* a,double* b,double* c){"
    print "int i,j,n;"
    print "#pragma simd"
    print "for (i=0;i<"+str(M)+";i++)"
    print "{"
    print "for (j=0;j<"+str(N)+";j++)"
    print "{"
    print "int n=i*"+str(M)+";"
    s="c[n+j]=c[n+j]"
    for k in range(0,K):
        s=s+"+a[n+"+str(k)+"]*b["+str(k*K)+"+j]"
    s=s+";"
    print s
    print "}"
    print "}"    
    print "}"
    print " "
    

def createigemm(M,K,N):
    iparts=int(math.floor(N/8))
    fparts=N%8
    maskval=(1<<fparts)-1
    maxmaskval=(1<<8)-1
    if fparts==0:
        nparts=iparts
    else:
        nparts=iparts+1
    print "__declspec( target (mic))"
    print "void micgemm_2_"+str(M)+"_"+str(K)+"_"+str(N)+"(double* a,double* b,double* c){"
    print "#ifdef __MIC__"
    print "int i;"
    for n in range(0,K):
        print "__m512d xa"+str(n)+";"
    for n in range(0,K):
        print "__m512d xb"+str(n)+";"
    print "__m512d xc0;"

    for n in range(0,8*nparts,8):
        nn=min(n+7,N-1)
        maskval=(1<<(nn-n+1))-1
#        print n, nn, maskval, maxmaskval
        if(maskval!=maxmaskval):
            for k in range(0,K):
                print " xb"+str(k)+" = _MM512_MASK_LOADU_PD(&b["+str(K*k)+"+"+str(n)+"]," +str(maskval)+");"
        else:
            for k in range(0,K):
                print "xb"+str(k)+" = _MM512_LOADU_PD(&b["+str(K*k)+"+"+str(n)+"]);"
        print "for(i=0;i<"+str(M*M)+";i+="+str(M)+"){"
        if(maskval!=maxmaskval):
            print "    xc0 = _MM512_MASK_LOADU_PD(&c[i+"+str(n)+"]," +str(maskval)+");"
        else:
            print "    xc0 = _MM512_LOADU_PD(&c[i+"+str(n)+"]);"

#        print "xc0 = _MM512_MASK_LOADU_PD(&c[i+16],127);"
        for k in range(0,K):
            print "    xa"+str(k)+"=_mm512_set1_pd(a[i+"+str(k)+"]);"
        for k in range(0,K):
            print "    xc0=_mm512_fmadd_pd(xa"+str(k)+",xb"+str(k)+",xc0);"
        if(maskval!=maxmaskval):
            print "    _MM512_MASK_STOREU_PD(&c[i+"+str(n)+"],xc0," +str(maskval)+");"
        else:
            print "    _MM512_STOREU_PD(&c[i+"+str(n)+"],xc0);"

        print "}"
    print "#else"    
#    print "micgemm_1_"+str(M)+"_"+str(K)+"_"+str(N)+"(a,b,c);"
    print "#endif"
    print "}"
    print " "


def create_symmetric_interface():
    print "__declspec( target (mic))"
    print "void micssm2(double* a, double* b, double* c, int M, int K, int N){"
    print "if((M<=32)&&(K<=32)&&(N<=32)){"
    print "   int v=((M-1)<<10)+((K-1)<<5)+(N-1);"
    print "   switch(v){"
    for m in range(1,33):
        for k in range(1,33):
            for n in range(1,3):
                print "      case "+str(((m-1)<<10)+((k-1)<<5)+(n-1))+":"
#                print "      __attribute__((target(mic)))"
                print "      micgemm_2_"+str(m)+"_"+str(k)+"_"+str(n)+"(a,b,c);"
                print "      break;"
    print "   }"    
    print "} else{"
    print "}"    
    print "}"



print "#include <immintrin.h>"
print "#include <micsmmmisc.h>"
if(len(sys.argv)==3):
    start=int(sys.argv[1])
    stop=int(sys.argv[2])
    for m in range(start,stop):
        for n in range(start,stop):
           for k in range(start,stop):
                createigemm(m,n,k)
#    for m in range(start,stop):
#        for k in range(start,stop):
#            for n in range(start,stop):
#                createvgemm(m,k,n)
#    for m in range(start,stop):
#        for k in range(start,stop):
#            for n in range(start,stop):
#                createsgemm(m,k,n)

create_symmetric_interface()






#creategemm(9,9,9)

#for n in range(0,N,nblocksize):
