import math
import sys


def createigemm2(M,K,N):
    iparts=int(math.floor(N/8))
    fparts=N%8
    maskval=(1<<fparts)-1
    maxmaskval=(1<<8)-1
    if fparts==0:
        nparts=iparts
    else:
        nparts=iparts+1
#    print "#include <immintrin.h>"
#    print "#include <micsmmmisc.h>"
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
#        if(maskval!=maxmaskval):
        for k in range(0,K):
            print " xb"+str(k)+" = _MM512_MASK_LOADU_PD(&b["+str(N*k)+"+"+str(n)+"]," +str(maskval)+");"
#        else:
#            for k in range(0,K):
#                print "xb"+str(k)+" = _MM512_LOADU_PD(&b["+str(N*k)+"+"+str(n)+"]);"
        print "for(i=0;i<"+str(M*N)+";i+="+str(N)+"){"
#        if(maskval!=maxmaskval):
        print "    xc0 = _MM512_MASK_LOADU_PD(&c[i+"+str(n)+"]," +str(maskval)+");"
#        else:
#            print "    xc0 = _MM512_LOADU_PD(&c[i+"+str(n)+"]);"

        for k in range(0,K):
            print "    xa"+str(k)+"=_mm512_set1_pd(a[i+"+str(k)+"]);"
        for k in range(0,K):
            print "    xc0=_mm512_fmadd_pd(xa"+str(k)+",xb"+str(k)+",xc0);"
#        if(maskval!=maxmaskval):
        print "    _MM512_MASK_STOREU_PD(&c[i+"+str(n)+"],xc0," +str(maskval)+");"
#        else:
#            print "    _MM512_STOREU_PD(&c[i+"+str(n)+"],xc0);"

        print "}"
    print "#else"    
    print "for(int m=0;m<"+str(M)+";m++){"
    print "   for(int n=0;n<"+str(N)+";n++){"
    print "      for(int k=0;k<"+str(K)+";k++){"
    print "         c[m*"+str(N)+"+n]+=a[m*"+str(K)+"+k]*b[k*"+str(N)+"+n];"
    print "      }"
    print "   }"
    print "}"
#    print "exit(0);"
#    print "micgemm_1_"+str(M)+"_"+str(K)+"_"+str(N)+"(a,b,c);"
    print "#endif"
    print "}"
    print " "

def createigemm20130803(M,K,N):
    iparts=int(math.floor(N/8))
    fparts=N%8
    maskval=(1<<fparts)-1
    if fparts==0:
        nparts=iparts
    else:
        nparts=iparts+1
    print "__declspec( target (mic))"
    print "void micgemm_2_"+str(M)+"_"+str(K)+"_"+str(N)+"(double* a,double* b,double* c){"
    print "#ifdef __MIC__"
    print "printf(\"micgemm_2_"+str(M)+"_"+str(K)+"_"+str(N)+"\\n\");"
    print "int i;"
    for n in range(0,K):
        print "__m512d xa"+str(n)+";"
    for n in range(0,K):
        print "__m512d xb"+str(n)+";"
    print "__m512d xc0;"

    for n in range(0,8*nparts,8):
        nn=min(n+7,N-1)
        maskval=(1<<(nn-n+1))-1
        for k in range(0,K):
            print " xb"+str(k)+" = _MM512_MASK_LOADU_PD(&b["+str(N*k)+"+"+str(n)+"]," +str(maskval)+");"
        print "for(i=0;i<"+str(M*N)+";i+="+str(N)+"){"
        print "    xc0 = _MM512_MASK_LOADU_PD(&c[i+"+str(n)+"]," +str(maskval)+");"

        for k in range(0,K):
            print "    xa"+str(k)+"=_mm512_set1_pd(a[i+"+str(k)+"]);"
        for k in range(0,K):
            print "    xc0=_mm512_mask3_fmadd_pd(xa"+str(k)+",xb"+str(k)+",xc0," +str(maskval)+");"
        print "    _MM512_MASK_STOREU_PD(&c[i+"+str(n)+"],xc0," +str(maskval)+");"
        print "}"
    print "#else"    
# ------------------------------------------------------
    print "printf(\"cppgemm_2_"+str(M)+"_"+str(K)+"_"+str(N)+"\\n\");"
    print "for(int m=0;m<"+str(M)+";m++){"
    print "   for(int n=0;n<"+str(N)+";n++){"
    print "      for(int k=0;k<"+str(K)+";k++){"
    print "         c[m*"+str(N)+"+n]+=a[m*"+str(K)+"+k]*b[k*"+str(N)+"+n];"
    print "      }"
    print "   }"
    print "}"
#    print "exit(0);"
#    print "micgemm_1_"+str(M)+"_"+str(K)+"_"+str(N)+"(a,b,c);"
    print "#endif"
    print "}"
    print " "
    

def createigemm(M,N,K):
    iparts=int(math.floor(N/8))
    fparts=N%8
    maskval=(1<<fparts)-1
    if fparts==0:
        nparts=iparts
    else:
        nparts=iparts+1
    print "#include <immintrin.h>"
    print "#include <micsmmmisc.h>"
    print "#include <mkl.h>"


    print "__declspec( target (mic))"
    print "void smm_dnn_"+str(M)+"_"+str(N)+"_"+str(K)+"(double* a,double* b,double* c){"
    print "#ifdef __MIC__"
#    print "printf(\"micgemm_2_"+str(M)+"_"+str(K)+"_"+str(N)+"\\n\");"
    print "int i;"
    for n in range(0,K):
        print "__m512d xa"+str(n)+";"
    for n in range(0,K):
        print "__m512d xb"+str(n)+";"
    print "__m512d xc0;"

    for n in range(0,8*nparts,8):
        nn=min(n+7,N-1)
        maskval=(1<<(nn-n+1))-1
        for k in range(0,K):
            print " xb"+str(k)+" = _MM512_MASK_LOADU_PD(&b["+str(N*k)+"+"+str(n)+"]," +str(maskval)+");"
        print "for(i=0;i<"+str(M)+";i+="+str(1)+"){"
        print "    xc0 = _MM512_MASK_LOADU_PD(&c[i*"+str(N)+"+"+str(n)+"]," +str(maskval)+");"
        for k in range(0,K):
            print "    xa"+str(k)+"=_mm512_set1_pd(a[i*"+str(K)+"+"+str(k)+"]);"
        for k in range(0,K):
            print "    xc0=_mm512_mask3_fmadd_pd(xa"+str(k)+",xb"+str(k)+",xc0," +str(maskval)+");"
        print "    _MM512_MASK_STOREU_PD(&c[i*"+str(N)+"+"+str(n)+"],xc0," +str(maskval)+");"
        print "}"
    print "#else"    
# ------------------------------------------------------
    print "printf(\"cppgemm_2_"+str(M)+"_"+str(K)+"_"+str(N)+"\\n\");"
    print "for(int m=0;m<"+str(M)+";m++){"
    print "   for(int n=0;n<"+str(N)+";n++){"
    print "      for(int k=0;k<"+str(K)+";k++){"
    print "         c[m*"+str(N)+"+n]+=a[m*"+str(K)+"+k]*b[k*"+str(N)+"+n];"
    print "      }"
    print "   }"
    print "}"
#    print "exit(0);"
#    print "micgemm_1_"+str(M)+"_"+str(K)+"_"+str(N)+"(a,b,c);"
    print "#endif"
    print "}"
    print " "




createigemm(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
