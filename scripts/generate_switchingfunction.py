import math
import sys

def create_symmetric_interface(M,K,N):
    print "#include<micsmm.h>"
    print "#include <mkl.h>"
    print "__declspec( target (mic))"
    print "void smm_dnn(int M, int N, int K, double* a, double* b, double* c){"
    print "if((M<="+str(M)+")&&(K<="+str(N)+")&&(N<="+str(K)+")){"
    print "   int v=((M-1)<<10)+((K-1)<<5)+(N-1);"
    print "   switch(v){"
    for m in range(1,M+1):
        for n in range(1,N+1):
           for k in range(1,K+1):
                print "      case "+str(((m-1)<<10)+((k-1)<<5)+(n-1))+":"
                print "            smm_dnn_"+str(m)+"_"+str(n)+"_"+str(k)+"(a,b,c);"
                print "            break;"  
    print "      default:"
    print "            printf(\"Can't find this matrix size \\n \");"
    print "            break;"
    print "   }"
    print "} else{"
    print "    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, (double)1,a, K, b,N, (double)1, c, N);"
    print "}"
    print "}"

create_symmetric_interface(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
