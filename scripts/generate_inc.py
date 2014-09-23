import sys
if(len(sys.argv)==4):
    for m in range(1,int(sys.argv[1])+1):
        for k in range(1,int(sys.argv[2])+1):
            for n in range(1,int(sys.argv[3])+1):
#                print "extern \"C\" {"
                print "__declspec(target(mic))"
                print "void xsmm_dnn_"+str(m)+"_"+str(n)+"_"+str(k)+"(const double* a, const double* b, double* c);"
#                print "}"

#    print "extern \"C\" {"
    print "__declspec(target(mic)) void xsmm_dnn(int M, int N, int K, const double* a, const double* b, double* c);"
#    print "}"
