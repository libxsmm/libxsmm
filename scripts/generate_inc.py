import sys
if(len(sys.argv)==4):
#    start=int(sys.argv[1])
#    stop=int(sys.argv[2])
    # for n in range(start,stop):
    #     print "extern \"C\" {"
    #     print "__declspec( target (mic))"
    #     print "void micgemm_0_"+str(n)+"_"+str(n)+"_"+str(n)+"(double* a,double* b,double* c);"
    #     print "}"
    # for n in range(start,stop):
    #     print "extern \"C\" {"
    #     print "__declspec( target (mic))"
    #     print "void micgemm_1_"+str(n)+"_"+str(n)+"_"+str(n)+"(double* a,double* b,double* c);"
    #     print "}"
    for m in range(1,int(sys.argv[1])):
        for k in range(1,int(sys.argv[2])):
            for n in range(1,int(sys.argv[3])):
#                print "extern \"C\" {"
                print "__declspec( target (mic))"
                print "void smm_dnn_"+str(m)+"_"+str(n)+"_"+str(k)+"(double* a,double* b,double* c);"
#                print "}"

#    print "extern \"C\" {"
    print "__declspec( target (mic)) void smm_dnn(int M, int N, int K, double* a, double* b, double* c);"
#    print "}"
