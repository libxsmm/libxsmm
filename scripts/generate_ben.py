###############################################################################
## Copyright (c) 2013-2014, Intel Corporation                                ##
## All rights reserved.                                                      ##
##                                                                           ##
## Redistribution and use in source and binary forms, with or without        ##
## modification, are permitted provided that the following conditions        ##
## are met:                                                                  ##
## 1. Redistributions of source code must retain the above copyright         ##
##    notice, this list of conditions and the following disclaimer.          ##
## 2. Redistributions in binary form must reproduce the above copyright      ##
##    notice, this list of conditions and the following disclaimer in the    ##
##    documentation and/or other materials provided with the distribution.   ##
## 3. Neither the name of the copyright holder nor the names of its          ##
##    contributors may be used to endorse or promote products derived        ##
##    from this software without specific prior written permission.          ##
##                                                                           ##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       ##
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         ##
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     ##
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      ##
## HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    ##
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  ##
## TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    ##
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    ##
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      ##
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        ##
## SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              ##
###############################################################################
## Christopher Dahnken (Intel Corp.), Hans Pabst (Intel Corp.),
## Alfio Lazzaro (CRAY Inc.), and Gilles Fourestey (CSCS)
###############################################################################
print "#include<stdio.h>"
print "#include <stdlib.h>"
print "#include <sys/time.h>"
print "#include <xsmmknc.h>"


print "__declspec(target(mic))"
print "double mytime() {"
print "  timeval a;"
print "  gettimeofday(&a, 0);"
print "  return (double)(a.tv_sec*1000 + a.tv_usec/1000.0);"
print "}"

for n in range(4,30):
    print "__declspec(target(mic))"
    print "void bench_"+str(n)+"(void){"
    print "int length=1;"
    print "double t1,t2;"
    print "double perf1,perf2;"
    print "long iterations = "+str(1000000000/(2*n*n*n))+";"
    print "double* a = new double["+str(n*n)+"];"
    print "double* b = new double["+str(n*n)+"];"
    print "double* c = new double["+str(n*n)+"];"
    print ""
    print "for(int i=0;i<"+str(n*n)+";i++){"
    print "a[i]=((double)i)/((double)"+str(n*n)+");"
    print "b[i]=((double)i)/((double)"+str(n*n)+");"
    print "c[i]=((double)i)/((double)"+str(n*n)+");"
    print "}"
#    print "t1=mytime();"
#    print "for(long n=0;n<iterations;n++){"
#    print "xgemm_1_"+str(n)+"_"+str(n)+"_"+str(n)+"(a,b,c);"
#    print "}"
#    print "t2=mytime();"
#    print "perf1=((double)iterations)*((double)"+str(2*n*n*n)+")/(t2-t1)/((double) 1000)/((double) 1000);"
    print ""
    print "for(int i=0;i<"+str(n*n)+";i++){"
    print "a[i]=((double)i)/((double)"+str(n*n)+");"
    print "b[i]=((double)i)/((double)"+str(n*n)+");"
    print "c[i]=((double)i)/((double)"+str(n*n)+");"
    print "}"
    print "t1=mytime();"
    print "for(long n=0;n<iterations;n++){"
    print "xsmm_dnn_"+str(n)+"_"+str(n)+"_"+str(n)+"(a,b,c);"
    print "}"
    print "t2=mytime();"
    print "perf2=((double)iterations)*((double)"+str(2*n*n*n)+")/(t2-t1)/((double) 1000)/((double) 1000);"
    print ""
    print "printf(\"%d \\t%f \\t %f \\n\", "+str(n)+", perf1, perf2);"
    print "}"


print "int main(void){"
print "#pragma offload target(mic:0)"
print "{"
print "printf(\"starting benchmark\\n\");"
for n in range(4,30):
    print "bench_"+str(n)+"();"
print "}"
print "}"
