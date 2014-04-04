icc -mkl  -I ../include/  -c benchmark.cpp
icc -mkl benchmark.o ../src/gemms/micsmm.a

