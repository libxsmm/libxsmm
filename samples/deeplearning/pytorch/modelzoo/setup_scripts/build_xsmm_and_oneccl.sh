
( cd  libxsmm && make realclean && make -j 16 AVX=2 CC=gcc CXX=g++ )

#git clone -b working_1.6 https://github.com/ddkalamk/oneCCL.git
( cd oneCCL && rm -rf build && mkdir -p build && cd build && CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_DISABLE_SYCL=1 && make -j install ) && source oneCCL/install/env/setvars.sh
