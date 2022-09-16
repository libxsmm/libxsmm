USE_RTM=0
STREAMING_WRITES=1
USE_MPI=0
USE_CCL=0
USE_FP16=0
VERIFY_CORRECTNESS=1
COUNT_UNIQUE=1
USE_LIBXSMM_JIT=1

ifeq ($(USE_CCL),1)
override USE_MPI=0
INCLUDE_DIR:=$(CCL_ROOT)/include
LIB_DIR:=$(CCL_ROOT)/lib
CCL_FLAGS:=-I$(INCLUDE_DIR) -lccl -L$(LIB_DIR)
else
CCL_FLAGS=
endif

ifeq ($(USE_LIBXSMM_JIT),1)
#CCL_FLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread  -I../../../include ../../../lib/libxsmmnoblas.a  ../../../lib/libxsmm.a
CCL_FLAGS += -I../../../include ../../../lib/libxsmmnoblas.a  ../../../lib/libxsmm.a -ldl
endif
#deflist +=

yesnolist += USE_RTM
yesnolist += FWD_ROW_BLOCKING
yesnolist += STREAMING_WRITES
yesnolist += USE_MPI
yesnolist += USE_CCL
yesnolist += USE_FP16
yesnolist += VERIFY_CORRECTNESS
yesnolist += COUNT_UNIQUE
yesnolist += USE_LIBXSMM_JIT

DEFS += $(strip $(foreach var, $(yesnolist), $(if $(filter 1, $($(var))), -D$(var))))
DEFS += $(strip $(foreach var, $(deflist), $(if $($(var)), -D$(var)=$($(var)))))

CXX=g++

ifeq ($(USE_MPI),1)
CXX=mpicxx
endif


all: main

main: main.cpp utils.cpp utils.h rtm.h EmbeddingBag.h dist.h
	$(CXX) -o main -g -O2 -mavx512f -mrtm -std=c++11 $(DEFS) utils.cpp main.cpp -fopenmp $(CCL_FLAGS)

clean:
	rm -f main
