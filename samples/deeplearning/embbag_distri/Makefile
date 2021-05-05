USE_RTM=1
STREAMING_WRITES=1
USE_MPI=1
USE_CCL=0
USE_FP16=0

ifeq ($(USE_CCL),1)
override USE_MPI=0
INCLUDE_DIR:=$(CCL_ROOT)/include
LIB_DIR:=$(CCL_ROOT)/lib
CCL_FLAGS:=-I$(INCLUDE_DIR) -lccl -L$(LIB_DIR)
else
CCL_FLAGS=
endif

ifeq ($(USE_LIBXSMM_JIT),1)
CCL_FLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread  -I../../../include ../../../lib/libxsmmnoblas.a  ../../../lib/libxsmm.a
endif
#deflist +=

yesnolist += USE_RTM
yesnolist += FWD_ROW_BLOCKING
yesnolist += STREAMING_WRITES
yesnolist += USE_MPI
yesnolist += USE_CCL
yesnolist += USE_FP16

DEFS += $(strip $(foreach var, $(yesnolist), $(if $(filter 1, $($(var))), -D$(var))))
DEFS += $(strip $(foreach var, $(deflist), $(if $($(var)), -D$(var)=$($(var)))))

CC=icc

ifeq ($(USE_MPI),1)
CC=mpiicc
endif


all: main

main: main.cpp utils.cpp utils.h rtm.h
	$(CC) -o main -g -O2 -xCOMMON-AVX512 -stdc++11 $(DEFS) utils.cpp main.cpp -qopenmp $(CCL_FLAGS)

clean:
	rm -f main
