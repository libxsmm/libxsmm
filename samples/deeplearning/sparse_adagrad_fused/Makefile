VERIFY_CORRECTNESS=1
USE_LIBXSMM_JIT=0

ifeq ($(USE_LIBXSMM_JIT),1)
EXTRA_FLAGS += -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread  -I../../../include ../../../lib/libxsmmnoblas.a  ../../../lib/libxsmm.a
endif
#deflist +=

yesnolist += VERIFY_CORRECTNESS
yesnolist += USE_LIBXSMM_JIT
#yesnolist +=

DEFS += $(strip $(foreach var, $(yesnolist), $(if $(filter 1, $($(var))), -D$(var))))
DEFS += $(strip $(foreach var, $(deflist), $(if $($(var)), -D$(var)=$($(var)))))

CC=icc

all: sparse_adagrad

sparse_adagrad: sparse_adagrad.cpp
	$(CC) -o sparse_adagrad -g -O2 -mavx512f -std=c++11 -fopenmp $(DEFS) sparse_adagrad.cpp $(EXTRA_FLAGS)

clean:
	rm -f sparse_adagrad
