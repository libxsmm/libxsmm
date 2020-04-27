ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
#TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
#TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
DEPDIR = ../../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = xsmm_lstm

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

GCC_OMP_FLAGS = -fopenmp
BLAS = 0
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

override CXX=g++

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
           $(DEPDIR)/include/libxsmm_source.h
CCXSRCS := $(wildcard $(SRCDIR)/*.cc)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(call qndir,$(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(call qndir,$(CSOURCS:.c=-c.o)))
SOURCES := $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CCXOBJS) $(COBJCTS)
XFILES := $(OUTDIR)/libxsmm_lstm.so
TF_FLAGS := $(BLDDIR)/tf_flags

-include $(TF_FLAGS)

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(TF_FLAGS):
	$(eval TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
	$(eval TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
	@echo "TF_CFLAGS = $(TF_CFLAGS)"
	@echo "TF_LFLAGS = $(TF_LFLAGS)"
	@mkdir -p $(BLDDIR)
	@echo "TF_CFLAGS=$(TF_CFLAGS)" > $@
	@echo "TF_LFLAGS=$(TF_LFLAGS)" >> $@

$(OUTDIR)/libxsmm_lstm.so: $(OUTDIR)/.make $(CCXOBJS) $(COBJCTS) $(LIBDEP) $(EXTDEP) $(TF_FLAGS)
	$(LD) -o $@ -shared $(CCXOBJS) $(COBJCTS) $(MAINLIB) $(TF_LFLAGS) -lsvml -liomp5 -fPIC

$(BLDDIR)/%-cc.o: $(SRCDIR)/%.cc .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc $(TF_FLAGS)
	g++ -std=c++11 $(GCC_OMP_FLAGS) $(TF_CFLAGS) -fPIC -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: install
install: $(XFILES) setup.py $(OUTDIR)/__init__.py
	python setup.py bdist_wheel
	pip install -U dist/xsmm_lstm-*.whl

.PHONY: uninstall
uninstall:
	pip uninstall xsmm_lstm

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS)
endif
	@rm -f .make .state $(XFILES) $(OUTDIR)/*.pyc
	@rm -rf dist xsmm_lstm.egg-info


