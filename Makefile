
# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise 
ROW_MAJOR ?= 1

# M, N, K values of the generated matrices
INDICES_M ?= $(shell seq 1 5)
INDICES_N ?= $(shell seq 1 5)
INDICES_K ?= $(shell seq 1 5)

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Use aligned Store and/or aligned Load instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

# THRESHOLD problem size (M x N x K); determines when to use BLAS 
THRESHOLD ?= $(shell echo $$((24 * 24 * 24)))

ROOTDIR ?= .
SCRDIR = $(ROOTDIR)/scripts
OBJDIR = $(ROOTDIR)/build
INCDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
LIBDIR = $(ROOTDIR)/lib

INDICES ?= $(foreach m,$(INDICES_M),$(foreach n,$(INDICES_N),$(foreach k,$(INDICES_K),$m_$n_$k)))

# prefer the Intel compiler and prefer a C compiler (over C++)
ifneq ($(shell which icc 2> /dev/null),)
	CC := icc
	AR := xiar
	CFLAGS     := -Wall -std=c99 -O2 -ipo -fPIC -fno-alias -ansi-alias -xHost -opt-assume-safe-padding -mkl=sequential
	CFLAGS_MIC := -Wall -std=c99 -O2 -ipo -fPIC -fno-alias -ansi-alias -mmic  -opt-assume-safe-padding -mkl=sequential
else ifneq ($(shell which icpc 2> /dev/null),)
	CC := icpc
	AR := xiar
	CFLAGS :=     -Wall -O2 -ipo -fPIC -fno-alias -ansi-alias -xHost -opt-assume-safe-padding -mkl=sequential
	CFLAGS_MIC := -Wall -O2 -ipo -fPIC -fno-alias -ansi-alias -mmic  -opt-assume-safe-padding -mkl=sequential
#else ifneq ($(shell which icl 2> /dev/null),)
#	CC := icl
#	AR := xilib
#else ifneq ($(shell which cl 2> /dev/null),)
#	CC := cl
else ifneq ($(shell which gcc 2> /dev/null),)
	CC := gcc
	CFLAGS := -Wall -std=c99 -O2 -march=native
else ifneq ($(shell which g++ 2> /dev/null),)
	CC := g++
	CFLAGS := -Wall -O2 -march=native
else ifneq ($(shell which pgc 2> /dev/null),)
	CC := pgc
else ifneq ($(shell which pgcpp 2> /dev/null),)
	CC := pgcpp
else ifneq ($(shell which cc 2> /dev/null),)
	CC := cc
else ifneq ($(shell which CC 2> /dev/null),)
	CC := CC
else ifneq ($(CXX),)
	CC := $(CXX)
endif

ifeq ($(CFLAGS),)
	CFLAGS := $(CXXFLAGS)
endif
ifeq ($(CFLAGS_MIC),)
	CFLAGS_MIC := $(CFLAGS)
endif

SRCFILES = $(patsubst %,mm_%.c,$(INDICES))
OBJFILES_HST = $(patsubst %,$(OBJDIR)/intel64/mm_%.o,$(INDICES))
OBJFILES_MIC = $(patsubst %,$(OBJDIR)/mic/mm_%.o,$(INDICES))

LIB_HST ?= $(LIBDIR)/intel64/libxsmm.a
LIB_MIC ?= $(LIBDIR)/mic/libxsmm.a
HEADER = $(INCDIR)/libxsmm.h
MAIN = $(SRCDIR)/libxsmm.c


lib_all: lib_hst lib_mic

header_mic: $(HEADER)
$(HEADER): $(INCDIR)/libxsmm.0 $(INCDIR)/libxsmm.1 $(INCDIR)/libxsmm.2
	@cat $(INCDIR)/libxsmm.0 > $@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) $(THRESHOLD) $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) >> $@
	@echo >> $@
	@cat $(INCDIR)/libxsmm.1 >> $@
	@echo >> $@
	@python $(SCRDIR)/libxsmm_interface.py $(ROW_MAJOR) $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) >> $@
	@cat $(INCDIR)/libxsmm.2 >> $@

source_mic: $(addprefix $(SRCDIR)/,$(SRCFILES))
$(SRCDIR)/%.c: $(HEADER)
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) -1 `echo $* | awk -F_ '{ print $$2" "$$3" "$$4 }'` > $@

main: $(MAIN)
$(MAIN): $(HEADER)
	@python $(SCRDIR)/libxsmm_dispatch.py $(words $(INDICES_M)) $(words $(INDICES_N)) $(INDICES_M) $(INDICES_N) $(INDICES_K) > $@

compile_mic: $(OBJFILES_MIC)
$(OBJDIR)/mic/%.o: $(SRCDIR)/%.c $(SRCDIR)/libxsmm_isa.h $(HEADER)
	@mkdir -p $(OBJDIR)/mic
	$(CC) $(CFLAGS_MIC) -I$(INCDIR) -c $< -o $@

compile_hst: $(OBJFILES_HST)
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.c $(SRCDIR)/libxsmm_isa.h $(HEADER)
	@mkdir -p $(OBJDIR)/intel64
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

lib_mic: $(LIB_MIC)
ifeq ($(origin NO_MAIN), undefined)
$(LIB_MIC): $(OBJFILES_MIC) $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/mic/%.o,$(MAIN))
else
$(LIB_MIC): $(OBJFILES_MIC)
endif
	@mkdir -p $(LIBDIR)/mic
	$(AR) -rs $@ $^

lib_hst: $(LIB_HST)
ifeq ($(origin NO_MAIN), undefined)
$(LIB_HST): $(OBJFILES_HST) $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/intel64/%.o,$(MAIN))
else
$(LIB_HST): $(OBJFILES_HST)
endif
	@mkdir -p $(LIBDIR)/intel64
	$(AR) -rs $@ $^

clean:
	rm -rf $(ROOTDIR)/*~ $(ROOTDIR)/*/*~ $(SRCDIR)/*.c $(OBJDIR)

realclean: clean
	rm -rf $(LIBDIR) $(HEADER)

install: lib_all clean
