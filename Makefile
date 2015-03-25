
# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise 
ROW_MAJOR ?= 1

# M, N, K sets generate value combinations according to the loop nest M(N(K)))
# with an empty set 
M ?= $(shell seq 1 5)

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

INDICES ?= $(shell python $(SCRDIR)/libxsmm_utilities.py $(words $(M)) $(words $(N)) $(M) $(N) $(K))

# prefer the Intel compiler
ifneq ($(shell which icc 2> /dev/null),)
	CC := icc
	AR := xiar
	FLAGS := -Wall -fPIC -fno-alias -ansi-alias -mkl=sequential -DNDEBUG
	ifeq ($(AVX),1)
		FLAGS += -xAVX
	else ifeq ($(AVX),2)
		FLAGS += -xCORE-AVX2
	else ifeq ($(AVX),3)
		FLAGS += -xCOMMON-AVX512
	else
		FLAGS += -xHost
	endif
	CFLAGS := $(FLAGS) -std=c99 -O3 -ipo -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
	CFLAGS_MIC := $(FLAGS) -std=c99 -O2 -ipo -mmic -opt-assume-safe-padding
	ifneq ($(shell which icpc 2> /dev/null),)
		CXX := icpc
		CXXFLAGS := $(FLAGS) -O3 -ipo -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
		CXXFLAGS_MIC := $(FLAGS) -O2 -ipo -mmic -opt-assume-safe-padding
	endif
#else ifneq ($(shell which icl 2> /dev/null),)
#	CC := icl
#	AR := xilib
#else ifneq ($(shell which cl 2> /dev/null),)
#	CC := cl
else ifneq ($(shell which gcc 2> /dev/null),)
	CC := gcc
	FLAGS := -Wall -O3 -DNDEBUG
	ifeq ($(AVX),1)
		FLAGS += -mavx
	else ifeq ($(AVX),2)
		FLAGS += -mavx2
	else ifeq ($(AVX),3)
		FLAGS += -mavx512f
	else
		FLAGS += -march=native
	endif
	CFLAGS := $(FLAGS) -std=c99
	ifneq ($(shell which g++ 2> /dev/null),)
		CXX := g++
		CXXFLAGS := $(FLAGS) 
	endif
endif

ifeq ($(CXX),)
	CXX := $(CC)
endif
ifeq ($(CC),)
	CC := $(CXX)
endif
ifeq ($(CFLAGS),)
	CFLAGS := $(CXXFLAGS)
endif
ifeq ($(CFLAGS_MIC),)
	CFLAGS_MIC := $(CFLAGS)
endif
ifeq ($(CXXFLAGS),)
	CXXFLAGS := $(CFLAGS)
endif
ifeq ($(CXXFLAGS_MIC),)
	CXXFLAGS_MIC := $(CXXFLAGS)
endif

MKL_DIRECT := 0
ifneq ($(MKL_DIRECT),0)
	CFLAGS := -DMKL_DIRECT_CALL_SEQ
	ifneq ($(MKL_DIRECT),1)
		CFLAGS_MIC := -DMKL_DIRECT_CALL_SEQ
	endif
endif

SRCFILES = $(patsubst %,mm_%.c,$(INDICES))
OBJFILES_HST = $(patsubst %,$(OBJDIR)/intel64/mm_%.o,$(INDICES))
OBJFILES_MIC = $(patsubst %,$(OBJDIR)/mic/mm_%.o,$(INDICES))

LIB_HST ?= $(LIBDIR)/intel64/libxsmm.a
LIB_MIC ?= $(LIBDIR)/mic/libxsmm.a
HEADER = $(INCDIR)/libxsmm.h
MAIN = $(SRCDIR)/libxsmm.cpp


lib_all: lib_hst lib_mic

header: $(HEADER)
$(HEADER): $(INCDIR)/libxsmm.0 $(INCDIR)/libxsmm.1 $(INCDIR)/libxsmm.2
	@cat $(INCDIR)/libxsmm.0 > $@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) $(THRESHOLD) $(words $(M)) $(words $(N)) $(M) $(N) $(K) >> $@
	@echo >> $@
	@cat $(INCDIR)/libxsmm.1 >> $@
	@echo >> $@
	@python $(SCRDIR)/libxsmm_interface.py $(words $(M)) $(words $(N)) $(M) $(N) $(K) >> $@
	@cat $(INCDIR)/libxsmm.2 >> $@

source: $(addprefix $(SRCDIR)/,$(SRCFILES))
$(SRCDIR)/%.c: $(HEADER)
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) -1 `echo $* | awk -F_ '{ print $$2" "$$3" "$$4 }'` > $@

main: $(MAIN)
$(MAIN): $(HEADER)
	@python $(SCRDIR)/libxsmm_dispatch.py $@ $(words $(M)) $(words $(N)) $(M) $(N) $(K) > $@

compile_mic: $(OBJFILES_MIC)
$(OBJDIR)/mic/%.o: $(SRCDIR)/%.c $(SRCDIR)/libxsmm_isa.h $(HEADER)
	@mkdir -p $(OBJDIR)/mic
	$(CC) $(CFLAGS_MIC) -I$(INCDIR) -c $< -o $@
$(OBJDIR)/mic/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/libxsmm_isa.h $(HEADER)
	$(CXX) $(CXXFLAGS_MIC) -I$(INCDIR) -c $< -o $@

compile_hst: $(OBJFILES_HST)
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.c $(SRCDIR)/libxsmm_isa.h $(HEADER)
	@mkdir -p $(OBJDIR)/intel64
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/libxsmm_isa.h $(HEADER)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

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
$(LIB_HST): $(OBJFILES_HST) $(patsubst $(SRCDIR)/%,$(OBJDIR)/intel64/%.o,$(basename $(MAIN)))
else
$(LIB_HST): $(OBJFILES_HST)
endif
	@mkdir -p $(LIBDIR)/intel64
	$(AR) -rs $@ $^

clean:
	rm -rf $(ROOTDIR)/*~ $(ROOTDIR)/*/*~ $(SRCDIR)/*.c $(SRCDIR)/*.cpp $(OBJDIR)

realclean: clean
	rm -rf $(LIBDIR) $(HEADER)

install: lib_all clean
