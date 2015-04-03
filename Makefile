
# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise 
ROW_MAJOR ?= 1

# M, N, K sets generate value combinations according to the loop nest M(N(K)))
# with an empty set 
M ?= $(shell seq 1 5)

# limit to certain code path(s)
SSE ?= 0
AVX ?= 0

# Use assembly kernel generator
GENASM ?= 0

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Use aligned Store and/or aligned Load instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

# THRESHOLD problem size (M x N x K); determines when to use BLAS 
THRESHOLD ?= $(shell echo $$((24 * 24 * 24)))

# SPARSITY = (LIBXSMM_MAX_M * LIBXSMM_MAX_M * LIBXSMM_MAX_M) / LIBXSMM_MAX_MNK
# Use binary search in auto-dispatch when SPARSITY exceeds the given value.
# With SPARSITY < 1, the binary search is enabled by default (no threshold).
SPARSITY ?= 2

ROOTDIR ?= .
SCRDIR = $(ROOTDIR)/scripts
OBJDIR = $(ROOTDIR)/build
INCDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
LIBDIR = $(ROOTDIR)/lib

LIB_HST ?= $(LIBDIR)/intel64/libxsmm.a
LIB_MIC ?= $(LIBDIR)/mic/libxsmm.a
HEADER = $(INCDIR)/libxsmm.h
MAIN = $(SRCDIR)/libxsmm.c

# prefer the Intel compiler
ifneq ($(shell which icc 2> /dev/null),)
	CC := icc
	AR := xiar
	FLAGS := -Wall -fPIC -fno-alias -ansi-alias -mkl=sequential -DNDEBUG
	CFLAGS := $(FLAGS) -std=c99 -O3 -ipo -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
	CFLMIC := $(FLAGS) -std=c99 -O2 -ipo -mmic -opt-assume-safe-padding
	ifneq ($(shell which icpc 2> /dev/null),)
		CXX := icpc
		CXXFLAGS := $(FLAGS) -O3 -ipo -offload-option,mic,compiler,"-O2 -opt-assume-safe-padding"
		CXXFLMIC := $(FLAGS) -O2 -ipo -mmic -opt-assume-safe-padding
	endif
	ifeq ($(AVX),1)
		CFLAGS += -xAVX
		CXXFLAGS += -xAVX
	else ifeq ($(AVX),2)
		CFLAGS += -xCORE-AVX2
		CXXFLAGS += -xCORE-AVX2
	else ifeq ($(AVX),3)
		CFLAGS += -xCOMMON-AVX512
		CXXFLAGS += -xCOMMON-AVX512
	else ifneq ($(SSE),0)
		CFLAGS += -xSSE3
		CXXFLAGS += -xSSE3
	else
		CFLAGS += -xHost
		CXXFLAGS += -xHost
	endif
else ifneq ($(shell which gcc 2> /dev/null),)
	CC := gcc
	FLAGS := -Wall -O2 -DNDEBUG
	CFLAGS := $(FLAGS) -std=c99
	ifneq ($(shell which g++ 2> /dev/null),)
		CXX := g++
		CXXFLAGS := $(FLAGS) 
	endif
	ifeq ($(AVX),1)
		CFLAGS += -mavx
		CXXFLAGS += -mavx
	else ifeq ($(AVX),2)
		CFLAGS += -mavx2
		CXXFLAGS += -mavx2
	else ifeq ($(AVX),3)
		CFLAGS += -mavx512f
		CXXFLAGS += -mavx512f
	else ifneq ($(SSE),0)
		CFLAGS += -xSSE3
		CXXFLAGS += -xSSE3
	else
		CFLAGS += -march=native
		CXXFLAGS += -march=native
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
ifeq ($(CFLMIC),)
	CFLMIC := $(CFLAGS)
endif
ifeq ($(CXXFLAGS),)
	CXXFLAGS := $(CFLAGS)
endif
ifeq ($(CXXFLMIC),)
	CXXFLMIC := $(CXXFLAGS)
endif

MKL_DIRECT := 0
ifneq ($(MKL_DIRECT),0)
	CFLAGS := -DMKL_DIRECT_CALL_SEQ
	CXXFLAGS := -DMKL_DIRECT_CALL_SEQ
	ifneq ($(MKL_DIRECT),1)
		CFLMIC := -DMKL_DIRECT_CALL_SEQ
		CXXFLMIC := -DMKL_DIRECT_CALL_SEQ
	endif
endif

ifeq ($(AVX),1)
	GENTARGET := snb
else ifeq ($(AVX),2)
	GENTARGET := hsw
else ifeq ($(AVX),3)
	GENTARGET := knl
else ifneq ($(SSE),0)
	GENTARGET := wsm
else
	GENTARGET := noarch
endif

INDICES ?= $(shell python $(SCRDIR)/libxsmm_utilities.py 0 $(words $(M)) $(words $(N)) $(M) $(N) $(K))
SRCFILES = $(addprefix $(SRCDIR)/,$(patsubst %,mm_%.c,$(INDICES)))
SRCFILES_GEN = $(patsubst %,$(SRCDIR)/%,GeneratorDriver.cpp GeneratorCSC.cpp GeneratorDense.cpp ReaderCSC.cpp)
OBJFILES_GEN = $(patsubst %,$(OBJDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN))))
OBJFILES_HST = $(patsubst %,$(OBJDIR)/intel64/mm_%.o,$(INDICES))
OBJFILES_MIC = $(patsubst %,$(OBJDIR)/mic/mm_%.o,$(INDICES))


lib_all: lib_hst lib_mic

header: $(HEADER)
$(HEADER): $(SRCDIR)/libxsmm.0.h $(SRCDIR)/libxsmm.1.h $(SRCDIR)/libxsmm.2.h
	@cat $(SRCDIR)/libxsmm.0.h > $@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) $(THRESHOLD) $(words $(M)) $(words $(N)) $(M) $(N) $(K) >> $@
	@echo >> $@
	@cat $(SRCDIR)/libxsmm.1.h >> $@
	@echo >> $@
	@python $(SCRDIR)/libxsmm_interface.py $(words $(M)) $(words $(N)) $(M) $(N) $(K) >> $@
	@cat $(SRCDIR)/libxsmm.2.h >> $@

ifneq ($(GENASM),0)
compile_gen: $(SRCFILES_GEN)
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)/intel64
	$(CXX) $(CXXFLAGS) -c $< -o $@

generator: $(OBJFILES_GEN)
$(SCRDIR)/generator: $(OBJFILES_GEN)
	$(CXX) $(OBJFILES_GEN) -o $@
endif

source: $(SRCFILES)
ifeq ($(GENASM),0)
$(SRCDIR)/%.c: $(HEADER)
else
$(SRCDIR)/%.c: $(HEADER) $(SCRDIR)/generator
endif
	$(eval MVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f2))
	$(eval NVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f3))
	$(eval KVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f4))
ifeq ($(GENASM),0)
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) -3 $(MVALUE) $(NVALUE) $(KVALUE) > $@
else
ifneq ($(ALIGNED_STORES),0) # aligned stores
ifneq ($(ROW_MAJOR),0) # row-major
	$(eval DPLDC := $(shell python $(SCRDIR)/libxsmm_utilities.py 8 $(NVALUE) $(ALIGNMENT) $(ALIGNED_STORES)))
	$(eval SPLDC := $(shell python $(SCRDIR)/libxsmm_utilities.py 4 $(NVALUE) $(ALIGNMENT) $(ALIGNED_STORES)))
else # column-major
	$(eval DPLDC := $(shell python $(SCRDIR)/libxsmm_utilities.py 8 $(MVALUE) $(ALIGNMENT) $(ALIGNED_STORES)))
	$(eval SPLDC := $(shell python $(SCRDIR)/libxsmm_utilities.py 4 $(MVALUE) $(ALIGNMENT) $(ALIGNED_STORES)))
endif
else # unaligned stores
ifneq ($(ROW_MAJOR),0) # row-major
	$(eval DPLDC := $(NVALUE))
	$(eval SPLDC := $(NVALUE))
else # column-major
	$(eval DPLDC := $(MVALUE))
	$(eval SPLDC := $(MVALUE))
endif
endif
	@if [[ 0 == $$(($(NVALUE) % 3)) ]]; then echo "#define LIBXSMM_GENTARGET_$(GENTARGET)" >> $@; fi
	@echo >> $@
	@echo >> $@
ifeq ($(GENTARGET),noarch)
	@if [[ 0 == $$(($(NVALUE) % 3)) ]]; then \
		PS4=''; set -x; \
		$(SCRDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(DPLDC) 1 wsm nopf DP; \
		$(SCRDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(SPLDC) 1 wsm nopf SP; \
		$(SCRDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(DPLDC) 1 snb nopf DP; \
		$(SCRDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(SPLDC) 1 snb nopf SP; \
		$(SCRDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(DPLDC) 1 hsw nopf DP; \
		$(SCRDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(SPLDC) 1 hsw nopf SP; \
	fi
else
	@if [[ 0 == $$(($(NVALUE) % 3)) ]]; then \
		PS4=''; set -x; \
		$(SCRDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(DPLDC) 1 $(GENTARGET) nopf DP; \
		$(SCRDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(SPLDC) 1 $(GENTARGET) nopf SP; \
	fi
endif
	@if [[ 30 -ge $(NVALUE) ]]; then \
		PS4=''; set -x; \
		$(SCRDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knc $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(DPLDC) 1 knc nopf DP; \
		$(SCRDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knc $(MVALUE) $(NVALUE) $(KVALUE) $(MVALUE) $(NVALUE) $(SPLDC) 1 knc nopf SP; \
	fi
	@sed -i \
		-e '1i#include <libxsmm.h>' \
		-e 's/void libxsmm_/LIBXSMM_INLINE LIBXSMM_TARGET(mic) void libxsmm_/' \
		-e 's/double\* A, double\* B, double\* C/const double\* A, const double\* B, double\* C/' \
		-e 's/float\* A, float\* B, float\* C/const float\* A, const float\* B, float\* C/' \
		-e 's/#ifndef NDEBUG/#ifdef LIBXSMM_NEVER_DEFINED/' \
		-e '/#pragma message ("KERNEL COMPILATION ERROR in: " __FILE__)/d' \
		-e '/#error No kernel was compiled, lacking support for current architecture?/d' \
		-e '/#pragma message ("KERNEL COMPILATION WARNING: compiling .\+ code on .\+ or newer architecture: " __FILE__)/d' \
		$@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(ALIGNED_STORES) $(ALIGNED_LOADS) $(ALIGNMENT) 0 $(MVALUE) $(NVALUE) $(KVALUE) >> $@
endif

main: $(MAIN)
$(MAIN): $(HEADER)
	@python $(SCRDIR)/libxsmm_dispatch.py $(THRESHOLD) $(SPARSITY) $(words $(M)) $(words $(N)) $(M) $(N) $(K) > $@

compile_mic: $(OBJFILES_MIC)
$(OBJDIR)/mic/%.o: $(SRCDIR)/%.c $(HEADER) $(SRCDIR)/libxsmm_isa.h
	@mkdir -p $(OBJDIR)/mic
	$(CC) $(CFLMIC) -I$(INCDIR) -c $< -o $@
$(OBJDIR)/mic/%.o: $(SRCDIR)/%.cpp $(HEADER) $(SRCDIR)/libxsmm_isa.h
	@mkdir -p $(OBJDIR)/mic
	$(CXX) $(CXXFLMIC) -I$(INCDIR) -c $< -o $@

compile_hst: $(OBJFILES_HST)
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.c $(HEADER) $(SRCDIR)/libxsmm_isa.h
	@mkdir -p $(OBJDIR)/intel64
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@
$(OBJDIR)/intel64/%.o: $(SRCDIR)/%.cpp $(HEADER) $(SRCDIR)/libxsmm_isa.h
	@mkdir -p $(OBJDIR)/intel64
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
	rm -rf $(ROOTDIR)/*~ $(ROOTDIR)/*/*~ $(OBJDIR) $(SCRDIR)/generator $(SCRDIR)/generator.exe $(SRCDIR)/mm_*_*_*.c $(MAIN)

realclean: clean
	rm -rf $(LIBDIR) $(HEADER)

install: lib_all clean
