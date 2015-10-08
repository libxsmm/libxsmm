# export all variables to sub-make processes
#.EXPORT_ALL_VARIABLES: #export
# Set MAKE_PARALLEL=0 for issues with parallel make (older GNU Make)
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif

# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise
ROW_MAJOR ?= 0

# Generates M,N,K-combinations for each comma separated group e.g., "1, 2, 3" gnerates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified spearately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# limit to certain code path(s)
SSE ?= 0
AVX ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Use aligned Store and/or aligned Load instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

# Generate prefetches
PREFETCH ?= 0

# THRESHOLD problem size (M x N x K) determining when to use BLAS; can be zero
THRESHOLD ?= $(shell echo $$((60 * 60 * 60)))

# Beta paramater of DGEMM
# we currently support 0 and 1
# 1 generates C += A * B
# 0 generates C = A * B
BETA ?= 1
ifneq (0,$(BETA))
ifneq (1,$(BETA))
$(error BETA needs to be eiter 0 or 1)
endif
endif


ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SPLDIR = $(ROOTDIR)/samples
SCRDIR = $(ROOTDIR)/scripts
SRCDIR = $(ROOTDIR)/src
INCDIR = include
BLDDIR = build
OUTDIR = lib
BINDIR = bin
DOCDIR = documentation

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(ROOTDIR)/include -I$(INCDIR) -I$(SRCDIR)

STATIC ?= 1
OMP ?= 0
SYM ?= 0
DBG ?= 0
IPO ?= 0

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
	MIC ?= 1
else
	MIC ?= 0
endif

ICPC    = $(notdir $(shell which icpc     2> /dev/null))
ICC     = $(notdir $(shell which icc      2> /dev/null))
IFORT   = $(notdir $(shell which ifort    2> /dev/null))
GPP     = $(notdir $(shell which g++      2> /dev/null))
GCC     = $(notdir $(shell which gcc      2> /dev/null))
GFC     = $(notdir $(shell which gfortran 2> /dev/null))

ifneq (,$(ICPC))
	CXX = $(ICPC)
	ifeq (,$(ICC))
		CC = $(CXX)
	endif
	AR = xiar
else
	CXX = $(GPP)
endif
ifneq (,$(ICC))
	CC = $(ICC)
	ifeq (,$(ICPC))
		CXX = $(CC)
	endif
	AR = xiar
else
	CC = $(GCC)
endif
ifneq (,$(IFORT))
	FC = $(IFORT)
else
	FC = $(GFC)
endif
ifneq (,$(CXX))
	LD = $(CXX)
endif
ifeq (,$(LD))
	LD = $(CC)
endif
ifeq (,$(LD))
	LD = $(FC)
endif

ifneq (,$(filter icpc icc ifort,$(CXX) $(CC) $(FC)))
	CXXFLAGS += -fPIC -Wall -std=c++0x
	CFLAGS += -fPIC -Wall -std=c89
	FCMTFLAGS += -threads
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O2
		CFLAGS += -fno-alias -ansi-alias -O2
		FCFLAGS += -O2
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -ipo
			CFLAGS += -ipo
			FCFLAGS += -ipo
		endif
		ifeq (1,$(AVX))
			TARGET = -xAVX
		else ifeq (2,$(AVX))
			TARGET = -xCORE-AVX2
		else ifeq (3,$(AVX))
			TARGET = -xCOMMON-AVX512
		else
			TARGET = -xHost
		endif
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifneq (0,$(SYM))
		ifneq (1,$(SYM))
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		else
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		endif
	endif
	ifneq (0,$(OMP))
		CXXFLAGS += -openmp
		CFLAGS += -openmp
		FCFLAGS += -openmp
		LDFLAGS += -openmp
	endif
	ifeq (0,$(OFFLOAD))
		CXXFLAGS += -no-offload
		CFLAGS += -no-offload
		FCFLAGS += -no-offload
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -no-intel-extensions -static-intel
	endif
	FCMODDIRFLAG = -module
else # GCC assumed
	VERSION = $(shell $(GCC) --version | grep "gcc (GCC)" | sed "s/gcc (GCC) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*$$/\1/")
	VERSION_MAJOR = $(shell echo "$(VERSION)" | cut -d"." -f1)
	VERSION_MINOR = $(shell echo "$(VERSION)" | cut -d"." -f2)
	VERSION_PATCH = $(shell echo "$(VERSION)" | cut -d"." -f3)
	MIC = 0
	CXXFLAGS += -Wall -std=c++0x -Wno-unused-function
	CFLAGS += -Wall -Wno-unused-function
	ifneq (Windows_NT,$(OS))
		CXXFLAGS += -fPIC
		CFLAGS += -fPIC
		FCFLAGS += -fPIC
		LDFLAGS += -fPIC
	endif
	ifeq (0,$(DBG))
		CXXFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		CFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		FCFLAGS += -O2 -ftree-vectorize -ffast-math -funroll-loops
		DFLAGS += -DNDEBUG
		ifneq (0,$(IPO))
			CXXFLAGS += -flto -ffat-lto-objects
			CFLAGS += -flto -ffat-lto-objects
			FCFLAGS += -flto -ffat-lto-objects
			LDFLAGS += -flto
		endif
		ifeq (1,$(AVX))
			TARGET = -mavx
		else ifeq (2,$(AVX))
			TARGET = -mavx2
		else ifeq (3,$(AVX))
			TARGET = -mavx512f
		else
			TARGET = -march=native
		endif
	else
		CXXFLAGS += -O0
		CFLAGS += -O0
		FCFLAGS += -O0
		SYM = $(DBG)
	endif
	ifneq (0,$(SYM))
		ifneq (1,$(SYM))
			CXXFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CXXFLAGS)
			CFLAGS := -g3 -gdwarf-2 -debug inline-debug-info $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		else
			CXXFLAGS := -g $(CXXFLAGS)
			CFLAGS := -g $(CFLAGS)
			FCFLAGS := -g $(FCFLAGS)
		endif
	endif
	ifneq (0,$(OMP))
		CXXFLAGS += -fopenmp
		CFLAGS += -fopenmp
		FCFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
	ifneq (0,$(STATIC))
		SLDFLAGS += -static
	endif
	FCMODDIRFLAG = -J
endif

ifeq (,$(CXXFLAGS))
	CXXFLAGS = $(CFLAGS)
endif
ifeq (,$(CFLAGS))
	CFLAGS = $(CXXFLAGS)
endif
ifeq (,$(FCFLAGS))
	FCFLAGS = $(CFLAGS)
endif
ifeq (,$(LDFLAGS))
	LDFLAGS = $(CFLAGS)
endif

ifneq (0,$(STATIC))
	LIBEXT = a
else
	LIBEXT = so
endif

ifeq (1,$(AVX))
	GENTARGET = snb
else ifeq (2,$(AVX))
	GENTARGET = hsw
else ifeq (3,$(AVX))
	GENTARGET = knl
else ifneq (0,$(SSE))
	GENTARGET = wsm
else
	GENTARGET = noarch
endif

parent = $(subst ?, ,$(firstword $(subst /, ,$(subst $(NULL) ,?,$(patsubst ./%,%,$1)))))

ifneq ("$(M)$(N)$(K)","")
	INDICES ?= $(shell python $(SCRDIR)/libxsmm_utilities.py -2 $(THRESHOLD) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
else
	INDICES ?= $(shell python $(SCRDIR)/libxsmm_utilities.py -1 $(THRESHOLD) '$(MNK)')
endif
NINDICES = $(words $(INDICES))

SRCFILES = $(addprefix $(BLDDIR)/,$(patsubst %,mm_%.c,$(INDICES)))
SRCFILES_GEN_LIB = $(patsubst %,$(SRCDIR)/%,generator_common.c, generator_dense.c, generator_dense_common.c, generator_dense_instructions.c \
                                            generator_dense_sse3_avx_avx2.c, generator_dense_sse3_microkernel.c, generator_dense_avx_microkernel.c, generator_dense_avx2_microkernel.c, \
                                            generator_dense_avx512_microkernel.c, generator_dense_imci_avx512.c, generator_dense_imci_microkernel.c, generator_dense_noarch.c, \
                                            generator_sparse.c, generator_sparse_csc_reader.c, generator_sparse_bsparse.c, generator_sparse_asparse.c)
SRCFILES_GEN_BIN = $(patsubst %,$(SRCDIR)/%,generator_driver.c)
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_GEN_BIN = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_BIN))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES))
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES))

.PHONY: lib_all
ifeq (0,$(OFFLOAD))
ifeq (0,$(MIC))
lib_all: fheader drytest lib_hst
else
lib_all: fheader drytest lib_hst lib_mic
endif
else
ifeq (0,$(MIC))
lib_all: fheader drytest lib_hst
else
lib_all: fheader drytest lib_hst lib_mic
endif
endif

.PHONY: all
all: lib_all samples

.PHONY: install
install: all clean

.PHONY: header
header: cheader fheader

ALIGNED_ST = $(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT))))
ALIGNED_LD = $(shell echo $$((1!=$(ALIGNED_LOADS)?$(ALIGNED_LOADS):$(ALIGNMENT))))

ifeq (1,$(PREFETCH)) # AL2_BL2viaC
	PREFETCH_ID = 6
else ifeq (pfsigonly,$(PREFETCH))
	PREFETCH_ID = 2
else ifeq (BL2viaC,$(PREFETCH))
	PREFETCH_ID = 3
else ifeq (AL2,$(PREFETCH))
	PREFETCH_ID = 4
else ifeq (curAL2,$(PREFETCH))
	PREFETCH_ID = 5
else ifeq (AL2_BL2viaC,$(PREFETCH))
	PREFETCH_ID = 6
else ifeq (curAL2_BL2viaC,$(PREFETCH))
	PREFETCH_ID = 7
else ifeq (AL2jpst,$(PREFETCH))
	PREFETCH_ID = 8
else ifeq (AL2jpst_BL2viaC,$(PREFETCH))
	PREFETCH_ID = 9
else # nopf
	PREFETCH_ID = 0
endif

PREFETCH_A = 0
PREFETCH_B = 0
PREFETCH_C = 0

ifeq (2,$(PREFETCH_ID))
	PREFETCH_SCHEME = pfsigonly
else ifeq (3,$(PREFETCH_ID))
	PREFETCH_SCHEME = BL2viaC
	PREFETCH_B = 1
else ifeq (4,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2
	PREFETCH_A = 1
else ifeq (5,$(PREFETCH_ID))
	PREFETCH_SCHEME = curAL2
	PREFETCH_A = 1
else ifeq (6,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2_BL2viaC
	PREFETCH_A = 1
	PREFETCH_B = 1
else ifeq (7,$(PREFETCH_ID))
	PREFETCH_SCHEME = curAL2_BL2viaC
	PREFETCH_A = 1
	PREFETCH_B = 1
else ifeq (8,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2jpst
	PREFETCH_A = 1
else ifeq (9,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2jpst_BL2viaC
	PREFETCH_A = 1
	PREFETCH_B = 1
else
	PREFETCH_SCHEME = nopf
endif

SUPPRESS_UNUSED_VARIABLE_WARNINGS = LIBXSMM_UNUSED(A); LIBXSMM_UNUSED(B); LIBXSMM_UNUSED(C);
ifneq (nopf,$(PREFETCH_SCHEME))
	SUPPRESS_UNUSED_VARIABLE_WARNINGS += LIBXSMM_UNUSED(A_prefetch); LIBXSMM_UNUSED(B_prefetch); LIBXSMM_UNUSED(C_prefetch);
endif

.PHONY: cheader
cheader: $(INCDIR)/libxsmm.h
$(INCDIR)/libxsmm.h: $(ROOTDIR)/Makefile $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py $(SRCDIR)/libxsmm.template.h $(ROOTDIR)/include/libxsmm_macros.h $(ROOTDIR)/include/libxsmm_prefetch.h $(ROOTDIR)/include/libxsmm_fallback.h
	@mkdir -p $(dir $@)
	@cp $(ROOTDIR)/include/libxsmm_macros.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_prefetch.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_fallback.h $(INCDIR) 2> /dev/null || true
	@python $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.h $(ROW_MAJOR) $(ALIGNMENT) \
		$(ALIGNED_ST) $(ALIGNED_LD) $(PREFETCH) $(PREFETCH_A) $(PREFETCH_B) $(PREFETCH_C) $(BETA) \
		$(OFFLOAD) $(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(INDICES) > $@

.PHONY: fheader
fheader: $(INCDIR)/libxsmm.f90
$(INCDIR)/libxsmm.f90: $(ROOTDIR)/Makefile $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py $(SRCDIR)/libxsmm.template.f90
	@mkdir -p $(dir $@)
	@python $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.f90 $(ROW_MAJOR) $(ALIGNMENT) \
		$(ALIGNED_ST) $(ALIGNED_LD) $(PREFETCH) $(PREFETCH_A) $(PREFETCH_B) $(PREFETCH_C) $(BETA) \
		$(OFFLOAD) $(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(INDICES) > $@
ifeq (0,$(OFFLOAD))
	@sed -i '/ATTRIBUTES OFFLOAD:MIC/d' $@
endif


.PHONY: compile_generator_lib
compile_generator_lib: $(SRCFILES_GEN_LIB)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@
.PHONY: build_generator_lib
build_generator_lib: $(OUTDIR)/intel64/libxsmmgen.$(LIBEXT)
$(OUTDIR)/intel64/libxsmmgen.$(LIBEXT): $(OBJFILES_GEN_LIB)
	@mkdir -p $(dir $@)
ifeq (0,$(STATIC))
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif
.PHONY: compile_generator
compile_generator: $(SRCFILES_GEN_BIN)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@
.PHONY: generator
generator: $(BINDIR)/generator
$(BINDIR)/generator: $(OBJFILES_GEN_BIN) $(OUTDIR)/intel64/libxsmmgen.$(LIBEXT) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) $(CLDFLAGS) $(OBJFILES_GEN_BIN) -L$(OUTDIR)/intel64/ -lxsmmgen -o $@

.PHONY: sources
sources: $(SRCFILES)
$(BLDDIR)/%.c: $(INCDIR)/libxsmm.h $(BINDIR)/generator $(SCRDIR)/libxsmm_utilities.py $(SCRDIR)/libxsmm_impl_mm.py
	$(eval MVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f2))
	$(eval NVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f3))
	$(eval KVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f4))
ifneq (0,$(ROW_MAJOR)) # row-major
	$(eval MVALUE2 := $(NVALUE))
	$(eval NVALUE2 := $(MVALUE))
else # column-major
	$(eval MVALUE2 := $(MVALUE))
	$(eval NVALUE2 := $(NVALUE))
endif
ifneq (0,$(ALIGNED_STORES)) # aligned stores
	$(eval LDCDP := $(shell python $(SCRDIR)/libxsmm_utilities.py 8 $(MVALUE2) $(ALIGNED_ST)))
	$(eval LDCSP := $(shell python $(SCRDIR)/libxsmm_utilities.py 4 $(MVALUE2) $(ALIGNED_ST)))
else # unaligned stores
	$(eval LDCDP := $(MVALUE2))
	$(eval LDCSP := $(MVALUE2))
endif
	$(eval LDA := $(MVALUE2))
	$(eval LDB := $(KVALUE))
	@mkdir -p $(dir $@)
	@echo "#include <libxsmm.h>" > $@
	@echo >> $@
ifneq (0,$(MIC))
	@echo "#define LIBXSMM_GENTARGET_knc_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_knc_sp" >> $@
endif
ifeq (noarch,$(GENTARGET))
	@echo "#define LIBXSMM_GENTARGET_knl_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_knl_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_sp" >> $@
	@echo >> $@
	@echo >> $@
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knl $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 $(BETA) 0 $(ALIGNED_ST) knl $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knl $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 $(BETA) 0 $(ALIGNED_ST) knl $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 $(BETA) 0 $(ALIGNED_ST) hsw $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 $(BETA) 0 $(ALIGNED_ST) hsw $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 $(BETA) 0 $(ALIGNED_ST) snb $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 $(BETA) 0 $(ALIGNED_ST) snb $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 $(BETA) 0 $(ALIGNED_ST) wsm $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 $(BETA) 0 $(ALIGNED_ST) wsm $(PREFETCH_SCHEME) SP
else
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_sp" >> $@
	@echo >> $@
	@echo >> $@
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 $(BETA) 0 $(ALIGNED_ST) $(GENTARGET) $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 $(BETA) 0 $(ALIGNED_ST) $(GENTARGET) $(PREFETCH_SCHEME) SP
endif
ifneq (0,$(MIC))
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knc $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 $(BETA) 0 $(ALIGNED_ST) knc $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knc $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 $(BETA) 0 $(ALIGNED_ST) knc $(PREFETCH_SCHEME) SP
endif
	@sed -i'' \
		-e 's/void libxsmm_/LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_/' \
		-e 's/#ifndef NDEBUG/#ifdef LIBXSMM_NEVER_DEFINED/' \
		-e 's/#pragma message (".*KERNEL COMPILATION ERROR in: " __FILE__)/  $(SUPPRESS_UNUSED_VARIABLE_WARNINGS)/' \
		-e '/#error No kernel was compiled, lacking support for current architecture?/d' \
		-e '/#pragma message (".*KERNEL COMPILATION WARNING: compiling .\+ code on .\+ or newer architecture: " __FILE__)/d' \
		$@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(MVALUE) $(NVALUE) $(KVALUE) >> $@

.PHONY: main
main: $(BLDDIR)/libxsmm.c
$(BLDDIR)/libxsmm.c: $(INCDIR)/libxsmm.h $(SCRDIR)/libxsmm_dispatch.py
	@mkdir -p $(dir $@)
	@python $(SCRDIR)/libxsmm_dispatch.py $(THRESHOLD) $(INDICES) > $@

ifneq (0,$(MIC))
.PHONY: compile_mic
compile_mic: $(OBJFILES_MIC)
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
$(BLDDIR)/mic/%.o: $(SRCDIR)/%.c $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif

.PHONY: compile_hst
compile_hst: $(OBJFILES_HST)
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@
$(BLDDIR)/intel64/%.o: $(SRCDIR)/%.c $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

ifneq (0,$(MIC))
.PHONY: lib_mic
lib_mic: $(OUTDIR)/mic/libxsmm.$(LIBEXT)
ifeq (undefined,$(origin NO_MAIN))
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OBJFILES_MIC) $(BLDDIR)/mic/libxsmm_crc32.o $(BLDDIR)/mic/libxsmm_dispatch.o $(BLDDIR)/mic/libxsmm.o
else
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OBJFILES_MIC) $(BLDDIR)/mic/libxsmm_crc32.o $(BLDDIR)/mic/libxsmm_dispatch.o
endif
	@mkdir -p $(dir $@)
ifeq (0,$(STATIC))
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif
endif

.PHONY: lib_hst
lib_hst: $(OUTDIR)/intel64/libxsmm.$(LIBEXT)
ifeq (undefined,$(origin NO_MAIN))
$(OUTDIR)/intel64/libxsmm.$(LIBEXT): $(OBJFILES_HST) $(BLDDIR)/intel64/libxsmm_crc32.o $(BLDDIR)/intel64/libxsmm_dispatch.o $(BLDDIR)/intel64/libxsmm.o
else
$(OUTDIR)/intel64/libxsmm.$(LIBEXT): $(OBJFILES_HST) $(BLDDIR)/intel64/libxsmm_crc32.o $(BLDDIR)/intel64/libxsmm_dispatch.o
endif
	@mkdir -p $(dir $@)
ifeq (0,$(STATIC))
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif

.PHONY: samples
samples: smm cp2k

.PHONY: smm
smm: lib_all
	@cd $(SPLDIR)/smm && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO)

.PHONY: smm_hst
smm_hst: lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) OFFLOAD=$(OFFLOAD)
.PHONY: smm_mic
smm_mic: lib_mic
	@cd $(SPLDIR)/smm && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=$(MIC)

.PHONY: cp2k
cp2k: lib_all
	@cd $(SPLDIR)/cp2k && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO)
.PHONY: cp2k_hst
cp2k_hst: lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) OFFLOAD=$(OFFLOAD)
.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@cd $(SPLDIR)/cp2k && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=$(MIC)

.PHONY: drytest
drytest: $(SPLDIR)/cp2k/cp2k-perf.sh
$(SPLDIR)/cp2k/cp2k-perf.sh: $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=cp2k-perf.txt" >> $@
	@echo "RUNS='$(INDICES)'" >> $@
	@echo >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | cut --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | cut --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | cut --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  \$${HERE}/cp2k.sh \$${MVALUE} 0 0 \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

.PHONY: test
test: $(SPLDIR)/cp2k/cp2k-perf.txt
$(SPLDIR)/cp2k/cp2k-perf.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_all
	@cd $(SPLDIR)/cp2k && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) realclean && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO)
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@

$(DOCDIR)/libxsmm.pdf: $(ROOTDIR)/README.md
	@mkdir -p $(dir $@)
	$(eval TEMPLATE := $(shell mktemp --tmpdir=. --suffix=.tex))
	@pandoc -D latex > $(TEMPLATE)
	@sed -i \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
		-e 's/\[\[.\+\](.\+)\]//' \
		-e '/!\[.\+\](.\+)/{n;d}' \
		$(ROOTDIR)/README.md | \
	pandoc \
		--latex-engine=xelatex \
		--template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSMM Documentation" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TEMPLATE)

$(DOCDIR)/cp2k.pdf: $(ROOTDIR)/documentation/cp2k.md
	@mkdir -p $(dir $@)
	$(eval TEMPLATE := $(shell mktemp --tmpdir=. --suffix=.tex))
	@pandoc -D latex > $(TEMPLATE)
	@sed -i \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
		-e 's/\[\[.\+\](.\+)\]//' \
		-e '/!\[.\+\](.\+)/{n;d}' \
		$(ROOTDIR)/documentation/cp2k.md | \
	pandoc \
		--latex-engine=xelatex \
		--template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures \
		-V documentclass=scrartcl \
		-V title-meta="CP2K with LIBXSMM" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TEMPLATE)

.PHONY: documentation
documentation: $(DOCDIR)/libxsmm.pdf $(DOCDIR)/cp2k.pdf

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
else
	@rm -f $(OBJECTS) $(BLDDIR)/libxsmm.c $(BLDDIR)/*.mod
endif
else
	@rm -f $(OBJECTS) $(BLDDIR)/libxsmm.c $(BLDDIR)/*.mod
endif

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
else
	@rm -f $(OUTDIR)/intel64/libxsmm.$(LIBEXT) $(OUTDIR)/mic/libxsmm.$(LIBEXT)
endif
else
	@rm -f $(OUTDIR)/intel64/libxsmm.$(LIBEXT) $(OUTDIR)/mic/libxsmm.$(LIBEXT)
endif
ifneq ($(abspath $(BINDIR)),$(ROOTDIR))
ifneq ($(abspath $(BINDIR)),$(abspath .))
	@rm -rf $(BINDIR)
else
	@rm -f $(BINDIR)/generator
endif
else
	@rm -f $(BINDIR)/generator
endif
	@rm -f $(SCRDIR)/libxsmm_utilities.pyc
	@rm -f $(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(INCDIR)/libxsmm.f90
	@rm -f $(INCDIR)/libxsmm.h

install: all clean

