# export all variables to sub-make processes
.EXPORT_ALL_VARIABLES: #export
.NOTPARALLEL:

# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise 
ROW_MAJOR ?= 1

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

# THRESHOLD problem size (M x N x K) determining when to use BLAS; can be zero
THRESHOLD ?= $(shell echo $$((60 * 60 * 60)))

# SPARSITY = (LIBXSMM_MAX_M * LIBXSMM_MAX_M * LIBXSMM_MAX_M) / LIBXSMM_MAX_MNK
# Use binary search in auto-dispatch when SPARSITY exceeds the given value.
# With SPARSITY < 1, the binary search is enabled by default (no threshold).
SPARSITY ?= 2

ROOTDIR = $(abspath $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SPLDIR = $(ROOTDIR)/samples
SCRDIR = $(ROOTDIR)/scripts
SRCDIR = $(ROOTDIR)/src
INCDIR = include
BLDDIR = build
OUTDIR = lib
BINDIR = bin

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(ROOTDIR)/include -I$(INCDIR)

STATIC ?= 1
OMP ?= 0
SYM ?= 0
DBG ?= 1
IPO ?= 0

OFFLOAD ?= 0
ifneq ($(OFFLOAD),0)
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
ifneq ($(CXX),)
	LD = $(CXX)
endif
ifeq ($(LD),)
	LD = $(CC)
endif
ifeq ($(LD),)
	LD = $(FC)
endif

ifneq (,$(filter icpc icc ifort,$(CXX) $(CC) $(FC)))
	CXXFLAGS += -fPIC -Wall -Werror -std=c++0x
	CFLAGS += -fPIC -Wall -Werror
	FCMTFLAGS += -threads
	FCFLAGS += -fPIC
	LDFLAGS += -fPIC
	ifeq (0,$(DBG))
		CXXFLAGS += -fno-alias -ansi-alias -O2
		CFLAGS += -fno-alias -ansi-alias -O2
		FCFLAGS += -O2
		DFLAGS += -DNDEBUG
		ifneq ($(IPO),0)
			CXXFLAGS += -ipo
			CFLAGS += -ipo
			FCFLAGS += -ipo
		endif
		ifeq ($(AVX),1)
			TARGET = -xAVX
		else ifeq ($(AVX),2)
			TARGET = -xCORE-AVX2
		else ifeq ($(AVX),3)
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
	ifneq ($(OMP),0)
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
	ifneq ($(STATIC),0)
		SLDFLAGS += -no-intel-extensions -static-intel
	endif
	FCMODDIRFLAG = -module
else # GCC assumed
	VERSION = $(shell $(GCC) --version | grep "gcc (GCC)" | sed "s/gcc (GCC) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*$$/\1/")
	VERSION_MAJOR = $(shell echo "$(VERSION)" | cut -d"." -f1)
	VERSION_MINOR = $(shell echo "$(VERSION)" | cut -d"." -f2)
	VERSION_PATCH = $(shell echo "$(VERSION)" | cut -d"." -f3)
	MIC = 0
	CXXFLAGS += -Wall -Werror -std=c++0x -Wno-unused-function
	CFLAGS += -Wall -Werror -Wno-unused-function
	ifneq ($(OS),Windows_NT)
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
		ifneq ($(IPO),0)
			CXXFLAGS += -flto -ffat-lto-objects
			CFLAGS += -flto -ffat-lto-objects
			FCFLAGS += -flto -ffat-lto-objects
			LDFLAGS += -flto
		endif
		ifeq ($(AVX),1)
			TARGET = -mavx
		else ifeq ($(AVX),2)
			TARGET = -mavx2
		else ifeq ($(AVX),3)
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
	ifneq ($(OMP),0)
		CXXFLAGS += -fopenmp
		CFLAGS += -fopenmp
		FCFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
	ifneq ($(STATIC),0)
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

ifneq ($(STATIC),0)
	LIBEXT = a
else
	LIBEXT = so
endif

ifeq ($(AVX),1)
	GENTARGET = snb
else ifeq ($(AVX),2)
	GENTARGET = hsw
else ifeq ($(AVX),3)
	GENTARGET = knl
else ifneq ($(SSE),0)
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
SRCFILES_GEN = $(patsubst %,$(SRCDIR)/%,generator_driver.c, generator_common.c, generator_dense.c, generator_dense_common.c, generator_dense_sse3_avx_avx2.c, generator_dense_instructions.c, generator_dense_sse3_avx_avx2_common.c, generator_dense_sse3_microkernel.c, generator_dense_avx_microkernel.c, generator_dense_avx2_microkernel.c, generator_dense_imci_avx512.c, generator_dense_avx512_microkernel.c, generator_dense_imci_microkernel.c)
OBJFILES_GEN = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES))
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES))

.PHONY: lib_all
ifeq ($(OFFLOAD),0)
ifeq ($(MIC),0)
lib_all: fheader drytest lib_hst
else
lib_all: fheader drytest lib_hst lib_mic
endif
else
ifeq ($(MIC),0)
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

.PHONY: cheader
cheader: $(INCDIR)/libxsmm.h
$(INCDIR)/libxsmm.h: $(ROOTDIR)/Makefile $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py $(SRCDIR)/libxsmm.template.h $(ROOTDIR)/include/libxsmm_macros.h
	@mkdir -p $(dir $@)
	@cp $(ROOTDIR)/include/libxsmm_macros.h $(INCDIR) 2> /dev/null || true
	@python $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.h $(ROW_MAJOR) $(ALIGNMENT) \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) \
		$(shell echo $$((1!=$(ALIGNED_LOADS)?$(ALIGNED_LOADS):$(ALIGNMENT)))) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(INDICES) > $@

.PHONY: fheader
fheader: $(INCDIR)/libxsmm.f90
$(INCDIR)/libxsmm.f90: $(ROOTDIR)/Makefile $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py $(SRCDIR)/libxsmm.template.f90
	@mkdir -p $(dir $@)
	@python $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.f90 $(ROW_MAJOR) $(ALIGNMENT) \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) \
		$(shell echo $$((1!=$(ALIGNED_LOADS)?$(ALIGNED_LOADS):$(ALIGNMENT)))) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(INDICES) > $@

.PHONY: compile_gen
compile_gen: $(SRCFILES_GEN)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) -c $< -o $@
.PHONY: generator
generator: $(OBJFILES_GEN)
$(BINDIR)/generator: $(OBJFILES_GEN) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(OBJFILES_GEN) -o $@
.PHONY: generator_backend
generator_backend: $(BINDIR)/generator

.PHONY: sources
sources: $(SRCFILES)
$(BLDDIR)/%.c: $(INCDIR)/libxsmm.h $(BINDIR)/generator $(SCRDIR)/libxsmm_utilities.py $(SCRDIR)/libxsmm_impl_mm.py
	$(eval MVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f2))
	$(eval NVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f3))
	$(eval KVALUE := $(shell echo $* | cut --output-delimiter=' ' -d_ -f4))
ifneq ($(ROW_MAJOR),0) # row-major
	$(eval MVALUE2 := $(NVALUE))
	$(eval NVALUE2 := $(MVALUE))
else # column-major
	$(eval MVALUE2 := $(MVALUE))
	$(eval NVALUE2 := $(NVALUE))
endif
ifneq ($(ALIGNED_STORES),0) # aligned stores
	$(eval LDCDP := $(shell python $(SCRDIR)/libxsmm_utilities.py 8 $(MVALUE2) $(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT))))))
	$(eval LDCSP := $(shell python $(SCRDIR)/libxsmm_utilities.py 4 $(MVALUE2) $(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT))))))
else # unaligned stores
	$(eval LDCDP := $(MVALUE2))
	$(eval LDCSP := $(MVALUE2))
endif
	$(eval LDA := $(MVALUE2))
	$(eval LDB := $(KVALUE))
	@mkdir -p $(dir $@)
	@echo "#include <libxsmm.h>" > $@
	@echo >> $@
	@echo "#define LIBXSMM_GENTARGET_knc_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_knc_sp" >> $@
ifeq ($(GENTARGET),noarch)
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
	@$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knl $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) knl nopf DP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knl $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) knl nopf SP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) hsw nopf DP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) hsw nopf SP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) snb nopf DP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) snb nopf SP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) wsm nopf DP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) wsm nopf SP > /dev/null
else
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_sp" >> $@
	@echo >> $@
	@echo >> $@
	@$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) $(GENTARGET) nopf DP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) $(GENTARGET) nopf SP > /dev/null
endif
	@$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knc $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCDP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) knc nopf DP > /dev/null
	@$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knc $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDA) $(LDB) $(LDCSP) 1 1 0 \
		$(shell echo $$((1!=$(ALIGNED_STORES)?$(ALIGNED_STORES):$(ALIGNMENT)))) knc nopf SP > /dev/null
	@sed -i'' \
		-e 's/void libxsmm_/LIBXSMM_INLINE LIBXSMM_TARGET(mic) void libxsmm_/' \
		-e 's/#ifndef NDEBUG/#ifdef LIBXSMM_NEVER_DEFINED/' \
		-e '/#pragma message ("KERNEL COMPILATION ERROR in: " __FILE__)/d' \
		-e '/#error No kernel was compiled, lacking support for current architecture?/d' \
		-e '/#pragma message ("KERNEL COMPILATION WARNING: compiling .\+ code on .\+ or newer architecture: " __FILE__)/d' \
		$@
	@python $(SCRDIR)/libxsmm_impl_mm.py $(ROW_MAJOR) $(MVALUE) $(NVALUE) $(KVALUE) >> $@

.PHONY: main
main: $(BLDDIR)/libxsmm.c
$(BLDDIR)/libxsmm.c: $(INCDIR)/libxsmm.h $(SCRDIR)/libxsmm_dispatch.py
	@mkdir -p $(dir $@)
	@python $(SCRDIR)/libxsmm_dispatch.py $(THRESHOLD) $(SPARSITY) $(INDICES) > $@

ifneq ($(MIC),0)
.PHONY: compile_mic
compile_mic: $(OBJFILES_MIC)
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.cpp $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif

.PHONY: compile_hst
compile_hst: $(OBJFILES_HST)
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.cpp $(INCDIR)/libxsmm.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

ifneq ($(MIC),0)
.PHONY: lib_mic
lib_mic: $(OUTDIR)/mic/libxsmm.$(LIBEXT)
ifeq ($(origin NO_MAIN), undefined)
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OBJFILES_MIC) $(BLDDIR)/mic/libxsmm.o
else
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OBJFILES_MIC)
endif
	@mkdir -p $(dir $@)
ifeq ($(STATIC),0)
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif
endif

.PHONY: lib_hst
lib_hst: $(OUTDIR)/intel64/libxsmm.$(LIBEXT)
ifeq ($(origin NO_MAIN),undefined)
$(OUTDIR)/intel64/libxsmm.$(LIBEXT): $(OBJFILES_HST) $(BLDDIR)/intel64/libxsmm.o
else
$(OUTDIR)/intel64/libxsmm.$(LIBEXT): $(OBJFILES_HST)
endif
	@mkdir -p $(dir $@)
ifeq ($(STATIC),0)
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif

.PHONY: samples
samples: smm cp2k

.PHONY: smm
smm: lib_all
	@cd $(SPLDIR)/smm && $(MAKE)
.PHONY: smm_hst
smm_hst: lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) OFFLOAD=$(OFFLOAD)
.PHONY: smm_mic
smm_mic: lib_mic
	@cd $(SPLDIR)/smm && $(MAKE) MIC=$(MIC)

.PHONY: cp2k
cp2k: lib_all
	@cd $(SPLDIR)/cp2k && $(MAKE)
.PHONY: cp2k_hst
cp2k_hst: lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) OFFLOAD=$(OFFLOAD)
.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@cd $(SPLDIR)/cp2k && $(MAKE) MIC=$(MIC)

.PHONY: test
test: $(SPLDIR)/cp2k/cp2k-perf.txt
$(SPLDIR)/cp2k/cp2k-perf.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_all
	@cd $(SPLDIR)/cp2k && $(MAKE) realclean && $(MAKE)
	@$(SPLDIR)/cp2k/cp2k-perf.sh

.PHONY: drytest
drytest: $(SPLDIR)/cp2k/cp2k-perf.sh
$(SPLDIR)/cp2k/cp2k-perf.sh: $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "RUNS='$(INDICES)'" >> $@
	@echo >> $@
	@echo >> $@
	@echo "cat /dev/null > cp2k-perf.txt" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | cut --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | cut --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | cut --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  \$${HERE}/cp2k.sh \$${MVALUE} 0 0 \$${NVALUE} \$${KVALUE} >> cp2k-perf.txt" >> $@
	@echo "  echo >> cp2k-perf.txt" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

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
	@rm -f $(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(INCDIR)/libxsmm.f90
	@rm -f $(INCDIR)/libxsmm.h

install: all clean

