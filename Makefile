# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explcitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(MAKE_PARALLEL))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

# Linux cut has features we use that do not work elsewhere. Mac, etc. users
# should install GNU coreutils and use "cut" from there.
# For example, if you use Homebrew, run "brew install coreutils" once and invoke:
# $ make CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
CUT ?= cut

# Python interpreter
PYTHON ?= python

# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise
ROW_MAJOR ?= 0

# Generates M,N,K-combinations for each comma separated group e.g., "1, 2, 3" gnerates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified separately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# Preferred precision when registering statically generated code versions
# 0: SP and DP code versions to be registered
# 1: SP only
# 2: DP only
PRECISION ?= 0

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Generate prefetches
PREFETCH ?= 0

# THRESHOLD problem size (M x N x K) determining when to use BLAS; can be zero
THRESHOLD ?= $(shell echo $$((80 * 80 * 80)))

# Generate code using aligned Load/Store instructions
# !=0: enable if lda/ldc (m) is a multiple of ALIGNMENT
# ==0: disable emitting aligned Load/Store instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

# Alpha argument of GEMM
# Supported: 1.0
ALPHA ?= 1
ifneq (1,$(ALPHA))
$(error ALPHA needs to be 1)
endif

# Beta argument of GEMM
# Supported: 0.0, 1.0
# 0: C  = A * B
# 1: C += A * B
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

# subdirectories for prefix based installation
PINCDIR = $(INCDIR)
POUTDIR = $(OUTDIR)
PBINDIR = $(BINDIR)
PDOCDIR = share/libxsmm

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -D__extern_always_inline=inline
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(SRCDIR)

STATIC ?= 1
OMP ?= 0
SYM ?= 0
DBG ?= 0

# Request strongest code conformance
PEDANTIC ?= 0

# Embed InterProcedural Optimization information into libraries
IPO ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0
BLAS ?= 0

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
	MIC ?= 1
else
	MIC ?= 0
endif

# JIT backend is enabled by default
JIT ?= 1
ifneq (0,$(JIT))
	SSE ?= 1
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

ifneq (0,$(STATIC))
	GENERATOR = $(BINDIR)/libxsmm_generator
	LIBEXT = a
else
	GENERATOR = env LD_LIBRARY_PATH=$(OUTDIR):$(LD_LIBRARY_PATH) $(BINDIR)/libxsmm_generator
	LIBEXT = so
endif

INDICES ?= $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py -1 $(THRESHOLD) $(words $(MNK)) $(MNK) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
NINDICES = $(words $(INDICES))

SRCFILES = $(patsubst %,$(BLDDIR)/mm_%.c,$(INDICES))
SRCFILES_GEN_LIB = $(patsubst %,$(SRCDIR)/%,$(wildcard $(SRCDIR)/generator_*.c) libxsmm_timer.c)
SRCFILES_GEN_BIN = $(patsubst %,$(SRCDIR)/%,libxsmm_generator_driver.c)
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_GEN_BIN = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_BIN))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES)) $(BLDDIR)/intel64/libxsmm.o $(BLDDIR)/intel64/libxsmm_crc32.o $(BLDDIR)/intel64/libxsmm_dispatch.o
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES)) $(BLDDIR)/mic/libxsmm.o $(BLDDIR)/mic/libxsmm_crc32.o $(BLDDIR)/mic/libxsmm_dispatch.o $(BLDDIR)/mic/libxsmm_timer.o

.PHONY: lib_all
ifeq (0,$(OFFLOAD))
ifeq (0,$(MIC))
lib_all: header drytest lib_hst
else
lib_all: header drytest lib_hst lib_mic
endif
else
ifeq (0,$(MIC))
lib_all: header drytest lib_hst
else
lib_all: header drytest lib_hst lib_mic
endif
endif

.PHONY: all
all: lib_all samples

.PHONY: header
header: cheader fheader

.PHONY: interface
interface: header

PREFETCH_ID = 0
PREFETCH_SCHEME = nopf
PREFETCH_TYPE = 0

ifneq (0,$(shell echo $$((2 <= $(PREFETCH) && $(PREFETCH) <= 9))))
	PREFETCH_ID = $(PREFETCH)
else ifeq (1,$(PREFETCH)) # AL2_BL2viaC
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
endif

# Mapping build options to libxsmm_prefetch_type (see include/libxsmm_typedefs.h)
ifeq (2,$(PREFETCH_ID))
	PREFETCH_SCHEME = pfsigonly
	PREFETCH_TYPE = 1
else ifeq (3,$(PREFETCH_ID))
	PREFETCH_SCHEME = BL2viaC
	PREFETCH_TYPE = 8
else ifeq (4,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2
	PREFETCH_TYPE = 2
else ifeq (5,$(PREFETCH_ID))
	PREFETCH_SCHEME = curAL2
	PREFETCH_TYPE = 16
else ifeq (8,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2jpst
	PREFETCH_TYPE = 4
else ifeq (6,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2_BL2viaC
	PREFETCH_TYPE = $(shell echo $$((8 | 2)))
else ifeq (7,$(PREFETCH_ID))
	PREFETCH_SCHEME = curAL2_BL2viaC
	PREFETCH_TYPE = $(shell echo $$((8 | 16)))
else ifeq (9,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2jpst_BL2viaC
	PREFETCH_TYPE = $(shell echo $$((8 | 4)))
endif

# Mapping build options to libxsmm_gemm_flags (see include/libxsmm_typedefs.h)
FLAGS = $(shell echo $$((((0!=$(ALIGNED_LOADS))*4) | ((0!=$(ALIGNED_STORES))*8))))

SUPPRESS_UNUSED_VARIABLE_WARNINGS = LIBXSMM_UNUSED(A); LIBXSMM_UNUSED(B); LIBXSMM_UNUSED(C);
ifneq (nopf,$(PREFETCH_SCHEME))
	SUPPRESS_UNUSED_VARIABLE_WARNINGS += LIBXSMM_UNUSED(A_prefetch); LIBXSMM_UNUSED(B_prefetch);
	SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXSMM_UNUSED(C_prefetch);\n
endif

.PHONY: cheader
cheader: $(INCDIR)/libxsmm.h
$(INCDIR)/libxsmm.h: $(INCDIR)/.make \
                     $(SRCDIR)/libxsmm.template.h $(ROOTDIR)/.hooks/install.sh $(ROOTDIR)/version.txt \
                     $(ROOTDIR)/include/libxsmm_macros.h $(ROOTDIR)/include/libxsmm_typedefs.h $(ROOTDIR)/include/libxsmm_frontend.h \
                     $(ROOTDIR)/include/libxsmm_generator.h $(ROOTDIR)/include/libxsmm_timer.h \
                     $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py \
                     $(ROOTDIR)/Makefile
	@$(ROOTDIR)/.hooks/install.sh
	@cp -u $(ROOTDIR)/include/libxsmm_macros.h $(INCDIR) 2> /dev/null || true
	@cp -u $(ROOTDIR)/include/libxsmm_typedefs.h $(INCDIR) 2> /dev/null || true
	@cp -u $(ROOTDIR)/include/libxsmm_frontend.h $(INCDIR) 2> /dev/null || true
	@cp -u $(ROOTDIR)/include/libxsmm_generator.h $(INCDIR) 2> /dev/null || true
	@cp -u $(ROOTDIR)/include/libxsmm_timer.h $(INCDIR) 2> /dev/null || true
	@$(PYTHON) $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.h $(PRECISION) $(MAKE_ILP64) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@
ifneq (0,$(OMP))
	$(info ==========================================================)
	$(info LIBXSMM is agnostic with respect to the threading runtime!)
	$(info Enabling OpenMP suppresses using OS primitives.)
	$(info ==========================================================)
endif
ifneq (0,$(BLAS))
	$(info =========================================================)
	$(info LIBXSMM is link-time agnostic with respect to BLAS/GEMM!)
	$(info Linking it now may prevent users to make an own decision.)
ifeq (1,$(BLAS))
	$(info LIBXSMM's THRESHOLD already prevents calling small GEMMs!)
	$(info A sequential BLAS is superfluous with respect to LIBXSMM.)
endif
	$(info =========================================================)
endif

.PHONY: fheader
fheader: $(INCDIR)/libxsmm.f
$(INCDIR)/libxsmm.f: $(INCDIR)/.make $(BLDDIR)/.make \
                     $(SRCDIR)/libxsmm.template.f $(ROOTDIR)/.hooks/install.sh $(ROOTDIR)/version.txt \
                     $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py \
                     $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@$(ROOTDIR)/.hooks/install.sh
	@$(PYTHON) $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.f $(PRECISION) $(MAKE_ILP64) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@
ifeq (0,$(OFFLOAD))
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} '/ATTRIBUTES OFFLOAD:MIC/d' $@
	@rm -f ${TMPFILE} 
endif
ifneq (,$(FC))
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $@ -o $(BLDDIR)/libxsmm-mod.o $(FMFLAGS) $(dir $@)
endif

.PHONY: compile_generator_lib
compile_generator_lib: $(OBJFILES_GEN_LIB)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@
.PHONY: build_generator_lib
build_generator_lib: $(OUTDIR)/libxsmmgen.$(LIBEXT)
$(OUTDIR)/libxsmmgen.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_GEN_LIB)
ifeq (0,$(STATIC))
	$(LD) -o $@ $(OBJFILES_GEN_LIB) -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $(OBJFILES_GEN_LIB)
endif

.PHONY: compile_generator
compile_generator: $(OBJFILES_GEN_BIN)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@
.PHONY: generator
generator: $(BINDIR)/libxsmm_generator
$(BINDIR)/libxsmm_generator: $(BINDIR)/.make $(OBJFILES_GEN_BIN) $(OUTDIR)/libxsmmgen.$(LIBEXT) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	$(CC) $(OBJFILES_GEN_BIN) -L$(OUTDIR) -lxsmmgen $(LDFLAGS) $(CLDFLAGS) -o $@

.PHONY: sources
sources: $(SRCFILES)
$(BLDDIR)/%.c: $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(BINDIR)/libxsmm_generator $(SCRDIR)/libxsmm_utilities.py $(SCRDIR)/libxsmm_specialized.py
ifneq (,$(SRCFILES))
	$(eval MVALUE := $(shell echo $(basename $@) | $(CUT) --output-delimiter=' ' -d_ -f2))
	$(eval NVALUE := $(shell echo $(basename $@) | $(CUT) --output-delimiter=' ' -d_ -f3))
	$(eval KVALUE := $(shell echo $(basename $@) | $(CUT) --output-delimiter=' ' -d_ -f4))
ifneq (0,$(ROW_MAJOR)) # row-major
	$(eval MNVALUE := $(NVALUE))
	$(eval NMVALUE := $(MVALUE))
else # column-major
	$(eval MNVALUE := $(MVALUE))
	$(eval NMVALUE := $(NVALUE))
endif
	$(eval ASTSP := $(shell echo $$((0!=$(ALIGNED_STORES)&&0==($(MNVALUE)*4)%$(ALIGNMENT)))))
	$(eval ASTDP := $(shell echo $$((0!=$(ALIGNED_STORES)&&0==($(MNVALUE)*8)%$(ALIGNMENT)))))
	$(eval ALDSP := $(shell echo $$((0!=$(ALIGNED_LOADS)&&0==($(MNVALUE)*4)%$(ALIGNMENT)))))
	$(eval ALDDP := $(shell echo $$((0!=$(ALIGNED_LOADS)&&0==($(MNVALUE)*8)%$(ALIGNMENT)))))
	@echo "#include <libxsmm.h>" > $@
	@echo >> $@
ifneq (0,$(MIC))
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knc_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knc_dp" >> $@
endif
endif
ifeq (noarch,$(GENTARGET))
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knl_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knl_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) knl $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) hsw $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) snb $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) wsm $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) knl $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) hsw $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) snb $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) wsm $(PREFETCH_SCHEME) DP
endif
else
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) $(GENTARGET) $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) $(GENTARGET) $(PREFETCH_SCHEME) DP
endif
endif
ifneq (0,$(MIC))
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTDP) knc $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTDP) knc $(PREFETCH_SCHEME) DP
endif
endif
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} \
		-e 's/void libxsmm_/LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_/' \
		-e 's/#ifndef NDEBUG/$(SUPPRESS_UNUSED_PREFETCH_WARNINGS)#ifdef LIBXSMM_NEVER_DEFINED/' \
		-e 's/#pragma message (".*KERNEL COMPILATION ERROR in: " __FILE__)/  $(SUPPRESS_UNUSED_VARIABLE_WARNINGS)/' \
		-e '/#error No kernel was compiled, lacking support for current architecture?/d' \
		-e '/#pragma message (".*KERNEL COMPILATION WARNING: compiling .\+ code on .\+ or newer architecture: " __FILE__)/d' \
		$@
	@rm -f ${TMPFILE}
	@$(PYTHON) $(SCRDIR)/libxsmm_specialized.py $(PRECISION) $(MVALUE) $(NVALUE) $(KVALUE) $(PREFETCH_TYPE) >> $@
endif

.PHONY: main
main: $(BLDDIR)/libxsmm_dispatch.h
$(BLDDIR)/libxsmm_dispatch.h: $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(SCRDIR)/libxsmm_dispatch.py
	@$(PYTHON) $(SCRDIR)/libxsmm_dispatch.py $(PRECISION) $(THRESHOLD) $(INDICES) > $@

ifneq (0,$(MIC))
.PHONY: compile_mic
compile_mic: $(OBJFILES_MIC)
$(BLDDIR)/mic/%.o: $(SRCDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxsmm.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxsmm.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif

.PHONY: compile_hst
compile_hst: $(OBJFILES_HST)
$(BLDDIR)/intel64/%.o: $(SRCDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxsmm.h $(BLDDIR)/libxsmm_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxsmm.h $(BLDDIR)/libxsmm_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

ifneq (0,$(MIC))
.PHONY: lib_mic
lib_mic: $(OUTDIR)/mic/libxsmm.$(LIBEXT) $(INCDIR)/libxsmm.f
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OUTDIR)/mic/.make $(OBJFILES_MIC)
ifeq (0,$(STATIC))
	$(LD) -o $@ $(OBJFILES_MIC) -mmic -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $(OBJFILES_MIC)
endif
endif

.PHONY: lib_hst
lib_hst: $(OUTDIR)/libxsmm.$(LIBEXT) $(INCDIR)/libxsmm.f
$(OUTDIR)/libxsmm.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_HST) $(OBJFILES_GEN_LIB)
ifeq (0,$(STATIC))
	$(LD) -o $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB) -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB)
endif

.PHONY: samples
samples: cp2k smm nek

.PHONY: cp2k
cp2k: lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && \
	$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && \
	$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=1 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: smm
smm: lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) clean && \
	$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: smm_mic
smm_mic: lib_mic
	@cd $(SPLDIR)/smm && $(MAKE) clean && \
	$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=1 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: nek
nek: lib_hst
	@cd $(SPLDIR)/nek && $(MAKE) clean && \
	$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: nek_mic
nek_mic: lib_mic
	@cd $(SPLDIR)/nek && $(MAKE) clean && \
	$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=1 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: drytest
drytest: $(SPLDIR)/cp2k/cp2k-perf.sh $(SPLDIR)/cp2k/.make $(SPLDIR)/smm/smmf-perf.sh $(SPLDIR)/nek/grad-perf.sh $(SPLDIR)/nek/axhm-perf.sh $(SPLDIR)/nek/rstr-perf.sh

$(SPLDIR)/cp2k/cp2k-perf.sh: $(SPLDIR)/cp2k/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=cp2k-perf.txt" >> $@
ifneq (,$(INDICES))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23\"" >> $@
endif
	@echo >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  SIZE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "else" >> $@
	@echo "  SIZE=0" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  \$${HERE}/cp2k.sh \$${MVALUE} \$${SIZE} 0 \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  if [[ "0" != "\$$?" ]] ; then"  >> $@
	@echo "    exit 1"  >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/smm/smmf-perf.sh: $(SPLDIR)/smm/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/smmf-perf.txt" >> $@
ifneq (,$(INDICES))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23\"" >> $@
endif
	@echo >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  SIZE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "else" >> $@
	@echo "  SIZE=0" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/smm \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${SIZE} >> \$${FILE}" >> $@
	@echo "  if [[ "0" != "\$$?" ]] ; then"  >> $@
	@echo "    exit 1"  >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/grad-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/grad-perf.txt" >> $@
ifneq (,$(INDICES))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23\"" >> $@
endif
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
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/grad \$${MVALUE} \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  if [[ "0" != "\$$?" ]] ; then"  >> $@
	@echo "    exit 1"  >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/axhm-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/axhm-perf.txt" >> $@
ifneq (,$(INDICES))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23\"" >> $@
endif
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
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/axhm \$${MVALUE} \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  if [[ "0" != "\$$?" ]] ; then"  >> $@
	@echo "    exit 1"  >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/rstr-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/rstr-perf.txt" >> $@
ifneq (,$(INDICES))
	@echo "RUNS=\"$(INDICES)\"" >> $@
	@echo "RUNT=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_4_4\"" >> $@
	@echo "RUNT=\"8_8_8\"" >> $@
endif
	@echo >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NRUNS=\$$(echo \$${RUNS} | wc -w)" >> $@
	@echo "NRUNT=\$$(echo \$${RUNT} | wc -w)" >> $@
	@echo "NMAX=\$$((NRUNS*NRUNT))" >> $@
	@echo "for RUN1 in \$${RUNS} ; do" >> $@
	@echo "  for RUN2 in \$${RUNT} ; do" >> $@
	@echo "  MVALUE=\$$(echo \$${RUN1} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN1} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN1} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  MMVALUE=\$$(echo \$${RUN2} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NNVALUE=\$$(echo \$${RUN2} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KKVALUE=\$$(echo \$${RUN2} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/rstr \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${MMVALUE} \$${NNVALUE} \$${KKVALUE} >> \$${FILE}" >> $@
	@echo "  if [[ "0" != "\$$?" ]] ; then"  >> $@
	@echo "    exit 1"  >> $@
	@echo "  fi" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

.PHONY: tests
tests: test-cp2k test-smm test-nek

.PHONY: test
test: test-cp2k

.PHONY: test-cp2k
test-cp2k: $(SPLDIR)/cp2k/cp2k-test.txt
$(SPLDIR)/cp2k/cp2k-test.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) cp2k
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@ 1000

.PHONY: perf-cp2k
perf-cp2k: $(SPLDIR)/cp2k/cp2k-perf.txt
$(SPLDIR)/cp2k/cp2k-perf.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) cp2k
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@

.PHONY: test-smm
test-smm: $(SPLDIR)/smm/smm-test.txt
$(SPLDIR)/smm/smm-test.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) smm
	@$(SPLDIR)/smm/smmf-perf.sh $@ 1000

.PHONY: perf-smm
perf-smm: $(SPLDIR)/smm/smmf-perf.txt
$(SPLDIR)/smm/smmf-perf.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) smm
	@$(SPLDIR)/smm/smmf-perf.sh $@

.PHONY: test-nek
test-nek: $(SPLDIR)/nek/grad-perf.txt $(SPLDIR)/nek/axhm-perf.txt
$(SPLDIR)/nek/grad-perf.txt: $(SPLDIR)/nek/grad-perf.sh lib_hst
	@cd $(SPLDIR)/nek && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) grad
	@$(SPLDIR)/nek/grad-perf.sh $@
$(SPLDIR)/nek/axhm-perf.txt: $(SPLDIR)/nek/axhm-perf.sh lib_hst
	@cd $(SPLDIR)/nek && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) axhm
	@$(SPLDIR)/nek/axhm-perf.sh $@
$(SPLDIR)/nek/rstr-perf.txt: $(SPLDIR)/nek/rstr-perf.sh lib_hst
	@cd $(SPLDIR)/nek && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) rstr
	@$(SPLDIR)/nek/rstr-perf.sh $@

$(DOCDIR)/libxsmm.pdf: $(DOCDIR)/.make $(ROOTDIR)/README.md
	$(eval TEMPLATE := $(shell mktemp --tmpdir=. --suffix=.tex))
	@pandoc -D latex > $(TEMPLATE)
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@rm -f ${TMPFILE}
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
		-e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxsmm.svg?branch=.\+)\](.\+)//' \
		-e 's/\[\[.\+\](.\+)\]//' -e '/!\[.\+\](.\+)/{n;d}' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		$(ROOTDIR)/README.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSMM Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TEMPLATE)

$(DOCDIR)/cp2k.pdf: $(DOCDIR)/.make $(ROOTDIR)/documentation/cp2k.md

	$(eval TEMPLATE := $(shell mktemp --tmpdir=. --suffix=.tex))
	@pandoc -D latex > $(TEMPLATE)
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@rm -f ${TMPFILE}
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
		-e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxsmm.svg?branch=.\+)\](.\+)//' \
		-e 's/\[\[.\+\](.\+)\]//' -e '/!\[.\+\](.\+)/{n;d}' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		$(ROOTDIR)/documentation/cp2k.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
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
	@rm -f $(OBJECTS) $(SRCFILES) $(BLDDIR)/libxsmm_dispatch.h
	@rm -f $(SCRDIR)/libxsmm_utilities.pyc
	@rm -rf $(SCRDIR)/__pycache__
	@touch $(SPLDIR)/cp2k/.make
	@touch $(SPLDIR)/smm/.make
	@touch $(SPLDIR)/nek/.make
	@touch $(INCDIR)/.make

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
else
	@rm -f $(OUTDIR)/libxsmm.$(LIBEXT) $(OUTDIR)/mic/libxsmm.$(LIBEXT) $(OUTDIR)/libxsmmgen.$(LIBEXT)
endif
else
	@rm -f $(OUTDIR)/libxsmm.$(LIBEXT) $(OUTDIR)/mic/libxsmm.$(LIBEXT) $(OUTDIR)/libxsmmgen.$(LIBEXT)
endif
ifneq ($(abspath $(BINDIR)),$(ROOTDIR))
ifneq ($(abspath $(BINDIR)),$(abspath .))
	@rm -rf $(BINDIR)
else
	@rm -f $(BINDIR)/libxsmm_generator
endif
else
	@rm -f $(BINDIR)/libxsmm_generator
endif
	@rm -f *.gcno *.gcda *.gcov
	@rm -f $(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(SPLDIR)/smm/smmf-perf.sh
	@rm -f $(SPLDIR)/nek/grad-perf.sh
	@rm -f $(SPLDIR)/nek/axhm-perf.sh
	@rm -f $(SPLDIR)/nek/rstr-perf.sh
	@rm -f $(INCDIR)/libxsmm.modmic
	@rm -f $(INCDIR)/libxsmm.mod
	@rm -f $(INCDIR)/libxsmm.f
	@rm -f $(INCDIR)/libxsmm.h

.PHONY: install-minimal
ifneq ($(abspath $(PREFIX)),$(abspath .))
install-minimal: lib_all
	@echo
	@echo "LIBXSMM installing binaries..."
	@mkdir -p $(PREFIX)/$(POUTDIR) $(PREFIX)/$(PBINDIR) $(PREFIX)/$(PINCDIR)
	@cp -uv $(OUTDIR)/libxsmmgen.so $(PREFIX)/$(POUTDIR) 2> /dev/null || true
	@cp -uv $(OUTDIR)/libxsmmgen.a $(PREFIX)/$(POUTDIR) 2> /dev/null || true
	@cp -uv $(OUTDIR)/libxsmm.so $(PREFIX)/$(POUTDIR) 2> /dev/null || true
	@cp -uv $(OUTDIR)/libxsmm.a $(PREFIX)/$(POUTDIR) 2> /dev/null || true
	@if [[ -e $(OUTDIR)/mic/libxsmm.so ]] ; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic ; \
		cp -uv $(OUTDIR)/mic/libxsmm.so $(PREFIX)/$(POUTDIR)/mic ; \
	fi
	@if [[ -e $(OUTDIR)/mic/libxsmm.a ]] ; then \
		mkdir -p $(PREFIX)/$(POUTDIR)/mic ; \
		cp -uv $(OUTDIR)/mic/libxsmm.a $(PREFIX)/$(POUTDIR)/mic ; \
	fi
	@cp -uv $(BINDIR)/libxsmm_generator $(PREFIX)/$(PBINDIR) 2> /dev/null || true
	@cp -uv $(INCDIR)/libxsmm*.h $(PREFIX)/$(PINCDIR)
	@cp -uv $(INCDIR)/libxsmm.f $(PREFIX)/$(PINCDIR)
	@cp -uv $(INCDIR)/*.mod* $(PREFIX)/$(PINCDIR)
else
install-minimal: lib_all
endif

.PHONY: install
install: install-minimal
	@echo
	@echo "LIBXSMM installing documentation..."
	@mkdir -p $(PREFIX)/$(PDOCDIR)
	@cp -uv $(ROOTDIR)/$(DOCDIR)/*.pdf $(PREFIX)/$(PDOCDIR)
	@cp -uv $(ROOTDIR)/$(DOCDIR)/*.md $(PREFIX)/$(PDOCDIR)
	@cp -uv $(ROOTDIR)/version.txt $(PREFIX)/$(PDOCDIR)
	@cp -uv $(ROOTDIR)/README.md $(PREFIX)/$(PDOCDIR)
	@cp -uv $(ROOTDIR)/LICENSE $(PREFIX)/$(PDOCDIR)

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

