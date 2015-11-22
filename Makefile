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

# Linux cut has features we use that do not work elsewhere
# Mac, etc. users should install GNU coreutils and use cut from there.
#
# For example, if you use Homebrew, run "brew install coreutils" once
# and then invoke the LIBXSMM make command with
# CUT=/usr/local/Cellar/coreutils/8.24/libexec/gnubin/cut
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

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Generate prefetches
PREFETCH ?= 0

# THRESHOLD problem size (M x N x K) determining when to use BLAS; can be zero
THRESHOLD ?= $(shell echo $$((80 * 80 * 80)))

# Use aligned Store and/or aligned Load instructions
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

# Select certain code path
SSE ?= 0
AVX ?= 0

# ILP64=0 (LP64 with 32-bit integers), and ILP64=0 (64-bit integers)
ILP64 ?= 0
BLAS ?= 0

OFFLOAD ?= 0
ifneq (0,$(OFFLOAD))
	MIC ?= 1
else
	MIC ?= 0
endif

# PLEASE NOTE THIS IS A PREVIEW OF OUR JITTING FEATURE, CURRENTLY THERE
# IS NO CLEAN-UP ROUTINE, JITTED MEMORY IS FREED AT PROGRAM EXIT ONLY!
JIT ?= 0
ifneq (0,$(JIT))
$(info =====================================================================)
$(info YOU ARE USING AN EXPERIMENTAL VERSION OF LIBXSMM WITH JIT SUPPORT)
$(info PLEASE NOTE THIS IS A PREVIEW OF OUR JITTING FEATURE, CURRENTLY THERE)
$(info IS NO CLEAN-UP ROUTINE, JITTED MEMORY IS FREED AT PROGRAM EXIT ONLY!)
$(info =====================================================================)
ifneq (0,$(ROW_MAJOR))
$(error ROW_MAJOR needs to be 0 for JIT support!)
endif
ifneq (0,$(OFFLOAD))
$(error OFFLOAD needs to be 0 for JIT support!)
endif
ifneq (0,$(MIC))
$(error MIC needs to be 0 for JIT support!)
endif
ifneq (0,$(SSE))
$(error SSE needs to be 0 for JIT support!)
endif
endif

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

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

INDICES ?= $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py -1 $(THRESHOLD) $(words $(MNK)) $(MNK) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
NINDICES = $(words $(INDICES))

SRCFILES = $(addprefix $(BLDDIR)/,$(patsubst %,mm_%.c,$(INDICES)))
SRCFILES_GEN_LIB = $(patsubst %,$(SRCDIR)/%,generator_common.c generator_dense.c generator_dense_common.c generator_dense_instructions.c \
                                            generator_dense_sse3_avx_avx2.c generator_dense_sse3_microkernel.c generator_dense_avx_microkernel.c generator_dense_avx2_microkernel.c \
                                            generator_dense_avx512_microkernel.c generator_dense_imci_avx512.c generator_dense_imci_microkernel.c generator_dense_noarch.c \
                                            generator_sparse.c generator_sparse_csc_reader.c generator_sparse_bsparse.c generator_sparse_asparse.c \
                                            libxsmm_timer.c)
SRCFILES_GEN_BIN = $(patsubst %,$(SRCDIR)/%,generator_driver.c)
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_GEN_BIN = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_BIN))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES)) $(BLDDIR)/intel64/libxsmm_crc32.o $(BLDDIR)/intel64/libxsmm_dispatch.o
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES)) $(BLDDIR)/mic/libxsmm_crc32.o $(BLDDIR)/mic/libxsmm_dispatch.o $(BLDDIR)/mic/libxsmm_timer.o

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

.PHONY: install
install: all clean

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
$(INCDIR)/libxsmm.h: $(SRCDIR)/libxsmm.template.h $(ROOTDIR)/.hooks/install.sh $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py \
                     $(ROOTDIR)/include/libxsmm_macros.h $(ROOTDIR)/include/libxsmm_typedefs.h $(ROOTDIR)/include/libxsmm_frontend.h \
                     $(ROOTDIR)/include/libxsmm_generator.h $(ROOTDIR)/include/libxsmm_timer.h \
                     $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@$(ROOTDIR)/.hooks/install.sh
	@cp $(ROOTDIR)/include/libxsmm_macros.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_typedefs.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_frontend.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_generator.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_timer.h $(INCDIR) 2> /dev/null || true
	@$(PYTHON) $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.h $(MAKE_ILP64) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@

.PHONY: fheader
fheader: $(INCDIR)/libxsmm.f
$(INCDIR)/libxsmm.f: $(SRCDIR)/libxsmm.template.f $(ROOTDIR)/.hooks/install.sh $(SCRDIR)/libxsmm_interface.py $(SCRDIR)/libxsmm_utilities.py \
                     $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@) $(BLDDIR)
	@$(ROOTDIR)/.hooks/install.sh
	@$(PYTHON) $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/libxsmm.template.f $(MAKE_ILP64) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@
ifeq (0,$(OFFLOAD))
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} '/ATTRIBUTES OFFLOAD:MIC/d' $@
	@rm -f ${TMPFILE} 
endif
	$(FC) $(FCFLAGS) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $@ -o $(BLDDIR)/libxsmm-mod.o $(FMFLAGS) $(dir $@)

.PHONY: compile_generator_lib
compile_generator_lib: $(OBJFILES_GEN_LIB)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@
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
compile_generator: $(OBJFILES_GEN_BIN)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(INCDIR)/libxsmm.h $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@
.PHONY: generator
generator: $(BINDIR)/generator
$(BINDIR)/generator: $(OBJFILES_GEN_BIN) $(OUTDIR)/intel64/libxsmmgen.$(LIBEXT) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) $(CLDFLAGS) $(OBJFILES_GEN_BIN) -L$(OUTDIR)/intel64 -lxsmmgen -o $@

.PHONY: sources
sources: $(SRCFILES)
$(BLDDIR)/%.c: $(INCDIR)/libxsmm.h $(BINDIR)/generator $(SCRDIR)/libxsmm_utilities.py $(SCRDIR)/libxsmm_specialized.py
	$(eval MVALUE := $(shell echo $* | $(CUT) --output-delimiter=' ' -d_ -f2))
	$(eval NVALUE := $(shell echo $* | $(CUT) --output-delimiter=' ' -d_ -f3))
	$(eval KVALUE := $(shell echo $* | $(CUT) --output-delimiter=' ' -d_ -f4))
ifneq (0,$(ROW_MAJOR)) # row-major
	$(eval MVALUE2 := $(NVALUE))
	$(eval NVALUE2 := $(MVALUE))
else # column-major
	$(eval MVALUE2 := $(MVALUE))
	$(eval NVALUE2 := $(NVALUE))
endif
ifneq (0,$(ALIGNED_LOADS)) # aligned loads
	$(eval LDASP := $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py $(MVALUE2) 16 $(ALIGNMENT)))
	$(eval LDADP := $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py $(MVALUE2)  8 $(ALIGNMENT)))
else # unaligned stores
	$(eval LDASP := $(MVALUE2))
	$(eval LDADP := $(MVALUE2))
endif
ifneq (0,$(ALIGNED_STORES)) # aligned stores
	$(eval LDCSP := $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py $(MVALUE2) 16 $(ALIGNMENT)))
	$(eval LDCDP := $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py $(MVALUE2)  8 $(ALIGNMENT)))
else # unaligned stores
	$(eval LDCSP := $(MVALUE2))
	$(eval LDCDP := $(MVALUE2))
endif
	$(eval LDB := $(KVALUE))
	@mkdir -p $(dir $@)
	@echo "#include <libxsmm.h>" > $@
	@echo >> $@
ifneq (0,$(MIC))
	@echo "#define LIBXSMM_GENTARGET_knc_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_knc_dp" >> $@
endif
ifeq (noarch,$(GENTARGET))
	@echo "#define LIBXSMM_GENTARGET_knl_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_knl_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_dp" >> $@
	@echo >> $@
	@echo >> $@
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knl $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDASP) $(LDB) $(LDCSP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) knl $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knl $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCDP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) knl $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCSP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) hsw $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCDP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) hsw $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCSP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) snb $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCDP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) snb $(PREFETCH_SCHEME) DP
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDASP) $(LDB) $(LDCSP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) wsm $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCDP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) wsm $(PREFETCH_SCHEME) DP
else
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_dp" >> $@
	@echo >> $@
	@echo >> $@
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDASP) $(LDB) $(LDCSP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) $(GENTARGET) $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCDP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) $(GENTARGET) $(PREFETCH_SCHEME) DP
endif
ifneq (0,$(MIC))
	$(BINDIR)/generator dense $@ libxsmm_s$(basename $(notdir $@))_knc $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDASP) $(LDB) $(LDCSP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) knc $(PREFETCH_SCHEME) SP
	$(BINDIR)/generator dense $@ libxsmm_d$(basename $(notdir $@))_knc $(MVALUE2) $(NVALUE2) $(KVALUE) $(LDADP) $(LDB) $(LDCDP) $(ALPHA) $(BETA) $(ALIGNED_LOADS) $(ALIGNED_STORES) knc $(PREFETCH_SCHEME) DP
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
	@$(PYTHON) $(SCRDIR)/libxsmm_specialized.py $(ROW_MAJOR) $(MVALUE) $(NVALUE) $(KVALUE) $(PREFETCH_TYPE) >> $@

.PHONY: main
main: $(BLDDIR)/libxsmm_dispatch.h
$(BLDDIR)/libxsmm_dispatch.h: $(INCDIR)/libxsmm.h $(SCRDIR)/libxsmm_dispatch.py
	@mkdir -p $(dir $@)
	@$(PYTHON) $(SCRDIR)/libxsmm_dispatch.py $(PREFETCH_TYPE) $(THRESHOLD) $(INDICES) > $@

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
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(INCDIR)/libxsmm.h $(BLDDIR)/libxsmm_dispatch.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@
$(BLDDIR)/intel64/%.o: $(SRCDIR)/%.c $(INCDIR)/libxsmm.h $(BLDDIR)/libxsmm_dispatch.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

ifneq (0,$(MIC))
.PHONY: lib_mic
lib_mic: $(OUTDIR)/mic/libxsmm.$(LIBEXT)
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OBJFILES_MIC)
	@mkdir -p $(dir $@)
ifeq (0,$(STATIC))
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif
endif

.PHONY: lib_hst
lib_hst: $(OUTDIR)/intel64/libxsmm.$(LIBEXT)
$(OUTDIR)/intel64/libxsmm.$(LIBEXT): $(OBJFILES_HST) $(OBJFILES_GEN_LIB)
	@mkdir -p $(dir $@)
ifeq (0,$(STATIC))
	$(LD) -o $@ $^ -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $^
endif
ifneq (0,$(JIT))
	$(info =====================================================================)
	$(info YOU ARE USING AN EXPERIMENTAL VERSION OF LIBXSMM WITH JIT SUPPORT)
	$(info PLEASE NOTE THIS IS A PREVIEW OF OUR JITTING FEATURE, CURRENTLY THERE)
	$(info IS NO CLEAN-UP ROUTINE, JITTED MEMORY IS FREED AT PROGRAM EXIT ONLY!)
	$(info =====================================================================)
endif

.PHONY: samples
samples: smm cp2k nek

.PHONY: smm
smm: lib_all
	@cd $(SPLDIR)/smm && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX)

.PHONY: nek
nek: lib_all
	@cd $(SPLDIR)/nek && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX)

.PHONY: smm_hst
smm_hst: lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD)
.PHONY: smm_mic
smm_mic: lib_mic
	@cd $(SPLDIR)/smm && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=$(MIC)

.PHONY: cp2k
cp2k: lib_all
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX)
.PHONY: cp2k_hst
cp2k_hst: lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD)
.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@cd $(SPLDIR)/cp2k && $(MAKE) clean && $(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=$(MIC)

.PHONY: drytest
drytest: $(SPLDIR)/cp2k/cp2k-perf.sh $(SPLDIR)/smm/smmf-perf.sh $(SPLDIR)/nek/grad-perf.sh $(SPLDIR)/nek/axhm-perf.sh $(SPLDIR)/nek/rstr-perf.sh
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
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  \$${HERE}/cp2k.sh \$${MVALUE} 0 0 \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/smm/smmf-perf.sh: $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/smmf-perf.txt" >> $@
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
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/smm \$${MVALUE} \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/grad-perf.sh: $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/grad-perf.txt" >> $@
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
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/grad \$${MVALUE} \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/axhm-perf.sh: $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/axhm-perf.txt" >> $@
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
	@echo "  MVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(echo \$${RUN} | $(CUT) --output-delimiter=' ' -d_ -f3)" >> $@
	@echo "  >&2 echo \"Test \$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})\"" >> $@
	@echo "  CHECK=1 \$${HERE}/axhm \$${MVALUE} \$${NVALUE} \$${KVALUE} >> \$${FILE}" >> $@
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/rstr-perf.sh: $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	@echo "#!/bin/bash" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "FILE=\$${HERE}/rstr-perf.txt" >> $@
	@echo "RUNS='$(INDICES)'" >> $@
	@echo "RUNT='$(INDICES)'" >> $@
	@echo >> $@
	@echo "if [[ \"\" != \"\$$1\" ]] ; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(echo \$${RUNS} | wc -w)" >> $@
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
	@echo "  echo >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN + 1))" >> $@
	@echo "done" >> $@
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

.PHONY: testf
testf: $(SPLDIR)/smm/smmf-perf.txt
$(SPLDIR)/smm/smmf-perf.txt: $(SPLDIR)/smm/smmf-perf.sh lib_all
	@cd $(SPLDIR)/smm && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) realclean && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO)
	@$(SPLDIR)/smm/smmf-perf.sh $@

.PHONY: testnek
testnek: $(SPLDIR)/nek/grad-perf.txt $(SPLDIR)/nek/axhm-perf.txt
$(SPLDIR)/nek/grad-perf.txt: $(SPLDIR)/nek/grad-perf.sh lib_all
	@cd $(SPLDIR)/nek && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) realclean && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO)
	@$(SPLDIR)/nek/grad-perf.sh $@
$(SPLDIR)/nek/axhm-perf.txt: $(SPLDIR)/nek/axhm-perf.sh lib_all
	@cd $(SPLDIR)/nek && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) realclean && \
		$(MAKE) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO)
	@$(SPLDIR)/nek/axhm-perf.sh $@

$(DOCDIR)/libxsmm.pdf: $(ROOTDIR)/README.md
	@mkdir -p $(dir $@)
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
		-e 's/\[\[.\+\](.\+)\]//' \
		-e '/!\[.\+\](.\+)/{n;d}' \
		$(ROOTDIR)/README.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSMM Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
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
	@TMPFILE=`mktemp`
	@sed -i ${TMPFILE} \
		-e 's/\(\\documentclass\[.\+\]{.\+}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TEMPLATE)
	@rm -f ${TMPFILE}
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxsmm\/master\///' \
		-e 's/\[!\[.\+\](https:\/\/travis-ci.org\/hfp\/libxsmm.svg?branch=.\+)\](.\+)//' \
		-e 's/\[\[.\+\](.\+)\]//' \
		-e '/!\[.\+\](.\+)/{n;d}' \
		$(ROOTDIR)/documentation/cp2k.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TEMPLATE) --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable \
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
	@rm -f $(OBJECTS) $(BLDDIR)/libxsmm_dispatch.h $(BLDDIR)/*.mod
endif
else
	@rm -f $(OBJECTS) $(BLDDIR)/libxsmm_dispatch.h $(BLDDIR)/*.mod
endif
	@rm -rf $(SCRDIR)/__pycache__
	@rm -f $(SCRDIR)/libxsmm_utilities.pyc

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
else
	@rm -f $(OUTDIR)/intel64/libxsmm.$(LIBEXT) $(OUTDIR)/mic/libxsmm.$(LIBEXT) $(OUTDIR)/intel64/libxsmmgen.$(LIBEXT)
endif
else
	@rm -f $(OUTDIR)/intel64/libxsmm.$(LIBEXT) $(OUTDIR)/mic/libxsmm.$(LIBEXT) $(OUTDIR)/intel64/libxsmmgen.$(LIBEXT)
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
	@rm -f $(SPLDIR)/smm/smmf-perf.sh
	@rm -f $(SPLDIR)/nek/grad-perf.sh
	@rm -f $(INCDIR)/libxsmm.mod
	@rm -f $(INCDIR)/libxsmm.f
	@rm -f $(INCDIR)/libxsmm.h

install: all clean
