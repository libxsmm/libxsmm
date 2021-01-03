ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

OMP = 1
SYM = 1

# explore AVX/ARCH=native
SSE = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
           $(DEPDIR)/include/libxsmm_source.h
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CXXSRCS := $(wildcard $(SRCDIR)/*.cxx)
CCXSRCS := $(wildcard $(SRCDIR)/*.cc)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))
XFILES := $(OUTDIR)/eltwise_reduce $(OUTDIR)/eltwise_opreduce $(OUTDIR)/eltwise_scale $(OUTDIR)/eltwise_layernorm $(OUTDIR)/eltwise_transform $(OUTDIR)/eltwise_dropout $(OUTDIR)/eltwise_relu
.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(OUTDIR)/eltwise_reduce: $(OUTDIR)/.make $(BLDDIR)/eltwise_reduce-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_reduce-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_opreduce: $(OUTDIR)/.make $(BLDDIR)/eltwise_opreduce-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_opreduce-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_scale: $(OUTDIR)/.make $(BLDDIR)/eltwise_scale-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_scale-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_layernorm: $(OUTDIR)/.make $(BLDDIR)/eltwise_layernorm-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_layernorm-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_transform: $(OUTDIR)/.make $(BLDDIR)/eltwise_transform-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_transform-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_dropout: $(OUTDIR)/.make $(BLDDIR)/eltwise_dropout-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_dropout-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/eltwise_relu: $(OUTDIR)/.make $(BLDDIR)/eltwise_relu-c.o $(LIBDEP)
	$(LD) -o $@ $(BLDDIR)/eltwise_relu-c.o $(MAINLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* fit.log *.dat
	@rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif
	@rm -f .make .state

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@rm -f $(OUTDIR)/libxsmm.$(DLIBEXT) $(OUTDIR)/*.stackdump
	@rm -f $(XFILES) $(MODULES)
endif

