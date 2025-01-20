ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
#DFLAGS = -DLIBXSMM_BLAS_CONST

#BLAS = 2
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))

OUTNAME := $(shell basename "$(ROOTDIR)")
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
XFILES := $(OUTDIR)/statictest

.PHONY: all
all: $(XFILES) libstatictest/libstatictest.h

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

.PHONY: libstatictest/libstatictest.h
libstatictest/libstatictest.h: $(DEPDIR)/scripts/libxsmm_staticlib.py
	$(info Creating static lib...)
	./$(DEPDIR)/scripts/libxsmm_staticlib.py -c statictest.json -o . -v

$(OUTDIR)/statictest: $(OUTDIR)/.make $(BLDDIR)/statictest-c.o libstatictest/libstatictest.a
	$(LD) -o $@ $(BLDDIR)/statictest-c.o libstatictest/libstatictest.a $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make libstatictest/libstatictest.h Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
	@-rm -f one one.o two two.o libstatictest.h libstatictest.a
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -rf $(BLDDIR)
	@-rm -f one one.o two two.o libstatictest.h libstatictest.a
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
	@-rm -f one one.o two two.o libstatictest.h libstatictest.a
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -rf $(OUTDIR)
	@-rm -f one one.o two two.o libstatictest.h libstatictest.a
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
