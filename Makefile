DIR_KNC    := .
SCRDIR_KNC := $(DIR_KNC)/scripts
OBJDIR_KNC := $(DIR_KNC)/build
INCDIR_KNC := $(DIR_KNC)/include
SRCDIR_KNC := $(DIR_KNC)/src
LIBDIR_KNC := $(DIR_KNC)/lib

INDICES_M := $(shell seq 1 3)
INDICES_N := $(shell seq 1 3)
INDICES_K := $(shell seq 1 3)
INDICES   := $(foreach m,$(INDICES_M),$(foreach n,$(INDICES_N),$(foreach k,$(INDICES_K),$m_$n_$k)))

TARGET_COMPILE_C_KNC := icc -offload-attribute-target=mic -mkl=sequential -std=c99 -openmp
AR := xiar -qoffload-build

SRCFILES_KNC := $(patsubst %,dc_small_dnn_%.c,$(INDICES))
OBJFILES_KNC := $(patsubst %,$(OBJDIR_KNC)/dc_small_dnn_%.o,$(INDICES))

LIB_KNC  := $(LIBDIR_KNC)/libxsmm.a
INC_KNC  := $(INCDIR_KNC)/xsmm_knc.h
MAIN_KNC := $(SRCDIR_KNC)/xsmm_knc.c

OBJFILES_KNC += $(patsubst $(SRCDIR_KNC)/%.c,$(OBJDIR_KNC)/%.o,$(MAIN_KNC))


all: lib_knc

lib_knc: $(LIB_KNC)
$(LIB_KNC): $(OBJFILES_KNC)
	@mkdir -p $(LIBDIR_KNC)
	$(AR) -rs $@ $^

compile_knc: $(OBJFILES_KNC)
$(OBJDIR_KNC)/%.o: $(SRCDIR_KNC)/%.c header_knc
	@mkdir -p $(OBJDIR_KNC)
	${TARGET_COMPILE_C_KNC} -I$(INCDIR_KNC) -c $< -o $@

source_knc: $(addprefix $(SRCDIR_KNC)/,$(SRCFILES_KNC))
$(SRCDIR_KNC)/%.c:
	@mkdir -p $(SRCDIR_KNC)
	@python $(SCRDIR_KNC)/xsmm_knc_gensrc.py `echo $* | awk -F_ '{ print $$4" "$$5" "$$6 }'` > $@

header_knc: $(INC_KNC)
$(INC_KNC):
	@echo "#ifndef XSMM_KNC_H" > $@
	@echo "#define XSMM_KNC_H" >> $@
	@echo >> $@
	@python $(SCRDIR_KNC)/xsmm_knc_geninc.py >> $@
	@bash -c 'for i in $(INDICES); do ( python $(SCRDIR_KNC)/xsmm_knc_geninc.py `echo $${i} | tr "_" " "` ); done' >> $@
	@echo >> $@
	@echo "#endif // XSMM_KNC_H" >> $@

main_knc: $(MAIN_KNC)
$(MAIN_KNC):
	@mkdir -p $(SRCDIR_KNC)
	@python $(SCRDIR_KNC)/xsmm_knc_genmain.py $(lastword $(INDICES_M)) $(lastword $(INDICES_K)) $(lastword $(INDICES_N)) > $@

clean:
	rm -rf $(SRCDIR_KNC) $(OBJDIR_KNC) $(DIR_KNC)/*~

realclean: clean
	rm -rf $(LIBDIR_KNC) $(INC_KNC)
