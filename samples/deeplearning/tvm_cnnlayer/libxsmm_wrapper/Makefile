CC= icpc
CFLAGS= -O3 -fPIC -std=c++11 -fopenmp
LDFLAGS= -shared
SOURCES = batch_reduce_plus_init.cc
LIBXSMMDIR=./../../../../

INC=-I$(LIBXSMMDIR)/include
LIBS = $(LIBXSMMDIR)/lib/libxsmm.a $(LIBXSMMDIR)/lib/libxsmmext.a \
       $(LIBXSMMDIR)/lib/libxsmmnoblas.a $(LIBXSMMDIR)/lib/libxsmmgen.a \
       $(LIBXSMMDIR)/lib/libxsmmf.a

TARGET= libxsmm_wrapper.so

all:
	$(CC) $(INC) $(CFLAGS) -fPIC $(SOURCES) $(LIBS)  -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

