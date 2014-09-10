SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

all: $(OBJECTS)
	xiar -r  -qoffload-build micsmm.a $(OBJECTS) 
.c.o:
	icc  -offload-attribute-target=mic -mkl=sequential -std=c99 -I../../include/ -c $<

print:
