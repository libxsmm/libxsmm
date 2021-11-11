
.PHONY: run
run:
	LD_LIBRARY_PATH=../../lib LIBXSMM_VERBOSE=-1 ./compiled

.PHONY: valgrind
valgrind:	
	LD_LIBRARY_PATH=../../lib LIBXSMM_VERBOSE=-1 valgrind ./compiled

.PHONY: compile
compile:
	g++ -g -DBLAS=0 -DLIBXSMM_NO_BLAS=1 -I../../include sve.cpp -L../../lib -lxsmm -pthread -o compiled

# -f is used to avoid reporting an error, when no file is found;
# this command is useful, so that libxsmm doesn't cry, when we run it twice, and it has to override the old debugging files
.PHONY: clean
clean:
	rm -f libxsmm_a64fx_tsize*
	rm -f core.*

