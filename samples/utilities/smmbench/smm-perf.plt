MPARM = 1
NPARM = 2
KPARM = 3
FLOPS = 5

HIM = -1
HIN = HIM
HIK = HIM

FILENAME = system("sh -c \"echo ${FILENAME}\"")
if (FILENAME eq "") {
  FILENAME = "smm-perf.pdf"
}

FILECOUNT = 1 # initial file number
# MULTI =-1: multiple files; no titles
# MULTI = 0: multiple files with titles
# MULTI = 1: single file with titles
MULTI = system("sh -c \"echo ${MULTI}\"")
if (MULTI eq "") {
  MULTI = 1
}

LIMIT = system("sh -c \"echo ${LIMIT}\"")
if (LIMIT eq "") {
  LIMIT = 128
}

if (MULTI==1) {
  set output FILENAME
}

FILEEXT = system("sh -c \"echo ".FILENAME." | sed 's/.\\+\\.\\(.\\+\\)/\\1/'\"")
set terminal FILEEXT
set termoption enhanced
#set termoption font ",12"
save_encoding = GPVAL_ENCODING
set encoding utf8


reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Selected Problem Instances" }
set style fill solid 0.4 border -1
set style data histograms
set style histogram cluster #gap 2
#set boxwidth 0.5 relative
set grid y2tics lc "grey"
set key left #spacing 0.5
set xtics rotate by -45 scale 0; set bmargin 6
set ytics format ""
set y2tics nomirror
set y2label "GFLOP/s"
set xrange [0:LIMIT+0.85]
set yrange [0:*]
set autoscale fix
if (0!=system("sh -c \"if [ -e smm-inlined.dat ]; then echo 1; else echo 0; fi\"")) {
plot  "smm-inlined.dat" using FLOPS title "Inlined", \
      "smm-blas.dat" using FLOPS title "BLAS", \
      "smm-dispatched.dat" using FLOPS title "Dispatched", \
      "smm-specialized.dat" using FLOPS:xtic("(".strcol(MPARM).",".strcol(NPARM).",".strcol(KPARM).")") title "Specialized"
} else {
plot  "smm-blas.dat"        using FLOPS title "BLAS", \
      "smm-specialized.dat" using FLOPS:xtic("(".strcol(MPARM).",".strcol(NPARM).",".strcol(KPARM).")") title "Specialized"
}
