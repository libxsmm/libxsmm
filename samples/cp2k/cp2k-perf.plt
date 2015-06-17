MPARM = 1
NPARM = 2
KPARM = 3
FLOPS = 8
MEMBW = 9

HIM = -1
HIN = HIM
HIK = HIM
MN = 23
PEAK = 0 #985.6

BASENAME = "cp2k-perf"
FILENAME = system("sh -c \"echo ${FILENAME}\"")
if (FILENAME eq "") {
  FILENAME = BASENAME.".pdf"
}

FILECOUNT = 1 # initial file number
# MULTI =-1: multiple files; no titles
# MULTI = 0: multiple files with titles
# MULTI = 1: single file with titles
MULTI = system("sh -c \"echo ${MULTI}\"")
if (MULTI eq "") {
  MULTI = 1
}

XFLOPS(M, N, K) = 2 * M * N * K
NFLOPS(M, N, K) = XFLOPS(column(M), column(N), column(K))
NBYTES(M, N, K, ELEMSIZE) = ELEMSIZE * (column(M) * column(K) + column(K) * column(N) + column(M) * column(N))
AI(M, N, K, ELEMSIZE) = NFLOPS(M, N, K) / NBYTES(M, N, K, ELEMSIZE)

stats BASENAME.".dat" using (NFLOPS(MPARM,NPARM,KPARM)) nooutput; MNK = STATS_stddev**(1.0/3.0); MAXMNK = int(STATS_max)
stats BASENAME.".dat" using (log(column(FLOPS))) nooutput; GEO = exp(STATS_sum/STATS_records)
stats BASENAME.".dat" using FLOPS nooutput; MED = STATS_median; MINFLOPS = STATS_min; MAXFLOPS = STATS_max
stats BASENAME.".dat" using NPARM nooutput; XN = int(STATS_max)

MAX(A, B) = A < B ? B : A
ACC = sprintf("%%.%if", ceil(MAX(1.0 / log10(MED) - 1.0, 0)))

IX(I1, J1, NJ) = int(MAX(I1 - 1, 0) * NJ + MAX(J1 - 1, 0))
I1(IX, NJ) = int(IX / NJ) + 1
J1(IX, NJ) = int(IX) % NJ + 1

set table BASENAME."-avg.dat"
plot BASENAME.".dat" using (IX(column(MPARM), column(NPARM), XN)):FLOPS smooth unique
unset table
set table BASENAME."-mbw.dat"
plot BASENAME.".dat" using MEMBW:(1.0) smooth cumulative
unset table
set table BASENAME."-cdf.dat"
plot BASENAME.".dat" using FLOPS:(1.0) smooth cumulative
unset table
stats BASENAME."-cdf.dat" using (("".strcol(3)."" eq "i")?($2):(1/0)) nooutput; FREQSUM = STATS_max; FREQN = STATS_records

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
if (MULTI>-1) { set title "Performance" }
set origin -0.03, 0
set pm3d interpolate 0, 0
#set colorbox horizontal user origin 0, 0.1 size 1, 0.1
#set autoscale fix
if (0<HIM) { set xrange [*:HIM] }
if (0<HIN) { set yrange [*:HIN] }
if (0<HIK) { set zrange [*:HIK] }
if (0>HIM) { set xrange [*:MNK] }
if (0>HIN) { set yrange [*:MNK] }
if (0>HIK) { set zrange [*:MNK] }
set xlabel "M"
set ylabel "N" offset -3.0
set zlabel "K" offset 1.0
set ticslevel 0
set cblabel "GFLOP/s" offset 1.5
set format x "%g"; set format y "%g"; set format z "%g"; set format cb "%g"
splot BASENAME.".dat" using MPARM:NPARM:KPARM:FLOPS notitle with points pointtype 7 linetype palette

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Performance (K-Average)" }
set origin -0.02, 0
set dgrid3d #9, 9
set pm3d interpolate 0, 0 map
set autoscale fix
set xlabel "M"
set ylabel "N" offset -1.5
set cblabel "GFLOP/s" offset 0.5
set format x "%g"; set format y "%g"; set format cb "%g"
set mxtics 2
splot BASENAME."-avg.dat" using (("".strcol(3)."" eq "i")?(I1($1, XN)):(1/0)):(("".strcol(3)."" eq "i")?(J1($1, XN)):(1/0)):2 notitle with pm3d

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Performance (Average per Bin)" }
set style fill solid 0.4 noborder
set boxwidth 0.5
set grid y2tics linecolor "grey"
unset key
unset xtics
set xtics ("MNK <= 13^3" 0, "13^3 < MNK <= 23^3" 1, "23^3 < MNK" 2) scale 0 offset 0, 0.2
set x2tics ("Small" 0, "Medium" 1, "Larger" 2) scale 0
set xlabel "Problem Size (MNK)"
set ytics format ""
set y2tics nomirror
set y2label "GFLOP/s"
set xrange [-0.5:2.5]
plot  BASENAME.".dat" \
      using (0.0):((NFLOPS(MPARM,NPARM,KPARM)<=XFLOPS(13,13,13))?column(FLOPS):1/0) notitle smooth unique with boxes linetype 1 linecolor "grey", \
  ""  using (1.0):(((XFLOPS(13,13,13)<NFLOPS(MPARM,NPARM,KPARM))&&(NFLOPS(MPARM,NPARM,KPARM)<=XFLOPS(23,23,23)))?column(FLOPS):1/0) notitle smooth unique with boxes linetype 1 linecolor "grey", \
  ""  using (2.0):((XFLOPS(23,23,23)<NFLOPS(MPARM,NPARM,KPARM))?column(FLOPS):1/0) notitle smooth unique with boxes linetype 1 linecolor "grey"

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Performance and Memory Bandwidth (CDF)" }
set xlabel "Probability\n\nMin.: ".sprintf(ACC, MINFLOPS)." GFLOP/s  Geo. Mean: ".sprintf(ACC, GEO)." GFLOP/s  Median: ".sprintf(ACC, MED)." GFLOP/s  Max.: ".sprintf(ACC, MAXFLOPS)." GFLOP/s"
set ylabel "GB/s"
set y2label "GFLOP/s"
set format x "%g%%"
set format y "%g"
set format y2 "%g"
set ytics nomirror
set y2tics nomirror
set grid x y2 linecolor "grey"
set xrange [0:100]
set yrange [0:*]
set y2range [0:*]
set fit quiet
f(x) = b * x + a
fit f(x) BASENAME."-cdf.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via a, b
g(x) = (x - a) / b
x50 = 0.5 * (100 + MAX(0, g(0)))
h(x) = d * x + c
dx = 100 / FREQN
fit [x50-1.5*dx:x50+1.5*dx] h(x) BASENAME."-cdf.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via c, d
set arrow from x50, second h(x50) to x50, second 0 front
set arrow from x50, second h(x50) to 100, second h(x50) front
set label sprintf("%.0f%%", x50) at x50, second 0.5 * h(x50) left offset 1 front
set label sprintf(ACC." GFLOP/s", h(x50)) at 0.5 * (x50 + 100.0), second h(x50) centre offset 0, -1 front
plot  BASENAME."-mbw.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 axes x1y1 notitle with lines linecolor "grey", \
      BASENAME."-cdf.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 axes x1y2 notitle with lines linewidth 2

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Performance and Arithmetic Intensity" }
set grid x y2 linecolor "grey"
set key left #spacing 0.5
set ytics format ""
set y2tics nomirror
set y2label "GFLOP/s"
set xlabel "FLOPS/Byte"
plot  BASENAME.".dat" using (AI(MPARM,NPARM,KPARM,8)):FLOPS notitle smooth sbezier with lines linecolor "grey", \
                   "" using (AI(MPARM,NPARM,KPARM,8)):FLOPS notitle smooth unique with points pointtype 7 pointsize 0.1

if (0 < PEAK) {
  reset
  if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
  if (MULTI>-1) { set title "Performance and Efficiency" }
  set xlabel "(M N K)^{-1/3}"
  set ylabel "Efficiency"
  set y2label "GFLOP/s"
  set y2tics
  set ytics nomirror
  set format x "%g"; set format y "%g%%"
  set mxtics 2
  set mytics 2
  set my2tics 2
  #set autoscale fix
  set yrange [0:100]
  set y2range [0:PEAK]
  plot BASENAME.".dat" using (floor(NFLOPS(MPARM,NPARM,KPARM)**(1.0/3.0)+0.5)):(100.0*column(FLOPS)/PEAK) notitle smooth sbezier with points pointtype 7 pointsize 0.5
}
