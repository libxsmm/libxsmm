MPARM = 1
NPARM = 2
KPARM = 3
MSIZE = 7
FLOPS = 8

# GEN =-1: multiple files; no titles
# GEN = 0: multiple files with titles
# GEN = 1: single file with titles
GEN = 1

HIM = -1
HIN = HIM
HIK = HIM
MN = 23
PEAK = 0 #985.6

FILECOUNT = 0
BASENAME = "smm-test"
stats BASENAME.".dat" using (column(MPARM)*column(NPARM)*column(KPARM)) nooutput; MNK = STATS_stddev**(1.0/3.0)
stats BASENAME.".dat" using (log(column(FLOPS))) nooutput; GEO = sprintf("%.1f", exp(STATS_sum/STATS_records))
stats BASENAME.".dat" using FLOPS nooutput; MED = sprintf("%.1f", STATS_median)
stats BASENAME.".dat" using NPARM nooutput; XN = int(STATS_max)

MAX(A, B) = A < B ? B : A
IX(I1, J1, NJ) = int(MAX(I1 - 1, 0) * NJ + MAX(J1 - 1, 0))
I1(IX, NJ) = int(IX / NJ) + 1
J1(IX, NJ) = int(IX) % NJ + 1

set table "smm-test-avg.txt"
plot BASENAME.".dat" using (IX(column(MPARM), column(NPARM), XN)):FLOPS smooth unique
unset table
set table "smm-test-cdf.txt"
plot BASENAME.".dat" using FLOPS:(1.0) smooth cumulative
unset table
stats "smm-test-cdf.txt" using (("".strcol(3)."" eq "i")?($2):(1/0)) nooutput; FREQSUM = STATS_max

TERMINAL = "pdf"
EXT = TERMINAL[1:3]


set terminal TERMINAL
set termoption enhanced
#set termoption font "Times-Roman,7"
save_encoding = GPVAL_ENCODING
set encoding utf8


if (GEN==1) set output BASENAME.".".EXT


reset
if (GEN<=0) { set output BASENAME."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
if (GEN>-1) { set title "Performance" }
set pm3d interpolate 0, 0
#set colorbox horizontal user origin 0, 0.1 size 1, 0.1
set autoscale fix
if (0<HIM) { set xrange [*:HIM] }
if (0<HIN) { set yrange [*:HIN] }
if (0<HIK) { set zrange [*:HIK] }
if (0>HIM) { set xrange [*:MNK] }
if (0>HIN) { set xrange [*:MNK] }
if (0>HIK) { set xrange [*:MNK] }
set xlabel "M"
set ylabel "N"
set zlabel "K"
set ticslevel 0
set cblabel "GFLOP/s" offset 1.0
set format x "%g"; set format y "%g"; set format z "%g"; set format cb "%g"
splot BASENAME.".dat" using MPARM:NPARM:KPARM:FLOPS notitle with points pointtype 7 linetype palette

reset
if (GEN<=0) { set output BASENAME."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
if (GEN>-1) { set title "Performance (K-Average)" }
set dgrid3d #9, 9
set pm3d interpolate 0, 0 map
set autoscale fix
set xlabel "M"
set ylabel "N" offset -1.0
set cblabel "GFLOP/s" offset 1.0
set format x "%g"; set format y "%g"; set format cb "%g"
set mxtics 2
#set offsets 1, 1, 1, 1
splot "smm-test-avg.txt" using (("".strcol(3)."" eq "i")?(I1($1, XN)):(1/0)):(("".strcol(3)."" eq "i")?(J1($1, XN)):(1/0)):2 notitle with pm3d

reset
if (GEN<=0) { set output BASENAME."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
if (GEN>-1) { set title "Performance (CDF)" }
set xlabel "Probability\n\nGeo. Mean: ".GEO." GFLOP/s  Median: ".MED." GFLOP/s"
set ylabel "GFLOP/s"
set format x "%g%%"
set format y "%g"
set fit quiet
f(x) = b * x + a
fit f(x) "smm-test-cdf.txt" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via a, b
g(x) = (x - a) / b
x = 0.5 * (100 + MAX(0, g(0)))
h(x) = d * x + c
fit [x-2:x+2] h(x) "smm-test-cdf.txt" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via c, d
set arrow 1 from x, h(x) to x, 0
set label 1 sprintf("%.1f%%", x) at x, 0.5 * h(x) left offset 1
set arrow 2 from x, h(x) to 0, h(x)
set label 2 sprintf("%.1f GFLOP/s", h(x)) at 0.5 * x, h(x) centre offset 0, 1
set autoscale fix
set xrange [0:100]
set yrange [0:*]
plot "smm-test-cdf.txt" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 notitle with lines

if (0 < PEAK) {
  reset
  if (GEN<=0) { set output BASENAME."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
  if (GEN>-1) { set title "Performance and Efficiency" }
  set xlabel "(M N K)^{-1/3}"
  set ylabel "Efficiency"
  set y2label "GFLOP/s"
  set y2tics
  set ytics nomirror
  set format x "%g"; set format y "%g%%"
  set mxtics 2
  set mytics 2
  set my2tics 2
  set autoscale fix
  set yrange [0:100]
  set y2range [0:PEAK]
  plot BASENAME.".dat" using (floor((column(MPARM)*column(NPARM)*column(KPARM))**(1.0/3.0)+0.5)):(100.0*column(FLOPS)/PEAK) notitle smooth sbezier with points pointtype 7 pointsize 0.5
}
