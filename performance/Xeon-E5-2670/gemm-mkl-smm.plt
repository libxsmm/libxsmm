XNAME = 1
FNAME = 2
TNAME = 3
PNAME = 4
NOTES = 5
MPARM = 6
KPARM = 7
NPARM = 8
NVALS = 8
MSIZE = 9
DURMS = 11
DEVMS = 13
VALID = 15
FLOPS = 19
XTEND = 21

# GEN =-1: multiple files; no titles
# GEN = 0: multiple files with titles
# GEN = 1: single file with titles
GEN = 1

GEMX = "gemm"
FUNC = "mkl"
NAME = "MKL/SMM/Xeon_E5-2670"
PREC = "f64"

#FUNCB = "goto"
NAMEB = "OpenBLAS"
SHIFT = 20

NCORES = 16
PEAK_SP_GFLOPS = 41.6 * NCORES
PEAK_DP_GFLOPS = 0.5 * PEAK_SP_GFLOPS
HIM = 9
HIN = HIM
HIK = HIM
MN = 4

FILECOUNT = 0
BASENAME = GEMX."-".FUNC."-".PREC
stats BASENAME.".txt" using (log(column(FLOPS)*NCORES)) nooutput; GEO = sprintf("%.1f", exp(STATS_sum/STATS_records))
stats BASENAME.".txt" using (column(FLOPS)*NCORES) nooutput; MED = sprintf("%.1f", STATS_median)
stats BASENAME.".txt" using NPARM nooutput; XN = int(STATS_max)

MAX(A, B) = A < B ? B : A
IX(I1, J1, NJ) = int(MAX(I1 - 1, 0) * NJ + MAX(J1 - 1, 0))
I1(IX, NJ) = int(IX / NJ) + 1
J1(IX, NJ) = int(IX) % NJ + 1

set table "gemx-avg.txt"
plot BASENAME.".txt" using (IX(column(MPARM), column(NPARM), XN)):(column(FLOPS)*NCORES) smooth unique
unset table
set table "gemx-cdf.txt"
plot BASENAME.".txt" using (column(FLOPS)*NCORES):(1.0) smooth cumulative
unset table
stats "gemx-cdf.txt" using (("".strcol(3)."" eq "i")?($2):(1/0)) nooutput; FREQSUM = STATS_max

TERMINAL = "pdf"
set terminal TERMINAL
#set termoption font "Times-Roman,7"

EXT = TERMINAL[1:3]

if (GEN==1) set output "gemm-mkl-smm.".EXT
if (PREC eq "f64") {
  PEAK = PEAK_DP_GFLOPS
  PRECISION = "D"
} else {
  PEAK = PEAK_SP_GFLOPS
  PRECISION = "S"
}


if (exists("FUNCB") && (FUNCB ne "") && exists("NAMEB") && (NAMEB ne "")) {
  reset
  if (GEN<=0) { set output GEMX."-".FUNC."-".PREC."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
  if (GEN>-1) { set title PRECISION.GEMX." - ".NAME."(+) vs. ".NAMEB."(-)" }
  stats "< paste ".BASENAME.".txt ".GEMX."-".FUNCB."-".PREC.".txt" using ((column(FLOPS)-column(FLOPS+SHIFT))*NCORES) nooutput
  set palette defined ( -1 "blue", 0 "white", 1 "red" )
  set pm3d interpolate 0, 0
  set autoscale fix
  if (0<HIM) { set xrange [*:HIM] }
  if (0<HIN) { set yrange [*:HIN] }
  if (0<HIK) { set zrange [*:HIK] }
  set xlabel "M"
  set ylabel "N"
  set zlabel "K"
  set ticslevel 0
  set cblabel "GFLOP/s" offset 0.5
  set format x "%g"; set format y "%g"; set format z "%g"; set format cb "%+g"
  set cbrange [-MAX(abs(STATS_min), abs(STATS_max)):MAX(abs(STATS_min), abs(STATS_max))]
  splot "< paste ".BASENAME.".txt ".GEMX."-".FUNCB."-".PREC.".txt" using MPARM:NPARM:KPARM:((column(FLOPS)-column(FLOPS+SHIFT))*NCORES) notitle with points pointtype 7 linetype palette
}

reset
if (GEN<=0) { set output GEMX."-".FUNC."-".PREC."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
if (GEN>-1) { set title PRECISION.GEMX."/".NAME." - Performance" }
set pm3d interpolate 0, 0
#set colorbox horizontal user origin 0, 0.1 size 1, 0.1
set autoscale fix
if (0<HIM) { set xrange [*:HIM] }
if (0<HIN) { set yrange [*:HIN] }
if (0<HIK) { set zrange [*:HIK] }
set xlabel "M"
set ylabel "N"
set zlabel "K"
set ticslevel 0
set cblabel "GFLOP/s" offset 1.0
set format x "%g"; set format y "%g"; set format z "%g"; set format cb "%g"
splot BASENAME.".txt" using MPARM:NPARM:KPARM:(column(FLOPS)*NCORES) notitle with points pointtype 7 linetype palette

reset
if (GEN<=0) { set output GEMX."-".FUNC."-".PREC."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
if (GEN>-1) { set title PRECISION.GEMX."/".NAME." - Performance (K-Average)" }
set dgrid3d #9, 9
set pm3d interpolate 0, 0 map
if (0<HIM) { set xrange [*:HIM] }
if (0<HIN) { set yrange [*:HIN] }
set autoscale fix
set xlabel "M"
set ylabel "N" offset -1.0
set cblabel "GFLOP/s" offset 1.0
set format x "%g"; set format y "%g"; set format cb "%g"
set mxtics 2
#set offsets 1, 1, 1, 1
splot "gemx-avg.txt" using (("".strcol(3)."" eq "i")?(I1($1, XN)):(1/0)):(("".strcol(3)."" eq "i")?(J1($1, XN)):(1/0)):2 notitle with pm3d

reset
if (GEN<=0) { set output GEMX."-".FUNC."-".PREC."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
if (GEN>-1) { set title PRECISION.GEMX."/".NAME." - Performance (CDF)" }
set xlabel "Probability\n\nGeo. Mean: ".GEO." GFLOP/s  Median: ".MED." GFLOP/s"
set ylabel "GFLOP/s"
set format x "%g%%"
set format y "%g"
set fit quiet
f(x) = b * x + a
fit f(x) "gemx-cdf.txt" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via a, b
g(x) = (x - a) / b
x = 0.5 * (100 + MAX(0, g(0)))
h(x) = d * x + c
fit [x-2:x+2] h(x) "gemx-cdf.txt" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via c, d
set arrow 1 from x, h(x) to x, 0
set label 1 sprintf("%.1f%%", x) at x, 0.5 * h(x) left offset 1
set arrow 2 from x, h(x) to 0, h(x)
set label 2 sprintf("%.1f GFLOP/s", h(x)) at 0.5 * x, h(x) centre offset 0, 1
set autoscale fix
set xrange [0:100]
set yrange [0:*]
plot "gemx-cdf.txt" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 notitle with lines

if (0 < PEAK) {
  reset
  if (GEN<=0) { set output GEMX."-".FUNC."-".PREC."-".FILECOUNT.".".EXT; FILECOUNT = FILECOUNT + 1 }
  if (GEN>-1) { set title PRECISION.GEMX."/".NAME." - Performance (M = N = ".MN.")" }
  set xlabel "K"
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
  if (exists("FUNCB") && (FUNCB ne "") && exists("NAMEB") && (NAMEB ne "")) {
    plot BASENAME.".txt" using KPARM:((MN==column(MPARM)&&MN==column(NPARM))?(100.0*column(FLOPS)*NCORES/PEAK):(1/0)) title NAME smooth sbezier with points pointtype 7 pointsize 0.5, \
         GEMX."-".FUNCB."-".PREC.".txt" using KPARM:((MN==column(MPARM)&&MN==column(NPARM))?(100.0*column(FLOPS)*NCORES/PEAK):(1/0)) title NAMEB smooth sbezier with points pointtype 7 pointsize 0.5
  } else {
    plot BASENAME.".txt" using KPARM:((MN==column(MPARM)&&MN==column(NPARM))?(100.0*column(FLOPS)*NCORES/PEAK):(1/0)) notitle smooth sbezier with points pointtype 7 pointsize 0.5
  }
}
