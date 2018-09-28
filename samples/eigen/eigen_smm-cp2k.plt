MPARM = 1
NPARM = 2
KPARM = 3
FLOPS = 5

HIM = -1
HIN = HIM
HIK = HIM

BASENAME = "eigen_smm"
FILENAME = system("sh -c \"echo ${FILENAME}\"")
if (FILENAME eq "") {
  FILENAME = BASENAME."-cp2k.pdf"
}

FILECOUNT = 1 # initial file number
# MULTI =-1: multiple files; no titles
# MULTI = 0: multiple files with titles
# MULTI = 1: single file with titles
MULTI = system("sh -c \"echo ${MULTI}\"")
if (MULTI eq "") {
  MULTI = 1
}

XFLOPS(M, N, K) = 2.0 * M * N * K
NFLOPS(M, N, K) = XFLOPS(column(M), column(N), column(K))
NBYTES(M, N, K, ELEMSIZE) = ELEMSIZE * (column(M) * column(K) + column(K) * column(N) + column(M) * column(N))
AI(M, N, K, ELEMSIZE) = NFLOPS(M, N, K) / NBYTES(M, N, K, ELEMSIZE)

TIME(M, N, K, F) = NFLOPS(M, N, K) * 1E-9 / column(F)
BW(M, N, K, F, ELEMSIZE) = (column(M) * column(K) + column(K) * column(N)) * ELEMSIZE / (TIME(M, N, K, F) * 1024 * 1024 * 1024)

stats BASENAME."-cp2k.dat" using (column(MPARM)*column(NPARM)*column(KPARM)) nooutput; MNK = STATS_stddev**(1.0/3.0); MAXMNK = int(STATS_max)
stats BASENAME."-cp2k.dat" using (log(column(FLOPS))) nooutput; NSAMPLES = STATS_records; GEOFLOPS = exp(STATS_sum/STATS_records)
stats BASENAME."-cp2k.dat" using FLOPS nooutput; MEDFLOPS = STATS_median; AVGFLOPS = STATS_mean; MINFLOPS = STATS_min; MAXFLOPS = STATS_max
stats BASENAME."-cp2k.dat" using NPARM nooutput; XN = int(STATS_max)
stats BASENAME."-cp2k.dat" using ((NFLOPS(MPARM,NPARM,KPARM)<=XFLOPS(13,13,13))?column(FLOPS):1/0) nooutput; BIN1_FLOPS = STATS_mean; BIN1_NSAMPLES = STATS_records
stats BASENAME."-cp2k.dat" using (((XFLOPS(13,13,13)<NFLOPS(MPARM,NPARM,KPARM))&&(NFLOPS(MPARM,NPARM,KPARM)<=XFLOPS(23,23,23)))?column(FLOPS):1/0) nooutput; BIN2_FLOPS = STATS_mean; BIN2_NSAMPLES = STATS_records
stats BASENAME."-cp2k.dat" using ((XFLOPS(23,23,23)<NFLOPS(MPARM,NPARM,KPARM))?column(FLOPS):1/0) nooutput; BIN3_FLOPS = STATS_mean; BIN3_NSAMPLES = STATS_records
stats BASENAME."-cp2k.dat" using ((NFLOPS(MPARM,NPARM,KPARM)<=XFLOPS(13,13,13))?BW(MPARM,NPARM,KPARM,FLOPS,8):1/0) nooutput; BIN1_MEMBW = STATS_mean
stats BASENAME."-cp2k.dat" using (((XFLOPS(13,13,13)<NFLOPS(MPARM,NPARM,KPARM))&&(NFLOPS(MPARM,NPARM,KPARM)<=XFLOPS(23,23,23)))?BW(MPARM,NPARM,KPARM,FLOPS,8):1/0) nooutput; BIN2_MEMBW = STATS_mean
stats BASENAME."-cp2k.dat" using ((XFLOPS(23,23,23)<NFLOPS(MPARM,NPARM,KPARM))?BW(MPARM,NPARM,KPARM,FLOPS,8):1/0) nooutput; BIN3_MEMBW = STATS_mean
stats BASENAME."-cp2k.dat" using (log(AI(MPARM,NPARM,KPARM,8))) nooutput; GEOAI = exp(STATS_sum/STATS_records)
stats BASENAME."-cp2k.dat" using (AI(MPARM,NPARM,KPARM,8)) nooutput; MEDAI = STATS_median; AVGAI = STATS_mean; MINAI = STATS_min; MAXAI = STATS_max
stats BASENAME."-cp2k.dat" using (log(BW(MPARM,NPARM,KPARM,FLOPS,8))) nooutput; GEOMEMBW = exp(STATS_sum/STATS_records)
stats BASENAME."-cp2k.dat" using (BW(MPARM,NPARM,KPARM,FLOPS,8)) nooutput; MEDMEMBW = STATS_median; AVGMEMBW = STATS_mean; MINMEMBW = STATS_min; MAXMEMBW = STATS_max

FORMAT(X) = sprintf("%%.%if", ceil(1.0 / MAX(log10(X), 1) - 1.0))
MAX(A, B) = A < B ? B : A

IX(I1, J1, NJ) = int(MAX(I1 - 1, 0) * NJ + MAX(J1 - 1, 0))
I1(IX, NJ) = int(IX / NJ) + 1
J1(IX, NJ) = int(IX) % NJ + 1

set table BASENAME."-plot-avg.dat"
plot BASENAME."-cp2k.dat" using (IX(column(MPARM), column(NPARM), XN)):FLOPS smooth unique
unset table
set table BASENAME."-plot-mbw.dat"
plot BASENAME."-cp2k.dat" using (BW(MPARM,NPARM,KPARM,FLOPS,8)):(1.0) smooth cumulative
unset table
set table BASENAME."-plot-cdf.dat"
plot BASENAME."-cp2k.dat" using FLOPS:(1.0) smooth cumulative
unset table
stats BASENAME."-plot-cdf.dat" using (("".strcol(3)."" eq "i")?($2):(1/0)) nooutput; FREQSUM = STATS_max; FREQN = STATS_records

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
if (MULTI>-1) { set title "Performance (Selected Kernels)" }
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
splot BASENAME."-cp2k.dat" using MPARM:NPARM:KPARM:FLOPS notitle with points pointtype 7 linetype palette

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Performance (K-Average for ".sprintf("%u Kernels", NSAMPLES).")" }
set origin -0.02, 0
set dgrid3d #9, 9
set pm3d interpolate 0, 0 map
set autoscale fix
set xlabel "M"
set ylabel "N" offset -1.5
set cblabel "GFLOP/s" offset 0.5
set format x "%g"; set format y "%g"; set format cb "%g"
set mxtics 2
splot BASENAME."-plot-avg.dat" using (("".strcol(3)."" eq "i")?(I1($1, XN)):(1/0)):(("".strcol(3)."" eq "i")?(J1($1, XN)):(1/0)):2 notitle with pm3d

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
set yrange [0:*]
set autoscale fix
set label sprintf("{/=9 ".FORMAT(BIN1_FLOPS)." GFLOP/s}", BIN1_FLOPS) at 0.0, BIN1_FLOPS centre offset 0, -1 front
set label sprintf("{/=9 ".FORMAT(BIN2_FLOPS)." GFLOP/s}", BIN2_FLOPS) at 1.0, BIN2_FLOPS centre offset 0, -1 front
set label sprintf("{/=9 ".FORMAT(BIN3_FLOPS)." GFLOP/s}", BIN3_FLOPS) at 2.0, BIN3_FLOPS centre offset 0, -1 front
set label sprintf("{/=9 (".FORMAT(BIN1_MEMBW)." GB/s)}", BIN1_MEMBW) at 0.0, BIN1_FLOPS centre offset 0, -2 front
set label sprintf("{/=9 (".FORMAT(BIN2_MEMBW)." GB/s)}", BIN2_MEMBW) at 1.0, BIN2_FLOPS centre offset 0, -2 front
set label sprintf("{/=9 (".FORMAT(BIN3_MEMBW)." GB/s)}", BIN3_MEMBW) at 2.0, BIN3_FLOPS centre offset 0, -2 front
set label sprintf("{/=9 N=%u}", BIN1_NSAMPLES) at 0.0, 0.0 centre offset 0, 0.5 front
set label sprintf("{/=9 N=%u}", BIN2_NSAMPLES) at 1.0, 0.0 centre offset 0, 0.5 front
set label sprintf("{/=9 N=%u}", BIN3_NSAMPLES) at 2.0, 0.0 centre offset 0, 0.5 front
plot BASENAME."-cp2k.dat" \
      using (0.0):(BIN1_FLOPS) notitle smooth unique with boxes linetype 1 linecolor "grey", \
  ""  using (1.0):(BIN2_FLOPS) notitle smooth unique with boxes linetype 1 linecolor "grey", \
  ""  using (2.0):(BIN3_FLOPS) notitle smooth unique with boxes linetype 1 linecolor "grey"

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Cummulative Performance Distribution (CDF for ".sprintf("%u Kernels", NSAMPLES).")" }
set xlabel "Probability\n\n{/=9 Min.: ".sprintf(FORMAT(MINFLOPS), MINFLOPS)." GFLOP/s   Geo.: ".sprintf(FORMAT(GEOFLOPS), GEOFLOPS)." GFLOP/s   Med.: ".sprintf(FORMAT(MEDFLOPS), MEDFLOPS)." GFLOP/s   Avg.: ".sprintf(FORMAT(AVGFLOPS), AVGFLOPS)." GFLOP/s   Max.: ".sprintf(FORMAT(MAXFLOPS), MAXFLOPS)." GFLOP/s}"
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
fit f(x) BASENAME."-plot-cdf.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via a, b
g(x) = (x - a) / b
x50 = 0.5 * (100 + MAX(0, g(0)))
h(x) = d * x + c
dx = 100.0 / FREQN
fit [x50-2.0*dx:x50+2.0*dx] h(x) BASENAME."-plot-cdf.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 via c, d
set arrow from x50, second h(x50) to x50, second 0 front
set arrow from x50, second h(x50) to 100, second h(x50) front
set label sprintf("%.0f%%", x50) at x50, second 0.5 * h(x50) left offset 1 front
set label sprintf(FORMAT(h(x50))." GFLOP/s", h(x50)) at 0.5 * (x50 + 100.0), second h(x50) centre offset 0, -1 front
set key left invert
plot BASENAME."-plot-mbw.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 axes x1y1 title "Memory Bandwidth" with lines linecolor "grey", \
     BASENAME."-plot-cdf.dat" using (("".strcol(3)."" eq "i")?(100*$2/FREQSUM):(1/0)):1 axes x1y2 title "Compute Performance" with lines linewidth 2

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Arithmetic Intensity (".sprintf("%u Kernels", NSAMPLES).")" }
set grid x y2 linecolor "grey"
set key left #spacing 0.5
set ytics format ""
set y2tics nomirror
set y2label "GFLOP/s"
#set xlabel "FLOPS/Byte\n\n{/=9 ".sprintf("N: %u", NSAMPLES)."   Min.: ".sprintf("%.1f", MINAI)."   Geo.: ".sprintf("%.1f", GEOAI)."   Med.: ".sprintf("%.1f", MEDAI)."   Avg.: ".sprintf("%.1f", AVGAI)."   Max.: ".sprintf("%.1f", MAXAI)."}"
set xlabel "FLOPS/Byte   (Min.: ".sprintf("%.1f", MINAI)."   Geo.: ".sprintf("%.1f", GEOAI)."   Med.: ".sprintf("%.1f", MEDAI)."   Avg.: ".sprintf("%.1f", AVGAI)."   Max.: ".sprintf("%.1f", MAXAI).")"
set yrange [0:*]
set autoscale fix
plot BASENAME."-cp2k.dat" using (AI(MPARM,NPARM,KPARM,8)):FLOPS notitle smooth sbezier with lines linecolor "grey" linewidth 2, \
                       "" using (AI(MPARM,NPARM,KPARM,8)):FLOPS notitle smooth unique with points pointtype 7 pointsize 0.1

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Memory Bandwidth Consumption (".sprintf("%u Kernels", NSAMPLES).")" }
set grid x y2 linecolor "grey"
set key left #spacing 0.5
set ytics format ""
set y2tics nomirror
set y2label "GB/s"
set xlabel "Problem Size (MNK^{1/3})\n\n{/=9 Min.: ".sprintf("%.0f GB/s", MINMEMBW)."   Geo.: ".sprintf("%.0f GB/s", GEOMEMBW)."   Med.: ".sprintf("%.0f GB/s", MEDMEMBW)."   Avg.: ".sprintf("%.0f GB/s", AVGMEMBW)."   Max.: ".sprintf("%.0f GB/s", MAXMEMBW)."}"
set yrange [0:*]
set autoscale fix
plot BASENAME."-cp2k.dat" using ((column(MPARM)*column(NPARM)*column(KPARM))**(1.0/3.0)):(BW(MPARM,NPARM,KPARM,FLOPS,8)) notitle smooth sbezier with lines linecolor "grey" linewidth 2, \
                       "" using ((column(MPARM)*column(NPARM)*column(KPARM))**(1.0/3.0)):(BW(MPARM,NPARM,KPARM,FLOPS,8)) notitle with points pointtype 7 pointsize 0.1

reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Compute Consumption (".sprintf("%u Kernels", NSAMPLES).")" }
set grid x y2 linecolor "grey"
set key left #spacing 0.5
set ytics format ""
set y2tics nomirror
set y2label "GFLOP/s"
set xlabel "Problem Size (MNK^{1/3})\n\n{/=9 Min.: ".sprintf(FORMAT(MINFLOPS), MINFLOPS)." GFLOP/s   Geo.: ".sprintf(FORMAT(GEOFLOPS), GEOFLOPS)." GFLOP/s   Med.: ".sprintf(FORMAT(MEDFLOPS), MEDFLOPS)." GFLOP/s   Avg.: ".sprintf(FORMAT(AVGFLOPS), AVGFLOPS)." GFLOP/s   Max.: ".sprintf(FORMAT(MAXFLOPS), MAXFLOPS)." GFLOP/s}"
set yrange [0:*]
set autoscale fix
plot BASENAME."-cp2k.dat" using ((column(MPARM)*column(NPARM)*column(KPARM))**(1.0/3.0)):FLOPS notitle smooth sbezier with lines linecolor "grey" linewidth 2, \
                       "" using ((column(MPARM)*column(NPARM)*column(KPARM))**(1.0/3.0)):FLOPS notitle with points pointtype 7 pointsize 0.1

if (0!=system("sh -c \"if [ -e eigen_smm-plot-join.dat ]; then echo 1; else echo 0; fi\"")) {
reset
if (MULTI<=0) { set output "".FILECOUNT."-".FILENAME; FILECOUNT = FILECOUNT + 1 }
if (MULTI>-1) { set title "Performance (Selected Kernels)" }
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
set yrange [0:*]
plot BASENAME."-plot-join.dat" using FLOPS:xtic("(".strcol(MPARM).",".strcol(NPARM).",".strcol(KPARM).")") notitle
}
