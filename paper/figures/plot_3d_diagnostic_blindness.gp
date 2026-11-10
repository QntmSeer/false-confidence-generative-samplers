#!/usr/bin/env gnuplot
# ==============================================================================
# The Diagnostic Blindness Dashboard (3-Panel Publication Figure)
# System: Alanine (2D/66D), LJ38 (114D RealNVP & SE(3)), Chignolin (414D)
# Palette: Theme A ("Typewriter Lab") - Rust / Jade / Ink / Grid
# ==============================================================================

set terminal pdfcairo enhanced color font "DejaVu Sans Mono,8" size 10.5in,3.2in
set output "fig_3d_diagnostic_blindness.pdf"

# Theme Colors
Rust  = "#c94f27"   # Trapped flows (mode omission)
Jade  = "#069a8e"   # Paired twin controls (full support)
Ink   = "#2a2723"   # Primary text & axes
Ink2  = "#6b655c"   # Subtitles & reference text
Grid  = "#e2dacb"   # Dotted grid lines
Paper = "#ffffff"   # Background

set style fill solid 1.0
set linetype 1 lc rgb Rust  pt 5 ps 1.1 lw 1.2   # Trapped: closed square
set linetype 2 lc rgb Jade  pt 7 ps 1.0 lw 1.2   # Twin: closed circle
set linetype 3 lc rgb Ink2  dt 2 lw 1.0           # Reference dashed lines
set linetype 4 lc rgb Grid  dt 3 lw 0.8           # Subtle grid lines

# Embedded Data: Trapped Unimodal Flows
# Columns: Name, ESS(%), G_rev(nats), k_hat, Coverage(%), G_fwd(nats)
$data_trapped << EOD
"Ala 2D"         81.0     1.70    0.00    15.0    0.030
"Ala 66D"        91.4     7.65    0.00     2.5    0.010
"LJ38 RealNVP"   45.4 58724.00   -0.90     0.0    0.340
"LJ38 SE(3)"     44.6 65059.00   -0.92     0.0    0.350
"Chig 414D"      83.3 32434.00   -0.72     0.0    0.090
EOD

# Embedded Data: Paired Mode-Covering Twin Controls
# Columns: Name, ESS(%), G_rev(nats), k_hat, Coverage(%), G_fwd(nats)
$data_twins << EOD
"Ala 2D Twin"    81.0     0.01    0.00    95.0    0.030
"Ala 66D Twin"   86.1    10.33    0.00     1.5    0.011
"LJ38 Twin"      24.2     1.05   -0.93    97.0   27.500
"LJ38 SE3 Twin"  23.5     1.04   -0.94    97.0   27.400
"Chig Twin"      83.2     0.10   -0.71    95.0    0.090
EOD

set multiplot layout 1,3 margins 0.08,0.96,0.18,0.86 spacing 0.09

# ------------------------------------------------------------------------------
# PANEL 1: High ESS masks massive G_rev
# ------------------------------------------------------------------------------
set title "{/:Bold High ESS masks massive G_{rev}}\n{/*0.85 3D molecular flows; squares: trapped; circles: twin}" \
    textcolor rgb Ink offset 0,0.5 left at graph 0,1.06

set xlabel "Effective Sample Size ESS (%)" textcolor rgb Ink offset 0,0.2
set ylabel "Reverse check G_{rev} (nats)" textcolor rgb Ink offset 1.0,0
set logscale y 10
set xrange [10:105]
set yrange [0.005:400000]
set format y "10^{%T}"
set ytics (0.01, 0.1, 1, 10, 100, 1000, 10000, 100000)
set grid ytics ls 4
unset grid xtics
set border 1 lc rgb Ink2 lw 1.0

# Reference line at G_rev = 2 nats
set arrow 1 from 10,2 to 105,2 nohead ls 3
set label 1 "G_{rev} threshold (2 nats)" at 14,2.8 textcolor rgb Ink2 font ",7"

# Leader lines for Panel 1
set arrow 2 from 45.4,58724 to 34,140000 head lc rgb Ink2 lw 0.6
set label 2 "LJ38 RealNVP" at 33,150000 right textcolor rgb Ink font ",7.5"

set arrow 3 from 44.6,65059 to 54,160000 head lc rgb Ink2 lw 0.6
set label 3 "LJ38 SE(3)" at 55,160000 left textcolor rgb Ink font ",7.5"

set arrow 4 from 83.3,32434 to 73,18000 head lc rgb Ink2 lw 0.6
set label 4 "Chig 414D" at 72,17000 right textcolor rgb Ink font ",7.5"

set label 5 "Ala 2D"  at 88,1.4 left textcolor rgb Ink font ",7.5"
set label 6 "Ala 66D" at 94,8.5 left textcolor rgb Ink font ",7.5"

plot $data_twins   using 2:3 with points ls 2 notitle, \
     $data_trapped using 2:3 with points ls 1 notitle

unset arrow 1; unset arrow 2; unset arrow 3; unset arrow 4
unset label 1; unset label 2; unset label 3; unset label 4; unset label 5; unset label 6

# ------------------------------------------------------------------------------
# PANEL 2: Pareto k_hat < 0.5 fails to detect 0% coverage
# ------------------------------------------------------------------------------
set title "{/:Bold k_{hat} < 0.5 fails to detect 0% coverage}\n{/*0.85 shaded: certified reliable by standard PSIS; dashed: nominal 95%}" \
    textcolor rgb Ink offset 0,0.5 left at graph 0,1.06

set xlabel "Pareto tail index k_{hat} (PSIS)" textcolor rgb Ink offset 0,0.2
set ylabel "95% CI Coverage (%)" textcolor rgb Ink offset 1.0,0
unset logscale y
set format y "%g"
set xrange [-1.2:0.75]
set yrange [-18:112]
set ytics 0,20,100
set grid ytics ls 4
set border 1 lc rgb Ink2 lw 1.0

# Shaded reliable zone k_hat < 0.5
set object 1 rect from -1.2,-18 to 0.5,112 fc rgb Grid fillstyle solid 0.55 behind
set label 1 "Reliable zone (k_{hat} < 0.5)" at -1.14,50 textcolor rgb Ink2 font ",7.5"

# Reference lines
set arrow 1 from 0.5,-18 to 0.5,112 nohead ls 3
set arrow 2 from -1.2,95 to 0.75,95 nohead ls 3
set label 2 "nominal 95%" at 0.52,95 left textcolor rgb Ink2 font ",7"

# Staggered leader lines at y = 0
set arrow 3 from -0.92,0 to -1.02,24 head lc rgb Ink2 lw 0.6
set label 3 "LJ38 SE(3)" at -1.03,25 right textcolor rgb Ink font ",7.5"

set arrow 4 from -0.90,0 to -0.90,-12 head lc rgb Ink2 lw 0.6
set label 4 "LJ38 RealNVP" at -0.90,-15 center textcolor rgb Ink font ",7.5"

set arrow 5 from -0.72,0 to -0.60,24 head lc rgb Ink2 lw 0.6
set label 5 "Chig 414D" at -0.59,25 left textcolor rgb Ink font ",7.5"

set label 6 "Ala 2D"  at 0.08,16 left textcolor rgb Ink font ",7.5"
set label 7 "Ala 66D" at 0.08,2  left textcolor rgb Ink font ",7.5"

set arrow 8 from -0.93,96 to -1.02,104 head lc rgb Jade lw 0.6
set label 8 "Twins (95-97%)" at -1.03,105 right textcolor rgb Jade font ",7.5"

plot $data_twins   using 4:5 with points ls 2 notitle, \
     $data_trapped using 4:5 with points ls 1 notitle

unset object 1
unset arrow 1; unset arrow 2; unset arrow 3; unset arrow 4; unset arrow 5; unset arrow 8
unset label 1; unset label 2; unset label 3; unset label 4; unset label 5; unset label 6; unset label 7; unset label 8

# ------------------------------------------------------------------------------
# PANEL 3: 5-order asymmetry unmasks omission
# ------------------------------------------------------------------------------
set title "{/:Bold 5-order asymmetry unmasks omission}\n{/*0.85 dashed: y = x; trapped flows lie 10^5 above diagonal}" \
    textcolor rgb Ink offset 0,0.5 left at graph 0,1.06

set xlabel "Forward Jensen gap G_{fwd} (nats)" textcolor rgb Ink offset 0,0.2
set ylabel "Reverse check G_{rev} (nats)" textcolor rgb Ink offset 1.0,0
set logscale x 10
set logscale y 10
set format x "10^{%T}"
set format y "10^{%T}"
set xrange [0.005:50]
set yrange [0.005:400000]
set xtics (0.01, 0.1, 1, 10)
set ytics (0.01, 0.1, 1, 10, 100, 1000, 10000, 100000)
set grid ytics ls 4
set border 1 lc rgb Ink2 lw 1.0

# Parity line y = x
set arrow 1 from 0.005,0.005 to 50,50 nohead ls 3
set label 1 "Bidirectional parity (y = x)" at 0.12,0.04 rotate by 33 textcolor rgb Ink2 font ",7"

# Leader lines for upper cluster
set arrow 2 from 0.09,32434 to 0.035,55000 head lc rgb Ink2 lw 0.6
set label 2 "Chig 414D" at 0.033,60000 right textcolor rgb Ink font ",7.5"

set arrow 3 from 0.34,58724 to 0.15,160000 head lc rgb Ink2 lw 0.6
set label 3 "LJ38 RealNVP" at 0.14,180000 right textcolor rgb Ink font ",7.5"

set arrow 4 from 0.35,65059 to 0.70,140000 head lc rgb Ink2 lw 0.6
set label 4 "LJ38 SE(3)" at 0.75,140000 left textcolor rgb Ink font ",7.5"

set arrow 5 from 0.01,7.65 to 0.006,16 head lc rgb Ink2 lw 0.6
set label 5 "Ala 66D" at 0.0058,18 right textcolor rgb Ink font ",7.5"

set label 6 "Ala 2D" at 0.05,1.2 left textcolor rgb Ink font ",7.5"

plot $data_twins   using 6:3 with points ls 2 notitle, \
     $data_trapped using 6:3 with points ls 1 notitle

unset multiplot
