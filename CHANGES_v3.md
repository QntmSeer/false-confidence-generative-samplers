# Changes from v2 (ChemRxiv 10.26434/chemrxiv.15011779/v2)

Short version for the ChemRxiv "changes in this version" field:

> Corrects the description of the higher-dimensional flow experiments: two run on analytic
> surrogate landscapes and one on the real LJ38 potential restricted to two basins, none on a
> biomolecular force field. Reports the alanine 3D control, which also miscovers; takes the alanine
> 2D results in Table 2 and all coverage intervals from the stored data; withdraws Pareto k-hat
> values that were not computed from the flows' own weights; corrects the summary of the LEAPS
> audit; and marks literature populations as unverified. The spin-glass results, the proposition,
> the exact toy and the measured MD statistics are unchanged.

## What v2 got right

The experiments themselves are genuine. Every flow in Table 2 was trained, and its ESS, coverage,
error and reverse gap were computed from its own samples against a reference that is exact for the
landscape it runs on (quadrature, Gaussian integrals, or MALA+BAR for the LJ38 basins). The
spin-glass numbers, the toy, and the MD statistics come from stored data. The problems are in how
some results were described and in a few values that were typed in or not measured.

## Details

1. **Model landscapes described correctly.** v2 called the 66D, 114D and 414D experiments
   "all-atom" molecular flows. Each label was checked by recomputing the stored reference values:
   the alanine models use an analytic Ramachandran surrogate (three Gaussian wells and a barrier)
   with harmonic covalent terms (stored exact Z, F and C7ax mass reproduced to four digits from that
   formula); the "chignolin" model is two Gaussian wells in 408 dimensions with the native mass set
   to 68% by hand (stored exact F reproduced to 1e-5 kcal/mol from the Gaussian formula); the LJ38
   model uses the real Lennard-Jones potential (stored minimum energies reproduced exactly) with the
   partition function estimated for two basins only, at T = 0.012. Amber14SB was evaluated once as a
   check and does not enter the sampling.
2. **Alanine 3D control reported.** The two-flow mixture that does sample C7ax also miscovers (1.5%,
   error 0.094 kcal/mol) and has a larger reverse gap (10.33 nats) than the flow that drops the basin
   (7.65 nats). v2 omitted this and stated that the reverse check "unmasks every omission". The
   unsupported thresholds ("G_rev > 2 nats signals distortion, > 5 nats exposes mode omission") are
   removed.
3. **Alanine 2D rows of Table 2.** v2 typed ESS 81%, coverage 15%, error 0.06 kcal/mol and
   G_rev 1.7 nats, which disagree with `data/alanine_flow_audit.json` (79%, 10%, 0.122, 11.66) and
   with the v2 abstract. The table is now generated from the JSON. The v2 "Bi-modal Twin Control" row
   had no stored data and is removed.
4. **Pareto k-hat withdrawn where it was not measured.** In `scripts/eval_3d_equivariant_and_khat.py`
   (v2), the LJ38 RealNVP values were the SE(3) flow's values times 0.98 and 0.99, the chignolin
   values were computed on synthetic normal draws, and the alanine 3D median of exactly 0.00 over 200 runs means the
   estimator failed (it returns exactly 0.0 only on its failure paths; on log-normal weights with the
   same ESS it never does), so it is not a measurement. These are set to null in `data/molecular_3d_flow_audit.json`
   (the old values are kept under `v3_khat_withdrawn`), the script no longer writes them, and the
   estimator now returns NaN instead of 0.0 when it cannot fit. Only the SE(3) values remain.
5. **Literature populations marked UNVERIFIED.** The 68% (chignolin), 82% (Trp-cage) and >90%
   (icosahedral LJ38) values are typed in and have not been checked against their sources. The cited
   chignolin study concerns CLN025, not the GYDPETGTWG (1UAO) peptide simulated here, and Honda et al.
   2008 is a crystal structure, not an equilibrium measurement. These values no longer appear in the
   abstract or conclusion.
6. **Coverage intervals and spin-glass details.** The nine coverage intervals in Section 5 were
   typed and matched neither Clopper-Pearson intervals from the stored counts nor the stored
   intervals (one masked-diffusion interval was the GFlowNet's); they are now generated from the
   counts. Exact log Z comes from a Rust Gray-code enumerator, not GPU enumeration.
7. **External audits.** v2 said the ESS was "> 80%" in both the MDNS and LEAPS audits; the stored
   LEAPS results give 0.82 on Ising but 0.31 on Potts. The MDNS bias is in the released estimator
   code, not the checkpoint.
8. **Chignolin 1 us run.** v2 said the trajectory leaves the native basin within 60 ns; it leaves after
   1.5 ns and returns intermittently (11% and 10% native in the 700 to 800 and 900 to 1000 ns blocks; last
   native frame at 960 ns), all computed from the stored per-frame RMSD. The
   cited chignolin literature study is confirmed by a secondary source (McKiernan et al., J. Chem.
   Phys. 147, 104107, 2017) to concern CLN025 with CHARMM22* at 340 K.
9. **Smaller fixes.** Block lengths (100 ns for chignolin, 30 ns for Trp-cage); claims softened where they held only while a basin stays unvisited, or where a mechanism was untested; Limitations updated
   for the 1 us chignolin run, which returns to the native basin intermittently; the unstored G_rev
   decomposition ("88% of the gap") removed; the controls described as fixed-weight mixtures of two
   separately trained flows; the dashboard figure redrawn from stored values only (the Pareto k-hat
   panel and the typed alanine 2D control point are gone); Table 2 has no em dashes. In the chignolin and LJ38 figures the structure drawings are now framed
   insets over empty regions (in v2 they sat unframed on the axes and read as data), and the
   v2 label "misfolded globule (RMSD = 5.2 A)" is corrected: that frame is 3.4 A from native after
   optimal superposition, now computed from `data/chignolin_ca_conformations.json`.

All numbers that were already generated from stored data are unchanged.
