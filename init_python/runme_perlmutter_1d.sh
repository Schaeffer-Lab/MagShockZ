#!/bin/bash
# Python-init 1D OSIRIS setup -- first point of the parameter scan.
# Every parameter is specified explicitly here: this file is the single source of
# truth for the run (FLASH_OSIRIS_define.py has no hidden defaults). A copy of these
# values is also written to run_manifest.yaml in the output directory.

conda activate osiris2

python FLASH_OSIRIS_define.py \
--data_path /pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_plt_cnt_0009 \
--dim 1 \
--inputfile_name magshockz_rqm100_dx0.3_ppc500_g20 \
--reference_density 5e18 \
--rqm_factor 100 \
--dx 0.3 \
--ppc 500 \
--tmax_gyroperiods 20 \
--algorithm cuda \
--start_point 0 0.07 0 \
--end_point 0 0.70 0 \
--node_number 4 \
--num_threads 1 \
--n_dump_total 512 \
--restart false \
--vpml_bnd_size 100 \
--emf_boundary pmc vpml \
--part_boundary thermal thermal \
--interpolation cubic \
--smooth_type binomial \
--smooth_order 2 \
--n_ave 6 \
--emf_reports b1 b2 b3 e1 e2 e3 \
--reports charge j1 j2 j3 q1 q2 q3 \
--rep_udist uth1 uth2 ufl1 ufl2 \
--phasespaces p1x1 p2x1 p3x1 gx1 \
--e_ps_pmin -1 -1 -0.5 \
--e_ps_pmax 1 1 0.5 \
--i_ps_pmin -0.1 -0.1 -0.05 \
--i_ps_pmax 0.1 0.1 0.05 \
--ps_np 4096 4096 4096 \
--ps_ngamma 128 \
--ps_gammamax 3.0 \
--ps_nx 1024 \
--ps_ny 512
