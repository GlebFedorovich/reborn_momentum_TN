#!/bin/bash
# condor_run_phase_scan.sh — wrapper Condor's `executable` invokes directly.
# Runs phase_scan_vmc_mpi.jl with K replica workers, all within ONE Condor
# job slot (vanilla universe, request_cpus = K+1) — no cross-machine MPI
# setup needed since every rank lives on the single machine Condor assigns
# this job. OPENBLAS_NUM_THREADS=1 is already baked into the .jl scripts via
# BLAS.set_num_threads(1), but set here too as a harmless belt-and-braces.
#
# $1 = number of MPI ranks (K+1 = 1 controller + K replicas), passed from
# the submit file so request_cpus and -n stay in sync in one place.

set -euo pipefail
export OPENBLAS_NUM_THREADS=1

N_RANKS="${1:?usage: condor_run_phase_scan.sh <n_ranks>}"

cd "$(dirname "$0")/../.."   # repo root (this script lives at grouped_ansatz/runs/) — adjust if the repo isn't shipped/mounted at a fixed path on the execute node

~/.julia/bin/mpiexecjl --project=. -n "$N_RANKS" julia grouped_ansatz/runs/phase_scan_vmc_mpi.jl
