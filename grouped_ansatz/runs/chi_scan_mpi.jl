# Load results later with, e.g.:
#   using TensorKit, JLD2
#   d = load("grouped_ansatz/output/chi_scan_mpi/chi_scan_L8_chi16.jld2")
#   d["ent_entropy_final"], d["opt_E"], d["mps_chain_final"], ...

using TensorKit, TensorOperations, LinearAlgebra, Printf, Statistics, Random
using MPSKit, Plots
using JSON, JLD2
using SparseArrays, KrylovKit

# Pin BLAS to 1 thread/process regardless of how this script is launched 
BLAS.set_num_threads(1)

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
const M = GroupedMomentumMPS

using Carlo, Carlo.JobTools
using MPI

MPI.Init()
const MC_COMM = MPI.COMM_WORLD
const N_RANKS = MPI.Comm_size(MC_COMM)
const IS_ROOT = MPI.Comm_rank(MC_COMM) == 0
# N_RANKS > 1 (launched via mpiexecjl -n K+1) => rank 0 is the Carlo
# controller, ranks 1..K each run their OWN independent run00xx copy of the
# SAME task (ranks_per_run left at its default, 1) and Carlo pools their raw
# accumulators via merge_results before fg() reads the merged results.json.
# With N_RANKS == 1 (plain `julia chi_scan_mpi.jl`) this falls back to
# ordinary single-chain SingleScheduler behavior, identical to chi_scan.jl.

StructUtils.lowerkey(::JSON.JSONStyle, s::ProductSector{Tuple{FermionParity, U1Irrep, ZNIrrep{N}}}) where {N} =
    string(s)

# ---------------------------------------------------------------------------
# SCAN SETTINGS
# ---------------------------------------------------------------------------
L        = 16
χ_list   = [8]
V_target = 0.5

folder_name = "grouped_ansatz/output/chi_scan_mpi"
IS_ROOT && mkpath(folder_name)
energy_plot_dir = folder_name * "/energy_convergence"
IS_ROOT && mkpath(energy_plot_dir)
gradnorm_plot_dir = folder_name * "/gradnorm_convergence"
IS_ROOT && mkpath(gradnorm_plot_dir)
MPI.Barrier(MC_COMM)

# ---------------------------------------------------------------------------
# Building blocks — identical logic to chi_scan.jl's fg / sr_step! /
# exact_gs_energy / compute_entropy, copied rather than shared so this file
# has no dependency on chi_scan.jl's current state. Only fg() differs
# (scheduler dispatch + root-only file cleanup).
# ---------------------------------------------------------------------------
function compute_entropy(ψ::FiniteMPS)
    ψ = normalize(ψ)
    entropy_list = Float64[]
    for b in 1:length(ψ)-1
        λ = entanglement_spectrum(ψ, b)
        λ_all = vcat(values(λ)...)
        s2 = sum(abs2, λ_all)
        isapprox(s2, 1; atol = 1e-6) || error(
            "compute_entropy bond $b: Σλ² = $s2 ≠ 1 — entanglement_spectrum convention mismatch")
        p = λ_all .^ 2
        p ./= sum(p)
        S = -sum(p .* log.(p .+ 1e-15))
        push!(entropy_list, S)
    end
    return entropy_list
end

function fg(mps_chain, zpairs, folder_name, id, maxiter, weight_in, config_start, L; V = 1.0, α = 1.0)
    tm = TaskMaker()
    tm.sweeps = Int(maxiter)
    tm.thermalization = Int(0)
    tm.binsize = Int(30)

    tm.mps_chain = mps_chain
    tm.L = L
    tm.zpairs = zpairs
    tm.config = copy(config_start)
    tm.t_hopping = 1.0
    tm.V_repulsion = V
    tm.idx_filled = findall(==(1), config_start)
    tm.idx_empty = findall(==(0), config_start)
    tm.weight = weight_in
    tm.alpha = α

    task(tm)

    job_path = folder_name * "/grouped_job_" * string(id)
    IS_ROOT && rm(job_path * ".data"; recursive = true, force = true)
    MPI.Barrier(MC_COMM)

    if N_RANKS > 1
        # ranks_per_run left at its default (1): each of the N_RANKS-1 worker
        # ranks runs its OWN independent run00xx copy of this single task
        # (Carlo splits the sweep budget among them; see
        # controller_react_idle/controller_react_busy in scheduler_mpi.jl),
        # rather than :all ranks cooperating on one run — the latter requires
        # a custom comm-aware Carlo.measure! override this MC type doesn't
        # have. merge_results pools the independent runs' accumulators once
        # the task is done.
        job = JobInfo(job_path, M.MC;
            run_time = "24:00:00",
            checkpoint_time = "12:00:00",
            tasks = make_tasks(tm),
        )
        start(Carlo.MPIScheduler, job)
    else
        job = JobInfo(job_path, M.MC;
            run_time = "24:00:00",
            checkpoint_time = "12:00:00",
            tasks = make_tasks(tm),
        )
        start(Carlo.SingleScheduler, job)
    end
    # start(MPIScheduler,...) has an internal barrier BEFORE the controller
    # writes/concatenates results.json, not after — worker ranks can return
    # from start() before rank 0 has finished writing the file. Every rank
    # calls fg() every SR step (SPMD) and needs the SAME merged dict to keep
    # its local mps_chain in sync, so wait here for rank 0 to actually finish.
    N_RANKS > 1 && MPI.Barrier(MC_COMM)

    stringdata = join(readlines(folder_name * "/grouped_job_" * string(id) * ".results.json"))
    dict = JSON.parse(stringdata; allownan = true)

    Wm = dict[end]["results"]["W"]["mean"]
    f = dict[end]["results"]["Energy"]["mean"] / Wm
    f_error = dict[end]["results"]["Energy"]["error"] / abs(Wm)

    return f, f_error, dict[end]
end

# Same iterative (KrylovKit eigsolve + linsolve) SR step as chi_scan.jl.
function sr_step!(mps_chain, dict_entry; η = 0.2, rcond = 1e-3)
    Wm      = dict_entry["results"]["W"]["mean"]
    E_mean  = dict_entry["results"]["Energy"]["mean"] / Wm
    O_mean  = Float64.(Iterators.flatten(dict_entry["results"]["O_flat"]["mean"]))  |> collect; O_mean ./= Wm
    OE_mean = Float64.(Iterators.flatten(dict_entry["results"]["OE_flat"]["mean"])) |> collect; OE_mean ./= Wm
    OO_flat = Float64.(Iterators.flatten(dict_entry["results"]["OO_flat"]["mean"])) |> collect; OO_flat ./= Wm
    N       = length(O_mean)
    S       = Symmetric(reshape(OO_flat, N, N) .- O_mean * O_mean')
    g       = OE_mean .- O_mean .* E_mean

    λs, _ = KrylovKit.eigsolve(S, randn(N), 1, :LR; ishermitian = true, tol = 1e-6, krylovdim = min(N, 20))
    λmax  = λs[1]
    shift = rcond * λmax
    δp, info = KrylovKit.linsolve(S + shift * I, g, zero(g);
                                   ishermitian = true, isposdef = true,
                                   tol = 1e-10, maxiter = 200)

    M.apply_update!(mps_chain, -η .* δp)
    return norm(g), info.converged
end

# Sparse, momentum-0-sector-restricted ED reference — identical to chi_scan.jl.
function exact_gs_energy(V, L)
    Nn = L ÷ 2
    k(i) = -π + π * (2i - 1) / L
    occ(s, i) = (s >> (i - 1)) & 1; cb(s, i) = count_ones(s & ((1 << (i - 1)) - 1))
    ann(s, i) = occ(s, i) == 0 ? (0, 0) : ((-1)^cb(s, i), s ⊻ (1 << (i - 1)))
    cre(s, i) = occ(s, i) == 1 ? (0, 0) : ((-1)^cb(s, i), s ⊻ (1 << (i - 1)))
    function ap4(s, a, b, c, d)
        x, st = ann(s, d); x == 0 && return (0, 0)
        y, st = ann(st, c); y == 0 && return (0, 0); x *= y
        y, st = cre(st, b); y == 0 && return (0, 0); x *= y
        y, st = cre(st, a); y == 0 && return (0, 0); x *= y
        (x, st)
    end
    states = [s for s in 0:(1 << L)-1 if count_ones(s) == Nn &&
              sum(i -> occ(s, i) == 1 ? (2i - 1) : 0, 1:L) % (2L) == 0]
    D = length(states)
    idxof = Dict(s => i for (i, s) in enumerate(states))
    I = Int[]; J = Int[]; W = Float64[]
    for (col, s) in enumerate(states)
        e = 0.0; for κ in 1:L; occ(s, κ) == 1 && (e += -2cos(k(κ))); end
        push!(I, col); push!(J, col); push!(W, e)
        for k1 in 1:L, k2 in 1:L, q in 0:L-1
            a = mod1(k1 + q, L); b = mod1(k2 - q, L); sg, st = ap4(s, a, b, k2, k1); sg == 0 && continue
            push!(I, idxof[st]); push!(J, col); push!(W, (1 / L) * V * cos(2π * q / L) * sg)
        end
    end
    H = sparse(I, J, W, D, D)
    vals, _ = eigsolve(H, randn(D), 1, :SR; ishermitian = true, tol = 1e-10, krylovdim = 40)
    return real(vals[1])
end

# ---------------------------------------------------------------------------
# SR hyperparameters — same recipe as chi_scan.jl, so results are directly
# comparable across the two files.
# ---------------------------------------------------------------------------
η0          = 0.2
η_min       = 0.05
n_opt_steps = 400
τ           = (n_opt_steps * 0.9) / (η0 / η_min - 1)
η_sched(step) = max(η_min, η0 / (1 + step / τ))
rcond       = 1e-5
α_temper    = 0.5
n_sweeps_lo = 1000
n_sweeps_hi = 2500
n_sweeps_sched(step) = round(Int, n_sweeps_lo + (n_sweeps_hi - n_sweeps_lo) * (step - 1) / (n_opt_steps - 1))

Egs_target = exact_gs_energy(V_target, L)
IS_ROOT && println("running with $N_RANKS MPI rank(s) ($(max(N_RANKS - 1, 1)) replica walker(s) per SR step)")
IS_ROOT && println("exact GS (L=$L, V=$V_target) = $(round(Egs_target, digits=6))")
IS_ROOT && println("χ scan: ", χ_list)

function momentum_frac_label(idx, L)
    k = M.idx_to_momentum(idx, L)
    num = round(Int, (k / π) * L)
    den = L
    g = gcd(abs(num), den); g = max(g, 1)
    num ÷= g; den ÷= g
    num == 0  && return "0"
    den == 1  && return num == 1 ? "π" : num == -1 ? "-π" : "$(num)π"
    return "$(num)π/$(den)"
end
Mn = L ÷ 2
site_ticks  = 1:Mn
site_labels = ["$(momentum_frac_label(s, L)),\n$(momentum_frac_label(s + L ÷ 2, L))" for s in site_ticks]

results = Dict{Int, NamedTuple}()

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
for χ in χ_list
    if IS_ROOT
        println("\n", "="^70)
        println("χ = $χ   (L=$L, V=$V_target)")
        println("="^70)
    end

    # get_start_fermi_sea draws from the GLOBAL default RNG (bare randn()),
    # and so does draw_from_B's sampling step (bare rand()) inside every
    # Carlo sweep — no RNG object is threaded through GroupedMomentumMPS.
    # Under ranks_per_run every MPI rank runs this entire script (SPMD) and
    # independently builds its own local mps_chain/job objects, so the K
    # replica workers only combine into a statistically valid pooled
    # estimate if they all sample the SAME wavefunction. Fix the seed here
    # (same value on every rank) so the initial mps_chain is bit-identical
    # across ranks, then immediately reseed per-rank so the actual MC
    # sampling decorrelates as intended.
    Random.seed!(hash((L, χ, V_target)))
    mps_chain, V_phys_list, V_left_list, zpairs = M.get_start_fermi_sea(L, χ; ε = 3.0)
    s_trial = M.fermi_sea_config(L)
    string_start = M.make_sample_string(s_trial, V_phys_list, V_left_list[1], zpairs, L)
    Random.seed!(hash((L, χ, V_target, MPI.Comm_rank(MC_COMM))))

    ent_entropy_initial = compute_entropy(FiniteMPS(mps_chain))

    opt_E     = Float64[]
    opt_Eerr  = Float64[]
    opt_gnorm = Float64[]
    opt_ess   = Float64[]
    opt_var   = Float64[]

    for step in 1:n_opt_steps
        η_t  = η_sched(step)
        n_sw = n_sweeps_sched(step)
        w = M.get_overlap(mps_chain, string_start)[1]
        Es, Es_err, de = fg(mps_chain, zpairs, folder_name, "chi$(χ)_step$(step)",
                             n_sw, w, s_trial, L; V = V_target, α = α_temper)
        gnorm, converged = sr_step!(mps_chain, de; η = η_t, rcond = rcond)

        Wm_v  = de["results"]["W"]["mean"]
        Wm2_v = de["results"]["W2"]["mean"]
        push!(opt_E, Es * L); push!(opt_Eerr, Es_err * L)
        push!(opt_gnorm, gnorm); push!(opt_ess, Wm_v^2 / Wm2_v)
        push!(opt_var, de["results"]["Energy2"]["mean"] / Wm_v - (de["results"]["Energy"]["mean"] / Wm_v)^2)

        if IS_ROOT && (step <= 3 || step % 20 == 0 || step == n_opt_steps)
            @printf("  step %3d (n_sw=%4d):  E·L = %9.5f ± %.1e   gap %+8.5f   ‖∇E‖=%.3e   ESS=%.3f   converged=%d\n",
                    step, n_sw, opt_E[end], Es_err * L, opt_E[end] - Egs_target, gnorm, opt_ess[end], converged)
        end

        # per-step Carlo job files are pure scratch once fg() has read them
        # into de — delete immediately (root only) so a long scan doesn't
        # pile up thousands of .data directories on disk.
        if IS_ROOT
            rm(folder_name * "/grouped_job_chi$(χ)_step$(step).data"; recursive = true, force = true)
            rm(folder_name * "/grouped_job_chi$(χ)_step$(step).results.json"; force = true)
        end
    end

    ent_entropy_final = compute_entropy(FiniteMPS(mps_chain))

    if IS_ROOT
        @printf("\n  χ=%d summary:  final E·L = %.5f   vs exact GS %.5f   (gap %+.5f)\n",
                χ, opt_E[end], Egs_target, opt_E[end] - Egs_target)

        plt_E_chi = scatter(1:n_opt_steps, opt_E, yerror = opt_Eerr, xlabel = "SR step", ylabel = "E·L",
                             title = "Energy convergence (L=$L, χ=$χ, V=$V_target)", label = "⟨E⟩·L",
                             markersize = 3, markerstrokewidth = 0.5, legend = :topright)
        plot!(plt_E_chi, 1:n_opt_steps, fill(Egs_target, n_opt_steps), ls = :dash, lw = 2, label = "exact GS")
        savefig(plt_E_chi, energy_plot_dir * "/chi_scan_energy_L$(L)_chi$(χ).png")
        println("  wrote $(energy_plot_dir)/chi_scan_energy_L$(L)_chi$(χ).png")

        plt_grad_chi = plot(1:n_opt_steps, opt_gnorm, xlabel = "SR step", ylabel = "‖∇E‖", yscale = :log10,
                             title = "Gradient norm convergence (L=$L, χ=$χ, V=$V_target)",
                             label = "‖∇E‖", lw = 1.5, legend = :topright)
        savefig(plt_grad_chi, gradnorm_plot_dir * "/chi_scan_gradnorm_L$(L)_chi$(χ).png")
        println("  wrote $(gradnorm_plot_dir)/chi_scan_gradnorm_L$(L)_chi$(χ).png")

        results[χ] = (opt_gnorm = copy(opt_gnorm), ent_entropy_final = copy(ent_entropy_final),
                      final_E = opt_E[end], final_Eerr = opt_Eerr[end], gap = opt_E[end] - Egs_target,
                      final_var = opt_var[end], final_ess = opt_ess[end])

        outfile = folder_name * "/chi_scan_L$(L)_chi$(χ).jld2"
        jldsave(outfile;
            L, χ, V_target, Egs_target,
            opt_E, opt_Eerr, opt_gnorm, opt_ess, opt_var,
            n_opt_steps, η0, η_min, τ, rcond, α_temper, n_sweeps_lo, n_sweeps_hi,
            ent_entropy_initial, ent_entropy_final,
            mps_chain_final = mps_chain, zpairs, V_phys_list, V_left_list,
        )
        println("  wrote $outfile")
    end
    MPI.Barrier(MC_COMM)
end

# ---------------------------------------------------------------------------
# Combined plots + summary table (root only).
# ---------------------------------------------------------------------------
if IS_ROOT
    plt_ent = plot(xlabel = "momentum modes (k, k+π) at each pair-site", ylabel = "entanglement entropy S",
                   title = "Final entanglement entropy vs χ (L=$L, V=$V_target)",
                   xticks = (site_ticks, site_labels), xtickfontsize = 7, xlims = (0.5, Mn + 0.5),
                   legend = :outertopright, left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
                   size = (780, 480))
    for χ in χ_list
        S = results[χ].ent_entropy_final
        plot!(plt_ent, (1:length(S)) .+ 0.5, S, label = "χ=$χ", lw = 2, marker = :circle, markersize = 3)
    end
    savefig(plt_ent, folder_name * "/chi_scan_entropy_L$(L).png")
    println("wrote $(folder_name)/chi_scan_entropy_L$(L).png")

    plt_grad = plot(xlabel = "SR step", ylabel = "‖∇E‖", yscale = :log10,
                     title = "Gradient norm vs χ (L=$L, V=$V_target)",
                     legend = :outertopright, left_margin = 10Plots.mm, size = (780, 480))
    for χ in χ_list
        plot!(plt_grad, 1:n_opt_steps, results[χ].opt_gnorm, label = "χ=$χ", lw = 1.5)
    end
    savefig(plt_grad, folder_name * "/chi_scan_gradnorm_L$(L).png")
    println("wrote $(folder_name)/chi_scan_gradnorm_L$(L).png")

    txtfile = folder_name * "/chi_scan_summary_L$(L).txt"
    open(txtfile, "w") do io
        @printf(io, "chi_scan_mpi summary — L=%d, V=%.2f, exact GS = %.6f, %d rank(s) (%d replica(s)/step)\n\n",
                L, V_target, Egs_target, N_RANKS, max(N_RANKS - 1, 1))
        @printf(io, "%6s %14s %12s %12s %14s %8s\n", "chi", "E_final", "err", "gap", "Var(E_loc)", "ESS")
        for χ in χ_list
            r = results[χ]
            @printf(io, "%6d %14.6f %12.2e %12.6f %14.3e %8.3f\n",
                    χ, r.final_E, r.final_Eerr, r.gap, r.final_var, r.final_ess)
        end
    end
    println("wrote $txtfile")

    println("\nchi_scan_mpi done. Files in $folder_name:")
    for f in sort(readdir(folder_name))
        (endswith(f, ".jld2") || endswith(f, ".png") || endswith(f, ".txt")) && println("  $f")
    end
end
