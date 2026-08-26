# chi_scan.jl — for FIXED L, scan several χ (bond dimension) values, running
# the same MC-SR recipe as demo.jl for each one, and save every diagnostic
# (energy, gradient norm, MC error bars, ESS, entanglement entropy — both
# initial Fermi-sea-start and final post-optimization) plus the converged MPS
# itself to a separate .jld2 file per χ. 

# Load results later with, e.g. (TensorKit MUST be `using`'d before load() so
# the saved TensorMap/GradedSpace types deserialize as themselves rather than
# JLD2 reconstructing dummy placeholder types):
#   using TensorKit, JLD2
#   d = load("grouped_ansatz/output/chi_scan/chi_scan_L16_chi16.jld2")
#   d["ent_entropy_final"], d["opt_E"], d["mps_chain_final"], ...

using TensorKit, TensorOperations, LinearAlgebra, Printf, Statistics, Random
using MPSKit, Plots
using JSON, JLD2
using SparseArrays, KrylovKit

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
const M = GroupedMomentumMPS

using Carlo, Carlo.JobTools

StructUtils.lowerkey(::JSON.JSONStyle, s::ProductSector{Tuple{FermionParity, U1Irrep, ZNIrrep{N}}}) where {N} =
    string(s)

# ---------------------------------------------------------------------------
# SCAN SETTINGS
# ---------------------------------------------------------------------------
L        = 8
χ_list   = [8, 12, 16]
V_target = 0.5

folder_name = "grouped_ansatz/output/chi_scan"
mkpath(folder_name)
energy_plot_dir = folder_name * "/energy_convergence"
mkpath(energy_plot_dir)
gradnorm_plot_dir = folder_name * "/gradnorm_convergence"
mkpath(gradnorm_plot_dir)

# ---------------------------------------------------------------------------
# Building blocks — same logic as demo.jl's fg / sr_step! / exact_gs_energy /
# compute_entropy, copied rather than shared so this file has no dependency
# on demo.jl's current (frequently-edited) state.
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
    rm(job_path * ".data"; recursive = true, force = true)

    job = JobInfo(job_path, M.MC;
        run_time = "24:00:00",
        checkpoint_time = "12:00:00",
        tasks = make_tasks(tm),
    )

    start(Carlo.SingleScheduler, job)

    stringdata = join(readlines(folder_name * "/grouped_job_" * string(id) * ".results.json"))
    dict = JSON.parse(stringdata; allownan = true)

    Wm = dict[end]["results"]["W"]["mean"]
    f = dict[end]["results"]["Energy"]["mean"] / Wm
    f_error = dict[end]["results"]["Energy"]["error"] / abs(Wm)

    return f, f_error, dict[end]
end

# Same iterative (KrylovKit eigsolve + linsolve) SR step as the current
# demo.jl — see there for the full rationale.
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

# Sparse, momentum-0-sector-restricted ED reference — identical to demo.jl's
# exact_gs_energy (see there for why sparse+sector-restricted, not dense).
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
# SR hyperparameters — current best-known recipe: fixed across all χ in the
# scan, so entropy-vs-χ comparisons aren't confounded by also changing the
# optimizer. η_min raised from 0.005 — the L=8 chi_scan's gap didn't improve
# monotonically with χ using that smaller floor (likely under-stepping at
# larger χ, same failure mode demo.jl saw at L=16/χ=24 before raising η_min).
# ---------------------------------------------------------------------------
η0          = 0.2
η_min       = 0.05
n_opt_steps = 400
τ           = (n_opt_steps * 0.9) / (η0 / η_min - 1)   # floor reached at ~90% through the run
η_sched(step) = max(η_min, η0 / (1 + step / τ))
rcond       = 1e-5
α_temper    = 0.5   # back to the value that's worked reliably all session — α=0.0 broke the
                     # sampler entirely, α=1.0 (no tempering) gave tiny error bars but got stuck
                     # (gap plateaued at 0.04-0.09, non-monotonic) — sampler collapse, precise
                     # but wrong. α=0.5 balances exploration against noise.
n_sweeps_lo = 1000
n_sweeps_hi = 2500   # doubled from 2000 for tighter error bars at the high end of the ramp
n_sweeps_sched(step) = round(Int, n_sweeps_lo + (n_sweeps_hi - n_sweeps_lo) * (step - 1) / (n_opt_steps - 1))

Egs_target = exact_gs_energy(V_target, L)
println("exact GS (L=$L, V=$V_target) = $(round(Egs_target, digits=6))")
println("χ scan: ", χ_list)

# x-tick labels for the entropy plot — same convention as demo.jl: pair-site s
# (1..L/2) holds momenta (k_s, k_s+π), points sit at bond midpoints (b+0.5).
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

# Collected across the scan for the combined end-of-run plots/table below.
results = Dict{Int, NamedTuple}()

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
for χ in χ_list
    println("\n", "="^70)
    println("χ = $χ   (L=$L, V=$V_target)")
    println("="^70)

    # ε raised from the default 0.5 — heavier random component relative to the
    # Fermi-sea product term, while still keeping that product term's guaranteed
    # nonzero-weight path (avoids the dead-end issue that broke the fully-random
    # M.get_start start).
    mps_chain, V_phys_list, V_left_list, zpairs = M.get_start_fermi_sea(L, χ; ε = 3.0)
    s_trial = M.fermi_sea_config(L)
    string_start = M.make_sample_string(s_trial, V_phys_list, V_left_list[1], zpairs, L)

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

        if step <= 3 || step % 20 == 0 || step == n_opt_steps
            @printf("  step %3d (n_sw=%4d):  E·L = %9.5f ± %.1e   gap %+8.5f   ‖∇E‖=%.3e   ESS=%.3f   converged=%d\n",
                    step, n_sw, opt_E[end], Es_err * L, opt_E[end] - Egs_target, gnorm, opt_ess[end], converged)
        end

        # per-step Carlo job files are pure scratch once fg() has read them
        # into de — delete immediately so a long scan doesn't pile up
        # thousands of .data directories on disk.
        rm(folder_name * "/grouped_job_chi$(χ)_step$(step).data"; recursive = true, force = true)
        rm(folder_name * "/grouped_job_chi$(χ)_step$(step).results.json"; force = true)
    end

    ent_entropy_final = compute_entropy(FiniteMPS(mps_chain))

    @printf("\n  χ=%d summary:  final E·L = %.5f   vs exact GS %.5f   (gap %+.5f)\n",
            χ, opt_E[end], Egs_target, opt_E[end] - Egs_target)

    # Per-χ energy-convergence plot (with MC error bars), saved separately so
    # each χ's SR trajectory can be inspected on its own rather than only via
    # the combined end-of-scan plots below.
    plt_E_chi = scatter(1:n_opt_steps, opt_E, yerror = opt_Eerr, xlabel = "SR step", ylabel = "E·L",
                         title = "Energy convergence (L=$L, χ=$χ, V=$V_target)", label = "⟨E⟩·L",
                         markersize = 3, markerstrokewidth = 0.5, legend = :topright)
    plot!(plt_E_chi, 1:n_opt_steps, fill(Egs_target, n_opt_steps), ls = :dash, lw = 2, label = "exact GS")
    savefig(plt_E_chi, energy_plot_dir * "/chi_scan_energy_L$(L)_chi$(χ).png")
    println("  wrote $(energy_plot_dir)/chi_scan_energy_L$(L)_chi$(χ).png")

    # Per-χ gradient-norm-convergence plot, same idea as the energy one above.
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

# ---------------------------------------------------------------------------
# Combined plots: entanglement entropy (final, post-optimization) and
# gradient-norm trajectory, one curve per χ, so the χ-dependence is visible
# directly rather than only comparable file-by-file later.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Final energies as a plain-text table.
# ---------------------------------------------------------------------------
txtfile = folder_name * "/chi_scan_summary_L$(L).txt"
open(txtfile, "w") do io
    @printf(io, "chi_scan summary — L=%d, V=%.2f, exact GS = %.6f\n\n", L, V_target, Egs_target)
    @printf(io, "%6s %14s %12s %12s %14s %8s\n", "chi", "E_final", "err", "gap", "Var(E_loc)", "ESS")
    for χ in χ_list
        r = results[χ]
        @printf(io, "%6d %14.6f %12.2e %12.6f %14.3e %8.3f\n",
                χ, r.final_E, r.final_Eerr, r.gap, r.final_var, r.final_ess)
    end
end
println("wrote $txtfile")

println("\nchi_scan done. Files in $folder_name:")
for f in sort(readdir(folder_name))
    (endswith(f, ".jld2") || endswith(f, ".png") || endswith(f, ".txt")) && println("  $f")
end
