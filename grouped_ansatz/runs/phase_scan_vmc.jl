# phase_scan_vmc.jl — same MC-SR χ-scan recipe as chi_scan.jl, but scanning
# ACROSS the LL→CDW phase transition

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
# SCAN SETTINGS — V_list straddles the V=2t transition (t=1 throughout).
# ---------------------------------------------------------------------------
L        = 8
χ_list   = [8, 12, 16]
V_list   = [1.0, 3.0]   # 1.0: deep Luttinger liquid (Δ=0.5).  3.0: deep CDW (Δ=1.5)

folder_name = "grouped_ansatz/output/phase_scan/chi_scan"
mkpath(folder_name)
energy_plot_dir = folder_name * "/energy_convergence"
mkpath(energy_plot_dir)
gradnorm_plot_dir = folder_name * "/gradnorm_convergence"
mkpath(gradnorm_plot_dir)

# ---------------------------------------------------------------------------
# Building blocks — identical to chi_scan.jl.
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
# SR hyperparameters — same recipe used throughout this project.
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

# ---------------------------------------------------------------------------
# Scan: V (phase) outer, χ inner — same as chi_scan.jl but repeated per V.
# ---------------------------------------------------------------------------
all_results = Dict{Float64, Dict{Int, NamedTuple}}()

for V_target in V_list
    phase_label = V_target < 2.0 ? "LL" : (V_target > 2.0 ? "CDW" : "critical")
    Egs_target = exact_gs_energy(V_target, L)
    println("\n", "#"^70)
    println("V = $V_target  ($phase_label phase, Δ=$(V_target/2))   exact GS = $(round(Egs_target, digits=6))")
    println("#"^70)

    results = Dict{Int, NamedTuple}()

    for χ in χ_list
        println("\n", "="^70)
        println("V=$V_target ($phase_label)   χ = $χ   (L=$L)")
        println("="^70)

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
            Es, Es_err, de = fg(mps_chain, zpairs, folder_name, "V$(V_target)_chi$(χ)_step$(step)",
                                 n_sw, w, s_trial, L; V = V_target, α = α_temper)
            gnorm, converged = sr_step!(mps_chain, de; η = η_t, rcond = rcond)

            Wm_v  = de["results"]["W"]["mean"]
            Wm2_v = de["results"]["W2"]["mean"]
            push!(opt_E, Es * L); push!(opt_Eerr, Es_err * L)
            push!(opt_gnorm, gnorm); push!(opt_ess, Wm_v^2 / Wm2_v)
            push!(opt_var, de["results"]["Energy2"]["mean"] / Wm_v - (de["results"]["Energy"]["mean"] / Wm_v)^2)

            if step <= 3 || step % 40 == 0 || step == n_opt_steps
                @printf("  step %3d (n_sw=%4d):  E·L = %9.5f ± %.1e   gap %+8.5f   ‖∇E‖=%.3e   ESS=%.3f   converged=%d\n",
                        step, n_sw, opt_E[end], Es_err * L, opt_E[end] - Egs_target, gnorm, opt_ess[end], converged)
            end

            rm(folder_name * "/grouped_job_V$(V_target)_chi$(χ)_step$(step).data"; recursive = true, force = true)
            rm(folder_name * "/grouped_job_V$(V_target)_chi$(χ)_step$(step).results.json"; force = true)
        end

        ent_entropy_final = compute_entropy(FiniteMPS(mps_chain))

        @printf("\n  V=%.1f χ=%d summary:  final E·L = %.5f   vs exact GS %.5f   (gap %+.5f)\n",
                V_target, χ, opt_E[end], Egs_target, opt_E[end] - Egs_target)

        plt_E_chi = scatter(1:n_opt_steps, opt_E, yerror = opt_Eerr, xlabel = "SR step", ylabel = "E·L",
                             title = "Energy convergence (L=$L, χ=$χ, V=$V_target, $phase_label)", label = "⟨E⟩·L",
                             markersize = 3, markerstrokewidth = 0.5, legend = :topright)
        plot!(plt_E_chi, 1:n_opt_steps, fill(Egs_target, n_opt_steps), ls = :dash, lw = 2, label = "exact GS")
        savefig(plt_E_chi, energy_plot_dir * "/chi_scan_energy_L$(L)_V$(V_target)_chi$(χ).png")

        plt_grad_chi = plot(1:n_opt_steps, opt_gnorm, xlabel = "SR step", ylabel = "‖∇E‖", yscale = :log10,
                             title = "Gradient norm convergence (L=$L, χ=$χ, V=$V_target, $phase_label)",
                             label = "‖∇E‖", lw = 1.5, legend = :topright)
        savefig(plt_grad_chi, gradnorm_plot_dir * "/chi_scan_gradnorm_L$(L)_V$(V_target)_chi$(χ).png")

        results[χ] = (opt_gnorm = copy(opt_gnorm), ent_entropy_final = copy(ent_entropy_final),
                      final_E = opt_E[end], final_Eerr = opt_Eerr[end], gap = opt_E[end] - Egs_target,
                      final_var = opt_var[end], final_ess = opt_ess[end])

        outfile = folder_name * "/chi_scan_L$(L)_V$(V_target)_chi$(χ).jld2"
        jldsave(outfile;
            L, χ, V_target, phase_label, Egs_target,
            opt_E, opt_Eerr, opt_gnorm, opt_ess, opt_var,
            n_opt_steps, η0, η_min, τ, rcond, α_temper, n_sweeps_lo, n_sweeps_hi,
            ent_entropy_initial, ent_entropy_final,
            mps_chain_final = mps_chain, zpairs, V_phys_list, V_left_list,
        )
        println("  wrote $outfile")
    end

    all_results[V_target] = results

    plt_ent = plot(xlabel = "momentum modes (k, k+π) at each pair-site", ylabel = "entanglement entropy S",
                   title = "Final entanglement entropy vs χ (L=$L, V=$V_target, $phase_label)",
                   xticks = (site_ticks, site_labels), xtickfontsize = 7, xlims = (0.5, Mn + 0.5),
                   legend = :outertopright, left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
                   size = (780, 480))
    for χ in χ_list
        S = results[χ].ent_entropy_final
        plot!(plt_ent, (1:length(S)) .+ 0.5, S, label = "χ=$χ", lw = 2, marker = :circle, markersize = 3)
    end
    savefig(plt_ent, folder_name * "/chi_scan_entropy_L$(L)_V$(V_target).png")
    println("wrote $(folder_name)/chi_scan_entropy_L$(L)_V$(V_target).png")
end

# ---------------------------------------------------------------------------
# Final summary table across both V/phase and χ.
# ---------------------------------------------------------------------------
txtfile = folder_name * "/phase_scan_summary_L$(L).txt"
open(txtfile, "w") do io
    @printf(io, "phase_scan_vmc summary — L=%d\n\n", L)
    @printf(io, "%6s %6s %14s %12s %12s %14s %8s %10s\n",
            "V", "chi", "E_final", "err", "gap", "Var(E_loc)", "ESS", "max(S)")
    for V_target in V_list
        for χ in χ_list
            r = all_results[V_target][χ]
            @printf(io, "%6.1f %6d %14.6f %12.2e %12.6f %14.3e %8.3f %10.4f\n",
                    V_target, χ, r.final_E, r.final_Eerr, r.gap, r.final_var, r.final_ess,
                    maximum(r.ent_entropy_final))
        end
    end
end
println("\nwrote $txtfile")
println("\nphase_scan_vmc done. Files in $folder_name")
