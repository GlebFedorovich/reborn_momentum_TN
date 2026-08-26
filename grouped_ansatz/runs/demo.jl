# demo.jl — dim-4 (k,k+π) grouped super-site momentum-MPS ansatz.
# Run from the repo root:   julia --project=. grouped_ansatz/runs/demo.jl

using TensorKit, TensorOperations, Plots, LaTeXStrings
using MPSKit, Random, LinearAlgebra, Printf, Statistics
using JSON, HDF5
using SparseArrays, KrylovKit   # for the momentum-0-sector exact_gs_energy reference (needed at L=16)

L = 8   # best-known L=12 recipe (from chi_scan.jl): χ=16 gave gap +0.039 — close to
         # χ=20/24's +0.026-0.027 but noticeably cheaper per step, good accuracy/cost point
χ = 8

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
const M = GroupedMomentumMPS

using Carlo, Carlo.JobTools

# Bipartite entanglement entropy at each bond of a given chain — same
# construction as demo/demo.jl's compute_entropy, and directly comparable to
# the momentum-space entanglement findings in [[finding-two-entanglement-notions]]
# / [[finding-mode-ordering-folding]]. Defined up here (rather than only at the
# end) so it can be called on the FRESH Fermi-sea start, before the SR loop
# mutates mps_chain_start in place via apply_update! — that's the only way to
# see the initial-state entropy profile alongside the optimized one later.
function compute_entropy(ψ::FiniteMPS)
    ψ = normalize(ψ)   # makes Σλ²=1 meaningful, and matches this function's convention check
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

# Generic over L (was hardcoded to ZNIrrep{16}, i.e. only L=8, before — the
# main project's demo.jl had to add a new hardcoded method by hand every time
# L changed; a free type parameter avoids repeating that).
StructUtils.lowerkey(::JSON.JSONStyle, s::ProductSector{Tuple{FermionParity, U1Irrep, ZNIrrep{N}}}) where {N} =
    string(s)

mps_chain_start, V_phys_list, V_left_list, zpairs = M.get_start_fermi_sea(L, χ; ε = 3.0);
# mps_chain_start, V_phys_list, V_left_list, zpairs = M.get_start(L, χ);

s_trial = M.fermi_sea_config(L)     # chain-order Fermi-sea config
string_start = M.make_sample_string(s_trial, V_phys_list, V_left_list[1], zpairs, L)
weight_start = M.get_overlap(mps_chain_start, string_start)[1]

@show space(mps_chain_start[end], 3)
@show space(string_start[end], 3)

# Captured BEFORE optimization (mps_chain_start is mutated in place by
# apply_update! inside the SR loop below) — the initial-state counterpart to
# ent_entropy_final, plotted together at the end.
ent_entropy_initial = compute_entropy(FiniteMPS(mps_chain_start))

function fg(mps_chain, zpairs, folder_name, id, maxiter, weight_in, config_start; V = 1.0, α = 1.0)
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
    total_attempts = dict[end]["results"]["total_attempts"]["mean"]
    accepted_flips = dict[end]["results"]["accepted_flips"]["mean"]

    G_out = []
    for i in eachindex(mps_chain)
        G_new = zeros(Float64, space(mps_chain[i]))
        for (sector, block_in) in blocks(mps_chain[i])
            key1 = M.block_key_gradient(i, sector)
            g_block = dict[end]["results"][key1]["mean"]
            g_block = Float64.(Iterators.flatten(g_block)) |> collect
            g_block_mat = reshape(g_block, size(block_in)) ./ Wm

            key2 = M.block_key_gradient_times_e_loc(i, sector)
            g_block_e_loc = dict[end]["results"][key2]["mean"]
            g_block_e_loc = Float64.(Iterators.flatten(g_block_e_loc)) |> collect
            g_block_e_loc_mat = reshape(g_block_e_loc, size(block_in)) ./ Wm

            G_true = g_block_e_loc_mat - f * g_block_mat
            block(G_new, sector) .= G_true
        end
        push!(G_out, G_new)
    end

    return f, f_error, accepted_flips, total_attempts, G_out, dict[end]
end

# Gauge-projected stochastic reconfiguration — same S/g construction as
# demo/demo.jl's sr_step! (centering ⟨O⟩ removes the global-scale gauge
# direction), but the (S + shift·I)δp = g linear solve is now done via
# KrylovKit.linsolve (CG on a Tikhonov-regularized S) instead of the
# truncated-eigendecomposition pseudo-inverse. shift = rcond·λmax plays the
# same role as the old rcond·λmax eigenvalue floor — it's a smooth analogue
# of the hard spectral cutoff, applied as a ridge term rather than truncating
# directions outright. λmax itself comes from KrylovKit.eigsolve (Lanczos,
# top eigenvalue only) rather than a dense eigmax(S), keeping the whole step
# iterative — no O(N³) full diagonalization anywhere.
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
    @show N, λmax, shift, norm(g), norm(δp), info.converged, info.normres

    M.apply_update!(mps_chain, -η .* δp)
    return δp, norm(g)
end

folder_name = "grouped_ansatz/output"
mkpath(folder_name)

# Exact ground-state energy of H_mom(V), restricted to the total-momentum-0
# sector (where the ansatz's own symmetry lives, and — per
# [[plan-importance-sampling]] — where the true GS lives for V<2 at L≡0 mod4)
# via sparse ED + KrylovKit. Dense full-sector ED (the previous version of
# this function) is O(D³) with D=C(L,L/2) — fine at L=8 (D=70) or L=12
# (D=924), but D=12870 at L=16 makes the dense eigendecomposition intractable
# (~2e12 flops). Momentum-0-sector-restricted D is far smaller (validated:
# L=8 D=10, L=12 D=80, L=16 D=810 — matches the dense answer exactly at L=8
# and L=12, 0.03s at L=16 vs. effectively unusable dense).
function exact_gs_energy(V; L = L)
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
# Single run: sanity-check ⟨E_loc⟩ before optimizing.
# ---------------------------------------------------------------------------
n_sweeps = 1000
V_target = 0.5

f, f_error, accepted, attempts, G_out, dict_entry =
    fg(mps_chain_start, zpairs, folder_name, "single", n_sweeps, weight_start, s_trial; V = V_target)

Wm_single = dict_entry["results"]["W"]["mean"]
E_var = dict_entry["results"]["Energy2"]["mean"] / Wm_single - f^2

println("\n===== single direct-sampler MC run (GROUPED, L=$L, χ=$χ, V=$V_target, sweeps=$n_sweeps) =====")
@printf("⟨E_loc⟩      = % .6f ± %.1e   (per site)\n", f, f_error)
@printf("⟨E_loc⟩·L    = % .6f ± %.1e   (total energy ⟨H⟩)\n", f * L, f_error * L)
@printf("Var(E_loc)   = %.3e\n", E_var)

# ---------------------------------------------------------------------------
# Fixed-V MC-SR: decaying step size + tempered importance sampling, exactly
# the recipe that reached the GS for the unpaired ansatz (see
# [[finding-exact-sr-stepsize]], [[plan-importance-sampling]]).
# ---------------------------------------------------------------------------
η0          = 0.2   # best-known L=12 recipe (chi_scan.jl): this η0/η_min/rcond/α combo
                       # gave gap +0.039 at χ=16 and +0.026-0.027 at χ=20/24
η_min       = 0.05
# τ rescaled to reach the floor at ~90% through THIS run's length (300 steps):
# η_sched hits η_min when step/τ = η0/η_min - 1 = 4, so τ = 270/4 = 67.5 puts
# the floor at step 270.
# τ           = 67.5
τ =  η0/η_min - 1
η_sched(step) = max(η_min, η0 / (1 + step / τ))
# REVERTED to 1e-6: tightening to 1e-4 DID suppress the norm(δp) spikes
# (confirmed the mechanism) but made the actual result worse (gap +0.271 vs
# +0.098) — count(keep) roughly halved (38-40 → 26-29), so the near-null
# directions being cut were carrying real, needed information, not just
# noise. Best config found so far: α=0.5, τ scaled for run length, rcond=1e-6.
rcond       = 1e-5
# At L=16: α=0.7 backfired (sampler collapse). α=0.3 clearly helped (gap
# +0.417 vs +0.528 at α=0.5) despite ESS dropping to ~0.22. α=0.1 pushed too
# far — ESS crashed to ~0.01-0.03 for the ENTIRE run (not just late), meaning
# the estimator itself was unreliable throughout (error bars routinely ±1-3,
# bigger than the energy scale) — that run's improved-looking gap isn't
# trustworthy. Now testing α=0.2 (between the working 0.3 and broken 0.1) —
# and back at L=12, where α has never been touched away from 0.5.
α_temper    = 0.5
n_opt_steps  = 400
# SWEEP RAMP (variance reduction where it matters): the ESS diagnostic showed
# ‖∇E‖ descends cleanly for a while then plateaus in noise at a level set by
# the sampling precision. Early on the true gradient is large, so a cheap/
# noisy estimate still points the right way; only once the true gradient
# shrinks toward the noise floor does the extra precision start to matter.
# Ramp sweeps_opt linearly over the run instead of a flat count — capped at
# 2000 (lower than the earlier 500→3000 ramp) since more steps (400 vs 300)
# already buys more total samples at fixed cost per step.
n_sweeps_lo  = 1500
n_sweeps_hi  = 3000
n_sweeps_sched(step) = round(Int, n_sweeps_lo + (n_sweeps_hi - n_sweeps_lo) * (step - 1) / (n_opt_steps - 1))
Egs_target   = exact_gs_energy(V_target)

opt_E      = Float64[]
opt_Eerr   = Float64[]   # MC error bar on E·L per step, for the energy plot
opt_Eref   = Float64[]
opt_var    = Float64[]
opt_gnorm  = Float64[]   # ‖∇E‖ = ‖OE_mean - O_mean*E_mean‖ per SR step — should → 0 at true convergence
opt_ess    = Float64[]   # effective sample size ⟨iw⟩²/⟨iw²⟩ ∈ (0,1] — tests whether ‖∇E‖ spikes
                          # coincide with heavy-tailed importance weights (tempered sampling, α<1)
opt_nsweep = Int[]        # sweeps/step actually used (the ramp), for the plot

println("\n----- MC-SR at FIXED V=$V_target (GROUPED dim-4 ansatz, χ=$χ, η=$η0→$η_min, α=$α_temper, " *
        "rcond=$rcond, $n_opt_steps steps, sweeps/step ramp $n_sweeps_lo→$n_sweeps_hi) -----")
println("      exact GS = $(round(Egs_target, digits=6))")
iter = 0
for step in 1:n_opt_steps
    global iter = step
    η_t = η_sched(step)
    n_sw = n_sweeps_sched(step)
    w = M.get_overlap(mps_chain_start, string_start)[1]
    Es, Es_err, _, _, _, de = fg(mps_chain_start, zpairs, folder_name, "anneal_$step",
                                  n_sw, w, s_trial; V = V_target, α = α_temper)
    _, gnorm = sr_step!(mps_chain_start, de; η = η_t, rcond = rcond)
    push!(opt_E, Es * L); push!(opt_Eerr, Es_err * L)
    push!(opt_Eref, Egs_target); push!(opt_gnorm, gnorm); push!(opt_nsweep, n_sw)
    Wm_v  = de["results"]["W"]["mean"]
    Wm2_v = de["results"]["W2"]["mean"]
    push!(opt_var, de["results"]["Energy2"]["mean"] / Wm_v - (de["results"]["Energy"]["mean"] / Wm_v)^2)
    push!(opt_ess, Wm_v^2 / Wm2_v)
    if step <= 5 || step % 10 == 0 || step == n_opt_steps
        @printf("  step %3d (n_sw=%4d):  E·L = %9.5f ± %.1e   gap %+8.5f   Var(E_loc)=%.2e   ‖∇E‖=%.3e   ESS=%.3f\n",
                step, n_sw, opt_E[end], Es_err * L, opt_E[end] - Egs_target, opt_var[end], gnorm, opt_ess[end])
    end
end

@printf("\nsummary:  V=%.2f  χ=%d  final E·L = %.5f  vs exact GS %.5f  (gap %+.5f)  %s\n",
        V_target, χ, opt_E[end], Egs_target, opt_E[end] - Egs_target,
        abs(opt_E[end] - Egs_target) < 2e-2 ? "*** at GS (within MC noise) ***" : "(off GS)")

# Bipartite entanglement entropy at each bond of the FINAL (post-optimization)
# chain — a flat-topped plateau (linear ordering) vs. a two-peak/dip shape
# (π-fold / grouped ordering) is exactly what ED studies predicted for this
# chain's pairing scheme; compared against ent_entropy_initial (captured
# above, before the SR loop) to see how the Fermi-sea start's entropy profile
# evolves under optimization.
ent_entropy_final = compute_entropy(FiniteMPS(mps_chain_start))

# x-tick labels for the entropy plot: each pair-site s (1..L/2) holds momenta
# (k_s, k_s+π) — see the pair-site-momenta breakdown discussed for L=8. Express
# each as a reduced fraction of π so ticks read like "-5π/8" instead of a
# decimal, and place them at each SITE position while the entropy points sit
# at bond MIDPOINTS (b+0.5, between site b and b+1) so the plot visually shows
# which momenta flank each cut.
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

plt_E = scatter(1:iter, opt_E, yerror = opt_Eerr, xlabel = "SR step", ylabel = "E·L",
                 title = "Grouped dim-4 ansatz MC-SR (L=$L, χ=$χ, V=$V_target)", label = "⟨E⟩·L",
                 markersize = 3, markerstrokewidth = 0.5)
plot!(plt_E, 1:iter, opt_Eref, ls = :dash, lw = 2, label = "exact GS")

plt_g = plot(1:iter, opt_gnorm, xlabel = "SR step", ylabel = "‖∇E‖", yscale = :log10,
             label = "‖∇E‖", lw = 2, legend = :topright, ylims = ((1e-5, 1e0)))

plt_ess = plot(1:iter, opt_ess, xlabel = "SR step", ylabel = "ESS = ⟨iw⟩²/⟨iw²⟩",
               label = "ESS (α=$α_temper)", lw = 2, color = :seagreen, ylims = (0, 1.02))

plt_S = plot((1:length(ent_entropy_initial)) .+ 0.5, ent_entropy_initial,
             xlabel = "momentum modes (k, k+π) at each pair-site",
             ylabel = "entanglement entropy S", label = "initial (Fermi-sea start)", lw = 2,
             marker = :circle, ls = :dash, color = :gray,
             xticks = (site_ticks, site_labels), xtickfontsize = 7, xlims = (0.5, Mn + 0.5))
plot!(plt_S, (1:length(ent_entropy_final)) .+ 0.5, ent_entropy_final,
      label = "final MPS (χ=$χ)", lw = 2, marker = :circle, color = :darkorange)

plt = plot(plt_E, plt_g, plt_ess, plt_S, layout = (4, 1), size = (780, 1200),
           left_margin = 12Plots.mm, bottom_margin = 4Plots.mm)
savefig(plt, folder_name * "/grouped_anneal_energy.png")
println("wrote $(folder_name)/grouped_anneal_energy.png")

# Does ESS crash exactly on the steps where ‖∇E‖ spikes? Correlation of
# -log(ESS) with ‖∇E‖ close to 1 (or a scatter that's visibly monotone) would
# confirm the heavy-tailed-importance-weight hypothesis; near 0 would refute it.
gnorm_med = median(opt_gnorm)
spike_steps = findall(>(3 * gnorm_med), opt_gnorm)
@printf("\n----- ESS vs ‖∇E‖-spike diagnostic -----\n")
@printf("median ‖∇E‖ = %.3e ; %d steps with ‖∇E‖ > 3×median\n", gnorm_med, length(spike_steps))
if !isempty(spike_steps)
    ess_at_spikes  = opt_ess[spike_steps]
    ess_elsewhere  = opt_ess[setdiff(1:iter, spike_steps)]
    @printf("  mean ESS at spike steps    = %.4f\n", mean(ess_at_spikes))
    @printf("  mean ESS at non-spike steps = %.4f\n", mean(ess_elsewhere))
end
@printf("correlation(‖∇E‖, -log(ESS)) = %.3f  (near +1 ⇒ heavy-tailed-weight hypothesis confirmed)\n",
        cor(opt_gnorm, -log.(opt_ess)))

@show opt_E[end]
@show Egs_target
@show opt_gnorm[end]
