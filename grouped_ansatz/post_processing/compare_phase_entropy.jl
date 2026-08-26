# Run from the repo root:   julia --project=. grouped_ansatz/post_processing/compare_phase_entropy.jl

using TensorKit, JLD2, Plots, Printf

L      = 8
V_list = [1.0, 3.0]
phase_label(V) = V < 2.0 ? "LL" : (V > 2.0 ? "CDW" : "critical")

vmc_dir  = "grouped_ansatz/output/phase_scan/chi_scan"
dmrg_dir = "grouped_ansatz/output/phase_scan/dmrg"

get_chi(f) = parse(Int, match(r"chi(\d+)\.jld2$", f).captures[1])

function chis_for(dir, prefix, V)
    files = filter(f -> occursin("$(prefix)_L$(L)_V$(V)_chi", f) && endswith(f, ".jld2"), readdir(dir))
    return sort(get_chi.(files))
end

vmc_chis  = Dict(V => chis_for(vmc_dir, "chi_scan", V) for V in V_list)
dmrg_chis = Dict(V => chis_for(dmrg_dir, "dmrg", V) for V in V_list)

colors = Dict(V_list[1] => :steelblue, V_list[2] => :firebrick)

# ---------------------------------------------------------------------------
# Panel 1: VMC momentum-space entropy vs momentum pair-site, at max χ, LL vs CDW.
# ---------------------------------------------------------------------------
function momentum_frac_label(idx, L)
    k = -π + π * (2 * idx - 1) / L
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

plt_vmc = plot(xlabel = "momentum modes (k, k+π) at each pair-site", ylabel = "entanglement entropy S",
                title = "VMC momentum-space, L=$L (max χ per phase)",
                xticks = (site_ticks, site_labels), xtickfontsize = 7, xlims = (0.5, Mn + 0.5),
                legend = :outertopright)
for V in V_list
    χmax = maximum(vmc_chis[V])
    d = load(joinpath(vmc_dir, "chi_scan_L$(L)_V$(V)_chi$(χmax).jld2"))
    S = d["ent_entropy_final"]
    plot!(plt_vmc, (1:length(S)) .+ 0.5, S, label = "V=$V ($(phase_label(V))), χ=$χmax",
          lw = 2, marker = :circle, markersize = 3, color = colors[V])
end

# ---------------------------------------------------------------------------
# Panel 2: DMRG real-space entropy vs bond, at max χ, LL vs CDW.
# ---------------------------------------------------------------------------
plt_dmrg = plot(xlabel = "bond index (real-space chain)", ylabel = "entanglement entropy S",
                 title = "DMRG real-space, L=$L (max χ per phase)",
                 legend = :outertopright)
for V in V_list
    χmax = maximum(dmrg_chis[V])
    d = load(joinpath(dmrg_dir, "dmrg_L$(L)_V$(V)_chi$(χmax).jld2"))
    S = d["ent_entropy"]
    plot!(plt_dmrg, 1:length(S), S, label = "V=$V ($(phase_label(V))), χ=$χmax",
          lw = 2, marker = :circle, markersize = 3, color = colors[V])
end

# shared y-axis so the phase-dependence is visually comparable within each panel
ymax = maximum(vcat(
    [maximum(load(joinpath(vmc_dir, "chi_scan_L$(L)_V$(V)_chi$(maximum(vmc_chis[V])).jld2"))["ent_entropy_final"]) for V in V_list],
    [maximum(load(joinpath(dmrg_dir, "dmrg_L$(L)_V$(V)_chi$(maximum(dmrg_chis[V])).jld2"))["ent_entropy"]) for V in V_list],
))
ylims!(plt_vmc, 0, 1.05 * ymax)
ylims!(plt_dmrg, 0, 1.05 * ymax)

plt = plot(plt_vmc, plt_dmrg, layout = (1, 2), size = (1300, 480),
           left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 4Plots.mm)

outdir = "grouped_ansatz/output/phase_scan"
outfile = joinpath(outdir, "phase_entropy_comparison_L$(L).png")
savefig(plt, outfile)
println("wrote $outfile")

# ---------------------------------------------------------------------------
# Numeric summary: max entropy + area-law χ_min = e^S estimate, per
# (method, phase), at every χ scanned — not just the largest.
# ---------------------------------------------------------------------------
println("\narea-law bond-dimension estimate (χ_min = e^S, NECESSARY not sufficient — see [[momentum-space VMC project]]):\n")
@printf("%-6s %-5s %-8s %10s %10s %12s\n", "method", "V", "phase", "chi", "max(S)", "chi_min=e^S")
for V in V_list
    for χ in vmc_chis[V]
        d = load(joinpath(vmc_dir, "chi_scan_L$(L)_V$(V)_chi$(χ).jld2"))
        S = maximum(d["ent_entropy_final"])
        @printf("%-6s %-5.1f %-8s %10d %10.4f %12.3f\n", "VMC", V, phase_label(V), χ, S, exp(S))
    end
end
for V in V_list
    for χ in dmrg_chis[V]
        d = load(joinpath(dmrg_dir, "dmrg_L$(L)_V$(V)_chi$(χ).jld2"))
        S = maximum(d["ent_entropy"])
        @printf("%-6s %-5.1f %-8s %10d %10.4f %12.3f\n", "DMRG", V, phase_label(V), χ, S, exp(S))
    end
end
