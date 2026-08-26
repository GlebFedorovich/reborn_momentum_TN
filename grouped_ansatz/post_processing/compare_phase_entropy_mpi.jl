# compare_phase_entropy_mpi.jl — L=16 version of compare_phase_entropy.jl,
# reading from the MPI phase-scan output (phase_scan_vmc_mpi.jl /
# phase_scan_dmrg.jl at L=16). Extended to show ALL scanned χ per phase
# (not just the largest), with each phase (V) assigned a fixed hue and χ
# within that phase mapped to a light->dark sequential ramp on that hue —
# so phase identity reads as color family, χ reads as shade, and a 6th χ
# doesn't require a 7th arbitrary color.
#
# Run from the repo root:   julia --project=. grouped_ansatz/post_processing/compare_phase_entropy_mpi.jl

using TensorKit, JLD2, Plots, Printf

L      = 16
V_list = [1.0, 3.0]
phase_label(V) = V < 2.0 ? "LL" : (V > 2.0 ? "CDW" : "critical")

vmc_dir  = "grouped_ansatz/output/phase_scan_mpi/chi_scan"
dmrg_dir = "grouped_ansatz/output/phase_scan_mpi/dmrg"

get_chi(f) = parse(Int, match(r"chi(\d+)\.jld2$", f).captures[1])

function chis_for(dir, prefix, V)
    files = filter(f -> occursin("$(prefix)_L$(L)_V$(V)_chi", f) && endswith(f, ".jld2"), readdir(dir))
    return sort(get_chi.(files))
end

vmc_chis  = Dict(V => chis_for(vmc_dir, "chi_scan", V) for V in V_list)
dmrg_chis = Dict(V => chis_for(dmrg_dir, "dmrg", V) for V in V_list)

# Fixed hue per phase (categorical), sequential light->dark ramp across χ
# within that phase — matches the blue=LL / orange=CDW convention used
# throughout this project, but now shade-coded by χ instead of collapsing to
# one curve per phase.
ramp = Dict(V_list[1] => cgrad(:Blues), V_list[2] => cgrad(:Oranges))
function shade(V, χ, χs)
    # reserve the lightest 15% of the ramp so even the smallest χ stays visible
    # against a white background
    frac = length(χs) == 1 ? 1.0 : 0.35 + 0.65 * (findfirst(==(χ), χs) - 1) / (length(χs) - 1)
    return ramp[V][frac]
end

# ---------------------------------------------------------------------------
# Panel 1: VMC momentum-space entropy vs momentum pair-site, ALL χ, both phases.
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
                title = "VMC momentum-space, L=$L (all χ)",
                xticks = (site_ticks, site_labels), xtickfontsize = 6,
                xlims = (0.5, Mn + 0.5), legend = :outertopright, legendfontsize = 7)
for V in V_list
    χs = vmc_chis[V]
    for χ in χs
        d = load(joinpath(vmc_dir, "chi_scan_L$(L)_V$(V)_chi$(χ).jld2"))
        S = d["ent_entropy_final"]
        plot!(plt_vmc, (1:length(S)) .+ 0.5, S, label = "V=$V ($(phase_label(V))), χ=$χ",
              lw = 2, marker = :circle, markersize = 2.5, color = shade(V, χ, χs))
    end
end

# ---------------------------------------------------------------------------
# Panel 2: DMRG real-space entropy vs bond, ALL χ, both phases.
# ---------------------------------------------------------------------------
plt_dmrg = plot(xlabel = "bond index (real-space chain)", ylabel = "entanglement entropy S",
                 title = "DMRG real-space, L=$L (all χ)",
                 legend = :outertopright, legendfontsize = 7)
for V in V_list
    χs = dmrg_chis[V]
    for χ in χs
        d = load(joinpath(dmrg_dir, "dmrg_L$(L)_V$(V)_chi$(χ).jld2"))
        S = d["ent_entropy"]
        plot!(plt_dmrg, 1:length(S), S, label = "V=$V ($(phase_label(V))), χ=$χ",
              lw = 2, marker = :circle, markersize = 2.5, color = shade(V, χ, χs))
    end
end

# shared y-axis so the phase-dependence is visually comparable within each panel
ymax = maximum(vcat(
    [maximum(load(joinpath(vmc_dir, "chi_scan_L$(L)_V$(V)_chi$(χ).jld2"))["ent_entropy_final"]) for V in V_list for χ in vmc_chis[V]],
    [maximum(load(joinpath(dmrg_dir, "dmrg_L$(L)_V$(V)_chi$(χ).jld2"))["ent_entropy"]) for V in V_list for χ in dmrg_chis[V]],
))
ylims!(plt_vmc, 0, 1.05 * ymax)
ylims!(plt_dmrg, 0, 1.05 * ymax)

plt = plot(plt_vmc, plt_dmrg, layout = (1, 2), size = (1500, 520),
           left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 4Plots.mm)

outdir = "grouped_ansatz/output/phase_scan_mpi"
outfile = joinpath(outdir, "phase_entropy_comparison_L$(L)_allchi.png")
savefig(plt, outfile)
println("wrote $outfile")

# ---------------------------------------------------------------------------
# Numeric summary: max entropy + area-law χ_min = e^S estimate.
# ---------------------------------------------------------------------------
println("\narea-law bond-dimension estimate (χ_min = e^S, NECESSARY not sufficient):\n")
@printf("%-6s %-5s %-8s %10s %10s %12s\n", "method", "V", "phase", "chi", "max(S)", "chi_min=e^S")
for V in V_list, χ in vmc_chis[V]
    d = load(joinpath(vmc_dir, "chi_scan_L$(L)_V$(V)_chi$(χ).jld2"))
    S = maximum(d["ent_entropy_final"])
    @printf("%-6s %-5.1f %-8s %10d %10.4f %12.3f\n", "VMC", V, phase_label(V), χ, S, exp(S))
end
for V in V_list, χ in dmrg_chis[V]
    d = load(joinpath(dmrg_dir, "dmrg_L$(L)_V$(V)_chi$(χ).jld2"))
    S = maximum(d["ent_entropy"])
    @printf("%-6s %-5.1f %-8s %10d %10.4f %12.3f\n", "DMRG", V, phase_label(V), χ, S, exp(S))
end
