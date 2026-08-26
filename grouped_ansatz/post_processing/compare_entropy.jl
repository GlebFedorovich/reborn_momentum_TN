# compare_entropy.jl — side-by-side entanglement entropy comparison between
# the momentum-space grouped ansatz (chi_scan.jl output) and real-space DMRG
# (dmrg_reference.jl output), at the same L. Two panels rather than one
# overlaid plot: the x-axes are genuinely different quantities — momentum-space
# bonds separate pair-sites labeled by (k,k+π), real-space bonds are literal
# chain positions — so overlaying would be visually misleading, not just
# differently scaled. See the running commentary in chi_scan.jl / dmrg_reference.jl
# for why the two entropy magnitudes/shapes differ physically (Fermi-surface
# peak vs. Calabrese-Cardy arch).
#
# Run from the repo root:   julia --project=. grouped_ansatz/post_processing/compare_entropy.jl

using TensorKit, JLD2, Plots, Printf

L = 8

vmc_dir  = "grouped_ansatz/output/chi_scan"
dmrg_dir = "grouped_ansatz/output/dmrg_reference"

vmc_files  = filter(f -> occursin("chi_scan_L$(L)_chi", f) && endswith(f, ".jld2"), readdir(vmc_dir))
dmrg_files = filter(f -> occursin("dmrg_L$(L)_chi", f) && endswith(f, ".jld2"), readdir(dmrg_dir))

get_chi(f) = parse(Int, match(r"chi(\d+)\.jld2$", f).captures[1])
vmc_chis  = sort(get_chi.(vmc_files))
dmrg_chis = sort(get_chi.(dmrg_files))

println("momentum-space (VMC) χ available: ", vmc_chis)
println("real-space (DMRG) χ available:    ", dmrg_chis)

# momentum-mode x-tick labels, matching chi_scan.jl's convention
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
                title = "Momentum-space grouped ansatz (VMC), L=$L",
                xticks = (site_ticks, site_labels), xtickfontsize = 7, xlims = (0.5, Mn + 0.5),
                legend = :outertopright)
for χ in vmc_chis
    d = load(joinpath(vmc_dir, "chi_scan_L$(L)_chi$(χ).jld2"))
    S = d["ent_entropy_final"]
    plot!(plt_vmc, (1:length(S)) .+ 0.5, S, label = "χ=$χ", lw = 2, marker = :circle, markersize = 3)
end

plt_dmrg = plot(xlabel = "bond index (real-space chain)", ylabel = "entanglement entropy S",
                 title = "Real-space DMRG (twisted boundary), L=$L",
                 legend = :outertopright)
for χ in dmrg_chis
    d = load(joinpath(dmrg_dir, "dmrg_L$(L)_chi$(χ).jld2"))
    S = d["ent_entropy"]
    plot!(plt_dmrg, 1:length(S), S, label = "χ=$χ", lw = 2, marker = :circle, markersize = 3)
end

# shared y-axis range so the ~3-6x magnitude gap is visually obvious at a glance
ymax = max(
    maximum(load(joinpath(vmc_dir, "chi_scan_L$(L)_chi$(maximum(vmc_chis)).jld2"))["ent_entropy_final"]),
    maximum(load(joinpath(dmrg_dir, "dmrg_L$(L)_chi$(maximum(dmrg_chis)).jld2"))["ent_entropy"]),
)
ylims!(plt_vmc, 0, 1.05 * ymax)
ylims!(plt_dmrg, 0, 1.05 * ymax)

plt = plot(plt_vmc, plt_dmrg, layout = (1, 2), size = (1200, 480),
           left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 4Plots.mm)

outdir = "grouped_ansatz/output"
mkpath(outdir)
outfile = joinpath(outdir, "entropy_comparison_L$(L).png")
savefig(plt, outfile)
println("wrote $outfile")

@printf "\nmax entropy: VMC(χ=%d) = %.4f   DMRG(χ=%d) = %.4f   ratio = %.2fx\n" maximum(vmc_chis) maximum(load(joinpath(vmc_dir, "chi_scan_L$(L)_chi$(maximum(vmc_chis)).jld2"))["ent_entropy_final"]) maximum(dmrg_chis) maximum(load(joinpath(dmrg_dir, "dmrg_L$(L)_chi$(maximum(dmrg_chis)).jld2"))["ent_entropy"]) (maximum(load(joinpath(dmrg_dir, "dmrg_L$(L)_chi$(maximum(dmrg_chis)).jld2"))["ent_entropy"]) / maximum(load(joinpath(vmc_dir, "chi_scan_L$(L)_chi$(maximum(vmc_chis)).jld2"))["ent_entropy_final"]))
