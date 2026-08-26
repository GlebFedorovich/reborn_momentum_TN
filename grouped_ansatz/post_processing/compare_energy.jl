# compare_energy.jl — energy convergence comparison between the
# momentum-space grouped ansatz (chi_scan.jl output) and real-space DMRG
# (dmrg_reference.jl output), at the same L. Companion to compare_entropy.jl.
#
# The two methods now target the IDENTICAL Hamiltonian at the same L —
# dmrg_reference.jl's real-space twisted-boundary construction was pinned
# down (via full-spectrum comparison, not just ground-state energy) to
# exactly match exact_gs_energy(V,L) from demo.jl/chi_scan.jl to ~1e-14 (see
# dmrg_reference.jl's header). So both panels below are directly comparable:
# raw E vs χ (panel 1) and |gap| = |E(χ) - E_exact| vs χ on a shared
# log-scale axis (panel 2, answering "which method converges faster/more
# reliably with bond dimension").
#
# Run from the repo root:   julia --project=. grouped_ansatz/post_processing/compare_energy.jl

using TensorKit, JLD2, Plots, Printf

L = 8

vmc_dir  = "grouped_ansatz/output/chi_scan"
dmrg_dir = "grouped_ansatz/output/dmrg_reference"

get_chi(f) = parse(Int, match(r"chi(\d+)\.jld2$", f).captures[1])
vmc_files  = filter(f -> occursin("chi_scan_L$(L)_chi", f) && endswith(f, ".jld2"), readdir(vmc_dir))
dmrg_files = filter(f -> occursin("dmrg_L$(L)_chi", f) && endswith(f, ".jld2"), readdir(dmrg_dir))
vmc_chis  = sort(get_chi.(vmc_files))
dmrg_chis = sort(get_chi.(dmrg_files))

vmc_E    = Float64[]; vmc_Eerr = Float64[]; vmc_gap  = Float64[]
for χ in vmc_chis
    d = load(joinpath(vmc_dir, "chi_scan_L$(L)_chi$(χ).jld2"))
    push!(vmc_E, d["opt_E"][end])
    push!(vmc_Eerr, d["opt_Eerr"][end])
    push!(vmc_gap, abs(d["opt_E"][end] - d["Egs_target"]))
end

dmrg_E   = Float64[]; dmrg_gap = Float64[]
for χ in dmrg_chis
    d = load(joinpath(dmrg_dir, "dmrg_L$(L)_chi$(χ).jld2"))
    push!(dmrg_E, d["E"])
    push!(dmrg_gap, abs(d["gap"]))
end

# ---------------------------------------------------------------------------
# Panel 1: raw energies vs χ, each against its OWN exact-GS reference (two
# y-axes worth of context packed into one panel via separate dashed lines,
# since the absolute scales differ but both converge from above).
# ---------------------------------------------------------------------------
Egs_vmc  = load(joinpath(vmc_dir, "chi_scan_L$(L)_chi$(vmc_chis[1]).jld2"))["Egs_target"]
Egs_dmrg = load(joinpath(dmrg_dir, "dmrg_L$(L)_chi$(dmrg_chis[1]).jld2"))["Eref"]

plt_E = plot(xlabel = "χ", ylabel = "E",
             title = "Energy convergence, L=$L (each vs its own exact GS)",
             legend = :bottomright)
plot!(plt_E, vmc_chis, vmc_E, yerror = vmc_Eerr, label = "VMC (momentum-space)",
      lw = 2, marker = :circle, color = :steelblue)
hline!(plt_E, [Egs_vmc], ls = :dash, lw = 1.5, color = :steelblue, label = "VMC exact GS")
plot!(plt_E, dmrg_chis, dmrg_E, label = "DMRG (real-space)",
      lw = 2, marker = :circle, color = :darkorange)
hline!(plt_E, [Egs_dmrg], ls = :dash, lw = 1.5, color = :darkorange, label = "DMRG exact GS")

# ---------------------------------------------------------------------------
# Panel 2: |gap| vs χ on a shared log-scale axis — the directly comparable
# "how converged is each method" plot.
# ---------------------------------------------------------------------------
plt_gap = plot(xlabel = "χ", ylabel = "|E(χ) - E_exact|", yscale = :log10,
               title = "Convergence to exact GS, L=$L",
               legend = :topright)
plot!(plt_gap, vmc_chis, vmc_gap, label = "VMC (momentum-space)", lw = 2, marker = :circle, color = :steelblue)
plot!(plt_gap, dmrg_chis, dmrg_gap, label = "DMRG (real-space)", lw = 2, marker = :circle, color = :darkorange)

plt = plot(plt_E, plt_gap, layout = (1, 2), size = (1200, 480),
           left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 4Plots.mm)

outdir = "grouped_ansatz/output"
mkpath(outdir)
outfile = joinpath(outdir, "energy_comparison_L$(L).png")
savefig(plt, outfile)
println("wrote $outfile")

@printf "\nVMC:  chi=%s -> gap=%s\n" vmc_chis round.(vmc_gap, digits=5)
@printf "DMRG: chi=%s -> gap=%s\n" dmrg_chis round.(dmrg_gap, digits=6)
