# test_grad_fd.jl — grouped-ansatz analogue of test/test_grad_fd.jl.
#
# Deterministic (zero-MC-noise) gate for the ENERGY GRADIENT of the dim-4
# (k,k+π) grouped ansatz: does get_partial_G + the ⟨O·E⟩−⟨O⟩⟨E⟩ combination
# (measure!/fg in grouped_mps.jl / demo.jl use it) equal the true ∂E/∂p of the
# variational energy? Same method as the original gate: enumerate the whole
# momentum-0 support (no sampling), compare the analytic gradient against
# central finite differences, parameter by parameter.

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
using LinearAlgebra, Printf
const M = GroupedMomentumMPS

const L = 8
const N = L ÷ 2
const t = 1.0
occ(s, i) = (s >> (i - 1)) & 1
const states = [s for s in 0:(1 << L)-1 if count_ones(s) == N]

Vplist(chain) = [M.space(chain[i], 2) for i in eachindex(chain)]
bleft(chain)  = M.space(chain[1], 1)

amp(chain, zpairs, config) =
    M.get_overlap(chain, M.make_sample_string(config, Vplist(chain), bleft(chain), zpairs, L))[1]

function mc_at(chain, zpairs, config, V)
    params = Dict(:mps_chain => chain, :L => L, :zpairs => zpairs, :t_hopping => t, :V_repulsion => V,
        :config => copy(config),
        :idx_filled => findall(==(1), config), :idx_empty => findall(==(0), config),
        :weight => amp(chain, zpairs, config))
    M.MC(params)
end

function support_data(chain, zpairs, V)
    cfgs = Vector{Vector{Int}}()
    ψ = Float64[]
    for s in states
        c = [occ(s, j) for j in 1:L]     # CHAIN-order config (matches test_paired_eloc.jl's convention)
        w = amp(chain, zpairs, c)
        push!(cfgs, c); push!(ψ, w)
    end
    keep = abs.(ψ) .> 1e-12 * maximum(abs, ψ)
    cfgs = cfgs[keep]; ψ = ψ[keep]
    p = abs2.(ψ) ./ sum(abs2, ψ)
    eloc = Float64[]
    for c in cfgs
        push!(eloc, M.get_E_loc(mc_at(chain, zpairs, c, V)))
    end
    return cfgs, ψ, p, eloc
end

exactE(chain, zpairs, V) = (d = support_data(chain, zpairs, V); sum(d[3] .* d[4]))

function analytic_grad(chain, zpairs, V)
    cfgs, ψ, p, eloc = support_data(chain, zpairs, V)
    E = sum(p .* eloc)
    Vp = Vplist(chain); bl = bleft(chain)
    Osum = nothing
    OEsum = nothing
    for (n, c) in enumerate(cfgs)
        ss = M.make_sample_string(c, Vp, bl, zpairs, L)
        G  = M.get_partial_G(chain, ss, bl, M.space(ss[end], 3))
        O  = G / ψ[n]
        if Osum === nothing
            Osum  = O * p[n]
            OEsum = O * (p[n] * eloc[n])
        else
            Osum  = Osum  + O * p[n]
            OEsum = OEsum + O * (p[n] * eloc[n])
        end
    end
    grad = (OEsum - E * Osum) * 2
    return M.flatten_gradient(grad)
end

function run(V; χ = 8, δ = 1e-5)
    println("="^70)
    chain0, _, _, zpairs = M.get_start_fermi_sea(L, χ; ε = 0.5)
    g_anal = analytic_grad(chain0, zpairs, V)
    nP = length(g_anal)
    @printf("V = %.2f,  χ = %d,  n_params = %d,  δ = %.0e  (GROUPED dim-4 ansatz)\n", V, χ, nP, δ)

    g_fd = zeros(nP)
    for k in 1:nP
        ek = zeros(nP); ek[k] = 1.0
        cp = [copy(chain0[i]) for i in eachindex(chain0)]; M.apply_update!(cp, +δ .* ek)
        cm = [copy(chain0[i]) for i in eachindex(chain0)]; M.apply_update!(cm, -δ .* ek)
        g_fd[k] = (exactE(cp, zpairs, V) - exactE(cm, zpairs, V)) / (2δ)
    end
    rel  = abs.(g_fd .- g_anal) ./ max.(abs.(g_anal), 1e-8)
    nbad = count(rel .> 1e-4)
    @printf("‖g_anal‖ = %.4e   ‖g_fd‖ = %.4e   mismatched %d/%d params   maxrel = %.2e   %s\n",
            norm(g_anal), norm(g_fd), nbad, nP, maximum(rel), nbad == 0 ? "MATCH" : "*** MISMATCH ***")
    println("   worst params (k, analytic, finite-diff, fd/anal):")
    for k in sortperm(rel, rev = true)[1:min(8, nP)]
        @printf("     k=%2d  anal=% .4e  fd=% .4e  ratio=% .3f\n", k, g_anal[k], g_fd[k], g_fd[k] / g_anal[k])
    end
end

run(0.0)
run(0.7)
