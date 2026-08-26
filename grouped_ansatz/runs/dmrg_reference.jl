# dmrg_reference.jl — real-space DMRG reference for the SAME t-V Hamiltonian
# the grouped momentum-space ansatz targets, at fixed L, scanning bond
# dimension χ and recording ground-state energy + entanglement entropy at
# each bond — a fully independent cross-check to compare against
# grouped_ansatz/chi_scan.jl's momentum-space entropy-vs-χ results.
#
# BOUNDARY CONDITION: closes the chain with a TWISTED hopping bond between
# site 1 and site L (c†_1 c_L picks up an extra phase e^{iθ}; θ=π is the
# "antiperiodic" twist, matching the half-odd-integer momentum quantization
# k(i) = -π + π(2i-1)/L used by exact_gs_energy() elsewhere in this project)
# plus a plain (untwisted) periodic interaction bond between site 1 and L —
# the interaction is phase-independent (density-density), so "twisting" only
# ever applies to the hopping term. Long-range fermionic MPO terms aren't
# built automatically by @mpoham the way nearest-neighbour terms are (unlike
# spin models, e.g. σᶻᶻ{i,j}, a c⁺c⁻-type term needs correctly-sided :L/:R
# operators AND — confirmed empirically — @mpoham DOES thread the Jordan-Wigner
# string through the intermediate sites correctly for a term{lattice[1],lattice[L]}
# placement, validated to 8 digits against an independent real-space ED at
# L=4 (E_DMRG = E_ED = -3.37228132) before trusting it here.
#
# EXACT MATCH TO exact_gs_energy(V,L) (demo.jl/chi_scan.jl): resolved by
# comparing the FULL eigenvalue spectrum (not just the ground state — a
# single-eigenvalue match can be coincidental at small L) between the
# momentum-space Hamiltonian's N=L/2 sector and several real-space
# candidates. The match: antiperiodic hopping (as above) + PLAIN nearest-
# neighbour interaction 2V·n_i·n_{i+1} (note the factor of 2 — NOT V·n_i·n_{i+1},
# and NOT the particle-hole-shifted (n-1/2)(n-1/2) form tried earlier).
# Verified to ~1e-14 at both L=4 and L=8 against exact_gs_energy. So this
# script now targets the IDENTICAL Hamiltonian used throughout the rest of
# this project — E_DMRG(L,V) IS directly, bit-for-bit comparable to
# exact_gs_energy(V,L) and to the momentum-space ansatz's own results.
#
# Real-space entanglement entropy is ALSO a different quantity than the
# momentum-space entropy tracked in chi_scan.jl — see [[finding-two-entanglement-notions]]:
# real-space entropy for a gapless 1D chain follows the Calabrese-Cardy
# ~(c/6)ln(L) open-chain scaling law (bond-position-dependent, peaked at the
# chain center), NOT the momentum-space Fermi-surface-peaked profile. Useful
# comparison here is: (a) energy convergence vs χ, and (b) how much bond
# dimension real-space DMRG needs vs. how much χ the momentum-space ansatz
# needs — not a literal one-to-one entropy match.
#
# Run from the repo root:   julia --project=. grouped_ansatz/runs/dmrg_reference.jl

using TensorKit, TensorOperations, LinearAlgebra, Printf, Statistics
using MPSKit, MPSKitModels, Plots
using JLD2
using SparseArrays, KrylovKit

# ---------------------------------------------------------------------------
# SCAN SETTINGS
# ---------------------------------------------------------------------------
L        = 8
χ_list   = [8]
t_hop    = 1.0
V_target = 0.5
twist_θ  = π    # antiperiodic (θ=π) closure of the 1↔L hopping bond; θ=0 would be plain periodic

folder_name = "grouped_ansatz/output/dmrg_reference"
mkpath(folder_name)

# ---------------------------------------------------------------------------
# U1(particle number) × FermionParity symmetric spinless-fermion operators.
# MPSKitModels' own c_plus/c_min (fermionoperators.jl) only carry FermionParity
# (no U1) — DMRG with just that symmetry doesn't conserve total particle
# number, and empirically converges to the WRONG total-N sector (confirmed:
# an unconstrained-N run at L=6 landed on E=-3.489 vs the true GS -4.196,
# fully converged (ϵ~1e-16) but in the wrong sector — total charge just isn't
# pinned without U1). Fix: hand-build the same operators (mirroring
# MPSKitModels' fermionoperators.jl construction exactly) with the combined
# sector FermionParity ⊠ U1Irrep, then pin the FiniteMPS boundary charges to
# force exactly L÷2 particles — this is the same boundary-charge-pinning
# trick src/src.jl's get_start uses for the momentum-space ansatz (there with
# an extra ZNIrrep momentum charge that's irrelevant here in real space).
# ---------------------------------------------------------------------------
const Sec = FermionParity ⊠ U1Irrep

function c_plus_sym(elt = ComplexF64; side = :L)
    vspace = Vect[Sec]((1, 1) => 1)
    if side === :L
        pspace = Vect[Sec]((0, 0) => 1, (1, 1) => 1)
        c⁺ = zeros(elt, pspace ← pspace ⊗ vspace)
        block(c⁺, Sec(1, 1)) .= one(elt)
    elseif side === :R
        C = c_plus_sym(elt; side = :L)
        F = isomorphism(storagetype(C), vspace, flip(vspace))
        @planar c⁺[-1 -2; -3] := C[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return c⁺
end

function c_min_sym(elt = ComplexF64; side = :L)
    if side === :L
        C = c_plus_sym(elt; side = :L)'
        F = isomorphism(flip(space(C, 2)), space(C, 2))
        @planar c⁻[-1; -2 -3] := C[-1 1; -2] * F[-3; 1]
    elseif side === :R
        c⁻ = TensorKit.permute(c_plus_sym(elt; side = :L)', ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return c⁻
end

function c_number_sym(elt = ComplexF64)
    pspace = Vect[Sec]((0, 0) => 1, (1, 1) => 1)
    n = zeros(elt, pspace ← pspace)
    block(n, Sec(1, 1)) .= one(elt)
    return n
end

# ---------------------------------------------------------------------------
# Real-space t-V chain MPO, twisted-boundary closure — EXACT match to
# exact_gs_energy(V,L):
#   H = -t Σ_{i=1}^{L-1} (c†_i c_{i+1} + h.c.) + 2V Σ_{i=1}^{L-1} n_i n_{i+1}
#       - t·e^{iθ} c†_1 c_L - t·e^{-iθ} c†_L c_1 + 2V n_1 n_L
# PLAIN interaction n_i n_{i+1} (not the particle-hole-shifted (n-1/2)(n-1/2)
# form tried earlier) with coefficient 2V (not V) — see the file header for
# how this was pinned down (full-spectrum comparison against the momentum-
# space Hamiltonian, not just ground-state energy). Since plain n·n interaction
# is NOT particle-hole symmetric, filling is no longer auto-pinned to L/2 by
# symmetry — the explicit U1 boundary-charge pinning (Vleft/Vright below) is
# now the ONLY thing enforcing L/2 filling, not "belt and braces" redundancy.
# For real θ (as here) the twisted hop coefficient e^{iθ} is just ±1, so TB
# gets scaled by -cos(θ) on the wraparound bond (θ=π ⇒ +t, the antiperiodic
# sign flip; θ=0 ⇒ -t, plain periodic).
# ---------------------------------------------------------------------------
function build_H(L, t, V; twist = π)
    n = c_number_sym(ComplexF64)
    pspace = space(n, 1)

    hop = MPSKitModels.contract_twosite(c_plus_sym(ComplexF64; side = :L), c_min_sym(ComplexF64; side = :R)) +
          MPSKitModels.contract_twosite(c_min_sym(ComplexF64; side = :L), c_plus_sym(ComplexF64; side = :R))
    Vint = MPSKitModels.contract_twosite(n, n)

    TB = rmul!(copy(hop), -t)
    VV = rmul!(copy(Vint), 2V)
    TBtwist = rmul!(copy(hop), -t * cos(twist))

    lattice = FiniteChain(L)
    H = @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return TB{i, j} + VV{i, j}
        end + TBtwist{lattice[1], lattice[L]} + VV{lattice[1], lattice[L]}
    end
    return H, pspace
end

# ---------------------------------------------------------------------------
# Independent exact-diagonalization reference, matched to build_H(): sparse +
# restricted to the N=L÷2 particle sector (dense full Fock space is 2^L × 2^L
# — infeasible at L=16; sector-restricted dimension is C(L, L÷2) = 12870 at
# L=16, trivial for a Krylov solve). Directly verified against
# exact_gs_energy(V,L) to ~1e-14 at L=4 and L=8 — see build_H's docstring for
# how this Hamiltonian form was pinned down and the L=4 long-range/JW-string
# validation against DMRG at L=4.
# ---------------------------------------------------------------------------
function ed_reference(L, t, V; twist = π)
    Nn = L ÷ 2
    occ(s, i) = (s >> (i - 1)) & 1
    cnt_before(s, i) = count_ones(s & ((1 << (i - 1)) - 1))
    ann(s, i) = occ(s, i) == 0 ? (0, 0) : ((-1)^cnt_before(s, i), s ⊻ (1 << (i - 1)))
    cre(s, i) = occ(s, i) == 1 ? (0, 0) : ((-1)^cnt_before(s, i), s ⊻ (1 << (i - 1)))

    hop_bonds = [(i, i + 1, -t) for i in 1:L-1]
    push!(hop_bonds, (1, L, -t * cos(twist)))
    int_bonds = [(i, i + 1, 2V) for i in 1:L-1]
    push!(int_bonds, (1, L, 2V))

    states = [s for s in 0:(1 << L)-1 if count_ones(s) == Nn]
    D = length(states)
    idxof = Dict(s => i for (i, s) in enumerate(states))
    I = Int[]; J = Int[]; W = Float64[]
    for (col, s) in enumerate(states)
        e = 0.0
        for (i, j, Vij) in int_bonds
            e += Vij * occ(s, i) * occ(s, j)
        end
        push!(I, col); push!(J, col); push!(W, e)
        for (i, j, coeff) in hop_bonds
            x, st = ann(s, j)
            if x != 0
                y, st2 = cre(st, i)
                y != 0 && (push!(I, idxof[st2]); push!(J, col); push!(W, coeff * x * y))
            end
            x, st = ann(s, i)
            if x != 0
                y, st2 = cre(st, j)
                y != 0 && (push!(I, idxof[st2]); push!(J, col); push!(W, coeff * x * y))
            end
        end
    end
    H = sparse(I, J, W, D, D)
    vals, _ = eigsolve(H, randn(D), 1, :SR; ishermitian = true, tol = 1e-10, krylovdim = 40)
    return real(vals[1])
end

Eref = ed_reference(L, t_hop, V_target; twist = twist_θ)
println("twisted-boundary (θ=$(round(twist_θ, digits=4))) exact-diagonalization GS (L=$L, t=$t_hop, V=$V_target) = $(round(Eref, digits=8))")
println("χ scan: ", χ_list)

# ---------------------------------------------------------------------------
# DMRG scan over χ
# ---------------------------------------------------------------------------
H, pspace = build_H(L, t_hop, V_target; twist = twist_θ)

Nhalf = L ÷ 2
Vleft  = Vect[Sec]((0, 0) => 1)
Vright = Vect[Sec]((mod(Nhalf, 2), Nhalf) => 1)

results = Dict{Int, NamedTuple}()

for χ in χ_list
    println("\n", "="^70)
    println("χ = $χ   (L=$L, t=$t_hop, V=$V_target, DMRG2, twisted boundary θ=$(round(twist_θ,digits=4)))")
    println("="^70)

    # initial bond space: generous per-charge-sector allocation so DMRG2's
    # trscheme=truncrank(χ) (not the initial space) is what actually limits χ.
    per_sector = max(2, χ)
    vspace_mid = Vect[Sec]([(mod(k, 2), k) => per_sector for k in 0:L]...)
    ψ0 = FiniteMPS(rand, ComplexF64, L, pspace, vspace_mid; left = Vleft, right = Vright)

    ψ, envs, ϵ = find_groundstate(
        ψ0, H, DMRG2(; trscheme = truncrank(χ), tol = 1e-10, maxiter = 200, verbosity = 0)
    )
    E = real(sum(expectation_value(ψ, H, envs)))

    ent = [entropy(ψ, b) for b in 1:L-1]
    ent = real.(ent)

    @printf "  E = %.8f   vs ED %.8f   (gap %+.3e)   final ϵ = %.2e\n" E Eref (E - Eref) ϵ
    @printf "  entanglement entropy per bond: %s\n" join(round.(ent, digits = 4), ", ")

    results[χ] = (E = E, gap = E - Eref, ent = ent, eps = ϵ)

    outfile = folder_name * "/dmrg_L$(L)_chi$(χ).jld2"
    jldsave(outfile; L, χ, t_hop, V_target, Eref, E, gap = E - Eref, ent_entropy = ent, eps = ϵ, mps_final = ψ)
    println("  wrote $outfile")
end

# ---------------------------------------------------------------------------
# Combined plots: entropy vs bond (one curve per χ), and energy vs χ
# convergence to the exact twisted-boundary reference.
# ---------------------------------------------------------------------------
plt_ent = plot(xlabel = "bond index (real-space chain)", ylabel = "entanglement entropy S",
               title = "Real-space DMRG entanglement entropy vs χ (L=$L, V=$V_target)",
               legend = :outertopright, left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
               size = (780, 480))
for χ in χ_list
    plot!(plt_ent, 1:(L-1), results[χ].ent, label = "χ=$χ", lw = 2, marker = :circle, markersize = 3)
end
savefig(plt_ent, folder_name * "/dmrg_entropy_L$(L).png")
println("\nwrote $(folder_name)/dmrg_entropy_L$(L).png")

plt_E = plot(χ_list, [results[χ].E for χ in χ_list], xlabel = "χ", ylabel = "E",
             title = "Real-space DMRG energy convergence (L=$L, V=$V_target)",
             label = "E_DMRG(χ)", lw = 2, marker = :circle, legend = :topright)
hline!(plt_E, [Eref], ls = :dash, lw = 2, label = "twisted-boundary exact GS")
savefig(plt_E, folder_name * "/dmrg_energy_L$(L).png")
println("wrote $(folder_name)/dmrg_energy_L$(L).png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
txtfile = folder_name * "/dmrg_summary_L$(L).txt"
open(txtfile, "w") do io
    @printf(io, "dmrg_reference summary — L=%d, t=%.2f, V=%.2f, twisted-boundary exact GS = %.8f\n\n",
            L, t_hop, V_target, Eref)
    @printf(io, "%6s %16s %14s %10s %14s\n", "chi", "E", "gap", "eps", "max(S)")
    for χ in χ_list
        r = results[χ]
        @printf(io, "%6d %16.8f %14.3e %10.2e %14.4f\n", χ, r.E, r.gap, r.eps, maximum(r.ent))
    end
end
println("wrote $txtfile")

println("\ndmrg_reference done. Files in $folder_name:")
for f in sort(readdir(folder_name))
    (endswith(f, ".jld2") || endswith(f, ".png") || endswith(f, ".txt")) && println("  $f")
end
