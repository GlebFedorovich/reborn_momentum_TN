# phase_scan_dmrg.jl — real-space DMRG counterpart to phase_scan_vmc.jl:
# same L, same χ_list, same V_list straddling the V=2t LL→CDW transition,
# using the exact-matched Hamiltonian construction validated in
# dmrg_reference.jl (antiperiodic hop + plain V·n_i·n_{i+1} interaction —
# see that file's header for how this was pinned down).
#
# Run from the repo root:   julia --project=. grouped_ansatz/runs/phase_scan_dmrg.jl
#
# Output: grouped_ansatz/output/phase_scan/dmrg/dmrg_L{L}_V{V}_chi{χ}.jld2

using TensorKit, TensorOperations, LinearAlgebra, Printf, Statistics
using MPSKit, MPSKitModels, Plots
using JLD2
using SparseArrays, KrylovKit

# ---------------------------------------------------------------------------
# SCAN SETTINGS — identical L/χ_list/V_list to phase_scan_vmc.jl.
# ---------------------------------------------------------------------------
L        = 16
χ_list   = [16, 20, 24]
V_list   = [1.0, 3.0]
t_hop    = 1.0
twist_θ  = π

folder_name = "grouped_ansatz/output/phase_scan_mpi/dmrg"
mkpath(folder_name)

# ---------------------------------------------------------------------------
# U1(particle number) × FermionParity symmetric operators — identical to
# dmrg_reference.jl.
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

function build_H(L, t, V; twist = π)
    n = c_number_sym(ComplexF64)
    pspace = space(n, 1)

    hop = MPSKitModels.contract_twosite(c_plus_sym(ComplexF64; side = :L), c_min_sym(ComplexF64; side = :R)) +
          MPSKitModels.contract_twosite(c_min_sym(ComplexF64; side = :L), c_plus_sym(ComplexF64; side = :R))
    Vint = MPSKitModels.contract_twosite(n, n)

    TB = rmul!(copy(hop), -t)
    VV = rmul!(copy(Vint), V)
    TBtwist = rmul!(copy(hop), -t * cos(twist))

    lattice = FiniteChain(L)
    H = @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return TB{i, j} + VV{i, j}
        end + TBtwist{lattice[1], lattice[L]} + VV{lattice[1], lattice[L]}
    end
    return H, pspace
end

function ed_reference(L, t, V; twist = π)
    Nn = L ÷ 2
    occ(s, i) = (s >> (i - 1)) & 1
    cnt_before(s, i) = count_ones(s & ((1 << (i - 1)) - 1))
    ann(s, i) = occ(s, i) == 0 ? (0, 0) : ((-1)^cnt_before(s, i), s ⊻ (1 << (i - 1)))
    cre(s, i) = occ(s, i) == 1 ? (0, 0) : ((-1)^cnt_before(s, i), s ⊻ (1 << (i - 1)))

    hop_bonds = [(i, i + 1, -t) for i in 1:L-1]
    push!(hop_bonds, (1, L, -t * cos(twist)))
    int_bonds = [(i, i + 1, V) for i in 1:L-1]
    push!(int_bonds, (1, L, V))

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

# ---------------------------------------------------------------------------
# Scan: V (phase) outer, χ inner.
# ---------------------------------------------------------------------------
all_results = Dict{Float64, Dict{Int, NamedTuple}}()

for V_target in V_list
    phase_label = V_target < 2.0 ? "LL" : (V_target > 2.0 ? "CDW" : "critical")
    Eref = ed_reference(L, t_hop, V_target; twist = twist_θ)
    println("\n", "#"^70)
    println("V = $V_target  ($phase_label phase, Δ=$(V_target/2))   twisted-boundary ED GS = $(round(Eref, digits=8))")
    println("#"^70)

    H, pspace = build_H(L, t_hop, V_target; twist = twist_θ)
    Nhalf = L ÷ 2
    Vleft  = Vect[Sec]((0, 0) => 1)
    Vright = Vect[Sec]((mod(Nhalf, 2), Nhalf) => 1)

    results = Dict{Int, NamedTuple}()

    for χ in χ_list
        println("\n", "="^70)
        println("V=$V_target ($phase_label)   χ = $χ   (L=$L, DMRG2, twisted boundary θ=$(round(twist_θ,digits=4)))")
        println("="^70)

        per_sector = max(2, χ)
        vspace_mid = Vect[Sec]([(mod(k, 2), k) => per_sector for k in 0:L]...)
        ψ0 = FiniteMPS(rand, ComplexF64, L, pspace, vspace_mid; left = Vleft, right = Vright)

        ψ, envs, ϵ = find_groundstate(
            ψ0, H, DMRG2(; trscheme = truncrank(χ), tol = 1e-10, maxiter = 200, verbosity = 0)
        )
        E = real(sum(expectation_value(ψ, H, envs)))

        ent = real.([entropy(ψ, b) for b in 1:L-1])

        @printf "  E = %.8f   vs ED %.8f   (gap %+.3e)   final ϵ = %.2e\n" E Eref (E - Eref) ϵ
        @printf "  entanglement entropy per bond: %s\n" join(round.(ent, digits = 4), ", ")

        results[χ] = (E = E, gap = E - Eref, ent = ent, eps = ϵ)

        outfile = folder_name * "/dmrg_L$(L)_V$(V_target)_chi$(χ).jld2"
        jldsave(outfile; L, χ, t_hop, V_target, phase_label, Eref, E, gap = E - Eref, ent_entropy = ent, eps = ϵ, mps_final = ψ)
        println("  wrote $outfile")
    end

    all_results[V_target] = results

    plt_ent = plot(xlabel = "bond index (real-space chain)", ylabel = "entanglement entropy S",
                   title = "Real-space DMRG entanglement entropy vs χ (L=$L, V=$V_target, $phase_label)",
                   legend = :outertopright, left_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
                   size = (780, 480))
    for χ in χ_list
        plot!(plt_ent, 1:(L-1), results[χ].ent, label = "χ=$χ", lw = 2, marker = :circle, markersize = 3)
    end
    savefig(plt_ent, folder_name * "/dmrg_entropy_L$(L)_V$(V_target).png")
    println("wrote $(folder_name)/dmrg_entropy_L$(L)_V$(V_target).png")
end

# ---------------------------------------------------------------------------
# Final summary table.
# ---------------------------------------------------------------------------
txtfile = folder_name * "/phase_scan_summary_L$(L).txt"
open(txtfile, "w") do io
    @printf(io, "phase_scan_dmrg summary — L=%d, t=%.2f\n\n", L, t_hop)
    @printf(io, "%6s %6s %16s %14s %10s %10s\n", "V", "chi", "E", "gap", "eps", "max(S)")
    for V_target in V_list
        for χ in χ_list
            r = all_results[V_target][χ]
            @printf(io, "%6.1f %6d %16.8f %14.3e %10.2e %10.4f\n",
                    V_target, χ, r.E, r.gap, r.eps, maximum(r.ent))
        end
    end
end
println("\nwrote $txtfile")
println("\nphase_scan_dmrg done. Files in $folder_name")
