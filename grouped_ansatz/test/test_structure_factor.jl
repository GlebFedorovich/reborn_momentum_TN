# test_structure_factor.jl
#
# Gate for the proposed density-structure-factor measurement S(q) =
# (1/L) Σ_k ⟨n_k n_{k+q}⟩ — the diagonal-in-occupation-number CDW order
# parameter we want to add to Carlo.measure! in grouped_mps.jl. Since n_k is
# diagonal in the sampling basis, the only real bug risk is index bookkeeping
# (mc.config lives in CHAIN order; to get a genuine physical momentum shift q
# you have to go through M.to_orig, exactly like get_E_loc's interaction term
# does — see grouped_mps.jl and test_paired_eloc.jl for the same caveat).
#
# Two fully independent Hamiltonian/eigenvector constructions, both already
# validated to reproduce the same ground-state ENERGY (test_paired_eloc.jl /
# chi_scan.jl's exact_gs_energy):
#   (A) build_Hmom_chain — chain (interleaved-pair) order, momentum via
#       M.to_orig(k,L) — same convention mc.config uses.
#   (B) exact_gs_state — plain ascending-mode order, no permutation at all
#       (chi_scan.jl's exact_gs_energy, extended to also return the
#       eigenvector).
# If S(q) computed via convention (A)+to_orig matches S(q) computed via
# convention (B) with no permutation, for the SAME physical ground state,
# the to_orig-based measurement formula is correct.

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
using LinearAlgebra, SparseArrays, Printf
const M = GroupedMomentumMPS

const L = 8
const N = L ÷ 2
const t = 1.0

idx_to_momentum(idx, L) = -π + π * (2idx - 1) / L

# ---------------------------------------------------------------------------
# (A) Chain-order Hamiltonian — identical to test_paired_eloc.jl's build_Hmom.
# ---------------------------------------------------------------------------
occ(state, i)         = (state >> (i - 1)) & 1
count_below(state, i) = count_ones(state & ((1 << (i - 1)) - 1))
const states = [s for s in 0:(1 << L)-1 if count_ones(s) == N]
const idxof  = Dict(s => i for (i, s) in enumerate(states))
const D      = length(states)

function annihilate(state, i)
    occ(state, i) == 0 && return (0, 0)
    ((-1)^count_below(state, i), state ⊻ (1 << (i - 1)))
end
function create(state, i)
    occ(state, i) == 1 && return (0, 0)
    ((-1)^count_below(state, i), state ⊻ (1 << (i - 1)))
end
function apply4(state, a, b, c, d)
    s,  st = annihilate(state, d); s  == 0 && return (0, 0)
    s2, st = annihilate(st,  c);   s2 == 0 && return (0, 0); s *= s2
    s2, st = create(st, b);        s2 == 0 && return (0, 0); s *= s2
    s2, st = create(st, a);        s2 == 0 && return (0, 0); s *= s2
    (s, st)
end

function build_Hmom_chain(V)
    Uq(qidx) = V * cos(2π * qidx / L)
    H = zeros(Float64, D, D)
    for (i, s) in enumerate(states)
        e = 0.0
        for k in 1:L
            occ(s, k) == 1 && (e += -2t * cos(idx_to_momentum(M.to_orig(k, L), L)))
        end
        H[i, i] += e
        for k1 in 1:L, k2 in 1:L, qidx in 0:L-1
            orig1 = M.to_orig(k1, L); orig2 = M.to_orig(k2, L)
            orig1p = mod1(orig1 + qidx, L); orig2p = mod1(orig2 - qidx, L)
            a = M.to_chain(orig1p, L); b = M.to_chain(orig2p, L)
            sgn, st = apply4(s, a, b, k2, k1)
            sgn == 0 && continue
            H[idxof[st], i] += (1 / L) * Uq(qidx) * sgn
        end
    end
    H
end

# S(q) via chain-order eigenvector: for each chain-order basis state, reorder
# its occupation bits into PHYSICAL momentum order via to_orig — the exact
# mapping the live Carlo.measure! code (mc.config -> n_orig) will use.
function Sq_chain_order(ψ, q)
    p = abs2.(ψ) ./ sum(abs2, ψ)
    acc = 0.0
    for (i, s) in enumerate(states)
        n_orig = zeros(Int, L)
        for k in 1:L
            n_orig[M.to_orig(k, L)] = occ(s, k)
        end
        acc += p[i] * sum(n_orig[κ] * n_orig[mod1(κ + q, L)] for κ in 1:L)
    end
    return acc / L
end

# ---------------------------------------------------------------------------
# (B) Plain ascending-mode-order ED — chi_scan.jl's exact_gs_energy, extended
# to also return the eigenvector. No permutation: chain-slot κ IS mode κ.
# ---------------------------------------------------------------------------
function exact_gs_state(V, L)
    Nn = L ÷ 2
    k(i) = idx_to_momentum(i, L)
    occ2(s, i) = (s >> (i - 1)) & 1; cb(s, i) = count_ones(s & ((1 << (i - 1)) - 1))
    ann(s, i) = occ2(s, i) == 0 ? (0, 0) : ((-1)^cb(s, i), s ⊻ (1 << (i - 1)))
    cre(s, i) = occ2(s, i) == 1 ? (0, 0) : ((-1)^cb(s, i), s ⊻ (1 << (i - 1)))
    function ap4(s, a, b, c, d)
        x, st = ann(s, d); x == 0 && return (0, 0)
        y, st = ann(st, c); y == 0 && return (0, 0); x *= y
        y, st = cre(st, b); y == 0 && return (0, 0); x *= y
        y, st = cre(st, a); y == 0 && return (0, 0); x *= y
        (x, st)
    end
    states2 = [s for s in 0:(1 << L)-1 if count_ones(s) == Nn &&
              sum(i -> occ2(s, i) == 1 ? (2i - 1) : 0, 1:L) % (2L) == 0]
    D2 = length(states2)
    idxof2 = Dict(s => i for (i, s) in enumerate(states2))
    I = Int[]; J = Int[]; W = Float64[]
    for (col, s) in enumerate(states2)
        e = 0.0; for κ in 1:L; occ2(s, κ) == 1 && (e += -2cos(k(κ))); end
        push!(I, col); push!(J, col); push!(W, e)
        for k1 in 1:L, k2 in 1:L, q in 0:L-1
            a = mod1(k1 + q, L); b = mod1(k2 - q, L); sg, st = ap4(s, a, b, k2, k1); sg == 0 && continue
            push!(I, idxof2[st]); push!(J, col); push!(W, (1 / L) * V * cos(2π * q / L) * sg)
        end
    end
    H = sparse(I, J, W, D2, D2)
    F = eigen(Symmetric(Matrix(H)))
    return F.values[1], F.vectors[:, 1], states2
end

function Sq_plain_order(ψ, states2, q)
    occ2(s, i) = (s >> (i - 1)) & 1
    p = abs2.(ψ) ./ sum(abs2, ψ)
    acc = 0.0
    for (i, s) in enumerate(states2)
        acc += p[i] * sum(occ2(s, κ) * occ2(s, mod1(κ + q, L)) for κ in 1:L)
    end
    return acc / L
end

function run_check(V)
    println("="^70)
    @printf("V = %.3f   L=%d\n", V, L)

    Fc = eigen(Symmetric(build_Hmom_chain(V)))
    ψ_chain, E_chain = Fc.vectors[:, 1], Fc.values[1]

    E_plain, ψ_plain, states2 = exact_gs_state(V, L)

    @printf("  E (chain-order Hmom) = %.10f\n", E_chain)
    @printf("  E (plain-order ED)   = %.10f\n", E_plain)
    @printf("  |ΔE| = %.3e   %s\n", abs(E_chain - E_plain), abs(E_chain - E_plain) < 1e-9 ? "energies MATCH" : "*** ENERGY MISMATCH ***")

    all_ok = true
    for q in 0:L-1
        Sc = Sq_chain_order(ψ_chain, q)
        Sp = Sq_plain_order(ψ_plain, states2, q)
        ok = abs(Sc - Sp) < 1e-9
        all_ok &= ok
        @printf("  q_idx=%2d (q=%+.4f):  S_chain=%.8f   S_plain=%.8f   |Δ|=%.3e   %s\n",
                q, 2π * q / L, Sc, Sp, abs(Sc - Sp), ok ? "MATCH" : "*** MISMATCH ***")
    end
    println(all_ok ? "  ALL S(q) MATCH" : "  *** S(q) MISMATCH DETECTED ***")
end

run_check(0.5)   # LL side
run_check(3.0)   # CDW side
