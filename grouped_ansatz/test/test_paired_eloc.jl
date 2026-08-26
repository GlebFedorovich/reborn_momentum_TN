# test_paired_eloc.jl
#
# Gate for the dim-4 (k,k+π) grouped-super-site ansatz's get_E_loc, mirroring
# test/test_eloc_mps.jl exactly (same identity, same ED reference Hamiltonian).
# This is the critical correctness check for the array-index relabeling trick
# described in grouped_ansatz/src/grouped_mps.jl: it exercises the momentum
# lookup (to_orig), the interaction q-shift remapping (to_orig/to_chain), and
# the 4-sector physical-space construction (sector_for) all at once, by
# checking that get_E_loc reproduces the exact variational energy of a REAL
# (non-ground-state, non-optimized) ansatz — no Monte Carlo noise, full
# deterministic sector enumeration:
#
#     ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩  ==  Σ_config  p(config) · E_loc(config) · L

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
using LinearAlgebra, Printf
const M = GroupedMomentumMPS

const L = 8
const N = L ÷ 2
const t = 1.0

idx_to_momentum(idx, L) = -π + π * (2idx - 1) / L

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
function apply4(state, a, b, c, d)               # c†_a c†_b c_c c_d, right-to-left
    s,  st = annihilate(state, d); s  == 0 && return (0, 0)
    s2, st = annihilate(st,  c);   s2 == 0 && return (0, 0); s *= s2
    s2, st = create(st, b);        s2 == 0 && return (0, 0); s *= s2
    s2, st = create(st, a);        s2 == 0 && return (0, 0); s *= s2
    (s, st)
end

# Reference momentum-space H, expressed in CHAIN (interleaved-pair) order —
# the same convention the grouped ansatz's get_overlap/get_E_loc use
# internally (see grouped_mps.jl's header comment). `states`/`occ`/`apply4`
# below index bits by CHAIN position; to get the correct physical momentum
# and interaction vertex we look up the ORIGINAL mode each chain slot
# represents via M.to_orig, exactly mirroring get_E_loc's own bookkeeping.
# (test/test_eloc_mps.jl's build_Hmom is the L-site analogue in the OTHER
# convention — plain ascending mode order — which is why it can't be reused
# verbatim here: the grouped ansatz's Fock-state convention is the
# interleaved creation order c_1† c_{1+L/2}† c_2† c_{2+L/2}† ... |0⟩, not the
# plain ascending order c_1† c_2† ... c_L†|0⟩.)
function build_Hmom(V)
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

# Build an MC at a fixed config, given DIRECTLY in CHAIN order (matching the
# ansatz's own convention — and now build_Hmom's convention too, via to_orig).
function mc_at_config(mps_chain, zpairs, config_chain, V)
    Vp = [M.space(mps_chain[i], 2) for i in eachindex(mps_chain)]
    w  = M.get_overlap(mps_chain, M.make_sample_string(config_chain, Vp, M.space(mps_chain[1], 1), zpairs, L))[1]
    params = Dict(
        :mps_chain => mps_chain, :L => L, :zpairs => zpairs, :t_hopping => t, :V_repulsion => V,
        :config => config_chain,
        :idx_filled => findall(==(1), config_chain), :idx_empty => findall(==(0), config_chain),
        :weight => w)
    return M.MC(params), w
end

function run_check(V; χ = 8)
    println("="^70)
    @printf("V = %.3f,  L = %d,  χ = %d,  sector dim D = %d  (GROUPED dim-4 ansatz)\n", V, L, χ, D)

    mps_chain, _, _, zpairs = M.get_start_fermi_sea(L, χ; ε = 0.5)

    ψ = zeros(Float64, D)
    for (i, s) in enumerate(states)
        _, ψ[i] = mc_at_config(mps_chain, zpairs, [occ(s, j) for j in 1:L], V)
    end

    psupp = abs.(ψ) .> 1e-12 * maximum(abs, ψ)
    @printf("  configs in momentum-0 support: %d / %d\n", count(psupp), D)

    eloc = zeros(Float64, D)
    for (i, s) in enumerate(states)
        psupp[i] || continue
        mc, _   = mc_at_config(mps_chain, zpairs, [occ(s, j) for j in 1:L], V)
        eloc[i] = M.get_E_loc(mc) * L
    end

    p = abs2.(ψ) ./ sum(abs2, ψ)

    Hmom    = build_Hmom(V)
    E_exact = (ψ' * Hmom * ψ) / (ψ' * ψ)
    E_vmc   = sum(p .* eloc)

    @printf("  exact ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩          : %.10f\n", E_exact)
    @printf("  Σ p·E_loc  (via get_E_loc)   : %.10f\n", E_vmc)
    @printf("  |difference|                 : %.3e   %s\n",
            abs(E_exact - E_vmc),
            abs(E_exact - E_vmc) < 1e-9 ? "MATCH" : "*** MISMATCH ***")
end

run_check(0.0)
run_check(0.7)
