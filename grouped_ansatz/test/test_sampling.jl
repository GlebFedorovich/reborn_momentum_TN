# test_sampling.jl — grouped-ansatz analogue of test/test_canonical.jl.
#
# Verifies the canonical-form foundation for the 4-way direct sampler
# (draw_from_B's generalization of _draw_branch from 2 to 4 branches):
#
#   (1) Gauge consistency: amplitudes from the cached right-canonical (AR)
#       tensors B equal those from the raw mps_chain up to ONE global scalar
#       (same for every config) — so B can be the single source of truth for
#       both sampling and get_E_loc's amplitude ratios.
#   (2) Right-canonical sampling: drawing pair-site-by-pair-site from B
#       reproduces the exact |ψ|²/‖ψ‖² distribution over the N=L/2 sector.
#
# Configs throughout are in CHAIN order (matching every other file in this
# directory) — draw_from_B/get_E_loc/get_overlap all assume that convention.

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
using TensorKit, MPSKit, Random, Printf
const M = GroupedMomentumMPS

L = 8; χ = 8; V = 0.7
Random.seed!(1)

mps, Vphys, Vleft, zpairs = M.get_start_fermi_sea(L, χ; ε = 0.6)

B = M.canonical_B(mps)

amp_raw(c)   = M.get_overlap(mps, M.make_sample_string(c, Vphys, Vleft[1], zpairs, L))[1]
amp_canon(c) = M.get_overlap(B,   M.make_sample_string(c, Vphys, Vleft[1], zpairs, L))[1]

# enumerate the N = L/2 sector (chain-order bitmask, same convention as
# test_paired_eloc.jl / test_grad_fd.jl)
N = L ÷ 2
configs = Vector{Int}[]
for s in 0:(1 << L) - 1
    count_ones(s) == N || continue
    push!(configs, [(s >> (i - 1)) & 1 for i in 1:L])
end
key(c) = sum(c[i] << (i - 1) for i in 1:L)
@show length(configs)

## --- (0) right-canonical isometry check -----------------------------------
for i in eachindex(B)
    A   = permute(B[i], ((1,), (2, 3)))
    ρ   = A * A'
    dev = norm(ρ - id(space(B[i], 1)))
    @printf("pair-site %d: ‖B·Bᵀ − 1‖ = %.2e   (dim V_left = %d)  %s\n",
            i, dev, dim(space(B[i], 1)), dev < 1e-10 ? "OK" : "*** not right-canonical ***")
end

## --- (1) AR vs raw amplitudes differ only by a single global scalar -------
ratios = Float64[]
for c in configs
    wr = amp_raw(c)
    abs(wr) < 1e-12 && continue
    push!(ratios, amp_canon(c) / wr)
end
r0 = ratios[1]
maxdev = maximum(abs.(ratios ./ r0 .- 1))
@printf("(1) global-scalar consistency: max rel. deviation = %.2e  %s\n",
        maxdev, maxdev < 1e-10 ? "OK (single scalar)" : "*** NOT a single scalar ***")
@printf("    amp_canon / amp_raw ≈ %.6g\n", r0)

## --- (2a) right-canonical normalization Σ|ψ_s^canon|² = ‖ψ‖²_AR -----------
@printf("(2a) Σ |ψ_s^canon|² over sector = %.6f\n", sum(abs2(amp_canon(c)) for c in configs))

## --- (2b) empirical draw frequencies (4-way branching) vs exact |ψ|²/Σ ----
probs = [abs2(amp_raw(c)) for c in configs]; probs ./= sum(probs)

Ndraw = 1000
print("    drawing $Ndraw configs (draw_from_B, α=1) ... "); flush(stdout)
counts = Dict{Int,Int}()
for _ in 1:Ndraw
    c, _, iw = M.draw_from_B(B, zpairs, L)   # α=1 ⇒ iw should be ≡ 1
    abs(iw - 1.0) < 1e-10 || error("α=1 draw has iw=$iw ≠ 1 — tempering bug")
    counts[key(c)] = get(counts, key(c), 0) + 1
end
println("done")

maxdev2 = maximum(abs(get(counts, key(c), 0) / Ndraw - probs[i]) for (i, c) in enumerate(configs))
@printf("(2b) max |empirical - exact| = %.4f   (3σ ≈ %.4f)  %s\n",
        maxdev2, 3 * sqrt(0.25 / Ndraw),
        maxdev2 < 4 * sqrt(0.25 / Ndraw) ? "OK" : "*** off ***")
