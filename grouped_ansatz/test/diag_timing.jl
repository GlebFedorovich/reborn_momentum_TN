# diag_timing.jl — isolate which piece of the pipeline (sampling / get_E_loc /
# get_partial_G+gradient-block-iteration) is responsible for the runaway
# compile-time blowup seen when running through Carlo.jl's scheduler, by
# timing each piece directly (bypassing Carlo entirely).

include("../src/grouped_mps.jl")
using .GroupedMomentumMPS
using TensorKit, Printf
const M = GroupedMomentumMPS

L = 8; χ = 8; V = 1.0
mps_chain, V_phys_list, V_left_list, zpairs = M.get_start_fermi_sea(L, χ)
B = M.canonical_B(mps_chain)

config, sample_string, iw = M.draw_from_B(B, zpairs, L)
w = M.get_overlap(mps_chain, sample_string)[1]

params = Dict(:mps_chain => mps_chain, :L => L, :zpairs => zpairs, :t_hopping => 1.0,
              :V_repulsion => V, :config => config,
              :idx_filled => findall(==(1), config), :idx_empty => findall(==(0), config),
              :weight => w)
mc = M.MC(params)
mc.B = B

println("--- draw_from_B / sample_direct! (N=200) ---")
@time for _ in 1:5
    M.sample_direct!(mc)
end
@time for _ in 1:200
    M.sample_direct!(mc)
end

println("--- get_E_loc (N=200) ---")
@time for _ in 1:5
    M.get_E_loc(mc)
end
@time for _ in 1:200
    M.get_E_loc(mc)
end

println("--- get_partial_G + flatten_gradient (N=200) ---")
V_phys_list2 = [M.space(mc.mps_chain[i], 2) for i in eachindex(mc.mps_chain)]
mps_init = M.make_sample_string(mc.config, V_phys_list2, M.space(mc.mps_chain[1], 1), mc.zpairs, mc.L)
@time for _ in 1:5
    G = M.get_partial_G(mc.mps_chain, mps_init, M.space(mps_init[1], 1), M.space(mps_init[end], 3))
    M.flatten_gradient(G)
end
@time for _ in 1:200
    G = M.get_partial_G(mc.mps_chain, mps_init, M.space(mps_init[1], 1), M.space(mps_init[end], 3))
    M.flatten_gradient(G)
end

println("--- FULL measure!-equivalent body (N=50) ---")
function measure_body(mc)
    f_current = M.get_E_loc(mc)
    V_phys_list3 = [M.space(mc.mps_chain[i], 2) for i in eachindex(mc.mps_chain)]
    mps_init2 = M.make_sample_string(mc.config, V_phys_list3, M.space(mc.mps_chain[1], 1), mc.zpairs, mc.L)
    Gradient = M.get_partial_G(mc.mps_chain, mps_init2, M.space(mps_init2[1], 1), M.space(mps_init2[end], 3)) * (1 / mc.weight)
    Gradient_times_E_loc = Gradient * f_current
    o_vec = M.flatten_gradient(Gradient)
    oe = o_vec .* f_current
    oo = vec(o_vec * o_vec')
    return f_current, o_vec, oe, oo
end
@time for _ in 1:5
    measure_body(mc)
end
@time for _ in 1:50
    measure_body(mc)
end

println("DONE")
