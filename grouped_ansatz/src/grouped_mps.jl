module GroupedMomentumMPS

# ============================================================================
# Dim-4 (k, k+π) grouped super-site momentum-space MPS.
#
# THE ANSATZ: instead of one MPS site per momentum mode k (local dim ≤2:
# empty/occupied), pair mode k with k+π into a SINGLE site of local dim ≤4:
# {∅, k only, k+π only, both}. 
# array index 1,2,3,4,...,L  ↔  original mode  1, 1+L/2, 2, 2+L/2, ..., L/2, L

using TensorKit
using Random
using Carlo, MPSKit, HDF5
using TensorOperations: @tensor

# ----------------------------------------------------------------------------
# index relabeling: array position (1..L, "chain order") <-> original
# ascending momentum-mode number (1..L, the numbering used by idx_to_momentum
# and by the ZNIrrep{2L} charge convention below).
#   chain slot 2s-1  <->  mode s        (s = 1..L/2)
#   chain slot 2s    <->  mode s + L/2
# ----------------------------------------------------------------------------
to_orig(idx::Int, L::Int) = isodd(idx) ? (idx + 1) ÷ 2 : idx ÷ 2 + L ÷ 2
to_chain(mode::Int, L::Int) = mode <= L ÷ 2 ? 2 * mode - 1 : 2 * (mode - L ÷ 2)

# helper function to convert an ORIGINAL momentum-mode index to its physical momentum value
idx_to_momentum(idx::Int, L::Int) = -π + π * (2 * idx - 1) / L

# ZNIrrep{2L} charge label of ORIGINAL mode i 
zcharge(i::Int, L::Int) = -2 * L - 1 + 2 * i

mutable struct MC <: AbstractMC
    mps_chain
    B                       # cached right-canonical (AR) tensors; see canonical_B
    L::Int                  # total number of ORIGINAL momentum modes (chain has L/2 sites)
    zpairs::Vector{Tuple{Int,Int}}   # (z1,z2) charge labels for each pair site
    t::Float64
    V::Float64
    config::Vector{Int}     # length L, CHAIN-order occupations
    idx_filled::Vector{Int}
    idx_empty::Vector{Int}
    weight::Float64
    total_attempts::Int64
    accepted_flips::Int64
    importance_weight::Float64
    alpha::Float64

    function MC(params::Dict)
        new(
            params[:mps_chain],
            canonical_B(params[:mps_chain]),
            params[:L],
            params[:zpairs],
            params[:t_hopping],
            params[:V_repulsion],
            params[:config],
            params[:idx_filled],
            params[:idx_empty],
            params[:weight],
            0,
            0,
            1.0,
            get(params, :alpha, 1.0),
        )
    end
end

# ----------------------------------------------------------------------------
# Physical-space construction
# ----------------------------------------------------------------------------

# The dim-4 physical space of pair site s: fuse(V_mode_i, V_mode_j) where
# i,j = to_orig(2s-1,L), to_orig(2s,L) = s, s+L/2. Returns the fused space and
# the two charge labels (needed later to pick out sectors by occupation).
function pair_phys_space(s::Int, L::Int, Sym)
    m1 = to_orig(2s - 1, L)
    m2 = to_orig(2s, L)
    z1 = zcharge(m1, L)
    z2 = zcharge(m2, L)
    V1 = Vect[Sym]((0, 0, 0) => 1, (1, 1, z1) => 1)
    V2 = Vect[Sym]((0, 0, 0) => 1, (1, 1, z2) => 1)
    return fuse(V1, V2), z1, z2
end

# Locate the (unique, multiplicity-1) sector of a pair-site physical space
# corresponding to occupations (n1,n2) of its two modes (charges z1,z2).
function sector_for(V_phys, n1::Int, n2::Int, z1::Int, z2::Int, twoL::Int)
    fp = (n1 + n2) % 2 # fermion parity of the pair site
    u1 = n1 + n2 # U(1) charge of the pair site
    zn = ZNIrrep{twoL}(n1 * z1 + n2 * z2) # ZN charge of the pair site
    return only(c for c in sectors(V_phys) if Int(c[1].isodd) == fp && c[2].charge == u1 && c[3] == zn)
end

# Build the L/2-site chain's physical + boundary bond spaces
function get_start(L::Int, χ_trunc::Int)
    M = L ÷ 2
    Sym = FermionParity ⊠ U1Irrep ⊠ ZNIrrep{2 * L}

    V_phys_list = []
    zpairs = Vector{Tuple{Int,Int}}(undef, M)
    V_left_list = [Vect[Sym]((0, 0, 0) => 1)]

    for s in 1:M-1
        Vp, z1, z2 = pair_phys_space(s, L, Sym)
        push!(V_phys_list, Vp)
        zpairs[s] = (z1, z2)
        V_right = fuse(V_left_list[end], Vp)
        push!(V_left_list, V_right)
    end

    Vp, z1, z2 = pair_phys_space(M, L, Sym)
    push!(V_phys_list, Vp)
    zpairs[M] = (z1, z2)
    V_right = Vect[Sym]((0, L ÷ 2, 0) => 1)
    push!(V_left_list, V_right)

    m = FiniteMPS([randn(Float64, V_left_list[i] ⊗ V_phys_list[i] ← V_left_list[i+1]) for i in 1:M])
    @show [dim(space(m[i], 3)) for i in 1:M]
    m_new = changebonds(m, SvdCut(trscheme = truncrank(χ_trunc)))
    @show [dim(space(m_new[i], 3)) for i in 1:M]

    return [m_new[i] for i in 1:M], V_phys_list, V_left_list, zpairs
end

# Occupation configuration of the free-fermion Fermi sea, stored in CHAIN order
function fermi_sea_config(L::Int)
    energies = [(-2 * cos(idx_to_momentum(i, L)), i) for i in 1:L]
    orig_config = zeros(Int, L)
    for (_, i) in sort(energies)[1:L÷2]
        orig_config[i] = 1
    end
    config = zeros(Int, L)
    for mode in 1:L
        config[to_chain(mode, L)] = orig_config[mode]
    end
    return config
end

# Fermi-sea-biased start plus a small random perturbation, SVD-truncated to χ_trunc.
function get_start_fermi_sea(L::Int, χ_trunc::Int, config = fermi_sea_config(L); ε = 0.5)
    M = L ÷ 2
    Sym = FermionParity ⊠ U1Irrep ⊠ ZNIrrep{2 * L}

    V_phys_list = []
    zpairs = Vector{Tuple{Int,Int}}(undef, M)
    V_left_list = [Vect[Sym]((0, 0, 0) => 1)]

    for s in 1:M-1
        Vp, z1, z2 = pair_phys_space(s, L, Sym)
        push!(V_phys_list, Vp)
        zpairs[s] = (z1, z2)
        V_right = fuse(V_left_list[end], Vp)
        push!(V_left_list, V_right)
    end

    Vp, z1, z2 = pair_phys_space(M, L, Sym)
    push!(V_phys_list, Vp)
    zpairs[M] = (z1, z2)
    V_right = Vect[Sym]((0, L ÷ 2, 0) => 1)
    push!(V_left_list, V_right)

    prod_tensors = make_sample_string(config, V_phys_list, V_left_list[1], zpairs, L)
    ψ_prod = FiniteMPS([prod_tensors[i] for i in 1:M])

    ψ_rand = FiniteMPS([randn(Float64, V_left_list[i] ⊗ V_phys_list[i] ← V_left_list[i+1]) for i in 1:M])

    ψ = normalize(ψ_prod) + ε * normalize(ψ_rand)
    @show [dim(left_virtualspace(ψ, i)) for i in 2:M]
    ψ = changebonds(ψ, SvdCut(trscheme = truncrank(χ_trunc)))
    @show [dim(left_virtualspace(ψ, i)) for i in 2:M]

    return [ψ[i] for i in 1:M], V_phys_list, V_left_list, zpairs
end

# Build a "sample string" MPS (trivial bond dims) representing a single
# configuration, in CHAIN order, over the L/2 pair sites
function make_sample_string(config, V_phys_list, V_boundary_left, zpairs, L::Int)
    M = length(V_phys_list)
    twoL = 2 * L

    n1 = config[1]; n2 = config[2]
    z1, z2 = zpairs[1]
    sec = sector_for(V_phys_list[1], n1, n2, z1, z2, twoL)
    Vn1 = typeof(V_phys_list[1])(sec => 1)
    V_left = V_boundary_left
    V_right = fuse(V_left ⊗ Vn1)
    first_tensor = ones(Float64, V_left ⊗ V_phys_list[1] ← V_right)
    mps_string = Vector{typeof(first_tensor)}(undef, M)
    mps_string[1] = first_tensor
    V_left = V_right

    for s in 2:M
        n1 = config[2s-1]; n2 = config[2s]
        z1, z2 = zpairs[s]
        sec = sector_for(V_phys_list[s], n1, n2, z1, z2, twoL)
        Vn_s = typeof(V_phys_list[s])(sec => 1)
        V_right = fuse(V_left ⊗ Vn_s)
        mps_string[s] = ones(Float64, V_left ⊗ V_phys_list[s] ← V_right)
        V_left = V_right
    end

    return mps_string
end

# ----------------------------------------------------------------------------
# Direct sampling
# ----------------------------------------------------------------------------

function _draw_branch_sector(left_env, Bk, V_string_left, V_phys, sec)
    Vn_s = typeof(V_phys)(sec => 1)
    V_right = fuse(V_string_left ⊗ Vn_s)
    sel = ones(Float64, V_string_left ⊗ V_phys ← V_right)
    @tensor le[-1; -2] := twist(left_env, 1)[2; 1] * Bk[1 3; -2] * conj(sel[2 3; -1])
    return le, sel
end

# enumerate the 4 possible occupation-number combinations (n1, n2)
const PAIR_COMBOS = ((0, 0), (1, 0), (0, 1), (1, 1))

# One independent autoregressive draw over the L/2 pair sites, given the
# precomputed right-canonical tensors B plus importance-weight machinery: q ∝ p^α.
function draw_from_B(B, zpairs, L::Int; α = 1.0)
    M = length(B)
    twoL = 2 * L
    config = Vector{Int}(undef, L)
    left_env = ones(Float64, space(B[1], 1) ← space(B[1], 1))
    local sample_string
    iw = 1.0

    for s in 1:M
        V_phys = space(B[s], 2)
        V_string_left = space(left_env, 1)
        z1, z2 = zpairs[s]

        sec1 = sector_for(V_phys, PAIR_COMBOS[1][1], PAIR_COMBOS[1][2], z1, z2, twoL)
        sec2 = sector_for(V_phys, PAIR_COMBOS[2][1], PAIR_COMBOS[2][2], z1, z2, twoL)
        sec3 = sector_for(V_phys, PAIR_COMBOS[3][1], PAIR_COMBOS[3][2], z1, z2, twoL)
        sec4 = sector_for(V_phys, PAIR_COMBOS[4][1], PAIR_COMBOS[4][2], z1, z2, twoL)
        branches = (
            _draw_branch_sector(left_env, B[s], V_string_left, V_phys, sec1),
            _draw_branch_sector(left_env, B[s], V_string_left, V_phys, sec2),
            _draw_branch_sector(left_env, B[s], V_string_left, V_phys, sec3),
            _draw_branch_sector(left_env, B[s], V_string_left, V_phys, sec4),
        )
        w = (norm(branches[1][1])^2, norm(branches[2][1])^2, norm(branches[3][1])^2, norm(branches[4][1])^2)

        Z = sum(w)

        p = w ./ Z
        qt = p .^ α; Zq = sum(qt); q = qt ./ Zq
        r = rand(); cum = 0.0; choice = 4
        for i in 1:4
            cum += q[i]
            if r < cum
                choice = i
                break
            end
        end
        iw *= p[choice] / q[choice]

        n1, n2 = PAIR_COMBOS[choice]
        le, sel = branches[choice]
        s == 1 && (sample_string = Vector{typeof(sel)}(undef, M))
        config[2s-1] = n1; config[2s] = n2
        sample_string[s] = sel
        left_env = le
    end

    return config, sample_string, iw
end

canonical_B(mps_chain) = (ψ = FiniteMPS([mps_chain[i] for i in 1:length(mps_chain)]); [ψ.AR[i] for i in 1:length(mps_chain)])

function sample_direct!(mc::MC)
    config, sample_string, iw = draw_from_B(mc.B, mc.zpairs, mc.L; α = mc.alpha)
    weight = get_overlap(mc.mps_chain, sample_string)[1]
    mc.config = config
    mc.idx_filled = findall(==(1), config)
    mc.idx_empty = findall(==(0), config)
    mc.weight = weight
    mc.importance_weight = iw
    return mc
end

function get_overlap(mps_state, product_state)
    right_env = ones(Float64, space(mps_state[length(mps_state)], 3)' ← space(product_state[length(product_state)], 3)')
    for i in length(mps_state):-1:1
        @tensor right_env_new[-1; -2] := mps_state[i][-1 2; 1] * right_env[1; 3] * conj(twist(product_state[i], 3)[-2 2; 3])
        right_env = right_env_new
    end
    right_env.data == twist(right_env, 1).data || error("Mismatch between right_env and its twist")
    return right_env.data
end

function Carlo.init!(mc::MC, ::MCContext, ::AbstractDict)
    mc.B = canonical_B(mc.mps_chain)
    return nothing
end

function Carlo.register_evaluables(::Type{MC}, eval::AbstractEvaluator, params::AbstractDict)
    return nothing
end

function Carlo.sweep!(mc::MC, ctx::MCContext)
    sample_direct!(mc)
    mc.total_attempts += 1
    mc.accepted_flips += 1
    return nothing
end

function apply_two_body(config, a, b, k2, k1)
    state = copy(config)
    sgn = 1

    state[k1] == 0 && return (0, state)
    sgn *= (-1)^sum(@view state[1:k1-1]); state[k1] = 0

    state[k2] == 0 && return (0, state)
    sgn *= (-1)^sum(@view state[1:k2-1]); state[k2] = 0

    state[b] == 1 && return (0, state)
    sgn *= (-1)^sum(@view state[1:b-1]); state[b] = 1

    state[a] == 1 && return (0, state)
    sgn *= (-1)^sum(@view state[1:a-1]); state[a] = 1

    return (sgn, state)
end

# ----------------------------------------------------------------------------
# E_loc for total energy
# ----------------------------------------------------------------------------

function get_E_loc(mc::MC)::Float64
    L = mc.L
    E = 0.0
    for id_filled in mc.idx_filled
        E += (-2) * mc.t * cos(idx_to_momentum(to_orig(id_filled, L), L))
    end

    E_int = 0.0
    V_phys_list = [space(mc.mps_chain[i], 2) for i in 1:length(mc.mps_chain)]
    V_left = space(mc.mps_chain[1], 1)

    overlap_cache = Dict{Vector{Int}, Float64}()
    amp(state) = get!(overlap_cache, collect(Int, state)) do
        get_overlap(mc.mps_chain, make_sample_string(state, V_phys_list, V_left, mc.zpairs, L))[1]
    end

    filled = mc.idx_filled
    for k1 in filled
        orig1 = to_orig(k1, L)
        for k2 in filled
            k1 == k2 && continue
            orig2 = to_orig(k2, L)

            for q_idx in 0:L-1
                orig1p = mod1(orig1 + q_idx, L)
                orig2p = mod1(orig2 - q_idx, L)
                k1p = to_chain(orig1p, L)
                k2p = to_chain(orig2p, L)

                sgn, state_trial = apply_two_body(mc.config, k1p, k2p, k2, k1)
                sgn == 0 && continue

                V_q = mc.V * cos(2π * q_idx / L)
                ratio = state_trial == mc.config ? 1.0 : amp(state_trial) / mc.weight
                E_int += (1 / L) * V_q * sgn * ratio
            end
        end
    end

    return (E + E_int) / L
end

# ----------------------------------------------------------------------------
# Gradient
# ----------------------------------------------------------------------------

function get_partial_G(mps_chain, sample_string, V_boundary_left, V_boundary_right)
    G_total = []
    for idx in 1:length(mps_chain)
        left_env = ones(Float64, space(sample_string[1], 1) ← space(mps_chain[1], 1))
        for k in 1:idx-1
            @tensor left_env[-1; -2] := twist(left_env, 1)[2; 1] * mps_chain[k][1 3; -2] * adjoint(sample_string[k])[-1; 2 3]
        end

        right_env = ones(Float64, space(sample_string[length(sample_string)], 3)' ← space(mps_chain[length(mps_chain)], 3)')
        for k in length(mps_chain):-1:idx+1
            @tensor right_env[-1; -2] := twist(right_env, 2)[1; 2] * mps_chain[k][-1 3; 1] * adjoint(sample_string[k])[2; -2 3]
        end

        @tensor G_idx[-1; -3 -2] := twist(left_env, 1)[1; -3] * conj(twist(sample_string[idx], 3)[1 -2; 2]) * twist(right_env, 2)[-1; 2]
        space(G_idx') == space(mps_chain[idx]) || error("Space mismatch between G_idx and MPS chain at site $idx")
        push!(G_total, deepcopy(G_idx'))
    end
    return G_total
end

function flatten_gradient(Gradient)
    result = Float64[]
    for i in eachindex(Gradient)
        for (_, block) in blocks(Gradient[i])
            append!(result, vec(convert(Array, block)))
        end
    end
    return result
end

function apply_update!(mps_chain, δp)
    offset = 1
    for i in eachindex(mps_chain)
        for (_, block) in blocks(mps_chain[i])
            n = length(block)
            block .+= reshape(view(δp, offset:offset+n-1), size(block))
            offset += n
        end
    end
end

function block_key_gradient(idx, sector)
    fp = Int(sector[1].isodd)
    u1 = sector[2].charge
    zn = sector[3].n
    return Symbol("G_fp_$(idx)_$(fp)_u$(u1)_zn$(zn)")
end

function block_key_gradient_times_e_loc(idx, sector)
    fp = Int(sector[1].isodd)
    u1 = sector[2].charge
    zn = sector[3].n
    return Symbol("G_times_E_loc_fp_$(idx)_$(fp)_u$(u1)_zn$(zn)")
end

function Carlo.measure!(mc::MC, ctx::MCContext)
    iw = mc.importance_weight
    f_current = get_E_loc(mc)
    measure!(ctx, :W, iw)
    measure!(ctx, :W2, iw^2)   # for effective-sample-size diagnostic: ESS = ⟨iw⟩²/⟨iw²⟩ ∈ (0,1]
    measure!(ctx, :Energy, iw * f_current)
    measure!(ctx, :Energy2, iw * f_current^2)

    V_phys_list = [space(mc.mps_chain[i], 2) for i in 1:length(mc.mps_chain)]
    mps_init = make_sample_string(mc.config, V_phys_list, space(mc.mps_chain[1], 1), mc.zpairs, mc.L)
    Gradient = get_partial_G(mc.mps_chain, mps_init, space(mps_init[1], 1), space(mps_init[end], 3)) * (1 / mc.weight)
    Gradient_times_E_loc = Gradient * f_current

    o_vec = flatten_gradient(Gradient)
    measure!(ctx, :O_flat, iw .* o_vec)
    measure!(ctx, :OE_flat, iw .* o_vec .* f_current)
    measure!(ctx, :OO_flat, iw .* vec(o_vec * o_vec'))

    for i in eachindex(mc.mps_chain)
        space(Gradient[i]) == space(mc.mps_chain[i]) || error("Space mismatch between Gradient and MPS chain at site $i")
        space(Gradient_times_E_loc[i]) == space(mc.mps_chain[i]) || error("Space mismatch between Gradient and MPS chain at site $i")
    end

    for i in eachindex(mc.mps_chain)
        for (sector, block) in blocks(Gradient[i])
            measure!(ctx, block_key_gradient(i, sector), iw .* convert(Array, block))
        end
        for (sector, block) in blocks(Gradient_times_E_loc[i])
            measure!(ctx, block_key_gradient_times_e_loc(i, sector), iw .* convert(Array, block))
        end
    end

    measure!(ctx, :total_attempts, mc.total_attempts)
    measure!(ctx, :accepted_flips, mc.accepted_flips)
    return nothing
end

end
