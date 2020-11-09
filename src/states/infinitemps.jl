"""
    struct InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor}

Represents an infinite matrix product state
The state is stored in the centergauge where
    state.AL[i]*state.CR[i] = state.AC[i] = state.CR[i-1]*state.AR[i]
"""
struct InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor}
    AL::PeriodicArray{A,1}
    AR::PeriodicArray{A,1}
    CR::PeriodicArray{B,1}
    AC::PeriodicArray{A,1}
end

Base.size(arr::InfiniteMPS,i) = size(arr.AL,i)
Base.length(arr::InfiniteMPS) = size(arr,1)
Base.eltype(arr::InfiniteMPS) = eltype(arr.AL[1])
Base.copy(m::InfiniteMPS) = InfiniteMPS(copy(m.AL),copy(m.AR),copy(m.CR),copy(m.AC));
Base.repeat(m::InfiniteMPS,i::Int) = InfiniteMPS(repeat(m.AL,i),repeat(m.AR,i),repeat(m.CR,i),repeat(m.AC,i));
Base.similar(st::InfiniteMPS) = InfiniteMPS(similar(st.AL),similar(st.AR),similar(st.CR),similar(st.AC))
TensorKit.norm(st::InfiniteMPS) = norm(st.AC[1]);
virtualspace(psi::InfiniteMPS, n::Integer) = _firstspace(psi.AL[n+1])

function InfiniteMPS(pspaces::AbstractArray{S,1},Dspaces::AbstractArray{S,1};kwargs...) where S
    InfiniteMPS([TensorMap(rand,Defaults.eltype,Dspaces[mod1(i-1,length(Dspaces))]*pspaces[i],Dspaces[i]) for i in 1:length(pspaces)];kwargs...)
end

#allow users to pass in simple arrays
function InfiniteMPS(A::AbstractArray{T,1}; kwargs...) where T<:GenericMPSTensor

    #we make a copy, and are therfore garantueeing no side effects for the user
    AR = PeriodicArray(A[:]);

    #initial guess for CR
    CR = PeriodicArray([isomorphism(Matrix{eltype(AR[i])},space(AR[i+1],1),space(AR[i+1],1)) for i in 1:length(A)]);

    AL = similar(AR);

    uniform_leftorth!(AL,CR,AR;kwargs...);
    uniform_rightorth!(AR,CR,AL;kwargs...);

    AC = similar(AR)
    for loc = 1:length(A)
        AC[loc] = AL[loc]*CR[loc]
    end

    CRtype = tensormaptype(spacetype(T),1,1,eltype(T));
    return InfiniteMPS{T,CRtype}(AL,AR,CR,AC)
end

function reorth!(dst::InfiniteMPS;from=:AL,kwargs...)
    @assert from == :AL

    #dst.AL changed, dst.CR may no longer fit
    if !reduce(&,map(x->_lastspace(x[1]) == _lastspace(x[2]),zip(dst.CR,dst.AL)))
        for i in 1:length(dst)
            dst.CR[i] = isomorphism(Matrix{eltype(dst.AL[i])},_lastspace(dst.AL[i])',_lastspace(dst.AL[i])')
        end
    end

    uniform_rightorth!(dst.AR,dst.CR,dst.AL;kwargs...);

    for loc in 1:length(dst)
        dst.AC[loc] = dst.AL[loc]*dst.CR[loc]
    end

    dst
end

function TensorKit.normalize!(st::InfiniteMPS)
    normalize!.(st.CR)
    normalize!.(st.AC)
    st
end

"
    l_RR(state,location)
    Left dominant eigenvector of the AR-AR transfermatrix
"
l_RR(state::InfiniteMPS,loc::Int=1) = @tensor toret[-1;-2]:=state.CR[loc-1][1,-2]*conj(state.CR[loc-1][1,-1])

"
    l_RL(state,location)
    Left dominant eigenvector of the AR-AL transfermatrix
"
l_RL(state::InfiniteMPS,loc::Int=1) = state.CR[loc-1]

"
    l_LR(state,location)
    Left dominant eigenvector of the AL-AR transfermatrix
"
l_LR(state::InfiniteMPS,loc::Int=1) = state.CR[loc-1]'

"
    l_LL(state,location)
    Left dominant eigenvector of the AL-AL transfermatrix
"
l_LL(state::InfiniteMPS{A},loc::Int=1) where A= isomorphism(Matrix{eltype(A)}, space(state.AL[loc],1),space(state.AL[loc],1))

"
    r_RR(state,location)
    Right dominant eigenvector of the AR-AR transfermatrix
"
r_RR(state::InfiniteMPS{A},loc::Int=length(state)) where A= isomorphism(Matrix{eltype(A)},domain(state.AR[loc]),domain(state.AR[loc]))

"
    r_RL(state,location)
    Right dominant eigenvector of the AR-AL transfermatrix
"
r_RL(state::InfiniteMPS,loc::Int=length(state)) = state.CR[loc]'

"
    r_LR(state,location)
    Right dominant eigenvector of the AL-AR transfermatrix
"
r_LR(state::InfiniteMPS,loc::Int=length(state)) = state.CR[loc]

"
    r_LL(state,location)
    Right dominant eigenvector of the AL-AL transfermatrix
"
r_LL(state::InfiniteMPS,loc::Int=length(state))= @tensor toret[-1;-2]:=state.CR[loc][-1,1]*conj(state.CR[loc][-2,1])

function TensorKit.dot(a::InfiniteMPS,b::InfiniteMPS;krylovdim = 30)
    init = TensorMap(rand,ComplexF64,space(a.AL[1],1),space(b.AL[1],1))
    num = lcm(length(a),length(b))
    (vals,vecs,convhist) = eigsolve(x->transfer_left(x,b.AL[1:num],a.AL[1:num]),init,1,:LM,Arnoldi(krylovdim=krylovdim))
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end

Base.circshift(st::InfiniteMPS,n) = InfiniteMPS(circshift(st.AL,n),circshift(st.AR,n),circshift(st.CR,n),circshift(st.AC,n))
