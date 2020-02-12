"
    FiniteMps(data::Array)

    finite one dimensional mps
    algorithms usually assume a right-orthormalized input
"
struct FiniteMps{T<:GenMpsType} <: AbstractArray{T,1}
    data::Array{T,1}

    function FiniteMps(data::Array{T,1}) where T<:GenMpsType
        #=
        ou = oneunit(space(data[1],1));
        @assert space(data[1],1) == ou
        @assert space(data[end],length(codomain(data[end]))+length(domain(data[end]))) == ou'
        =#
        new{T}(data)
    end
end

FiniteMps{T}(init,len) where T = FiniteMps(Array{T,1}(init,len))

Base.length(arr::FiniteMps) = length(arr.data)
Base.size(arr::FiniteMps) = size(arr.data)
Base.getindex(arr::FiniteMps,I::Int) = arr.data[I]
Base.setindex!(arr::FiniteMps{T},v::T,I::Int) where T = setindex!(arr.data,v,I)
Base.eltype(arr::FiniteMps{T}) where T = T
Base.iterate(arr::FiniteMps,state=1) = Base.iterate(arr.data,state)
Base.lastindex(arr::FiniteMps) = lastindex(arr.data)
Base.lastindex(arr::FiniteMps,d) = lastindex(arr.data,d)
Base.copy(arr::FiniteMps) where T= FiniteMps(copy(arr.data));
Base.deepcopy(arr::FiniteMps) where T = FiniteMps(deepcopy(arr.data));
Base.similar(arr::FiniteMps) = FiniteMps(similar(arr.data))
r_RR(state::FiniteMps{T}) where T = isomorphism(Matrix{eltype(T)},domain(state[end]),domain(state[end]))
l_LL(state::FiniteMps{T}) where T = isomorphism(Matrix{eltype(T)},space(state[1],1),space(state[1],1))

"take the sum of 2 finite mpses"
function Base.:+(v1::FiniteMps,v2::FiniteMps) #untested and quite horrible code, but not sure how to make it nice
    @assert length(v1)==length(v2)

    function fuseutils(v)
        v = permute(v,(1,2),(3,4,5))
        v = TensorMap(v.data,fuse(codomain(v)),domain(v))
        v = permute(v,(1,2),(3,4))
        v = TensorMap(v.data,codomain(v),fuse(domain(v)))
    end

    ou=oneunit(space(v1[1],1))

    v1m = TensorMap(ou⊕ou,ou⊕ou) do x
        [1 0;0 0]
    end

    v2m = TensorMap(ou⊕ou,ou⊕ou) do x
        [0 0;0 1]
    end

    tot = similar(v1)
    for i in 1:length(v1)
        u1 = isomorphism(space(v1[i],1),space(v1[i],3)')
        u2 = isomorphism(space(v2[i],1),space(v2[i],3)')

        @tensor v1e[-1 -2; -3 -4 -5]:=v1[i][-1,-3,-5]*u2[-2,-4]
        @tensor v2e[-1 -2; -3 -4 -5]:=v2[i][-2,-3,-4]*u1[-1,-5]

        v1e = fuseutils(v1e)
        v2e = fuseutils(v2e)

        @tensor v1e[-1 -2; -3 -4 -5]:=v1e[-2,-3,-4]*v1m[-1,-5]
        @tensor v2e[-1 -2; -3 -4 -5]:=v2e[-2,-3,-4]*v2m[-1,-5]

        tot[i] = fuseutils(v1e)+fuseutils(v2e)
    end

    edge1 = TensorMap(ones,oneunit(space(tot[1],1)),space(tot[1],1))
    edge2 = TensorMap(ones,space(tot[end],3)',oneunit(space(tot[end],3)))
    @tensor tot[1][-1 -2;-3] := edge1[-1,1]*tot[1][1,-2,-3]
    @tensor tot[end][-1 -2;-3] := tot[end][-1,-2,1]*edge2[1,-3]
    return tot
end


function LinearAlgebra.dot(v1::FiniteMps,v2::FiniteMps)
    @assert length(v1)==length(v2)

    @tensor start[-1;-2]:=v2[1][1,2,-2]*conj(v1[1][1,2,-1])
    for i in 2:length(v1)-1
        start=transfer_left(start,v2[i],v1[i])
    end

    @tensor start[1,2]*v2[end][2,3,4]*conj(v1[end][1,3,4])
end