"""
GradientGrassmann is an optimisation methdod that keeps the MPS in left-canonical form, and
treats the tensors as points on Grassmann manifolds. It then applies one of the standard
gradient optimisation methods, e.g. conjugate gradient, to the MPS, making use of the
Riemannian manifold structure. A preconditioner is used, either so that effectively the
metric used on the manifold is that given by the Hilbert space inner product, or by applying
the approximate inverse of the local Hessian, see below.

The arguments to the constructor are
method = OptimKit.ConjugateGradient
    The gradient optimisation method to be used. Should either be an instance or a subtype
    of `OptimKit.OptimizationAlgorithm`. If it's an instance, this `method` is simply used
    to do the optimisation. If it's a subtype, then an instance is constructed as
    `method(; maxiter=maxiter, verbosity=verbosity, gradtol=tol)`

finalize! = OptimKit._finalize!
    A function that gets called once each iteration. See OptimKit for details.

precondition = :statespace
    Which preconditioner to use. Options are `:statespace`, `:localhess`, and `:neighhess`.
    TODO explain the options.

tol = Defaults.tol
maxiter = Defaults.maxiter
verbosity = 2
    Arguments passed to the `method` constructor. If `method` is an instance of
    `OptimKit.OptimizationAlgorithm`, these argument are ignored.

In other words, by default conjugate gradient is used. One can easily set `tol`, `maxiter`
and `verbosity` for it, or switch to LBFGS or gradient descent by setting `method`. If more
control is wanted over things like specifics of the linesearch, CG flavor or the `m`
parameter of LBFGS, then the user should create the `OptimKit.OptimizationAlgorithm`
instance manually and pass it as `method`.
"""
struct GradientGrassmann <: Algorithm
    method::OptimKit.OptimizationAlgorithm
    finalize!::Function
    precondition::Function

    function GradientGrassmann(; method = ConjugateGradient,
                               finalize! = OptimKit._finalize!,
                               precondition = :statespace,
                               tol = Defaults.tol,
                               maxiter = Defaults.maxiter,
                               verbosity = 2)
        if isa(method, OptimKit.OptimizationAlgorithm)
            # We were given an optimisation method, just use it.
            m = method
        elseif method <: OptimKit.OptimizationAlgorithm
            # We were given an optimisation method type, construct an instance of it.
            m = method(; maxiter=maxiter, verbosity=verbosity, gradtol=tol)
        else
            msg = "method should be either an instance or a subtype of OptimKit.OptimizationAlgorithm."
            throw(ArgumentError(msg))
        end
        if precondition === :statespace
            precfunc = GrassmannMPS.precondition
        elseif precondition === :localhess
            precfunc = GrassmannMPS.precondition_localhess
        elseif precondition === :neighhess
            precfunc = GrassmannMPS.precondition_neighhess
        else
            msg = "Unknown preconditioning methods: $precondition"
            throw(ArgumentError(msg))
        end
        return new(m, finalize!, precfunc)
    end
end

function find_groundstate(state::S, H::HT, alg::GradientGrassmann,
                          envs::P=environments(state, H))::Tuple{S,P,Float64} where {S,HT<:Hamiltonian,P}
    normalize!(state)
    res = optimize(GrassmannMPS.fg, (state, envs), alg.method;
                   transport! = GrassmannMPS.transport!,
                   retract = GrassmannMPS.retract,
                   inner = GrassmannMPS.inner,
                   scale! = GrassmannMPS.scale!,
                   add! = GrassmannMPS.add!,
                   finalize! = alg.finalize!,
                   precondition = alg.precondition,
                   isometrictransport = true)
    (x, fx, gx, numfg, normgradhistory) = res
    (state, envs) = x
    return state, envs, normgradhistory[end]
end


# We separate some of the internals needed for implementing GradientGrassmann into a
# submodule, to keep the MPSKit module namespace cleaner.
"""
A module for functions related to treating an InfiniteMPS in left-canonical form as a bunch
of points on Grassmann manifolds, and performing things like retractions and transports
on these Grassmann manifolds.

The module exports nothing, and all references to it should be qualified, e.g.
`GrassmannMPS.fg`.
"""
module GrassmannMPS

using ..MPSKit
using TensorKit
import TensorKitManifolds.Grassmann

"""
Compute the expectation value, and its gradient with respect to the tensors in the unit
cell as tangent vectors on Grassmann manifolds.
"""
function fg(x)
    (state, envs) = x
    # The partial derivative with respect to AL, al_d, is the partial derivative with
    # respect to AC times CR'.
    ac_d = [ac_prime(state.AC[v], v, state, envs) for v in 1:length(state)]
    al_d = [d*c' for (d, c) in zip(ac_d, state.CR[1:end])]
    g = [Grassmann.project(2*d, a) for (d, a) in zip(al_d, state.AL)]
    f = real(sum(expectation_value(state, envs.opp, envs)))
    return f, g
end

"""
Retract a left-canonical infinite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::Tuple{<:InfiniteMPS,<:Cache}, g, alpha)
    (state, envs) = x

    nenvs = deepcopy(envs);
    nstate = nenvs.dependency; # The end-point
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        (nstate.AL[i], h[i]) = Grassmann.retract(state.AL[i], g[i], alpha)
    end

    reorth!(nstate)
    recalculate!(nenvs,nstate)

    return (nstate,nenvs), h
end

"""
Retract a left-canonical finite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::Tuple{<:FiniteMPS,<:Cache}, g, alpha)
    (state, envs) = x
    y = copy(state)  # The end-point
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        (yal, h[i]) = Grassmann.retract(state.AL[i], g[i], alpha)
        y.AC[i] = (yal,state.CR[i])
    end
    normalize!(y)
    return (y,envs), h
end

"""
Transport a tangent vector `h` along the retraction from `x` in direction `g` by distance
`alpha`. `xp` is the end-point of the retraction.
"""
function transport!(h, x, g, alpha, xp)
    (state, envs) = x
    for i in 1:length(state)
        h[i] = Grassmann.transport!(h[i], state.AL[i], g[i], alpha, xp[1].AL[i])
    end
    return h
end

"""
Euclidean inner product between two Grassmann tangents of an infinite MPS.
"""
function inner(x, g1, g2)
    (state, envs) = x
    tot = sum(Grassmann.inner(a, d1, d2) for (a, d1, d2) in zip(state.AL, g1, g2))
    return real(tot)
end

"""
Scale a tangent vector by scalar `alpha`.
"""
scale!(g, alpha) = g .* alpha

"""
Add two tangents vectors, scaling the latter by `alpha`.
"""
add!(g1, g2, alpha) = g1 + g2 .* alpha

"""
Precondition a given Grassmann tangent `g` at state `x` by the Hilbert space inner product.

This requires inverting the right MPS transfer matrix. This is done using `reginv`, with a
regularisation parameter that is the norm of the tangent `g`.
"""
function precondition(x, g)
    (state, envs) = x
    #hacky workaround - what is eltype(state)?
    delta = min(real(one(eltype(state.AL[1]))), sqrt(inner(x, g, g)))
    crinvs = [MPSKit.reginv(cr, delta) for cr in state.CR[1:end]]
    g_prec = [Grassmann.project(d[]*(crinv'*crinv), a)
              for (d, crinv, a) in zip(g, crinvs, state.AL)]
    return g_prec
end

"""
Precondition a given Grassmann tangent `g` at state `x` by the local Hessian.
"""
function precondition_localhess(x, g)
    (state, envs) = x
    gnorm = sqrt(inner(x, g, g))
    innr(x, y) = real(sum(dot(xi, yi) for (xi, yi) in zip(x, y)))

    # TODO Remove the hard-coding of these parameters
    verbosity = 3
    #delta_cr = min(real(one(eltype(state.AL[1]))), gnorm)
    delta_cr = min(1e-2, gnorm)
    delta_newton = 0.0
    gamma = 0.0

    # We'll precondition the truncated Newton with an inverse of the CR matrix at each site.
    # This inverse is regularised with delta_cr.
    crinvs = [MPSKit.reginv(cr, delta_cr) for cr in state.CR]
    crinvcrs = [crinv * cr for (cr, crinv) in zip(state.CR, crinvs)]

    # TODO The projection here is actually unnecessary.
    g_prec = [
        Grassmann.project!(d[] * crinv', a)
        for (d, crinv, a) in zip(g, crinvs, state.AL)
    ]

    Bs = [localhess(state, envs, v) for v in 1:length(state.AL)]
    # The M and MC matrices are needed for the term in the Hessian arising from the
    # curvature of the retraction. They, too, are preconditioned by multiplying with crinv.
    Ms = [c_prime(state.CR[v], v, state, envs) for v in 1:length(state.AL)]
    MCs = [
        (crinv * m * crinvcr' + crinvcr * m' * crinv') / 2
        for (m, crinvcr, crinv) in zip(Ms, crinvcrs, crinvs)
    ]
    B(x) = [
        2 * Grassmann.project!(Bi(xi[] * crinvcr) * crinvcr' - xi[] * mc, a)
        for (Bi, xi, crinvcr, mc, a) in zip(Bs, x, crinvcrs, MCs, state.AL)
    ]
    g_prec = truncated_newton(g_prec, B, innr, delta_newton, gamma; verbosity=verbosity)

    # TODO The projection here is actually unnecessary.
    g_prec = [
        Grassmann.project!(d[] * crinv, a)
        for (d, crinv, a) in zip(g_prec, crinvs, state.AL)
    ]
    return g_prec
end

"""
    localhess(state, envs, v)

Return a function that is the linear operator for the "local" Hessian term, i.e.
```
           ┌──A──┐
 ──A──     │  │  │
   │   ->  l──H──r
           │  │  │
           └─   ─┘
```
`A` is a tangent vector for the MPS tensor in centre-canonical form. `v` is the the site
that `A` is at.
"""
function localhess(state, envs, v)
    f(ac_tan) = ac_prime(ac_tan, v, state, envs)
    return f
end

# TODO Finish the following neighhess stuff.
#function precondition_neighhess(x, gL)
#    verbosity = 3
#    (state, envs) = x
#    gnorm = sqrt(inner(x, gL, gL))
#    delta_cr = min(real(one(eltype(state.AL[1]))), gnorm)
#    gamma = 0.0
#    delta_newton = 0.0
#    #gamma = 1e-8*gnorm
#    #delta_newton = 1e-2
#    innr(x, y) = real(sum(dot(xi, yi) for (xi, yi) in zip(x, y)))
#
#    crinvs = [MPSKit.reginv(cr, delta_cr) for cr in state.CR[1:end]]
#    gC = [d[]*crinv' for (d, crinv) in zip(gL, crinvs)]
#    gC = [Grassmann.project(d, a) for (d, a) in zip(gC, state.AL)]
#
#    B = neighhess(state, envs, crinvs)
#    gC_prec = truncated_newton(gC, B, innr, delta_newton, gamma; verbosity=verbosity)
#
#    gL_prec = [d[]*crinv for (d, crinv) in zip(gC_prec, crinvs)]
#    gL_prec = [Grassmann.project(d, a) for (d, a) in zip(gL_prec, state.AL)]
#    return gL_prec
#end
#
#function neighhess(state, envs, crinvs)
#    ham = envs.opp
#    Ms = [c_prime(state.CR[v], v, state, envs) for v in 1:length(state.AL)]
#    MCs = [(crinv*m + m' * crinv')/2 for (m, crinv) in zip(Ms, crinvs)]
#    function f(gC)
#        gC = [g[] for g in gC]
#        N = length(gC)
#        gL = [gC[i] * crinvs[i] for i in 1:N]
#        gR = [
#            @tensor r[-1 -2; -3] := crinvs[mod1(i-1, N)][-1; 1] * gC[i][1 -2; -3]
#            for i in 1:N
#        ]
#        gC_prec = [ac_prime(gC[pos], pos, state, envs) for pos in 1:N]
#        for pos in 1:N
#            l = leftenv(envs, pos, state)
#            lprev = leftenv(envs, pos-1, state)
#            r = rightenv(envs, pos, state)
#            rnext = rightenv(envs, pos+1, state)
#
#            posnext = mod1(pos+1, N)
#            posprev = mod1(pos-1, N)
#            renv = (
#                transfer_right(rnext, ham, posnext, gR[posnext], state.AR[posnext])
#                #+ transfer_right(rnext, ham, posnext, state.AR[posnext], gR[posnext])
#            )
#            lenv = (
#                transfer_left(lprev, ham, posprev, gL[posprev], state.AL[posprev])
#                #+ transfer_left(lprev, ham, posprev, state.AL[posprev], gL[posprev])
#            )
#            # TODO Could use scalkeys and opkeys to speed up.
#            for (i, j) in MPSKit.keys(ham, pos)
#                @tensor(
#                    gC_prec[pos][-1 -2; -3] +=
#                    lenv[i][-1 5 4] *
#                    state.AC[pos][4 2 1] *
#                    ham[pos, i, j][5 -2 3 2] *
#                    r[j][1 3 -3]
#                )
#                @tensor(
#                    gC_prec[pos][-1 -2; -3] +=
#                    l[i][-1 5 4] *
#                    state.AC[pos][4 2 1] *
#                    ham[pos, i, j][5 -2 3 2] *
#                    renv[j][1 3 -3]
#                )
#            end
#        end
#        gC_prec = [
#            2 * Grassmann.project!(gi_prec - gi * mc, a)
#            for (gi_prec, mc, gi, a) in zip(gC_prec, MCs, gC, state.AL)
#        ]
#        return gC_prec
#    end
#    return f
#end

"""
    truncated_newton(p0, B, inner, delta=0, gamma=0, maxiter=100; verbosity=0)

Solve the equation (B + delta I) x = p0 for x, using a linear conjugate gradient algorithm
that aborts and returns the current iterate if at any point inner(x, B x) <= gamma.

The termination threshold for the residual is high, `ϵ = min(0.5, norm(p0)) * norm(p0)`.
`inner` is the inner product function, `maxiter` and `verbosity` are obvious.
"""
function truncated_newton(p0, B, inner, delta=0, gamma=0, maxiter=1000; verbosity=0)
    counter = 0
    d = p0
    r = -p0
    z = zero.(p0)
    rr = inner(r, r)
    rnorm = sqrt(abs(rr))
    ϵ = min(0.5, sqrt(rnorm)) * rnorm
    while counter < maxiter
        Bd = B(d) + delta*d
        dBd = inner(d, Bd)
        if dBd <= gamma
            if verbosity > 2
                @info "Truncated Newton got a negative Hessian element after $counter steps"
            end
            return counter == 0 ? p0 : z
        end
        α = rr / dBd
        z = z + α * d
        r = r + α * Bd
        counter += 1
        rrold = rr
        rr = inner(r, r)
        sqrt(abs(rr)) < ϵ && break
        β = rr / rrold
        d = -r + β*d
    end
    verbosity > 2 && @info "Truncated Newton done in $counter steps"
    return z
end

end  # module GrassmannMPS
