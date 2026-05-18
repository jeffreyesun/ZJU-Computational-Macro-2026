"""
    Param{T}

Wrapper for a stage-level parameter that may be either *calibrated* (carry
a literal value of leaf type `T`) or *swept* (carry a `Symbol` naming an
`env` field from which the value is read at evaluation time).

The struct is mutable so that mode-flip (calibrated ↔ swept) is a field
mutation, not a stage rebuild:

```julia
ε = Param(0.25)       # calibrated
ε.val = :ξ_logit       # now swept; reads env.ξ_logit at use
```

`Union{T, Symbol}` is a small union; Julia's union-splitting makes
`resolve(p, env)` type-stable.
"""
mutable struct Param{T}
    val :: Union{T, Symbol}
end

Param(x::Real)        = Param{typeof(x)}(x)
Param(sym::Symbol)    = Param{Float64}(sym)

"""
Resolve a `Param` against a runtime `env`: if `p.val` is a `Symbol`, read
`env[p.val]`; otherwise return `p.val`.
"""
function resolve(p::Param{T}, env)::T where {T}
    v = p.val
    return v isa Symbol ? env[v]::T : v
end

"True iff `p` is currently in swept mode (its `.val` is a `Symbol`)."
is_swept(p::Param) = p.val isa Symbol

"The `env` key `p` reads from if swept, else `nothing`."
swept_key(p::Param) = p.val isa Symbol ? p.val : nothing
