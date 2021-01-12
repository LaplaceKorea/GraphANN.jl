repeated(f::F, n) where {F} = [f() for _ in 1:n]

# Can we find the first `x` where `f(x) > target` but `f(x-1) < target`.
function binarysearch(f::F, target, lo::T, hi::T) where {F, T <: Integer}
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = Base.Sort.midpoint(lo, hi)
        @show m f(m) target
        if isless(f(m), target)
            lo = m
        else
            hi = m
        end
    end
    return hi
end

function nines(x; max = 5, rev = false)
    x = sort(x; rev = rev, alg = QuickSort)
    vals = eltype(x)[]
    for pow in 1:max
        frac = 1.0 - (0.1)^pow
        ind = ceil(Int, frac * length(x))

        ind >= length(x) && break
        push!(vals, x[ind])
    end
    @show Int.(vals)
    return vals
end

struct Memoize{F}
    f::F
    saved::Dict{Any,Any}
end

memoize(f::F) where {F} = Memoize{F}(f, Dict{Any,Any}())
(f::Memoize)(x...) = get!(() -> f.f(x...), f.saved, x)

