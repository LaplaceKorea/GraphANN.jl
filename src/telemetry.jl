module _Telemetry

# Here, we provide a `Telemetry` type with a function `ifhasa`.
# The idea is that we can search `Telemetry` for an object of a certain type.
#
# If that object is found, then the function passed to `ifhasa` is called with that
# object as an argument.
#
# If the object is not found, then the passed function is not called and we rely on
# Dead Code Elimination to remove the passed function.
#
# The implementation uses recursive descent through a tuple using dispatch to determine
# if an object is found.
# If no instances are found, than `nothing` is returned, which acts as a sentinal value
# to control. the calling of `maybecall`.

export Telemetry, ifhasa

struct Telemetry{names,T}
    val::NamedTuple{names,T}

    # -- inner constructor to allow a keyword counstructor.
    Telemetry(x::NamedTuple{names, T}) where {names, T} = new{names, T}(x)
end

# Overload `getproperty` so we can access entries in the wrapped `NamedTuple` using
# the dot syntax.
function Base.getproperty(x::Telemetry, v::Symbol)
    val = getfield(x, :val)
    return v == :val ? val : getproperty(val, v)
end

# Turn keywords into a NamedTuple.
Telemetry(; kw...) = Telemetry((;kw...,))
ifhasa(f::F, x::Telemetry, ::Type{T}) where {F,T} = maybecall(f, find(T, x))

# Entry point.
find(::Type{T}, x::Telemetry) where {T} = find(T, Tuple(x.val)...)

# 1. Matching path.
# 2. Non-matching path.
# 3. Bottom case (end of recursion).
find(::Type{T}, x::U, args...) where {T, U <: T} = x
find(::Type{T}, x, args...) where {T} = find(T, args...)
find(::Type) = nothing

# If object is found, then call the passed function.
# Otherwise, do nothing.
maybecall(f::F, ::Nothing) where {F} = nothing
maybecall(f::F, x) where {F} = f(x)

end # module

