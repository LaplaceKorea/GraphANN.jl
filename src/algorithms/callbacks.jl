#####
##### Callback Implementations
#####

struct Latencies{T <: MaybeThreadLocal{Vector{UInt64}}}
    values::T
end

Base.empty!(x::Latencies{Vector{UInt64}}) = empty!(x.values)
Base.empty!(x::Latencies{<:ThreadLocal}) = foreach(empty!, getall(x.values))

Base.getindex(x::Latencies{Vector{UInt64}}) = x.values
Base.getindex(x::Latencies{<:ThreadLocal}) = x.values[]

Base.get(x::Latencies{Vector{UInt64}}) = x.values
Base.get(x::Latencies{<:ThreadLocal}) = reduce(vcat, getall(x.values))

Latencies(::Any) = Latencies(UInt64[])
Latencies(::ThreadLocal) = Latencies(ThreadLocal(UInt64[]))

# single threaded version
function latency_callbacks(runner::MaybeThreadLocal{T}) where {T}
    latencies = Latencies(runner)
    prequery = () -> push!(latencies[], time_ns())
    postquery = () -> begin
        _latencies = latencies[]
        _latencies[end] = time_ns() - _latencies[end]
    end
    # Create the correct callback wrapper based on the type of the "Runner"
    return (latencies = latencies, callbacks = _latency_callbacks(T, prequery, postquery))
end

function _latency_callbacks(::Type{<:DiskANNRunner}, prequery, postquery)
    return DiskANNCallbacks(; prequery, postquery)
end

function _latency_callbacks(::Type{<:SPTAGRunner}, prequery, postquery)
    return SPTAGCallbacks(; prequery, postquery)
end
