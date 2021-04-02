# Dataframe Wrapper
struct Record
    df::DataFrame
    path::String
end

function Record(path::AbstractString, new = false)
    df = (new == false && ispath(path)) ? (deserialize(path)::DataFrame) : (DataFrame())
    return Record(df, path)
end

function _transpose(df)
    return DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
end
Base.show(io::IO, record::Record) = show(io, _transpose(record.df))

lower(x) = x
lower(x::AbstractDict) = lowerdict(x)
lowerdict(x) = Dict(key => lower(value) for (key, value) in pairs(x))
function Base.push!(record::Record, row; cols = :union)
    return push!(record.df, lowerdict(row); cols = cols)
end

function save(record::Record)
    mktemp() do path, io
        serialize(io, record.df)
        close(io)
        mv(path, record.path; force = true)
    end
end

makeresult(v) = makeresult([v])
makeresult(v::AbstractVector) = SortedDict(reduce(merge, v))

# For computing 9999999 latencies
# NOTE: the input vector must be sorted
getnine(x::AbstractVector, frac) = x[ceil(Int, frac * length(x))]
