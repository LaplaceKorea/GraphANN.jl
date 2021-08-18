# Compression routine for a dataset.

# Computation of `η` - the coefficient for the parallel loss.
# TODO: Check for `t ≥ norm(x)`
function η(t, x::MaybePtr{SVector})
    n = GraphANN._Base.norm(x)
    v = t / n^2
    return v / (1 - v)
end
