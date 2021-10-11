function atomic_ptr_cas!(x::Ptr{Float32}, cmp::Float32, new::Float32)
    return Base.llvmcall( """
         %iptr = inttoptr i64 %0 to i32*
         %icmp = bitcast float %1 to i32
         %inew = bitcast float %2 to i32
         %irs = cmpxchg i32* %iptr, i32 %icmp, i32 %inew acq_rel acquire
         %irv = extractvalue { i32, i1 } %irs, 0
         %rv = bitcast i32 %irv to float
         ret float %rv
         """,
         Float32, Tuple{Ptr{Float32},Float32,Float32},
         x,
         cmp,
         new,
    )
end

function atomic_ptr_add!(var::Ptr{Float32}, val::Float32)
    old = unsafe_load(var)
    while true
        new = old + val
        cmp = old
        old = atomic_ptr_cas!(var, cmp, new)
        reinterpret(Int32, old) == reinterpret(Int32, cmp) && return old
    end
end
