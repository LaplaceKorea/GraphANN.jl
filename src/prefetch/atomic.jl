# Normal Julia uses a mutable Atomic type to perform atomic operations.
# For the prefetch tracker, we want potentially billions of atomically modifiable objects,
# so an allocation for each will just not do.
#
# Here, we copy over the code from `base/atomics.jl` and convert it to work on raw pointers.

const inttypes = (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
const IntType = Union{inttypes...}

import Base.Sys: WORD_SIZE

# Mapping between Julia types and LLVM names
const llvmtypes = IdDict{Any,String}(
    Int8 => "i8",
    Int16 => "i16",
    Int32 => "i32",
    Int64 => "i64",
    UInt8 => "i8",
    UInt16 => "i16",
    UInt32 => "i32",
    UInt64 => "i64",
)

for typ in inttypes
    lt = llvmtypes[typ]

    # Create a bunch of atomic_cas! functions
    @eval function unsafe_atomic_cas!(ptr::Ptr{$typ}, cmp::$typ, new::$typ)
        Base.llvmcall($"""
            %ptr = inttoptr i$WORD_SIZE %0 to $lt*
            %rs = cmpxchg $lt* %ptr, $lt %1, $lt %2 acq_rel acquire
            %rv = extractvalue { $lt, i1 } %rs, 0
            ret $lt %rv
            """,
            $typ,
            Tuple{Ptr{$typ}, $typ, $typ},
            ptr, cmp, new,
        )
    end

    @eval function unsafe_atomic_or!(ptr::Ptr{$typ}, v::$typ)
        Base.llvmcall($"""
            %ptr = inttoptr i$WORD_SIZE %0 to $lt*
            %rv = atomicrmw or $lt* %ptr, $lt %1 acq_rel
            ret $lt %rv
            """,
            $typ,
            Tuple{Ptr{$typ}, $typ},
            ptr, v
        )
    end

    @eval function unsafe_atomic_nand!(ptr::Ptr{$typ}, v::$typ)
        Base.llvmcall($"""
            %ptr = inttoptr i$WORD_SIZE %0 to $lt*
            %rv = atomicrmw nand $lt* %ptr, $lt %1 acq_rel
            ret $lt %rv
            """,
            $typ,
            Tuple{Ptr{$typ}, $typ},
            ptr, v
        )
    end
end
