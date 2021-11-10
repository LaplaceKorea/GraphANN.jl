if get(ENV, "JULIA_EXCLUSIVE", 0) != 0
    @info "Running JULIA_EXCLUSIVE tests!"
    @test GraphANN._Base.NUM_NUMA_NODES[] > 1
end
