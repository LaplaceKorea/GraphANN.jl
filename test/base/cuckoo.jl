@testset "Testing Cuckoo Filter" begin
    @testset "Testing Bucket" begin
        # Imports for convenience.
        _Base = GraphANN._Base
        Bucket = _Base.Bucket

        # Tests for the number of padding bits required for each UInt64 word.
        @test _Base.padding_bits(Bucket{4,1}) == 0
        @test _Base.padding_bits(Bucket{5,1}) == 4
        @test _Base.padding_bits(Bucket{6,1}) == 4
        @test _Base.padding_bits(Bucket{7,1}) == 1
        @test _Base.padding_bits(Bucket{8,1}) == 0
        @test _Base.padding_bits(Bucket{12,1}) == 4

        @test _Base.fingerprints_per_word(Bucket{4,1}) == 16
        @test _Base.fingerprints_per_word(Bucket{5,1}) == 12
        @test _Base.fingerprints_per_word(Bucket{6,1}) == 10
        @test _Base.fingerprints_per_word(Bucket{7,1}) == 9
        @test _Base.fingerprints_per_word(Bucket{8,1}) == 8
        @test _Base.fingerprints_per_word(Bucket{12,1}) == 5

        # Search Mask
        @test _Base.searchmask(Bucket{4,1}) == 0x1111_1111_1111_1111
        @test _Base.searchmask(Bucket{5,1}) == 0x0084_2108_4210_8421
        @test _Base.searchmask(Bucket{6,1}) == 0x0041_0410_4104_1041
        @test _Base.searchmask(Bucket{7,1}) == 0x0102_0408_1020_4081
        @test _Base.searchmask(Bucket{8,1}) == 0x0101_0101_0101_0101
        @test _Base.searchmask(Bucket{12,1}) == 0x0001_0010_0100_1001

        # Fingerprint Mask
        @test _Base.fingerprintmask(Bucket{4,1}) == 0b1111
        @test _Base.fingerprintmask(Bucket{5,1}) == 0b1_1111
        @test _Base.fingerprintmask(Bucket{6,1}) == 0b11_1111
        @test _Base.fingerprintmask(Bucket{7,1}) == 0b111_1111
        @test _Base.fingerprintmask(Bucket{8,1}) == 0b1111_1111
        @test _Base.fingerprintmask(Bucket{12,1}) == 0b1111_1111_1111

        # Make sure we can match all positions within a 64-bit word.
        ntrials = 10
        for i in [4,5,6,7,8,12]
            # Test each position where a fingerprint can be considered valid.
            T = Bucket{i,1}
            maxfingerprint = (2^i) - 1

            # Select a nonzero fingerprint
            fingerprint = zero(UInt64)
            while true
                fingerprint = rand(UInt64) & _Base.fingerprintmask(T)
                iszero(fingerprint) || break
            end

            numfinds = 0
            fingerprints_per_word = _Base.fingerprints_per_word(T)
            for _ in 1:ntrials
                for j in 1:fingerprints_per_word
                    for n in 0:maxfingerprint
                        # Shift into position
                        word = UInt(n) << (i * (j - 1))
                        if n == fingerprint
                            @test _Base.inword(T, fingerprint, word) == true
                            if !_Base.inword(T, fingerprint, word) == true
                                @show fingerprint word T n j
                                error()
                            end
                            numfinds += 1
                        else
                            @test _Base.inword(T, fingerprint, word) == false
                        end
                    end
                end
            end
            @test numfinds == ntrials * fingerprints_per_word
        end
    end

    @testset "Testing Inserts" begin
        _Base = GraphANN._Base
        Bucket = _Base.Bucket

        num_words = [1,2,4]
        bits_per_fingerprint = [8,10,12,16,21,32]
        for (nwords, nbits) in Iterators.product(num_words, bits_per_fingerprint)
            word = zeros(UInt64, nwords)
            bucket = Bucket{nbits,nwords}(pointer(word))

            GC.@preserve word begin
                none_in(bucket, itr) = !any(x -> in(x, bucket), itr)

                # Get for unique fingerprints
                fingerprints = UInt[]
                num_fingerprints = _Base.fingerprints_per_word(typeof(bucket)) * nwords
                mask = _Base.fingerprintmask(typeof(bucket))
                while length(fingerprints) < num_fingerprints
                    candidate = rand(UInt64) & mask
                    if !in(candidate, fingerprints) && !iszero(candidate)
                        push!(fingerprints, candidate)
                    end
                end
                @test allunique(fingerprints)
                @test none_in(bucket, fingerprints)

                # Add fingerprint 1
                for i in 1:length(fingerprints)
                    @test _Base.trypush!(bucket, fingerprints[i])
                    @test in(fingerprints[i], bucket)

                    @test none_in(bucket, fingerprints) == false
                    @test none_in(bucket, fingerprints[i+1:end]) == true
                end
                @test all(x -> in(x, bucket), fingerprints)

                # With this done, trying to add another fingerprint should fail.
                new_fingerprint = zero(UInt64)
                while !iszero(new_fingerprint) && !in(new_fingerprint, fingerprints)
                    new_fingerprint = UInt64(rand(UInt16))
                end
                @test _Base.trypush!(bucket, new_fingerprint) == false

                # Adding existing fingerprints should work though since they already exist.
                @test _Base.trypush!(bucket, first(fingerprints)) == true
            end
        end
    end
end
