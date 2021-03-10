# Approximate Similarity Search Library

**WHAT IS IT**

## Installation Instructions

### Installing Julia

This library is implemented in [Julia](https://julialang.org/), so naturally the Julia runtime/compiler is required.
For best performance, use Julia 1.6.0-rc1 or higher.
Installing Julia can be be done using the commands below:
```sh
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.0-rc1-linux-x86_64.tar.gz
tar -xvf julia-1.6.0-rc1-linux-x86_64.tar.gz
```
Julia can then be launched from `./julia-1.6.0-rc1/bin/julia`.

### Installing GraphANN

At the moment, the GraphANN library is visible to Julia's package manager.
However, using the library is straightforward.
Navigate to the repo's directory on your local machine and start Julia with the `--project` flag.
```
cd GraphANN
<path/to/julia> --project
```
Once in the Julia REPL (Read-Eval-Print loop), activate the project with
```julia
julia> using GraphANN
```
To run the test suite, use:
```julia
julia> using Pkg; Pkg.test()
```
