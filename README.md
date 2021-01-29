# GraphANN

## Getting the Code

Simply download using Git:
```sh
git clone ssh://git@gitlab.devtools.intel.com:29418/stg-ai-sw-team/GraphAnn.jl.git GraphANN
```

## Getting Julia

Installing Julia is straightforward.
Precompiled binaries for many systems are found at: <https://julialang.org/downloads/>.

Example Linux commands for downloading and unpacking Julia 1.5.3 is shown below.
```sh
wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
tar -xvf julia-1.5.3-linux-x86_64.tar.gz
```
Julia can then be run with the command
```sh
./julia-1.5.3/bin/julia
```

## Generating Ground Truth

Generating ground truth consists of several steps.
The high level flow is presented first, with detailed information later.

1. Start Julia with a large number of threads (optionally, in a `tmux` session).
2. Load the dataset and the query set.
3. Compute ground truth.
4. Save results to a file.

### Starting Julia (and tmux)

The program `tmux` <https://github.com/tmux/tmux/wiki> enables applications to run in the background and persist across logins.
It's very helpful for long-running applications.

Now, for best results, start Julia with as many threads as possible by setting the `JULIA_NUM_THREADS` environment variable.
For example,
```sh
JULIA_NUM_THREADS=$(nproc)
```

Then, navigate to the `GraphANN` directory and launch Julia (be sure to use the `--project` flag).
```sh
</path/to/julia> --project
```
If this is your first time running the `GraphANN` code, you will need to download the dependencies (make sure your proxy settings are correct).
```julia
using Pkg; Pkg.instantiate();
```
Now that all the dependencies are downloaded, import the `GraphANN` code using
```julia
using GraphANN
```

### Loading Data and Queries

Now that Julia is up and running, we need to get some data into the program.
At the moment, the code only knows how to load and save data in the `.[i|b|f]vecs` form, but more can be added if desired.
To load the dataset, run
```julia
dataset = GraphANN.load_vecs(GraphANN.Euclidean{128,UInt8}, "/path/to/dataset"; [maxlines = 1_000_000])
```
This will, as the name implies, load data from the `vecs` file format.
The optional keyword `maxlines` allows you to limit the number of points retrieved.
If it is not set, then `load_vecs` will load the entire dataset.

Loading the query set is similar.
```julia
queries = GraphANN.load_vecs(GraphANN.Euclidean{128,UInt8}, "/path/to/queries")
```

### Generating the Ground Truth

A single function call will generate the ground truth:
```julia
gt = GraphANN.bruteforce_search(queries, data, [numneighbors]; [groupsize = 32], [savefile = "qroundtruth.ivecs"])
```
This will generate the ground truth for `queries` with respect to `data` and store the results as a 2D matrix `gt`.
The optional argument `numneighbors` controls the number of exact nearest neighbors neighbors found (defaulting to 100).

Two keyword arguments are useful:

* `groupsize` - This is an implementation detail, but controls how many queries are assigned to a thread at a time.
    Set this somewhere between 32 and 64 for the fastest results.

    **Note**: This argument has no effect on the results, only on computation time.

* `savefile` - If this argument is set, then the function will automatically (and periodically) save the results to the corresponding file.
    This is especially useful when computing the ground truth for a large number of queries as the results are saved periodically, ensuring that if the program is interrupted for some reason, partial progress is not lost.

### Saving Results

Althrough `GraphANN.bruteforce_search` will automatically save results if the `savefile` keyword argument is set, you might want to manually save results.
Simply use
```sh
GraphANN.save_vecs("groundtruth.ivecs", gt)
```

### Example run

Here, we show an example of running the whole pipeline.

```julia
using GraphANN

# Get the first 10 million entries in the Sift 1B dataset.
# Note: trailing semi-colons are optional and just surpress the printing of the results to the REPL.
dataset = GraphANN.load_vecs(GraphANN.Euclidean{128,UInt8}, "/data/bigann.bvecs"; maxlines = 10_000_000);
queries = GraphANN.load_vecs(GraphANN.Euclidean{128,UInt8}, "/data/query.bvecs");

gt = GraphANN.bruteforce_search(queries, dataset, 100; groupsize = 64, savefile = "groundtruth.ivecs")
GraphANN.save_vecs("another_groundtruth.ivecs", gt)
```

## Next Steps

1. Measure Bandwidth - are we bandwidth limited?
2. Ways of improving accuracy or performance.
    - Do something else while waiting for data?
    - Gather data more quickly.

3. Measure accuracy after only a single round of pruning.

4. Put documentation on graph serialization format ... actually with the serialized graphs.
(and probably in the git repo as well)

5. Measure exploration vs direct path when navigating to nearest neighbor (vague, but have general idea).


# TODO:

Picture of graphs and adjacency lists
Picture of dense adjacency list - How is it saving space.
