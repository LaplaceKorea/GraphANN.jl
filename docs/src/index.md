# Threading Constructs

## ThreadPool

Julia has initial support for multithreading, but we need slightly finer grained control for two reasons.
First, many of our routines are long running and the default static schedule provided by Julia does not do a good enough job with load balancing.
Second, we want the ability to launch prefetch threads in the background that are running asynchronously with query threads.
These background threads should run on different physical cores (or physical threads I guess) than than the query threads.
To achieve this, we need to abstract away the notion of a `ThreadPool` that will launch tasks on a subset of the available threadids.

This is achieved by the `ThreadPool` type.
Basically, this is just a wrapper around a vector of thread ids that belong to the thread pool.
Thus, if a thread pool is created around the range `1:10`, then the corresponding function will be run on threads 1 to 10.
Two methods are provided for launching tasks on thread pools

* `on_threads` - Take a zero-argument function and a thread pool.
    The function is run on each thread in the thread pool.
    By default, this function blocks until thread execution is complete.
    To achieve non-blocking behavior, pass `false` to the third argument.

    If this is the case, than `on_threads` will return a `TaskHandle` immediately after launching all tasks.
    This handle contains references to the launched tasks and can be waited for by calling `wait`.
    At some point, `wait` should be called since at this point, any error that occured in the launched tasks will propagate to the top level, allowing for debugging.

* `dynamic_thread` - By default, Julia uses a static schedule.
    For better load balancing, the `dynamic_thread` construct will dynamically assign work to threads in a `ThreadPool` across some `domain`.
    To improve locality, an optional `worksize` argument is included which will partition the domain into `worksize`d batches.

If the finer grained thread control is not needed, the function `allthreads()` will return a thread pool containing all available threads.

## ThreadLocal

A common pattern in Julia is to pre-allocate data structures and mutate these data structures (because allocation is somewhat expensive due to garbage collection.)
When multithreading, it is convenient to give each thread some local storage that only it has access to, avoiding synchronization overheads.
The `ThreadLocal` type is a type that helps with this, creating a deep copy for each thread of whatever object is passed to its constructors.
When accessed with the `getindex` syntax (like `threadlocal[]`), the `ThreadLocal` type will automatically return the storage for the current thread.
This is a powerful pattern when combined with a `ThreadPool` to provide local storage.

