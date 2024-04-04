# Conflux

[![Latest Release](https://img.shields.io/github/release/MurrellGroup/Conflux.jl.svg)](https://github.com/MurrellGroup/Conflux.jl/releases/latest)
[![MIT license](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/license/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://MurrellGroup.github.io/Conflux.jl/dev/)

Conflux.jl is a toolkit designed to enable data parallelism for [Flux.jl](https://github.com/FluxML/Flux.jl) models by simplifying the process of replicating them across multiple GPUs on a single node, and by leveraging [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) for efficient inter-GPU communication. This package aims to provide a straightforward and intuitive interface for multi-GPU training, requiring minimal changes to existing code and training loops.

## Features

- Easy replication of objects across multiple GPUs with the **replicate** function
- Efficient synchronization of models and averaging of gradients with the **allreduce!** function, which takes an operator (e.g. `+`, `*`, `avg`) and a set of replicas, and reduces all their parameters with the given operator, leaving the replicas identical.
- A **withdevices** function that allows you to run code on each device asynchronously.

See the documentation for more details, examples, and important caveats.

## Installation

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add https://github.com/MurrellGroup/Conflux.jl#main
```

## Example usage

```julia
# Specify the default devices to use
ENV["CUDA_VISIBLE_DEVICES"] = "0,1"

using Conflux

using Flux, Optimisers

model = Chain(Dense(1 => 256, tanh), Dense(256 => 512, tanh), Dense(512 => 256, tanh), Dense(256 => 1))

# This will use the available devices. If you want to use a specific device, you can pass them in a second argument.
models = replicate(model)

opt = Optimisers.Adam(0.001f0)

# Instantiate the optimiser states on each device
state = Optimisers.setup(opt, model) |> models.devices[1]
master_model = models[1]

# A single batch, stored on CPU. Could use a more sophisticated mechanism to distribute multiple batches.
X = rand(1, 16)
Y = X .^ 2

loss(y, Y) = sum(abs2, y .- Y)

losses = []
for epoch in 1:10
    # Get the gradients for each batch on each device
    ∇models = Conflux.withdevices() do (i, device)
        x, y = device(X), device(Y)
        # The second return value is a tuple because `Flux.withgradient` takes `args...`, and the model is the first argument.
        l, (∇model,) = Flux.withgradient(m -> loss(m(x), y), models[i])
        push!(losses, l)
        ∇model
    end

    Conflux.reduce!(∇models...)
    Optimisers.update!(master_model, state, ∇models[1])

    Conflux.synchronize!(models)
end
```