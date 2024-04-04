export allreduce!, avg

#=import NCCL: avg

"""
    allreduce!(op, arrays...)

Perform all-reduce with operator `op` on `arrays`. This mutates the arrays in place,
and leaves them as identical copies. The arrays must have the same size.

# Examples

```jldoctest
julia> import Conflux, CUDA

julia> arrays = map(0:1) do i
           CUDA.device!(i)
           CUDA.fill(Float32(i), 4)
       end
2-element Vector{CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}:
 Float32[0.0, 0.0, 0.0, 0.0]
 Float32[1.0, 1.0, 1.0, 1.0]

julia> Conflux.allreduce!(Conflux.avg, arrays...) # set the arrays to be the average of all arrays

julia> arrays
2-element Vector{CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}:
 Float32[0.5, 0.5, 0.5, 0.5]
 Float32[0.5, 0.5, 0.5, 0.5]

julia> Conflux.allreduce!(+, arrays...) # set the arrays to be the sum of all arrays

julia> arrays
2-element Vector{CUDA.CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}:
 Float32[1.0, 1.0, 1.0, 1.0]
 Float32[1.0, 1.0, 1.0, 1.0]
```
"""
function allreduce!(op, arrays::Vararg{<:CUDA.CuArray})
    allequal(size.(arrays)) || throw(ArgumentError("Arrays must have the same size"))
    NCCL.group() do
        for array in arrays
            device = CUDA.device(array)
            comm = COMMS[][findfirst(==(device), collect(CUDA.devices()))]
            CUDA.device!(device)
            NCCL.Allreduce!(array, array, op, comm)
        end
    end
    return nothing
end

"""
    allreduce!(op, xs...)

Perform all-reduce with operator `op` on the parameters of each element in `xs`, e.g. models or optimizer states.
Useful for synchronizing parameters across devices with the `avg` operator

# Examples

```jldoctest
julia> import Conflux, Flux, CUDA

julia> model = Dense(1,1)
Dense(1 => 1)       # 2 parameters

julia> models = replicate(model)
2 replicas on devices [CuDevice(0), CuDevice(1)]

julia> Flux.params.(models)
2-element Vector{Zygote.Params{Zygote.Buffer{Any, Vector{Any}}}}:
 Params([Float32[0.39819413;;], Float32[0.0]])
 Params([Float32[0.39819413;;], Float32[0.0]])

julia> Conflux.allreduce!(+, models)

julia> Flux.params.(models)
2-element Vector{Zygote.Params{Zygote.Buffer{Any, Vector{Any}}}}:
 Params([Float32[0.79638827;;], Float32[0.0]])
 Params([Float32[0.79638827;;], Float32[0.0]])
```
"""
function allreduce!(op, xs...)
    for arrays in zip(collect.(Flux.params.(xs))...)
        allreduce!(op, arrays...)
    end
    return nothing
end=#

"""
Manually averages parameters with CUDA instead of using NCCL.
"""
function mul!(ps::Flux.Params, x::Real)
    device!(first(collect(ps)))
    for array in ps
        array .*= x
    end
    return ps
end

function add!(dest_params::Flux.Params, src_params::Flux.Params; free=false)
    dest_arrays = collect(dest_params)
    dest_device = flux_device(first(dest_arrays))
    src_arrays = transfer_to_device(collect(src_params), dest_device; free=free)
    device!(dest_device)
    CUDA.@sync for (dest_array, src_array) in zip(dest_arrays, src_arrays)
        CUDA.@async begin
            dest_array .+= src_array
            CUDA.unsafe_free!(src_array)
        end
    end
    return dest_params
end

function reducefirst!(replicas::Replicas; free = false)
    params_vector = Flux.params.(replicas)
    master_params = foldl((dest_params, src_params) -> add!(dest_params, src_params; free=free), params_vector)
    mul!(master_params, 1/length(params_vector))
    return replicas
end

using Base.Threads

function synchronize!(replicas::Replicas)
    params_vector = Flux.params.(replicas)
    master_params = params_vector[1]
    @sync for params in params_vector[2:end]
        Threads.@spawn CUDA.@sync for (param, master_param) in zip(params, master_params)
            CUDA.@async CUDA.CUDA.copyto!(param, master_param)
        end
    end
    return replicas
end