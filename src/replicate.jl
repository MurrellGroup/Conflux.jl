export replicate

"""
    Replicas{T, D <: Flux.AbstractDevice} <: AbstractVector{T}

A vector of instances of type `T`, replicated across devices with type `D`.
Can be indexed by device to get the replica on that device.
"""
struct Replicas{T, D <: Flux.AbstractDevice} <: AbstractVector{T}
    replicas::Vector{T}
    devices::Vector{D}

    function Replicas(replicas::Vector, devices::Vector{D}) where D <: Flux.AbstractDevice
        replicas = replicas .|> devices
        T = eltype(replicas)
        return new{T, D}(replicas, devices)
    end
end

Base.size(replicas::Replicas) = size(replicas.replicas)
Base.getindex(replicas::Replicas, i::Integer) = replicas.replicas[i]
Base.getindex(replicas::Replicas{D}, device::D) where D = replicas[findfirst(==(device), replicas.devices)]
Base.getindex(replicas::Replicas{D}, device) where D = replicas[flux_device(device)]

Base.summary(replicas::Replicas) = "$(length(replicas)) replicas on devices [$(join(map(d -> d.deviceID, replicas.devices), ", "))]"
Base.show(io::IO, replicas::Replicas) = print(io, summary(replicas))
Base.show(io::IO, ::MIME"text/plain", replicas::Replicas) = show(io, replicas)

"""
    replicate(original, devices=flux_cuda_devices(), f=identity)

Replicate `original` across `devices`, optionally applying `f` to each replica (e.g. `deepcopy`).
"""
replicate(original, devices=flux_cuda_devices(), f=identity) = Replicas([f(original) for _ in devices], flux_devices(devices))