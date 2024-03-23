export withdevices

flux_device(x::Flux.AbstractDevice) = x

flux_device(x::CUDA.CuDevice) = Flux.get_device("CUDA", Int(x.handle))
flux_device(x::CUDA.CuArray) = flux_device(CUDA.device(x))

cuda_device(x::Flux.FluxCUDADevice) = x.deviceID
cuda_device(x) = cuda_device(flux_device(x))

flux_devices() = flux_device.(CUDA.devices())
flux_devices(xs) = flux_device.(xs)

cuda_devices() = collect(CUDA.devices())
cuda_devices(xs) = cuda_device.(xs)

set_device!(x::Flux.FluxCUDADevice) = CUDA.device!(Int(cuda_device(x).handle))
set_device!(x) = set_device!(flux_device(x))

"""
    withdevices(f, devices=flux_devices())

Run `f` on each device in `devices`, returning a vector of the results.
"""
function withdevices(f, devices::Vector{Flux.FluxCUDADevice}=flux_devices())
    CUDA.@sync map(enumerate(devices)) do (i, device)
        CUDA.@async begin
            set_device!(device)
            f((i, device))
        end
    end .|> fetch
end

withdevices(f, devices) = withdevices(f, flux_devices(devices))