export withdevices

flux_device(x::Flux.AbstractDevice) = x

flux_device(x::CUDA.CuDevice) = Flux.get_device("CUDA", Int(x.handle))
flux_device(x::CUDA.CuArray) = flux_device(CUDA.device(x))

cuda_device(x::Flux.FluxCUDADevice) = x.deviceID
cuda_device(x) = cuda_device(flux_device(x))

flux_devices(xs) = flux_device.(xs)

cuda_devices() = collect(CUDA.devices())
cuda_devices(xs) = cuda_device.(xs)

device!(x::Flux.FluxCUDADevice) = CUDA.device!(Int(cuda_device(x).handle))
device!(x) = device!(flux_device(x))

function device!(f, x::Flux.FluxCUDADevice)
    device!(x)
    f()
end

flux_cuda_devices() = flux_device.(cuda_devices())

"""
    withdevices(f, devices=flux_cuda_devices())

Run `f` on each device in `devices`, returning a vector of the results.
"""
function withdevices(f, devices=flux_cuda_devices(); async=true)
    if async
        CUDA.@sync map(enumerate(flux_devices(devices))) do (i, device)
            CUDA.@async begin
                device!(device)
                f((i, device))
            end
        end .|> fetch
    else
        map(enumerate(flux_devices(devices))) do (i, device)
            device!(device)
            f((i, device))
        end
    end
end