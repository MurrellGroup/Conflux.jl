function transfer_to_device(arrays, device::Flux.AbstractDevice; free=false)
    device!(device)
    new_arrays = arrays |> device
    free && CUDA.unsafe_free!.(arrays)
    return new_arrays
end

function transfer_to_device(params::Flux.Params, device::Flux.AbstractDevice; free=false)
    return Flux.Params(transfer_to_device(collect(params), device; free=free))
end