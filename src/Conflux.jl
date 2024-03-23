module Conflux

import Flux
import CUDA, NCCL
CUDA.allowscalar(false)

# Communicators would get finalized after precompilation, so they are initialized in __init__
const COMMS = Ref{Vector{NCCL.Communicator}}(NCCL.Communicator[])

function __init__()
    append!(COMMS[], NCCL.Communicators(CUDA.devices()))
end

include("reduce.jl")
include("device.jl")
include("replicate.jl")

end