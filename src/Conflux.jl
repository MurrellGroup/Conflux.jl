module Conflux

import Flux
import CUDA
CUDA.allowscalar(false)

include("device.jl")
include("replicate.jl")
include("omg.jl")
include("reduce.jl")

end