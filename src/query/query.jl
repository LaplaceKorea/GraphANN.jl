# Telemetry allows for optional expansion for metrics gathered during query.
# It's designed to allow telemetry to be opt in with no performance loss when telemetry
# is not applied.
#
# NOTE: It's the algorithm's responsibility to implement the correct telemetry hooks
include("telemetry.jl"); import ._Telemetry: Telemetry, ifhasa
include("greedy.jl")
