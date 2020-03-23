module CCBladeOpenMDAOExample

using CCBlade: Rotor, Section, OperatingPoint

# convenience function to set fields within an array of structs
function Base.setproperty!(obj::Array{Rotor{TF, TI, TB, TF2}, N}, sym::Symbol, x) where {TF, TI, TB, TF2, N}
    setfield!.(obj, sym, x)
end

# convenience function to set fields within an array of structs
function Base.setproperty!(obj::Array{Section{TF1, TF2, TF3, TAF}, N}, sym::Symbol, x) where {TF1, TF2, TF3, TAF, N}
    setfield!.(obj, sym, x)
end

# convenience function to set fields within an array of structs
function Base.setproperty!(obj::Array{OperatingPoint{TF, TF2}, N}, sym::Symbol, x) where {TF, TF2, N}
    setfield!.(obj, sym, x)
end

include("derivatives.jl")
include("openmdao.jl")

end # module
