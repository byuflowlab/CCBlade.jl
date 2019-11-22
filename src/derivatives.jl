import ForwardDiff

struct PartialsWrt{TF}
    phi::TF

    r::TF
    chord::TF
    theta::TF

    Vx::TF
    Vy::TF
    rho::TF
    mu::TF
    asound::TF

    Rhub::TF
    Rtip::TF
    pitch::TF
    precone::TF

end

# Need this for the mapslices call in output_partials.
PartialsWrt(x::AbstractArray) = PartialsWrt(x...)
    
function residual_partials(phi, rotor, section, op)
    # unpack inputs
    r = section.r
    chord = section.chord
    theta = section.theta
    af = section.af
    Rhub = rotor.Rhub
    Rtip = rotor.Rtip
    B = rotor.B
    turbine = rotor.turbine
    pitch = rotor.pitch
    precone = rotor.precone
    Vx = op.Vx
    Vy = op.Vy
    rho = op.rho
    mu = op.mu
    asound = op.asound

    # Get a version of the residual function that's compatible with ForwardDiff.
    function R(inputs)
        # The order of inputs should always match the order of fields in the
        # PartialsWrt struct.
        _phi = inputs[1]
        _r = inputs[2]
        _chord = inputs[3]
        _theta = inputs[4]
        _Vx, _Vy, _rho, _mu, _asound = inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]
        _Rhub, _Rtip, _pitch, _precone = inputs[10], inputs[11], inputs[12], inputs[13]
        _section = Section(_r, _chord, _theta, af)
        _op = OperatingPoint(_Vx, _Vy, _rho, _mu, _asound)
        _rotor = Rotor(_Rhub, _Rtip, B, turbine, _pitch, _precone)
        res, out = residual(_phi, _rotor, _section, _op)
        return res
    end

    # Do it.
    x = [phi, r, chord, theta, Vx, Vy, rho, mu, asound, Rhub, Rtip, pitch, precone]
    return PartialsWrt(ForwardDiff.gradient(R, x))

end

function output_partials(phi, rotor, section, op)
    # unpack inputs
    r = section.r
    chord = section.chord
    theta = section.theta
    af = section.af
    Rhub = rotor.Rhub
    Rtip = rotor.Rtip
    B = rotor.B
    turbine = rotor.turbine
    pitch = rotor.pitch
    precone = rotor.precone
    Vx = op.Vx
    Vy = op.Vy
    rho = op.rho
    mu = op.mu
    asound = op.asound

    # Get a version of the output function that's compatible with ForwardDiff.
    function R(inputs)
        # The order of inputs should always match the order of fields in the
        # PartialsWrt struct.
        _phi = inputs[1]
        _r = inputs[2]
        _chord = inputs[3]
        _theta = inputs[4]
        _Vx, _Vy, _rho, _mu, _asound = inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]
        _Rhub, _Rtip, _pitch, _precone = inputs[10], inputs[11], inputs[12], inputs[13]
        _section = Section(_r, _chord, _theta, af)
        _op = OperatingPoint(_Vx, _Vy, _rho, _mu, _asound)
        _rotor = Rotor(_Rhub, _Rtip, B, turbine, _precone)
        res, out = residual(_phi, _rotor, _section, _op)
        return [getfield(out, i) for i in fieldnames(typeof(out))]
    end

    # Do it.
    x = [phi, r, chord, theta, Vx, Vy, rho, mu, asound, Rhub, Rtip, pitch, precone]
    return Outputs(mapslices(PartialsWrt, ForwardDiff.jacobian(R, x), dims=2)...)
end

