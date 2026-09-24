# Blade surface export for ParaView. CCBlade only knows airfoil polars, not shapes, so
# the section shape is a NACA 4-digit profile chosen for visualization (circles for the
# NREL "Cylinder" stations). Coordinates: rotor axis = z, blade 1 along +x, pitch axis
# at the quarter chord, units meters.

"NACA 4-digit coordinates as a closed loop (TE -> upper -> LE -> lower -> TE) with 2n-1 points."
function naca4_coordinates(digits::AbstractString; n = 50)
    (length(digits) == 4 && all(isdigit, digits)) || fail("airfoil_shape must be a NACA 4-digit string such as 4412 or 0012")
    m = parse(Int, digits[1]) / 100
    pp = parse(Int, digits[2]) / 10
    t = parse(Int, digits[3:4]) / 100
    beta = range(0, pi, length = n)
    xc = (1 .- cos.(beta)) ./ 2
    yt = 5t .* (0.2969 .* sqrt.(xc) .- 0.1260 .* xc .- 0.3516 .* xc .^ 2 .+ 0.2843 .* xc .^ 3 .- 0.1036 .* xc .^ 4)
    yc = zeros(n); dyc = zeros(n)
    if m > 0 && pp > 0
        for (i, x) in enumerate(xc)
            if x < pp
                yc[i] = m / pp^2 * (2pp * x - x^2); dyc[i] = 2m / pp^2 * (pp - x)
            else
                yc[i] = m / (1 - pp)^2 * ((1 - 2pp) + 2pp * x - x^2); dyc[i] = 2m / (1 - pp)^2 * (pp - x)
            end
        end
    end
    th = atan.(dyc)
    xu = xc .- yt .* sin.(th); yu = yc .+ yt .* cos.(th)
    xl = xc .+ yt .* sin.(th); yl = yc .- yt .* cos.(th)
    return vcat(reverse(xu), xl[2:end]), vcat(reverse(yu), yl[2:end])
end

"Circle of diameter 1 centered at mid chord, 2n-1 points, for cylindrical root stations."
function circle_coordinates(; n = 50)
    th = range(0, 2pi, length = 2n - 1)
    return 0.5 .+ 0.5 .* cos.(th), 0.5 .* sin.(th)
end

"Structured surface arrays (points x stations) for one blade before azimuthal rotation."
function blade_surface(g::Geometry; shape = g.airfoil_shape, n = 50)
    npts = 2n - 1
    nst = length(g.r)
    X = zeros(npts, nst); Y = zeros(npts, nst); Z = zeros(npts, nst)
    for j in 1:nst
        cyl = startswith(lowercase(g.airfoil_names[j]), "cylinder")
        xa, ya = cyl ? circle_coordinates(n = n) : naca4_coordinates(shape; n = n)
        c = g.chord_over_R[j] * g.Rtip
        th = g.twist_deg[j] * pi / 180
        xi = (0.25 .- xa) .* c        # chordwise, leading edge positive, pitch axis at c/4
        zeta = ya .* c                # thickness direction
        X[:, j] .= g.r[j]
        Y[:, j] .= xi .* cos(th) .+ zeta .* sin(th)
        Z[:, j] .= -xi .* sin(th) .+ zeta .* cos(th)
    end
    return X, Y, Z
end

"""
    export_blade_vtk(geom; out=nothing, shape, output_dir, name, n) -> Dict

Write a multiblock VTK file with every blade and a hub cylinder. When CCBlade outputs
are given, the spanwise loads and angles are attached as point data for coloring.
"""
function export_blade_vtk(g::Geometry; out = nothing, shape = g.airfoil_shape, output_dir = OUTPUT_DIR,
                          name = "blade", n = 50)
    mkpath(output_dir)
    X, Y, Z = blade_surface(g; shape = shape, n = n)
    npts, nst = size(X)
    base = joinpath(output_dir, name)
    vtm = vtk_multiblock(base)

    # WriteVTK wants 3D arrays for a surface embedded in 3D: use a singleton third axis.
    grid3(A) = reshape(A, size(A, 1), size(A, 2), 1)
    spanwise(v) = grid3(repeat(reshape(collect(v), 1, :), npts, 1))   # one value per station
    o = out === nothing ? nothing : (out isa AbstractMatrix ? out[:, 1] : out)
    point_data = ["chord_m", "twist_deg", "r_over_R"]
    for k in 1:g.num_blades
        psi = 2pi * (k - 1) / g.num_blades
        Xk = X .* cos(psi) .- Y .* sin(psi)
        Yk = X .* sin(psi) .+ Y .* cos(psi)
        vtk = vtk_grid(vtm, grid3(Xk), grid3(Yk), grid3(Z))
        vtk["chord_m"] = spanwise(g.chord_over_R .* g.Rtip)
        vtk["twist_deg"] = spanwise(g.twist_deg)
        vtk["r_over_R"] = spanwise(g.r_over_R)
        if o !== nothing
            vtk["Np_N_per_m"] = spanwise(field(o, :Np))
            vtk["Tp_N_per_m"] = spanwise(field(o, :Tp))
            vtk["alpha_deg"] = spanwise(field(o, :alpha) .* (180 / pi))
            vtk["cl"] = spanwise(field(o, :cl))
            vtk["cd"] = spanwise(field(o, :cd))
            k == 1 && append!(point_data, ["Np_N_per_m", "Tp_N_per_m", "alpha_deg", "cl", "cd"])
        end
    end

    # hub cylinder
    nth = 37
    hl = 0.6 * maximum(g.chord_over_R) * g.Rtip
    ths = range(0, 2pi, length = nth)
    zs = (-hl / 2, hl / 2)
    HX = [g.Rhub * cos(t) for t in ths, z in zs]
    HY = [g.Rhub * sin(t) for t in ths, z in zs]
    HZ = [z for t in ths, z in zs]
    vtk_grid(vtm, grid3(HX), grid3(HY), grid3(HZ))

    files = vtk_save(vtm)
    return Dict{String,Any}(
        "main_file" => base * ".vtm",
        "files" => files,
        "num_blades" => g.num_blades,
        "points_per_section" => npts,
        "num_sections" => nst,
        "section_shape" => shape,
        "point_data" => point_data,
        "coordinate_system" => "rotor axis = z, blade 1 along +x, pitch axis at quarter chord, meters",
        "paraview" => "Open the .vtm in ParaView (File > Open, then Apply). Color by twist_deg or, if loads were attached, Np_N_per_m. The hub is the last block.",
    )
end
