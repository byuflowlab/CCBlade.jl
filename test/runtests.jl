using CCBlade
using Test


@testset "normal operation" begin


# --- verification using example from "Wind Turbine Blade Analysis using the Blade Element Momentum Method"
# --- by Grant Ingram: https://community.dur.ac.uk/g.l.ingram/download/wind_turbine_design.pdf

# Note: There were various problems with the pdf data. Fortunately the excel
# spreadsheet was provided: http://community.dur.ac.uk/g.l.ingram/download.php

# - They didn't actually use the NACA0012 data.  According to the spreadsheet they just used cl = 0.084*alpha with alpha in degrees.
# - There is an error in the spreadsheet where CL at the root is always 0.7.  This is because the classical approach doesn't converge properly at that station.
# - the values for gamma and chord were rounded in the pdf.  The spreadsheet has more precise values.
# - the tip is not actually converged (see significant error for Delta A).  I converged it further using their method.


# --- rotor definition ---
Rhub = 0.01
Rtip = 5.0
Rtip_eff = 5.0*100  # to eliminate tip effects as consistent with their study.
B = 3  # number of blades

rotor = Rotor(Rhub, Rtip_eff, B, turbine=true)

# --- section definitions ---

r = [0.2, 1, 2, 3, 4, 5]
gamma = [61.0, 74.31002131, 84.89805553, 89.07195504, 91.25038415, 92.58003871]
theta = (90.0 .- gamma)*pi/180
chord = [0.7, 0.706025153, 0.436187551, 0.304517933, 0.232257636, 0.187279622]

function affunc(alpha, Re, M)

    cl = 0.084*alpha*180/pi

    return cl, 0.0
end

sections = Section.(r, chord, theta, Ref(affunc))


# --- inflow definitions ---

Vinf = 7.0
tsr = 8
Omega = tsr*Vinf/Rtip
rho = 1.0

ops = simple_op.(Vinf, Omega, r, rho)

# --- evaluate ---

out = solve.(Ref(rotor), sections, ops)

ivec = out.phi*180/pi .- theta*180/pi
betavec = 90 .- out.phi*180/pi


# outputs[1] is uncomparable because the classical method fails at the root so they fixed cl

@test isapprox(out.a[2], 0.2443, atol=1e-4)
@test isapprox(out.ap[2], 0.0676, atol=1e-4)
@test isapprox(out.a[3], 0.2497, atol=1e-4)
@test isapprox(out.ap[3], 0.0180, atol=1e-4)
@test isapprox(out.a[4], 0.2533, atol=1e-4)
@test isapprox(out.ap[4], 0.0081, atol=1e-4)
@test isapprox(out.a[5], 0.2556, atol=1e-4)
@test isapprox(out.ap[5], 0.0046, atol=1e-4)
@test isapprox(out.a[6], 0.25725, atol=1e-4)  # note that their spreadsheet is not converged so I ran their method longer.
@test isapprox(out.ap[6], 0.0030, atol=1e-4)

@test isapprox(betavec[2], 66.1354, atol=1e-3)
@test isapprox(betavec[3], 77.0298, atol=1e-3)
@test isapprox(betavec[4], 81.2283, atol=1e-3)
@test isapprox(betavec[5], 83.3961, atol=1e-3)
@test isapprox(betavec[6], 84.7113, atol=1e-3)  # using my more converged solution


#
#

# idx = 6

# aguess = avec[idx]
# apguess = apvec[idx]

# # aguess = 0.2557
# # apguess = 0.0046

# for i = 1:100

#     sigmap = B*chord[idx]/(2*pi*r[idx])
#     lambdar = inflows[idx].Vy/inflows[idx].Vx
#     beta2 = atan(lambdar*(1 + apguess)/(1 - aguess))
#     inc2 = gamma[idx]*pi/180 - beta2
#     cl2, cd2 = affunc(inc2, 1.0, 1.0)
#     global aguess = 1.0/(1 + 4*cos(beta2)^2/(sigmap*cl2*sin(beta2)))
#     global apguess = sigmap*cl2/(4*lambdar*cos(beta2))*(1 - a2)
#     # global apguess = 1.0 / (4*sin(beta2)/(sigmap*cl2) - 1)

# end

# aguess
# apguess



# --------------------------------------------------------------



# -------- verification: propellers.  using script at http://www.aerodynamics4students.com/propulsion/blade-element-propeller-theory.php ------
# I increased their tolerance to 1e-6


# --- rotor definition ---
D = 1.6
Rhub = 0.0
Rtip = D/2
Rhub_eff = 1e-6  # something small to eliminate hub effects
Rtip_eff = 100.0  # something large to eliminate tip effects
B = 2  # number of blades

rotor_no_F = Rotor(Rhub_eff, Rtip_eff, B)
rotor = Rotor(Rhub, Rtip, B)


# --- section definitions ---

R = D/2.0
r = range(R/10, stop=R, length=11)
pitch = 1.0  # pitch distance in meters.
theta = atan.(pitch./(2*pi*r))
chord = 0.10


function affunc2(alpha, Re, M)

    cl = 6.2*alpha
    cd = 0.008 - 0.003*cl + 0.01*cl*cl

    return cl, cd
end

sections = Section.(r, chord, theta, Ref(affunc2))


# --- inflow definitions ---
RPM = 2100
rho = 1.225


tsim = 1e3*[1.045361193032356, 1.025630300048415, 1.005234466788998, 0.984163367036026, 0.962407923825140, 0.939960208707079, 0.916813564966455, 0.892962691000145, 0.868403981825492, 0.843134981103815, 0.817154838249790, 0.790463442573673, 0.763063053839278, 0.734956576558370, 0.706148261507327, 0.676643975451150, 0.646450304160057, 0.615575090105131, 0.584027074365864, 0.551815917391907, 0.518952127358381, 0.485446691671386, 0.451311288662196, 0.416557935286392, 0.381199277009438, 0.345247916141561, 0.308716772800348, 0.271618894441869, 0.233967425339051, 0.195775319296371, 0.157055230270717, 0.117820154495231, 0.078082266879117, 0.037854059197644, -0.002852754149850, -0.044026182837742, -0.085655305814570, -0.127728999394140, -0.170237722799272, -0.213169213043848, -0.256515079286031, -0.300266519551194, -0.344414094748869, -0.388949215983616, -0.433863576642539, -0.479150401337354, -0.524801553114807, -0.570810405128802, -0.617169893200684, -0.663873474163182, -0.710915862524620, -0.758291877949762, -0.805995685105502, -0.854022273120508, -0.902366919041604, -0.951025170820984, -0.999992624287163, -1.049265666456123, -1.098840222937414, -1.148712509929845]
qsim = 1e2*[0.803638686218187, 0.806984572453978, 0.809709290183008, 0.811743686838315, 0.813015017103876, 0.813446921530685, 0.812959654049620, 0.811470393912576, 0.808893852696513, 0.805141916379142, 0.800124489784850, 0.793748780791057, 0.785921727832179, 0.776548246109426, 0.765532528164390, 0.752778882688809, 0.738190986274448, 0.721673076180745, 0.703129918771009, 0.682467282681955, 0.659592296506578, 0.634413303042323, 0.606840565246423, 0.576786093366321, 0.544164450503912, 0.508891967461804, 0.470887571011192, 0.430072787279711, 0.386371788290446, 0.339711042057184, 0.290019539402947, 0.237229503458026, 0.181274942660876, 0.122093307308376, 0.059623821454727, -0.006190834182631, -0.075406684829235, -0.148076528546541, -0.224253047813501, -0.303980950928302, -0.387309291734422, -0.474283793689904, -0.564946107631716, -0.659336973911858, -0.757495165410553, -0.859460291551374, -0.965266648683888, -1.074949504731187, -1.188540970723477, -1.306072104649531, -1.427575034895290, -1.553080300508925, -1.682614871422754, -1.816205997296014, -1.953879956474228, -2.095662107769925, -2.241576439746701, -2.391647474158875, -2.545897099743367, -2.704346566395035]

for i = 1:60

    Vinf = float(i)
    Omega = RPM * pi/30

    ops = simple_op.(Vinf, Omega, r, rho)

    # --- evaluate ---

    out = solve.(Ref(rotor_no_F), sections, ops)

    # Np, Tp = loads(outputs)
    # T, Q = thrusttorque(r[1], r, r[end], Np, Tp, B)
    # their spreadsheet did not use trapzezoidal rule, so just do a rect sum.
    T = sum(out.Np*(r[2]-r[1]))*B
    Q = sum(r.*out.Tp*(r[2]-r[1]))*B

    @test isapprox(T, tsim[i], atol=1e-2)  # 2 decimal places
    @test isapprox(Q, qsim[i], atol=2e-3)

end


Vinf = 20.0
Omega = RPM * pi/30

op = simple_op.(Vinf, Omega, r, rho)

out = solve.(Ref(rotor_no_F), sections, op)

T = sum(out.Np*(r[2]-r[1]))*B
Q = sum(r.*out.Tp*(r[2]-r[1]))*B
eff, CT, CQ = nondim(T, Q, Vinf, Omega, rho, rotor, "propeller")


@test isapprox(CT, 0.056110238632657, atol=1e-7)
@test isapprox(CQ, 0.004337202960642, atol=1e-8)
@test isapprox(eff, 0.735350632777002, atol=1e-6)


# ---------------------------------------------


# ------ camber ----------

alpha0 = -3*pi/180

function affunc3(alpha, Re, M)

    cl = 6.2*(alpha - alpha0)
    cd = 0.008 - 0.003*cl + 0.01*cl*cl

    return cl, cd
end


sections = Section.(r, chord, theta, Ref(affunc3))

Vinf = 5.0
Omega = RPM * pi/30
ops = simple_op.(Vinf, Omega, r, rho)

out = solve.(Ref(rotor_no_F), sections, ops)

T = sum(out.Np*(r[2]-r[1]))*B
Q = sum(r.*out.Tp*(r[2]-r[1]))*B

@test isapprox(T, 1223.0506862888788, atol=1e-8)
@test isapprox(Q, 113.79919472569034, atol=1e-8)

@test isapprox(out[4].Np, 427.3902632382494, atol=1e-8)
@test isapprox(out[4].Tp, 122.38414345762305, atol=1e-8)
@test isapprox(out[4].a, 2.2845512476210943, atol=1e-8)
@test isapprox(out[4].ap, 0.05024950801920044, atol=1e-8)
@test isapprox(out[4].u, 11.422756238105471, atol=1e-8)
@test isapprox(out[4].v, 3.2709314141649575, atol=1e-8)
@test isapprox(out[4].phi, 0.2596455971546484, atol=1e-8)
@test isapprox(out[4].alpha, 0.23369406105568025, atol=1e-8)
@test isapprox(out[4].W, 63.96697566502531, atol=1e-8)
@test isapprox(out[4].cl, 1.773534419416163, atol=1e-8)
@test isapprox(out[4].cd, 0.03413364011028978, atol=1e-8)
@test isapprox(out[4].cn, 1.7053239640124302, atol=1e-8)
@test isapprox(out[4].ct, 0.48832327407767123, atol=1e-8)
@test isapprox(out[4].F, 1.0, atol=1e-8)
@test isapprox(out[4].G, 1.0, atol=1e-8)

theta = atan.(pitch./(2*pi*r)) .- 3*pi/180

sections = Section.(r, chord, theta, Ref(affunc3))
out = solve.(Ref(rotor_no_F), sections, ops)
T = sum(out.Np*(r[2]-r[1]))*B
Q = sum(r.*out.Tp*(r[2]-r[1]))*B

@test isapprox(T, 1e3*0.962407923825140, atol=1e-3)
@test isapprox(Q, 1e2*0.813015017103876, atol=1e-4)


# -----------------------------------------------

end


@testset "qualitative check" begin


# --------- qualitative --------------
# The following tests are only qualitative.  They are not comparisons to known
# empirical data.  Rather they are from the figures used in the documentaiton.  Qualitatively
# the output is about right.  Minor changes in, for example, the airfoil interpolation method
# coudl slightly change the outputs.  The main purpose of these tests is to alert us if something
# significant changes.

Rhub = 1.5
Rtip = 63.0
B = 3
precone = 2.5*pi/180

rotor = Rotor(Rhub, Rtip, B; precone=precone, turbine=true)

r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
    56.1667, 58.9000, 61.6333]
chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
    3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
theta = pi/180*[13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
    6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]

# Define airfoils.  In this case we have 8 different airfoils that we load into an array.
# These airfoils are defined in files.
aftypes = Array{AlphaAF}(undef, 8)
aftypes[1] = AlphaAF("airfoils/Cylinder1.dat", radians=false)
aftypes[2] = AlphaAF("airfoils/Cylinder2.dat", radians=false)
aftypes[3] = AlphaAF("airfoils/DU40_A17.dat", radians=false)
aftypes[4] = AlphaAF("airfoils/DU35_A17.dat", radians=false)
aftypes[5] = AlphaAF("airfoils/DU30_A17.dat", radians=false)
aftypes[6] = AlphaAF("airfoils/DU25_A17.dat", radians=false)
aftypes[7] = AlphaAF("airfoils/DU21_A17.dat", radians=false)
aftypes[8] = AlphaAF("airfoils/NACA64_A17.dat", radians=false)

# indices correspond to which airfoil is used at which station
af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

# create airfoil array
airfoils = aftypes[af_idx]

sections = Section.(r, chord, theta, airfoils)

# operating point for the turbine
yaw = 0.0*pi/180
tilt = 5.0*pi/180
hubHt = 90.0
shearExp = 0.2

Vinf = 10.0
tsr = 7.55
rotorR = Rtip*cos(precone)
Omega = Vinf*tsr/rotorR
azimuth = 0.0*pi/180
rho = 1.225
pitch = 0.0

op = windturbine_op.(Vinf, Omega, pitch, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

out = solve.(Ref(rotor), sections, op)

# plot distributed loads
# figure()
# plot(r/Rtip, out.Np/1e3)
# plot(r/Rtip, out.Tp/1e3)
# xlabel("r/Rtip")
# ylabel("distributed loads (kN/m)")
# legend(["flapwise", "lead-lag"])

nr = length(r)
Npnorm = [0.09718339956327719, 0.13149361678303992, 0.12220741751313423, 1.1634860128761517, 1.7001259694801125, 2.0716257635881257, 2.5120015027019678, 3.1336171133495045, 3.6916824972696465, 4.388661772599469, 5.068896486058948, 5.465165634664408, 6.035059239683594, 6.539134070994739, 6.831387531628286, 6.692665814418597, 4.851568452578296]
Tpnorm = [-0.03321034106737285, -0.08727189682145081, -0.12001897678344217, 0.4696423333976085 , 0.6226256283641799 , 0.6322961942049257 , 0.6474145670774534 , 0.6825021056687035 , 0.6999861694557595 , 0.7218774840801262 , 0.7365515905555542 , 0.7493905698765747 , 0.7529143446199785 , 0.7392483947274653, 0.6981206044592225, 0.614524256128813, 0.40353047553570615]
for i = 1:nr
    @test isapprox.(out.Np[i]/1e3, Npnorm[i], atol=1e-3)
    @test isapprox.(out.Tp[i]/1e3, Tpnorm[i], atol=1e-3)
end

# T, Q = thrusttorque(rotor, sections, out)

# azangles = pi/180*[0.0, 90.0, 180.0, 270.0]
# ops = windturbine_op.(Vinf, Omega, r, precone, yaw, tilt, azangles', hubHt, shearExp, rho)
# outs = solve.(Ref(rotor), sections, ops)

# T, Q = thrusttorque(rotor, sections, outs)

# CP, CT, CQ = nondim(T, Q, Vinf, Omega, rho, rotor)

ntsr = 20  # number of tip-speed ratios
tsrvec = range(2, stop=15, length=ntsr)
cpvec = zeros(ntsr)  # initialize arrays
ctvec = zeros(ntsr)

azangles = pi/180*[0.0, 90.0, 180.0, 270.0]

for i = 1:ntsr
    Omega = Vinf*tsrvec[i]/rotorR

    ops = windturbine_op.(Vinf, Omega, pitch, r, precone, yaw, tilt, azangles', hubHt, shearExp, rho)
    outs = solve.(Ref(rotor), sections, ops)
    T, Q = thrusttorque(rotor, sections, outs)

    cpvec[i], ctvec[i], _ = nondim(T, Q, Vinf, Omega, rho, rotor, "windturbine")
end

cpvec_test = [0.02350364982213779, 0.07009444382724848, 0.1389142676582008, 0.21796006904596332, 0.30793648085907793, 0.39220377428841097, 0.4358126426438294, 0.45890373500510734, 0.4692861367774551, 0.4675297063104674, 0.45763823228659106, 0.442190136700292, 0.4231819091376136, 0.4013611296914438, 0.37661228050658546, 0.3487266071360946, 0.31761016554953614, 0.2831513131286119, 0.24519843834864297, 0.20383577257448457]
ctvec_test = [0.12346062066554207, 0.19135957414241508, 0.2753773487370145, 0.3637320163247399, 0.4608728263820876, 0.5702746116812709, 0.6510654098500409, 0.7116957736180881, 0.759888802150446, 0.799015527437428, 0.8326366210606186, 0.8629179420352486, 0.8913475399157195, 0.9187322806175379, 0.9452557920123842, 0.971065072969849, 0.9962534829381495, 1.0208678404660274, 1.0447602003736, 1.067546263425622]

for i = 1:ntsr
    @test isapprox(cpvec[i], cpvec_test[i], atol=1e-3)
    @test isapprox(ctvec[i], ctvec_test[i], atol=1e-3)
end



# rotor definition
Rhub = 0.0254*.5
Rtip = 0.0254*3.0
B = 2  # number of blades

rotor = Rotor(Rhub, Rtip, B)

r = .0254*[0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332,
     1.1128, 1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911,
     1.6708, 1.7505, 1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287,
     2.3084, 2.3881, 2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]
chord = .0254*[0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054, 0.5973, 0.5887,
          0.5794, 0.5695, 0.5590, 0.5479, 0.5362, 0.5240, 0.5111, 0.4977,
          0.4836, 0.4689, 0.4537, 0.4379, 0.4214, 0.4044, 0.3867, 0.3685,
          0.3497, 0.3303, 0.3103, 0.2897, 0.2618, 0.1920]
theta = pi/180.0*[40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899, 31.6400,
                   29.7730, 28.0952, 26.5833, 25.2155, 23.9736, 22.8421, 21.8075,
                   20.8586, 19.9855, 19.1800, 18.4347, 17.7434, 17.1005, 16.5013,
                   15.9417, 15.4179, 14.9266, 14.4650, 14.0306, 13.6210, 13.2343,
                   12.8685, 12.5233, 12.2138]

af = AlphaAF("airfoils/NACA64_A17.dat", radians=false)

sections = Section.(r, chord, theta, Ref(af))


rho = 1.225
Vinf = 10.0
Omega = 8000.0*pi/30.0

op = simple_op.(Vinf, Omega, r, rho)

outputs = solve.(Ref(rotor), sections, op)

Nptest = [1.8660880922356378, 2.113489633244873, 2.35855792055661, 2.60301402945597, 2.844874233881403, 3.1180230827072126, 3.560077224628854, 4.024057801497014, 4.480574891998562, 4.9279550384928275, 5.366395080074933, 5.79550918136406, 6.21594163851808, 6.622960527391846, 7.017012349498324, 7.3936834781240774, 7.751945902955048, 8.086176029603802, 8.393537672577372, 8.67090062789216, 8.912426896510306, 9.111379449026037, 9.264491105426602, 9.361598738055728, 9.397710628068818, 9.360730779314666, 9.236967116872792, 9.002418776792911, 8.617229305924996, 7.854554211296309, 5.839491141636506]
Tptest = [1.481919153409856, 1.5816880353415623, 1.6702432911163534, 1.7502397903925069, 1.822089134395204, 1.8965254874252537, 2.0022647148294554, 2.097706171361262, 2.178824887386094, 2.2475057498944886, 2.3058616094094666, 2.355253913018444, 2.3970643308370168, 2.4307239254050717, 2.4574034513165794, 2.4763893383410522, 2.488405728268889, 2.492461784055084, 2.4887264544021237, 2.4772963155708783, 2.457435891854637, 2.4282986089025607, 2.3902927838322237, 2.3418848562229155, 2.283388513786012, 2.2134191689454954, 2.130781255778788, 2.0328865955896, 1.9153448642630952, 1.7308451522888118, 1.2736544011110416]
unormtest = [0.1138639166930624, 0.1215785884908478, 0.12836706426605704, 0.13442694075150818, 0.13980334658952898, 0.14527249541450538, 0.15287706861957473, 0.15952130275430465, 0.16497198426904225, 0.16942902209255595, 0.17308482185019125, 0.17607059901193356, 0.17850357997819633, 0.1803781237713692, 0.18177831882512596, 0.18267665167815783, 0.18311924527883217, 0.18305367052760668, 0.182487470134022, 0.1814205780851561, 0.17980424363698078, 0.1775794222894007, 0.1747493664535781, 0.1712044765873724, 0.1669243629724566, 0.16177907188793106, 0.155616714947619, 0.14815634397029886, 0.13888287431637295, 0.12464247057085576, 0.09292645450646951]
vnormtest = [0.09042291182918308, 0.090986677078917, 0.09090479653774426, 0.09038728871285233, 0.08954144817337718, 0.08836143378908702, 0.08598138211326231, 0.08315706129440237, 0.08022297890584436, 0.07727195122065926, 0.07437202068064472, 0.07155384528141737, 0.06883664444997291, 0.0662014244622726, 0.06365995181515205, 0.061184457505937574, 0.05878201223442723, 0.056423985398129595, 0.05410845965501752, 0.05183227774671869, 0.04957767474023305, 0.04732717658478895, 0.04508635659098153, 0.04282827989707323, 0.04055808783300249, 0.03825394697198818, 0.0358976247398962, 0.033456013675480394, 0.030869388594900515, 0.027466462150913407, 0.020268236545135546]
nr = length(r)
for i = 1:nr
    @test isapprox(outputs.Np[i], Nptest[i], atol=1e-3)
    @test isapprox(outputs.Tp[i], Tptest[i], atol=1e-3)
    @test isapprox(outputs.u[i]/Vinf, unormtest[i], atol=1e-3)
    @test isapprox(outputs.v[i]/Vinf, vnormtest[i], atol=1e-3)
end


nJ = 20  # number of advance ratios

J = range(0.1, 0.9, length=nJ)  # advance ratio

Omega = 8000.0*pi/30
n = Omega/(2*pi)
D = 2*Rtip

eff = zeros(nJ)
CT = zeros(nJ)
CQ = zeros(nJ)

for i = 1:nJ
    Vinf = J[i] * D * n

    op = simple_op.(Vinf, Omega, r, rho)
    outputs = solve.(Ref(rotor), sections, op)
    T, Q = thrusttorque(rotor, sections, outputs)
    eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, rotor, "propeller")

end

CTtest = [0.12007272369269596, 0.11685168049381163, 0.113039580103141, 0.1085198115279716, 0.10319003920478352, 0.09725943970861005, 0.09086388629270399, 0.0839976424307642, 0.07671734636145504, 0.06920018426158829, 0.06146448769092457, 0.05351803580676398, 0.04536760940952122, 0.037021384017275026, 0.028444490366496933, 0.019460037067085302, 0.0101482712198453, 0.0009431056296630114, -0.008450506440868786, -0.018336716057292365]
CPtest = [0.04881364095025621 , 0.0495814889974431, 0.05010784517185014, 0.0503081363797726, 0.05016808708430908, 0.04959928236064606, 0.04857913603557237, 0.0470530147170992, 0.045034787733797786, 0.042554367414565766, 0.03958444136790737, 0.036104210007643946, 0.032085430933805684, 0.02750011065614528, 0.022307951002430132, 0.016419276860939747, 0.009846368564156775, 0.002833135813003493, -0.004876593891333574, -0.013591625085158215]
etatest = [0.24598190455626265, 0.3349080300487075, 0.4155652767326253, 0.48818637673414306, 0.5521115225679999, 0.6089123481436948, 0.6595727776885079, 0.7046724703349897, 0.7441662053086512, 0.7788447616541276, 0.8090611349633181, 0.8347808848055981, 0.8558196582739432, 0.8715046719672315, 0.8791362131978436, 0.8670633642311274, 0.7974063895510229, 0.2715632768892098, 0.0, 0.0]

for i = 1:nJ
    @test isapprox(CT[i], CTtest[i], atol=1e-3)
    @test isapprox(CQ[i]*2*pi, CPtest[i], atol=1e-3)
    @test isapprox(eff[i], etatest[i], atol=1e-3)
end




# ------ hover verification --------

# https://rotorcraft.arc.nasa.gov/Publications/files/RamasamyGB_ERF10_836.pdf


Rhub = 0.19*Rtip
Rtip = 0.656
B = 3
rotor = Rotor(Rhub, Rtip, B)

r = range(Rhub + 0.01*Rtip, Rtip - 0.01*Rtip, length=30)
chord = 0.060
theta = 0.0

af = AlphaAF("airfoils/naca0012v2.txt", radians=false)
function af2(alpha, Re, M)
    cl, cd = afeval(af, alpha, Re, M)
    return cl, cd+0.014
end
sections = Section.(r, chord, theta, Ref(af2))


rho = 1.225
Omega = 800*pi/30
Vinf = 0.0


nP = 40
pitch = range(1e-4, 20*pi/180, length=nP)

CT = zeros(nP)
CQ = zeros(nP)


for i = 1:nP

    op = simple_op.(Vinf, Omega, r, rho, pitch=pitch[i])
    outputs = solve.(Ref(rotor), sections, op)
    T, Q = thrusttorque(rotor, sections, outputs)
    _, CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, rotor, "helicopter")
end

# these are not directly from the experimental data, but have been compared to the experimental data and compare favorably.
# this is more of a regression test on the Vx=0 case
CTcomp = [9.452864991304056e-9, 6.569947366946672e-5, 0.00022338783939012262 , 0.0004420355541809959, 0.0007048495858030926, 0.0010022162314665929, 0.0013268531109981317, 0.0016736995380106938, 0.0020399354072946035, 0.0024223576277264307, 0.0028189858460418893, 0.0032281290309981213, 0.003649357660426685, 0.004081628946214875, 0.004526034348853718, 0.004982651929181267, 0.0054553705714941, 0.005942700094508395, 0.006447634897014323, 0.006963626871239654, 0.007492654931894796, 0.00803866268066438, 0.008597914974368199, 0.009163315934297088, 0.00973817187875574, 0.010309276997090536, 0.010827599471613264, 0.011322361524464346, 0.01180210507896255, 0.012276543435307877, 0.012749323136224754, 0.013223371028562213, 0.013697833731701945, 0.01417556699620018, 0.014646124777465859, 0.015112116772851365, 0.015576452747370885, 0.01602507607909594, 0.016461827164870473, 0.016880126012974343]
CQcomp = [0.000226663607327854, 0.0002270862930229147, 0.0002292742856722754, 0.00023412703235791698 , 0.00024192624628054639 , 0.0002525855612031453, 0.00026638347417704255 , 0.00028314784456601373 , 0.00030299181501156373 , 0.0003259970210015136, 0.00035194661281707764 , 0.00038102864688744595 , 0.0004132249034847219, 0.00044859355432807347 , 0.0004873204055790553, 0.0005293656187218555, 0.0005753409000182888, 0.0006250099998058788, 0.0006788861946930185, 0.0007361096750412038, 0.0007970800153713466, 0.0008624036743669367, 0.0009315051772818803, 0.0010035766105979213, 0.0010791941808362153, 0.0011566643573792704, 0.001229236439467123, 0.0013007334425769355, 0.001372124993921022, 0.0014449961686871802, 0.0015197156782734364, 0.0015967388663224156, 0.0016761210460920718, 0.0017578748614666766, 0.0018409716992061841, 0.0019248522013432586, 0.0020103360819251357, 0.002096387027559033, 0.002182833604491109, 0.0022686470790128036]

for i = 1:nP
    @test isapprox(CT[i], CTcomp[i], atol=1e-4)
    @test isapprox(CQ[i], CQcomp[i], atol=1e-4)
end

end



@testset "correction methods" begin

# --- Mach ---

cl0 = 0.5
cd0 = 0.001
Mach = 0.6

cl, cd = mach_correction(PrandtlGlauert(), cl0, cd0, Mach)

@test cl == 0.625
@test cd == 0.001


# --- Re ---

Re0 = 1e6
sf = TurbulentSkinFriction(Re0)

Re = 2e6
cl, cd = re_correction(sf, cl0, cd0, Re)

@test cl == 0.5
@test isapprox(cd, 0.000870551, atol=1e-6)

cl, cd = re_correction(sf, cl0, cd0, 0.5e6)

@test cl == 0.5
@test isapprox(cd, 0.001148698, atol=1e-6)


# ----- Du Selig lift -----
D = 5.35
rR = 0.3
cr = 0.37
Omega = 158*pi/30
Vinf = 8.8

tsr = Omega*D/2/Vinf

# 2D data from Du-Selig paper
alphavec = pi/180*[0.0, 2.5155348977710403, 5.008413994160788, 7.49992519310865, 10.032131340952713, 12.497054033373846, 14.982409693833208, 17.44339968110891, 19.969450290464472, 22.590227797670874, 25.098125268058034]
cl2d = [0.0, 0.2519530298215895, 0.499873042018675, 0.6993694847728795, 0.7394671184527943, 0.7977304301609993, 0.7793208104222391, 0.6983663599821608, 0.5205579311691111, 0.49608168627557214, 0.47565213798095574]

# lift curve slope / angle of attack
m, alpha0 = CCBlade.linearliftcoeff(alphavec[1:3], cl2d[1:3])
du = DuSeligEggers(1.0, 1.0, 1.0, m, alpha0)

na = length(alphavec)
cl3d = zeros(na)
for i = 1:na
    cl3d[i], _ = rotation_correction(du, cl2d[i], cd, cr, rR, tsr, alphavec[i])
end

# data extracted from paper. match is not quite ok, but not exact.  not enough info from paper to determine what they did differently.
# alphavec3 = pi/180*[0.0, 2.5155348977710403, 5.009097942881734, 7.4334966735869, 10.014092193437797, 12.43661006516037, 14.970982050620755, 17.504385108726478, 20.01382146373576, 22.593334064778496, 25.10510724458434]
cl37 = [0.0, 0.2519530298215895, 0.5240848267401155, 0.74779989370297, 0.900881296424801, 1.058013955403694, 1.174782240701504, 1.2572504976439398, 1.2912974649725646, 1.406043541885448, 1.5228141070123276]

# comparison from airfoil prep excel worksheet
clexcel = [0.0002, 0.2516, 0.5000, 0.7285, 0.8937, 1.0624, 1.2008, 1.3120, 1.3871, 1.5309, 1.6698]

@test all(isapprox.(cl3d, clexcel, atol=5e-5))


rR = 0.55
cr = 0.16
for i = 1:na
    cl3d[i], _ = rotation_correction(du, cl2d[i], cd, cr, rR, tsr, alphavec[i])
end

# from original paper
# alphavec2 = pi/180*[0.0, 2.492879096389748, 5.008527985614281, 7.5228659731236664, 10.010615454106329, 12.52130571510401, 14.984974501536751, 17.537442630238804, 20.019606530000416, 22.618925146087175, 25.105135742447715]
cl16 = [0.0, 0.2499376609238726, 0.5039083394722486, 0.7114730973045302, 0.7778047240908117, 0.8562399634087439, 0.8701150031276412, 0.8274867591802311, 0.6960888107414154, 0.7119678202126802, 0.7238229313757207]

# from airfoil prep excel worksheet
clexcel = [0.0000, 0.2519, 0.4999, 0.7071, 0.7802, 0.8677, 0.8908, 0.8606, 0.7497, 0.7697, 0.7914]

@test all(isapprox.(cl3d, clexcel, atol=5e-5))


# -- second test case from Du-Selig paper

cr = 0.301
rR = 0.3
Omega = 72*pi/30
Vinf = 10.0
D = 10.0

tsr = Omega*D/2/Vinf

alphavec = pi/180*[0.05300353356890675, 2.5265017667844507, 4.905771495877504, 7.214369846878679, 9.899882214369846, 12.396937573616018, 14.917550058892814, 17.48527679623086, 19.958775029446404, 22.479387514723207, 25.0, 27.49705535924618, 29.970553592461723]
cl2d = [0.13100179507357756, 0.39492390225335816, 0.6916073880907927, 0.8987789384891554, 0.9814551925975084, 1.0139336800037035, 1.0660602506930839, 1.0002777477741605, 0.9432391562639841, 0.8490775173463776, 0.8204180618348846, 0.8092284269703383, 0.8002247699579783]
m, alpha0 = CCBlade.linearliftcoeff(alphavec[1:4], cl2d[1:4])
du = DuSeligEggers(1.0, 1.0, 1.0, m, alpha0)

na = length(alphavec)
cl3d = zeros(na)
for i = 1:na
    cl3d[i], _ = rotation_correction(du, cl2d[i], cd, cr, rR, tsr, alphavec[i])
end

# data from original paper
# alphavec2 = pi/180*[0.05300353356890497, 2.502944640753828, 4.905771495877504, 7.285041224970556, 9.8527679623086, 12.255594817432275, 14.823321554770313, 17.367491166077738, 19.93521790341578, 22.45583038869258, 24.95288574793875, 27.520612485276803, 29.970553592461716]
cl30 = [0.12663498284650343, 0.39274306787847024, 0.6872405758637183, 0.9118716599544292, 1.0753467989569032, 1.192988411745645, 1.3368128957262853, 1.4042207374717754, 1.4781762258192277, 1.525935984281534, 1.6239166550938433, 1.7218896106902037, 1.8176920188662753]

# data from airfoil prep excel sheet
clexcel = [0.1324, 0.3988, 0.6794, 0.9057, 1.0786, 1.2140, 1.3617, 1.4444, 1.5276, 1.5919, 1.6935, 1.8040, 1.9147]

@test all(isapprox.(cl3d, clexcel, atol=5e-5))



cr = 0.181
rR = 0.47
for i = 1:na
    cl3d[i], _ = rotation_correction(du, cl2d[i], cd, cr, rR, tsr, alphavec[i])
end

# data from original paper
# alphavec3 = pi/180*[0.05300353356890675, 2.5265017667844543, 4.929328621908127, 7.332155477031801, 9.852767962308597, 12.349823321554773, 14.893992932862197, 17.438162544169607, 19.958775029446404, 22.502944640753825, 24.95288574793875, 27.497055359246172, 29.99411071849234]
cl18 = [0.12663498284650343, 0.39929071448043185, 0.6916048163521438, 0.9140499225906671, 1.0054778033237153, 1.0641571640923564, 1.1424820364055326, 1.1116366030418527, 1.0851605536438969, 1.0390312774854573, 1.0365804105523586, 1.0581367239135695, 1.07969818075208]

# data from airofilprep excel worksheet
clexcel = [0.1316, 0.3966, 0.6862, 0.9018, 1.0243, 1.1021, 1.1964, 1.1960, 1.2008, 1.1764, 1.2052, 1.2476, 1.2914]

@test all(isapprox.(cl3d, clexcel, atol=5e-5))

cr = 0.113
rR = 0.80
for i = 1:na
    cl3d[i], _ = rotation_correction(du, cl2d[i], cd, cr, rR, tsr, alphavec[i])
end

# data from original paper
# alphavec4 = pi/180*[0.05300353356890675, 2.5265017667844525, 4.952885747938751, 7.285041224970556, 9.805653710247352, 12.396937573616018, 14.89399293286219, 17.438162544169607, 19.958775029446404, 22.502944640753825, 24.95288574793875, 27.473498233215548, 29.99411071849234]
cl11 = [0.12663498284650343, 0.4014741205939689, 0.6937856507270308, 0.9075048477273546, 0.9836488856656436, 1.0270341166849262, 1.0900802896806416, 1.0308505768409797, 0.9803570601941154, 0.901476692332619, 0.8815585764912233, 0.8790999943421753, 0.8744580060795908]

# data from airfoil prep excel worksheet
clexcel = [0.1311, 0.3953, 0.6904, 0.8995, 0.9911, 1.0338, 1.0954, 1.0444, 1.0013, 0.9229, 0.9071, 0.9080, 0.9109]

@test all(isapprox.(cl3d, clexcel, atol=5e-5))

end


@testset "derivatives" begin

D = 1.6
R = D/2.0
Rhub = 0.01
Rtip = D/2
r = range(R/10, stop=9/10*R, length=11)
chord = 0.1*ones(length(r))
proppitch = 1.0  # pitch distance in meters.
theta = atan.(proppitch./(2*pi*r))

function affunc(alpha, Re, M)

    cl = 6.2*alpha
    cd = 0.008 - 0.003*cl + 0.01*cl*cl

    return cl, cd
end

n = length(r)
airfoils = fill(affunc, n)

B = 2  # number of blades
turbine = false
pitch = 0.0
precone = 0.0

rho = 1.225
Vinf = 30.0
RPM = 2100
Omega = RPM * pi/30

function ccbladewrapper(x)

    # unpack
    nall = length(x)
    nvec = nall - 7
    n = nvec ÷ 3

    rp = x[1:n]
    chordp = x[n+1:2*n]
    thetap = x[2*n+1:3*n]
    Rhubp = x[3*n+1]
    Rtipp = x[3*n+2]
    pitchp = x[3*n+3]
    preconep = x[3*n+4]
    Vinfp = x[3*n+5]
    Omegap = x[3*n+6]
    rhop = x[3*n+7]

    rotor = Rotor(Rhubp, Rtipp, B; turbine=turbine, precone=preconep)
    sections = Section.(rp, chordp, thetap, airfoils)
    ops = simple_op.(Vinfp, Omegap, rp, rhop; pitch=pitchp)

    outputs = solve.(Ref(rotor), sections, ops)

    T, Q = thrusttorque(rotor, sections, outputs)

    return [T; Q]
end

import ForwardDiff

x = [r; chord; theta; Rhub; Rtip; pitch; precone; Vinf; Omega; rho]

J = ForwardDiff.jacobian(ccbladewrapper, x)

# using BenchmarkTools
# @btime ForwardDiff.jacobian($ccbladewrapper, $x)
# original: 584.041 μs (9910 allocations: 1.15 MiB)
# with ImplicitAD: 323.208 μs (11862 allocations: 747.50 KiB)

import FiniteDiff

J2 = FiniteDiff.finite_difference_jacobian(ccbladewrapper, x, Val{:central})

@test maximum(abs.(J - J2)) < 1e-6

end

@testset "type stability" begin


function ccbladewrapper(x)

    B = 2  # number of blades
    turbine = false

    function af(alpha, Re, M)

        cl = 6.2*alpha
        cd = 0.008 - 0.003*cl + 0.01*cl*cl

        return cl, cd
    end

    # unpack
    nall = length(x)
    nvec = nall - 7
    # n = nvec ÷ 3
    n = 11
    # println(n, "hi")

    rp = x[1:n]
    chordp = x[n+1:2*n]
    thetap = x[2*n+1:3*n]
    Rhubp = x[3*n+1]
    Rtipp = x[3*n+2]
    pitchp = x[3*n+3]
    preconep = x[3*n+4]
    Vinfp = x[3*n+5]
    Omegap = x[3*n+6]
    rhop = x[3*n+7]

    rotor = Rotor(Rhubp, Rtipp, B; turbine=turbine, precone=preconep)
    sections = Section.(rp, chordp, thetap, Ref(af))
    ops = simple_op.(Vinfp, Omegap, rp, rhop; pitch=pitchp)

    outputs = solve.(Ref(rotor), sections, ops)

    T, Q = thrusttorque(rotor, sections, outputs)

    return [T; Q]
end

D = 1.6
R = D/2.0
Rhub = 0.01
Rtip = D/2
r = range(R/10, stop=9/10*R, length=11)
chord = 0.1*ones(length(r))
proppitch = 1.0  # pitch distance in meters.
theta = atan.(proppitch./(2*pi*r))

pitch = 0.0
precone = 0.0

rho = 1.225
Vinf = 30.0
RPM = 2100
Omega = RPM * pi/30

xvec = [r; chord; theta; Rhub; Rtip; pitch; precone; Vinf; Omega; rho]

# @code_warntype ccbladewrapper(xvec)

function checkstability()
    try
        @inferred Vector{Float64} ccbladewrapper(xvec)
        return true
    catch err
        println(err)
        return false
    end
end

@test checkstability()


end