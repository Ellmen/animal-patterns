from ngsolve import *
# from ngsolve import Draw
from netgen.geom2d import SplineGeometry


def make_geometry():
    geometry = SplineGeometry()
    from squid_pts import pnts
    pnums = [geometry.AppendPoint(*p) for p in pnts]
    for i in range(len(pnts)):
        p1 = pnums[i]
        p2 = pnums[i+1] if i+1 < len(pnts) else pnums[0]
        geometry.Append(
            ['line', p1, p2],
            bc=1,
            leftdomain=1,
            rightdomain=0
        )
    return geometry


mesh = Mesh(make_geometry().GenerateMesh (maxh=5))

fes = H1(mesh, order=3, dirichlet=[1], autoupdate=True)
u = fes.TrialFunction()
v = fes.TestFunction()

# one heat conductivity coefficient per sub-domain (we currently only have 1)
lam = CoefficientFunction([1])
a = BilinearForm(fes, symmetric=False)
a += lam*grad(u)*grad(v)*dx


# heat-source in sub-domain
f = LinearForm(fes)
f += CoefficientFunction([1])*v*dx

c = MultiGridPreconditioner(a, inverse = "sparsecholesky")

gfu = GridFunction(fes, autoupdate=True)
Draw (gfu)

# finite element space and gridfunction to represent
# the heatflux:
space_flux = HDiv(mesh, order=2, autoupdate=True)
gf_flux = GridFunction(space_flux, "flux", autoupdate=True)

def SolveBVP():
    a.Assemble()
    f.Assemble()
    inv = CGSolver(a.mat, c.mat)
    gfu.vec.data = inv * f.vec
    Redraw (blocking=True)



l = []

# def CalcError():
#     flux = lam * grad(gfu)
#     # interpolate finite element flux into H(div) space:
#     gf_flux.Set (flux)

#     # Gradient-recovery error estimator
#     err = 1/lam*(flux-gf_flux)*(flux-gf_flux)
#     elerr = Integrate (err, mesh, VOL, element_wise=True)

#     maxerr = max(elerr)
#     l.append ( (fes.ndof, sqrt(sum(elerr)) ))
#     print ("maxerr = ", maxerr)

#     for el in mesh.Elements():
#         mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25*maxerr)


# with TaskManager():
#     while fes.ndof < 100000:  
#         SolveBVP()
#         CalcError()
#         mesh.Refine()
    
SolveBVP()

# import matplotlib.pyplot as plt

# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel("ndof")
# plt.ylabel("H1 error-estimate")
# ndof,err = zip(*l)
# plt.plot(ndof,err, "-*")

# plt.ion()
# plt.show()

# input("<press enter to quit>")

