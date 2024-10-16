from mpi4py import MPI
import pyvista
import numpy as np
import matplotlib.pyplot as plt
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

def u1D_ex(x):
    return x[0]**2 - x[0]

errorsL2_1D = np.empty((2, 3))
errorsH1_1D = np.empty((2, 3))
for i, dim in enumerate([1, 2]):
    dimV1 = dim
    errL2_1D = []
    errH1_1D = []
    taille_1D = []

    plt.figure(1, figsize=(12,12))
    X = np.linspace(0,1, 100)
    

    for N in [3, 11, 21]:
        mesh1D = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=N)

        V1D = fem.functionspace(mesh1D, ("Lagrange", dimV1))

        #Résolution problème 1D
        u, v = ufl.TrialFunction(V1D), ufl.TestFunction(V1D)
        x = ufl.SpatialCoordinate(mesh1D)
        f1 = x[0]**2 - x[0] - 2
        u1D_sol = fem.Function(V1D)
        u1D_sol.interpolate(u1D_ex)
        alpha1, beta1 = 1, 1

        a1 = ufl.Dx(u,0)*ufl.Dx(v,0)*dx + beta1*u*v*dx
        l1 = f1*v*dx + alpha1*v*ds
        
        problem1D = LinearProblem(a1, l1, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh_1D = problem1D.solve()
        if dim==1:
            cells, types, x = plot.vtk_mesh(V1D)
            plt.plot ( x[:,0], uh_1D.x.array.real, label= f"Sol. app. pour N = {N}")
        #Calcul erreur L2 en 1D
        comm = uh_1D.function_space.mesh.comm
        errorL2  = fem.form((uh_1D-u1D_sol)**2*dx)
        EL2 = np.sqrt(comm.allreduce(fem.assemble_scalar(errorL2), MPI.SUM))
        errL2_1D.append(EL2)

        #Calcul erreur H1 en 1D
        errorH1 = fem.form(ufl.Dx(uh_1D - u1D_sol, 0)**2*dx + (uh_1D-u1D_sol)**2*dx)
        EH1 = np.sqrt(comm.allreduce(fem.assemble_scalar(errorH1), MPI.SUM))
        errH1_1D.append(EH1)
        #Calcule taille maillage en 1D
        taille_1D.append(1/N)

    if dim==1:
        plt.plot(X, X**2-X, "--", label = "Sol. Ex.", color="black")
        plt.title("Comparaison de la solution approchée pour maillages de diferente taille")
        plt.legend()
        plt.show()
    errorsL2_1D[i, :] = errL2_1D
    errorsH1_1D[i, :] = errH1_1D

taille_1D_np = np.array(taille_1D)
plt.figure()
plt.loglog(taille_1D_np, errorsL2_1D[0,:], label="L² dim 1")
plt.loglog(taille_1D_np, errorsL2_1D[1,:], label="L² dim 2")
plt.legend()
plt.title("Convergence des erreurs poour la norme L²")
plt.show
plt.figure()
plt.loglog(taille_1D_np, errorsH1_1D[0,:], label="H¹ dim 1")
plt.loglog(taille_1D_np, errorsH1_1D[1,:], label="H¹ dim 2")
plt.legend()
plt.title("Convergence des erreurs pour la norme H¹")
plt.show()



    



def u2D_ex(x):
    return x[0]**2 + x[1]**2

errorsH1_2D = np.empty((2,3))
errorsL2_2D = np.empty((2,3))

for i,dim in enumerate([1, 2]):
    dimV2 = dim
    errL2_2D = []
    errH1_2D = []
    taille_2D = []

    for j,N in enumerate([3, 11, 21]):
        mesh2D = mesh.create_unit_square(MPI.COMM_WORLD, nx=N, ny=N)
        V2D = fem.functionspace(mesh2D, ("Lagrange", dimV2))

        #Résolution du problème 2D
        x = ufl.SpatialCoordinate(mesh2D)
        u2D_exa = u2D_ex(x)
        n = ufl.FacetNormal(mesh2D)
        g = ufl.dot(n, grad(u2D_exa))
        u2D_sol = fem.Function(V2D)
        u2D_sol.interpolate(u2D_ex)
        f2 = x[0]**2 + x[1]**2 - 4
        alpha2 = 1

        uu, vv = ufl.TrialFunction(V2D), ufl.TestFunction(V2D)
        a2 = inner(grad(uu), grad(vv))*dx + uu*vv*dx
        l2 = f2*vv*dx + g*vv*ds

        problem2D = LinearProblem(a2, l2, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh_2D = problem2D.solve()

        #Calcul erreur L2 en 2D
        comm = uh_2D.function_space.mesh.comm
        errorL2  = fem.form((uh_2D-u2D_sol)**2*dx)
        EL2 = np.sqrt(comm.allreduce(fem.assemble_scalar(errorL2), MPI.SUM))
        errL2_2D.append(EL2)

        #Calcul erreur H1 en 2D
        errorH1 = fem.form(grad(uh_2D-u2D_sol)**2*dx + (uh_2D-u2D_sol)**2*dx)
        EH1 = np.sqrt(comm.allreduce(fem.assemble_scalar(errorH1), MPI.SUM))
        errH1_2D.append(EH1)

        #Calcule taille maillage en 2D
        taille_2D.append(1/N)

        if dim==1:
            pyvista_cells, cell_types, geometry = plot.vtk_mesh(V2D)

            grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
            grid.point_data["u"] = uh_2D.x.array
            grid.set_active_scalars("u")

            plotter = pyvista.Plotter()
            plotter.add_text(f"Sol. ap. pour maille de {N}x{N} points", position="upper_edge", font_size=14, color="black")
            plotter.add_mesh(grid, show_edges=True)
            plotter.view_xy()
            plotter.save_graphic(f"sol_app(dim1)_2D[{j}](Poly).eps")
            plotter.show()
            
            
            
            

    errorsL2_2D[i, :] = errL2_2D
    errorsH1_2D[i,:] = errH1_2D

    
x, y = np.linspace(0, 1, 200), np.linspace(0, 1, 200)
X,Y = np.meshgrid(x, y)
plt.figure()
plt.pcolormesh(X, Y, X**2+Y**2)
plt.colorbar()
plt.title("Solution exacte du problème en 2D")
plt.show()



taille_2D_np = np.array(taille_2D)

plt.figure()
plt.loglog(taille_2D_np, errorsL2_2D[0,:], label="L² dim 1")
plt.loglog(taille_2D_np, errorsL2_2D[1,:], label="L² dim 2")
plt.legend()
plt.title("Convergence des erreurs poour la norme L²")
plt.show
plt.figure()
plt.loglog(taille_2D_np, errorsH1_2D[0,:], label="H¹ dim 1")
plt.loglog(taille_2D_np, errorsH1_2D[1,:], label="H¹ dim 2")
plt.legend()
plt.title("Convergence des erreurs pour la norme H¹")
plt.show()