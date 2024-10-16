import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.io, dolfinx.fem as fem, dolfinx.mesh
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri  # noqa: I2023
import ufl
from python_utils import *
import seaborn as sns
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import assemble_scalar, form
from dolfinx import plot
import multiphenicsx.fem
import multiphenicsx.fem.petsc
import petsc4py.PETSc

sns.set()
Plot = False
Ghost = True
model_rank = 0
mesh_comm = MPI.COMM_WORLD

degV = 3
degPhi = degV + 1

#Définition de la fonction level-set
def phi_np(x, y):
    return ((x - 0.5)**2 + (y - 0.5)**2 - 0.16)*((x - 0.5)**2 + (y - 0.5)**2 - 0.0625)

def phi_expr_function(x):
    return phi_np(x[0], x[1])

#Définition d'une fonction pour vérifier la proximité entre deux points selon une tolerance
def near(a, b, tol=3e-16):
    """
    Check if two numbers 'a' and 'b' are close to each other within a tolerance 'tol'.
    """
    return np.abs(a - b) <= tol

#Définition des expressions pour le problème: solution exacte, fonction source et conditions de Dirichlet
class MyExpression_u_ex:
    def evalnp(self, x):
        return ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.16)*((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*np.exp(x[0])*np.sin(2*np.pi*x[1])
    def evalufl(sel, x):
        return ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.16)*((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*ufl.exp(x[0])*ufl.sin(2*ufl.pi*x[1])

class MyExpression_F:
    def eval(self, x):
        return -2*((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*np.exp(x[0])*np.sin(2*np.pi*x[1]) - 2*(x[0] - 0.5)*(2*(x[0] - 0.5)*np.exp(x[0])*np.sin(2*np.pi*x[1]) + ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*np.exp(x[0])*np.sin(2*np.pi*x[1])) - 2*(x[0] - 0.5)*(2*(x[0] - 0.5)*np.exp(x[0])*np.sin(2*np.pi*x[1]) + ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*np.exp(x[0])*np.sin(2*np.pi*x[1])) - ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.16)*(2*np.exp(x[0])*np.sin(2*np.pi*x[1]) + 4*(x[0] - 0.5)*np.exp(x[0])*np.sin(2*np.pi*x[1]) + ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*np.exp(x[0])*np.sin(2*np.pi*x[1])) - 2*((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*np.exp(x[0])*np.sin(2*np.pi*x[1]) - 2*(x[1] - 0.5)*(2*(x[1] - 0.5)*np.exp(x[0])*np.sin(2*np.pi*x[1]) + ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*2*np.pi*np.exp(x[0])*np.cos(2*np.pi*x[1])) - 2*(x[1] - 0.5)*(2*(x[1] - 0.5)*np.exp(x[0])*np.sin(2*np.pi*x[1]) + ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*2*np.pi*np.exp(x[0])*np.cos(2*np.pi*x[1])) - ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.16)*(2*np.exp(x[0])*np.sin(2*np.pi*x[1]) + 4*(x[1] - 0.5)*2*np.pi*np.exp(x[0])*np.cos(2*np.pi*x[1]) - ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.0625)*4*(np.pi**2)*np.exp(x[0])*np.sin(2*np.pi*x[1]))

class MyExpression_u_d:
    def eval(self, x):
        return 0.0 * x[0]
    
#Création des tableaux pour stocker les erreurs pour chaque méthode et la taille du maillage
errorsdirect_l2 = []
errorsdirect_h1 = []
errorsdual_l2 = []
errorsdual_h1 = []
sizes = []

#Boucle for pour computer direct et dual dirichlet pour maillages de taille différente
for size in [16, 32, 64, 128]:

    #########################################################################################
    #                             Création de chaque maillage                               #
    #########################################################################################

    mesh_macro = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        size - 1,
        size - 1,
    )
    num_cells_macro = (
        mesh_macro.topology.index_map(mesh_macro.topology.dim).size_local
        + mesh_macro.topology.index_map(mesh_macro.topology.dim).num_ghosts
    )

    mesh_macro.topology.create_connectivity(0, 2)
    v_to_c = mesh_macro.topology.connectivity(0, 2)
    vert = np.where(
        (
            phi_np(mesh_macro.geometry.x[:, 0], mesh_macro.geometry.x[:, 1])
            <= 0.0 + 3e-16
        )
    )[0]
    cells = [v_to_c.links(vert[i]) for i in range(len(vert))]
    cells = np.unique(cells)
    mesh = dolfinx.mesh.create_submesh(mesh_macro, mesh_macro.topology.dim, cells)[0]

    num_cells = (
        mesh.topology.index_map(mesh.topology.dim).size_local
        + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    )

    #########################################################################################
    #                     Création des Meshtags pour chaque partie du domain                #
    #########################################################################################

    #Création de toutes les connectivities dont on aura besoin dans le code
    mesh.topology.create_connectivity(2,2) #À utiliser pour les restrictions dans dual Dirichlet
    mesh.topology.create_connectivity(2,1)
    mesh.topology.create_connectivity(1,2)
    mesh.topology.create_connectivity(1,0)
    
    c_to_f = mesh.topology.connectivity(2,1) #dictionnaire des facets connectés à une celle
    f_to_c = mesh.topology.connectivity(1,2) #dictionnaire de cells connectés à une facet
    f_to_v = mesh.topology.connectivity(1,0) #dictionnaire de points connectés à une cell

    facets = np.unique([c_to_f.links(i)[:] for i in range(num_cells)]) #liste de toutes les facets du domain
    points = np.array([f_to_v.links(f)[:] for f in facets]) #liste de tous les points du domain

    selected_faces = np.where(
        phi_np(mesh.geometry.x[points[:, 0], 0], mesh.geometry.x[points[:, 0], 1])
        * phi_np(mesh.geometry.x[points[:, 1], 0], mesh.geometry.x[points[:, 1], 1])
        <= 0.0 + 3e-16
    )[0] #liste de toutes les cells du domain

    boundary_cells = np.unique(
        np.hstack(
            [f_to_c.links(selected_faces[i])[:] for i in range(len(selected_faces))]
        )
    ) #liste des cells qui font partie du bord du domain

    boundary_facets = np.unique(
        np.hstack(
            [c_to_f.links(boundary_cells[i])[:] for i in range(len(boundary_cells))]
        )
    ) #liste des facets qui font partie du bord du domain

    boundary_cells = np.unique(
        np.hstack(
            [f_to_c.links(boundary_facets[i])[:] for i in range(len(boundary_facets))]
        )
    ) #adjustement de la liste des boundary_cells

    boundary_facets, boundary_cells = np.unique(boundary_facets), np.unique(boundary_cells)

    sorted_facets = np.argsort(boundary_facets) #liste final des entities pour selectionner les boundary facets
    sorted_cells = np.argsort(boundary_cells) #liste final des entities pour selectionner les boundary cells

    #Création des meshtags for cells
    Dirichlet = 1
    values_dirichlet = Dirichlet * np.ones(len(boundary_cells), dtype=np.intc)
    values_cells = np.hstack([values_dirichlet])
    entities_cells = np.hstack([boundary_cells])
    sorted_cells = np.argsort(entities_cells)

    subdomains_cell = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        entities_cells[sorted_cells],
        values_cells[sorted_cells],
    )  #meshtags pour les cells

    #Création des meshtags for facets
    values_dirichlet = Dirichlet * np.ones(len(boundary_facets), dtype=np.intc)
    values_facets = np.hstack([values_dirichlet])
    entities_facets = np.hstack([boundary_facets])
    sorted_facets = np.argsort(entities_facets)

    subdomains_facet = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        entities_facets[sorted_facets],
        values_facets[sorted_facets],
    )  #facets du bord du domain

    #Pour plotter les maillages du domain
    if Plot:
        mesh_macro.topology.create_connectivity(1,0)
        mesh_macro.topology.create_connectivity(2,1)
        mesh.topology.create_connectivity(1,0)
        f_to_v = mesh.topology.connectivity(1,0)
        c_to_f_macro = mesh_macro.topology.connectivity(2,1)
        f_to_v_macro = mesh_macro.topology.connectivity(1, 0)
        facets_macro = np.unique([c_to_f_macro.links(i)[:] for i in range(num_cells_macro)])

        indices = subdomains_facet.find(Dirichlet)
        faces = []

        plt.figure(figsize=(12,12))

        for f in facets: #plot maillage du subdomain
            vert = f_to_v.links(f)
            p1, p2 = mesh.geometry.x[vert]
            face = np.array([p1, p2])
            plt.plot(face[:, 0], face[:, 1], "-+", color="yellow")

        for f in facets_macro: #plot maillage du domain plus génñeral
            vert = f_to_v_macro.links(f)
            p1, p2 = mesh_macro.geometry.x[vert]
            face = np.array([p1, p2])
            plt.plot(face[:, 0], face[:, 1], "-+", color="black", alpha=0.2)
        
        #plot du domain exacte
        angle = np.linspace(0, 2 * np.pi, 1000)
        radius1 = 0.4
        radius2 = 0.25
        x = 0.5 + radius1*np.cos(angle)
        y = 0.5 + radius1*np.sin(angle)
        xx = 0.5 + radius2*np.cos(angle)
        yy = 0.5 + radius2*np.sin(angle)
        plt.plot(x, y, "-", color="purple")
        plt.plot(xx, yy, "-", color="purple")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        plt.figure(figsize=(12, 12))

        for f in boundary_facets: #plot des bords du maillage avec le subdomain
            if f in indices:
                vert = f_to_v.links(f)
                p1, p2 = mesh.geometry.x[vert]
                face = np.array([p1, p2])
                plt.plot(face[:, 0], face[:, 1], "-+", color="b")
        
        plt.plot(x, y, "-", color="r")
        plt.plot(xx, yy, "-", color="r")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.show()
    
    #########################################################################################
    #                              Solution pour direct Dirichlet                           #
    #########################################################################################

    #Création des deux espaces de fonctions
    V = fem.functionspace(mesh, ("CG", degV))
    V_phi = fem.functionspace(mesh, ("CG", degPhi))

    x = ufl.SpatialCoordinate(mesh) #création des coordonnés dans le subdomain

    #Création et interpolation des fonctions à utiliser
    u_ex = MyExpression_u_ex()
    f_expr = MyExpression_F()
    u_D_expr = MyExpression_u_d()

    f = dolfinx.fem.Function(V)
    u = dolfinx.fem.Function(V)
    u_D = dolfinx.fem.Function(V)
    phi = dolfinx.fem.Function(V_phi)

    phi.interpolate(phi_expr_function)
    u.interpolate(u_ex.evalnp)
    f.interpolate(f_expr.eval)
    u_D.interpolate(u_D_expr.eval)

    #Définition de quelques valeurs pour le problème
    h = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)
    sigma = 20.0 #paramètre de la Ghost penalty

    #Definition des mesures pour les intégrales
    dx = ufl.Measure("dx", mesh, subdomain_data=subdomains_cell)
    ds = ufl.Measure("ds", mesh)
    dS = ufl.Measure("dS", mesh, subdomain_data=subdomains_facet)

    #Définition et solution du problème
    w, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    Ghost_penalty_a = sigma*ufl.avg(h)*ufl.dot(ufl.jump(ufl.grad(phi*w),n),ufl.jump(ufl.grad(phi*v),n))*dS(1) + sigma*h**2*ufl.inner(ufl.div(ufl.grad(phi*w)),ufl.div(ufl.grad(phi*v)))*dx(1)
    Ghost_penalty_l = - sigma*h**2*ufl.inner(f,ufl.div(ufl.grad(phi*v)))*dx(1)

    if Ghost:
        a = ufl.inner(ufl.grad(phi*w),ufl.grad(phi*v))*dx - ufl.dot(ufl.inner(ufl.grad(phi*w),n),phi*v)*ds + Ghost_penalty_a
        l = f*v*phi*dx + Ghost_penalty_l
    else:
        a = ufl.inner(ufl.grad(phi*w),ufl.grad(phi*v))*dx - ufl.dot(ufl.inner(ufl.grad(phi*w),n),phi*v)*ds
        l = f*v*phi*dx
    
    dirDir_Poisson = LinearProblem(a, l, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) #Création du problème de Poisson
    wh = dirDir_Poisson.solve() #Résolution du problème de Poisson
    uh = phi*wh + u_D #Obtention de notre solution avec direct Dirichlet
    
    #########################################################################################
    #                         Calcul des erreurs pour direct Dirichlet                      #
    #########################################################################################

    diff = uh - u #difference entre la solution exacte et la solution approchée

    #Calcul erreur relative L2
    errorL2 = form(ufl.inner(diff, diff)*dx)
    errorL2div = form(ufl.inner(u,u)*dx)
    errorL2_local = np.sqrt((fem.assemble_scalar(errorL2)))/np.sqrt(fem.assemble_scalar(errorL2div))
    errorL2_global = mesh.comm.allreduce(errorL2_local, op=MPI.SUM)

    errorsdirect_l2.append(errorL2_global)

    #Calcul erreur relative H1
    errorH1 = form(ufl.inner(ufl.grad(diff),ufl.grad(diff))*dx)
    errorH1div = form(ufl.inner(ufl.grad(u),ufl.grad(u))*dx)
    errorH1_local = np.sqrt(fem.assemble_scalar(errorH1))/np.sqrt(fem.assemble_scalar(errorH1div))
    errorH1_global = mesh.comm.allreduce(errorH1_local, op=MPI.SUM)

    errorsdirect_h1.append(errorH1_global)

    #Calcul taille du maillage
    hh = 0.15/size 
    sizes.append(hh)

    #########################################################################################
    #                                     Solution pour dual Dirichlet                      #
    #########################################################################################

    #Création d'un nouveau espace des fonction pour les deux incognites
    W = V.clone() #C'est le même que V au niveau théorique
    W_phi = V_phi.clone()

    #Créations des fonctions test et à trouver
    (uu, p) = (ufl.TrialFunction(V), ufl.TrialFunction(W))
    (vv, q) = (ufl.TestFunction(V), ufl.TestFunction(W))
    phi_dual = dolfinx.fem.Function(W_phi)
    phi_dual.interpolate(phi_expr_function)


    #Définition des formes variationnelles
    gamma = 20.0 #paramètre attaché à l'implementation de dual Dirichlet

    Ghost_penalty_A = sigma*ufl.avg(h)*ufl.dot(ufl.jump(ufl.grad(uu),n),ufl.jump(ufl.grad(vv),n))*dS(1) + sigma*h**2*ufl.inner(ufl.div(ufl.grad(uu)),ufl.div(ufl.grad(vv)))*dx(1)
    Ghost_penalty_L = - sigma*h**2*ufl.inner(f,ufl.div(ufl.grad(vv)))*dx(1)
    

    #Pour la matrice A
    auuvv = ufl.inner(ufl.grad(uu),ufl.grad(vv))*dx - ufl.dot(ufl.inner(ufl.grad(uu),n),vv)*ds - gamma*(h**(-2))*ufl.inner(uu,vv)*dx(1) + Ghost_penalty_A
    auuq = -gamma*(h**(-3))*ufl.inner(uu, phi_dual*q)*dx(1)
    apvv = -gamma*(h**(-3))*ufl.inner(phi_dual*p,vv)*dx(1)
    apq = gamma*(h**(-4))*ufl.inner(phi_dual*p,phi_dual*q)*dx(1)

    #Pour le vecteur L
    lvv = f*vv*dx + gamma*(h**(-2))*ufl.inner(u_D, vv)*dx(1) + Ghost_penalty_L
    lq = -gamma*(h**(-3))*ufl.dot(u_D,phi_dual*q)*dx(1)

    #Définition de la matrice A et le vecteur L
    A = [[auuvv, apvv], [auuq, apq]]
    L = [lvv, lq]

    A_cpp = dolfinx.fem.form(A)
    L_cpp = dolfinx.fem.form(L)

    #Définition des restrictions pour la méthode
    dofs_V = np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
    dofs_W = dolfinx.fem.locate_dofs_topological(W, mesh.topology.dim, entities_cells[sorted_cells])
    restriction_V = multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)
    restriction_W = multiphenicsx.fem.DofMapRestriction(W.dofmap, dofs_W)
    restrictions = [restriction_V, restriction_W]

    #Assemblage du système linéaire
    AA = multiphenicsx.fem.petsc.assemble_matrix_block(A_cpp, bcs=[], restriction=(restrictions, restrictions))
    AA.assemble()

    LL = multiphenicsx.fem.petsc.assemble_vector_block(L_cpp, A_cpp, bcs=[], restriction=restrictions)

    #Résolution du système linéaire
    up = multiphenicsx.fem.petsc.create_vector_block(L_cpp, restriction=restrictions)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(AA)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(LL, up)
    up.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()

    #Division de la solution
    (uh_dual, ph) = (dolfinx.fem.Function(V), dolfinx.fem.Function(W))
    with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(up, [V.dofmap, W.dofmap], restrictions) as up_wrapper:
        for up_wrapper_local, component in zip(up_wrapper, (uh_dual, ph)):
            with component.x.petsc_vec.localForm() as component_local:
                component_local[:] = up_wrapper_local
    up.destroy()

    #########################################################################################
    #                      Calcul des erreurs pour dual Dirichlet                           #
    #########################################################################################

    diff = (uh_dual + phi*ph) - u #difference entre la solution exacte et la solution approchée

    #Calcul erreur relative L2
    errorL2 = form(ufl.inner(diff, diff)*dx)
    errorL2div = form(ufl.inner(u,u)*dx)
    errorL2_local = np.sqrt((fem.assemble_scalar(errorL2)))/np.sqrt(fem.assemble_scalar(errorL2div))
    errorL2_global = mesh.comm.allreduce(errorL2_local, op=MPI.SUM)

    errorsdual_l2.append(errorL2_global)

    #Calcul erreur relative H1
    errorH1 = form(ufl.inner(ufl.grad(diff),ufl.grad(diff))*dx)
    errorH1div = form(ufl.inner(ufl.grad(u),ufl.grad(u))*dx)
    errorH1_local = np.sqrt(fem.assemble_scalar(errorH1))/np.sqrt(fem.assemble_scalar(errorH1div))
    errorH1_global = mesh.comm.allreduce(errorH1_local, op=MPI.SUM)

    errorsdual_h1.append(errorH1_global)


#########################################################################################
#                      Plot pour la convergence des erreurs                             #
#########################################################################################

sizesnp = np.array(sizes)

plt.figure(figsize=(12,12))
plt.loglog(sizes, errorsdirect_l2, color="blue", label="Erreur L2")
plt.loglog(sizes, errorsdirect_h1, color="red", label = "Erreur H1")
plt.loglog(sizesnp, sizesnp**3, "-.", color="black", label = "h³")
plt.loglog(sizesnp, sizesnp**4, "--", color="black", label="h⁴")
plt.legend()
plt.title("Erreurs d'approximation pour direct Dirichlet")
plt.show()

pL2 = np.log((errorsdirect_l2[-1]/errorsdirect_l2[-2]))/np.log((sizes[-1]/sizes[-2]))
pH1 = np.log((errorsdirect_h1[-1]/errorsdirect_h1[-2]))/np.log((sizes[-1]/sizes[-2]))

print("Pente L2 pour direct Dirichlet: ", pL2)
print("Pente H1 pour direct Dirichlet: ", pH1)

plt.figure(figsize=(12,12))
plt.loglog(sizes, errorsdual_l2, color="blue", label="Erreur L2")
plt.loglog(sizes, errorsdual_h1, color="red", label = "Erreur H1")
plt.loglog(sizesnp, sizesnp**3, "-.", color="black", label = "h³")
plt.loglog(sizesnp, sizesnp**4, "--", color="black", label="h⁴")
plt.legend()
plt.title("Erreurs d'approximation pour dual Dirichlet")
plt.show()

pdL2 = np.log((errorsdual_l2[-1]/errorsdual_l2[-2]))/np.log((sizes[-1]/sizes[-2]))
pdH1 = np.log((errorsdual_h1[-1]/errorsdual_h1[-2]))/np.log((sizes[-1]/sizes[-2]))

print("Pente L2 pour dual Dirichlet: ", pdL2)
print("Pente H1 pour dual Dirichlet: ", pdH1)

