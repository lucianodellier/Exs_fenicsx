from dolfinx import fem, io, mesh, cpp
import ufl
from ufl import dx, grad, inner
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

'''
import gmsh
gmsh.initialize()

membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
status = gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin",0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.05)
gmsh.model.mesh.generate(gdim)

if MPI.COMM_WORLD.rank == 0:
    # Get mesh geometry
    geometry_data = io.extract_gmsh_geometry(gmsh.model)
    # Get mesh topology for each element
    topology_data = io.extract_gmsh_topology_and_markers(gmsh.model)
    # Extract the cell type and number of nodes per cell and broadcast
    # it to the other processors 
    gmsh_cell_type = list(topology_data.keys())[0]    
    properties = gmsh.model.mesh.getElementProperties(gmsh_cell_type)
    name, dim, order, num_nodes, local_coords, _ = properties
    cells = topology_data[gmsh_cell_type]["topology"]
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([gmsh_cell_type, num_nodes], root=0)

else:        
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    cells, geometry_data = np.empty([0, num_nodes]), np.empty([0, gdim]) 
    
# Permute topology data from MSH-ordering to dolfinx-ordering
ufl_domain = io.ufl_mesh_from_gmsh(cell_id, gdim)
gmsh_cell_perm = io.cell_perm_gmsh(cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]

# Create distributed mesh
msh = mesh.create_mesh(MPI.COMM_WORLD, cells, geometry_data[:, :gdim], ufl_domain)       
'''



msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(16,16),
                            cell_type=mesh.CellType.triangle)                 #cell_type=mesh.CellType.quadrilateral


Vd = fem.FunctionSpace(msh, ("Lagrange", 1))

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
facets = np.flatnonzero(mesh.compute_boundary_facets(msh.topology))
dofs = fem.locate_dofs_topological(Vd, fdim, facets)
u_boundary = fem.Constant(msh, ScalarType(0.0))
bc = fem.dirichletbc(u_boundary, dofs, Vd)

u  = ufl.TrialFunction(Vd)
v  = ufl.TestFunction(Vd)
x  = ufl.SpatialCoordinate(msh)
f  = fem.Constant(msh, ScalarType(1.0))
##mu = fem.Constant(msh, ScalarType(1.0))
mu = 0.01 + ufl.exp(-100 * ((x[0]-0.5)**2 + (x[1] - 0.5)**2))
a  = inner(mu * grad(u), grad(v)) * dx
L  = inner(f, v) * dx

one = fem.Constant(msh, ScalarType(1))
area = fem.form(one * dx)
print(fem.assemble_scalar(area))

opts={"ksp_type": "preonly", "pc_type": "lu"}
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts)
uh = problem.solve()
uh.name = "u"

with io.XDMFFile(msh.comm, "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
