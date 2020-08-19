import json
from dolfin import *
from math import *
#from multiphenics import *

# Mesh
mesh = Mesh("data/mesh.xml")
subdomains = MeshFunction("size_t", mesh, "data/mesh_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/mesh_facet_region.xml")

#interface_restriction = MeshRestriction(mesh, "data/mesh_interface_restriction.rtc.xml")

# Mesh_id 
id_upper_domain=20
id_lower_domain=19
id_bottom_boundary=12
id_top_boundary=15
id_right_lower_boundary=13
id_right_upper_boundary=14
id_left_lower_boundary=17
id_left_upper_boundary=16
id_interface=18

#Material
f=open("data/poroelastic_properties.json")
material_data = json.load(f)
f.close()
material_1= material_data["GulfMexicoShale"]
material_2= material_data["CoarseSand"]
material_1['Permeability']=material_2['Permeability']

#Assign material 
subdomain_materials={id_upper_domain:material_1,id_lower_domain:material_2}

class Subdomain_Property(UserExpression):
    def __init__(self, subdomains,subdomain_materials,property_name,**kwargs):
        self.subdomain_materials=subdomain_materials
        self.subdomains=subdomains.array()
        self.property_name=property_name
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        values[0] =subdomain_materials[subdomains[ufc_cell.index]][self.property_name]

k=Subdomain_Property(subdomains,subdomain_materials,'Permeability',degree=0) 
kf=1.0 
        
print("Creating function spaces.")
V = FunctionSpace(mesh, "CG", 1)
print("Defining trial functions.")
h = TrialFunction(V)
print("Defining test functions.")
dh = TestFunction(V)

print("Defining subdomains discretized volumes.")
dx = Measure("dx")(subdomain_data=subdomains)
print("Defining boundaries discretized areas.")
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(id_interface)
n = FacetNormal(mesh)

print("Defining bilinear form a(h,dh).")

gradn_h=inner(grad(h("+")),n("+"))*n("+")
gradn_dh=inner(grad(dh("+")),n("+"))*n("+")

gradt_h=grad(h("+"))-gradn_h
gradt_dh=grad(dh("+"))-gradn_dh


b = kf*inner(gradt_h,gradt_dh)*dS
a = k*inner(grad(h),grad(dh))*dx+b

print("Defining linear form L(v).")
L = Constant(0.)*dh*dx

print("Assigning Dirichlet boundary conditions.")
bc1 = DirichletBC(V, Constant(4.), boundaries, 12)
bc2 = DirichletBC(V, Constant(1.), boundaries, 15)
bcs = [bc1, bc2]

print("Solving variational problem a(h,dh)=L(dh).")
H = Function(V)
problem = LinearVariationalProblem(a, L, H, bcs)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "mumps"
solver.solve()

    
import matplotlib.pyplot as plt
plot(H,title="Head")
plt.show()

H.rename("head","head")
XDMFFile("Results.xdmf").write(H)

   
        

        

        
        
        
        













    
    
    
    
    
    









