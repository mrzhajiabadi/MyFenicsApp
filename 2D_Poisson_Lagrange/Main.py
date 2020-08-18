import json
from dolfin import *
from multiphenics import *
parameters["ghost_mode"] = "shared_facet" # required by dS

# Mesh
mesh = Mesh("data/mesh.xml")
subdomains = MeshFunction("size_t", mesh, "data/mesh_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/mesh_facet_region.xml")

upper_restriction = MeshRestriction(mesh, "data/mesh_upper_restriction.rtc.xml")
lower_restriction = MeshRestriction(mesh, "data/mesh_lower_restriction.rtc.xml")
interface_restriction = MeshRestriction(mesh, "data/mesh_interface_restriction.rtc.xml")




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



#upper_2dim=(subdomains.array()==id_upper_domain)
#lower_2dim=(subdomains.array()==id_lower_domain)
#interface_1dim=(boundaries.array()==id_interface)

#Material
f=open("data/poroelastic_properties.json")
material_data = json.load(f)
f.close()
material_1= material_data["GulfMexicoShale"]
material_2= material_data["CoarseSand"]
#material_1['Permeability']=material_2['Permeability']
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
        
V = FunctionSpace(mesh, "CG", 1)
W = BlockFunctionSpace([V, V, V], restrict=[upper_restriction, lower_restriction, interface_restriction])
h1h2l = BlockTrialFunction(W)
dh1dh2dl = BlockTestFunction(W)
(h1, h2, l) = block_split(h1h2l)
(dh1, dh2, dl) = block_split(dh1dh2dl)

dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)


dx1 , dx2 = dx(id_upper_domain) , dx(id_lower_domain)
dS=dS(id_interface)

A11 = inner(k*grad(h1), grad(dh1))*dx1
A22 = inner(k*grad(h2), grad(dh2))*dx2
A31 =  h1("+")*dl("+")*dS
A32 =  -h2("+")*dl("+")*dS
A13=   l("+")*dh1("+")*dS
A23=   -l("+")*dh2("+")*dS   

a=[[A11,0,A13],
   [0,A22,A23],
   [A31,A32,0]]

A = block_assemble(a)
      
F1 = Constant(0.)*dh1*dx1
F2 = Constant(0.)*dh2*dx2
f=[F1,F2,0.0]
F = block_assemble(f)

bc1 = DirichletBC(W.sub(0), Constant(1.), boundaries, 15)
bc2 = DirichletBC(W.sub(1), Constant(4.), boundaries, 12)
bcs = BlockDirichletBC([bc2,bc1])
        
bcs.apply(A)
print("Assigning Dirichlet boundary conditions to independent vector F.")
bcs.apply(F)

print("Defining solution vector H.")
H = BlockFunction(W)
print("Solving linear system AH=F.")
block_solve(A, H.block_vector(), F)        

h1, h2, l = block_split(H)
      

h1.rename("h1","h1")
h2.rename("h2","h2")
infile=XDMFFile("Results.xdmf") 
infile.parameters["rewrite_function_mesh"] = False
infile.parameters["functions_share_mesh"] = True
infile.write(h1,0.0)
infile.write(h2,0.0)
infile.close()
	









        













    
    
    
    
    
    









