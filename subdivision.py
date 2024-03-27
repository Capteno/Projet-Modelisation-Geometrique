import numpy as np
import meshio
import pyvista as pv

def load_mesh(path_to_model):
    maillage = meshio.read(path_to_model)

    sommets = maillage.points
    faces = maillage.cells_dict['quad']
    aretes = set([tuple(sorted([faces[i][j], faces[i][(j+1)%len(faces[i])]])) for i in range(len(faces)) for j in range(len(faces[i]))])

    dico = {}
    dico["faces"] = faces
    dico["aretes"] = aretes
    dico["sommets"] = sommets
    dico['nsommets'] = len(sommets)
    dico['sharp_sommets'] = []
    dico['sharp_aretes'] = []

    return maillage, dico

def get_faces_from_edge(dico, edge):
    faces = []
    for face in dico["faces"]:
        if set(edge).issubset(set(face)):
            faces.append(face)
    return faces

def affichage(dico, dico_init, titre=""):
    plotter = pv.Plotter()

    # ============ Affichage du modele 3D ============
    faces = np.hstack((np.full((len(dico["faces"]), 1), len(dico["faces"][0])), dico["faces"]))
    mesh = pv.PolyData(dico["sommets"], faces)
    plotter.add_mesh(mesh, color="blue", show_edges=True, edge_color="black", line_width=1)

    # ============ Affichage des aretes sharp ============
    linessharp = [[2]+list(edge) for edge in dico_init["sharp_aretes"]]
    if len(linessharp) != 0:
        linessharp = [e for l in linessharp for e in l]
        mesh_sharp = pv.PolyData(dico_init["sommets"], lines=np.array(linessharp))
        plotter.add_mesh(mesh_sharp, color="red", line_width=2)

    # ============ Affichage des aretes non-sharp ============
    linesnonsharp = [[2]+list(edge) for edge in dico_init["aretes"] if edge not in dico_init["sharp_aretes"]]
    if len(linesnonsharp) != 0:
        linesnonsharp = [e for l in linesnonsharp for e in l]
        mesh_nonsharp = pv.PolyData(dico_init["sommets"], lines=np.array(linesnonsharp))
        plotter.add_mesh(mesh_nonsharp, color="yellow", line_width=2)

    plotter.add_title(titre, font_size=10)

    plotter.show()

def catmull_clark(dico):
    new_faces = []
    new_vertices = []
    face_points = []
    edge_points = []
    sharp_sommets = []
    sharp_edgs = []

    face_to_index = {tuple(face): i for i, face in enumerate(dico["faces"])}
    edge_to_index = {edge: i for i, edge in enumerate(dico["aretes"])}

    face_points = [np.mean(dico["sommets"][face], axis=0) for face in dico["faces"]]
    new_vertices.extend(face_points)
    
    edge_points = []
    for edge in dico["aretes"]:
        if edge in dico["sharp_aretes"]:
            edge_point = np.mean([dico["sommets"][v] for v in edge], axis=0)

        else:
            adjacent_faces = get_faces_from_edge(dico, edge)
            edge_point = np.mean([dico["sommets"][v] for v in edge] + [face_points[face_to_index[tuple(face)]] for face in adjacent_faces], axis=0)
        edge_points.append(edge_point)
    new_vertices.extend(edge_points)

    for v in range(dico['nsommets']):
        sharp_edges = [edge for edge in dico["sharp_aretes"] if v in edge]
        for kk in range(len(sharp_edgs)):
            if sharp_edgs[kk][0] == v:
                sharp_edgs[kk] = sorted(tuple((len(new_vertices),sharp_edgs[kk][1])))
            elif sharp_edgs[kk][1] == v:
                sharp_edgs[kk] = sorted(tuple((len(new_vertices),sharp_edgs[kk][0])))

        if v in dico['sharp_sommets']:
            new_vertex = dico["sommets"][v]
            sharp_sommets.append(len(new_vertices))
        elif len(sharp_edges) < 2:
            adjacent_faces = [face for face in dico["faces"] if v in face]

            face_point_avg = np.mean([face_points[face_to_index[tuple(face)]] for face in adjacent_faces], axis=0)
            edge_point_avg = np.mean([edge_points[edge_to_index[edge]] for edge in dico["aretes"] if v in edge], axis=0)
        

            new_vertex = (1*face_point_avg + 2*edge_point_avg + 1*dico["sommets"][v]) / 4
        elif len(sharp_edges) == 2:
            new_vertex = 3/4 * dico["sommets"][v] + 1/8 * (dico["sommets"][sharp_edges[0][1 - sharp_edges[0].index(v)]] + dico["sommets"][sharp_edges[1][1 - sharp_edges[1].index(v)]])
        else:
            new_vertex = dico["sommets"][v]
        new_vertices.append(new_vertex)
    
    for face in dico["faces"]:
        face_point_index = face_to_index[tuple(face)]
        for i in range(len(face)):
            v1, v2, v3 = face[i], face[(i+1)%len(face)], face[(i+2)%len(face)]
            edge_index1 = edge_to_index[tuple(sorted((v1, v2)))]
            edge_index2 = edge_to_index[tuple(sorted((v2, v3)))]
            new_faces.append([face_point_index, len(dico["faces"]) + edge_index1, len(dico["faces"]) + len(dico["aretes"]) + v2, len(dico["faces"]) + edge_index2])

    return {"faces": np.array(new_faces), "sommets": np.array(new_vertices), "aretes": set(tuple(sorted([face[i], face[(i+1)%len(face)]])) for face in new_faces for i in range(len(face))), 'nsommets': len(new_vertices), 'sharp_sommets': sharp_sommets, 'sharp_aretes': sharp_edgs}

model, dico_init = load_mesh("./modele_3D/cube.obj")

dico_init['sharp_sommets'] = []
dico_init['sharp_aretes'] = [(4,5), (5,6), (6,7), (4,7), (0,1), (1,2), (2,3), (0,3)]

dico = dico_init.copy()
affichage(dico, dico_init)

for i in range(5):
    dico = catmull_clark(dico)
    affichage(dico, dico_init, "Subdivision " + str(i+1) + "/5")
