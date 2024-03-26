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

    return maillage, dico

def get_faces_from_edge(dico, edge):
    faces = []
    for face in dico["faces"]:
        if set(edge).issubset(set(face)):
            faces.append(face)
    return faces

def affichage(dico):
    faces = np.hstack((np.full((len(dico["faces"]), 1), len(dico["faces"][0])), dico["faces"]))

    mesh = pv.PolyData(dico["sommets"], faces)
    mesh.plot(color="orange", show_edges=True, edge_color="black", line_width=1)

def catmull_clark(dico):
    new_faces = []
    new_vertices = []
    face_points = []
    edge_points = []

    face_to_index = {tuple(face): i for i, face in enumerate(dico["faces"])}
    edge_to_index = {edge: i for i, edge in enumerate(dico["aretes"])}

    face_points = [np.mean(dico["sommets"][face], axis=0) for face in dico["faces"]]
    new_vertices.extend(face_points)
    
    edge_points = []
    for edge in dico["aretes"]:
        adjacent_faces = get_faces_from_edge(dico, edge)
        edge_point = np.mean([dico["sommets"][v] for v in edge] + [face_points[face_to_index[tuple(face)]] for face in adjacent_faces], axis=0)
        edge_points.append(edge_point)
    new_vertices.extend(edge_points)

    for v in range(dico['nsommets']):
        adjacent_faces = [face for face in dico["faces"] if v in face]
        n = 4

        face_point_avg = np.mean([face_points[face_to_index[tuple(face)]] for face in adjacent_faces], axis=0)
        edge_point_avg = np.mean([edge_points[edge_to_index[edge]] for edge in dico["aretes"] if v in edge], axis=0)

        new_vertex = (1*face_point_avg + 2*edge_point_avg + (n-3)*dico["sommets"][v]) / n
        new_vertices.append(new_vertex)
    
    for face in dico["faces"]:
        face_point_index = face_to_index[tuple(face)]
        for i in range(len(face)):
            v1, v2, v3 = face[i], face[(i+1)%len(face)], face[(i+2)%len(face)]
            edge_index1 = edge_to_index[tuple(sorted((v1, v2)))]
            edge_index2 = edge_to_index[tuple(sorted((v2, v3)))]
            new_faces.append([face_point_index, len(dico["faces"]) + edge_index1, len(dico["faces"]) + len(dico["aretes"]) + v2, len(dico["faces"]) + edge_index2])

    return {"faces": np.array(new_faces), "sommets": np.array(new_vertices), "aretes": set(tuple(sorted([face[i], face[(i+1)%len(face)]])) for face in new_faces for i in range(len(face))), 'nsommets': len(new_vertices)}

def calculate_angle(dico, edge):
    faces = get_faces_from_edge(dico, edge)

    normals = [np.cross(dico["sommets"][face[1]] - dico["sommets"][face[0]], dico["sommets"][face[2]] - dico["sommets"][face[0]]) for face in faces]

    normals = [normal / np.linalg.norm(normal) for normal in normals]

    angle = np.arccos(np.dot(normals[0], normals[1]))

    return angle



model, dico = load_mesh("./modele_3D/cube.obj")

affichage(dico)

for i in range(5):
    print("Subdivision", i)
    dico = catmull_clark(dico)
    affichage(dico)
