# python -m pip install pygltflib
# python -m pip install Pillow

import base64
import io
import numpy as np

from PIL import Image
from pygltflib import GLTF2

from geometry_utils import *

def get_raw_buffer_bytes(glb_model, buffer_index):
    """
    Retorna os bytes de um buffer (seja data:, arquivo externo ou chunk binário do GLB).
    """
    buffer = glb_model.buffers[buffer_index]
    if buffer.uri:
        if buffer.uri.startswith("data:"):
            comma = buffer.uri.find(",")
            return base64.b64decode(buffer.uri[comma + 1:])
        else:
            # caminho relativo ao arquivo GLB/Gltf atual
            return open(buffer.uri, "rb").read()
    else:
        # Para GLB o binário fica no chunk; pygltflib oferece binary_blob()
        raw = glb_model.binary_blob()
        if raw is None:
            raise ValueError("Buffer.uri é None e gltf.binary_blob() retornou None.")
        return raw
   

def get_buffer_data(glb_model, accessor_id):
    """
    Retorna um numpy array com os dados do accessor.
    Lida com bufferView.byteOffset e accessor.byteOffset, além dos tipos de componente.
    """
    accessor = glb_model.accessors[accessor_id]
    if accessor.bufferView is None:
        raise NotImplementedError("Accessors sem bufferView (sparse ou outras formas) não implementado aqui.")

    bv = glb_model.bufferViews[accessor.bufferView]
    buffer_index = bv.buffer

    raw_buffer = get_raw_buffer_bytes(glb_model, buffer_index)

    # offsets (podem ser None)
    bv_offset = bv.byteOffset or 0
    acc_offset = accessor.byteOffset or 0
    start = bv_offset + acc_offset

    # tipo do componente
    component_type_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32
    }
    if accessor.componentType not in component_type_map:
        raise ValueError(f"componentType {accessor.componentType} não suportado")

    dtype = component_type_map[accessor.componentType]

    # quantidade por elemento (SCALAR, VEC2, VEC3, VEC4, MAT2...)
    type_count_map = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16
    }
    if accessor.type not in type_count_map:
        raise ValueError(f"accessor.type {accessor.type} não reconhecido")

    type_count = type_count_map[accessor.type]
    count = accessor.count * type_count

    # tamanho em bytes para garantir que não estamos passando além do buffer
    itemsize = np.dtype(dtype).itemsize
    needed_bytes = accessor.count * type_count * itemsize

    # Se bufferView.byteLength estiver definido, podemos limitar
    bv_length = bv.byteLength if bv.byteLength is not None else len(raw_buffer) - bv_offset
    if start + needed_bytes > bv_offset + bv_length:
        # Ajuste: tente não falhar misteriosamente — corte se necessário
        available = (bv_offset + bv_length) - start
        max_count = available // itemsize
        if max_count <= 0:
            raise ValueError("Não há bytes suficientes no bufferView para o accessor solicitado.")
        
        # reduz o count para o máximo possível (aviso)
        count = max_count

    # np.frombuffer aceita offset em bytes
    arr = np.frombuffer(raw_buffer, dtype=dtype, count=count, offset=start)

    if type_count > 1:
        arr = arr.reshape((-1, type_count))

    return arr


def load_glb_mesh(glb_model, mesh_index):
    mesh = glb_model.meshes[mesh_index]
    primitive = mesh.primitives[0]

    # positions
    vertices_pos = get_buffer_data(glb_model, primitive.attributes.POSITION)

    vertices_normals = None
    if hasattr(primitive.attributes, "NORMAL") and primitive.attributes.NORMAL is not None:
        vertices_normals = get_buffer_data(glb_model, primitive.attributes.NORMAL)

    vertices_uvs = None
    if hasattr(primitive.attributes, "TEXCOORD_0") and primitive.attributes.TEXCOORD_0 is not None:
        vertices_uvs = get_buffer_data(glb_model, primitive.attributes.TEXCOORD_0)

    faces = None
    if primitive.indices is not None:
        # indices normalmente são SCALAR -> devolvemos um array 1D
        idx_arr = get_buffer_data(glb_model, primitive.indices)
        faces = idx_arr.flatten().reshape(-1, 3, 1)

    # TEXTURA: suporte para image.uri (data: ou arquivo) ou image.bufferView
    texture_image = None
    material = glb_model.materials[primitive.material] if primitive.material is not None and len(glb_model.materials) > primitive.material else None
    if material and material.pbrMetallicRoughness and material.pbrMetallicRoughness.baseColorTexture:
        tex_index = material.pbrMetallicRoughness.baseColorTexture.index
        img_index = glb_model.textures[tex_index].source
        image = glb_model.images[img_index]

        if image.uri:
            if image.uri.startswith("data:"):
                comma = image.uri.find(",")
                img_bytes = base64.b64decode(image.uri[comma + 1:])
                texture_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            else:
                # caminho relativo
                texture_image = Image.open(image.uri).convert("RGBA")

        elif image.bufferView is not None:
            # imagem embutida via bufferView (com image.mimeType)
            bv = glb_model.bufferViews[image.bufferView]
            buffer_bytes = get_raw_buffer_bytes(glb_model, bv.buffer)
            start = (bv.byteOffset or 0)
            length = bv.byteLength or 0
            img_bytes = buffer_bytes[start:start+length]
            texture_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
   
    if (len(vertices_normals) == 0):
        vertices_normals = compute_vertices_normals(vertices_pos, faces)

    return vertices_pos, vertices_normals, vertices_uvs, faces, texture_image


def load_glb_model_no_texture(glb_file_path):
    vertices_pos = []
    vertices_normals = []
    faces = []

    glb_model = GLTF2().load(glb_file_path)
    num_meshes = len(glb_model.meshes)

    for mesh_index in range(num_meshes):
        mesh_vertices_pos, mesh_vertices_normals, _, mesh_faces, _ = load_glb_mesh(glb_model, mesh_index)

        if len(mesh_vertices_pos) > 0:
            valor = len(vertices_pos)
            mesh_faces = [[[f[0][0] + valor], [f[1][0] + valor], [f[2][0] + valor]] for f in mesh_faces]

            vertices_pos.extend(mesh_vertices_pos)
            vertices_normals.extend(mesh_vertices_normals)
            faces.extend(mesh_faces)

    vertices_pos = np.array(vertices_pos, dtype=np.float32)
    vertices_normals = np.array(vertices_normals, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)

    faces_normals = compute_faces_normals(vertices_pos, faces)
    if (len(vertices_normals) == 0):
        vertices_normals = compute_vertices_normals(vertices_pos, faces)

    bounding_box = compute_bounding_box(vertices_pos, faces)

    bounding_box_sizes = bounding_box[1] - bounding_box[0]
    if max(bounding_box_sizes) > 100000:
        vertices_pos = vertices_pos * 0.001
        bounding_box = bounding_box * 0.001

    return vertices_pos, vertices_normals, faces, faces_normals


def load_glb_model_with_texture(glb_file_path):
    model = []

    glb_model = GLTF2().load(glb_file_path)
    num_meshes = len(glb_model.meshes)

    for mesh_index in range(num_meshes):
        mesh_vertices_pos, mesh_vertices_normals, mesh_vertices_uvs, mesh_faces, mesh_texture_image = load_glb_mesh(glb_model, mesh_index)

        if len(mesh_vertices_pos) > 0:
            if (len(mesh_vertices_normals) == 0):
                mesh_vertices_normals = compute_vertices_normals(mesh_vertices_pos, mesh_faces)

            mesh_faces_normals = compute_faces_normals(mesh_vertices_pos, mesh_faces)
            model.append([mesh_vertices_pos, mesh_vertices_normals, mesh_vertices_uvs, mesh_faces, mesh_faces_normals, mesh_texture_image])

    return model