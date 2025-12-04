import ctypes
import math
import os
import random
import sys

import glfw
import glm
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as gls
from PIL import Image

from geometry_utils import (
    compute_bounding_box,
    get_bounding_box_center,
    union_bounding_boxes,
)
from FBX_utils import load_fbx_model

try:
    from noise import pnoise2  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover - dependencia externa obrigatoria
    raise RuntimeError(
        "O pacote 'noise' e obrigatorio para gerar o terreno procedural. "
        "Execute 'pip install noise==1.2.2' no mesmo interpretador antes de iniciar a cena."
    ) from exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != BASE_DIR:
    os.chdir(BASE_DIR)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FIELD_OF_VIEW = 60.0
NEAR_PLANE = 0.1
FAR_PLANE = 2500.0

TERRAIN_RESOLUTION = 256
TERRAIN_SIZE = 360.0  # Req 1a: dimensao do terreno >= 300 m
TERRAIN_PEAK_HEIGHT = 8.0  # Reduz altitude maxima para diminuir montanhas
TERRAIN_TEXTURE_REPEAT = 24.0
TERRAIN_TEXTURE_CANDIDATES = [  # Req 1b: texturas candidatas aplicadas no solo
    os.path.join("Textures", "rocky_terrain_diff_1k.jpg"),
    os.path.join("Textures", "terrain_grass.png"),  # fallback se a nova textura nao existir
]
TERRAIN_SEED = 2024
TERRAIN_HEIGHTMAP_PATH = os.path.join("Heightmaps", "heightmap_02.png")
USE_IMAGE_HEIGHTMAP = True  # Req 1a: aceita heightmap externo ou gera proceduralmente
TERRAIN_GENTLE_FACTOR = 0.22  # controla ondulacao suave
TERRAIN_RIDGE_FACTOR = 0.6  # controla altura das cadeias mais altas
TERRAIN_FLAT_CENTER_RADIUS = 0.32  # raio normalizado de planicie no centro
TERRAIN_EDGE_FULL_RADIUS = 0.78  # raio onde montanhas entram em plena forca

NUM_INSTANCES_PER_CHARACTER = 12

PLAYER_WALK_SPEED = 7.5
PLAYER_RUN_MULTIPLIER = 1.8
PLAYER_JUMP_SPEED = 8.5
GRAVITY = -20.0
PLAYER_MOUSE_SENSITIVITY = 0.12
ORIENTATION_CORRECTION_RAD = math.pi * 1.5  # corrige orientacao inicial em 270° (frente alinhada com W)
FACING_CORRECTION_RAD = math.pi  # alinhamento de movimento/rotacao igual ao open_world_simulation

CAMERA_MIN_PITCH = -65.0
CAMERA_MAX_PITCH = -5.0
CAMERA_DISTANCE = 8.5
CAMERA_SAFETY_HEIGHT = 1.0

FOG_DENSITY = 0.0009  # Req 1c: densidade do fog atmosferico

SUN_ORBIT_RADIUS = 900.0
SHADOW_MAP_RESOLUTION = 2048
SIM_HOURS_PER_REAL_SECOND = 24.0 / 60.0  # Req 3a: cada minuto real avanca 1h simulada do sol leste->oeste

FBX_CHARACTERS = [
    {
        "name": "Future Car (FBX sem textura)",
        "path": os.path.join("FBX models", "fbx_futureCar", "CraneoFBX.fbx"),
        "color": [0.8, 0.85, 1.0],
        "yaw_offset_deg": 180.0,
        "pitch_offset_deg": -90.0,  # postura original (sem inclinar para frente)
        "roll_offset_deg": 0.0,
        "target_height": 1.2,  # caveira menor
        "hover_offset": -0.6,  # afunda mais para parecer enterrada
        "facing_correction_deg": 180.0,  # orbita olhando para frente (igual open_world_simulation)
    },
    {
        "name": "Pikachu",
        "path": os.path.join("FBX models", "fbx_Lobo", "PikachuF.FBX"),
        "color": [1.0, 0.95, 0.4],
        "yaw_offset_deg": 180.0,
        "pitch_offset_deg": -90.0,  # alinhado com os demais personagens
        "roll_offset_deg": 0.0,
        "target_height": 1.2,
        "facing_correction_deg": 180.0,
    },
    {
        "name": "Black Dragon",   
        "path": os.path.join("FBX models", "fbx_blackDragon", "Dragon_Baked_Actions_fbx_7.4_binary.fbx"),
        "color": [0.7, 0.7, 0.7],
        "yaw_offset_deg": 0.0,  # corrige para nao andar de costas
        "pitch_offset_deg": -90.0,  # igual ao lobo para manter o angulo correto
        "target_width": 4.5,
        "target_height": 2.2,
        "hover_offset": 4.0,  # deixa o dragao bem acima do solo
        "wing_flap": True,  # aplica efeito simples de batimento nas asas
        "wing_amplitude": 1.2,
        "wing_frequency": 3.2,
    },
    {
        "name": "Shun Gold",
        "path": os.path.join("FBX models", "fbx_Camel", "Shun gold.FBX"),
        "color": [0.85, 0.8, 0.65],
        "yaw_offset_deg": 180.0,
        "pitch_offset_deg": 0.0,
        "roll_offset_deg": 180.0,  # corrigir inversao (cabeca para baixo)
        "facing_correction_deg": 180.0,
    },
]

CONTROLLED_CHARACTER_START_INDEX = 2  # 0=Future Car, 1=Wolf, 2=Black Dragon, 3=Shun Gold

CHARACTER_SWITCH_KEY = glfw.KEY_C
CHARACTER_DIRECT_KEYS = [glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4]


window = None
window_size = [WINDOW_WIDTH, WINDOW_HEIGHT]

main_shader = None
depth_shader = None
sun_shader = None
clock_shader = None
clock_vao = None
clock_vbo = None

terrain_renderable = None
terrain_heightmap = None
terrain_bbox = None
scene_center = glm.vec3(0.0, 0.0, 0.0)

fbx_assets = []
crowd_instances = []

shadow_fbo = None
shadow_texture = None
sun_quad_vao = None

camera_pos = glm.vec3(0.0, 10.0, 5.0)
camera_front = glm.vec3(0.0, -0.2, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
camera_orbit = {
    "yaw": 0.0,
    "pitch": -25.0,
    "distance": CAMERA_DISTANCE,
}

sun_direction = glm.vec3(-1.0, -1.0, 0.0)
sun_color = glm.vec3(1.0, 1.0, 1.0)
sky_color = glm.vec3(0.25, 0.35, 0.7)
fog_color = glm.vec3(0.25, 0.35, 0.7)
sun_position = glm.vec3(0.0, 0.0, 0.0)
light_space_matrix = glm.mat4(1.0)
sun_light_factor = 0.0
sun_fog_density = 0.002
sun_ambient_color = glm.vec3(0.08, 0.08, 0.1)
sim_hours = 6.0

player_state = {
    "position": glm.vec3(0.0, 0.0, 5.0),
    "yaw_rad": 0.0,
    "asset_index": CONTROLLED_CHARACTER_START_INDEX,
    "asset": None,
    "base_offset": 0.0,
    "vertical_velocity": 0.0,
    "jump_offset": 0.0,
    "camera_anchor_height": 1.8,
    "model_matrix": glm.mat4(1.0),
    "normal_matrix": glm.mat3(1.0),
    "is_moving": False,
}

key_states = {}

cursor_first_update = True
last_cursor_x = WINDOW_WIDTH / 2.0
last_cursor_y = WINDOW_HEIGHT / 2.0


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_text_file(path):
    resolved = _resolve_path(path)
    with open(resolved, "r", encoding="utf-8") as file:
        return file.read()


def compile_shader_program(vertex_path, fragment_path):
    vertex_source = load_text_file(vertex_path)
    fragment_source = load_text_file(fragment_path)

    vertex_shader = gls.compileShader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = gls.compileShader(fragment_source, GL_FRAGMENT_SHADER)
    program = gls.compileProgram(vertex_shader, fragment_shader)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program


def create_texture(path, repeat=True):
    full_path = _resolve_path(path)
    if not os.path.isfile(full_path):
        print(f"[WARN] Textura nao encontrada: {full_path}")
        return None

    try:
        image = Image.open(full_path).convert("RGBA")
    except Exception as exc:
        print(f"[WARN] Falha ao abrir textura '{full_path}': {exc}")
        return None

    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    data = np.array(image, dtype=np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        image.width,
        image.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data,
    )

    wrap_mode = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id


color_texture_cache = {}
missing_texture_report = set()


def create_solid_color_texture(color):
    """Cria uma textura 1x1 com a cor difusa do material para evitar modelos sem textura."""
    key = tuple(round(float(c), 4) for c in color)
    if key in color_texture_cache:
        return color_texture_cache[key]

    rgba = np.array([[int(max(0.0, min(1.0, c)) * 255) for c in (*color, 1.0)]], dtype=np.uint8)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    color_texture_cache[key] = texture_id
    return texture_id


def _report_missing_texture(asset_name, mat_idx, candidates):
    key = (asset_name, mat_idx)
    if key in missing_texture_report:
        return
    missing_texture_report.add(key)
    print(f"[WARN] Textura nao encontrada para '{asset_name}' (material {mat_idx}). Tentativas: {', '.join(os.path.basename(c) for c in candidates if c)}")


def load_terrain_texture():
    for candidate in TERRAIN_TEXTURE_CANDIDATES:
        texture = create_texture(candidate, repeat=True)
        if texture:
            print(f"[info] Textura do terreno carregada: {os.path.basename(candidate)}")
            return texture
        full_path = _resolve_path(candidate)
        if os.path.exists(full_path):
            print(f"[WARN] Falha ao carregar textura do terreno '{os.path.basename(candidate)}', tentando proxima.")
    print("[WARN] Nenhuma textura valida encontrada para o terreno; usando cor solida.")
    return create_solid_color_texture((0.45, 0.55, 0.45))


def create_vao(buffers):
    vao_id = glGenVertexArrays(1)
    glBindVertexArray(vao_id)

    for buffer_data, attribute_index in buffers:
        vbo_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBufferData(GL_ARRAY_BUFFER, buffer_data.nbytes, buffer_data, GL_STATIC_DRAW)
        glVertexAttribPointer(
            attribute_index,
            buffer_data.shape[1],
            GL_FLOAT,
            GL_FALSE,
            0,
            ctypes.c_void_p(0),
        )
        glEnableVertexAttribArray(attribute_index)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return vao_id


def create_sun_billboard_resources():
    """Cria um quad em NDC para desenhar o disco do sol."""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    # quad em triângulo strip: (-1,-1), (1,-1), (1,1), (-1,1)
    vertices = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao


def flatten_geometry(vertices, normals, uvs, faces):
    vertex_pos = []
    vertex_normals = []
    vertex_uvs = []

    has_normals = len(normals) > 0
    has_uvs = len(uvs) > 0

    for face in faces:
        for vertex in face:
            idx = vertex[0]
            vertex_pos.append(vertices[idx])
            if has_normals:
                vertex_normals.append(normals[idx])
            else:
                vertex_normals.append([0.0, 1.0, 0.0])

            if has_uvs:
                vertex_uvs.append(uvs[idx])
            else:
                vertex_uvs.append([0.0, 0.0])

    return (
        np.array(vertex_pos, dtype=np.float32),
        np.array(vertex_normals, dtype=np.float32),
        np.array(vertex_uvs, dtype=np.float32),
    )


def _build_rotation_matrix(yaw_rad, pitch_rad, roll_rad):
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)

    rot_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rot_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return rot_yaw @ rot_pitch @ rot_roll


def _rotate_vertices(vertices, rotation_matrix):
    # vertices: (N, 3); rotation_matrix: (3, 3)
    return (rotation_matrix @ vertices.T).T




def _smoothstep(edge0, edge1, x):
    if edge0 == edge1:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _edge_mountain_mask(u, v, inner=TERRAIN_FLAT_CENTER_RADIUS, outer=TERRAIN_EDGE_FULL_RADIUS):
    # Normaliza o raio para 0..1 (0 centro, 1 cantos) e suaviza para ativar montanhas nas bordas.
    radius = math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 1.41421356237
    return _smoothstep(inner, outer, radius)


def _generate_planar_uvs(vertices):
    """UV XZ de fallback apenas para assets problemáticos (não usado pelo Shun)."""
    if len(vertices) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    verts = np.asarray(vertices, dtype=np.float32)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    u = (verts[:, 0] - mins[0]) / extent[0]
    v = (verts[:, 2] - mins[2]) / extent[2]
    return np.stack([u, v], axis=1).astype(np.float32)


def _find_local_texture(base_dir):
    """Escolhe a primeira textura encontrada no diretório do modelo."""
    if not base_dir:
        return None

    search_dirs = [
        base_dir,
        os.path.join(base_dir, "textures"),
        os.path.join(base_dir, "images"),
    ]
    common_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".dds")
    candidates = []

    for root in search_dirs:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.lower().endswith(common_exts):
                candidates.append(os.path.join(root, name))

    if not candidates:
        return None

    def _score(name):
        lower = os.path.basename(name).lower()
        score = 0
        for kw in ("color", "diff", "albedo", "base", "body", "fur", "tex", "_c", "_d"):
            if kw in lower:
                score += 2
        for kw in ("normal", "nrm", "_n", "spec", "rough", "metal", "ao"):
            if kw in lower:
                score -= 1
        return score

    return max(candidates, key=lambda path: (_score(path), os.path.getsize(path)))


def _filter_diffuse_candidates(paths):
    blocklist = ("ao", "curvature", "mask", "normal", "_n", "rough", "metal", "orm", "spec")
    filtered = []
    for candidate in paths:
        if not candidate:
            continue
        base = os.path.basename(candidate).lower()
        if any(token in base for token in blocklist):
            continue
        filtered.append(candidate)
    return filtered


def resolve_texture_path(texture_candidates, base_dir=None, asset_name=None):
    prioritized = []
    if asset_name and "dragon" in asset_name.lower():
        dragon_dir = base_dir or os.path.join(BASE_DIR, "FBX models", "fbx_blackDragon")
        prioritized = [
            os.path.join(dragon_dir, "textures", "Dragon_Bump_Col2.jpg"),
            os.path.join(dragon_dir, "Dragon_Bump_Col2.jpg"),
            os.path.join(dragon_dir, "textures", "Dragon_ground_color.jpg"),
            os.path.join(dragon_dir, "Dragon_ground_color.jpg"),
        ]

    search_dirs = []
    if base_dir:
        search_dirs.append(base_dir)
        search_dirs.append(os.path.join(base_dir, "textures"))
    if base_dir:
        search_dirs.append(os.path.join(base_dir, "images"))
    search_dirs.append(BASE_DIR)
    search_dirs.append(os.path.join(BASE_DIR, "Textures"))

    common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".dds"]

    search_candidates = list(dict.fromkeys(list(prioritized) + list(texture_candidates)))

    for candidate in search_candidates:
        if os.path.isfile(candidate):
            return candidate
        filename = os.path.basename(candidate)
        name_no_ext, ext = os.path.splitext(filename)
        for root in search_dirs:
            local_candidate = os.path.join(root, filename)
            if os.path.isfile(local_candidate):
                return local_candidate
            # tenta variantes de extensao
            for alt_ext in common_exts:
                alt_candidate = os.path.join(root, name_no_ext + alt_ext)
                if os.path.isfile(alt_candidate):
                    return alt_candidate

    # Nenhum caminho diretamente resolvido; tenta localizar qualquer textura local
    return _find_local_texture(base_dir)


def apply_texture_hints(asset_config, texture_paths, asset_dir):
    """Reordena candidatos para favorecer o mapa difuso correto por asset (não altera Shun)."""
    name = asset_config.get("name", "").lower()
    uniq = list(dict.fromkeys(texture_paths))

    def _preferred(paths):
        return [p for p in paths if p]

    if any(key in name for key in ("alien animal", "futuristic car", "qishilong")):
        preferred = _preferred(
            [
                os.path.join(asset_dir, "textures", "T_M_B_44_Qishilong_body01_B.png"),
                os.path.join(asset_dir, "textures", "T_M_B_44_Qishilong_body02_B.png"),
                os.path.join(asset_dir, "T_M_B_44_Qishilong_body01_B.png"),
                os.path.join(asset_dir, "T_M_B_44_Qishilong_body02_B.png"),
                "T_M_B_44_Qishilong_body01_B.png",
                "T_M_B_44_Qishilong_body02_B.png",
                os.path.join(asset_dir, "textures", "Base Color.jpg"),
                "Base Color.jpg",
                os.path.join(asset_dir, "textures", "Diffuse_2.jpg"),
                "Diffuse_2.jpg",
            ]
        )
        filtered = _filter_diffuse_candidates(uniq)
        return list(dict.fromkeys(preferred + filtered))

    if "wolf" in name:
        preferred = _preferred(
            [
                os.path.join(asset_dir, "textures", "Wolf_Body.jpg"),
                "Wolf_Body.jpg",
            ]
        )
        return list(dict.fromkeys(preferred + uniq))

    if "pikachu" in name:
        preferred = _preferred(
            [
                os.path.join(asset_dir, "images", "pm0025_00_BodyA1.png"),
                os.path.join(asset_dir, "images", "pm0025_00_BodyB1.png"),
                "pm0025_00_BodyA1.png",
                "pm0025_00_BodyB1.png",
            ]
        )
        return list(dict.fromkeys(preferred + uniq))

    if "dragon" in name:
        preferred = _preferred(
            [
                os.path.join(asset_dir, "textures", "Dragon_Bump_Col2.jpg"),
                "Dragon_Bump_Col2.jpg",
                os.path.join(asset_dir, "textures", "Dragon_ground_color.jpg"),
                "Dragon_ground_color.jpg",
            ]
        )
        return list(dict.fromkeys(preferred + uniq))

    return uniq


def create_procedural_heightmap(size, height_scale, resolution, seed):
    heightmap = np.zeros((resolution, resolution), dtype=np.float32)
    positions = np.zeros((resolution * resolution, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    uvs = np.zeros((resolution * resolution, 2), dtype=np.float32)

    for j in range(resolution):
        for i in range(resolution):
            u = i / (resolution - 1)
            v = j / (resolution - 1)
            x = (u - 0.5) * size
            z = (v - 0.5) * size

            gentle = pnoise2(x * 0.02, z * 0.02, repeatx=1024, repeaty=1024, base=seed) * 2.0
            ridged = pnoise2(x * 0.005, z * 0.005, repeatx=1024, repeaty=1024, base=seed + 1337)

            edge_mask = _edge_mountain_mask(u, v)
            ridge = max(0.0, ridged - 0.25)
            rolling = gentle * height_scale * TERRAIN_GENTLE_FACTOR * (0.45 + 0.55 * edge_mask)
            height = rolling + (ridge ** 2.2) * height_scale * TERRAIN_RIDGE_FACTOR * edge_mask

            idx = j * resolution + i
            heightmap[j, i] = height
            positions[idx] = [x, height, z]
            uvs[idx] = [u * TERRAIN_TEXTURE_REPEAT, v * TERRAIN_TEXTURE_REPEAT]

    step = size / (resolution - 1)
    for j in range(resolution):
        for i in range(resolution):
            left = heightmap[j, max(i - 1, 0)]
            right = heightmap[j, min(i + 1, resolution - 1)]
            down = heightmap[max(j - 1, 0), i]
            up = heightmap[min(j + 1, resolution - 1), i]

            normal = glm.normalize(
                glm.vec3(
                    (left - right) / (2.0 * step),
                    1.0,
                    (down - up) / (2.0 * step),
                )
            )
            idx = j * resolution + i
            normals[idx] = [normal.x, normal.y, normal.z]

    quad_count = (resolution - 1) * (resolution - 1)
    indices = np.zeros(quad_count * 6, dtype=np.uint32)
    pointer = 0
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            top_left = j * resolution + i
            top_right = top_left + 1
            bottom_left = (j + 1) * resolution + i
            bottom_right = bottom_left + 1

            indices[pointer: pointer + 6] = [
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            ]
            pointer += 6

    faces = indices.reshape(-1, 3, 1)
    return heightmap, positions, normals, faces, uvs


def create_heightmap_from_image(size, height_scale, resolution, image_path):
    """Carrega um heightmap de imagem (grayscale) e gera vertices/uv/normais."""
    img = Image.open(image_path).convert("L").resize((resolution, resolution), Image.BILINEAR)
    data = np.array(img, dtype=np.float32) / 255.0
    heightmap = data * height_scale

    positions = np.zeros((resolution * resolution, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    uvs = np.zeros((resolution * resolution, 2), dtype=np.float32)

    for j in range(resolution):
        for i in range(resolution):
            u = i / (resolution - 1)
            v = j / (resolution - 1)
            x = (u - 0.5) * size
            z = (v - 0.5) * size
            edge_mask = _edge_mountain_mask(u, v)
            flatten = 0.45 + 0.55 * edge_mask  # mantem centro mais plano e bordas mais movimentadas
            y = heightmap[j, i] * flatten
            heightmap[j, i] = y

            idx = j * resolution + i
            positions[idx] = [x, y, z]
            uvs[idx] = [u * TERRAIN_TEXTURE_REPEAT, v * TERRAIN_TEXTURE_REPEAT]

    step = size / (resolution - 1)
    for j in range(resolution):
        for i in range(resolution):
            left = heightmap[j, max(i - 1, 0)]
            right = heightmap[j, min(i + 1, resolution - 1)]
            down = heightmap[max(j - 1, 0), i]
            up = heightmap[min(j + 1, resolution - 1), i]

            normal = glm.normalize(
                glm.vec3(
                    (left - right) / (2.0 * step),
                    1.0,
                    (down - up) / (2.0 * step),
                )
            )
            idx = j * resolution + i
            normals[idx] = [normal.x, normal.y, normal.z]

    quad_count = (resolution - 1) * (resolution - 1)
    indices = np.zeros(quad_count * 6, dtype=np.uint32)
    pointer = 0
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            top_left = j * resolution + i
            top_right = top_left + 1
            bottom_left = (j + 1) * resolution + i
            bottom_right = bottom_left + 1

            indices[pointer: pointer + 6] = [
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            ]
            pointer += 6

    faces = indices.reshape(-1, 3, 1)
    return heightmap, positions, normals, faces, uvs


def get_heightmap_height(heightmap, bbox, x_world, z_world):
    xmin, _, zmin = bbox[0]
    xmax, _, zmax = bbox[1]

    x_prop = (x_world - xmin) / (xmax - xmin)
    z_prop = (z_world - zmin) / (zmax - zmin)

    height, width = heightmap.shape
    x = np.clip(x_prop * (width - 1), 0, width - 1)
    z = np.clip(z_prop * (height - 1), 0, height - 1)

    x1, z1 = int(np.floor(x)), int(np.floor(z))
    x2, z2 = min(x1 + 1, width - 1), min(z1 + 1, height - 1)

    if x1 == x2 and z1 == z2:
        return float(heightmap[z1, x1])

    dx = x - x1
    dz = z - z1

    h00 = heightmap[z1, x1]
    h10 = heightmap[z1, x2]
    h01 = heightmap[z2, x1]
    h11 = heightmap[z2, x2]

    h_top = h00 * (1.0 - dx) + h10 * dx
    h_bottom = h01 * (1.0 - dx) + h11 * dx
    return float(h_top * (1.0 - dz) + h_bottom * dz)


def create_terrain_renderable():
    global terrain_heightmap, terrain_bbox, scene_center

    # Req 1a: terreno pode vir de heightmap externo ou ser gerado proceduralmente
    if USE_IMAGE_HEIGHTMAP and os.path.isfile(TERRAIN_HEIGHTMAP_PATH):
        (
            terrain_heightmap,
            vertices,
            vertex_normals,
            faces,
            vertex_uvs,
        ) = create_heightmap_from_image(TERRAIN_SIZE, TERRAIN_PEAK_HEIGHT, TERRAIN_RESOLUTION, TERRAIN_HEIGHTMAP_PATH)
        print(f"Heightmap carregado de imagem: {TERRAIN_HEIGHTMAP_PATH}")
    else:
        (
            terrain_heightmap,
            vertices,
            vertex_normals,
            faces,
            vertex_uvs,
        ) = create_procedural_heightmap(TERRAIN_SIZE, TERRAIN_PEAK_HEIGHT, TERRAIN_RESOLUTION, TERRAIN_SEED)
        if USE_IMAGE_HEIGHTMAP:
            print(f"Heightmap de imagem nao encontrado, usando procedural: {TERRAIN_HEIGHTMAP_PATH}")

    terrain_bbox = compute_bounding_box(vertices, faces)
    center = get_bounding_box_center(terrain_bbox)
    scene_center = glm.vec3(center[0], center[1], center[2])

    v_pos, v_normals, v_uvs = flatten_geometry(vertices, vertex_normals, vertex_uvs, faces)
    vao_id = create_vao(
        [
            (v_pos, 0),
            (v_normals, 1),
            (v_uvs, 2),
        ]
    )

    terrain_texture = load_terrain_texture()  # Req 1b: aplica textura no terreno
    return {
        "vao": vao_id,
        "vertex_count": len(v_pos),
        "texture_id": terrain_texture,
        "model_matrix": glm.mat4(1.0),
        "normal_matrix": glm.mat3(1.0),
    }


def load_fbx_asset(asset_config):
    fbx_path_abs = os.path.join(BASE_DIR, asset_config["path"])
    asset_dir = os.path.dirname(fbx_path_abs)
    name_lower = asset_config.get("name", "").lower()

    yaw_off_rad = math.radians(asset_config.get("yaw_offset_deg", 0.0))
    pitch_off_rad = math.radians(asset_config.get("pitch_offset_deg", 0.0))
    roll_off_rad = math.radians(asset_config.get("roll_offset_deg", 0.0))
    facing_correction_rad = math.radians(asset_config.get("facing_correction_deg", 0.0))
    if facing_correction_rad == 0.0 and abs(asset_config.get("yaw_offset_deg", 0.0) - 180.0) < 1e-3:
        facing_correction_rad = FACING_CORRECTION_RAD
    rotation_matrix = _build_rotation_matrix(yaw_off_rad, pitch_off_rad, roll_off_rad)

    model_data = load_fbx_model(fbx_path_abs)
    if not model_data or (isinstance(model_data, tuple) and all(m is None for m in model_data)):
        raise RuntimeError(f"Falha ao carregar {asset_config['name']} em {fbx_path_abs}")
    if isinstance(model_data, tuple):
        # caso de retorno invalido, limpa Nones
        model_data = [m for m in model_data if m is not None]
    if not model_data:
        raise RuntimeError(f"Falha ao carregar {asset_config['name']} em {fbx_path_abs}")

    # Aplica rotações de correção antes de calcular bbox/escala para evitar modelos achatados ao rodar.
    rotated_bboxes = []
    raw_bboxes = []
    for mesh_data in model_data:
        rotated_vertices = _rotate_vertices(mesh_data[0], rotation_matrix)
        rotated_bboxes.append(compute_bounding_box(rotated_vertices, mesh_data[3]))
        raw_bboxes.append(compute_bounding_box(mesh_data[0], mesh_data[3]))

    bbox = rotated_bboxes[0]
    for mesh_bbox in rotated_bboxes[1:]:
        bbox = union_bounding_boxes(bbox, mesh_bbox)

    local_bbox = raw_bboxes[0]
    for mesh_bbox in raw_bboxes[1:]:
        local_bbox = union_bounding_boxes(local_bbox, mesh_bbox)

    bbox_extents = bbox[1] - bbox[0]
    local_extents = local_bbox[1] - local_bbox[0]
    rotated_height = max(bbox_extents[1], 1e-3)
    target_height = asset_config.get("target_height", 2.0)
    target_width = asset_config.get("target_width", None)

    scale_y = target_height / rotated_height
    if target_width is not None:
        max_horizontal = max(bbox_extents[0], bbox_extents[2], 1e-3)
        scale_xz = target_width / max_horizontal
    else:
        scale_xz = scale_y

    scale_vec = glm.vec3(scale_xz, scale_y, scale_xz)
    hover_offset = float(asset_config.get("hover_offset", 0.0))
    base_offset = -bbox[0][1] * scale_y + hover_offset
    visual_height = rotated_height * scale_y
    wing_center = get_bounding_box_center(local_bbox)

    texture_cache = {}
    meshes = []
    for mesh in model_data:
        # FBX_utils may now return [pos, normals, uvs, faces, faces_normals, texture_paths, diffuse_color, face_materials, material_textures, material_diffuse_colors]
        face_materials = None
        material_textures = None
        material_diffuse_colors = None
        if len(mesh) >= 10:
            (
                vertices_pos,
                vertices_normals,
                vertices_uvs,
                faces,
                _,
                texture_paths,
                diffuse_color,
                face_materials,
                material_textures,
                material_diffuse_colors,
            ) = mesh[:10]
        elif len(mesh) >= 7:
            vertices_pos, vertices_normals, vertices_uvs, faces, _, texture_paths, diffuse_color = mesh[:7]
        elif len(mesh) >= 6:
            vertices_pos, vertices_normals, vertices_uvs, faces, _, texture_paths = mesh[:6]
            diffuse_color = [1.0, 1.0, 1.0]
        else:
            vertices_pos, vertices_normals, vertices_uvs, faces, _, texture_paths = mesh
            diffuse_color = [1.0, 1.0, 1.0]

        # Fallback de UV apenas para assets problematicos (mantem textura mesmo sem UV original)
        if len(vertices_uvs) == 0 and any(tag in name_lower for tag in ("futuristic car", "alien animal", "qishilong")):
            vertices_uvs = _generate_planar_uvs(vertices_pos)
        # name_lower ja resolvido no inicio do carregamento

        # Agrupa as faces por material para usar a textura correta (olhos/boca/escamas).
        if face_materials is None or len(face_materials) == 0:
            face_materials = np.zeros(len(faces), dtype=np.int32)
        faces_np = np.asarray(faces)
        mat_indices = np.asarray(face_materials).flatten() if face_materials is not None else np.zeros(len(faces_np), dtype=np.int32)
        unique_materials = sorted(set(int(idx) for idx in mat_indices)) if len(mat_indices) else [0]

        for mat_idx in unique_materials:
            selected = np.where(mat_indices == mat_idx)[0] if len(mat_indices) else np.arange(len(faces_np))
            if selected.size == 0:
                continue
            selected_faces = faces_np[selected]

            v_pos, v_normals, v_uvs = flatten_geometry(vertices_pos, vertices_normals, vertices_uvs, selected_faces)
            vao_id = create_vao(
                [
                    (v_pos, 0),
                    (v_normals, 1),
                    (v_uvs, 2),
                ]
            )

            # Seleciona candidatas de textura para o material atual.
            material_candidates = texture_paths
            if material_textures and 0 <= mat_idx < len(material_textures):
                material_candidates = material_textures[mat_idx] or texture_paths

            material_candidates = apply_texture_hints(asset_config, material_candidates, asset_dir)
            if any(tag in name_lower for tag in ("futuristic car", "alien animal", "qishilong")):
                forced = [
                    os.path.join(asset_dir, "textures", "T_M_B_44_Qishilong_body01_B.png"),
                    os.path.join(asset_dir, "textures", "T_M_B_44_Qishilong_body02_B.png"),
                    os.path.join(asset_dir, "T_M_B_44_Qishilong_body01_B.png"),
                    os.path.join(asset_dir, "T_M_B_44_Qishilong_body02_B.png"),
                    "T_M_B_44_Qishilong_body01_B.png",
                    "T_M_B_44_Qishilong_body02_B.png",
                    os.path.join(asset_dir, "textures", "Base Color.jpg"),
                    os.path.join(asset_dir, "textures", "Diffuse_2.jpg"),
                    "Base Color.jpg",
                    "Diffuse_2.jpg",
                ]
                material_candidates = list(dict.fromkeys(forced + _filter_diffuse_candidates(material_candidates)))
            elif name_lower == "wolf":
                forced = [
                    os.path.join(asset_dir, "textures", "Wolf_Body.jpg"),
                    "Wolf_Body.jpg",
                ]
                material_candidates = list(dict.fromkeys(forced + material_candidates))
            elif "pikachu" in name_lower:
                forced = [
                    os.path.join(asset_dir, "images", "pm0025_00_BodyA1.png"),
                    os.path.join(asset_dir, "images", "pm0025_00_BodyB1.png"),
                    "pm0025_00_BodyA1.png",
                    "pm0025_00_BodyB1.png",
                    os.path.join(asset_dir, "images", "pm0025_00_Eye1.png"),
                    "pm0025_00_Eye1.png",
                    os.path.join(asset_dir, "images", "pm0025_00_Mouth1.png"),
                    "pm0025_00_Mouth1.png",
                ]
                material_candidates = list(dict.fromkeys(forced + material_candidates))
            elif "dragon" in name_lower:
                forced = [
                    os.path.join(asset_dir, "textures", "Dragon_ground_color.jpg"),
                    "Dragon_ground_color.jpg",
                    os.path.join(asset_dir, "textures", "Dragon_Bump_Col2.jpg"),
                    "Dragon_Bump_Col2.jpg",
                ]
                material_candidates = list(dict.fromkeys(forced + material_candidates))

            texture_id = None
            resolved = resolve_texture_path(
                material_candidates,
                base_dir=asset_dir,
                asset_name=asset_config.get("name"),
            )
            if resolved:
                if resolved in texture_cache:
                    texture_id = texture_cache[resolved]
                else:
                    texture_id = create_texture(resolved, repeat=True)
                    texture_cache[resolved] = texture_id
            else:
                # Para Alien Animal nao criamos fallback; demais recebem cor solida
                if "alien animal" not in name_lower:
                    mat_color = diffuse_color
                    if material_diffuse_colors and 0 <= mat_idx < len(material_diffuse_colors):
                        mat_color = material_diffuse_colors[mat_idx]
                    texture_id = create_solid_color_texture(mat_color)
                    _report_missing_texture(asset_config.get("name", "asset"), mat_idx, material_candidates)

            mat_color_vec = glm.vec3(*diffuse_color)
            if material_diffuse_colors and 0 <= mat_idx < len(material_diffuse_colors):
                mat_color_vec = glm.vec3(*material_diffuse_colors[mat_idx])

            meshes.append(
                {
                    "vao": vao_id,
                    "vertex_count": len(v_pos),
                    "texture_id": texture_id,
                    "diffuse_color": mat_color_vec,
                }
            )

    return {
        "name": asset_config["name"],
        "color": glm.vec3(*asset_config["color"]),
        "scale": scale_y,
        "scale_vec": scale_vec,
        "base_offset": base_offset,
        "visual_height": visual_height,
        "meshes": meshes,
        "yaw_offset_rad": yaw_off_rad,
        "pitch_offset_rad": pitch_off_rad,
        "roll_offset_rad": roll_off_rad,
        "facing_correction_rad": facing_correction_rad,
        "wing_enabled": bool(asset_config.get("wing_flap", False)),
        "wing_center": glm.vec3(*wing_center.tolist()),
        "wing_size": glm.vec3(*local_extents.tolist()),
        "wing_amplitude": float(asset_config.get("wing_amplitude", 0.0)),
        "wing_frequency": float(asset_config.get("wing_frequency", 0.0)),
    }


def spawn_crowd_instances():
    global crowd_instances

    crowd_instances = []
    random.seed(1337)
    half = TERRAIN_SIZE * 0.5 - 8.0

    for asset in fbx_assets:
        for _ in range(NUM_INSTANCES_PER_CHARACTER):
            orbit_radius = random.uniform(6.0, 18.0)
            center_x = random.uniform(-half + orbit_radius, half - orbit_radius)
            center_z = random.uniform(-half + orbit_radius, half - orbit_radius)
            crowd_instances.append(
                {
                    "asset": asset,
                    "orbit_center": glm.vec2(center_x, center_z),
                    "orbit_radius": orbit_radius,
                    "orbit_speed": random.uniform(0.15, 0.35),
                    "phase_offset": random.uniform(0.0, math.tau),
                    "bob_amplitude": random.uniform(0.05, 0.25),
                    "bob_speed": random.uniform(0.8, 1.8),
                    "scale_variation": random.uniform(0.9, 1.1),
                    "model_matrix": glm.mat4(1.0),
                    "normal_matrix": glm.mat3(1.0),
                }
            )


def update_crowd_instances(_delta_time):
    if not crowd_instances:
        return

    time_now = glfw.get_time()
    half = TERRAIN_SIZE * 0.5 - 2.0

    for instance in crowd_instances:
        asset_name = instance["asset"]["name"].lower()
        is_skull = any(key in asset_name for key in ("future car", "craneo", "skull", "caveira"))

        if is_skull:
            angle = instance["phase_offset"]
            bob = 0.0
            sway_speed = 0.0
            sway_amp_pitch = 0.0
            sway_amp_roll = 0.0
        else:
            angle = instance["phase_offset"] + time_now * instance["orbit_speed"]
            bob = math.sin(time_now * instance["bob_speed"] + instance["phase_offset"]) * instance["bob_amplitude"]
            sway_speed = 1.4
            sway_amp_pitch = 0.08
            sway_amp_roll = 0.05
        x = instance["orbit_center"].x + math.cos(angle) * instance["orbit_radius"]
        z = instance["orbit_center"].y + math.sin(angle) * instance["orbit_radius"]
        x = max(-half, min(half, x))
        z = max(-half, min(half, z))

        ground = get_heightmap_height(terrain_heightmap, terrain_bbox, x, z)
        scale_factor = instance.get("scale_variation", 1.0)
        base_offset = instance["asset"]["base_offset"] * scale_factor
        y = ground + base_offset + bob

        direction = glm.vec3(-math.sin(angle), 0.0, math.cos(angle))
        yaw = math.atan2(direction.x, direction.z) + instance["asset"].get("yaw_offset_rad", 0.0)
        yaw += instance["asset"].get("facing_correction_rad", 0.0)

        translation = glm.translate(glm.mat4(1.0), glm.vec3(x, y, z))
        rotation_yaw = glm.rotate(glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0))
        rotation_pitch_off = glm.rotate(glm.mat4(1.0), instance["asset"].get("pitch_offset_rad", 0.0), glm.vec3(1.0, 0.0, 0.0))
        rotation_roll_off = glm.rotate(glm.mat4(1.0), instance["asset"].get("roll_offset_rad", 0.0), glm.vec3(0.0, 0.0, 1.0))

        # anima??o simples: leve oscila??o de pitch/roll para quebrar rigidez (desativada para caveiras)
        pitch = math.sin(time_now * sway_speed + instance["phase_offset"]) * sway_amp_pitch if sway_speed > 0.0 else 0.0
        roll = math.cos(time_now * sway_speed * 0.8 + instance["phase_offset"] * 1.3) * sway_amp_roll if sway_speed > 0.0 else 0.0

        rotation_pitch = glm.rotate(glm.mat4(1.0), pitch, glm.vec3(1.0, 0.0, 0.0))
        rotation_roll = glm.rotate(glm.mat4(1.0), roll, glm.vec3(0.0, 0.0, 1.0))

        scale_vec = instance["asset"].get("scale_vec", glm.vec3(instance["asset"]["scale"])) * scale_factor
        scale_matrix = glm.scale(glm.mat4(1.0), scale_vec)
        model = translation * rotation_yaw * rotation_pitch_off * rotation_roll_off * rotation_pitch * rotation_roll * scale_matrix

        instance["model_matrix"] = model
        instance["normal_matrix"] = glm.mat3(glm.transpose(glm.inverse(model)))

def set_controlled_character(index):
    global player_state

    if not fbx_assets:
        return

    clamped_index = max(0, min(index, len(fbx_assets) - 1))
    asset = fbx_assets[clamped_index]
    player_state["asset_index"] = clamped_index
    player_state["asset"] = asset
    player_state["base_offset"] = asset["base_offset"]
    # Orienta o personagem para "frente" logo ao carregar (mesma direção que W usa)
    player_state["yaw_rad"] = ORIENTATION_CORRECTION_RAD + asset.get("yaw_offset_rad", 0.0)
    player_state["camera_anchor_height"] = max(1.6, asset["visual_height"] * 0.9)
    rebuild_player_matrices()


def cycle_character(step):
    new_index = (player_state["asset_index"] + step) % len(fbx_assets)
    set_controlled_character(new_index)


def rebuild_player_matrices():
    asset = player_state.get("asset")
    if asset is None:
        return

    # sem animação automática no jogador controlado
    pitch = 0.0
    roll = 0.0

    translation = glm.translate(
        glm.mat4(1.0),
        glm.vec3(player_state["position"].x, player_state["position"].y + player_state["base_offset"], player_state["position"].z),
    )
    rotation_yaw = glm.rotate(glm.mat4(1.0), player_state["yaw_rad"], glm.vec3(0.0, 1.0, 0.0))
    rotation_pitch_off = glm.rotate(glm.mat4(1.0), player_state["asset"].get("pitch_offset_rad", 0.0), glm.vec3(1.0, 0.0, 0.0))
    rotation_roll_off = glm.rotate(glm.mat4(1.0), player_state["asset"].get("roll_offset_rad", 0.0), glm.vec3(0.0, 0.0, 1.0))
    rotation_pitch = glm.rotate(glm.mat4(1.0), pitch, glm.vec3(1.0, 0.0, 0.0))
    rotation_roll = glm.rotate(glm.mat4(1.0), roll, glm.vec3(0.0, 0.0, 1.0))
    scale_vec = asset.get("scale_vec", glm.vec3(asset["scale"]))
    scale_matrix = glm.scale(glm.mat4(1.0), scale_vec)
    model = translation * rotation_yaw * rotation_pitch_off * rotation_roll_off * rotation_pitch * rotation_roll * scale_matrix
    normal_matrix = glm.mat3(glm.transpose(glm.inverse(model)))

    player_state["model_matrix"] = model
    player_state["normal_matrix"] = normal_matrix


def setup_shadow_resources():
    global shadow_fbo, shadow_texture

    # Req 3c: shadow mapping avancado para projetar sombras do sol
    shadow_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, shadow_texture)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_DEPTH_COMPONENT,
        SHADOW_MAP_RESOLUTION,
        SHADOW_MAP_RESOLUTION,
        0,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        None,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    border_color = (GLfloat * 4)(1.0, 1.0, 1.0, 1.0)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

    shadow_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_texture, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def update_sun_state(hours_of_day: float):
    global sun_direction, sun_color, sky_color, fog_color, sun_position, light_space_matrix, sun_light_factor, sun_fog_density, sun_ambient_color

    # Req 3a: sol orbita de leste (x+) para oeste (x-) em 24h simuladas (~60s reais).
    angle = math.tau * ((hours_of_day - 6.0) / 24.0)  # 06h no leste (x+), 18h no oeste (x-)

    sun_position = glm.vec3(
        math.cos(angle) * SUN_ORBIT_RADIUS,
        math.sin(angle) * SUN_ORBIT_RADIUS * 0.65,
        math.sin(angle * 0.35) * SUN_ORBIT_RADIUS * 0.3,
    )
    sun_direction = glm.normalize(-sun_position)

    day_sky = glm.vec3(0.35, 0.6, 0.9)
    dusk_sky = glm.vec3(0.95, 0.45, 0.25)
    night_sky = glm.vec3(0.04, 0.05, 0.12)

    # Transicoes suaves: noite -> alvorada -> dia -> entardecer -> noite
    altitude = -sun_direction.y  # positivo quando o sol esta acima, negativo abaixo do horizonte
    dawn_mix = _smoothstep(-0.25, 0.05, altitude)
    day_mix = _smoothstep(0.05, 0.45, altitude)
    sun_light_factor = day_mix
    # Req 3b: cor do ceu (sky_color) muda conforme hora do dia simulada.
    sky_color = glm.mix(night_sky, dusk_sky, dawn_mix)
    sky_color = glm.mix(sky_color, day_sky, day_mix)
    # Req 1c: fog (cor/densidade) ajustado pelo horario.
    fog_color = glm.mix(night_sky, day_sky, day_mix)

    sun_color = glm.mix(glm.vec3(1.0, 0.45, 0.25), glm.vec3(1.0, 0.98, 0.9), day_mix)
    sun_fog_density = 0.002 - 0.0015 * day_mix
    sun_ambient_color = glm.mix(glm.vec3(0.08, 0.08, 0.1), glm.vec3(0.4, 0.4, 0.42), day_mix)

    light_target = scene_center
    light_view = glm.lookAt(-sun_direction * SUN_ORBIT_RADIUS, light_target, glm.vec3(0.0, 1.0, 0.0))
    extent = TERRAIN_SIZE
    light_projection = glm.ortho(-extent, extent, -extent, extent, -1200.0, 1200.0)
    light_space_matrix = light_projection * light_view

    # Req 3b: aplica cor do ceu calculada ao clear da cena.
    glClearColor(sky_color.x, sky_color.y, sky_color.z, 1.0)


# ---- Clock overlay (simulated time HUD) ----
CLOCK_FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    ":": ["000", "010", "000", "010", "000"],
}
CLOCK_CELL_PX = 8
CLOCK_MARGIN_PX = 12


def _clock_text_width_px(text: str, cell_px: int = CLOCK_CELL_PX, gap_px: int = 2) -> int:
    width = 0
    first = True
    for ch in text:
        pattern = CLOCK_FONT.get(ch)
        if not pattern:
            continue
        if not first:
            width += gap_px
        width += len(pattern[0]) * cell_px
        first = False
    return width


def _clock_vertices(text: str) -> np.ndarray:
    cell = CLOCK_CELL_PX
    gap_px = 2
    text_width = _clock_text_width_px(text, cell, gap_px)
    start_x = window_size[0] - CLOCK_MARGIN_PX - text_width
    start_y = CLOCK_MARGIN_PX

    verts: list[list[float]] = []
    cursor_x = start_x
    for ch in text:
        pattern = CLOCK_FONT.get(ch)
        if not pattern:
            continue
        char_w = len(pattern[0])
        for r, row in enumerate(pattern):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = cursor_x + c * cell
                y0 = start_y + r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                x0_ndc = (x0 / (window_size[0] * 0.5)) - 1.0
                x1_ndc = (x1 / (window_size[0] * 0.5)) - 1.0
                y0_ndc = 1.0 - (y0 / (window_size[1] * 0.5))
                y1_ndc = 1.0 - (y1 / (window_size[1] * 0.5))
                verts.extend(
                    [
                        [x0_ndc, y0_ndc],
                        [x1_ndc, y0_ndc],
                        [x1_ndc, y1_ndc],
                        [x0_ndc, y0_ndc],
                        [x1_ndc, y1_ndc],
                        [x0_ndc, y1_ndc],
                    ]
                )
        cursor_x += (char_w * cell) + gap_px

    return np.array(verts, dtype=np.float32)


def _ensure_clock_resources():
    global clock_shader, clock_vao, clock_vbo
    if clock_shader is not None:
        return
    vertex_src = """
    #version 330 core
    layout(location = 0) in vec2 aPos;
    void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
    """
    fragment_src = """
    #version 330 core
    uniform vec3 uColor;
    out vec4 FragColor;
    void main() { FragColor = vec4(uColor, 1.0); }
    """
    clock_shader = gls.compileProgram(
        gls.compileShader(vertex_src, GL_VERTEX_SHADER),
        gls.compileShader(fragment_src, GL_FRAGMENT_SHADER),
    )
    clock_vao = glGenVertexArrays(1)
    clock_vbo = glGenBuffers(1)
    glBindVertexArray(clock_vao)
    glBindBuffer(GL_ARRAY_BUFFER, clock_vbo)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)


def _format_clock(hours_of_day: float) -> str:
    h = int(hours_of_day) % 24
    m = int((hours_of_day % 1.0) * 60.0)
    return f"{h:02d}:{m:02d}"


def render_clock_overlay(hours_of_day: float):
    _ensure_clock_resources()
    text = _format_clock(hours_of_day)
    vertices = _clock_vertices(text)
    if vertices.size == 0:
        return

    glUseProgram(clock_shader)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBindVertexArray(clock_vao)
    glBindBuffer(GL_ARRAY_BUFFER, clock_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glUniform3f(glGetUniformLocation(clock_shader, "uColor"), 1.0, 1.0, 1.0)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices))

    glBindVertexArray(0)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glUseProgram(0)


def update_camera_from_player():
    global camera_pos, camera_front

    # Req 4: camera acompanha jogador usando yaw/pitch do mouse (sensacao de primeira pessoa)
    target = player_state["position"] + glm.vec3(0.0, player_state["camera_anchor_height"], 0.0)
    yaw_rad = math.radians(camera_orbit["yaw"])
    pitch_rad = math.radians(glm.clamp(camera_orbit["pitch"], CAMERA_MIN_PITCH, CAMERA_MAX_PITCH))

    offset = glm.vec3(
        math.cos(pitch_rad) * math.cos(yaw_rad),
        math.sin(pitch_rad),
        math.cos(pitch_rad) * math.sin(yaw_rad),
    ) * camera_orbit["distance"]

    desired_position = target - offset
    terrain_height = get_heightmap_height(terrain_heightmap, terrain_bbox, desired_position.x, desired_position.z)
    min_camera_height = terrain_height + CAMERA_SAFETY_HEIGHT
    corrected_y = max(desired_position.y, min_camera_height)

    camera_pos = glm.vec3(desired_position.x, corrected_y, desired_position.z)
    camera_front = glm.normalize(target - camera_pos)


def handle_character_shortcuts(key, action):
    if action != glfw.PRESS:
        return
    if key == CHARACTER_SWITCH_KEY:
        cycle_character(1)
    elif key in CHARACTER_DIRECT_KEYS:
        idx = CHARACTER_DIRECT_KEYS.index(key)
        set_controlled_character(idx)


def key_callback(window, key, scancode, action, mods):
    global cursor_first_update

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        return

    if key == glfw.KEY_TAB and action == glfw.PRESS:
        current_mode = glfw.get_input_mode(window, glfw.CURSOR)
        new_mode = glfw.CURSOR_NORMAL if current_mode == glfw.CURSOR_DISABLED else glfw.CURSOR_DISABLED
        glfw.set_input_mode(window, glfw.CURSOR, new_mode)
        cursor_first_update = True
        return

    handle_character_shortcuts(key, action)
    key_states[key] = action in (glfw.PRESS, glfw.REPEAT)


def mouse_callback(window, xpos, ypos):
    global cursor_first_update, last_cursor_x, last_cursor_y

    if glfw.get_input_mode(window, glfw.CURSOR) == glfw.CURSOR_NORMAL:
        cursor_first_update = True
        return

    if cursor_first_update:
        last_cursor_x = xpos
        last_cursor_y = ypos
        cursor_first_update = False
        return

    dx = xpos - last_cursor_x
    dy = last_cursor_y - ypos
    last_cursor_x = xpos
    last_cursor_y = ypos

    camera_orbit["yaw"] += dx * PLAYER_MOUSE_SENSITIVITY
    camera_orbit["pitch"] += dy * PLAYER_MOUSE_SENSITIVITY
    camera_orbit["pitch"] = glm.clamp(camera_orbit["pitch"], CAMERA_MIN_PITCH, CAMERA_MAX_PITCH)


def framebuffer_size_callback(window, width, height):
    global window_size
    width = max(1, width)
    height = max(1, height)
    window_size = [width, height]
    glViewport(0, 0, width, height)


def update_player(delta_time):
    asset = player_state.get("asset")
    if asset is None:
        return

    # Req 4: movimento FPS com W/S/A/D, corrida (shift), virar pelo mouse e pulo (espaco)
    yaw_rad = math.radians(camera_orbit["yaw"])
    forward = glm.normalize(glm.vec3(math.cos(yaw_rad), 0.0, math.sin(yaw_rad)))
    right = glm.normalize(glm.cross(forward, glm.vec3(0.0, 1.0, 0.0)))

    move = glm.vec3(0.0, 0.0, 0.0)
    if key_states.get(glfw.KEY_W, False):
        move += forward
    if key_states.get(glfw.KEY_S, False):
        move -= forward
    if key_states.get(glfw.KEY_A, False):
        move -= right
    if key_states.get(glfw.KEY_D, False):
        move += right

    if glm.length(move) > 0.0:
        player_state["is_moving"] = True
        speed = PLAYER_WALK_SPEED * delta_time
        if key_states.get(glfw.KEY_LEFT_SHIFT, False):
            speed *= PLAYER_RUN_MULTIPLIER
        move = glm.normalize(move) * speed
        yaw_correction = asset.get("yaw_offset_rad", 0.0) + asset.get("facing_correction_rad", 0.0)
        player_state["yaw_rad"] = math.atan2(move.x, move.z) + yaw_correction
    else:
        player_state["is_moving"] = False

    new_x = player_state["position"].x + move.x
    new_z = player_state["position"].z + move.z
    half = TERRAIN_SIZE * 0.5 - 2.0
    new_x = max(-half, min(half, new_x))
    new_z = max(-half, min(half, new_z))

    if key_states.get(glfw.KEY_SPACE, False) and player_state["jump_offset"] == 0.0:
        player_state["vertical_velocity"] = PLAYER_JUMP_SPEED

    player_state["vertical_velocity"] += GRAVITY * delta_time
    player_state["jump_offset"] = max(0.0, player_state["jump_offset"] + player_state["vertical_velocity"] * delta_time)
    if player_state["jump_offset"] == 0.0:
        player_state["vertical_velocity"] = 0.0

    ground = get_heightmap_height(terrain_heightmap, terrain_bbox, new_x, new_z)
    player_state["position"] = glm.vec3(new_x, ground + player_state["jump_offset"], new_z)
    rebuild_player_matrices()


def get_view_matrix():
    return glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)


def get_projection_matrix():
    aspect_ratio = window_size[0] / float(window_size[1])
    return glm.perspective(glm.radians(FIELD_OF_VIEW), aspect_ratio, NEAR_PLANE, FAR_PLANE)


def _project_to_ndc(world_pos, view, projection):
    clip = projection * view * glm.vec4(world_pos, 1.0)
    if clip.w <= 0.0:
        return None
    ndc = clip / clip.w
    if ndc.z > 1.0 or ndc.z < 0.0:
        return None
    return glm.vec3(ndc.x, ndc.y, ndc.z)


def render_sun_billboard(projection, view):
    if sun_shader is None or sun_quad_vao is None:
        return

    center_ndc = _project_to_ndc(sun_position, view, projection)
    if center_ndc is None:
        return

    glUseProgram(sun_shader)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_ONE, GL_ONE)

    aspect = window_size[0] / float(max(1, window_size[1]))
    glUniform2f(glGetUniformLocation(sun_shader, "center"), center_ndc.x, center_ndc.y)
    glUniform1f(glGetUniformLocation(sun_shader, "radius"), 0.08)
    glUniform1f(glGetUniformLocation(sun_shader, "aspect"), aspect)
    glUniform3fv(glGetUniformLocation(sun_shader, "sunColor"), 1, glm.value_ptr(sun_color))

    glBindVertexArray(sun_quad_vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glBindVertexArray(0)

    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glUseProgram(0)


def set_wing_uniforms(program, asset):
    """Configura uniforms para o efeito de batimento de asas (apenas para o dragao)."""
    use_wings = int(asset.get("wing_enabled", False)) if asset else 0
    center = asset.get("wing_center", glm.vec3(0.0)) if asset else glm.vec3(0.0)
    size = asset.get("wing_size", glm.vec3(1.0)) if asset else glm.vec3(1.0)
    amplitude = float(asset.get("wing_amplitude", 0.0)) if asset else 0.0
    frequency = float(asset.get("wing_frequency", 0.0)) if asset else 0.0

    glUniform1i(glGetUniformLocation(program, "useWingFlap"), use_wings)
    glUniform3fv(glGetUniformLocation(program, "wingCenter"), 1, glm.value_ptr(center))
    glUniform3fv(glGetUniformLocation(program, "wingSize"), 1, glm.value_ptr(size))
    glUniform1f(glGetUniformLocation(program, "wingAmplitude"), amplitude)
    glUniform1f(glGetUniformLocation(program, "wingFrequency"), frequency)


def draw_mesh(mesh_entry, model_matrix, normal_matrix, color, use_texture, asset=None):
    glUniformMatrix4fv(glGetUniformLocation(main_shader, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))
    glUniformMatrix3fv(glGetUniformLocation(main_shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normal_matrix))
    glUniform3fv(glGetUniformLocation(main_shader, "objectColor"), 1, glm.value_ptr(color))
    glUniform1i(glGetUniformLocation(main_shader, "useTexture"), int(use_texture and mesh_entry["texture_id"] is not None))
    set_wing_uniforms(main_shader, asset)

    glActiveTexture(GL_TEXTURE0)
    if use_texture and mesh_entry["texture_id"]:
        glBindTexture(GL_TEXTURE_2D, mesh_entry["texture_id"])
    else:
        glBindTexture(GL_TEXTURE_2D, 0)

    glBindVertexArray(mesh_entry["vao"])
    glDrawArrays(GL_TRIANGLES, 0, mesh_entry["vertex_count"])
    glBindVertexArray(0)


def render_depth_pass(time_now: float):
    # Req 3c: primeiro passe de profundidade para gerar shadow map do sol
    glViewport(0, 0, SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
    glClear(GL_DEPTH_BUFFER_BIT)

    glUseProgram(depth_shader)
    glUniformMatrix4fv(glGetUniformLocation(depth_shader, "lightSpaceMatrix"), 1, GL_FALSE, glm.value_ptr(light_space_matrix))
    glUniform1f(glGetUniformLocation(depth_shader, "time"), time_now)

    set_wing_uniforms(depth_shader, None)
    glUniformMatrix4fv(glGetUniformLocation(depth_shader, "model"), 1, GL_FALSE, glm.value_ptr(terrain_renderable["model_matrix"]))
    glBindVertexArray(terrain_renderable["vao"])
    glDrawArrays(GL_TRIANGLES, 0, terrain_renderable["vertex_count"])
    glBindVertexArray(0)

    for instance in crowd_instances:
        set_wing_uniforms(depth_shader, instance["asset"])
        glUniformMatrix4fv(glGetUniformLocation(depth_shader, "model"), 1, GL_FALSE, glm.value_ptr(instance["model_matrix"]))
        for mesh in instance["asset"]["meshes"]:
            glBindVertexArray(mesh["vao"])
            glDrawArrays(GL_TRIANGLES, 0, mesh["vertex_count"])

    if player_state["asset"]:
        set_wing_uniforms(depth_shader, player_state["asset"])
        glUniformMatrix4fv(glGetUniformLocation(depth_shader, "model"), 1, GL_FALSE, glm.value_ptr(player_state["model_matrix"]))
        for mesh in player_state["asset"]["meshes"]:
            glBindVertexArray(mesh["vao"])
            glDrawArrays(GL_TRIANGLES, 0, mesh["vertex_count"])

    glBindVertexArray(0)
    glUseProgram(0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def render_main_pass(time_now: float):
    glViewport(0, 0, window_size[0], window_size[1])
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(main_shader)
    glUniform1f(glGetUniformLocation(main_shader, "time"), time_now)

    projection = get_projection_matrix()
    view = get_view_matrix()

    glUniformMatrix4fv(glGetUniformLocation(main_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(glGetUniformLocation(main_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(main_shader, "lightSpaceMatrix"), 1, GL_FALSE, glm.value_ptr(light_space_matrix))
    glUniform3fv(glGetUniformLocation(main_shader, "viewPos"), 1, glm.value_ptr(camera_pos))
    glUniform3fv(glGetUniformLocation(main_shader, "sunDirection"), 1, glm.value_ptr(sun_direction))
    glUniform3fv(glGetUniformLocation(main_shader, "lightColor"), 1, glm.value_ptr(sun_color))
    glUniform3fv(glGetUniformLocation(main_shader, "skyColor"), 1, glm.value_ptr(sky_color))
    glUniform3fv(glGetUniformLocation(main_shader, "fogColor"), 1, glm.value_ptr(fog_color))
    glUniform3fv(glGetUniformLocation(main_shader, "ambientColor"), 1, glm.value_ptr(sun_ambient_color))
    glUniform1f(glGetUniformLocation(main_shader, "shininess"), 32.0)
    glUniform1f(glGetUniformLocation(main_shader, "fogDensity"), sun_fog_density)  # Req 1c: fog no fragment shader
    glUniform1f(glGetUniformLocation(main_shader, "fogBaseDensity"), sun_fog_density)
    glUniform1f(glGetUniformLocation(main_shader, "fogHeightFalloff"), 0.05)
    glUniform1f(glGetUniformLocation(main_shader, "fogAnisotropy"), 0.2)
    glUniform1i(glGetUniformLocation(main_shader, "fogSteps"), 8)
    glUniform1f(glGetUniformLocation(main_shader, "lightRadiusUV"), 2.5 / float(SHADOW_MAP_RESOLUTION))
    glUniform1f(glGetUniformLocation(main_shader, "minBias"), 0.0004)
    glUniform1f(glGetUniformLocation(main_shader, "maxBias"), 0.002)
    glUniform1i(glGetUniformLocation(main_shader, "blockerSamples"), 8)
    glUniform1i(glGetUniformLocation(main_shader, "pcfSamples"), 16)
    glUniform1i(glGetUniformLocation(main_shader, "diffuseMap"), 0)

    glActiveTexture(GL_TEXTURE1)
    # Req 3c: usa shadow map calculado no passe de profundidade para projetar sombras.
    glBindTexture(GL_TEXTURE_2D, shadow_texture)
    glUniform1i(glGetUniformLocation(main_shader, "shadowMap"), 1)

    draw_mesh(
        terrain_renderable,
        terrain_renderable["model_matrix"],
        terrain_renderable["normal_matrix"],
        glm.vec3(1.0, 1.0, 1.0),
        True,
        None,
    )

    for instance in crowd_instances:
        asset = instance["asset"]
        for mesh in asset["meshes"]:
            draw_mesh(
                mesh,
                instance["model_matrix"],
                instance["normal_matrix"],
                asset["color"],
                True,
                asset,
            )

    if player_state["asset"]:
        for mesh in player_state["asset"]["meshes"]:
            draw_mesh(
                mesh,
                player_state["model_matrix"],
                player_state["normal_matrix"],
                player_state["asset"]["color"],
                True,
                player_state["asset"],
            )

    render_sun_billboard(projection, view)

    glBindVertexArray(0)
    glUseProgram(0)


def initialize_player():
    ground = get_heightmap_height(terrain_heightmap, terrain_bbox, 0.0, 5.0)
    player_state["position"] = glm.vec3(0.0, ground, 5.0)
    player_state["yaw_rad"] = math.radians(camera_orbit["yaw"])
    player_state["vertical_velocity"] = 0.0
    player_state["jump_offset"] = 0.0
    set_controlled_character(CONTROLLED_CHARACTER_START_INDEX)
    rebuild_player_matrices()
    update_camera_from_player()


def main():
    global window, main_shader, depth_shader, sun_shader, terrain_renderable, fbx_assets, sun_quad_vao

    if not glfw.init():
        raise RuntimeError("Falha ao iniciar GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == "darwin":
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "A3 - Mundo Procedural 3a Pessoa", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Nao foi possivel criar a janela GLFW")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    main_shader = compile_shader_program("05 - advanced_world_vs.glsl", "05 - advanced_world_fs.glsl")
    sun_shader = compile_shader_program("05 - sun_billboard_vs.glsl", "05 - sun_billboard_fs.glsl")
    depth_shader = compile_shader_program("05 - depth_shader_vs.glsl", "05 - depth_shader_fs.glsl")
    sun_quad_vao = create_sun_billboard_resources()

    terrain_renderable = create_terrain_renderable()

    fbx_assets = []
    for cfg in FBX_CHARACTERS:
        try:
            asset = load_fbx_asset(cfg)
            fbx_assets.append(asset)
        except Exception as exc:
            print(f"[WARN] Nao foi possivel carregar {cfg.get('name')} ({cfg.get('path')}): {exc}")

    if not fbx_assets:
        raise RuntimeError("Nenhum FBX carregado. Verifique caminhos e formatos.")
    if len(fbx_assets) < len(FBX_CHARACTERS):
        print(f"[INFO] Carregados {len(fbx_assets)} de {len(FBX_CHARACTERS)}. Confira caminhos e versoes dos arquivos restantes.")

    spawn_crowd_instances()
    update_crowd_instances(0.0)
    setup_shadow_resources()
    initialize_player()

    last_time = glfw.get_time()
    global sim_hours
    sim_hours = 6.0
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        # Req 3a: avanca 1h simulada por minuto real para animar a trajetoria solar.
        sim_hours = (sim_hours + delta_time * SIM_HOURS_PER_REAL_SECOND) % 24.0

        glfw.poll_events()
        update_player(delta_time)
        update_crowd_instances(delta_time)
        update_camera_from_player()
        update_sun_state(sim_hours)

        render_depth_pass(current_time)
        render_main_pass(current_time)

        render_clock_overlay(sim_hours)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
