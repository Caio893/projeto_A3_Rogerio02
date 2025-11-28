"""Advanced OpenGL scene with procedural terrain, FBX crowd and FPS camera."""

from __future__ import annotations

from runtime_bootstrap import ensure_supported_runtime

ensure_supported_runtime()

import ctypes
import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence

import glfw
import glm
import numpy as np
try:
    from noise import pnoise2  # pyright: ignore[reportMissingImports]
except ImportError:
    from functools import lru_cache

    _FALLBACK_GRADS = (
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    @lru_cache(maxsize=64)
    def _permutation(seed: int) -> tuple[int, ...]:
        values = list(range(256))
        random.Random(seed).shuffle(values)
        return tuple(values + values)

    def _grad(hash_value: int, x: float, y: float) -> float:
        gx, gy = _FALLBACK_GRADS[hash_value & 7]
        return gx * x + gy * y

    def pnoise2(x: float, y: float, repeatx: int = 0, repeaty: int = 0, base: int = 0) -> float:
        """Fallback Perlin noise implementation used when the 'noise' package is absent."""
        # Wrap coordinates when repeat is requested to keep the terrain tilable.
        if repeatx > 0:
            x = (x % repeatx + repeatx) % repeatx
        if repeaty > 0:
            y = (y % repeaty + repeaty) % repeaty

        xi = math.floor(x)
        yi = math.floor(y)
        xf = x - xi
        yf = y - yi

        perm = _permutation(base & 0xFFFFFFFF)
        xi &= 255
        yi &= 255

        aa = perm[perm[xi] + yi]
        ab = perm[perm[xi] + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]

        u = _fade(xf)
        v = _fade(yf)

        x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
        x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
        return _lerp(x1, x2, v)

    print(
        "[info] Pacote 'noise' não encontrado. Usando implementação interna de Perlin noise. "
        "Instale 'pip install noise' para obter desempenho ideal."
    )
from PIL import Image
from OpenGL.GL import *
import OpenGL.GL.shaders as gls

from geometry_utils import compute_bounding_box
from FBX_utils import load_fbx_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != BASE_DIR:
    os.chdir(BASE_DIR)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FIELD_OF_VIEW = 70.0
NEAR_PLANE = 0.1
FAR_PLANE = 2000.0

TERRAIN_RESOLUTION = 256
TERRAIN_SIZE = 400.0  # meters
TERRAIN_HEIGHT = 9.0
TERRAIN_TEXTURE_REPEAT = 24.0
TERRAIN_TEXTURE_CANDIDATES = [
    os.path.join("Textures", "rocky_terrain_diff_1k.jpg"),
    os.path.join("Textures", "terrain_grass.png"),  # fallback
]
SELECTED_TERRAIN_TEXTURE_PATH: str | None = None

HEIGHTMAP_OCTAVES = 6
HEIGHTMAP_PERSISTENCE = 0.42
HEIGHTMAP_LACUNARITY = 2.1
HEIGHTMAP_SCALE = 3.6
TERRAIN_GENTLE_FACTOR = 0.28
TERRAIN_RIDGE_FACTOR = 0.6
TERRAIN_FLAT_CENTER_RADIUS = 0.32
TERRAIN_EDGE_FULL_RADIUS = 0.82

CHARACTER_INSTANCES = 20
CHARACTER_TARGET_HEIGHT = 2.1

SHADOW_MAP_SIZE = 4096

CAMERA_EYE_HEIGHT = 1.75
MOVE_SPEED = 6.0
RUN_MULTIPLIER = 1.8
JUMP_STRENGTH = 7.5
GRAVITY = -18.0
MOUSE_SENSITIVITY = 0.12
FACING_CORRECTION_RAD = math.pi  # Corrige modelos que estao voltados para tras

SIM_HOURS_PER_REAL_SECOND = 24.0 / 60.0  # ciclo completo de dia/noite em ~60s reais

CHARACTER_SPECS = [
    {
        "name": "Future Car (FBX sem textura)",
        "path": os.path.join("fbx_futureCar", "CraneoFBX.fbx"),
        "tint": glm.vec3(0.8, 0.85, 1.0),
        "yaw_offset_deg": 180.0,
        "pitch_offset_deg": -90.0,  # postura original (sem inclinar para frente)
        "roll_offset_deg": 0.0,
        "target_height": 1.2,       # caveira menor
        "hover_offset": -0.6,       # afunda mais para parecer enterrada
        "allow_texture_fallback": False,  # modelo novo sem texturas; usa apenas cor base/tinta
        "base_color": (1.0, 1.0, 1.0),
    },
    {
        "name": "Pikachu",
        "path": os.path.join("fbx_Lobo", "PikachuF.FBX"),
        "tint": glm.vec3(1.0, 0.95, 0.4),
        "yaw_offset_deg": 180.0,
        "pitch_offset_deg": -90.0,
        "roll_offset_deg": 0.0,
        "target_height": 1.2,
    },
    {
        "name": "Black Dragon",
        "path": os.path.join("fbx_blackDragon", "Dragon_Baked_Actions_fbx_7.4_binary.fbx"),
        "tint": glm.vec3(0.7, 0.7, 0.7),
        "yaw_offset_deg": 180.0,  # compensa a correcao global para o dragao andar para frente
        "pitch_offset_deg": -90.0,
        "roll_offset_deg": 0.0,
        "target_width": 9.0,
        "target_height": 4.4,
        "hover_offset": 4.0,
        "wing_flap": True,
        "wing_amplitude": 1.2,
        "wing_frequency": 3.2,
    },
    {
        "name": "Shun Gold",
        "path": os.path.join("fbx_Camel", "Shun gold.FBX"),
        "tint": glm.vec3(0.85, 0.8, 0.65),
        "yaw_offset_deg": 180.0,
        "pitch_offset_deg": 0.0,
        "roll_offset_deg": 180.0,
        "target_height": 2.0,
    },
]


@dataclass
class MeshResource:
    vao: int
    count: int
    use_elements: bool
    texture_id: int
    base_color: glm.vec3
    buffers: Sequence[int]
    ebo: int | None = None


@dataclass
class SceneInstance:
    meshes: Sequence[MeshResource]
    model_matrix: glm.mat4
    normal_matrix: glm.mat3
    tint: glm.vec3
    asset: CharacterAsset | None


@dataclass
class TerrainResource:
    mesh: MeshResource
    heightmap: np.ndarray
    texture_path: str
    size: float
    resolution: int
    min_bounds: np.ndarray
    max_bounds: np.ndarray

    def sample_height(self, x_world: float, z_world: float) -> float:
        half = 0.5 * self.size
        x = np.clip((x_world + half) / self.size, 0.0, 0.9999) * (self.resolution - 1)
        z = np.clip((z_world + half) / self.size, 0.0, 0.9999) * (self.resolution - 1)

        x0 = int(np.floor(x))
        z0 = int(np.floor(z))
        x1 = min(x0 + 1, self.resolution - 1)
        z1 = min(z0 + 1, self.resolution - 1)

        tx = x - x0
        tz = z - z0

        h00 = self.heightmap[z0, x0]
        h10 = self.heightmap[z0, x1]
        h01 = self.heightmap[z1, x0]
        h11 = self.heightmap[z1, x1]

        h_top = (1.0 - tx) * h00 + tx * h10
        h_bottom = (1.0 - tx) * h01 + tx * h11
        return (1.0 - tz) * h_top + tz * h_bottom


@dataclass
class ShadowMap:
    fbo: int
    texture: int
    size: int


sun_shader = None
sun_quad_vao = None
clock_shader = None
clock_vao = None
clock_vbo = None


@dataclass
class CharacterAsset:
    name: str
    meshes: Sequence[MeshResource]
    bbox: np.ndarray
    scale_vec: glm.vec3
    base_offset: float
    yaw_offset_rad: float
    pitch_offset_rad: float
    roll_offset_rad: float
    wing_enabled: bool
    wing_center: glm.vec3
    wing_size: glm.vec3
    wing_amplitude: float
    wing_frequency: float


def _build_rotation_matrix(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)

    rot_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rot_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_yaw @ rot_pitch @ rot_roll


def _rotate_vertices(vertices: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return (rotation_matrix @ vertices.T).T


class CameraController:
    def __init__(self, terrain: TerrainResource):
        center_height = terrain.sample_height(0.0, 0.0)
        self.position = glm.vec3(0.0, center_height + CAMERA_EYE_HEIGHT, 5.0)
        self.front = glm.normalize(glm.vec3(0.0, -0.05, -1.0))
        self.world_up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))
        self.yaw = -90.0
        self.pitch = -10.0
        self.vertical_velocity = 0.0
        self.height_offset = 0.0
        self.is_grounded = True
        self.terrain = terrain

    def update_vectors(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        front = glm.vec3(
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
        )
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def clamp_position(self):
        half = 0.5 * self.terrain.size - 2.0
        self.position.x = float(np.clip(self.position.x, -half, half))
        self.position.z = float(np.clip(self.position.z, -half, half))


def resource_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def read_shader_source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def compile_shader_program(vertex_path: str, fragment_path: str) -> int:
    vertex_code = read_shader_source(resource_path(vertex_path))
    fragment_code = read_shader_source(resource_path(fragment_path))
    program = gls.compileProgram(
        gls.compileShader(vertex_code, GL_VERTEX_SHADER),
        gls.compileShader(fragment_code, GL_FRAGMENT_SHADER),
    )
    return program


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _edge_mountain_mask(u: float, v: float, inner: float = TERRAIN_FLAT_CENTER_RADIUS,
                        outer: float = TERRAIN_EDGE_FULL_RADIUS) -> float:
    radius = math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 1.41421356237
    return _smoothstep(inner, outer, radius)


def load_texture_2d(filepath: str, linear: bool = True, repeat: bool = True) -> int:
    if not os.path.isfile(filepath):
        print(f"[warn] Textura nao encontrada: {filepath}")
        return 0

    try:
        with Image.open(filepath) as image:
            image = image.convert("RGBA")
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(image, dtype=np.uint8)
            width, height = image.width, image.height
    except Exception as exc:  # pragma: no cover - protecao contra texturas invalidas
        print(f"[warn] Falha ao abrir textura '{filepath}': {exc}")
        return 0

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        img_data,
    )

    filter_mode = GL_LINEAR if linear else GL_NEAREST
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode)
    wrap_mode = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_mode)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id


def _find_local_texture(base_dir: str | None) -> str | None:
    """Procura por uma textura difusa perto do FBX quando o arquivo nao referencia nada."""
    if not base_dir:
        return None

    search_dirs = [
        base_dir,
        os.path.join(base_dir, "textures"),
        os.path.join(base_dir, "images"),
    ]
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".dds")
    candidates: list[str] = []

    for root in search_dirs:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.lower().endswith(exts):
                candidates.append(os.path.join(root, name))

    if not candidates:
        return None

    def _score(path: str) -> tuple[int, int]:
        lower = os.path.basename(path).lower()
        score = 0
        for kw in ("color", "diff", "albedo", "base", "body", "gold", "tex", "_c", "_d"):
            if kw in lower:
                score += 2
        for kw in ("normal", "nrm", "_n", "spec", "rough", "metal", "ao"):
            if kw in lower:
                score -= 1
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return score, size

    return max(candidates, key=_score)


def _filter_diffuse_candidates(paths: Sequence[str], blocked: Sequence[str]) -> list[str]:
    blocked_lower = tuple(b.lower() for b in blocked)
    filtered: list[str] = []
    for path in paths:
        if not path:
            continue
        base = os.path.basename(path).lower()
        if any(tag in base for tag in blocked_lower):
            continue
        filtered.append(path)
    return filtered


def resolve_terrain_texture_path() -> str | None:
    if SELECTED_TERRAIN_TEXTURE_PATH and os.path.isfile(SELECTED_TERRAIN_TEXTURE_PATH):
        return SELECTED_TERRAIN_TEXTURE_PATH
    for candidate in TERRAIN_TEXTURE_CANDIDATES:
        candidate_path = candidate if os.path.isabs(candidate) else resource_path(candidate)
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def pick_terrain_texture() -> tuple[int, str]:
    global SELECTED_TERRAIN_TEXTURE_PATH
    for candidate in TERRAIN_TEXTURE_CANDIDATES:
        candidate_path = candidate if os.path.isabs(candidate) else resource_path(candidate)
        texture_id = load_texture_2d(candidate_path)
        if texture_id:
            SELECTED_TERRAIN_TEXTURE_PATH = candidate_path
            print(f"[info] Textura do terreno carregada: {os.path.basename(candidate_path)}")
            return texture_id, candidate_path
        if os.path.isfile(candidate_path):
            print(f"[warn] Falha ao carregar textura do terreno '{os.path.basename(candidate_path)}', tentando proxima.")
    fallback_path = resolve_terrain_texture_path()
    SELECTED_TERRAIN_TEXTURE_PATH = fallback_path
    if fallback_path:
        print(f"[warn] Nenhuma textura nova funcionou; usando '{os.path.basename(fallback_path)}' como fallback sem textura carregada.")
    else:
        print("[warn] Nenhuma textura valida encontrada para o terreno; usando apenas cor base.")
    return 0, fallback_path or ""


def fractal_noise(x: float, y: float, seed: int) -> float:
    amplitude = 1.0
    frequency = 1.0
    total = 0.0
    max_value = 0.0
    for octave in range(HEIGHTMAP_OCTAVES):
        total += amplitude * pnoise2(
            x * frequency,
            y * frequency,
            repeatx=1024,
            repeaty=1024,
            base=seed + octave,
        )
        max_value += amplitude
        amplitude *= HEIGHTMAP_PERSISTENCE
        frequency *= HEIGHTMAP_LACUNARITY
    return total / max_value if max_value > 0 else 0.0


def create_terrain_mesh(seed: int) -> TerrainResource:
    resolution = TERRAIN_RESOLUTION
    size = TERRAIN_SIZE
    half = 0.5 * size

    heightmap = np.zeros((resolution, resolution), dtype=np.float32)
    positions = np.zeros((resolution * resolution, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    uv = np.zeros((resolution * resolution, 2), dtype=np.float32)

    for j in range(resolution):
        for i in range(resolution):
            u = i / (resolution - 1)
            v = j / (resolution - 1)
            x = (u - 0.5) * size
            z = (v - 0.5) * size
            edge_mask = _edge_mountain_mask(u, v)
            raw = fractal_noise(x / (HEIGHTMAP_SCALE * 10.0), z / (HEIGHTMAP_SCALE * 10.0), seed)
            rolling = raw * TERRAIN_HEIGHT * TERRAIN_GENTLE_FACTOR * (0.45 + 0.55 * edge_mask)
            ridge = max(0.0, raw - 0.05)
            h = rolling + (ridge ** 2.0) * TERRAIN_HEIGHT * TERRAIN_RIDGE_FACTOR * edge_mask
            heightmap[j, i] = h
            idx = j * resolution + i
            positions[idx] = [x, h, z]
            uv[idx] = [u * TERRAIN_TEXTURE_REPEAT, v * TERRAIN_TEXTURE_REPEAT]

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

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    buffers = glGenBuffers(3)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, buffers[2])
    glBufferData(GL_ARRAY_BUFFER, uv.nbytes, uv, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(2)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    texture_id, texture_path = pick_terrain_texture()

    mesh = MeshResource(
        vao=vao,
        count=indices.size,
        use_elements=True,
        texture_id=texture_id,
        base_color=glm.vec3(0.45, 0.55, 0.45),
        buffers=buffers,
        ebo=ebo,
    )

    min_y = float(np.min(heightmap))
    max_y = float(np.max(heightmap))
    bounds_min = np.array([-half, min_y, -half], dtype=np.float32)
    bounds_max = np.array([half, max_y, half], dtype=np.float32)

    return TerrainResource(
        mesh=mesh,
        heightmap=heightmap,
        size=size,
        resolution=resolution,
        min_bounds=bounds_min,
        max_bounds=bounds_max,
        texture_path=texture_path,
    )


def create_mesh_from_arrays(vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray,
                            texture_id: int, base_color: glm.vec3) -> MeshResource:
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    buffers = glGenBuffers(3)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, buffers[2])
    glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(2)

    glBindVertexArray(0)

    return MeshResource(
        vao=vao,
        count=int(len(vertices)),
        use_elements=False,
        texture_id=texture_id,
        base_color=base_color,
        buffers=buffers,
    )


def create_sun_billboard_resources() -> int:
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
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


def resolve_texture(texture_candidates: Sequence[str], model_dir: str | None = None, asset_name: str | None = None) -> str | None:
    """Resolve possible texture locations for FBX meshes.

    We prioritize files next to the FBX (inside 'FBX models') before falling back
    to the shared terrain textures.
    """
    models_root = resource_path("FBX models")
    textures_root = resource_path("Textures")

    prioritized: list[str] = []
    blocked: tuple[str, ...] = ()
    if asset_name:
        asset_lower = asset_name.lower()
        if "dragon" in asset_lower:
            # Forca o dragao a usar as texturas corretas dentro da pasta original.
            dragon_dir = model_dir or os.path.join(models_root, "fbx_blackDragon")
            prioritized = [
                os.path.join(dragon_dir, "textures", "Dragon_Bump_Col2.jpg"),
                os.path.join(dragon_dir, "Dragon_Bump_Col2.jpg"),
                os.path.join(dragon_dir, "textures", "Dragon_ground_color.jpg"),
                os.path.join(dragon_dir, "Dragon_ground_color.jpg"),
            ]
        elif "qishilong" in asset_lower:
            qish_dir = model_dir or os.path.join(models_root, "fbx_futureCar")
            prioritized = [
                os.path.join(qish_dir, "textures", "T_M_B_44_Qishilong_body01_B.png"),
                os.path.join(qish_dir, "textures", "T_M_B_44_Qishilong_body02_B.png"),
                os.path.join(qish_dir, "T_M_B_44_Qishilong_body01_B.png"),
                os.path.join(qish_dir, "T_M_B_44_Qishilong_body02_B.png"),
                "T_M_B_44_Qishilong_body01_B.png",
                "T_M_B_44_Qishilong_body02_B.png",
            ]
            blocked = ("ao", "curvature", "mask", "normal", "_n", "rough", "metal", "orm", "spec")

    candidates = list(dict.fromkeys(list(prioritized) + list(texture_candidates)))
    if blocked:
        candidates = _filter_diffuse_candidates(candidates, blocked)

    for candidate in candidates:
        if not candidate:
            continue

        attempts = []
        if os.path.isabs(candidate):
            attempts.append(candidate)
        else:
            normalized = os.path.normpath(candidate)
            attempts.append(normalized)
            if model_dir:
                attempts.append(os.path.normpath(os.path.join(model_dir, normalized)))
                attempts.append(os.path.normpath(os.path.join(model_dir, os.path.basename(normalized))))
            attempts.append(os.path.normpath(resource_path(normalized)))
            attempts.append(os.path.normpath(resource_path(os.path.basename(normalized))))
            attempts.append(os.path.normpath(os.path.join(models_root, normalized)))
            attempts.append(os.path.normpath(os.path.join(models_root, os.path.basename(normalized))))
            attempts.append(os.path.normpath(os.path.join(textures_root, os.path.basename(normalized))))

        for path in attempts:
            if os.path.isfile(path):
                return path

    fallback = _find_local_texture(model_dir)
    return fallback


def load_character_asset(spec: dict) -> CharacterAsset:
    name = spec["name"]
    path = spec["path"]
    model_path = resource_path("FBX models", path) if not os.path.isabs(path) else path
    model_dir = os.path.dirname(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"FBX nao encontrado: {model_path}")

    meshes_data = load_fbx_model(model_path)
    if not meshes_data:
        raise RuntimeError(f"Falha ao carregar {name}")

    yaw_off_rad = math.radians(spec.get("yaw_offset_deg", 0.0))
    pitch_off_rad = math.radians(spec.get("pitch_offset_deg", 0.0))
    roll_off_rad = math.radians(spec.get("roll_offset_deg", 0.0))
    rotation_matrix = _build_rotation_matrix(yaw_off_rad, pitch_off_rad, roll_off_rad)
    allow_texture_fallback = bool(spec.get("allow_texture_fallback", True))
    base_color_raw = spec.get("base_color", (1.0, 1.0, 1.0))
    if isinstance(base_color_raw, glm.vec3):
        base_color_vec = glm.vec3(base_color_raw)
    else:
        base_color_vec = glm.vec3(*base_color_raw)

    mesh_resources: List[MeshResource] = []
    rotated_bboxes: List[np.ndarray] = []
    raw_bboxes: List[np.ndarray] = []
    logged_missing_texture = False

    for mesh in meshes_data:
        vertices_pos, vertices_normals, vertices_uvs, faces = mesh[:4]
        texture_paths = mesh[5] if len(mesh) > 5 else []
        if len(vertices_uvs) == 0:
            vertices_uvs = np.zeros((len(vertices_pos), 2), dtype=np.float32)

        texture_path = resolve_texture(texture_paths, model_dir=model_dir, asset_name=name)
        if texture_path is None and allow_texture_fallback:
            texture_path = resolve_terrain_texture_path()
        elif texture_path is None and not logged_missing_texture:
            print(f"[info] Modelo '{name}' sem texturas detectadas; renderizando apenas com cor base/tinta.")
            logged_missing_texture = True
        texture_id = load_texture_2d(texture_path) if texture_path else 0

        mesh_resource = create_mesh_from_arrays(
            vertices_pos.astype(np.float32),
            vertices_normals.astype(np.float32),
            vertices_uvs.astype(np.float32),
            texture_id,
            base_color=base_color_vec,
        )
        mesh_resources.append(mesh_resource)

        rotated_vertices = _rotate_vertices(vertices_pos, rotation_matrix)
        bbox = compute_bounding_box(rotated_vertices, faces)
        rotated_bboxes.append(bbox)
        raw_bboxes.append(compute_bounding_box(vertices_pos, faces))

    if not rotated_bboxes:
        raise RuntimeError(f"Nao foi possivel calcular bbox para {name}")

    bbox_min = rotated_bboxes[0][0]
    bbox_max = rotated_bboxes[0][1]
    for bbox in rotated_bboxes[1:]:
        bbox_min = np.minimum(bbox_min, bbox[0])
        bbox_max = np.maximum(bbox_max, bbox[1])

    bbox = np.array([bbox_min, bbox_max], dtype=np.float32)
    local_min = raw_bboxes[0][0]
    local_max = raw_bboxes[0][1]
    for lb in raw_bboxes[1:]:
        local_min = np.minimum(local_min, lb[0])
        local_max = np.maximum(local_max, lb[1])
    local_bbox = np.array([local_min, local_max], dtype=np.float32)
    local_extents = local_max - local_min
    wing_center = (local_min + local_max) * 0.5
    bbox_extent = bbox_max - bbox_min
    rotated_height = max(float(bbox_extent[1]), 1e-5)
    target_height = spec.get("target_height", CHARACTER_TARGET_HEIGHT)
    target_width = spec.get("target_width")

    scale_y = target_height / rotated_height
    if target_width is not None:
        max_horizontal = max(float(bbox_extent[0]), float(bbox_extent[2]), 1e-5)
        scale_xz = target_width / max_horizontal
    else:
        scale_xz = scale_y

    scale_vec = glm.vec3(scale_xz, scale_y, scale_xz)
    hover_offset = float(spec.get("hover_offset", 0.0))
    base_offset = -bbox_min[1] * scale_y + hover_offset
    wing_enabled = bool(spec.get("wing_flap", False))
    wing_amplitude = float(spec.get("wing_amplitude", 0.0))
    wing_frequency = float(spec.get("wing_frequency", 0.0))

    return CharacterAsset(
        name=name,
        meshes=mesh_resources,
        bbox=bbox,
        scale_vec=scale_vec,
        base_offset=base_offset,
        yaw_offset_rad=yaw_off_rad,
        pitch_offset_rad=pitch_off_rad,
        roll_offset_rad=roll_off_rad,
        wing_enabled=wing_enabled,
        wing_center=glm.vec3(*wing_center.tolist()),
        wing_size=glm.vec3(*local_extents.tolist()),
        wing_amplitude=wing_amplitude,
        wing_frequency=wing_frequency,
    )


def create_instances(terrain: TerrainResource) -> tuple[List[SceneInstance], List[dict]]:
    random.seed(1337)
    loaded_assets = []
    for spec in CHARACTER_SPECS:
        try:
            asset = load_character_asset(spec)
            loaded_assets.append((spec, asset))
        except FileNotFoundError as exc:
            print(f"[warn] {exc}; pulando modelo '{spec.get('name', '?')}'")
            continue
        except RuntimeError as exc:
            print(f"[warn] {exc}; pulando modelo '{spec.get('name', '?')}'")

    if not loaded_assets:
        raise RuntimeError("Nenhum modelo FBX encontrado. Verifique a pasta 'FBX models'.")

    instances: List[SceneInstance] = []
    motion_states: List[dict] = []
    half = terrain.size * 0.5 - 8.0

    for spec, asset in loaded_assets:
        base_tint = spec["tint"]
        for _ in range(CHARACTER_INSTANCES):
            orbit_radius = random.uniform(6.0, 18.0)
            center_x = random.uniform(-half + orbit_radius, half - orbit_radius)
            center_z = random.uniform(-half + orbit_radius, half - orbit_radius)
            phase = random.uniform(0.0, math.tau)
            orbit_speed = random.uniform(0.15, 0.35)
            bob_amp = random.uniform(0.05, 0.25)
            bob_speed = random.uniform(0.8, 1.8)
            scale_variation = random.uniform(0.9, 1.1)

            tint_variation = glm.vec3(
                base_tint.x * random.uniform(0.9, 1.1),
                base_tint.y * random.uniform(0.9, 1.1),
                base_tint.z * random.uniform(0.9, 1.1),
            )
            tint = glm.clamp(tint_variation, 0.2, 1.2)

            instance = SceneInstance(asset.meshes, glm.mat4(1.0), glm.mat3(1.0), tint, asset)
            instances.append(instance)
            motion_states.append(
                {
                    "instance": instance,
                    "asset": asset,
                    "orbit_center": glm.vec2(center_x, center_z),
                    "orbit_radius": orbit_radius,
                    "orbit_speed": orbit_speed,
                    "phase_offset": phase,
                    "bob_amplitude": bob_amp,
                    "bob_speed": bob_speed,
                    "scale_variation": scale_variation,
                }
            )

    return instances, motion_states


def update_character_instances(motion_states: Sequence[dict], terrain: TerrainResource, time_now: float | None = None) -> None:
    if not motion_states:
        return
    if time_now is None:
        time_now = glfw.get_time()

    half = 0.5 * terrain.size - 2.0

    for state in motion_states:
        asset: CharacterAsset = state["asset"]
        instance: SceneInstance = state["instance"]

        asset_name = asset.name.lower()
        is_skull = any(key in asset_name for key in ("future car", "craneo", "skull", "caveira"))

        if is_skull:
            angle = state["phase_offset"]
            bob = 0.0
            sway_speed = 0.0
            sway_amp_pitch = 0.0
            sway_amp_roll = 0.0
        else:
            angle = state["phase_offset"] + time_now * state["orbit_speed"]
            bob = math.sin(time_now * state["bob_speed"] + state["phase_offset"]) * state["bob_amplitude"]
            sway_speed = 1.4
            sway_amp_pitch = 0.08
            sway_amp_roll = 0.05

        x = state["orbit_center"].x + math.cos(angle) * state["orbit_radius"]
        z = state["orbit_center"].y + math.sin(angle) * state["orbit_radius"]
        x = max(-half, min(half, x))
        z = max(-half, min(half, z))

        ground = terrain.sample_height(x, z)

        direction = glm.vec3(-math.sin(angle), 0.0, math.cos(angle))
        yaw = math.atan2(direction.x, direction.z) + asset.yaw_offset_rad + FACING_CORRECTION_RAD

        pitch = math.sin(time_now * sway_speed + state["phase_offset"]) * sway_amp_pitch if sway_speed > 0.0 else 0.0
        roll = math.cos(time_now * sway_speed * 0.8 + state["phase_offset"] * 1.3) * sway_amp_roll if sway_speed > 0.0 else 0.0

        scale_factor = state["scale_variation"]
        scale_vec = asset.scale_vec * scale_factor
        base_offset = asset.base_offset * scale_factor

        translation = glm.translate(glm.mat4(1.0), glm.vec3(x, ground + base_offset + bob, z))
        rotation_yaw = glm.rotate(glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0))
        rotation_pitch_off = glm.rotate(glm.mat4(1.0), asset.pitch_offset_rad, glm.vec3(1.0, 0.0, 0.0))
        rotation_roll_off = glm.rotate(glm.mat4(1.0), asset.roll_offset_rad, glm.vec3(0.0, 0.0, 1.0))
        rotation_pitch = glm.rotate(glm.mat4(1.0), pitch, glm.vec3(1.0, 0.0, 0.0))
        rotation_roll = glm.rotate(glm.mat4(1.0), roll, glm.vec3(0.0, 0.0, 1.0))
        scale_matrix = glm.scale(glm.mat4(1.0), scale_vec)

        model_matrix = translation * rotation_yaw * rotation_pitch_off * rotation_roll_off * rotation_pitch * rotation_roll * scale_matrix
        instance.model_matrix = model_matrix
        instance.normal_matrix = glm.mat3(glm.transpose(glm.inverse(model_matrix)))


def create_shadow_map(size: int) -> ShadowMap:
    depth_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_texture)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_DEPTH_COMPONENT,
        size,
        size,
        0,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        None,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    border = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border)

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return ShadowMap(fbo=fbo, texture=depth_texture, size=size)


def compute_environment(sim_hours: float) -> dict:
    sun_angle = math.tau * ((sim_hours - 6.0) / 24.0)  # 06h nasce no leste (x+), 18h se poe no oeste (x-)
    radius = 900.0
    sun_pos = glm.vec3(
        math.cos(sun_angle) * radius,
        math.sin(sun_angle) * radius * 0.65,
        math.sin(sun_angle * 0.35) * radius * 0.3,
    )
    sun_dir = glm.normalize(-sun_pos)
    altitude = -sun_dir.y  # positivo quando o sol esta acima, negativo abaixo do horizonte

    day_sky = glm.vec3(0.36, 0.62, 0.92)
    night_sky = glm.vec3(0.01, 0.01, 0.04)
    dawn_sky = glm.vec3(0.98, 0.55, 0.35)

    dawn_mix = _smoothstep(-0.25, 0.05, altitude)
    day_mix = _smoothstep(0.05, 0.45, altitude)
    sky_color = glm.mix(night_sky, dawn_sky, dawn_mix)
    sky_color = glm.mix(sky_color, day_sky, day_mix)

    fog_color = glm.mix(night_sky, day_sky, day_mix)
    fog_density = 0.002 - 0.0015 * day_mix
    ambient = glm.mix(glm.vec3(0.08, 0.08, 0.1), glm.vec3(0.4, 0.4, 0.42), day_mix)
    sun_color = glm.mix(glm.vec3(1.0, 0.45, 0.25), glm.vec3(1.0, 0.97, 0.85), day_mix)

    return {
        "sun_position": sun_pos,
        "sun_direction": sun_dir,
        "sun_color": sun_color,
        "ambient": ambient,
        "fog_color": fog_color,
        "fog_density": fog_density,
        "sky_color": sky_color,
    }


def compute_light_space_matrix(light_pos: glm.vec3) -> glm.mat4:
    target = glm.vec3(0.0, 0.0, 0.0)
    up = glm.vec3(0.0, 1.0, 0.0)
    light_view = glm.lookAt(light_pos, target, up)
    extent = TERRAIN_SIZE
    near_plane = 0.1
    far_plane = 2000.0
    light_projection = glm.ortho(-extent, extent, -extent, extent, near_plane, far_plane)
    return light_projection * light_view


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
    start_x = WINDOW_WIDTH - CLOCK_MARGIN_PX - text_width
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
                x0_ndc = (x0 / (WINDOW_WIDTH * 0.5)) - 1.0
                x1_ndc = (x1 / (WINDOW_WIDTH * 0.5)) - 1.0
                y0_ndc = 1.0 - (y0 / (WINDOW_HEIGHT * 0.5))
                y1_ndc = 1.0 - (y1 / (WINDOW_HEIGHT * 0.5))
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


def _format_clock(sim_hours: float) -> str:
    h = int(sim_hours) % 24
    m = int((sim_hours % 1.0) * 60.0)
    return f"{h:02d}:{m:02d}"


def render_clock_overlay(sim_hours: float):
    _ensure_clock_resources()
    text = _format_clock(sim_hours)
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


def set_wing_uniforms(program: int, asset: CharacterAsset | None):
    use_wings = int(asset.wing_enabled) if asset else 0
    center = asset.wing_center if asset else glm.vec3(0.0, 0.0, 0.0)
    size = asset.wing_size if asset else glm.vec3(1.0, 1.0, 1.0)
    amp = float(asset.wing_amplitude) if asset else 0.0
    freq = float(asset.wing_frequency) if asset else 0.0

    glUniform1i(glGetUniformLocation(program, "useWingFlap"), use_wings)
    glUniform3fv(glGetUniformLocation(program, "wingCenter"), 1, glm.value_ptr(center))
    glUniform3fv(glGetUniformLocation(program, "wingSize"), 1, glm.value_ptr(size))
    glUniform1f(glGetUniformLocation(program, "wingAmplitude"), amp)
    glUniform1f(glGetUniformLocation(program, "wingFrequency"), freq)


def render_instances(instances: Sequence[SceneInstance], model_loc: int, normal_loc: int,
                     tint_loc: int, has_texture_loc: int, base_color_loc: int, program: int):
    for instance in instances:
        set_wing_uniforms(program, instance.asset)
        for mesh in instance.meshes:
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(instance.model_matrix))
            glUniformMatrix3fv(normal_loc, 1, GL_FALSE, glm.value_ptr(instance.normal_matrix))
            glUniform3fv(tint_loc, 1, glm.value_ptr(instance.tint))
            glUniform3fv(base_color_loc, 1, glm.value_ptr(mesh.base_color))
            glUniform1i(has_texture_loc, GL_TRUE if mesh.texture_id else GL_FALSE)

            glBindVertexArray(mesh.vao)
            if mesh.texture_id:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, mesh.texture_id)

            if mesh.use_elements:
                glDrawElements(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, mesh.count)

            if mesh.texture_id:
                glBindTexture(GL_TEXTURE_2D, 0)
            glBindVertexArray(0)


def render_shadow_pass(depth_shader: int, instances: Sequence[SceneInstance], light_space_matrix: glm.mat4,
                      shadow_map: ShadowMap, time_now: float):
    glViewport(0, 0, shadow_map.size, shadow_map.size)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_map.fbo)
    glClear(GL_DEPTH_BUFFER_BIT)
    glUseProgram(depth_shader)

    light_loc = glGetUniformLocation(depth_shader, "lightSpaceMatrix")
    model_loc = glGetUniformLocation(depth_shader, "model")
    glUniform1f(glGetUniformLocation(depth_shader, "time"), time_now)

    glUniformMatrix4fv(light_loc, 1, GL_FALSE, glm.value_ptr(light_space_matrix))

    glCullFace(GL_FRONT)
    for instance in instances:
        set_wing_uniforms(depth_shader, instance.asset)
        for mesh in instance.meshes:
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(instance.model_matrix))
            glBindVertexArray(mesh.vao)
            if mesh.use_elements:
                glDrawElements(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, mesh.count)
            glBindVertexArray(0)
    glCullFace(GL_BACK)

    glUseProgram(0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def _project_to_ndc(world_pos: glm.vec3, view: glm.mat4, projection: glm.mat4):
    clip = projection * view * glm.vec4(world_pos, 1.0)
    if clip.w <= 0.0:
        return None
    ndc = clip / clip.w
    if ndc.z > 1.0 or ndc.z < 0.0:
        return None
    return glm.vec3(ndc.x, ndc.y, ndc.z)


def render_sun_billboard(view: glm.mat4, projection: glm.mat4, sun_pos: glm.vec3, sun_color: glm.vec3):
    if sun_shader is None or sun_quad_vao is None:
        return
    center = _project_to_ndc(sun_pos, view, projection)
    if center is None:
        return

    glUseProgram(sun_shader)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_ONE, GL_ONE)

    aspect = WINDOW_WIDTH / float(max(1, WINDOW_HEIGHT))
    glUniform2f(glGetUniformLocation(sun_shader, "center"), center.x, center.y)
    glUniform1f(glGetUniformLocation(sun_shader, "radius"), 0.08)
    glUniform1f(glGetUniformLocation(sun_shader, "aspect"), aspect)
    glUniform3fv(glGetUniformLocation(sun_shader, "sunColor"), 1, glm.value_ptr(sun_color))

    glBindVertexArray(sun_quad_vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glBindVertexArray(0)

    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glUseProgram(0)


def render_scene(scene_shader: int, instances: Sequence[SceneInstance], camera: CameraController,
                 view: glm.mat4, projection: glm.mat4, env: dict, light_space_matrix: glm.mat4,
                 shadow_map: ShadowMap, time_now: float):
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(scene_shader)
    glUniform1f(glGetUniformLocation(scene_shader, "time"), time_now)

    glUniformMatrix4fv(glGetUniformLocation(scene_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(scene_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(glGetUniformLocation(scene_shader, "lightSpaceMatrix"), 1, GL_FALSE, glm.value_ptr(light_space_matrix))
    glUniform3fv(glGetUniformLocation(scene_shader, "viewPos"), 1, glm.value_ptr(camera.position))
    glUniform3fv(glGetUniformLocation(scene_shader, "lightDir"), 1, glm.value_ptr(env["sun_direction"]))
    glUniform3fv(glGetUniformLocation(scene_shader, "lightColor"), 1, glm.value_ptr(env["sun_color"]))
    glUniform3fv(glGetUniformLocation(scene_shader, "ambientColor"), 1, glm.value_ptr(env["ambient"]))
    glUniform3fv(glGetUniformLocation(scene_shader, "fogColor"), 1, glm.value_ptr(env["fog_color"]))
    glUniform1f(glGetUniformLocation(scene_shader, "fogDensity"), env["fog_density"])
    glUniform1f(glGetUniformLocation(scene_shader, "fogBaseDensity"), env["fog_density"])
    glUniform1f(glGetUniformLocation(scene_shader, "fogHeightFalloff"), 0.05)
    glUniform1f(glGetUniformLocation(scene_shader, "fogAnisotropy"), 0.2)
    glUniform1i(glGetUniformLocation(scene_shader, "fogSteps"), 8)
    glUniform1f(glGetUniformLocation(scene_shader, "lightRadiusUV"), 2.5 / float(SHADOW_MAP_SIZE))
    glUniform1f(glGetUniformLocation(scene_shader, "minBias"), 0.0004)
    glUniform1f(glGetUniformLocation(scene_shader, "maxBias"), 0.002)
    glUniform1i(glGetUniformLocation(scene_shader, "blockerSamples"), 8)
    glUniform1i(glGetUniformLocation(scene_shader, "pcfSamples"), 16)

    glUniform1i(glGetUniformLocation(scene_shader, "diffuseTexture"), 0)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, shadow_map.texture)
    glUniform1i(glGetUniformLocation(scene_shader, "shadowMap"), 1)

    model_loc = glGetUniformLocation(scene_shader, "model")
    normal_loc = glGetUniformLocation(scene_shader, "normalMatrix")
    tint_loc = glGetUniformLocation(scene_shader, "tintColor")
    has_texture_loc = glGetUniformLocation(scene_shader, "hasTexture")
    base_color_loc = glGetUniformLocation(scene_shader, "baseColor")

    render_instances(instances, model_loc, normal_loc, tint_loc, has_texture_loc, base_color_loc, scene_shader)

    render_sun_billboard(view, projection, env["sun_position"], env["sun_color"])

    glBindTexture(GL_TEXTURE_2D, 0)
    glUseProgram(0)


keys_state: dict[int, bool] = {}
mouse_data = {"first": True, "x": 0.0, "y": 0.0}
jump_requested = False


def key_callback(window, key, scancode, action, mods):
    global jump_requested
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        return

    if action == glfw.PRESS:
        keys_state[key] = True
        if key == glfw.KEY_SPACE:
            jump_requested = True
    elif action == glfw.RELEASE:
        keys_state[key] = False


def mouse_callback(window, xpos, ypos):
    if mouse_data["first"]:
        mouse_data["x"] = xpos
        mouse_data["y"] = ypos
        mouse_data["first"] = False
    dx = xpos - mouse_data["x"]
    dy = mouse_data["y"] - ypos
    mouse_data["x"] = xpos
    mouse_data["y"] = ypos
    window_data = glfw.get_window_user_pointer(window)
    if not window_data:
        return
    camera: CameraController = window_data["camera"]
    camera.yaw += dx * MOUSE_SENSITIVITY
    camera.pitch += dy * MOUSE_SENSITIVITY
    camera.pitch = max(-89.0, min(89.0, camera.pitch))
    camera.update_vectors()


def framebuffer_size_callback(window, width, height):
    global WINDOW_WIDTH, WINDOW_HEIGHT
    WINDOW_WIDTH = max(1, width)
    WINDOW_HEIGHT = max(1, height)


def process_movement(camera: CameraController, delta_time: float):
    global jump_requested
    speed = MOVE_SPEED * delta_time
    if keys_state.get(glfw.KEY_LEFT_SHIFT, False):
        speed *= RUN_MULTIPLIER

    move_direction = glm.vec3(0.0, 0.0, 0.0)
    forward = glm.normalize(glm.vec3(camera.front.x, 0.0, camera.front.z))
    right = glm.normalize(glm.cross(forward, camera.world_up))

    if keys_state.get(glfw.KEY_W, False):
        move_direction += forward
    if keys_state.get(glfw.KEY_S, False):
        move_direction -= forward
    if keys_state.get(glfw.KEY_A, False):
        move_direction -= right
    if keys_state.get(glfw.KEY_D, False):
        move_direction += right

    if glm.length(move_direction) > 0:
        move_direction = glm.normalize(move_direction)
        camera.position += move_direction * speed

    ground_height = camera.terrain.sample_height(camera.position.x, camera.position.z) + CAMERA_EYE_HEIGHT

    if jump_requested and camera.is_grounded:
        camera.vertical_velocity = JUMP_STRENGTH
        camera.is_grounded = False
    jump_requested = False

    camera.vertical_velocity += GRAVITY * delta_time
    camera.height_offset += camera.vertical_velocity * delta_time

    if camera.height_offset <= 0.0:
        camera.height_offset = 0.0
        camera.vertical_velocity = 0.0
        camera.is_grounded = True

    camera.position.y = ground_height + camera.height_offset
    camera.clamp_position()


def main():
    if not glfw.init():
        raise SystemExit("Falha ao iniciar GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "A3 - Mundo Procedural", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Falha ao criar janela")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glEnable(GL_MULTISAMPLE)

    scene_shader = compile_shader_program("05 - world_scene_vs.glsl", "05 - world_scene_fs.glsl")
    depth_shader = compile_shader_program("05 - depth_vs.glsl", "05 - depth_fs.glsl")
    global sun_shader, sun_quad_vao
    sun_shader = compile_shader_program("05 - sun_billboard_vs.glsl", "05 - sun_billboard_fs.glsl")
    sun_quad_vao = create_sun_billboard_resources()

    terrain = create_terrain_mesh(seed=42)
    character_instances, motion_states = create_instances(terrain)
    terrain_instance = SceneInstance(
        meshes=[terrain.mesh],
        model_matrix=glm.mat4(1.0),
        normal_matrix=glm.mat3(1.0),
        tint=glm.vec3(1.0, 1.0, 1.0),
        asset=None,
    )
    all_instances = [terrain_instance] + character_instances

    camera = CameraController(terrain)
    glfw.set_window_user_pointer(window, {"camera": camera})
    mouse_data["first"] = True

    shadow_map = create_shadow_map(SHADOW_MAP_SIZE)
    update_character_instances(motion_states, terrain, glfw.get_time())

    last_time = glfw.get_time()
    sim_hours = 6.0

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        sim_hours = (sim_hours + delta_time * SIM_HOURS_PER_REAL_SECOND) % 24.0

        glfw.poll_events()
        process_movement(camera, delta_time)
        update_character_instances(motion_states, terrain, current_time)

        env = compute_environment(sim_hours)
        glClearColor(env["sky_color"].x, env["sky_color"].y, env["sky_color"].z, 1.0)

        light_space_matrix = compute_light_space_matrix(env["sun_position"])

        render_shadow_pass(depth_shader, all_instances, light_space_matrix, shadow_map, current_time)

        view = glm.lookAt(camera.position, camera.position + camera.front, camera.up)
        projection = glm.perspective(glm.radians(FIELD_OF_VIEW), WINDOW_WIDTH / float(max(1, WINDOW_HEIGHT)), NEAR_PLANE, FAR_PLANE)
        render_scene(scene_shader, all_instances, camera, view, projection, env, light_space_matrix, shadow_map, current_time)
        render_clock_overlay(sim_hours)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
