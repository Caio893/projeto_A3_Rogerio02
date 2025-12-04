import glfw
import glm

from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders as gls

import tkinter as tk
from tkinter import filedialog

from geometry_utils import *
from glb_utils import *


field_of_view = 60
window_size = [600, 600]

camera_pos = np.array([0, 0, 10], dtype=np.float32)
camera_aim = np.array([0, 0, 0], dtype=np.float32)
camera_up = np.array([0, 0, 1], dtype=np.float32)
far = 1000.0

shader_gouraud_shading_program = None
shader_phong_shading_program = None

vao_flat_shading_id = None
vao_vertex_shading_id = None

current_shading_mode = 2
model_vertices_size = 0

keys_used = {}


# ################################################################################################
# GLSL (Shaders) - Funções auxiliares
# ################################################################################################

def create_shader_program(vertex_shader_filepath, fragment_shader_filepath):
    # ler o código do vertex shader no caso de estar em um arquivo separado
    vertex_shader_source = ''
    with open(vertex_shader_filepath, 'r') as file:
        vertex_shader_source = file.read()

    # criar o objeto vertex shader
    vertex_shader = gls.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    
    # ler o código do fragment shader no caso de estar em um arquivo separado
    fragment_shader_source = ''
    with open(fragment_shader_filepath, 'r') as file:
        fragment_shader_source = file.read()

    # criar o objeto fragment shader
    fragment_shader = gls.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)

    # criar o programa com os shaders
    shader_program = gls.compileProgram(vertex_shader, fragment_shader)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


def create_vbo(buffer_type, buffer, index):
    vbo_id = glGenBuffers(1)
    glBindBuffer(buffer_type, vbo_id)
    glBufferData(buffer_type,     # tipo do buffer
                 buffer.nbytes,   # tamanho do buffer
                 buffer,          # dados
                 GL_STATIC_DRAW)  # forma do uso do buffer
    
    glVertexAttribPointer(index,              # código do atributo posição
                          buffer.shape[1],    # 2D ou 3D
                          GL_FLOAT,           # tipo dos valores do atributo
                          GL_FALSE,           # não desejo normalizar
                          0,                  # 2 floats de 4 bytes - quantidade de bytes entre um atributo e o próximo
                          ctypes.c_void_p(0)) # 0 pois começa no inicio do buffer (offset)
    glEnableVertexAttribArray(index)

    glBindBuffer(buffer_type, 0)


def create_vao(model_vertices_pos, model_vertices_normals):
    vao_id = glGenVertexArrays(1)
    glBindVertexArray(vao_id)

    create_vbo(GL_ARRAY_BUFFER, model_vertices_pos, 0)
    create_vbo(GL_ARRAY_BUFFER, model_vertices_normals, 1)

    glBindVertexArray(0)

    return vao_id


# ################################################################################################
# OpenGL - Funções auxiliares
# ################################################################################################

def get_vertices_lists_flat_shading(vertices, faces, faces_normals):
    vertex_pos_list = []
    vertex_normals_list = []

    for i in range(len(faces)):
        face = faces[i]
        face_normal = faces_normals[i]

        for j in range(3):
            vertex_index = face[j][0]
            vertex = vertices[vertex_index]
            vertex_pos_list.append(vertex)
            vertex_normals_list.append(face_normal)

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32)


def get_vertices_lists_vertex_shading(vertices, faces, vertices_normals):
    vertex_pos_list = []
    vertex_normals_list = []

    for i in range(len(faces)):
        face = faces[i]

        for j in range(3):
            vertex_index = face[j][0]
            
            vertex = vertices[vertex_index]
            vertex_normal = vertices_normals[vertex_index]

            vertex_pos_list.append(vertex)
            vertex_normals_list.append(vertex_normal)

    return np.array(vertex_pos_list, dtype=np.float32), np.array(vertex_normals_list, dtype=np.float32)


def my_init(fbx_file_path):
    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)
    
    # carrega o modelo
    vertices_pos, vertices_normals, faces, faces_normals = load_glb_model_no_texture(fbx_file_path)
    bounding_box = compute_bounding_box(vertices_pos, faces)

    # calcula posição inicial da camera
    global camera_pos, camera_aim, camera_up, far
    camera_pos, camera_aim, camera_up, far = compute_camera_position(bounding_box, field_of_view)

    # compila os programas shaders:
    # 1 - um que implementa o shading de Gouraud, implementado pelo OpenGL Legado
    # 2 - outro que implementa o shading de Phong
    global shader_gouraud_shading_program
    shader_gouraud_shading_program = create_shader_program('01 - glb_viewer_light_shader_no_texture_gouraud_vs.glsl', '01 - glb_viewer_light_shader_no_texture_gouraud_fs.glsl')

    global shader_phong_shading_program
    shader_phong_shading_program = create_shader_program('01 - glb_viewer_light_shader_no_texture_phong_vs.glsl', '01 - glb_viewer_light_shader_no_texture_phong_fs.glsl')

    # cria dois VAOs: 
    # 1 - um para flat shading com a mesma normal para os 3 vértices de uma mesma face e
    # 2 - outro para gouraud e phong shading com a normal calculada por vértice como sendo a média das normais das faces em torno do vértice
    global vao_flat_shading_id
    model_vertices_pos, model_vertices_normals = get_vertices_lists_flat_shading(vertices_pos, faces, faces_normals)
    vao_flat_shading_id = create_vao(model_vertices_pos, model_vertices_normals)

    global vao_vertex_shading_id
    model_vertices_pos, model_vertices_normals = get_vertices_lists_vertex_shading(vertices_pos, faces, vertices_normals)
    vao_vertex_shading_id = create_vao(model_vertices_pos, model_vertices_normals)

    global model_vertices_size
    model_vertices_size = len(model_vertices_pos)


def get_current_shading_name():
    if current_shading_mode % 3 == 0:
        return "Flat Shading"
    
    if current_shading_mode % 3 == 1:
       return "Gouraud Shading"

    return "Phong Shading"


def get_current_vao():
    if current_shading_mode % 3 == 0:
        return vao_flat_shading_id
    
    if current_shading_mode % 3 == 1:
       return vao_vertex_shading_id

    return vao_vertex_shading_id


def get_current_shader_program():
    if current_shading_mode % 3 == 0:
        return shader_gouraud_shading_program
    
    if current_shading_mode % 3 == 1:
       return shader_gouraud_shading_program

    return shader_phong_shading_program


def my_render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, window_size[0], window_size[1])

    current_shader_program = get_current_shader_program()
    glUseProgram(current_shader_program)

    # Matrizes
    model = glm.mat4(1.0)
    view = glm.lookAt(glm.vec3(camera_pos[0], camera_pos[1], camera_pos[2]), 
                      glm.vec3(camera_aim[0], camera_aim[1], camera_aim[2]), 
                      glm.vec3(camera_up[0], camera_up[1], camera_up[2]))    
    projection = glm.perspective(glm.radians(field_of_view), float(window_size[0])/float(window_size[1]), 0.01, far)
    light_pos = camera_pos + camera_up * 10

    # Uniforms de transformação para o vertex shader
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(current_shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

    # Uniforms de iluminação para o fragment shader
    glUniform3f(glGetUniformLocation(current_shader_program, "objectColor"), 1.0, 1.0, 1.0)
    glUniform3fv(glGetUniformLocation(current_shader_program, "viewPos"), 1, camera_pos)

    glUniform3f(glGetUniformLocation(current_shader_program, "ambientColor"), 0.3, 0.3, 0.3)
    glUniform3f(glGetUniformLocation(current_shader_program, "lightColor"), 1.0, 1.0, 0.5)
    glUniform3fv(glGetUniformLocation(current_shader_program, "lightPos"), 1, light_pos)
    glUniform1f(glGetUniformLocation(current_shader_program, "shininess"), 64.0)

    glBindVertexArray(get_current_vao())
 
    glDrawArrays(GL_TRIANGLES, 0, model_vertices_size)
    
    glBindVertexArray(0)
    glUseProgram(0)


# ################################################################################################
# GLFW - Funções auxiliares
# ################################################################################################

def my_update_window_size(window, width, height):
    global window_size
    window_size = [width, height]


def rotate_view_pos_horizontal(angle_rotate_rads):
    global camera_pos

    view_direction = np.array(camera_pos) - np.array(camera_aim)
    side_direction = np.cross(view_direction, np.array([0.0, 1.0, 0.0]))
    up = np.cross(side_direction, view_direction)
    up = up / np.linalg.norm(up)

    rotate_matrix = get_rotate_matrix(angle_rotate_rads, up)

    camera_pos = np.dot(view_direction, rotate_matrix.T) + np.array(camera_aim)
    camera_pos = camera_pos.tolist()


def rotate_view_pos_vertical(angle_rotate_rads):
    global camera_pos

    view_direction = np.array(camera_pos) - np.array(camera_aim)
    side_direction = np.cross(view_direction, np.array([0.0, 1.0, 0.0]))
    side_direction = side_direction / np.linalg.norm(side_direction)

    rotate_matrix = get_rotate_matrix(angle_rotate_rads, side_direction)

    camera_pos = np.dot(view_direction, rotate_matrix.T) + np.array(camera_aim)
    camera_pos = camera_pos.tolist()


def my_keyboard(window, key, scancode, action, mods):
    global keys_used, current_shading_mode

    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    if action == glfw.PRESS and key == glfw.KEY_F12:
        current_shading_mode = current_shading_mode + 1
        glfw.set_window_title(window, "GLB Viewer - " + get_current_shading_name())

    if action == glfw.REPEAT or action == glfw.PRESS:
        keys_used[key] = True
    elif action == glfw.RELEASE:
        keys_used[key] = False


def process_user_interaction(window):
    angle_rotate_rads = 0.08

    if keys_used.get(glfw.KEY_LEFT, False) or keys_used.get(glfw.KEY_A, False):
        rotate_view_pos_horizontal(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_RIGHT, False) or keys_used.get(glfw.KEY_D, False):
        rotate_view_pos_horizontal(angle_rotate_rads)

    if keys_used.get(glfw.KEY_UP, False) or keys_used.get(glfw.KEY_W, False):
        rotate_view_pos_vertical(-angle_rotate_rads)

    if keys_used.get(glfw.KEY_DOWN, False) or keys_used.get(glfw.KEY_S, False):
        rotate_view_pos_vertical(angle_rotate_rads)


# ################################################################################################
# Main - Programa Principal
# ################################################################################################

def open_glb_file():
    # Cria janela "oculta"
    root = tk.Tk()
    root.withdraw()  

    # Abre diálogo de seleção
    arquivo = filedialog.askopenfilename(
    title="Selecione um arquivo",
        filetypes=(("Arquivos modelos glb", "*.glb"), 
                   ("Todos os arquivos", "*.*"))
    )

    return arquivo

def main():
    glb_file_path = open_glb_file()
    if not open_glb_file or len(glb_file_path) == 0:
        return
    
    glfw.init()

    window = glfw.create_window(window_size[0], window_size[1], "GLB Viewer - " + get_current_shading_name(), None, None)
    glfw.make_context_current(window) 
    glfw.swap_interval(1)

    glfw.set_framebuffer_size_callback(window, my_update_window_size)
    glfw.set_key_callback(window, my_keyboard)

    glfw.maximize_window(window)

    my_init(glb_file_path)

    while not glfw.window_should_close(window):
        glfw.poll_events() # escutando eventos
        process_user_interaction(window)

        my_render()

        glfw.swap_buffers(window) # alternando os buffers

    glfw.terminate()


if __name__ == "__main__":
    main()