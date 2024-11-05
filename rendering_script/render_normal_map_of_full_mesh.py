import os, sys, time
import bpy  # Blender의 Python API
import numpy as np
import shutil
from math import radians
from mathutils import Matrix, Vector  # Blender의 수학 유틸리티

# 실행 지침 (주제 '0501'에서 '90'도 요각으로 실행):
# 이 폴더('rendering_script')로 이동
# blender blank.blend -b -P render_normal_map_of_full_mesh.py -- 0501 90

use_gpu = True  # GPU 사용 여부 설정

curpath = os.path.abspath(os.path.dirname("."))  # 현재 경로 설정
sys.path.insert(0, curpath)  # 현재 경로를 시스템 경로에 추가

argv = sys.argv  # 명령줄 인자 가져오기
argv = argv[argv.index("--") + 1:]  # '--' 이후의 인자만 사용

subject = argv[0]  # 주제 설정
angle = int(argv[1])  # 각도 설정
base_path = argv[2]  # 세 번째 인자를 경로로 설정

# 360도 이상의 각도를 변환
if angle >= 360:
    angle = angle - 360

light_energy = 0.5e+02  # 조명 에너지 설정

RESOLUTION = 1024  # 렌더링 해상도 설정

BUFFER_PATH = "buffer_normal_maps_of_full_mesh"  # 출력 폴더 설정
front_normal_map_filename = "rendered_nmlF"  # 전면 노멀맵 파일 이름
back_normal_map_filename = "rendered_nmlB"  # 후면 노멀맵 파일 이름

shape_file = os.path.join(base_path, subject, subject + ".obj")

save_folder_path = os.path.join(curpath, BUFFER_PATH, subject)  # 저장 폴더 경로 설정
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)  # 폴더가 없으면 생성

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    # OBJ 파일에서 메쉬 데이터를 로드하는 함수
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # 쿼드 메쉬
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # 삼각형 메쉬
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # 텍스처 처리
            if len(values[1].split('/')) >= 2:
                # 쿼드 메쉬
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # 삼각형 메쉬
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # 노멀 처리
            if len(values[1].split('/')) == 3:
                # 쿼드 메쉬
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # 삼각형 메쉬
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

def make_rotate(rx, ry, rz):
    # x, y, z 축에 대한 회전 행렬 생성
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R

def setupBlender(angle_to_rotate):
    # Blender 설정 함수
    global overall_vertices

    scene = bpy.context.scene
    camera = bpy.data.objects["Camera"]

    bpy.context.scene.render.engine = 'CYCLES'  # 렌더 엔진 설정

    if use_gpu:
        # GPU 사용 설정
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # 또는 "OPENCL"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # 정점 회전
    R = make_rotate(0, radians(angle_to_rotate), 0)
    vertices = np.matmul(R, overall_vertices.T).T  # [num_of_pts, 3] 형태

    vmin = vertices.min(0)
    vmax = vertices.max(0)

    vcenter_vertical = (vmax[1] + vmin[1]) / 2

    vrange = vmax - vmin
    orthographic_scale_to_use = np.max(vrange) * 1.1

    camera.data.clip_end = 50.0  # 카메라 클리핑 거리 설정
    camera.data.type = "ORTHO"  # 카메라 타입 설정

    # 카메라 위치 설정
    camera.location[0] = 0
    camera.location[1] = 0
    camera.location[2] = 10.0

    small_x_range = vrange[0] * 0.005
    small_y_range = vrange[1] * 0.005
    temp_bool_y = (vertices[:, 1] > (vcenter_vertical - small_y_range)) & (vertices[:, 1] < (vcenter_vertical + small_y_range))
    horizontal_line = vertices[temp_bool_y, :]  # 수평선
    vcenter_horizontal = np.median(horizontal_line, 0)[0]
    temp_bool_x = (vertices[:, 0] > (vcenter_horizontal - small_x_range)) & (vertices[:, 0] < (vcenter_horizontal + small_x_range))
    vertical_line = vertices[temp_bool_x, :]  # 수직선
    temp_bool_x_and_y = temp_bool_x & temp_bool_y
    small_cube = vertices[temp_bool_x_and_y, :]  # 중심 주변의 작은 큐브
    pt_nearest_to_cam = small_cube.max(0)
    z_coor = pt_nearest_to_cam[2]

    mesh_center = np.array([vcenter_horizontal, vcenter_vertical, z_coor])

    # 카메라 회전 설정 (라디안)
    camera.rotation_euler[0] = 0.0 / 180 * np.pi
    camera.rotation_euler[1] = 0.0 / 180 * np.pi
    camera.rotation_euler[2] = 0.0 / 180 * np.pi

    camera.data.ortho_scale = orthographic_scale_to_use

    # 조명 설정
    light = bpy.data.objects["Light"]
    light.data.energy = light_energy

    light.location[0] = (0 + vmin[0] - vcenter_horizontal) / 2
    light.location[1] = (0 + vmax[1] - vcenter_vertical) / 2
    light.location[2] = vmax[2] * 8

    # Blender에서 컴포지션 사용 시작
    scene.render.use_compositing = True
    scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    rl = bpy.context.scene.node_tree.nodes["Render Layers"]

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = save_folder_path  # 저장 경로 설정
    fo.format.file_format = "OPEN_EXR"

    bpy.context.scene.view_layers["View Layer"].use_pass_normal = True

    tree.links.new(rl.outputs["Normal"], fo.inputs["Image"])

    # 렌더링 해상도 설정
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION

    # 배경 색상 설정
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    r, g, b = 0, 0, 0
    bg.inputs[0].default_value[:3] = (r, g, b)

    return scene, camera, fo, mesh_center

# OBJ 파일을 Blender로 가져오기
# bpy.ops.import_scene.obj(filepath=shape_file)
bpy.ops.wm.obj_import(filepath=shape_file)

mesh_name = ""
for _object in bpy.data.objects:
    if subject in _object.name:
        mesh_name = _object.name

mesh = bpy.data.objects[mesh_name]  # 방금 로드된 메쉬 가져오기

# overall_vertices는 전역 변수
overall_vertices, _ = load_obj_mesh(shape_file, with_normal=False, with_texture=False)

scene, camera, fo, mesh_center = setupBlender(angle_to_rotate=angle)

# 메쉬의 위치/변환 및 회전 시작
rot_mat = Matrix.Rotation(radians(angle), 4, 'Y')
backside_rot_mat = Matrix.Rotation(radians(180), 4, 'Y')

orig_loc, orig_rot, orig_scale = mesh.matrix_world.decompose()

vec = Vector((-mesh_center[0], -mesh_center[1], -mesh_center[2]))
new_orig_loc_mat = Matrix.Translation(vec)

orig_scale_mat = Matrix.Scale(orig_scale[0], 4, (1, 0, 0)) * Matrix.Scale(orig_scale[1], 4, (0, 1, 0)) * Matrix.Scale(orig_scale[2], 4, (0, 0, 1))

# 새로운 행렬 조립
mesh.matrix_world = new_orig_loc_mat @ rot_mat @ orig_scale_mat

bpy.ops.render.render(write_still=False)
os.rename(os.path.join(save_folder_path, "Image0001.exr"), os.path.join(save_folder_path, front_normal_map_filename + "_" + "{0:03d}".format(int(angle)) + ".exr"))

# 후면 노멀맵 생성 시작
mesh.matrix_world = backside_rot_mat @ new_orig_loc_mat @ rot_mat @ orig_scale_mat

bpy.ops.render.render(write_still=False)
os.rename(os.path.join(save_folder_path, "Image0001.exr"), os.path.join(save_folder_path, back_normal_map_filename + "_" + "{0:03d}".format(int(angle)) + ".exr"))