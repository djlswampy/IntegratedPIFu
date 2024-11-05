import os, sys, time
import bpy  # Blender의 Python API
import numpy as np
import shutil
from math import radians
from mathutils import Matrix, Vector  # Blender에서 사용하는 수학 유틸리티

# 실행 지침 (예: 주제 '0501'을 '90도'로 회전시킨 각도에서 실행):
# 이 폴더('rendering_script')로 이동한 후 아래 명령 실행:
# blender blank.blend -b -P render_full_mesh.py -- 0501 90

use_gpu = True  # GPU를 사용할지 여부를 설정하는 플래그

# 현재 경로를 절대 경로로 설정
curpath = os.path.abspath(os.path.dirname("."))
sys.path.insert(0, curpath)  # 현재 경로를 시스템 경로에 추가하여 패키지 접근 가능하게 설정

argv = sys.argv  # Blender 실행 시 명령줄 인자 가져오기
argv = argv[argv.index("--") + 1:]  # '--' 이후의 인자만 사용 (주제와 각도)

subject = argv[0]  # 첫 번째 인자를 주제(subject)로 설정
angle = int(argv[1])  # 두 번째 인자는 회전 각도로 설정
base_path = argv[2]  # 세 번째 인자를 경로로 설정

# 각도가 360도 이상인 경우, 360도에서 뺀 값을 사용
if angle >= 360:
    angle = angle - 360

light_energy = 0.5e+02  # 조명의 에너지를 설정 (빛의 강도)

RESOLUTION = 1024  # 렌더링 해상도를 설정 (1024x1024)

BUFFER_PATH = "buffer_fixed_full_mesh"  # 출력 폴더 경로 설정
image_filename = "rendered_image"  # 렌더링된 이미지 파일 이름
mask_filename = "rendered_mask"  # 렌더링된 마스크 파일 이름

# THuman2.0_Release 폴더에서 주제(subject)에 해당하는 OBJ 파일 경로를 설정
shape_file = os.path.join(base_path, subject, subject + ".obj")

# 주제에 해당하는 저장 폴더 경로 설정
save_folder_path = os.path.join(curpath, BUFFER_PATH, subject)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)  # 폴더가 없으면 새로 생성


# OBJ 파일에서 메쉬 데이터를 읽어오는 함수
def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []  # 정점 데이터를 저장할 리스트
    norm_data = []  # 노멀 데이터를 저장할 리스트
    uv_data = []  # 텍스처 좌표 데이터를 저장할 리스트

    face_data = []  # 면 데이터를 저장할 리스트
    face_norm_data = []  # 면에 대한 노멀 데이터 저장 리스트
    face_uv_data = []  # 면에 대한 텍스처 좌표 저장 리스트

    # 파일 읽기
    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file

    # 파일을 한 줄씩 읽어서 OBJ 파일의 구조를 분석
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):  # 주석은 무시
            continue
        values = line.split()  # 라인의 데이터를 공백으로 분리
        if not values:
            continue

        # 정점 데이터 (v로 시작하는 라인)
        if values[0] == 'v':
            v = list(map(float, values[1:4]))  # x, y, z 좌표를 float로 변환 후 저장
            vertex_data.append(v)
        # 노멀 데이터 (vn으로 시작하는 라인)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))  # 노멀 벡터를 저장
            norm_data.append(vn)
        # 텍스처 좌표 (vt로 시작하는 라인)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))  # 텍스처 좌표를 저장
            uv_data.append(vt)
        # 면 정보 (f로 시작하는 라인)
        elif values[0] == 'f':
            # 쿼드 메쉬일 경우
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))  # 첫 번째 정점 인덱스
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))  # 두 번째 정점 인덱스
                face_data.append(f)
            # 삼각형 메쉬일 경우
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # 텍스처 좌표 처리
            if len(values[1].split('/')) >= 2:
                if len(values) > 4:  # 쿼드 메쉬일 경우
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                elif len(values[1].split('/')[1]) != 0:  # 삼각형 메쉬일 경우
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)

            # 노멀 처리
            if len(values[1].split('/')) == 3:
                if len(values) > 4:  # 쿼드 메쉬일 경우
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                elif len(values[1].split('/')[2]) != 0:  # 삼각형 메쉬일 경우
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)  # 정점 데이터를 numpy 배열로 변환
    faces = np.array(face_data) - 1  # 면 데이터는 1을 빼서 0부터 시작하도록 맞춤

    # 텍스처와 노멀 데이터를 모두 사용할 경우
    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)  # 노멀 데이터가 없으면 계산
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    # 텍스처만 사용할 경우
    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    # 노멀만 사용할 경우
    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces  # 기본적으로 정점과 면 데이터만 반환


# 회전 행렬을 생성하는 함수 (x, y, z 축에 대해 회전)
def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    # X축 회전 행렬
    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    # Y축 회전 행렬
    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    # Z축 회전 행렬
    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    # X, Y, Z 축의 회전 행렬을 곱하여 최종 회전 행렬을 반환
    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


# Blender의 설정을 준비하는 함수
def setupBlender(angle_to_rotate):
    global overall_vertices  # 전역 변수로 선언된 메쉬의 정점 데이터를 사용

    scene = bpy.context.scene  # 현재 Blender 씬을 가져옴
    camera = bpy.data.objects["Camera"]  # 카메라 객체 가져옴

    bpy.context.scene.render.engine = 'CYCLES'  # Cycles 렌더 엔진을 사용

    if use_gpu:
        # GPU 설정을 위한 장치 구성
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # GPU를 CUDA로 설정
        bpy.context.scene.cycles.device = "GPU"  # GPU 사용 설정
        bpy.context.preferences.addons["cycles"].preferences.get_devices()  # GPU 장치 탐지

    # 정점 데이터를 각도에 따라 회전 (Y축 기준)
    R = make_rotate(0, radians(angle_to_rotate), 0)
    vertices = np.matmul(R, overall_vertices.T).T  # 회전된 정점 데이터

    vmin = vertices.min(0)  # 최소 좌표값
    vmax = vertices.max(0)  # 최대 좌표값

    vcenter_vertical = (vmax[1] + vmin[1]) / 2  # 수직 중앙 좌표

    vrange = vmax - vmin  # 좌표 범위 계산
    orthographic_scale_to_use = np.max(vrange) * 1.1  # 카메라의 Orthographic Scale을 좌표 범위의 1.1배로 설정

    camera.data.clip_end = 10000000000  # 클리핑 거리를 크게 설정하여 메쉬가 잘리지 않도록 함
    camera.data.type = "ORTHO"  # 카메라를 Orthographic 타입으로 설정

    # 카메라 위치 설정 (정면에서 1000의 거리)
    camera.location[0] = 0
    camera.location[1] = 0
    camera.location[2] = 1000.0

    # 메쉬의 중심점을 계산하여 카메라의 시점이 맞도록 설정
    small_x_range = vrange[0] * 0.005
    small_y_range = vrange[1] * 0.005
    temp_bool_y = (vertices[:, 1] > (vcenter_vertical - small_y_range)) & (vertices[:, 1] < (vcenter_vertical + small_y_range))
    horizontal_line = vertices[temp_bool_y, :]  # 수평선 좌표 추출
    vcenter_horizontal = np.median(horizontal_line, 0)[0]  # 수평 중앙 좌표 계산
    temp_bool_x = (vertices[:, 0] > (vcenter_horizontal - small_x_range)) & (vertices[:, 0] < (vcenter_horizontal + small_x_range))
    vertical_line = vertices[temp_bool_x, :]  # 수직선 좌표 추출
    temp_bool_x_and_y = temp_bool_x & temp_bool_y
    small_cube = vertices[temp_bool_x_and_y, :]  # 중앙에 위치한 작은 큐브 추출
    pt_nearest_to_cam = small_cube.max(0)  # 카메라에 가장 가까운 점의 좌표 추출
    z_coor = pt_nearest_to_cam[2]  # Z축 좌표

    mesh_center = np.array([vcenter_horizontal, vcenter_vertical, z_coor])  # 메쉬의 중심 좌표

    # 카메라의 회전 설정 (각도는 0)
    camera.rotation_euler[0] = 0.0 / 180 * np.pi
    camera.rotation_euler[1] = 0.0 / 180 * np.pi
    camera.rotation_euler[2] = 0.0 / 180 * np.pi

    camera.data.ortho_scale = orthographic_scale_to_use  # 카메라의 Orthographic Scale을 설정

    # 조명 설정
    light = bpy.data.objects["Light"]
    light.data.energy = light_energy  # 조명 강도 설정

    light.location[0] = (0 + vmin[0] - vcenter_horizontal) / 2
    light.location[1] = (0 + vmax[1] - vcenter_vertical) / 2
    light.location[2] = vmax[2] * 8

    # 컴포지션 설정 시작
    scene.render.use_compositing = True
    scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    rl = bpy.context.scene.node_tree.nodes["Render Layers"]

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = save_folder_path  # 저장 경로 설정

    # 렌더 레이어와 파일 출력 노드를 연결
    tree.links.new(rl.outputs["Image"], fo.inputs["Image"])

    # 렌더링 해상도 설정
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION

    # 배경 색상 설정 (검정색)
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    r, g, b = 0, 0, 0
    bg.inputs[0].default_value[:3] = (r, g, b)

    return scene, camera, fo, mesh_center  # 설정 완료 후 반환


# 메쉬의 마스크를 생성하는 함수
def renderMask(mesh):
    pass_index = 1  # 메쉬의 ID 값 설정
    mesh.pass_index = pass_index

    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True  # 객체 인덱스 사용 설정

    bpy.context.scene.render.engine = 'CYCLES'  # Cycles 렌더 엔진 사용

    if use_gpu:
        # GPU 설정
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()

    tree = bpy.context.scene.node_tree
    rl = bpy.context.scene.node_tree.nodes["Render Layers"]

    # ID 마스크 노드 추가
    id_mask_node = tree.nodes.new("CompositorNodeIDMask")
    id_mask_node.index = pass_index

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = save_folder_path  # 저장 경로 설정

    # 객체 인덱스와 마스크 노드를 연결
    tree.links.new(rl.outputs["IndexOB"], id_mask_node.inputs["ID value"])
    tree.links.new(id_mask_node.outputs["Alpha"], fo.inputs["Image"])

    # 배경 색상 설정
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    r, g, b = 0, 0, 0
    bg.inputs[0].default_value[:3] = (r, g, b)

    # 해상도 설정
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION

    # 렌더링 수행
    bpy.ops.render.render(write_still=False)

    # 렌더링이 완료된 후 링크를 제거하고 기본 이미지를 출력
    tree.links.remove(tree.links[-1])
    tree.links.new(rl.outputs["Image"], fo.inputs["Image"])


# OBJ 파일을 Blender로 가져옴
# bpy.ops.import_scene.obj(filepath=shape_file)
bpy.ops.wm.obj_import(filepath=shape_file)

mesh_name = ""
# 메쉬의 이름을 주제(subject)와 매칭하여 가져옴
for _object in bpy.data.objects:
    if subject in _object.name:
        mesh_name = _object.name

mesh = bpy.data.objects[mesh_name]  # 방금 불러온 메쉬 가져오기

# 전역 변수에 메쉬 데이터 저장
overall_vertices, _ = load_obj_mesh(shape_file, with_normal=False, with_texture=False)

# Blender 설정 시작
scene, camera, fo, mesh_center = setupBlender(angle_to_rotate=angle)

# 메쉬의 위치 및 회전을 설정
rot_mat = Matrix.Rotation(radians(angle), 4, 'Y')  # Y축을 기준으로 회전 행렬 생성

# 원래 메쉬의 위치, 회전, 스케일 분해
orig_loc, orig_rot, orig_scale = mesh.matrix_world.decompose()

# 메쉬의 중심점에 맞춰 위치 변경
vec = Vector((-mesh_center[0], -mesh_center[1], -mesh_center[2]))
new_orig_loc_mat = Matrix.Translation(vec)

# 원래 스케일을 적용
orig_scale_mat = Matrix.Scale(orig_scale[0], 4, (1, 0, 0)) * Matrix.Scale(orig_scale[1], 4, (0, 1, 0)) * Matrix.Scale(orig_scale[2], 4, (0, 0, 1))

# 새롭게 계산된 변환 행렬을 메쉬에 적용
mesh.matrix_world = new_orig_loc_mat @ rot_mat @ orig_scale_mat

# 렌더링 수행
bpy.ops.render.render(write_still=False)

# 렌더링된 이미지 파일 이름을 변경하여 저장
os.rename(os.path.join(save_folder_path, "Image0001.png"), os.path.join(save_folder_path, image_filename + "_" + "{0:03d}".format(int(angle)) + ".png"))

# 메쉬의 마스크 생성
renderMask(mesh)

# 마스크 파일 이름을 변경하여 저장
os.rename(os.path.join(save_folder_path, "Image0001.png"), os.path.join(save_folder_path, mask_filename + "_" + "{0:03d}".format(int(angle)) + ".png"))

# 메쉬의 파라미터 (중심, 회전 행렬, 스케일) 저장
rotation_matrix = make_rotate(0, radians(angle), 0)  # 회전 행렬 생성
scale_factor = RESOLUTION / camera.data.ortho_scale  # 스케일 값 계산

params_dic = {'center': mesh_center, 'R': rotation_matrix, 'scale_factor': scale_factor}  # 파라미터 딕셔너리 생성

# 파라미터를 npy 파일로 저장
file_name_to_save = os.path.join(save_folder_path, "rendered_params_" + "{0:03d}".format(int(angle)) + ".npy")
np.save(file_name_to_save, params_dic)