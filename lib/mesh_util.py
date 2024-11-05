from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid


def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    네트워크가 예측한 SDF로부터 메쉬를 재구성합니다.
    :param net: BasePixImpNet 객체. 이미지 필터를 미리 호출해야 합니다.
    :param cuda: CUDA 장치
    :param calib_tensor: 캘리브레이션 텐서
    :param resolution: 그리드 셀의 해상도
    :param b_min: 바운딩 박스의 최소 좌표 [x_min, y_min, z_min]
    :param b_max: 바운딩 박스의 최대 좌표 [x_max, y_max, z_max]
    :param use_octree: 옥트리 가속을 사용할지 여부
    :param num_samples: 각 GPU 반복에서 쿼리할 포인트 수
    :return: marching cubes 결과.
    '''
    # 먼저 그리드를 해상도로 생성하고
    # 그리드 좌표를 실제 월드 좌표 xyz로 변환하는 행렬을 생성합니다.
    # coords(coordinates(whkvy))는 3D 공간에 균일하게 배치된 격자점의 좌표 집합입니다.
    # mat은 그리드 좌표계를 실제 월드 좌표계로 변환하는 변환 행렬입니다.
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    # 셀 평가를 위한 람다 함수를 정의합니다.
    def eval_func(points):
        points = np.expand_dims(points, axis=0)  # 포인트 배열에 새로운 축 추가
        # points = np.repeat(points, net.num_views, axis=0)
        points = np.repeat(points, 1, axis=0)  # 포인트를 1번 반복하여 확장
        samples = torch.from_numpy(points).to(device=cuda).float()  # 텐서로 변환하여 CUDA로 이동
        net.query(samples, calib_tensor)  # 네트워크에 쿼리 수행
        pred = net.get_preds()[0][0]  # 예측 값 추출
        
        return pred.detach().cpu().numpy()  # numpy 배열로 반환

    # 그리드 평가를 수행합니다.
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)  # 옥트리 사용
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)  # 기본 그리드 사용

    # 마지막으로 marching cubes를 수행합니다.
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, 0.5, method="lewiner")
        # 정점을 월드 좌표계로 변환
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]  # 좌표 변환
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')  # 에러 발생 시 메시지 출력
        return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()