'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([-1, -1, -1]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ] # coords has shape of (3, 512, 512, 512)
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX   # coords_matrix transform points from 'resolution' dimension to 'bounding box' dimension
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)   # coords still has shape of (3, 512, 512, 512)
    return coords, coords_matrix  # coords_matrix has shape of [4,4] and it transforms 3D points from 'resolution' dimension to 'bounding box' dimension


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    # `points`의 점 개수를 확인하여 `num_pts`에 저장합니다.
    # points.shape[1]은 점의 총 개수를 나타냅니다.
    num_pts = points.shape[1]
    
    # 결과 SDF 값을 저장할 배열을 초기화합니다. 모든 점에 대해 SDF 값을 계산하여 저장할 공간입니다.
    # 처음에는 모두 0으로 초기화됩니다.
    sdf = np.zeros(num_pts)
    
    # `points`를 한 번에 평가하기에 너무 클 수 있으므로, `num_samples` 크기로 나누어 평가합니다.
    # 전체 점 개수에서 배치 크기 `num_samples`을 나눈 값으로 몇 배치가 필요한지 계산합니다.
    num_batches = num_pts // num_samples

    # 각 배치별로 `eval_func`를 통해 SDF 값을 평가하고, 그 결과를 `sdf` 배열에 저장합니다.
    # `num_batches`만큼 반복합니다.
    for i in range(num_batches):
        # i번째 배치의 인덱스 범위만큼 `points`에서 데이터를 슬라이싱하여 `eval_func`에 전달합니다.
        # `eval_func(points[:, i * num_samples:i * num_samples + num_samples])`는
        # `points`에서 현재 배치에 해당하는 점들을 평가합니다.
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    
    # 위에서 `num_batches`로 나누어 떨어지지 않는 경우, 마지막에 남은 데이터를 처리합니다.
    # 예를 들어, `num_pts`가 `num_samples`로 나누어 떨어지지 않는 경우, 나머지 점들을 평가합니다.
    if num_pts % num_samples:
        # 마지막 배치의 인덱스 범위를 설정하여 `eval_func`에 전달하고, 남은 점들을 평가합니다.
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])
    
    # 평가 결과가 모두 저장된 `sdf` 배열을 반환합니다.
    return sdf


def batch_eval_tensor(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.size(1)

    num_batches = num_pts // num_samples
    vals = []
    for i in range(num_batches):
        vals.append(eval_func(points[:, i * num_samples:i * num_samples + num_samples]))
    if num_pts % num_samples:
        vals.append(eval_func(points[:, num_batches * num_samples:]))

    return np.concatenate(vals,0)


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    # coords 배열의 해상도를 가져옵니다. 예를 들어 (3, 256, 256, 256)일 경우, (256, 256, 256)만 가져옵니다.
    resolution = coords.shape[1:4]
    
    # coords 배열을 2차원으로 변환하여 [3, -1] 형태로 재구성합니다.
    # 이로써 3 x (256*256*256) 모양이 됩니다.
    coords = coords.reshape([3, -1])

    # batch_eval 함수를 통해 SDF 값을 계산합니다.
    # 모든 포인트에 대해 SDF 값을 계산하여 sdf에 저장합니다.
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)

    # 원래의 해상도(resolution)로 SDF 배열을 재구성하여 반환합니다.
    return sdf.reshape(resolution)


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.05,
                     num_samples=512 * 512 * 512):
    # coords 배열의 해상도를 가져옵니다. 예를 들어 (3, 256, 256, 256)일 경우 (256, 256, 256)입니다.
    resolution = coords.shape[1:4]  # coords의 shape은 (3, 256, 256, 256)
    
    # SDF 값을 저장할 배열을 초기화합니다. 크기는 resolution과 동일하며, 모든 요소를 0으로 초기화합니다.
    sdf = np.zeros(resolution)  # (256, 256, 256)

    # 아직 처리되지 않은 그리드 셀을 표시할 배열을 생성하고, True로 초기화합니다.
    notprocessed = np.zeros(resolution, dtype=bool) # 크기는 (256, 256, 256)
    notprocessed[:-1,:-1,:-1] = True  # 마지막 인덱스를 제외하고 True로 설정
    # 그리드 마스크를 초기화합니다. 여기에 각 스텝마다 평가할 셀들이 표시됩니다.
    grid_mask = np.zeros(resolution, dtype=bool) # 크기는 (256, 256, 256)

    # 초기 해상도에서 resolution을 나누어 서브샘플링할 단계 크기를 계산합니다.
    reso = resolution[0] // init_resolution  # 예: 256/64 = 4

    # reso를 절반씩 줄여가며 모든 셀이 평가될 때까지 반복합니다.
    while reso > 0:
        # 그리드를 서브샘플링하여 해당 스텝에서 평가할 셀 위치를 지정합니다.
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # 아직 처리되지 않은 셀 중 이번 스텝에서 평가할 셀을 선택합니다.
        test_mask = np.logical_and(grid_mask, notprocessed)
        
        # test_mask가 True인 좌표들만 선택해 `points` 배열을 생성합니다.
        points = coords[:, test_mask]

        # 선택한 셀에 대해 batch_eval 함수를 사용하여 SDF 값을 계산합니다.
        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        # 평가가 끝난 셀을 처리 완료로 표시합니다.
        notprocessed[test_mask] = False

        # 그리드 해상도가 최소 단계로 내려가면 반복문을 종료합니다.
        if reso <= 1:
            break

        # 현재 해상도에서 x, y, z 좌표의 그리드 포인트를 생성합니다.
        x_grid = np.arange(0, resolution[0], reso)
        y_grid = np.arange(0, resolution[1], reso)
        z_grid = np.arange(0, resolution[2], reso)

        # 현재 스텝에서 평가된 SDF 값을 가져옵니다.
        v = sdf[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        # 8개의 서브그리드로 나누어, 각 셀에 대한 SDF 값을 가져옵니다.
        v0 = v[:-1,:-1,:-1]
        v1 = v[:-1,:-1,1:]
        v2 = v[:-1,1:,:-1]
        v3 = v[:-1,1:,1:]
        v4 = v[1:,:-1,:-1]
        v5 = v[1:,:-1,1:]
        v6 = v[1:,1:,:-1]
        v7 = v[1:,1:,1:]

        # 서브그리드의 중앙 위치 좌표를 생성합니다.
        x_grid = x_grid[:-1] + reso // 2
        y_grid = y_grid[:-1] + reso // 2
        z_grid = z_grid[:-1] + reso // 2

        # 아직 처리되지 않은 중앙 위치 그리드의 값을 가져옵니다.
        nonprocessed_grid = notprocessed[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        # 8개 서브그리드의 최소값과 최대값을 계산하여 v의 범위를 구합니다.
        v = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], 0)
        v_min = v.min(0)
        v_max = v.max(0)
        v = 0.5 * (v_min + v_max)

        # 최소-최대 차이가 threshold 이하인 그리드를 스킵하도록 마스크 설정
        skip_grid = np.logical_and(((v_max - v_min) < threshold), nonprocessed_grid)

        # 현재 단계 크기를 n_x, n_y, n_z에 저장하여 그리드 내 서브그리드 갯수를 나타냅니다.
        n_x = resolution[0] // reso
        n_y = resolution[1] // reso
        n_z = resolution[2] // reso

        # skip_grid가 True인 셀은 평가 없이 보간된 값을 사용하여 SDF 값을 설정합니다.
        xs, ys, zs = np.where(skip_grid)
        for x, y, z in zip(xs * reso, ys * reso, zs * reso):
            sdf[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = v[x//reso, y//reso, z//reso]
            notprocessed[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = False

        # 해상도를 절반으로 줄여가며 반복
        reso //= 2

    # 최종적으로 sdf 배열을 반환하여 SDF가 포함된 3D 그리드로 반환합니다.
    return sdf.reshape(resolution)