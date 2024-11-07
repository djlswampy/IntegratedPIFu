import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import os
import sys

# 상위 디렉토리를 sys.path에 추가하여 모듈을 임포트할 수 있게 함
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Numpy를 다시 불러오는 중복 코드
import numpy as np

# 필요한 변수들을 설정합니다.
source_mesh_folder = "apps/results/Date_11_Jul_22_Time_01_12_07"  # 불러올 메쉬 파일이 있는 폴더 경로
use_Buff_dataset = False  # Buff 데이터셋 사용 여부

# 소스 메쉬 폴더 경로 출력
print('source_mesh_folder:', source_mesh_folder)

# 사용할 샘플 개수 설정
num_samples_to_use = 10000


# 사전 준비된 테스트 메쉬에 대해 Chamfer 거리 및 Point-to-Surface 거리를 계산하는 함수 정의
def run_test_mesh_already_prepared():
        # 테스트할 subject 목록을 설정. Buff 데이터셋을 사용할 경우 해당 파일을, 그렇지 않으면 test_set_list.txt를 불러옴
        if use_Buff_dataset:
            test_subject_list = np.loadtxt("buff_subject_testing.txt", dtype=str).tolist()
        else:
            test_subject_list = np.loadtxt("test_set_list.txt", dtype=str).tolist()

        # 총 Chamfer 거리와 Point-to-Surface 거리를 저장할 리스트 초기화
        total_chamfer_distance = []
        total_point_to_surface_distance = []

        # 각 테스트 subject에 대해 Chamfer 거리와 Point-to-Surface 거리 계산
        for subject in test_subject_list:
            # Buff 데이터셋을 사용하는 경우와 그렇지 않은 경우의 GT(Ground Truth) 메쉬 경로 설정
            if use_Buff_dataset:
                folder_name = subject.split('_')[0]
                attire_name = subject.split('_', 1)[1]
                GT_mesh_path = os.path.join("<Path to BUFF folder>", folder_name, attire_name + '.ply')
            else:
                GT_mesh_path = os.path.join("rendering_script", 'THuman2.0_Release', subject, '%s.obj' % subject)

            # 테스트할 소스 메쉬 경로 설정
            source_mesh_path = os.path.join(source_mesh_folder, 'test_%s.obj' % subject)
            # 경로가 없으면 '_1'이 추가된 파일을 찾음
            if not os.path.exists(source_mesh_path):
                source_mesh_path = os.path.join(source_mesh_folder, 'test_%s_1.obj' % subject)

            # Ground Truth(GT) 메쉬와 소스 메쉬를 trimesh 라이브러리를 사용하여 불러옴
            GT_mesh = trimesh.load(GT_mesh_path)
            source_mesh = trimesh.load(source_mesh_path)

            # 각 메쉬에 대해 Chamfer 거리와 Point-to-Surface 거리 계산
            chamfer_distance = get_chamfer_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use)
            point_to_surface_distance = get_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use)

            # 계산된 거리를 출력하고 리스트에 추가
            print("subject: ", subject)
            print("chamfer_distance: ", chamfer_distance)
            total_chamfer_distance.append(chamfer_distance)

            print("point_to_surface_distance: ", point_to_surface_distance)
            total_point_to_surface_distance.append(point_to_surface_distance)

        # 모든 subject에 대해 Chamfer 거리와 Point-to-Surface 거리의 평균을 계산
        average_chamfer_distance = np.mean(total_chamfer_distance)
        average_point_to_surface_distance = np.mean(total_point_to_surface_distance)

        # 평균 Chamfer 거리와 Point-to-Surface 거리를 출력
        print("average_chamfer_distance:", average_chamfer_distance)
        print("average_point_to_surface_distance:", average_point_to_surface_distance)


# Chamfer 거리를 계산하는 함수 정의
def get_chamfer_dist(src_mesh, tgt_mesh, num_samples=10000):
    # 소스와 타겟 메쉬의 표면에서 num_samples 만큼 샘플링하여 점을 얻음
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    # 소스 점들에서 타겟 메쉬의 가장 가까운 점까지의 거리 계산
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    # 타겟 점들에서 소스 메쉬의 가장 가까운 점까지의 거리 계산
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    # NaN 값을 0으로 대체하여 계산 오류 방지
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    # 각 거리의 제곱 평균 계산
    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    # 두 거리의 평균을 Chamfer 거리로 반환
    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    return chamfer_dist


# Point-to-Surface 거리를 계산하는 함수 정의
def get_surface_dist(src_mesh, tgt_mesh, num_samples=10000):
    # 소스 메쉬에서 num_samples 만큼 샘플링하여 표면 점을 얻음
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)

    # 소스 점들에서 타겟 메쉬의 가장 가까운 점까지의 거리 계산
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)

    # NaN 값을 0으로 대체하여 계산 오류 방지
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0

    # 거리의 제곱 평균을 계산하여 Point-to-Surface 거리로 반환
    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    return src_tgt_dist


# Chamfer 거리와 Point-to-Surface 거리를 함께 계산하는 함수 정의
def quick_get_chamfer_and_surface_dist(src_mesh, tgt_mesh, num_samples=10000):
    # 소스와 타겟 메쉬의 표면에서 num_samples 만큼 샘플링하여 점을 얻음
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    # 소스 점들에서 타겟 메쉬의 가장 가까운 점까지의 거리 계산
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    # 타겟 점들에서 소스 메쉬의 가장 가까운 점까지의 거리 계산
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    # NaN 값을 0으로 대체하여 계산 오류 방지
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    # 각 거리의 제곱 평균 계산
    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    # Chamfer 거리와 Point-to-Surface 거리 계산
    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    surface_dist = src_tgt_dist

    # Chamfer 거리와 Point-to-Surface 거리를 함께 반환
    return chamfer_dist, surface_dist


# 메인 함수로 run_test_mesh_already_prepared 실행
if __name__ == "__main__":
    run_test_mesh_already_prepared()
