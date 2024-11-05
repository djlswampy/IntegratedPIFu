import os
import random
import numpy as np 
from PIL import Image, ImageOps
import cv2
import torch
import json
import trimesh
import logging
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from numpy.linalg import inv

# trimesh의 로그 레벨을 40으로 설정하여 로그 출력을 제한
log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir, training_subject_list = None):    
    # root_dir에 있는 폴더 리스트를 가져옴
    folders = os.listdir(root_dir)
    
    # 빈 딕셔너리 생성
    meshs = {}
    # 각 폴더를 순회하며 메쉬 데이터를 로드
    for index, f in enumerate(folders):
        if f == ".DS_Store": # macOS에서 자동 생성되는 파일 무시
            continue

        if f not in training_subject_list:  # 학습용 데이터셋에 포함되지 않은 경우 무시
            continue

        # 메쉬 파일을 로드하여 딕셔너리에 저장 (메쉬 이름을 키로 사용)
        meshs[f] = trimesh.load(os.path.join(root_dir, f, '%s.obj' % f))
        print(f"Loaded mesh for subject {index + 1}/{len(training_subject_list)}: {f}")

    # 메쉬들이 저장된 딕셔너리 반환(key = subject, value = mesh)
    return meshs

class TrainDataset(Dataset):
    def __init__(self, opt, projection='orthogonal', phase = 'train', evaluation_mode=False, validation_mode=False):
        self.opt = opt
        self.projection_mode = projection
        self.training_subject_list = np.loadtxt("integratedpifu_train_set_list.txt", dtype=str) # 학습용 데이터셋 목록 로드

        #if opt.debug_mode:
        #    self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/fake_train_set_list.txt", dtype=str)

        self.evaluation_mode = evaluation_mode # 평가 모드 여부 설정       
        self.validation_mode = validation_mode # 검증 모드 여부 설정

        self.phase = phase # 현재 단계 (학습 or 검증)
        self.is_train = (self.phase == 'train') # 학습 모드인지 여부
        
        # 검증용 데이터셋을 사용할지 여부 확인
        if self.opt.useValidationSet:
            # 학습용 데이터셋을 섞고 10%를 검증용 데이터로 사용
            indices = np.arange( len(self.training_subject_list) )
            np.random.seed(10)
            np.random.shuffle(indices)
            lower_split_index = round( len(self.training_subject_list)* 0.1 )
            val_indices = indices[:lower_split_index]
            train_indices = indices[lower_split_index:]

            if self.validation_mode:
                self.training_subject_list = self.training_subject_list[val_indices] # 검증용 데이터셋 설정
                self.is_train = False
            else:
                self.training_subject_list = self.training_subject_list[train_indices] # 학습용 데이터셋 설정

        self.training_subject_list = self.training_subject_list.tolist()

        # 평가 모드일 경우 test_set_list를 사용하여 데이터를 교체
        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("integratedpifu_test_set_list.txt", dtype=str).tolist()
            self.is_train = False

        # 루트 디렉터리 및 메쉬 디렉터리 설정
        self.root = "rendering_script/buffer_fixed_full_mesh"
        self.mesh_directory = "/home/public/data/integratedpifu_data/thuman_data_sample"
        
        # 평가 모드가 아닌 경우 메쉬 로드
        if (evaluation_mode):
            pass 
        else:
            # 학습용 메쉬를 로드하여 딕셔너리에 저장
            self.mesh_dic = load_trimesh(self.mesh_directory,  training_subject_list = self.training_subject_list)  # a dict containing the meshes of all the CAD models.

        # groundtruth 노멀 맵을 사용할지 여부에 따라 경로 설정
        if self.opt.use_groundtruth_normal_maps:
            self.normal_directory_high_res = "rendering_script/buffer_normal_maps_of_full_mesh"
        else:
            self.normal_directory_high_res = "pretrained_normal_maps"

        # GT 깊이 맵을 사용할지 여부에 따라 경로 설정
        if self.opt.useGTdepthmap:
            self.depth_map_directory = "rendering_script/buffer_depth_maps_of_full_mesh"
        else:

            self.depth_map_directory = "trained_depth_maps" # New version (Depth maps trained with only normal - Second Stage maps)

        # groundtruth 인간 파싱 맵을 사용할지 여부에 따라 경로 설정
        if self.opt.use_groundtruth_human_parse_maps:
            self.human_parse_map_directory = "rendering_script/render_human_parse_results"
        else:
            self.human_parse_map_directory = "trained_parse_maps"

        # 주제 리스트와 이미지 파일 리스트 초기화
        self.subjects = self.training_subject_list  
        self.load_size = self.opt.loadSize    
        self.num_sample_inout = self.opt.num_sample_inout 

        # 주제별로 이미지 경로를 찾고 리스트에 저장
        self.img_files = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.root, training_subject)
            subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
            self.img_files = self.img_files + subject_render_paths_list

        # 이미지 파일 정렬
        self.img_files = sorted(self.img_files)

        # PIL 이미지에서 텐서로 변환하는 함수 설정 (정규화 포함)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), # ToTensor는 입력 데이터를 (C x H x W) 형태로 변환하며, 각 차원의 값 범위는 [0.0, 1.0]입니다.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 각 차원을 평균 0.5, 표준편차 0.5로 정규화합니다. 최종적으로 각 차원의 값 범위는 [-1, 1]이 됩니다.
        ])

    # 데이터셋 크기를 반환
    def __len__(self):
        return len(self.img_files)

    # 샘플링 방법을 선택하는 함수
    def select_sampling_method(self, subject, calib, b_min, b_max, R = None):
        compensation_factor = 4.0

        # 주어진 주제에 대한 메쉬 로드
        mesh = self.mesh_dic[subject] # 1개의 주제/CAD의 메쉬

        # note, this is the solution for when dataset is "THuman"
        # adjust sigma according to the mesh's size (measured using the y-coordinates)
        # 메쉬의 y축 크기에 따라 sigma를 조정
        y_length = np.abs(np.max(mesh.vertices, axis=0)[1])  + np.abs(np.min(mesh.vertices, axis=0)[1] )
        sigma_multiplier = y_length/188

        # 메쉬 표면에서 샘플링한 표면 포인트와 면 인덱스를 반환
        surface_points, face_indices = trimesh.sample.sample_surface(mesh, int(compensation_factor * 4 * self.num_sample_inout) )  # self.num_sample_inout is no. of sampling points and is default to 8000.

        # 이미지 공간 내에 무작위 포인트 추가
        length = b_max - b_min # has shape of (3,)
        
        # DOS를 사용하지 않는 경우
        if not self.opt.useDOS:
            random_points = np.random.rand( int(compensation_factor * self.num_sample_inout // 4) , 3) * length + b_min # [compensation_factor*num_sample_inout/4, 3] 크기의 랜덤 포인트 생성
            surface_points_shape = list(surface_points.shape)
            random_noise = np.random.normal(scale= self.opt.sigma_low_resolution_pifu * sigma_multiplier, size=surface_points_shape) # 정규분포에 따라 표면 근처에 랜덤 노이즈 추가
            sample_points_low_res_pifu = surface_points + random_noise # sample_points는 표면 근처의 점들. sigma는 정규분포의 표준편차를 의미
            sample_points_low_res_pifu = np.concatenate([sample_points_low_res_pifu, random_points], 0) # [compensation_factor*4.25*num_sample_inout, 3] 크기로 결합
            np.random.shuffle(sample_points_low_res_pifu) # 점들을 랜덤으로 섞음
            
            inside_low_res_pifu = mesh.contains(sample_points_low_res_pifu) # return a boolean 1D array of size (num of sample points,)
            inside_points_low_res_pifu = sample_points_low_res_pifu[inside_low_res_pifu]

        # DOS를 사용하는 경우
        if self.opt.useDOS:
            num_of_pts_in_section = self.num_sample_inout // 3  # 샘플링할 점들의 수를 3등분으로 나눔

            normal_vectors = mesh.face_normals[face_indices] # 메쉬의 표면 노말 벡터들 [num_of_sample_pts, 3]

            directional_vector = np.array([[0.0,0.0,1.0]]) # Z방향으로 향하는 방향 벡터 (1x3 크기)
            directional_vector = np.matmul(inv(R), directional_vector.T) # 방향 벡터를 회전 행렬로 변환 (3x1)

            # 점들의 내적 계산
            normal_vectors_to_use = normal_vectors[ 0: self.num_sample_inout ,:] # 샘플링된 표면 노말 벡터들 선택
            dot_product = np.matmul(directional_vector.T, normal_vectors_to_use.T ) # Z방향과 노말 벡터 간의 내적 계산 [1 x num_of_sample_pts]

            # 내적 결과를 기반으로, 후면을 향하는 표면은 -1, 전면을 향하는 표면은 1로 설정
            dot_product[dot_product<0] = -1.0 # 후면을 향하는 점들
            dot_product[dot_product>=0] = 1.0 # 전면을 향하는 점들
            z_displacement = np.matmul(dot_product.T, directional_vector.T) # Z축 방향으로 점을 이동시킴 [num_of_sample_pts, 3]. 후면을 향하는 점들은 뒤로, 전면을 향하는 점들은 앞으로 이동

            # 정규분포를 사용하여 표면에서의 이동량을 설정
            normal_sigma = np.random.normal(loc=0.0, scale= 1.0, size= [4 * self.num_sample_inout, 1] ) # [num_of_sample_pts, 1] 크기의 정규분포 샘플
            normal_sigma_mask = (normal_sigma[:,0] < 1.0)  &  (normal_sigma[:,0] > -1.0) # 이동량이 -1과 1 사이에 있는지 확인
            normal_sigma = normal_sigma[normal_sigma_mask,:] # 유효한 이동량만 선택
            normal_sigma = normal_sigma[0:self.num_sample_inout, :] # 샘플 수에 맞춰 크기를 조정
            surface_points_with_normal_sigma = surface_points[ 0:self.num_sample_inout ,:] - z_displacement * sigma_multiplier * normal_sigma * 2.0 # 표면에서 Z축 방향으로 점들을 이동시킴
            labels_with_normal_sigma = normal_sigma.T / 2.0 * 0.8 # 이동량을 0.8 범위 내에서 설정, -0.4에서 0.4 사이의 범위로 설정
            labels_with_normal_sigma = labels_with_normal_sigma + 0.5 # 최종적으로 0.1에서 0.9 사이의 값으로 조정 [1, self.num_sample_inout]

            # 메쉬의 깊이 안쪽에 있는 점들을 생성
            num_of_way_inside_pts = round(self.num_sample_inout * self.opt.ratio_of_way_inside_points) # 메쉬 깊숙한 곳에 위치한 점들의 수를 계산
            way_inside_pts = surface_points[0: num_of_way_inside_pts ] - z_displacement[0:num_of_way_inside_pts] * sigma_multiplier * (4.0  + np.random.uniform(low=0.0, high=2.0, size=None) ) # 깊이 안쪽으로 점들을 이동
            proximity = trimesh.proximity.longest_ray(mesh, way_inside_pts, -z_displacement[0:num_of_way_inside_pts]) # 메쉬와의 근접성 계산 [num_of_sample_pts]
            way_inside_pts[ proximity< (sigma_multiplier* 4.0 ) ] = 0 # 너무 가까운 점들을 제거
            proximity = trimesh.proximity.signed_distance(mesh, way_inside_pts) # 메쉬와의 거리 계산 [num_of_sample_pts]
            way_inside_pts[proximity<0, :] = 0 # 메쉬 바깥에 있는 점들은 제거

            inside_points_low_res_pifu = np.concatenate([surface_points_with_normal_sigma, way_inside_pts], 0) # 표면에 있는 점들과 메쉬 안쪽 점들을 결합

            # 메쉬 바깥에 있는 점들을 생성
            num_of_outside_pts = round(self.num_sample_inout * self.opt.ratio_of_outside_points) # 바깥 점들의 수를 계산
            outside_surface_points = surface_points[0: num_of_outside_pts ] + z_displacement[0:num_of_outside_pts] * sigma_multiplier * (5.0 + np.random.uniform(low=0.0, high=50.0, size=None) ) # 바깥으로 이동시킴
            proximity = trimesh.proximity.longest_ray(mesh, outside_surface_points, z_displacement[0:num_of_outside_pts]) # 메쉬와의 근접성 계산 [num_of_sample_pts]
            outside_surface_points[ proximity< (sigma_multiplier* 5.0 ) ] = 0 # 너무 가까운 점들을 제거

            all_points_low_res_pifu = np.concatenate([inside_points_low_res_pifu, outside_surface_points], 0) # 모든 점들을 결합
            
        else:
            outside_points_low_res_pifu = sample_points_low_res_pifu[np.logical_not(inside_low_res_pifu)]

        # 내부 점이 너무 많으면 내부와 외부 점의 수를 줄임. ("nin > self.num_sample_inout // 2"가 참일 가능성이 높음)
        nin = inside_points_low_res_pifu.shape[0]  # 내부 점의 수
        if not self.opt.useDOS:  # Depth-Oriented Sampling(DOS)을 사용하지 않는 경우
            inside_points_low_res_pifu = inside_points_low_res_pifu[
                            :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points_low_res_pifu  # 내부 점이 너무 많으면 절반으로 줄임. 최종 크기는 [2500, 3]이어야 함
            outside_points_low_res_pifu = outside_points_low_res_pifu[
                             :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points_low_res_pifu[
                                                                                                   :(self.num_sample_inout - nin)]  # 외부 점의 수도 절반으로 줄임. 최종 크기는 [2500, 3]이어야 함

            samples_low_res_pifu = np.concatenate([inside_points_low_res_pifu, outside_points_low_res_pifu], 0).T  # 내부 점과 외부 점을 결합하고 전치시켜 최종 크기를 [3, 5000]으로 설정

        if self.opt.useDOS:  # Depth-Oriented Sampling(DOS)을 사용하는 경우
            samples_low_res_pifu = all_points_low_res_pifu.T  # 모든 점을 전치시켜 최종 크기를 [3, 5000]으로 설정
            
            labels_low_res_pifu = np.concatenate([labels_with_normal_sigma, np.ones((1, way_inside_pts.shape[0])) * 1.0, np.ones((1, outside_surface_points.shape[0])) * 0.0], 1)  # 레이블을 결합. 내부 점은 1.0, 외부 점은 0.0으로 설정. 최종 크기는 [1, 5000]이어야 함
            
        else:  # DOS를 사용하지 않는 경우
            labels_low_res_pifu = np.concatenate([np.ones((1, inside_points_low_res_pifu.shape[0])), np.zeros((1, outside_points_low_res_pifu.shape[0]))], 1)  # 내부 점은 1, 외부 점은 0으로 설정. 최종 크기는 [1, 5000]이어야 함

        # 샘플 점과 레이블을 PyTorch 텐서로 변환하고, float 타입으로 설정
        samples_low_res_pifu = torch.Tensor(samples_low_res_pifu).float()
        labels_low_res_pifu = torch.Tensor(labels_low_res_pifu).float()

        # 메쉬 데이터를 삭제하여 메모리에서 해제
        del mesh

        return {
            'samples_low_res_pifu': samples_low_res_pifu,
            'labels_low_res_pifu': labels_low_res_pifu
            }

    def get_item(self, index):

            # 이미지 파일 경로 가져오기
            img_path = self.img_files[index]
            img_name = os.path.splitext(os.path.basename(img_path))[0]  # 파일명에서 확장자를 제거한 부분을 가져옴

            # yaw(회전 각도) 정보 가져오기
            yaw = img_name.split("_")[-1]  # 파일명에서 마지막 부분이 yaw 각도임
            yaw = int(yaw)  # 문자열을 정수로 변환

            # subject(대상) 정보 가져오기
            subject = img_path.split('/')[-2]  # 경로에서 상위 폴더 이름이 subject임 (예: "0507")

            # 경로 설정 (렌더링된 파라미터, 이미지, 마스크 파일 경로)
            param_path = os.path.join(self.root, subject, "rendered_params_" + "{0:03d}".format(yaw) + ".npy")  # 파라미터 파일 경로
            render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")  # 렌더링 이미지 경로
            mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")  # 마스크 파일 경로

            # Ground Truth Normal Maps 경로 설정
            if self.opt.use_groundtruth_normal_maps:
                nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".exr")
                nmlB_high_res_path = os.path.join(self.normal_directory_high_res, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".exr")
            else:
                nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy")
                nmlB_high_res_path = os.path.join(self.normal_directory_high_res, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".npy")

            # Depth Map 경로 설정
            if self.opt.useGTdepthmap:
                depth_map_path = os.path.join(self.depth_map_directory, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".exr")
            else:
                depth_map_path = os.path.join(self.depth_map_directory, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".npy")

            # Human Parse Map 경로 설정
            human_parse_map_path = os.path.join(self.human_parse_map_directory, subject, "rendered_parse_" + "{0:03d}".format(yaw) + ".npy")

            load_size_associated_with_scale_factor = 1024  # 이미지 스케일 크기 설정

            # 파라미터 로드 (npy 파일에서 파라미터 읽기)
            param = np.load(param_path, allow_pickle=True)  # np.array 형식의 파라미터 로드
            center = param.item().get('center')  # 카메라 3D 중심 위치
            R = param.item().get('R')  # CAD 모델을 회전시키기 위한 행렬
            scale_factor = param.item().get('scale_factor')  # 카메라 스케일링 정보
            print('center: ', center)
            print('scale_factor: ', scale_factor)

            # b_range, b_center, b_min, b_max 계산 (바운딩 박스 설정)
            b_range = load_size_associated_with_scale_factor / scale_factor  # 예: 512 / scale_factor
            b_center = center
            b_min = b_center - b_range / 2
            b_max = b_center + b_range / 2

            # Extrinsic 행렬 생성 (3D 좌표 회전을 위한 행렬)
            translate = -center.reshape(3, 1)  # 카메라 중심 위치를 이동
            extrinsic = np.concatenate([R, translate], axis=1)  # 회전 행렬 R과 중심 이동 행렬을 결합
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)  # 마지막 행 추가

            # Intrinsic 행렬 생성 (이미지 UV 좌표를 월드 좌표에 맞춤)
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = 1.0 * scale_factor
            scale_intrinsic[1, 1] = -1.0 * scale_factor
            scale_intrinsic[2, 2] = 1.0 * scale_factor

            # 단위 행렬 생성
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2)
            uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2)
            uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2)

            # 마스크와 렌더링 이미지 불러오기
            mask = Image.open(mask_path).convert('L')  # 마스크는 그레이스케일로 변환
            render = Image.open(render_path).convert('RGB')  # 렌더링 이미지는 RGB로 변환

            # Intrinsic과 Extrinsic 행렬 결합하여 최종 calibration 행렬 생성
            intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            # 마스크와 렌더링 이미지 텐서로 변환
            mask = transforms.ToTensor()(mask).float()
            render = self.to_tensor(render)  # 렌더링 이미지를 정규화하여 텐서로 변환
            render = mask.expand_as(render) * render  # 마스크 영역만 남기고 나머지는 제거

            # PIFu용 저해상도 렌더링 이미지와 마스크 생성
            render_low_pifu = F.interpolate(torch.unsqueeze(render, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            render_low_pifu = render_low_pifu[0]
            mask_low_pifu = mask_low_pifu[0]

            # Normal Map 처리
            if self.opt.use_groundtruth_normal_maps:
                # high-res normal map 불러오기 (EXR 형식)
                nmlF_high_res = cv2.imread(nmlF_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # 앞쪽 normal map
                nmlB_high_res = cv2.imread(nmlB_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # 뒤쪽 normal map
                nmlB_high_res = nmlB_high_res[:, ::-1, :].copy()  # 좌우 대칭 변환
                nmlF_high_res = np.transpose(nmlF_high_res, [2, 0, 1])  # 채널 순서 변경
                nmlB_high_res = np.transpose(nmlB_high_res, [2, 0, 1])
            else:
                # npy 형식으로 저장된 normal map 로드
                nmlF_high_res = np.load(nmlF_high_res_path)
                nmlB_high_res = np.load(nmlB_high_res_path)

            # 텐서로 변환
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlB_high_res = torch.Tensor(nmlB_high_res)

            # 마스크 적용
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
            nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

            # 저해상도 Normal Map 생성
            nmlF = F.interpolate(torch.unsqueeze(nmlF_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            nmlF = nmlF[0]
            nmlB = F.interpolate(torch.unsqueeze(nmlB_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            nmlB = nmlB[0]

            # Depth Map 처리
            if self.opt.use_depth_map:
                if self.opt.useGTdepthmap:
                    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # EXR 형식 depth map 로드
                    depth_map = depth_map[:, :, 0]  # 첫 번째 채널 선택
                    mask_depth = depth_map > 100  # 특정 값을 초과하는 영역에 대해 마스크 생성
                    camera_position = 10.0  # 카메라 중심 위치 설정
                    depth_map = depth_map - camera_position  # 카메라 중심으로부터의 거리를 0으로 설정
                    depth_map = depth_map / (b_range / self.opt.resolution)  # 단위를 바운딩 큐브로 변환
                    depth_map = depth_map / (self.opt.resolution / 2)  # [-1, 1] 범위로 정규화
                    depth_map = depth_map + 1.0  # [0, 2] 범위로 변환
                    depth_map[mask_depth] = 0  # 유효하지 않은 값들은 0으로 설정
                    depth_map = np.expand_dims(depth_map, 0)  # [1,1024,1024] 크기로 변경
                    depth_map = torch.Tensor(depth_map)
                    depth_map = mask.expand_as(depth_map) * depth_map  # 마스크 적용
                    if self.opt.depth_in_front:
                        depth_map_low_res = F.interpolate(torch.unsqueeze(depth_map, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                        depth_map_low_res = depth_map_low_res[0]
                    else:
                        depth_map_low_res = 0
                else:
                    # npy 형식 depth map 로드
                    depth_map = np.load(depth_map_path)
                    depth_map = torch.Tensor(depth_map)
                    depth_map = mask.expand_as(depth_map) * depth_map  # 마스크 적용
                    if self.opt.depth_in_front:
                        depth_map_low_res = F.interpolate(torch.unsqueeze(depth_map, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                        depth_map_low_res = depth_map_low_res[0]
                    else:
                        depth_map_low_res = 0
            else:
                depth_map = None
                depth_map_low_res = 0

            # Human Parse Map 처리
            if self.opt.use_human_parse_maps:
                human_parse_map = np.load(human_parse_map_path)  # (1024,1024) 크기의 human parse map 로드
                human_parse_map = torch.Tensor(human_parse_map)
                human_parse_map = torch.unsqueeze(human_parse_map, 0)  # (1,1024,1024) 크기로 변경
                human_parse_map = mask.expand_as(human_parse_map) * human_parse_map  # 마스크 적용
                if self.opt.use_groundtruth_human_parse_maps:
                    human_parse_map_1 = (human_parse_map == 0.5).float()
                    human_parse_map_2 = (human_parse_map == 0.6).float()
                    human_parse_map_3 = (human_parse_map == 0.7).float()
                    human_parse_map_4 = (human_parse_map == 0.8).float()
                    human_parse_map_5 = (human_parse_map == 0.9).float()
                    human_parse_map_6 = (human_parse_map == 1.0).float()
                    human_parse_map_list = [human_parse_map_1, human_parse_map_2, human_parse_map_3, human_parse_map_4, human_parse_map_5, human_parse_map_6]
                else:
                    human_parse_map_0 = (human_parse_map == 0).float()
                    human_parse_map_1 = (human_parse_map == 1).float()
                    human_parse_map_2 = (human_parse_map == 2).float()
                    human_parse_map_3 = (human_parse_map == 3).float()
                    human_parse_map_4 = (human_parse_map == 4).float()
                    human_parse_map_5 = (human_parse_map == 5).float()
                    human_parse_map_6 = (human_parse_map == 6).float()
                    human_parse_map_list = [human_parse_map_0, human_parse_map_1, human_parse_map_2, human_parse_map_3, human_parse_map_4, human_parse_map_5, human_parse_map_6]
                human_parse_map = torch.cat(human_parse_map_list, dim=0)
                human_parse_map = F.interpolate(torch.unsqueeze(human_parse_map, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
                human_parse_map = human_parse_map[0]
            else:
                human_parse_map = 0

            # Depth map이 사용되지 않을 경우 0으로 설정
            if not self.opt.use_depth_map:
                depth_map = 0

            # evaluation_mode인 경우 샘플 데이터를 0으로 설정
            if self.evaluation_mode:
                sample_data = {'samples_low_res_pifu': 0, 'labels_low_res_pifu': 0}
            else:
                # 샘플 데이터 선택 (num_sample_inout 값이 있을 때만 샘플링 수행)
                if self.opt.num_sample_inout:
                    sample_data = self.select_sampling_method(subject, calib, b_min=b_min, b_max=b_max, R=R)

            # 최종 반환되는 데이터 구조
            data = {
                'name': subject,  # 대상 이름
                'render_path': render_path,  # 렌더링 이미지 경로
                'render_low_pifu': render_low_pifu,  # 저해상도 렌더링 이미지
                'mask_low_pifu': mask_low_pifu,  # 저해상도 마스크
                'original_high_res_render': render,  # 고해상도 렌더링 이미지
                'mask': mask,  # 고해상도 마스크
                'calib': calib,  # 캘리브레이션 행렬
                'extrinsic': extrinsic,  # 외부 행렬
                'samples_low_res_pifu': sample_data['samples_low_res_pifu'],  # 저해상도 샘플
                'labels_low_res_pifu': sample_data['labels_low_res_pifu'],  # 샘플 레이블
                'b_min': b_min,  # 바운딩 박스 최소값
                'b_max': b_max,  # 바운딩 박스 최대값
                'nmlF': nmlF,  # 저해상도 앞쪽 Normal Map
                'nmlB': nmlB,  # 저해상도 뒤쪽 Normal Map
                'nmlF_high_res': nmlF_high_res,  # 고해상도 앞쪽 Normal Map
                'nmlB_high_res': nmlB_high_res,  # 고해상도 뒤쪽 Normal Map
                'depth_map': depth_map,  # 고해상도 Depth Map
                'depth_map_low_res': depth_map_low_res,  # 저해상도 Depth Map
                'human_parse_map': human_parse_map  # Human Parse Map
            }

            return data

    # __getitem__ 함수: 주어진 인덱스의 데이터를 가져옴
    def __getitem__(self, index):
        return self.get_item(index)  # 위에서 정의한 get_item 함수 호출