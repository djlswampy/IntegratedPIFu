import os
import random

import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import cv2
import torch
import json
import trimesh
import logging
import OpenEXR
import Imath

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

# OpenEXR 이미지를 읽기 위해 OpenCV에서 EXR 지원 활성화
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# 카메라와 메쉬 사이의 거리 (렌더링 스크립트에서 설정된 값, 필요 시 변경 가능)
CAMERA_TO_MESH_DISTANCE = 10.0

# PyTorch Dataset 클래스를 상속받아 깊이 데이터셋 정의
class DepthDataset(Dataset):

    def __init__(self, opt, evaluation_mode=False):
        """
        DepthDataset 초기화 메서드.
        
        opt: 옵션 객체
        evaluation_mode: 평가 모드로 설정 시 True
        """
        self.opt = opt
        
        # 학습에 사용할 주제 목록을 텍스트 파일에서 로드
        self.training_subject_list = np.loadtxt("train_set_list.txt", dtype=str).tolist()

        # 평가 모드인 경우, 학습 데이터 대신 테스트 데이터 목록을 로드
        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("test_set_list.txt", dtype=str).tolist()
            self.is_train = False

        # 법선 맵, 깊이 맵, 렌더링 이미지를 저장한 디렉토리 경로 설정
        self.depth_map_directory = "rendering_script/buffer_depth_maps_of_full_mesh"
        self.root = "rendering_script/buffer_fixed_full_mesh"

        # 두 번째 스테이지 깊이 데이터 학습 사용 시, 코스 깊이 맵 경로 설정
        if self.opt.second_stage_depth:
            self.coarse_depth_map_directory = "trained_coarse_depth_maps"
            
        # 깊이 학습에 법선 맵 사용 시, 법선 맵 디렉토리 설정
        if self.opt.use_normal_map_for_depth_training:
            self.normal_directory_high_res = "trained_normal_maps"

        # 주제 리스트 저장
        self.subjects = self.training_subject_list  

        # 이미지 파일 리스트 생성
        self.img_files = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.root, training_subject)
            # 해당 주제의 모든 이미지 파일 경로를 리스트에 추가
            subject_render_paths_list = [os.path.join(subject_render_folder, f) for f in os.listdir(subject_render_folder) if "image" in f]
            self.img_files = self.img_files + subject_render_paths_list
        
        # 파일 경로 정렬
        self.img_files = sorted(self.img_files)

        # 이미지를 PyTorch 텐서로 변환하는 변환 정의
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # PIL 이미지를 (C x H x W) 형식으로 변환, 값 범위는 [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 각 채널을 평균 0.5, 표준 편차 0.5로 정규화하여 범위를 [-1, 1]로 변환
        ])


    def load_exr_image(self, file_path):
        exr_file = OpenEXR.InputFile(file_path)
        
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ["V"]  # EXR 파일에 있는 채널명을 정확히 기입해야 함
        channel_data = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in channels]
        channel_data = [c.reshape(size[1], size[0]) for c in channel_data]

        return np.stack(channel_data, axis=-1)

    def __len__(self):
        """
        데이터셋의 전체 길이를 반환하는 메서드 (필수 구현)
        """
        return len(self.img_files)

    def get_item(self, index):
        """
        주어진 인덱스의 데이터를 가져오는 함수. 데이터 로드 로직 구현.
        
        index: 데이터셋에서 가져올 항목의 인덱스
        """
        # 이미지 경로와 이름 가져오기
        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 이미지 파일명에서 yaw(회전 각도) 값 추출
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # 이미지 경로에서 주제 추출 (예: "0507")
        subject = img_path.split('/')[-2]

        # 관련된 파일 경로 설정 (파라미터, 렌더링, 마스크 등)
        param_path = os.path.join(self.root, subject, "rendered_params_" + "{0:03d}".format(yaw) + ".npy")
        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")
        depth_map_path = os.path.join(self.depth_map_directory, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".exr")
        
        # 두 번째 스테이지 깊이 데이터 사용 시 코스 깊이 맵 경로 설정
        if self.opt.second_stage_depth:
            coarse_depth_map_path = os.path.join(self.coarse_depth_map_directory, subject, "rendered_depthmap_" + "{0:03d}".format(yaw) + ".npy")

        # 마스크 이미지 로드 및 그레이스케일로 변환
        mask = Image.open(mask_path).convert('L')
        render = Image.open(render_path).convert('RGB')

        # 마스크를 텐서로 변환
        mask = transforms.ToTensor()(mask).float()

        # 렌더링 이미지를 텐서로 변환하고, 마스크를 곱하여 배경 제거
        render = self.to_tensor(render)
        render = mask.expand_as(render) * render

        # 파라미터 파일에서 scale factor 값 로드
        param = np.load(param_path, allow_pickle=True)
        scale_factor = param.item().get('scale_factor')

        # scale factor에 따른 이미지 크기 계산
        load_size_associated_with_scale_factor = 1024
        b_range = load_size_associated_with_scale_factor / scale_factor

        # 중심 표시를 위한 인디케이터 생성
        center_indicator = np.zeros([1, 1024, 1024])
        center_indicator[:, 511:513, 511:513] = 1.0
        center_indicator = torch.Tensor(center_indicator).float()

        # 고해상도 이미지를 저해상도로 리사이즈하여 PIFu에 사용
        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        # 법선 맵 사용 시 해당 파일 로드
        if self.opt.use_normal_map_for_depth_training:
            nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy")
            nmlF_high_res = np.load(nmlF_high_res_path)  # [3, 1024, 1024] 형태로 로드
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
        else:
            nmlF_high_res = 0

        # EXR 형식의 깊이 맵 로드
        # depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_map = self.load_exr_image(depth_map_path)
        depth_map = depth_map[:, :, 0]
        mask_depth = depth_map > 100  # 유효하지 않은 깊이 값에 대한 마스크 생성

        # 깊이 값 정규화 및 보정
        depth_map = depth_map - CAMERA_TO_MESH_DISTANCE
        depth_map = depth_map / (b_range / self.opt.resolution)
        depth_map = depth_map / (self.opt.resolution / 2)
        depth_map = depth_map + 1.0
        depth_map[mask_depth] = 0
        depth_map = np.expand_dims(depth_map, 0)
        depth_map = torch.Tensor(depth_map)
        depth_map = mask.expand_as(depth_map) * depth_map

        # 두 번째 스테이지 깊이 데이터 사용 시, 코스 깊이 맵 로드
        if self.opt.second_stage_depth:
            coarse_depth_map = np.load(coarse_depth_map_path)
            coarse_depth_map = torch.Tensor(coarse_depth_map)
            coarse_depth_map = mask.expand_as(coarse_depth_map) * coarse_depth_map
        else:
            coarse_depth_map = 0

        # 데이터 딕셔너리로 반환
        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'center_indicator': center_indicator,
            'depth_map': depth_map,
            'coarse_depth_map': coarse_depth_map,
            'nmlF_high_res': nmlF_high_res,
            'mask': mask
        }

    def __getitem__(self, index):
        """
        PyTorch의 DataLoader가 호출하는 메서드로, 데이터셋의 요소를 가져옴.
        """
        return self.get_item(index)