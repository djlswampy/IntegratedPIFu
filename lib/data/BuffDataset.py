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

# 3D 모델 생성용 추가 지도 맵의 생성 여부를 설정합니다.
produce_normal_maps = True 
produce_coarse_depth_maps = True
produce_fine_depth_maps = True
produce_parse_maps = True


# 데이터셋 클래스를 정의합니다.
class BuffDataset(Dataset):

    # 초기화 함수. 데이터셋 초기 설정과 파일 경로 등을 설정합니다.
    def __init__(self, opt):
        self.opt = opt  # 옵션 설정을 가져옵니다.
        self.projection_mode = 'orthogonal'  # 투영 모드를 '정사영'으로 설정합니다.
        
        # 테스트할 데이터셋의 subject(개체) 목록을 읽어옵니다.
        self.subjects = np.loadtxt("buff_subject_testing.txt", dtype=str).tolist()

        # 데이터셋의 루트 디렉토리를 설정합니다.
        self.root = "buff_dataset/buff_rgb_images"

        # 이미지를 텐서로 변환하고 정규화하기 위한 전처리 파이프라인 설정
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 (C x H x W) 형태의 텐서로 변환합니다.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화하여 [-1, 1] 범위로 변환합니다.
        ])

    # 데이터셋의 길이를 반환합니다.
    def __len__(self):
        return len(self.subjects)  # subjects 리스트의 길이 반환



    # 특정 인덱스의 데이터를 반환하는 함수입니다.
    def get_item(self, index):
        # 현재 인덱스의 subject 이름을 가져옵니다.
        subject = self.subjects[index]

        # subject에 대한 각 파일 경로를 설정합니다.
        param_path = os.path.join(self.root, "rendered_params_" + subject + ".npy") 
        render_path = os.path.join(self.root, "rendered_image_" + subject + ".png")
        mask_path = os.path.join(self.root, "rendered_mask_" + subject + ".png")

        # 선택한 경우, 각종 지도(노멀 맵, 깊이 맵 등) 파일 경로 설정
        if produce_normal_maps:
            nmlF_high_res_path = os.path.join("buff_dataset/buff_normal_maps", "rendered_nmlF_" + subject + ".npy")
            nmlB_high_res_path = os.path.join("buff_dataset/buff_normal_maps", "rendered_nmlB_" + subject + ".npy")

        if produce_coarse_depth_maps:
            coarse_depth_map_path = os.path.join("buff_dataset/buff_depth_maps", "rendered_coarse_depthmap_" + subject + ".npy")
        
        if produce_fine_depth_maps:
            fine_depth_map_path = os.path.join("buff_dataset/buff_depth_maps", "rendered_depthmap_" + subject + ".npy")

        if produce_parse_maps:
            parse_map_path = os.path.join("buff_dataset/buff_parse_maps", "rendered_parse_" + subject + ".npy")

        # 스케일 팩터에 따른 이미지 크기 설정
        load_size_associated_with_scale_factor = 1024

        # 카메라 매개변수를 포함하는 파일 로드
        param = np.load(param_path, allow_pickle=True)
        center = param.item().get('center')  # 카메라의 중심 위치
        R = param.item().get('R')  # 회전 행렬
        scale_factor = param.item().get('scale_factor')  # 스케일 팩터

        # 카메라 매개변수를 바탕으로 3D 공간의 범위 정의
        b_range = load_size_associated_with_scale_factor / scale_factor
        b_center = center
        b_min = b_center - b_range / 2
        b_max = b_center + b_range / 2

        # 3D 포인트의 회전 및 이동을 위한 외적 행렬 설정
        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)  # 회전 후 이동
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

        # 스케일링 행렬 설정
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor
        scale_intrinsic[1, 1] = -1.0 * scale_factor
        scale_intrinsic[2, 2] = 1.0 * scale_factor

        # UV 좌표계와 픽셀 공간 매칭을 위한 행렬 설정
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2)
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2)
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2)

        # 마스크와 렌더 이미지를 불러옵니다.
        mask = Image.open(mask_path).convert('L')
        render = Image.open(render_path).convert('RGB')

        # 총 내적 행렬을 만들어서 캘리브레이션 매트릭스로 설정
        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()

        # 마스크를 텐서로 변환
        mask = transforms.ToTensor()(mask).float()

        # 렌더 이미지를 정규화하여 텐서로 변환하고, 마스크를 적용합니다.
        render = self.to_tensor(render)  # [0,255] 범위를 [-1,1] 범위로 정규화
        render = mask.expand_as(render) * render

        # 로우 레졸루션 PIFu용으로 이미지를 512x512 크기로 축소
        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        # 각종 지도(노멀 맵, 깊이 맵 등)를 불러오고 마스크를 적용한 뒤 로우 레졸루션으로 변환
        if produce_normal_maps:
            nmlF_high_res = np.load(nmlF_high_res_path)
            nmlB_high_res = np.load(nmlB_high_res_path)
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlB_high_res = torch.Tensor(nmlB_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
            nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

            nmlF = F.interpolate(torch.unsqueeze(nmlF_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            nmlF = nmlF[0]
            nmlB = F.interpolate(torch.unsqueeze(nmlB_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            nmlB = nmlB[0]
        else:
            nmlF_high_res = nmlB_high_res = 0
            nmlF = nmlB = 0 

        # 깊이 맵 데이터 불러오기 및 전처리
        if produce_coarse_depth_maps:
            coarse_depth_map = np.load(coarse_depth_map_path)
            coarse_depth_map = torch.Tensor(coarse_depth_map)
            coarse_depth_map = mask.expand_as(coarse_depth_map) * coarse_depth_map

        else:
            coarse_depth_map = 0

        if produce_fine_depth_maps:
            fine_depth_map = np.load(fine_depth_map_path)
            fine_depth_map = torch.Tensor(fine_depth_map)
            fine_depth_map = mask.expand_as(fine_depth_map) * fine_depth_map
            depth_map_low_res = F.interpolate(torch.unsqueeze(fine_depth_map, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
            depth_map_low_res = depth_map_low_res[0]
        else: 
            fine_depth_map = 0
            depth_map_low_res = 0 

        # 구문 분석 맵(파싱 맵) 데이터 불러오기 및 전처리
        if produce_parse_maps:
            human_parse_map = np.load(parse_map_path)
            human_parse_map = torch.Tensor(human_parse_map)
            human_parse_map = torch.unsqueeze(human_parse_map, 0)
            human_parse_map = mask.expand_as(human_parse_map) * human_parse_map

            # 파싱 맵을 개별 채널로 나누고 병합하여 로우 레졸루션으로 변환
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

        # 중심 표시용 인디케이터 설정
        center_indicator = np.zeros([1, 1024, 1024])
        center_indicator[:, 511:513, 511:513] = 1.0
        center_indicator = torch.Tensor(center_indicator).float()

        # 데이터셋의 현재 항목을 딕셔너리 형태로 반환
        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'mask': mask,
            'calib': calib,
            'b_min': b_min,
            'b_max': b_max,
            'nmlF': nmlF,
            'nmlB': nmlB,
            'nmlF_high_res': nmlF_high_res,
            'nmlB_high_res': nmlB_high_res,
            'depth_map': fine_depth_map,
            'depth_map_low_res': depth_map_low_res,
            'human_parse_map': human_parse_map,
            'center_indicator': center_indicator,
            'coarse_depth_map': coarse_depth_map
        }

    # PyTorch에서 데이터를 인덱스로 접근할 수 있게 하는 함수
    def __getitem__(self, index):
        return self.get_item(index)
