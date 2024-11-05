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

# 데이터셋 로드 시 특정 맵을 로드할지 여부를 결정하는 플래그
produce_normal_maps = True
produce_coarse_depth_maps = False
produce_fine_depth_maps = False 
produce_parse_maps = False 

class BuffDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt  # 옵션 객체 저장
        self.projection_mode = 'orthogonal'  # 투영 모드 설정
        # 테스트할 주제 목록을 텍스트 파일에서 로드
        self.subjects = np.loadtxt("/home/jo/IntegratedPIFu/buff_subject_testing.txt", dtype=str).tolist()
        self.root = "/home/jo/IntegratedPIFu/dataset"  # 이미지 파일의 루트 디렉토리 설정

        # PIL 이미지를 텐서로 변환하는 변환기 설정
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 텐서를 정규화
        ])

    def __len__(self):
        return len(self.subjects)  # 데이터셋의 길이 반환

    def get_item(self, index):
        subject = self.subjects[index]  # 주제 이름 가져오기

        # NOTE: 여기에 원래 카메라 파라미터 경로도 있었는데 구할 방법이 없어서 지우고 PIFu에서 기본 카메라 파라미터 설정하는 코드 가지고 와서 사용했음
        # 각 주제에 대한 파일 경로 설정
        render_path = os.path.join(self.root, "rendered_image_" + subject + ".png")
        mask_path = os.path.join(self.root, "rendered_mask_" + subject + ".png")

        # 노멀 맵 경로 설정
        if produce_normal_maps:
            nmlF_high_res_path = os.path.join("/home/jo/IntegratedPIFu/dataset/normal_maps", "rendered_nmlF_" + subject + ".npy")
            nmlB_high_res_path = os.path.join("/home/jo/IntegratedPIFu/dataset/normal_maps", "rendered_nmlB_" + subject + ".npy")

        # 깊이 맵 경로 설정
        if produce_coarse_depth_maps:
            coarse_depth_map_path = os.path.join("buff_dataset/buff_depth_maps", "rendered_coarse_depthmap_" + subject + ".npy")
        
        if produce_fine_depth_maps:
            fine_depth_map_path = os.path.join("buff_dataset/buff_depth_maps", "rendered_depthmap_" + subject + ".npy")

        # 파싱 맵 경로 설정
        if produce_parse_maps:
            parse_map_path = os.path.join("buff_dataset/buff_parse_maps", "rendered_parse_" + subject + ".npy")

        load_size_associated_with_scale_factor = 1024  # 스케일 팩터와 관련된 로드 크기 설정

        # NOTE: 여기에 원래 이미지 카메라 파라미터 데이터를 사용해야하는데 구할 방법이 없어서 지우고 PIFu에서 기본 카메라 파라미터 설정하는 코드 가지고 와서 사용했음
        # 카메라 파라미터를 기본값으로 설정
        center = np.array([0.0, 0.0, 0.0])  # 카메라 중심 위치를 원점으로 설정
        R = np.eye(3)  # 회전 행렬을 단위 행렬로 설정
        scale_factor = 900.0  # 스케일 팩터를 1로 설정

        # 바운딩 박스 계산
        b_range = load_size_associated_with_scale_factor / scale_factor 
        b_center = center
        b_min = b_center - b_range / 2
        b_max = b_center + b_range / 2

        # 외부 행렬 설정
        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        
        # 내부 행렬 설정
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor  
        scale_intrinsic[1, 1] = -1.0 * scale_factor  
        scale_intrinsic[2, 2] = 1.0 * scale_factor   

        # 이미지 픽셀 공간을 UV 공간에 맞추기 위한 행렬 설정
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2)  
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2)  
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

        # 최종 캘리브레이션 행렬 계산
        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float() 
        extrinsic = torch.Tensor(extrinsic).float()

        # 마스크 및 렌더 이미지 로드
        mask = Image.open(mask_path).convert('L')  
        render = Image.open(render_path).convert('RGB')

        # 마스크를 텐서로 변환
        mask = transforms.ToTensor()(mask).float()

        # 렌더 이미지를 텐서로 변환 및 정규화
        render = self.to_tensor(render)
        render = mask.expand_as(render) * render

        # 저해상도 PIFu를 위한 이미지 크기 조정
        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))[0]
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))[0]

        # 노멀 맵 로드 및 처리
        if produce_normal_maps:
            nmlF_high_res = np.load(nmlF_high_res_path)
            nmlB_high_res = np.load(nmlB_high_res_path)
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlB_high_res = torch.Tensor(nmlB_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
            nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

            nmlF = F.interpolate(torch.unsqueeze(nmlF_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))[0]
            nmlB = F.interpolate(torch.unsqueeze(nmlB_high_res, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))[0]
        else:
            nmlF_high_res = nmlB_high_res = None
            nmlF = nmlB = None 

        # 깊이 맵 로드 및 처리
        if produce_coarse_depth_maps:
            coarse_depth_map = np.load(coarse_depth_map_path)
            coarse_depth_map = torch.Tensor(coarse_depth_map)
            coarse_depth_map = mask.expand_as(coarse_depth_map) * coarse_depth_map
        else:
            coarse_depth_map = None

        if produce_fine_depth_maps:
            fine_depth_map = np.load(fine_depth_map_path)
            fine_depth_map = torch.Tensor(fine_depth_map)
            fine_depth_map = mask.expand_as(fine_depth_map) * fine_depth_map
            depth_map_low_res = F.interpolate(torch.unsqueeze(fine_depth_map, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))[0]
        else: 
            fine_depth_map = None
            depth_map_low_res = None 

        # 파싱 맵 로드 및 처리
        if produce_parse_maps:
            human_parse_map = np.load(parse_map_path)
            human_parse_map = torch.Tensor(human_parse_map)
            human_parse_map = torch.unsqueeze(human_parse_map, 0)
            human_parse_map = mask.expand_as(human_parse_map) * human_parse_map

            # 파싱 맵을 여러 채널로 분리
            human_parse_map_list = []
            for i in range(7):  # 클래스 수에 따라 조정
                human_parse_map_i = (human_parse_map == i).float()
                human_parse_map_list.append(human_parse_map_i)
            human_parse_map = torch.cat(human_parse_map_list, dim=0)
            human_parse_map = F.interpolate(torch.unsqueeze(human_parse_map, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))[0]
        else:
            human_parse_map = None

        # 중심 지시자 생성
        center_indicator = np.zeros([1, 1024, 1024])
        center_indicator[:, 511:513, 511:513] = 1.0
        center_indicator = torch.Tensor(center_indicator).float()
        
        # NOTE: 여기서 노멀맵 생성하는 경우랑 그냥 데이터 로딩하는 경우로 나뉨. 노멀맵 생성하는 경우는 위쪽에 처리할 맵에 대한 변수 전부 False임
        # NOTE: 그런데 처리할 맵에 대한 변수를 False로 설정하면 그와 관련된 속성들이 None으로 찍힘. 그로 인해 데이터 로더에서 None과 관련된 오류가 출력됨
        # NOTE: 그래서 처리할 맵에 대한 변수를 False로 설정한 값들과 관련된 속성들은 전부 지우고 실행해야함
        
        """
        이 코드를 이용해서 None으로 출력되는 속성은 제거하고 실행해야함
        print(f"Subject: {subject}")
        print(f"Render path: {render_path}")
        print(f"Mask path: {mask_path}")
        print(f"Render low PIFu: {render_low_pifu}")
        print(f"Mask low PIFu: {mask_low_pifu}")
        print(f"Original high res render: {render}")
        print(f"Mask: {mask}")
        print(f"Calib: {calib}")
        print(f"Normal Map Front (high res): {nmlF_high_res}")
        print(f"Normal Map Back (high res): {nmlB_high_res}")
        print(f"Center Indicator: {center_indicator}")
        print(f"Coarse Depth Map: {coarse_depth_map}")
        print(f"Fine Depth Map: {fine_depth_map}")
        print(f"Depth Map Low Res: {depth_map_low_res}")
        print(f"Human Parse Map: {human_parse_map}")
        print(f"Normal Map Front: {nmlF}")
        print(f"Normal Map Back: {nmlB}")
        print(f"B Min: {b_min}")
        print(f"B Max: {b_max}")
        """

        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'mask': mask,
            'calib': calib,
            'nmlF_high_res': nmlF_high_res,
            'nmlB_high_res': nmlB_high_res,
            'center_indicator': center_indicator,
            'nmlF': nmlF,
            'nmlB': nmlB,
            'b_min': b_min,
            'b_max': b_max
        }
        
        """
        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'mask': mask,
            'calib': calib,
            'nmlF_high_res': nmlF_high_res,
            'nmlB_high_res': nmlB_high_res,
            'center_indicator': center_indicator,
            'coarse_depth_map': coarse_depth_map,
            'depth_map': fine_depth_map,
            'depth_map_low_res': depth_map_low_res,
            'human_parse_map': human_parse_map,
            'nmlF': nmlF,
            'nmlB': nmlB,
            'b_min': b_min,
            'b_max': b_max
        }
        """

    def __getitem__(self, index):
        return self.get_item(index) 