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

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

# HumanParseDataset 클래스 정의 - PyTorch Dataset 클래스를 상속받아 구현
class HumanParseDataset(Dataset):

    def __init__(self, opt, evaluation_mode=False):
        """
        데이터셋 초기화 함수. 학습 및 평가 데이터를 준비.
        Args:
            opt: 옵션 객체, 데이터셋 설정 및 전처리에 필요한 인자 포함.
            evaluation_mode: 평가 모드 여부. True일 경우 테스트 데이터 로드.
        """
        self.opt = opt
        # 학습 데이터 리스트 로드
        self.training_subject_list = np.loadtxt("train_set_list.txt", dtype=str).tolist()

        # 평가 모드일 경우, 테스트 데이터 리스트로 대체
        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("test_set_list.txt", dtype=str).tolist()
        
        # 데이터 디렉토리 경로 설정
        self.human_parse_map_directory = "rendering_script/render_human_parse_results"
        self.root = "rendering_script/buffer_fixed_full_mesh"
        
        # 법선 맵 사용 옵션이 활성화된 경우, 법선 맵 디렉토리 설정
        if self.opt.use_normal_map_for_parse_training:
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
        self.img_files = sorted(self.img_files)

        # PIL 이미지를 PyTorch 텐서로 변환하기 위한 변환 정의
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # PIL 이미지를 (C x H x W) 형태의 텐서로 변환. 각 채널 범위 [0.0, 1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 각 채널을 평균 0.5, 표준 편차 0.5로 정규화. 최종적으로 [-1,1] 범위
        ])

    def __len__(self):
        """
        데이터셋의 전체 길이 반환. (총 이미지 파일 수)
        """
        return len(self.img_files)

    def get_item(self, index):
        """
        주어진 인덱스에 해당하는 데이터를 가져와서 전처리.
        Args:
            index: 데이터 인덱스
        Returns:
            데이터 딕셔너리
        """

        # 인덱스를 사용해 이미지 경로 가져오기
        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 이미지 파일명에서 yaw (회전 각도) 값 추출
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # 주제 추출 (예: "0507")
        subject = img_path.split('/')[-2]

        # 렌더링 이미지 및 마스크 파일 경로 구성
        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")
        
        # 사람 파싱 맵 파일 경로 구성
        human_parse_map_path = os.path.join(self.human_parse_map_directory, subject, "rendered_parse_" + "{0:03d}".format(yaw) + ".npy")

        # 마스크 이미지 로드 및 그레이스케일로 변환
        mask = Image.open(mask_path).convert('L')
        # 렌더링 이미지 로드 및 RGB로 변환
        render = Image.open(render_path).convert('RGB')

        # 마스크를 텐서로 변환
        mask = transforms.ToTensor()(mask).float()

        # 렌더링 이미지를 텐서로 변환하고, 마스크를 곱하여 배경 제거
        render = self.to_tensor(render)
        render = mask.expand_as(render) * render

        # 1024x1024 이미지를 512x512로 리사이즈하여 PIFu에 사용
        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        # 법선 맵 사용 옵션이 활성화된 경우, 법선 맵 로드
        if self.opt.use_normal_map_for_parse_training:
            nmlF_high_res_path = os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npy")
            nmlF_high_res = np.load(nmlF_high_res_path)  # [3, 1024, 1024] 형태
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
        else:
            nmlF_high_res = 0

        # 사람 파싱 맵 로드 및 전처리
        human_parse_map_high_res = np.load(human_parse_map_path)  # (1024, 1024) 형태
        human_parse_map_high_res = torch.Tensor(human_parse_map_high_res)
        human_parse_map_high_res = torch.unsqueeze(human_parse_map_high_res, 0)  # (1, 1024, 1024) 형태로 변환
        human_parse_map_high_res = mask.expand_as(human_parse_map_high_res) * human_parse_map_high_res

        # 각 클래스(카테고리)별로 파싱 맵을 생성
        human_parse_map_0 = (human_parse_map_high_res == 0).float()
        human_parse_map_1 = (human_parse_map_high_res == 0.5).float()
        human_parse_map_2 = (human_parse_map_high_res == 0.6).float()
        human_parse_map_3 = (human_parse_map_high_res == 0.7).float()
        human_parse_map_4 = (human_parse_map_high_res == 0.8).float()
        human_parse_map_5 = (human_parse_map_high_res == 0.9).float()
        human_parse_map_6 = (human_parse_map_high_res == 1.0).float()
        human_parse_map_list = [human_parse_map_0, human_parse_map_1, human_parse_map_2, human_parse_map_3, human_parse_map_4, human_parse_map_5, human_parse_map_6]
        human_parse_map_high_res = torch.cat(human_parse_map_list, dim=0)

        # 최종 데이터 딕셔너리 반환
        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'nmlF_high_res': nmlF_high_res,
            'human_parse_map_high_res': human_parse_map_high_res,
            'mask': mask
        }

    def __getitem__(self, index):
        """
        DataLoader가 호출할 메서드로, 데이터셋의 요소를 가져옴.
        """
        return self.get_item(index)