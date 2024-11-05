import os
import random

import numpy as np
from PIL import Image, ImageOps
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

# NormalDataset 클래스 정의 - Dataset 클래스를 상속받아 PyTorch 데이터셋을 구현
class NormalDataset(Dataset):
    # 클래스 초기화 함수
    def __init__(self, opt, evaluation_mode=False):
        self.opt = opt  # 옵션 객체를 인스턴스 변수로 저장
        # 학습 데이터로 사용할 주제 목록을 텍스트 파일에서 로드
        self.training_subject_list = np.loadtxt("normal_train_set_list.txt", dtype=str).tolist()

        # 평가 모드일 경우, 학습 데이터 대신 테스트 데이터 목록을 로드
        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("test_set_list.txt", dtype=str).tolist()
            self.is_train = False

        # Ground Truth 법선 맵과 렌더링된 이미지를 저장한 디렉토리 경로 설정
        self.groundtruth_normal_map_directory = "rendering_script/buffer_normal_maps_of_full_mesh"
        self.root = "rendering_script/buffer_fixed_full_mesh"

        # 주제 리스트 저장
        self.subjects = self.training_subject_list   

        # 이미지 파일 리스트 생성
        self.img_files = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.root, training_subject)
            print("subject_render_folder: ", subject_render_folder)
            # 해당 주제의 모든 이미지 파일 경로를 리스트에 추가
            subject_render_paths_list = [os.path.join(subject_render_folder, f) for f in os.listdir(subject_render_folder) if "image" in f]
            self.img_files = self.img_files + subject_render_paths_list
        self.img_files = sorted(self.img_files)  # 파일 경로를 정렬하여 저장

        # 이미지 데이터를 PyTorch 텐서로 변환하기 위한 변환 정의
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # PIL 이미지를 텐서로 변환 (C x H x W) 형식으로 변환, 값 범위는 [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 각 채널을 평균 0.5, 표준 편차 0.5로 정규화하여 범위를 [-1, 1]로 변환
        ])

    def load_exr_image(self, file_path):
        """
        EXR 파일에서 이미지를 로드하여 (H, W, 3) 형태의 NumPy 배열로 반환하는 함수.
        """
        
        # EXR 파일 열기
        exr_file = OpenEXR.InputFile(file_path)
        
        # 데이터 윈도우 가져오기 (이미지의 크기 정보)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # (너비, 높이)
        
        # 픽셀 타입 설정 (FLOAT 타입으로 설정)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # EXR 파일의 채널 이름을 설정 (X, Y, Z 채널)
        channels = ["X", "Y", "Z"]
        
        # 각 채널 데이터를 읽고, numpy 배열로 변환
        channel_data = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in channels]
        
        # 읽은 채널 데이터를 이미지 형태로 재구성 (행렬 형태로 변환)
        channel_data = [c.reshape(size[1], size[0]) for c in channel_data]
        
        # 채널 데이터를 스택으로 결합하여 최종 이미지를 반환 (H, W, 3) 형태
        return np.stack(channel_data, axis=-1)

    # 데이터셋의 전체 길이를 반환하는 메서드 (필수 구현)
    def __len__(self):
        return len(self.img_files)

    # 주어진 인덱스의 데이터를 가져오는 함수 (데이터 로드 로직을 구현)
    def get_item(self, index):
        # 인덱스를 사용하여 이미지 경로 가져오기
        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 이미지 파일명에서 yaw(회전 각도) 값 추출
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # 이미지 경로에서 주제 추출 (예: "0507")
        subject = img_path.split('/')[-2]

        # 렌더링 이미지 및 마스크 파일 경로 구성
        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png")
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png")
        
        # 법선 맵 파일 경로 구성 (앞면과 뒷면)
        nmlF_high_res_path = os.path.join(self.groundtruth_normal_map_directory, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".exr")
        nmlB_high_res_path = os.path.join(self.groundtruth_normal_map_directory, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".exr")

        # 마스크 이미지 로드 및 그레이스케일로 변환
        mask = Image.open(mask_path).convert('L')
        # 렌더링 이미지 로드 및 RGB로 변환
        render = Image.open(render_path).convert('RGB')

        # 마스크를 텐서로 변환
        mask = transforms.ToTensor()(mask).float()

        # 렌더링 이미지를 텐서로 변환하고, 마스크를 곱하여 배경 제거
        render = self.to_tensor(render)  # [-1, 1]로 정규화된 텐서로 변환
        render = mask.expand_as(render) * render

        # 고해상도 이미지를 저해상도로 리사이즈하여 PIFu에 사용
        render_low_pifu = F.interpolate(torch.unsqueeze(render, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask, 0), size=(self.opt.loadSizeGlobal, self.opt.loadSizeGlobal))
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        """
        OpenCV를 사용해 고해상도 법선 맵 로드
        nmlF_high_res = cv2.imread(nmlF_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # [1024, 1024, 3]
        nmlB_high_res = cv2.imread(nmlB_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        이 코드 써서 불러오니까 채널 수가 줄어듬. opencv에서 exr 파일을 읽는 과정에서 마지막 채널이 소실된듯함
        그래서 load_exr_image 함수 추가
        """
        nmlF_high_res = self.load_exr_image(nmlF_high_res_path) # [1024, 1024, 3]
        nmlB_high_res = self.load_exr_image(nmlB_high_res_path)
        
        # 뒷면 법선 맵 좌우 반전
        nmlB_high_res = nmlB_high_res[:, ::-1, :].copy()  

        # OpenCV로 로드된 이미지를 텐서로 변환
        nmlF_high_res = np.transpose(nmlF_high_res, [2, 0, 1])  # [C, H, W] 형태로 변환
        nmlF_high_res = torch.Tensor(nmlF_high_res)
        nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res

        nmlB_high_res = np.transpose(nmlB_high_res, [2, 0, 1])  # [C, H, W] 형태로 변환
        nmlB_high_res = torch.Tensor(nmlB_high_res)
        nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

        # 데이터 딕셔너리로 반환
        return {
            'name': subject,
            'render_path': render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render': render,
            'nmlB_high_res': nmlB_high_res,
            'nmlF_high_res': nmlF_high_res
        }

    # __getitem__ 메서드 - DataLoader가 이 메서드를 호출하여 데이터셋의 요소를 가져옴
    def __getitem__(self, index):
        return self.get_item(index)