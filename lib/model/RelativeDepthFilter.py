import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..net_util import init_net
import cv2

# translated_Tanh 클래스 정의: tanh 활성화 함수의 출력을 +1로 이동시킴
class translated_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Tanh 활성화 값을 1로 이동시켜 반환
        return self.tanh(x) + 1.0

# widened_Tanh 클래스 정의: tanh 활성화 함수의 출력을 2배로 확장
class widened_Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Tanh 활성화 값을 2배로 확장하여 반환
        return self.tanh(x) * 2.0

# RelativeDepthFilter 클래스 정의: 상대 깊이 필터를 위한 모델 클래스
class RelativeDepthFilter(nn.Module):
    def __init__(self, opt):
        """
        RelativeDepthFilter 초기화 메서드.
        
        opt: 옵션 객체 (모델 구성에 필요한 설정 포함)
        """
        super(RelativeDepthFilter, self).__init__()

        self.name = 'depthfilter'  # 모델 이름 설정
        self.opt = opt  # 옵션 객체를 인스턴스 변수로 저장

        # 두 번째 스테이지가 아닌 경우 (기본적인 깊이 필터링)
        if not self.opt.second_stage_depth:
            from .UNet import UNet
            n_channels = 4  # 기본 채널 수 (예: RGB + Depth)

            # 깊이 학습에 법선 맵 사용 시 채널 수 증가
            if self.opt.use_normal_map_for_depth_training:
                n_channels += 3  # RGB + Depth + Normal Maps

            # UNet 모델 초기화 (마지막 활성화는 translated_Tanh 사용)
            self.image_filter = UNet(
                n_channels=n_channels, 
                n_classes=1, 
                bilinear=False, 
                last_op=translated_Tanh()
            )
        # 두 번째 스테이지 깊이 필터링 (DifferenceUNet 사용)
        elif self.opt.second_stage_depth:
            n_channels = 4
            if self.opt.use_normal_map_for_depth_training:
                n_channels += 3
            
            from .UNet import DifferenceUNet
            self.image_filter = DifferenceUNet(
                n_channels=n_channels, 
                n_classes=1, 
                bilinear=False, 
                last_op=widened_Tanh(), 
                scale_factor=2
            )
        else:
            raise Exception("Incorrect config")  # 잘못된 설정 예외 처리

        self.im_feat_list = []  # 이미지 특징을 저장할 리스트

        init_net(self)  # 네트워크 가중치 초기화

    def filter(self, images):
        '''
        이미지를 입력으로 받아 완전 합성곱 네트워크를 적용하여 특징을 추출.
        결과 특징은 im_feat_list에 저장됨.
        
        args:
            images: [B, C, H, W] 형태의 텐서 (배치 크기, 채널, 높이, 너비)
        '''
        self.im_feat_list = self.image_filter(images)  # 이미지 필터 적용 후 특징 추출

    def get_im_feat(self):
        """
        이미지 특징 반환 메서드.
        """
        return self.im_feat_list

    def generate_depth_map(self):
        """
        깊이 맵 생성 메서드. 현재 저장된 이미지 특징에서 깊이 맵 반환.
        """
        return self.get_im_feat()

    def get_error(self):
        '''
        예측 결과와 정답 레이블 간의 손실 계산하여 반환.
        '''
        error = {}
        
        # 예측된 깊이 맵과 실제 깊이 맵 사이의 Smooth L1 Loss 계산
        error['Err'] = nn.SmoothL1Loss()(self.im_feat_list, self.groundtruth_depthmap)

        return error

    def forward(self, images, groundtruth_depthmap):
        """
        모델의 순방향 패스 정의. 입력 이미지에 대해 특징을 추출하고 손실 계산.
        
        images: 입력 이미지 텐서 [B, C, H, W]
        groundtruth_depthmap: 실제 깊이 맵 텐서 [B, C, H, W]
        """
        # 입력 이미지에 대해 필터 적용 (특징 추출)
        self.filter(images)

        # 실제 깊이 맵을 인스턴스 변수로 저장
        self.groundtruth_depthmap = groundtruth_depthmap
            
        # 손실 계산
        err = self.get_error()

        return err