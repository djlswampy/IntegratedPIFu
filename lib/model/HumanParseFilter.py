import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..net_util import init_net
import cv2

# HumanParseFilter 클래스 정의 - PyTorch nn.Module을 상속받아 신체 파싱을 위한 필터 모델 구현
class HumanParseFilter(nn.Module):

    def __init__(self, 
                 opt, 
                 criteria={'err': nn.CrossEntropyLoss() }
                 ):
        """
        HumanParseFilter 클래스 초기화 함수.
        Args:
            opt: 옵션 객체, 모델 설정과 관련된 인자 포함.
            criteria: 손실 계산에 사용될 기준. 기본적으로 CrossEntropyLoss 사용.
        """
        super(HumanParseFilter, self).__init__()

        self.name = 'humanparsefilter'  # 모델 이름 지정
        self.criteria = criteria  # 손실 기준 정의

        self.opt = opt  # 옵션 저장
        
        # UNet 모델을 사용하여 이미지 필터 정의. 채널 수 결정
        from .UNet import UNet
        n_channels = 3  # 기본적으로 RGB 이미지 입력
        if self.opt.use_normal_map_for_parse_training:
            n_channels += 3  # 법선 맵(Normal Map)이 있을 경우 채널 추가

        # UNet을 이용해 이미지 필터링. 출력 클래스는 7개 (사람 신체 부위 분류)
        self.image_filter = UNet(n_channels=n_channels, n_classes=7, bilinear=False)

        # 이미지 특징 저장 리스트
        self.im_feat_list = []

        # 네트워크 가중치 초기화
        init_net(self)

    def filter(self, images):
        '''
        이미지를 필터링하여 신경망을 통해 결과 생성.
        생성된 특징 맵은 self.im_feat_list에 저장됨.
        Args:
            images: 입력 이미지 텐서 (형태: [B, C, H, W])
        '''
        # UNet을 사용하여 이미지 필터링 후 특징 맵 저장
        self.im_feat_list = self.image_filter(images)

    def get_im_feat(self):
        """
        저장된 이미지 특징 맵 반환.
        """
        return self.im_feat_list

    def generate_parse_map(self):
        """
        신체 파싱 맵 생성. 각 픽셀의 클래스를 결정.
        Returns:
            신체 파싱 맵 텐서 (형태: [B, H, W])
        """
        im_feat = self.im_feat_list  # [B, C, H, W] 형태
        # 각 픽셀의 가장 높은 점수를 갖는 클래스를 선택하여 파싱 맵 생성
        im_feat = torch.argmax(im_feat, dim=1)  # [B, H, W]

        return im_feat

    def get_error(self):
        '''
        손실(오차)을 계산하여 반환.
        Returns:
            error: 손실 딕셔너리
        '''
        error = {}
        # CrossEntropyLoss를 사용하여 예측과 실제 레이블 간의 손실 계산
        error['Err'] = self.criteria['err'](self.im_feat_list, self.groundtruth_parsemap)

        return error

    def forward(self, images, groundtruth_parsemap):
        """
        모델의 순방향 전달을 정의.
        Args:
            images: 입력 이미지 텐서 [B, C, H, W]
            groundtruth_parsemap: 실제 신체 파싱 맵 [B, C, H, W]
        Returns:
            err: 손실 값
        """
        # 입력 이미지를 필터링하여 특징 추출
        self.filter(images)

        # 실제 파싱 맵에서 각 픽셀의 가장 높은 점수를 갖는 클래스를 추출
        self.groundtruth_parsemap = torch.argmax(groundtruth_parsemap, dim=1)  # [B, H, W]

        # 손실 계산
        err = self.get_error()

        return err