import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from ..net_util import init_net
from ..net_util import CustomBCELoss
from ..networks import define_G
import cv2

class HGPIFuNetwNML(BasePIFuNet):
    '''
    HGPIFuNetwNML 클래스는 스택된 호그스허드(hourglass) 구조를 이미지 인코더로 사용하는 HGPIFu 모델을 구현합니다.
    이 클래스는 인간 파싱과 깊이 맵 생성을 위한 다양한 기능을 제공합니다.
    '''
    
    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 use_High_Res_Component=False
                 ):
        """
        HGPIFuNetwNML 클래스의 초기화 메서드.
        
        Args:
            opt: 옵션 객체로, 모델 구성에 필요한 설정을 포함합니다.
            projection_mode (str): 투영 모드 설정 (기본값: 'orthogonal').
            criteria (dict): 손실 함수 기준을 포함하는 딕셔너리 (기본값: {'occ': nn.MSELoss()}).
            use_High_Res_Component (bool): 고해상도 컴포넌트 사용 여부 (기본값: False).
        """
        super(HGPIFuNetwNML, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu_low_res'  # 모델 이름 설정
        self.opt = opt  # 옵션 객체 저장
        self.use_High_Res_Component = use_High_Res_Component  # 고해상도 컴포넌트 사용 여부 설정

        in_ch = 3  # 초기 입력 채널 수 설정 (RGB)
        try:
            # 프론트 및 백 노멀 맵 사용 여부에 따라 입력 채널 수 증가
            if opt.use_front_normal: 
                in_ch += 3  # 프론트 노멀 맵 추가
            if opt.use_back_normal: 
                in_ch += 3  # 백 노멀 맵 추가
        except:
            pass  # 옵션에 해당 속성이 없으면 무시

        # 깊이 맵 사용 및 프론트에 깊이 맵을 포함할 경우 입력 채널 수 증가
        if self.opt.use_depth_map and self.opt.depth_in_front:
            if not self.use_High_Res_Component:
                in_ch += 1  # 깊이 맵 추가
            elif self.use_High_Res_Component and self.opt.allow_highres_to_use_depth:
                in_ch += 1  # 고해상도 컴포넌트 사용 시 깊이 맵 추가
            else:
                pass  # 기타 경우는 무시

        # 인간 파싱 맵 사용 여부에 따라 입력 채널 수 증가
        if self.opt.use_human_parse_maps:
            if not self.use_High_Res_Component:
                if self.opt.use_groundtruth_human_parse_maps:
                    in_ch += 6  # 정답 인간 파싱 맵 추가
                else:
                    in_ch += 7  # 예측 인간 파싱 맵 추가
            else:
                pass  # 고해상도 컴포넌트 사용 시 추가 없음

        # 고해상도 컴포넌트 사용 여부에 따라 적절한 이미지 필터 초기화
        if self.use_High_Res_Component:
            from .DifferenceIntegratedHGFilters import DifferenceIntegratedHGFilter
            self.image_filter = DifferenceIntegratedHGFilter(
                1, 2, in_ch, 256,   
                opt.norm, opt.hg_down, False)  # DifferenceIntegratedHGFilter 초기화
        else:
            self.image_filter = HGFilter(
                opt.num_stack_low_res, opt.hg_depth_low_res, in_ch, opt.hg_dim_low_res,   
                opt.norm, opt.hg_down, False)  # HGFilter 초기화

        # 깊이 맵을 프론트에 사용하지 않을 경우, MLP의 입력 차원 수정
        if self.opt.use_depth_map and not self.opt.depth_in_front:
            self.opt.mlp_dim_low_res[0] += 1  # 깊이 맵을 위한 차원 추가
            print("Overwriting self.opt.mlp_dim_low_res to add in 1 dim for depth map!")

        # MLP 초기화: 필터 특징을 바탕으로 깊이 예측
        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim_low_res,  
            merge_layer=self.opt.merge_layer_low_res,  
            res_layers=self.opt.mlp_res_layers_low_res,   
            norm="no_norm",
            last_op=nn.Sigmoid())  # 최종 활성화 함수로 Sigmoid 사용

        # 깊이 정규화기 초기화
        self.spatial_enc = DepthNormalizer(opt)

        self.im_feat_list = []  # 이미지 특징 리스트 초기화
        self.tmpx = None  # 임시 변수 초기화
        self.normx = None  # 정규화된 변수 초기화
        self.phi = None  # 중간 활성화 변수 초기화

        self.intermediate_preds_list = []  # 중간 예측 리스트 초기화

        init_net(self)  # 네트워크 가중치 초기화

        self.netF = None  # 추가 네트워크 변수 초기화
        self.netB = None

        self.nmlF = None  # 프론트 노멀 맵 변수 초기화
        self.nmlB = None  # 백 노멀 맵 변수 초기화

        self.gamma = None  # 감마 변수 초기화

        self.current_depth_map = None  # 현재 깊이 맵 변수 초기화

    def filter(self, images, nmlF=None, nmlB=None, current_depth_map=None, netG_output_map=None, human_parse_map=None, mask_low_res_tensor=None, mask_high_res_tensor=None):
        '''
        이미지에 완전 합성곱 네트워크를 적용하여 특징을 추출합니다.
        추출된 특징은 self.im_feat_list에 저장됩니다.
        
        Args:
            images (Tensor): 입력 이미지 텐서 [배치 크기, 채널, 높이, 너비]
            nmlF (Tensor, optional): 프론트 노멀 맵 텐서
            nmlB (Tensor, optional): 백 노멀 맵 텐서
            current_depth_map (Tensor, optional): 현재 깊이 맵 텐서
            netG_output_map (Tensor, optional): 네트워크 G의 출력 맵
            human_parse_map (Tensor, optional): 인간 파싱 맵 텐서
            mask_low_res_tensor (Tensor, optional): 저해상도 마스크 텐서
            mask_high_res_tensor (Tensor, optional): 고해상도 마스크 텐서
        '''
        # 깊이 맵을 프론트에 사용하지 않을 경우, 현재 깊이 맵 저장
        if self.opt.use_depth_map and not self.opt.depth_in_front:
            self.current_depth_map = current_depth_map

        # 마스크 텐서 저장
        self.mask_high_res_tensor = mask_high_res_tensor
        self.mask_low_res_tensor = mask_low_res_tensor

        nmls = []  # 노멀 맵 리스트 초기화

        # 노멀 맵을 추가적으로 사용하는 경우
        with torch.no_grad():
            if self.opt.use_front_normal:
                if nmlF is None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")  # 노멀 맵이 없을 경우 예외 발생
                self.nmlF = nmlF  # 프론트 노멀 맵 저장
                nmls.append(self.nmlF)  # 노멀 맵 리스트에 추가
            if self.opt.use_back_normal:
                if nmlB is None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")  # 노멀 맵이 없을 경우 예외 발생
                self.nmlB = nmlB  # 백 노멀 맵 저장
                nmls.append(self.nmlB)  # 노멀 맵 리스트에 추가

        # 노멀 맵이 존재할 경우, 이미지에 노멀 맵을 추가
        if len(nmls) != 0:
            nmls = torch.cat(nmls, 1)  # 채널 방향으로 노멀 맵 합성
            # 이미지와 노멀 맵의 크기가 다를 경우, 업샘플링하여 맞춤
            if images.size()[2:] != nmls.size()[2:]:
                nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
            images = torch.cat([images, nmls], 1)  # 이미지에 노멀 맵 추가

        # 깊이 맵을 프론트에 사용하고 현재 깊이 맵이 존재할 경우, 이미지에 깊이 맵 추가
        if self.opt.use_depth_map and self.opt.depth_in_front and (current_depth_map is not None):
            images = torch.cat([images, current_depth_map], 1)

        # 인간 파싱 맵을 사용하는 경우, 이미지에 인간 파싱 맵 추가
        if self.opt.use_human_parse_maps and (human_parse_map is not None):
            images = torch.cat([images, human_parse_map], 1)

        # 고해상도 컴포넌트를 사용하는 경우
        if self.use_High_Res_Component: 
            self.im_feat_list, self.normx = self.image_filter(images, netG_output_map)  # DifferenceIntegratedHGFilter 적용
        else:
            self.im_feat_list, self.normx = self.image_filter(images)  # HGFilter 적용

        # 훈련 모드가 아닌 경우, 마지막 스택의 특징만 사용
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True):
        '''
        주어진 3D 포인트에 대해 2D 프로젝션을 계산하고, 예측을 수행합니다.
        필터는 미리 호출되어야 합니다.
        예측은 self.preds에 저장됩니다.
        
        Args:
            points (Tensor): 3D 포인트 [배치 크기, 3, N]
            calibs (Tensor): 카메라 캘리브레이션 매트릭스 [배치 크기, 3, 4]
            transforms (Tensor, optional): 이미지 공간 좌표 변환 매트릭스 [배치 크기, 2, 3]
            labels (Tensor, optional): 실제 레이블 [배치 크기, C, N]
            update_pred (bool, optional): 예측 업데이트 여부 (기본값: True)
            update_phi (bool, optional): 중간 활성화 변수 업데이트 여부 (기본값: True)
        
        Returns:
            Tensor: 예측 결과 [배치 크기, C, N]
        '''
        # 3D 포인트를 2D 이미지 공간으로 프로젝션
        xyz = self.projection(points, calibs, transforms)  # [배치 크기, 3, N]
        xy = xyz[:, :2, :]  # [배치 크기, 2, N]

        # 마스크 값을 적용하여 예측 범위 설정
        if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
            mask_values = self.index(self.mask_high_res_tensor, xy) 
        if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
            mask_values = self.index(self.mask_low_res_tensor, xy) 

        # 바운딩 박스 내에 있는지 확인
        in_bb = (xyz >= -1) & (xyz <= 1)  # [배치 크기, 3, N]
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]  # [배치 크기, N]
        in_bb = in_bb[:, None, :].detach().float()  # [배치 크기, 1, N]

        # (0,0,0) 포인트는 제외
        is_zero_bool = (xyz == 0)  # [배치 크기, 3, N]
        is_zero_bool = is_zero_bool[:, 0, :] & is_zero_bool[:, 1, :] & is_zero_bool[:, 2, :]  # [배치 크기, N]
        not_zero_bool = torch.logical_not(is_zero_bool)  # [배치 크기, N]
        not_zero_bool = not_zero_bool[:, None, :].detach().float()  # [배치 크기, 1, N]

        # 레이블이 제공된 경우, 마스크를 적용하여 레이블 설정
        if labels is not None:
            self.labels = in_bb * labels  # [배치 크기, C, N]
            self.labels = not_zero_bool * self.labels  # (0,0,0) 포인트 제외

            size_of_batch = self.labels.shape[0]  # 배치 크기 저장

        # 공간적 특징 추출 (정규화된 z 값)
        sp_feat = self.spatial_enc(xyz, calibs=calibs)  # [배치 크기, 1, N]

        intermediate_preds_list = []  # 중간 예측 리스트 초기화
        phi = None  # 중간 활성화 변수 초기화

        # 각 스택의 이미지 특징을 순회하며 예측 수행
        for i, im_feat in enumerate(self.im_feat_list):
            if self.opt.use_depth_map and not self.opt.depth_in_front:
                # 깊이 맵을 프론트에 사용하지 않을 경우, 깊이 맵과 공간적 특징 추가
                point_local_feat_list = [self.index(im_feat, xy), self.index(self.current_depth_map, xy), sp_feat]
            else:
                # 기본적으로 이미지 특징과 공간적 특징만 사용
                point_local_feat_list = [self.index(im_feat, xy), sp_feat]

            # 로컬 특징 합성
            point_local_feat = torch.cat(point_local_feat_list, 1)  # [배치 크기, 필터 채널, N]

            # MLP를 통해 예측 수행
            pred, phi = self.mlp(point_local_feat)  # pred: [배치 크기, 1, N], phi: 중간 활성화

            # 마스크를 적용하여 예측 범위 설정
            pred = in_bb * pred
            pred = not_zero_bool * pred
            if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
                pred = mask_values * pred
            if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
                pred = mask_values * pred

            intermediate_preds_list.append(pred)  # 중간 예측 리스트에 추가

        # phi 업데이트 여부에 따라 업데이트
        if update_phi:
            self.phi = phi

        # 예측 업데이트 여부에 따라 업데이트
        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list  # 중간 예측 리스트 저장
            self.preds = self.intermediate_preds_list[-1]  # 최종 예측 저장

    def calc_normal(self, points, calibs, transforms=None, labels=None, delta=0.01, fd_type='forward'):
        '''
        모델 공간에서 표면 노멀을 계산합니다.
        마지막 스택에서만 노멀을 계산합니다.
        현재 구현은 전진 차분(forward difference)을 사용합니다.
        
        Args:
            points (Tensor): 3D 포인트 [배치 크기, 3, N]
            calibs (Tensor): 카메라 캘리브레이션 매트릭스 [배치 크기, 3, 4]
            transforms (Tensor, optional): 이미지 공간 좌표 변환 매트릭스 [배치 크기, 2, 3]
            labels (Tensor, optional): 실제 레이블 [배치 크기, C, N]
            delta (float, optional): 유한 차분을 위한 섭동 값 (기본값: 0.01)
            fd_type (str, optional): 유한 차분 유형 (forward/backward/central) (기본값: 'forward')
        '''
        # x, y, z 축에 대해 섭동을 가해 새로운 포인트 생성
        pdx = points.clone()
        pdx[:, 0, :] += delta  # x 축 섭동
        pdy = points.clone()
        pdy[:, 1, :] += delta  # y 축 섭동
        pdz = points.clone()
        pdz[:, 2, :] += delta  # z 축 섭동

        if labels is not None:
            self.labels_nml = labels  # 노멀 레이블 저장

        # 모든 포인트를 하나의 텐서로 쌓기
        points_all = torch.stack([points, pdx, pdy, pdz], 3)  # [배치 크기, 3, N, 4]
        points_all = points_all.view(*points.size()[:2], -1)  # [배치 크기, 3, N*4]

        # 포인트를 2D 이미지 공간으로 프로젝션
        xyz = self.projection(points_all, calibs, transforms)  # [배치 크기, 3, N*4]
        xy = xyz[:, :2, :]  # [배치 크기, 2, N*4]

        # 마지막 스택의 이미지 특징 추출
        im_feat = self.im_feat_list[-1]
        sp_feat = self.spatial_enc(xyz, calibs=calibs)  # [배치 크기, 1, N*4]

        # 포인트의 로컬 특징 합성
        point_local_feat_list = [self.index(im_feat, xy), sp_feat]
        point_local_feat = torch.cat(point_local_feat_list, 1)  # [배치 크기, 필터 채널, N*4]

        # MLP를 통해 예측 수행
        pred = self.mlp(point_local_feat)[0]  # [배치 크기, 1, N*4]

        # 예측을 4개의 섭동된 포인트에 대해 분할
        pred = pred.view(*pred.size()[:2], -1, 4)  # [배치 크기, 1, N, 4]

        # 유한 차분을 통한 노멀 계산 (delta로 나누는 부분은 생략됨)
        dfdx = pred[:, :, :, 1] - pred[:, :, :, 0]  # x 방향 변화
        dfdy = pred[:, :, :, 2] - pred[:, :, :, 0]  # y 방향 변화
        dfdz = pred[:, :, :, 3] - pred[:, :, :, 0]  # z 방향 변화

        # 노멀 벡터 계산 및 정규화
        nml = -torch.cat([dfdx, dfdy, dfdz], 1)  # [배치 크기, 3, N]
        nml = F.normalize(nml, dim=1, eps=1e-8)  # 정규화된 노멀 벡터

        self.nmls = nml  # 노멀 벡터 저장

    def get_im_feat(self):
        '''
        마지막 스택의 이미지 필터 특징을 반환합니다.
        
        Returns:
            Tensor: 마지막 스택의 이미지 필터 특징 [배치 크기, 채널, 높이, 너비]
        '''
        return self.im_feat_list[-1]

    def get_error(self, points=None):
        '''
        예측과 실제 레이블 간의 손실을 계산하여 반환합니다.
        
        Args:
            points (Tensor, optional): 사용되지 않음. (기본값: None)
        
        Returns:
            dict: 손실 값이 포함된 딕셔너리
        '''
        error = {}
        error['Err(occ)'] = 0  # 초기화

        # 모든 중간 예측에 대해 손실 계산
        for preds in self.intermediate_preds_list:
            error['Err(occ)'] += self.criteria['occ'](preds, self.labels)

        error['Err(occ)'] /= len(self.intermediate_preds_list)  # 평균 손실 계산

        # 노멀 손실 계산 (노멀 맵이 존재할 경우)
        if self.nmls is not None and self.labels_nml is not None:
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)

        return error  # 손실 딕셔너리 반환

    def forward(self, images, points, calibs, labels, points_nml=None, labels_nml=None, nmlF=None, nmlB=None, current_depth_map=None, netG_output_map=None, human_parse_map=None, mask_low_res_tensor=None, mask_high_res_tensor=None):
        '''
        모델의 순방향 전달을 정의합니다. 이미지를 필터링하고, 포인트에 대한 예측을 수행하며, 필요시 노멀 계산을 수행합니다.
        
        Args:
            images (Tensor): 입력 이미지 텐서 [배치 크기, 채널, 높이, 너비]
            points (Tensor): 3D 포인트 [배치 크기, 3, N]
            calibs (Tensor): 카메라 캘리브레이션 매트릭스 [배치 크기, 3, 4]
            labels (Tensor): 실제 레이블 [배치 크기, C, N]
            points_nml (Tensor, optional): 노멀 계산을 위한 3D 포인트 [배치 크기, 3, N]
            labels_nml (Tensor, optional): 노멀 계산을 위한 실제 레이블 [배치 크기, 3, N]
            nmlF (Tensor, optional): 프론트 노멀 맵 텐서
            nmlB (Tensor, optional): 백 노멀 맵 텐서
            current_depth_map (Tensor, optional): 현재 깊이 맵 텐서
            netG_output_map (Tensor, optional): 네트워크 G의 출력 맵
            human_parse_map (Tensor, optional): 인간 파싱 맵 텐서
            mask_low_res_tensor (Tensor, optional): 저해상도 마스크 텐서
            mask_high_res_tensor (Tensor, optional): 고해상도 마스크 텐서
        
        Returns:
            tuple: (손실 딕셔너리, 예측 결과 텐서)
        '''
        # 이미지 필터링
        self.filter(images, 
                    nmlF=nmlF, 
                    nmlB=nmlB, 
                    current_depth_map=current_depth_map, 
                    netG_output_map=netG_output_map, 
                    human_parse_map=human_parse_map, 
                    mask_low_res_tensor=mask_low_res_tensor, 
                    mask_high_res_tensor=mask_high_res_tensor)
        
        # 포인트에 대한 예측 수행
        self.query(points, calibs, labels=labels)

        # 노멀 계산 (노멀 포인트와 레이블이 제공된 경우)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, labels=labels_nml)

        # 예측 결과 가져오기
        res = self.get_preds()
        
        # 손실 계산
        err = self.get_error()

        return err, res  # 손실과 예측 결과 반환