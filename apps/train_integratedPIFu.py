import sys
import os
import json

# 시스템 경로에 상위 디렉토리를 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import io
import gc

"""
low, high에 따라서 옵션 선택
"""
# from lib.options import BaseOptions
from lib.options_lowResPIFu import BaseOptions
from lib.model import HGPIFuNetwNML 
from lib.data import TrainDataset
from lib.mesh_util import save_obj_mesh_with_color, reconstruction, save_obj_mesh
from lib.geometry import index

# 랜덤 시드 설정
seed = 0 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 옵션 파서 초기화
parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

# 디버그 및 테스트 모드 설정
debug_mode = False
test_script_activate = True
test_script_activate_option_use_BUFF_dataset = False

# 모델 및 옵티마이저 체크포인트 설정
load_model_weights = True  # 모델 체크포인트 로드 여부
load_model_weights_for_high_res_too = False  # 하이 피푸 모델 체크포인트 로드 여부
load_model_weights_for_low_res_finetuning_config = 0 # 하이 피푸를 사용할 때, 로우 피푸의 가중치도 함께 수정하는 경우, 로우 피푸 옵티마이저 체크포인트 설정 (0: 체크포인트 로드 안 함, 1 또는 2: 특정 경로에서 체크포인트 로드)
checkpoint_folder_to_load_low_res = '/home/public/IntegratedPIFu/checkpoints/first_low_pifu_test' # 로우 피푸 옵티마이저 체크포인트 경로
checkpoint_folder_to_load_high_res = 'apps/checkpoints/Date_28_Jun_22_Time_02_49_38' # 하이 피푸 옵티마이저 체크포인트 경로
epoch_to_load_from_low_res = 1 # 로우 피푸 옵티마이저 에포크 번호
epoch_to_load_from_high_res = 2 # 하이 피푸 옵티마이저 에포크 번호

# 고해상도 구성 요소 사용 시 옵션 수정
if opt.use_High_Res_Component:
    # 로우 레졸루션 피푸의 시그마 값을 하이 레졸루션 컴포넌트의 시그마 값으로 수정
    opt.sigma_low_resolution_pifu = opt.High_Res_Component_sigma
    print("Modifying sigma_low_resolution_pifu to {0} for high resolution component!".format(opt.High_Res_Component_sigma))

# CPU에서 피클 파일을 로드하기 위한 클래스 정의
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 특정 모듈과 이름에 대해 커스텀 로딩 함수 반환
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# 샘플 시각화를 ply 파일로 저장하는 함수 정의
def save_samples_truncted_prob(fname, points, prob):
    r = (prob >= 0.5).reshape([-1, 1]) * 255  # 확률이 0.5 이상인 점의 빨간색 값 설정
    g = (prob < 0.5).reshape([-1, 1]) * 255   # 확률이 0.5 미만인 점의 초록색 값 설정
    b = np.zeros(r.shape)                     # 파란색 값은 0으로 설정

    # 포인트와 색상 정보를 합침
    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

# 학습률을 조정하는 함수 정의
def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    # 현재 에포크가 학습률 스케줄에 포함되어 있는지 확인
    if epoch in schedule:
        lr *= learning_rate_decay  # 학습률 감소 비율만큼 감소
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 옵티마이저의 모든 파라미터 그룹에 새로운 학습률 적용
    return lr  # 업데이트된 학습률 반환

# 메쉬를 생성하는 함수 정의
def gen_mesh(resolution, net, device, data, save_path, thresh=0.5, use_octree=True):
    # 데이터를 GPU 또는 CPU로 이동시키고 텐서로 변환
    calib_tensor = data['calib'].to(device=device)  # 캘리브레이션 데이터 텐서를 디바이스로 이동
    calib_tensor = torch.unsqueeze(calib_tensor,0)  # 배치 차원 추가
    
    b_min = data['b_min']  # 바운딩 박스 최소값
    print('b_min: ', b_min)
    b_max = data['b_max']  # 바운딩 박스 최대값
    print('b_max: ', b_max)

    # 저해상도 렌더링 이미지 불러오기
    image_low_tensor = data['render_low_pifu'].to(device=device)  
    image_low_tensor = image_low_tensor.unsqueeze(0)  # 배치 차원 추가

    # 앞면 normal map 사용 시
    if opt.use_front_normal:
        nmlF_low_tensor = data['nmlF'].to(device=device)
        nmlF_low_tensor = nmlF_low_tensor.unsqueeze(0)
    else:
        nmlF_low_tensor = None

    # 뒷면 normal map 사용 시
    if opt.use_back_normal:
        print("use_back_normal")
        nmlB_low_tensor = data['nmlB'].to(device=device)
        nmlB_low_tensor = nmlB_low_tensor.unsqueeze(0)
    else:
        nmlB_low_tensor = None

    # depth map 사용 시
    if opt.use_depth_map:
        depth_map_low_res = data['depth_map_low_res'].to(device=device)
        depth_map_low_res = depth_map_low_res.unsqueeze(0)
    else: 
        depth_map_low_res = None

    # human parse map 사용 시
    if opt.use_human_parse_maps:
        human_parse_map = data['human_parse_map'].to(device=device)
        human_parse_map = human_parse_map.unsqueeze(0)
    else:
        human_parse_map=None

    # 고해상도 구성 요소 사용 시
    if opt.use_High_Res_Component:
        netG, highRes_netG = net  # 고해상도 네트워크와 기본 네트워크를 가져옴
        net = highRes_netG  # 고해상도 네트워크를 net 변수에 할당

        # 고해상도 이미지 불러오기
        image_high_tensor = data['original_high_res_render'].to(device=device)
        image_high_tensor = torch.unsqueeze(image_high_tensor,0)
        
        # 앞면 고해상도 normal map 불러오기
        if opt.use_front_normal:
            nmlF_high_tensor = data['nmlF_high_res'].to(device=device)
            nmlF_high_tensor = nmlF_high_tensor.unsqueeze(0)
        else:
            nmlF_high_tensor = None

        # 뒷면 고해상도 normal map 불러오기
        if opt.use_back_normal:
            nmlB_high_tensor = data['nmlB_high_res'].to(device=device)
            nmlB_high_tensor = nmlB_high_tensor.unsqueeze(0)
        else:
            nmlB_high_tensor = None

        # 고해상도 depth map 불러오기 (옵션에 따라 다름)
        if opt.use_depth_map and opt.allow_highres_to_use_depth:
            depth_map_high_res = data['depth_map'].to(device=device)
            depth_map_high_res = depth_map_high_res.unsqueeze(0)
        else: 
            depth_map_high_res = None

        # 고해상도 마스크 불러오기
        if opt.use_mask_for_rendering_high_res:
            mask_high_res_tensor = data['mask'].to(device=device)
            mask_high_res_tensor = mask_high_res_tensor.unsqueeze(0)
        else:
            mask_high_res_tensor = None

        # 고해상도 네트워크 필터링 및 특징 추출
        netG.filter(image_low_tensor, 
                   nmlF=nmlF_low_tensor, 
                   nmlB=nmlB_low_tensor, 
                   current_depth_map=depth_map_low_res, 
                   human_parse_map=human_parse_map)
        netG_output_map = netG.get_im_feat()  # Low-PIFu의 출력 특징 맵 획득

        # 고해상도 네트워크 필터링 수행
        net.filter(image_high_tensor, 
                   nmlF=nmlF_high_tensor, 
                   nmlB=nmlB_high_tensor, 
                   current_depth_map=depth_map_high_res, 
                   netG_output_map=netG_output_map, 
                   mask_low_res_tensor=None, 
                   mask_high_res_tensor=mask_high_res_tensor)
        image_tensor = image_high_tensor  # 고해상도 이미지 텐서로 설정

    else:
        # 저해상도 마스크 사용 시
        if opt.use_mask_for_rendering_low_res:
            mask_low_res_tensor = data['mask_low_pifu'].to(device=device)
            mask_low_res_tensor = mask_low_res_tensor.unsqueeze(0)
        else:
            mask_low_res_tensor = None

        # 저해상도 네트워크 필터링 수행
        net.filter(image_low_tensor, 
                   nmlF=nmlF_low_tensor, 
                   nmlB=nmlB_low_tensor, 
                   current_depth_map=depth_map_low_res, 
                   human_parse_map=human_parse_map, 
                   mask_low_res_tensor=mask_low_res_tensor, 
                   mask_high_res_tensor=None)
        image_tensor = image_low_tensor  # 저해상도 이미지 텐서로 설정

    try:
        # 이미지 저장 (렌더링 결과)
        save_img_path = save_path[:-4] + '.png'  # .obj 파일을 .png로 변경하여 저장 경로 설정
        save_img_list = []
        # image_tensor.shape[0]는 첫 번째 차원의 크기를 반환
        for v in range(image_tensor.shape[0]):
            # 이미지 텐서를 저장 가능한 형식으로 변환하고 저장
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)  # 여러 이미지를 하나로 합침
        cv2.imwrite(save_img_path, save_img)  # 이미지 파일로 저장

        # 메쉬 재구성 (marching cubes 알고리즘 사용)
        verts, faces, _, _ = reconstruction(
            net, device, calib_tensor, 
            resolution, b_min=b_min, b_max=b_max, 
            use_octree=use_octree, num_samples=50000)
        
        save_obj_mesh(save_path, verts, faces)

        """
        컬러 넣는 코드(안씀)
        
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=device).float()  # 정점 텐서 생성 및 디바이스로 이동

        # 3D 좌표를 2D 이미지 좌표로 변환
        xyz_tensor = net.projection(verts_tensor, calib_tensor)  
        uv = xyz_tensor[:, :2, :]  # 2D 좌표 추출
        color = index(image_tensor, uv).detach().cpu().numpy()[0].T  # 색상 정보 추출
        color = color * 0.5 + 0.5  # 색상 정규화

        # 메쉬 저장 (색상 포함)
        save_obj_mesh_with_color(save_path, verts, faces, color)  # 색상 정보가 포함된 메쉬 파일 저장
        """
        
    except Exception as e:
        print(e)  # 예외 발생 시 에러 메시지 출력
        print("Cannot create marching cubes at this time.")  # 메쉬 생성 실패 메시지 출력

# 학습 함수 정의
def train(opt):
    global gen_test_counter  # 전역 변수 gen_test_counter 사용
    currently_epoch_to_update_low_res_pifu = True  # 현재 에포크에서 저해상도 PIFu 업데이트 여부 초기화

    # 디바이스 설정 (GPU 또는 CPU)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device))  # 사용 중인 디바이스 출력

    # 디버그 모드 설정
    if debug_mode:
        opt.debug_mode = True
    else:
        opt.debug_mode = False
    
    # 테스트 스크립트 활성화 시 데이터셋 설정
    if test_script_activate:
        if test_script_activate_option_use_BUFF_dataset:
            from lib.data.BuffDataset import BuffDataset  # BUFF 데이터셋 모듈 임포트
            train_dataset = BuffDataset(opt)  # BUFF 데이터셋을 사용하여 데이터셋 초기화
        else:
            train_dataset = TrainDataset(opt, projection='orthogonal', phase='train', evaluation_mode=True)  # 평가 모드로 학습 데이터셋 초기화
    else:
        train_dataset = TrainDataset(opt, projection='orthogonal', phase='train')  # 일반 학습 모드로 데이터셋 초기화
    
    projection_mode = train_dataset.projection_mode  # 데이터셋의 투영 모드 가져오기

    # 디버그 모드일 경우 데이터셋 디렉토리 경로 수정
    if debug_mode:
        train_dataset.normal_directory_high_res = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_normal_maps"
        train_dataset.depth_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_refined_depth_maps_usingNormalOnly"
        train_dataset.human_parse_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_parse_maps"

    # 데이터 로더 설정
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, 
                                   shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, 
                                   pin_memory=opt.pin_memory)

    print('num_threads: ', opt.num_threads)
    print('num_epoch: ', opt.num_epoch)
    print('pin_memory: ', opt.pin_memory)
    print('use_back_normal: ', opt.use_back_normal)
    print('use_High_Res_Component: ', opt.use_High_Res_Component)
    print('batch_size: ', opt.batch_size)
    print('num_sample_inout: ', opt.num_sample_inout)
    print('update_low_res_pifu: ', opt.update_low_res_pifu)
    print('use_mask_for_rendering_high_res: ', opt.use_mask_for_rendering_high_res)
    print('train loader size: ', len(train_data_loader))  # 학습 데이터 로더의 크기 출력

    # 검증 데이터셋 설정
    if opt.useValidationSet:
        print("opt.useValidationSet: True")  # 검증 데이터셋 사용 여부 출력
        validation_dataset = TrainDataset(opt, projection='orthogonal', phase='validation', 
                                         evaluation_mode=False, validation_mode=True)  # 검증 데이터셋 초기화

        validation_epoch_cd_dist_list = []  # 에포크별 Chamfer 거리 리스트 초기화
        validation_epoch_p2s_dist_list = []  # 에포크별 Point-to-Surface 거리 리스트 초기화

        validation_graph_path = os.path.join(opt.results_path, opt.name, 'ValidationError_Graph.png')  # 검증 그래프 저장 경로 설정

    # 디버그 모드일 경우 검증 데이터셋 디렉토리 경로 수정
    if debug_mode:
        validation_dataset.normal_directory_high_res = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_normal_maps"
        validation_dataset.depth_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_refined_depth_maps_usingNormalOnly"
        validation_dataset.human_parse_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_parse_maps"

    # 네트워크 초기화
    netG = HGPIFuNetwNML(opt, projection_mode, use_High_Res_Component=False)  # Low-Resolution PIFu 네트워크 초기화

    if opt.use_High_Res_Component:
        highRes_netG = HGPIFuNetwNML(opt, projection_mode, use_High_Res_Component=True)  # High-Resolution Integrator 네트워크 초기화

    # 체크포인트 및 결과 경로 생성 (존재하지 않을 경우 생성)
    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)
    if not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name)):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if not os.path.exists('%s/%s' % (opt.results_path, opt.name)):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))

    print('opt.name:', opt.name)  # 옵션 이름 출력

    # 모델 가중치 로드
    if load_model_weights:
        modelG_path = "/home/public/IntegratedPIFu/checkpoints/first_high_pifu_train/netG_model_state_dict_epoch1.pickle"  # 로드할 netG 가중치 경로 설정
        print('Resuming from ', modelG_path)  # 가중치 로드 경로 출력

        if device == 'cpu':
            # CPU에서 피클 파일 로드
            with open(modelG_path, 'rb') as handle:
               netG_state_dict = CPU_Unpickler(handle).load()
        else:
            # GPU에서 피클 파일 로드
            with open(modelG_path, 'rb') as handle:
               netG_state_dict = pickle.load(handle)

        netG.load_state_dict(netG_state_dict, strict=True)  # netG에 가중치 로드
        
        # 고해상도 모델 가중치 로드
        if opt.use_High_Res_Component and load_model_weights_for_high_res_too:
            modelhighResG_path = "/home/public/IntegratedPIFu/checkpoints/first_high_pifu_train/highRes_netG_model_state_dict_epoch1.pickle"  # 고해상도 netG 가중치 경로 설정
            print('Resuming from ', modelhighResG_path)  # 가중치 로드 경로 출력

            if device == 'cpu':
                # CPU에서 고해상도 피클 파일 로드
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = CPU_Unpickler(handle).load()
            else:
                # GPU에서 고해상도 피클 파일 로드
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = pickle.load(handle)
            highRes_netG.load_state_dict(highResG_state_dict, strict=True)  # highRes_netG에 가중치 로드

    # 테스트 스크립트 활성화 시 메쉬 생성
    if test_script_activate:
        with torch.no_grad():  # 그래디언트 계산 비활성화
            print('generate mesh (test) ...')  # 테스트 메쉬 생성 시작 메시지 출력
            train_dataset.is_train = False  # 데이터셋을 학습 모드에서 테스트 모드로 전환
            netG = netG.to(device=device)  # netG를 디바이스로 이동
            netG.eval()  # netG를 평가 모드로 설정

            if opt.use_High_Res_Component:
                highRes_netG = highRes_netG.to(device=device)  # highRes_netG를 디바이스로 이동
                highRes_netG.eval()  # highRes_netG를 평가 모드로 설정

            if test_script_activate_option_use_BUFF_dataset:
                len_to_iterate = len(train_dataset)  # BUFF 데이터셋을 사용할 경우 전체 데이터셋 길이 사용
            else:
                len_to_iterate = 72  # 그렇지 않으면 72로 설정
            for gen_idx in tqdm(range(len_to_iterate)):  # 진행 상황을 표시하며 반복
                if test_script_activate_option_use_BUFF_dataset:
                    index_to_use = gen_idx  # BUFF 데이터셋 사용 시 인덱스 설정
                else:
                    index_to_use = gen_test_counter % len(train_dataset)  # 그렇지 않으면 모듈로 연산을 통해 인덱스 설정
                gen_test_counter += 10  # 테스트 카운터 증가
                train_data = train_dataset.get_item(index=index_to_use)  # 데이터셋에서 항목 가져오기
                save_path = '%s/%s/test_%s.obj' % (opt.results_path, opt.name, train_data['name'])  # 저장 경로 설정

                if opt.use_High_Res_Component:
                    # 고해상도 구성 요소를 사용할 경우, netG와 highRes_netG를 함께 사용하여 메쉬 생성
                    gen_mesh(resolution=opt.resolution, 
                             net=[netG, highRes_netG], 
                             device=device, 
                             data=train_data, 
                             save_path=save_path)
                else:
                    # 그렇지 않으면 기본 네트워크만 사용하여 메쉬 생성
                    gen_mesh(resolution=opt.resolution, 
                             net=netG, 
                             device=device, 
                             data=train_data, 
                             save_path=save_path)

        print("Testing is Done! Exiting...")  # 테스트 완료 메시지 출력
        return  # 함수 종료

    # 옵션 로그 저장
    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')  # 옵션 로그 파일 경로 설정
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))  # 옵션 설정을 JSON 형식으로 저장

    # 네트워크 및 옵티마이저 설정
    netG = netG.to(device=device)  # netG를 디바이스로 이동
    lr_G = opt.learning_rate_G  # netG의 초기 학습률 설정
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr_G, momentum=0, weight_decay=0)  # netG용 옵티마이저 초기화

    # 옵티마이저 가중치 로드
    if load_model_weights:
        optimizerG_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res))
        with open(optimizerG_path, 'rb') as handle:
           optimizerG_state_dict = pickle.load(handle)  # 옵티마이저 상태 로드
        optimizerG.load_state_dict(optimizerG_state_dict)  # 옵티마이저에 상태 적용

    if opt.use_High_Res_Component:
        highRes_netG = highRes_netG.to(device=device)  # highRes_netG를 디바이스로 이동
        lr_highRes = opt.learning_rate_MR  # highRes_netG의 초기 학습률 설정
        optimizer_highRes = torch.optim.RMSprop(highRes_netG.parameters(), lr=lr_highRes, momentum=0, weight_decay=0)  # highRes_netG용 옵티마이저 초기화

        if load_model_weights and load_model_weights_for_high_res_too:
            optimizer_highRes_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_highRes_epoch{0}.pickle".format(epoch_to_load_from_high_res))
            with open(optimizer_highRes_path, 'rb') as handle:
                optimizer_highRes_state_dict = pickle.load(handle)  # highRes_netG 옵티마이저 상태 로드
            optimizer_highRes.load_state_dict(optimizer_highRes_state_dict)  # highRes_netG 옵티마이저에 상태 적용
        
        if opt.update_low_res_pifu:
            # Low-Res Fine-Tuning 옵티마이저 초기화
            optimizer_lowResFineTune = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate_low_res_finetune, momentum=0, weight_decay=0)

            # load_model_weights_for_low_res_finetuning_config = 0 인 경우 스킵
            if load_model_weights and (load_model_weights_for_low_res_finetuning_config != 0):
                # 파인튜닝 구성에 따라 옵티마이저 상태 로드 경로 설정
                if load_model_weights_for_low_res_finetuning_config == 1:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res))
                elif load_model_weights_for_low_res_finetuning_config == 2:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_lowResFineTune_epoch{0}.pickle".format(epoch_to_load_from_high_res))
                else:
                    raise Exception('Incorrect use of load_model_weights_for_low_res_finetuning_config!')  # 잘못된 구성 시 예외 발생
                
                with open(optimizer_lowResFineTune_path, 'rb') as handle:
                   optimizer_lowResFineTune_state_dict = pickle.load(handle)  # 파인튜닝 옵티마이저 상태 로드
                optimizer_lowResFineTune.load_state_dict(optimizer_lowResFineTune_state_dict)  # 파인튜닝 옵티마이저에 상태 적용

    # 학습 시작
    start_epoch = 0  # 학습을 시작할 에포크 번호 설정
    for epoch in range(start_epoch, opt.num_epoch):  # 지정된 에포크 수만큼 반복
        print("start of epoch {}".format(epoch))  # 현재 에포크 번호 출력

        netG.train()  # 기본 네트워크를 학습 모드로 설정
        if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 여부 확인
            if opt.update_low_res_pifu:  # 저해상도 PIFu를 업데이트할지 여부 확인
                if (epoch < opt.epoch_to_start_update_low_res_pifu):  # 업데이트 시작 에포크 이전인지 확인
                    currently_epoch_to_update_low_res_pifu = False 
                    print("currently_epoch_to_update_low_res_pifu remains at False for this epoch")  # 업데이트 비활성화 상태 출력
                elif (epoch >= opt.epoch_to_end_update_low_res_pifu):  # 업데이트 종료 에포크 이후인지 확인
                    currently_epoch_to_update_low_res_pifu = False
                    print("No longer updating low_res_pifu! In the Finetune Phase")  # 업데이트 종료 상태 출력
                elif (epoch % opt.epoch_interval_to_update_low_res_pifu == 0):  # 특정 간격의 에포크인지 확인
                    currently_epoch_to_update_low_res_pifu = not currently_epoch_to_update_low_res_pifu  # 업데이트 상태 토글
                    print("Updating currently_epoch_to_update_low_res_pifu to: ",currently_epoch_to_update_low_res_pifu)  # 상태 변경 출력
                else:
                    pass  # 다른 경우는 아무 동작도 하지 않음

            # 현재 에포크에서 저해상도 PIFu를 업데이트할지 여부에 따라 네트워크 모드 설정
            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                netG.train()  # 기본 네트워크를 학습 모드로 설정
                highRes_netG.eval()  # 고해상도 네트워크를 평가 모드로 설정
            else:
                netG.eval()  # 기본 네트워크를 평가 모드로 설정
                highRes_netG.train()  # 고해상도 네트워크를 학습 모드로 설정

        """
        이 때 데이터 로더와 반복문을 통해 데이터셋에서 배치 단위로 데이터를 로드하고, 로드 과정에서 __getitem__()이 호출된다.
        """
        for train_idx, train_data in enumerate(train_data_loader):  # 학습 데이터 로더를 통해 배치 단위로 데이터 반복
            print("batch {}".format(train_idx))  # 현재 배치 번호 출력

            # 가비지 컬렉터
            gc.collect()

            # 데이터 준비
            calib_tensor = train_data['calib'].to(device=device)  # 캘리브레이션 데이터 텐서를 디바이스로 이동

            if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
                render_low_pifu_tensor = train_data['render_low_pifu'].to(device=device)  # 저해상도 렌더링 이미지 텐서를 디바이스로 이동
                render_pifu_tensor = train_data['original_high_res_render'].to(device=device)  # 원본 고해상도 렌더링 이미지 텐서를 디바이스로 이동
                
                if opt.use_front_normal:  # 앞면 노멀 맵 사용 여부 확인
                    nmlF_low_tensor = train_data['nmlF'].to(device=device)  # 저해상도 앞면 노멀 맵 텐서를 디바이스로 이동
                    nmlF_tensor = train_data['nmlF_high_res'].to(device=device)  # 고해상도 앞면 노멀 맵 텐서를 디바이스로 이동
                else:
                    nmlF_tensor = None  # 노멀 맵 사용하지 않으면 None으로 설정

                if opt.use_back_normal:  # 뒷면 노멀 맵 사용 여부 확인
                    nmlB_low_tensor = train_data['nmlB'].to(device=device)  # 저해상도 뒷면 노멀 맵 텐서를 디바이스로 이동
                    nmlB_tensor = train_data['nmlB_high_res'].to(device=device)  # 고해상도 뒷면 노멀 맵 텐서를 디바이스로 이동
                else:
                    nmlB_low_tensor = None  # 노멀 맵 사용하지 않으면 None으로 설정
                    nmlB_tensor = None

            else:  # 고해상도 컴포넌트 사용하지 않을 경우
                render_pifu_tensor = train_data['render_low_pifu'].to(device=device)  # 저해상도 렌더링 이미지 텐서를 디바이스로 이동
                
                if opt.use_front_normal:  # 앞면 노멀 맵 사용 여부 확인
                    nmlF_tensor = train_data['nmlF'].to(device=device)  # 앞면 노멀 맵 텐서를 디바이스로 이동
                else:
                    nmlF_tensor = None  # 노멀 맵 사용하지 않으면 None으로 설정

                if opt.use_back_normal:  # 뒷면 노멀 맵 사용 여부 확인
                    nmlB_tensor = train_data['nmlB'].to(device=device)  # 뒷면 노멀 맵 텐서를 디바이스로 이동
                else:
                    nmlB_tensor = None  # 노멀 맵 사용하지 않으면 None으로 설정

            # depth map 사용 여부에 따른 처리
            if opt.use_depth_map:
                current_depth_map = train_data['depth_map'].to(device=device)  # 현재 깊이 맵을 디바이스로 이동
                if opt.depth_in_front and (not opt.use_High_Res_Component):  # 특정 옵션 조건 확인
                    current_depth_map = train_data['depth_map_low_res'].to(device=device)  # 저해상도 깊이 맵을 사용
                if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
                    current_low_depth_map = train_data['depth_map_low_res'].to(device=device)  # 저해상도 깊이 맵을 디바이스로 이동
                    if not opt.allow_highres_to_use_depth:  # 고해상도에서 깊이 맵 사용을 허용하지 않는 경우
                        current_depth_map = None  # 현재 깊이 맵을 None으로 설정
            else: 
                current_depth_map = None  # 깊이 맵 사용하지 않으면 None으로 설정
                current_low_depth_map = None  # 저해상도 깊이 맵도 None으로 설정

            # human parse map 사용 여부에 따른 처리
            if opt.use_human_parse_maps:
                human_parse_map = train_data['human_parse_map'].to(device=device)  # 인간 파싱 맵을 디바이스로 이동
            else:
                human_parse_map = None  # 사용하지 않으면 None으로 설정

            # 샘플링된 포인트와 레이블을 텐서로 이동
            samples_low_res_pifu_tensor = train_data['samples_low_res_pifu'].to(device=device)
            labels_low_res_pifu_tensor = train_data['labels_low_res_pifu'].to(device=device)

            if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
                # Low-Resolution PIFu 네트워크(netG)를 사용하여 입력 데이터를 필터링 및 특징 추출
                netG.filter(render_low_pifu_tensor, nmlF=nmlF_low_tensor, nmlB=nmlB_low_tensor, current_depth_map=current_low_depth_map, human_parse_map=human_parse_map)
                # 필터링된 특징 받아옴
                netG_output_map = netG.get_im_feat()

                # High-Resolution Integrator 네트워크를 사용하여 깊이 예측 및 오차 계산
                error_high_pifu, res_high_pifu = highRes_netG.forward(
                    images=render_pifu_tensor,  # 고해상도 렌더링 이미지
                    points=samples_low_res_pifu_tensor,  # 샘플 포인트
                    calibs=calib_tensor,  # 캘리브레이션 데이터
                    labels=labels_low_res_pifu_tensor,  # 포인트 레이블
                    points_nml=None,  # 노멀 포인트 (사용하지 않음)
                    labels_nml=None,  # 노멀 레이블 (사용하지 않음)
                    nmlF=nmlF_tensor,  # 앞면 노멀 맵
                    nmlB=nmlB_tensor,  # 뒷면 노멀 맵
                    current_depth_map=current_depth_map,  # 현재 깊이 맵
                    netG_output_map=netG_output_map  # Low-PIFu의 출력 특징 맵
                )
                # Low-Res PIFu를 업데이트해야 하는지 여부를 확인
                if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                    # Low-Res Fine-Tuning 옵티마이저의 그래디언트 초기화
                    optimizer_lowResFineTune.zero_grad()
                    
                    # High-Resolution Integrator의 오차 'Err(occ)'에 대해 역전파 수행하여 그래디언트 계산
                    error_high_pifu['Err(occ)'].backward()
                    
                    # 현재 에포크에서의 고해상도 손실 값을 가져옴
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    
                    # Low-Res Fine-Tuning 옵티마이저를 사용하여 Low-Res PIFu의 파라미터 업데이트
                    optimizer_lowResFineTune.step()
                else:
                    # High-Res Integrator 옵티마이저의 그래디언트 초기화
                    optimizer_highRes.zero_grad()
                    
                    # High-Res Integrator의 오차 'Err(occ)'에 대해 역전파 수행하여 그래디언트 계산
                    error_high_pifu['Err(occ)'].backward()
                    
                    # 현재 에포크에서의 고해상도 손실 값을 가져옴
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    
                    # High-Res Integrator 옵티마이저를 사용하여 High-Res Integrator의 파라미터 업데이트
                    optimizer_highRes.step()

                # 현재 에포크, 손실 값, 학습률 출력
                print(
                    'Name: {0} | Epoch: {1} | error_high_pifu: {2:.06f} | LR: {3:.06f} '.format(
                        opt.name, epoch, curr_high_loss, lr_highRes)
                )

                r = res_high_pifu  # 예측 결과 저장

            else:  # 고해상도 컴포넌트 사용하지 않을 경우
                # Low-Resolution PIFu 네트워크를 사용하여 입력 데이터에 대해 순전파 수행 및 오차 계산
                error_low_res_pifu, res_low_res_pifu = netG.forward(
                    images=render_pifu_tensor,  # 렌더링 이미지
                    points=samples_low_res_pifu_tensor,  # 샘플 포인트
                    calibs=calib_tensor,  # 캘리브레이션 데이터
                    labels=labels_low_res_pifu_tensor,  # 포인트 레이블
                    points_nml=None,  # 노멀 포인트 (사용하지 않음)
                    labels_nml=None,  # 노멀 레이블 (사용하지 않음)
                    nmlF=nmlF_tensor,  # 앞면 노멀 맵
                    nmlB=nmlB_tensor,  # 뒷면 노멀 맵
                    current_depth_map=current_depth_map,  # 현재 깊이 맵
                    human_parse_map=human_parse_map  # 인간 파싱 맵
                )
                optimizerG.zero_grad()  # 옵티마이저의 그래디언트 초기화
                error_low_res_pifu['Err(occ)'].backward()  # 오차에 대해 역전파 수행하여 그래디언트 계산
                curr_low_res_loss = error_low_res_pifu['Err(occ)'].item()  # 현재 에포크의 저해상도 손실 값 저장
                optimizerG.step()  # 옵티마이저를 사용하여 네트워크 파라미터 업데이트

                # 현재 에포크, 손실 값, 학습률 출력
                print(
                    'Name: {0} | Epoch: {1} | error_low_res_pifu: {2:.06f} | LR: {3:.06f} '.format(
                        opt.name, epoch, curr_low_res_loss, lr_G)
                )

                r = res_low_res_pifu  # 예측 결과 저장

        # 학습률 조정
        lr_G = adjust_learning_rate(optimizerG, epoch, lr_G, opt.schedule, opt.learning_rate_decay)  # 기본 옵티마이저의 학습률 조정
        if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                lr_highRes = adjust_learning_rate(optimizer_lowResFineTune, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)  # Low-Res Fine-Tuning 옵티마이저의 학습률 조정
            else:
                lr_highRes = adjust_learning_rate(optimizer_highRes, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)  # High-Res Integrator 옵티마이저의 학습률 조정

        with torch.no_grad():  # 그래디언트 계산 비활성화
            if True:  # 항상 실행됨
                # 모델 상태 저장
                with open('%s/%s/netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)), 'wb') as handle:
                    pickle.dump(netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)  # 기본 네트워크의 상태 저장

                with open('%s/%s/optimizerG_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)), 'wb') as handle:
                    pickle.dump(optimizerG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)  # 기본 옵티마이저의 상태 저장

                if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
                    with open('%s/%s/highRes_netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)), 'wb') as handle:
                        pickle.dump(highRes_netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)  # 고해상도 네트워크의 상태 저장

                    with open('%s/%s/optimizer_highRes_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)), 'wb') as handle:
                        pickle.dump(optimizer_highRes.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)  # High-Res Integrator 옵티마이저의 상태 저장
                    
                    if opt.update_low_res_pifu:  # Low-Res PIFu를 업데이트하는 경우
                        with open('%s/%s/optimizer_lowResFineTune_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)), 'wb') as handle:
                            pickle.dump(optimizer_lowResFineTune.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)  # Low-Res Fine-Tuning 옵티마이저의 상태 저장

                    highRes_netG.eval()  # 고해상도 네트워크를 평가 모드로 설정

                # 학습 중 메쉬 생성
                print('generate mesh (train) ...')
                train_dataset.is_train = False  # 데이터셋을 학습 모드에서 테스트 모드로 전환
                netG.eval()  # 기본 네트워크를 평가 모드로 설정
                for gen_idx in tqdm(range(1)):  # 단일 메쉬 생성 반복 (진행 상황 표시)
                    index_to_use = gen_test_counter % len(train_dataset)  # 사용할 데이터 인덱스 결정
                    gen_test_counter += 10  # 테스트 카운터 증가
                    train_data = train_dataset.get_item(index=index_to_use)  # 데이터셋에서 항목 가져오기
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (opt.results_path, opt.name, epoch, train_data['name'])  # 저장 경로 설정

                    if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
                        gen_mesh(
                            resolution=opt.resolution, 
                            net=[netG, highRes_netG], 
                            device=device, 
                            data=train_data, 
                            save_path=save_path
                        )  # 메쉬 생성 및 저장
                    else:
                        gen_mesh(
                            resolution=opt.resolution, 
                            net=netG, 
                            device=device, 
                            data=train_data, 
                            save_path=save_path
                        )  # 메쉬 생성 및 저장

                try:
                    # 모델 성능 시각화 저장
                    save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)  # 저장 경로 설정
                    r = r[0].cpu()  # 예측 결과를 CPU로 이동
                    points = samples_low_res_pifu_tensor[0].transpose(0, 1).cpu()  # 샘플 포인트를 CPU로 이동하고 전치
                    save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())  # 포인트 클라우드 저장
                except:
                    print("Unable to save point cloud.")  # 저장 실패 시 메시지 출력

                train_dataset.is_train = True  # 데이터셋을 다시 학습 모드로 전환

            # 검증 데이터셋 사용 시
            if opt.useValidationSet:
                import trimesh  # 3D 메쉬 처리 라이브러리 임포트
                from evaluate_model import quick_get_chamfer_and_surface_dist  # 평가 함수 임포트
                num_samples_to_use = 5000  # 평가에 사용할 샘플 수 설정

                print('Commencing validation..')  # 검증 시작 메시지 출력
                print('generate mesh (validation) ...')  # 검증 메쉬 생성 메시지 출력

                netG.eval()  # 기본 네트워크를 평가 모드로 설정
                if opt.use_High_Res_Component:
                    highRes_netG.eval()  # 고해상도 네트워크를 평가 모드로 설정
                val_len = len(validation_dataset)  # 검증 데이터셋의 전체 길이
                num_of_val_subjects = val_len // 10  # 검증할 주제 수 계산
                val_mesh_paths = []  # 검증 메쉬 경로 리스트 초기화
                index_to_use_list = []  # 사용할 인덱스 리스트 초기화
                num_of_validation_subjects_to_use = 10  # 사용할 검증 주제 수 설정
                for gen_idx in tqdm(range(num_of_validation_subjects_to_use)):  # 검증 주제 수만큼 반복 (진행 상황 표시)
                    print('[Validation] generating mesh #{0}'.format(gen_idx))  # 현재 검증 메쉬 생성 메시지 출력
                    index_to_use = np.random.randint(low=0, high=num_of_val_subjects)  # 무작위로 검증할 인덱스 선택
                    while index_to_use in index_to_use_list:  # 중복 인덱스 방지
                        print('repeated index_to_use is selected, re-sampling')  # 중복 선택 시 재샘플링 메시지 출력
                        index_to_use = np.random.randint(low=0, high=num_of_val_subjects)
                    index_to_use_list.append(index_to_use)  # 선택한 인덱스를 리스트에 추가
                    val_data = validation_dataset.get_item(index=index_to_use * 10)  # 검증 데이터셋에서 항목 가져오기

                    save_path = '%s/%s/val_eval_epoch%d_%s.obj' % (opt.results_path, opt.name, epoch, val_data['name'])  # 메쉬 저장 경로 설정

                    val_mesh_paths.append(save_path)  # 저장 경로를 리스트에 추가

                    if opt.use_High_Res_Component:  # 고해상도 컴포넌트 사용 시
                        gen_mesh(
                            resolution=opt.resolution, 
                            net=[netG, highRes_netG], 
                            device=device, 
                            data=val_data, 
                            save_path=save_path
                        )  # 메쉬 생성 및 저장
                    else:
                        gen_mesh(
                            resolution=opt.resolution, 
                            net=netG, 
                            device=device, 
                            data=val_data, 
                            save_path=save_path
                        )  # 메쉬 생성 및 저장

                total_chamfer_distance = []  # Chamfer 거리 리스트 초기화
                total_point_to_surface_distance = []  # Point-to-Surface 거리 리스트 초기화
                for val_path in val_mesh_paths:  # 검증 메쉬 경로 리스트를 순회
                    subject = val_path.split('_')[-1]  # 파일명에서 주제 이름 추출
                    subject = subject.replace('.obj', '')  # 확장자 제거
                    GT_mesh = validation_dataset.mesh_dic[subject]  # Ground Truth 메쉬 로드

                    try: 
                        print('Computing CD and P2S for {0}'.format(os.path.basename(val_path)))  # Chamfer 거리 및 Point-to-Surface 거리 계산 메시지 출력
                        source_mesh = trimesh.load(val_path)  # 생성된 메쉬 로드
                        chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(
                            src_mesh=source_mesh, 
                            tgt_mesh=GT_mesh, 
                            num_samples=num_samples_to_use
                        )  # Chamfer 거리 및 Point-to-Surface 거리 계산
                        total_chamfer_distance.append(chamfer_distance)  # Chamfer 거리 리스트에 추가
                        total_point_to_surface_distance.append(point_to_surface_distance)  # Point-to-Surface 거리 리스트에 추가
                    except:
                        print('Unable to compute chamfer_distance and/or point_to_surface_distance!')  # 계산 실패 시 메시지 출력
                
                # 평균 Chamfer 거리 계산
                if len(total_chamfer_distance) == 0:
                    average_chamfer_distance = 0
                else:
                    average_chamfer_distance = np.mean(total_chamfer_distance)

                # 평균 Point-to-Surface 거리 계산
                if len(total_point_to_surface_distance) == 0:
                    average_point_to_surface_distance = 0 
                else:
                    average_point_to_surface_distance = np.mean(total_point_to_surface_distance)

                # 검증 에포크별 거리 리스트에 추가
                validation_epoch_cd_dist_list.append(average_chamfer_distance)
                validation_epoch_p2s_dist_list.append(average_point_to_surface_distance)

                # 검증 결과 출력
                print("[Validation] Overall Epoch {0}- Avg CD: {1}; Avg P2S: {2}".format(epoch, average_chamfer_distance, average_point_to_surface_distance))

                # 검증을 위해 생성된 파일 삭제
                for file_path in val_mesh_paths:
                    mesh_path = file_path
                    image_path = file_path.replace('.obj', '.png')
                    os.remove(mesh_path)  # 메쉬 파일 삭제
                    os.remove(image_path)  # 이미지 파일 삭제

                # 검증 그래프 저장
                plt.plot(np.arange(epoch + 1), np.array(validation_epoch_cd_dist_list))  # 에포크별 Chamfer 거리 플롯
                plt.plot(np.arange(epoch + 1), np.array(validation_epoch_p2s_dist_list), '-.')  # 에포크별 Point-to-Surface 거리 플롯
                plt.xlabel('Epoch')  # x축 레이블 설정
                plt.ylabel('Validation Error (CD + P2D)')  # y축 레이블 설정
                plt.title('Epoch Against Validation Error (CD + P2D)')  # 그래프 제목 설정
                plt.savefig(validation_graph_path)  # 그래프 이미지로 저장

# 메인 함수
if __name__ == '__main__':
    train(opt)