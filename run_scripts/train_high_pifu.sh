# 1. train_integratedPIFu에서 high-pifu 옵션 import
# 2. load_model_weights = True
# 3. load_model_weights_for_high_res_too = False
# 4. modelG_path = 로드할 netG 가중치 경로 설정
# 5. test_script_activate = False
# 6. checkpoint_folder_to_load_low_res = 로우 피푸 옵티마이저 체크포인트 경로

NAME='first_high_pifu_train'
NUM_EPOCH='2'
SERIAL_BATCHES='--serial_batches'  # 랜덤 배치 옵션
BATCH_SIZE='2'

# 훈련 명령어 실행
python apps/train_integratedPIFu.py ${SERIAL_BATCHES} \
  --name ${NAME} \
  --num_epoch ${NUM_EPOCH} \
  --batch_size ${BATCH_SIZE}