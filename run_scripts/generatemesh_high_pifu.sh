# train_integratedPIFu에서 high-pifu 옵션 import
# load_model_weights = True
# load_model_weights_for_high_res_too = True
# modelG_path = 로드할 low-pifu 가중치 경로
# modelhighResG_path = 로드할 high-pifu 가중치 경로
# test_script_activate = True

NAME='high_pifu_generate_1'
SERIAL_BATCHES='--serial_batches'  # 랜덤 배치 옵션
BATCH_SIZE='1'

# 훈련 명령어 실행
python apps/train_integratedPIFu.py ${SERIAL_BATCHES} \
  --name ${NAME} \
  --batch_size ${BATCH_SIZE}