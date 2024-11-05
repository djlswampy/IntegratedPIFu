NAME='first_normal_test'
NUM_EPOCH='2'
SERIAL_BATCHES='--serial_batches'  # 랜덤 배치 옵션
PIN_MEMORY='--pin_memory'  # 메모리 고정 옵션
NUM_THREADS='8'  # 스레드 수

# 훈련 명령어 실행
python apps/train_normalmodel.py ${SERIAL_BATCHES} ${PIN_MEMORY} \
  --name ${NAME} \
  --num_epoch ${NUM_EPOCH} \
  --num_threads ${NUM_THREADS}