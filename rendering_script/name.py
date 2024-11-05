import os

"""
디렉토리 안에 있는 디렉토리명을 뽑아서 출력하는 스크립트
"""

# 디렉토리 경로 설정
directory_path = '/home/public/IntegratedPIFu/rendering_script/buffer_fixed_full_mesh'

# 해당 경로에서 디렉토리 이름만 추출
dir_names = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

# 디렉토리 이름 출력
for dir_name in dir_names:
    print(dir_name)

with open('directory_names.txt', 'w') as f:
     for dir_name in dir_names:
         f.write(dir_name + '\n')