import os
import shutil

def move_same_files(source_dir1, source_dir2, destination_dir):
    # 두 폴더 내의 파일 리스트 가져오기
    files1 = os.listdir(source_dir1)
    files2 = os.listdir(source_dir2)

    # 두 폴더에 존재하는 동일한 파일 찾기
    same_files = list(set(files1) & set(files2))

    # 이동할 폴더가 존재하지 않으면 생성
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 동일한 파일을 이동할 폴더로 이동
    for file in same_files:
        file_path1 = os.path.join(source_dir1, file)
        file_path2 = os.path.join(source_dir2, file)
        destination_path = os.path.join(destination_dir, file)

        # 파일을 이동
        if os.path.exists(file_path1):
            shutil.copy(file_path1, destination_path)
        elif os.path.exists(file_path2):
            shutil.copy(file_path2, destination_path)

# 예시 폴더 경로 및 이동할 폴더 경로 설정
source_directory_1 = 'D:/201_SampleData/01_Classification_Output\Vit_SGD_CE_MultiLR_torchvision_l_16\miss_classified'
source_directory_2 = 'D:/201_SampleData/01_Classification_Output\weights_finetune\miss_classified'
destination_directory = 'D:/201_SampleData/01_Classification_Output\Common'

# 함수 호출
move_same_files(source_directory_1, source_directory_2, destination_directory)
