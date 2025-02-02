# '''
# This script generates a train/test/val split json for VoxCeleb dataset

# vox1_meta.csv could be download from VoxCeleb official website:
#     https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
# '''


# import os
# import json
# import pandas as pd

# VOX_DIR = os.path.join('./data/VoxCeleb')
# vox1_meta_csv = os.path.join(VOX_DIR, 'vox1', 'vox1_meta.csv')
# vox2_meta_csv = os.path.join(VOX_DIR, 'vox2', 'full_vox2_meta.csv')
# # output:
# split_json = os.path.join(VOX_DIR, 'split.json')


# def main():
#     vox1_df = pd.read_csv(vox1_meta_csv, sep='\t')
#     vox2_df = pd.read_csv(vox2_meta_csv, sep=',')

#     print(vox1_df.head())
#     print(vox2_df.head())

#     split_dict = {
#         'vox1': {'train':[], 'val':[], 'test':[]},
#         'vox2': {'train':[], 'val':[], 'test':[]}
#         }

#     for i in range(len(vox1_df)):
#         name = vox1_df['VGGFace1 ID'].iloc[i]
#         if i % 10 == 8:
#             split_dict['vox1']['test'].append(name)
#         elif i % 10 == 9:
#             split_dict['vox1']['val'].append(name)
#         else:
#             split_dict['vox1']['train'].append(name)

#     for i in range(len(vox2_df)):
#         name = vox2_df['Name'].iloc[i]
#         if i % 10 == 8:
#             split_dict['vox2']['test'].append(name)
#         elif i % 10 == 9:
#             split_dict['vox2']['val'].append(name)
#         else:
#             split_dict['vox2']['train'].append(name)

#     '''
#     for i in range(len(vox1_df)):
#         name = vox1_df['VGGFace1 ID'].iloc[i]
#         set = vox1_df['Set'].iloc[i]
#         if set == 'test':
#             split_dict['vox1']['test'].append(name)
#         if set == 'dev':
#             if i % 10 == 9:
#                 split_dict['vox1']['val'].append(name)
#             else:
#                 split_dict['vox1']['train'].append(name)

#     for i in range(len(vox2_df)):
#         name = vox2_df['Name'].iloc[i]
#         set = vox2_df['Set'].iloc[i]
#         if set == 'test':
#             split_dict['vox2']['test'].append(name)
#         if set == 'dev':
#             if i % 10 == 9:
#                 split_dict['vox2']['val'].append(name)
#             else:
#                 split_dict['vox2']['train'].append(name)
#     '''

#     print(len(split_dict['vox1']['train']),
#           len(split_dict['vox1']['val']),
#           len(split_dict['vox1']['test']))
#     print(len(split_dict['vox2']['train']),
#           len(split_dict['vox2']['val']),
#           len(split_dict['vox2']['test']))

#     with open(split_json, 'w') as outfile:
#         json.dump(split_dict, outfile)

#     return 0


# if __name__ == '__main__':
#     main()

import os
import json

# Đường dẫn tới dataset
DATASET_DIR = './example_dataset'  # Đặt đường dẫn phù hợp với bạn
MASKED_FACES_DIR = os.path.join(DATASET_DIR, 'masked_faces')

# File đầu ra
split_json = os.path.join(DATASET_DIR, 'split.json')

def main():
    # Lấy danh sách người từ thư mục masked_faces
    people = sorted(os.listdir(MASKED_FACES_DIR))  # Đảm bảo danh sách ổn định

    # Tạo dictionary lưu trữ train/val/test
    split_dict = {'train': [], 'val': [], 'test': []}

    for i, person in enumerate(people):
        if i % 10 == 8:
            split_dict['test'].append(person)
        elif i % 10 == 9:
            split_dict['val'].append(person)
        else:
            split_dict['train'].append(person)

    # In ra số lượng mỗi tập
    print(f"Train: {len(split_dict['train'])}, Val: {len(split_dict['val'])}, Test: {len(split_dict['test'])}")

    # Lưu vào file JSON
    with open(split_json, 'w') as outfile:
        json.dump(split_dict, outfile, indent=4)

if __name__ == '__main__':
    main()
