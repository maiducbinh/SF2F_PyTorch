import os
import pickle
import argparse
from PIL import Image

def convert_images_to_pickle(source_folder, destination_folder):
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(destination_folder, exist_ok=True)
    
    for person in os.listdir(source_folder):
        person_folder = os.path.join(source_folder, person)
        
        if os.path.isdir(person_folder):
            # Tạo thư mục con cho người
            person_output_folder = os.path.join(destination_folder, person)
            os.makedirs(person_output_folder, exist_ok=True)
            
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Mở ảnh
                    img = Image.open(img_path)
                    img_data = img.convert("RGB")  # Đảm bảo ảnh ở định dạng RGB
                    
                    # Tạo tên file pickle theo định dạng người_tên_file_ảnh.pickle
                    pickle_file_name = f"{person}_{os.path.splitext(img_file)[0]}.pickle"
                    pickle_file_path = os.path.join(person_output_folder, pickle_file_name)
                    
                    # Lưu ảnh dưới dạng pickle
                    with open(pickle_file_path, 'wb') as pickle_file:
                        pickle.dump(img_data, pickle_file)

def main():
    # Thiết lập argparse để xử lý đầu vào từ dòng lệnh
    parser = argparse.ArgumentParser(description="Chuyển đổi ảnh sang định dạng pickle.")
    parser.add_argument('--input_dir', type=str, help='Thư mục chứa ảnh')
    parser.add_argument('--output_dir', type=str, help='Thư mục lưu file pickle')
    
    args = parser.parse_args()
    
    # Gọi hàm chuyển đổi
    convert_images_to_pickle(args.input_dir, args.output_dir)
    print("Chuyển đổi hoàn tất!")

if __name__ == "__main__":
    main()
    #python /home/iec/DucBinh/SF2F_PyTorch/scripts/convert_img_to_pickle.py /home/iec/DucBinh/SF2F_PyTorch/vox_dataset/masked_faces /home/iec/DucBinh/SF2F_PyTorch/example_dataset/masked_faces