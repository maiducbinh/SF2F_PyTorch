"""
Chuyển đổi các file audio thô trong dataset sang mel spectrogram.

Cấu trúc folder của dataset:
    data/
      ├── <tên người 1>/
      │      ├── audios/          # chứa file audio (.wav hoặc .m4a)
      │      └── masked_faces/    # chứa file ảnh (.jpg)
      ├── <tên người 2>/
      │      ├── audios/
      │      └── masked_faces/
      └── ...

Logic:
  1. Duyệt qua các folder con trong folder data, mỗi folder ứng với một người.
  2. Trong mỗi folder người, đọc các file audio từ folder "audios".
  3. Với mỗi file audio, chuyển đổi sang mel spectrogram sử dụng hàm wav_to_mel,
     lưu kết quả (cùng với thông tin meta) vào file pickle trong folder output, được tổ chức theo tên người.
  4. Hỗ trợ chạy song song với nhiều luồng.
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import gc
import pickle
from tensorflow.io import gfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append('./')
from utils.wav2mel import wav_to_mel

class WavConvertor:
    def __init__(self, data_dir, output_dir):
        """
        Args:
            data_dir (str): Đường dẫn đến folder chứa dataset (cấu trúc theo mô tả).
            output_dir (str): Đường dẫn đến folder lưu kết quả (mel spectrograms).
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.get_person_dirs()
        self.create_output_dir()

    def create_output_dir(self):
        if not gfile.exists(self.output_dir):
            gfile.mkdir(self.output_dir)

    def get_person_dirs(self):
        """
        Duyệt qua các folder con trong data_dir, chọn những folder có chứa folder "audios".
        """
        self.person_dirs = []
        for name in os.listdir(self.data_dir):
            person_path = os.path.join(self.data_dir, name)
            audios_path = os.path.join(person_path, "audios")
            if os.path.isdir(person_path) and gfile.exists(audios_path):
                self.person_dirs.append(person_path)
        if not self.person_dirs:
            print("Không tìm thấy folder nào có chứa 'audios' trong dataset.")

    def convert_person(self, person_dir):
        """
        Chuyển đổi các file audio của một người sang mel spectrogram và lưu vào folder output.
        """
        # Lấy tên người từ tên folder
        person_name = os.path.basename(person_dir.rstrip("/"))
        audios_dir = os.path.join(person_dir, "audios")
        if not gfile.exists(audios_dir):
            print(f"Không tồn tại folder 'audios' trong {person_dir}. Bỏ qua.")
            return

        # Tạo folder output cho người này (nếu chưa có)
        out_person_dir = os.path.join(self.output_dir, person_name)
        if not gfile.exists(out_person_dir):
            gfile.mkdir(out_person_dir)

        audio_files = os.listdir(audios_dir)
        for audio_file in audio_files:
            if audio_file.lower().endswith(('.wav', '.m4a')):
                audio_path = os.path.join(audios_dir, audio_file)
                try:
                    base_audio_name = os.path.splitext(audio_file)[0]
                    # Đặt tên file pickle theo định dạng: <tên người>_<tên file audio>.pickle
                    pickle_name = f"{person_name}_{base_audio_name}.pickle"
                    pickle_path = os.path.join(out_person_dir, pickle_name)
                    if os.path.exists(pickle_path):
                        # Nếu file pickle đã tồn tại thì bỏ qua
                        continue
                    # Chuyển đổi wav sang mel spectrogram
                    log_mel = wav_to_mel(audio_path)
                    pickle_dict = {
                        'LogMel_Features': log_mel,
                        'person_name': person_name,
                        'audio_file': audio_file
                    }
                    with open(pickle_path, "wb") as f:
                        pickle.dump(pickle_dict, f)
                except Exception as e:
                    print(f"Lỗi khi xử lý file {audio_path}: {e}")
        gc.collect()

    def _worker(self, job_id, person_list):
        """
        Hàm xử lý cho một luồng: chuyển đổi các folder người trong list được chỉ định.
        """
        for i, person_dir in enumerate(person_list):
            self.convert_person(person_dir)
            print(f"Job #{job_id}: đã xử lý {i+1}/{len(person_list)} - {person_dir}")

    def convert_all(self, n_jobs=1):
        """
        Chuyển đổi toàn bộ dữ liệu với số luồng song song n_jobs.
        """
        n_persons = len(self.person_dirs)
        if n_persons == 0:
            print("Không có folder người hợp lệ để xử lý.")
            return

        n_jobs = min(n_jobs, n_persons)
        persons_per_job = n_persons // n_jobs
        process_index = []
        for ii in range(n_jobs):
            start = ii * persons_per_job
            # Luồng cuối cùng nhận phần dư
            end = (ii + 1) * persons_per_job if ii < n_jobs - 1 else n_persons
            process_index.append((start, end))

        futures = set()
        with ProcessPoolExecutor() as executor:
            for job_id, (start, end) in enumerate(process_index):
                person_subset = self.person_dirs[start:end]
                future = executor.submit(self._worker, job_id, person_subset)
                futures.add(future)
                print(f"Đã submit job {job_id} với danh sách từ index {start} đến {end}")
            for future in as_completed(futures):
                # Chờ tất cả job hoàn thành
                pass

        print("Xử lý hoàn tất.")


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi file audio sang mel spectrogram")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Đường dẫn đến folder chứa dataset (mỗi folder con là tên người có chứa 'audios')")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Đường dẫn đến folder lưu kết quả mel spectrograms")
    parser.add_argument('--n_jobs', type=int, default=1,
                        help="Số luồng xử lý song song (mặc định: 1)")
    args = parser.parse_args()

    converter = WavConvertor(args.data_dir, args.output_dir)
    converter.convert_all(args.n_jobs)


if __name__ == '__main__':
    main()
