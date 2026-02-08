"""
script untuk download model dari Google Drive.

fitur:
- skip download jika model sudah ada
- progress bar saat download
- verifikasi file setelah download
"""

import os
import sys
from pathlib import Path
import gdown

# konfigurasi
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
SENTIMENT_DIR = MODELS_DIR / "sentiment"
SUMMARIZATION_DIR = MODELS_DIR / "summarization"

# load .env file jika ada
def load_env_file(env_path: Path = None):
    """Load environment variables dari .env file."""
    if env_path is None:
        env_path = BASE_DIR / ".env"
    
    if not env_path.exists():
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # skip comment dan empty line
            if not line or line.startswith('#'):
                continue
            # parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # hanya set jika belum ada di environment
                if key and value and key not in os.environ:
                    os.environ[key] = value

# load .env file
load_env_file()

# google drive folder IDs - WAJIB dari environment variable (.env)
# untuk keamanan, folder IDs TIDAK di-hardcode di kode
# set di .env file sebelum menjalankan script ini
SENTIMENT_FOLDER_ID = os.getenv("SENTIMENT_FOLDER_ID")
SUMMARIZATION_FOLDER_ID = os.getenv("SUMMARIZATION_FOLDER_ID")

# validasi: pastikan folder IDs sudah di-set
if not SENTIMENT_FOLDER_ID or not SUMMARIZATION_FOLDER_ID:
    print("="*60)
    print("ERROR: Folder IDs belum di-set!")
    print("="*60)
    print("\nUntuk download model, Anda HARUS set folder IDs di .env file:")
    print("\n  1. Copy .env.example menjadi .env:")
    print("     cp .env.example .env")
    print("\n  2. Edit .env dan uncomment + isi folder IDs:")
    print("     SENTIMENT_FOLDER_ID=your-sentiment-folder-id")
    print("     SUMMARIZATION_FOLDER_ID=your-summarization-folder-id")
    print("\n  3. Jalankan script lagi:")
    print("     python3 download_models.py")
    print("\n" + "="*60)
    sys.exit(1)

# file penting yang harus ada setelah download
REQUIRED_SENTIMENT_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    # vocab.txt tidak wajib untuk tokenizer modern (DistilBERT menggunakan tokenizer.json)
]

REQUIRED_SUMMARIZATION_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "spiece.model"  #  sentencepiece digunakan untuk t5
]

# file model (bisa safetensors atau bin)
MODEL_FILES = ["model.safetensors", "pytorch_model.bin"]


def check_model_exists(model_dir: Path, required_files: list) -> bool:
    # cek apakah folder model ada
    if not model_dir.exists():
        return False
    
    # cek file penting
    for file in required_files:
        if not (model_dir / file).exists():
            return False
    
    # cek file model (minimal salah satu: safetensors atau bin)
    has_model_file = any((model_dir / f).exists() for f in MODEL_FILES)
    if not has_model_file:
        return False
    
    return True


def verify_download(model_dir: Path, required_files: list) -> bool:
    # verifikasi model di folder
    print(f"\n{'='*60}")
    print(f"Verifikasi model di: {model_dir}")
    print(f"{'='*60}")
    
    all_files = list(model_dir.glob("*"))
    print(f"Total file ditemukan: {len(all_files)}")
    
    # cek file penting
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  [OK] {file} ({size:.2f} MB)")
        else:
            print(f"  [MISSING] {file}")
            missing_files.append(file)
    
    # cek file model
    has_model = False
    for model_file in MODEL_FILES:
        file_path = model_dir / model_file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  [OK] {model_file} ({size:.2f} MB)")
            has_model = True
            break
    
    if not has_model:
        print(f"  [MISSING] Model file (perlu {MODEL_FILES})")
        missing_files.append("model_file")
    
    # list semua file
    print(f"\nSemua file di folder:")
    for f in sorted(all_files):
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {f.name} ({size:.2f} MB)")
    
    # validasi fleksibel: vocab.txt tidak wajib jika tokenizer.json ada
    if "vocab.txt" in missing_files and (model_dir / "tokenizer.json").exists():
        print(f"\n[INFO] Catatan: vocab.txt tidak ditemukan, tapi tokenizer.json ada (OK untuk tokenizer modern)")
        missing_files.remove("vocab.txt")
    
    # validasi fleksibel: spiece.model tidak wajib jika tokenizer.json ada (untuk T5)
    if "spiece.model" in missing_files and (model_dir / "tokenizer.json").exists():
        print(f"\n[INFO] Catatan: spiece.model tidak ditemukan, tapi tokenizer.json ada (OK untuk tokenizer modern T5)")
        missing_files.remove("spiece.model")
    
    if missing_files:
        print(f"\n[WARNING] PERINGATAN: File berikut tidak ditemukan:")
        for f in missing_files:
            print(f"    - {f}")
        return False
    
    print(f"\n[OK] Verifikasi berhasil! Model lengkap.")
    return True


def download_model(folder_id: str, output_dir: Path, model_name: str, required_files: list):

    print(f"\n{'='*60}")
    print(f"Download Model: {model_name}")
    print(f"{'='*60}")
    
    # cek apakah model sudah ada
    if check_model_exists(output_dir, required_files):
        print(f"[OK] Model {model_name} sudah ada di {output_dir}")
        print("  Skip download.")
        
        # verifikasi ulang
        if verify_download(output_dir, required_files):
            return
        else:
            print("Model tidak lengkap, akan download ulang...")
    
    # buat folder jika belum ada
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URL google drive
    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    
    print(f"Download dari: {url}")
    print(f"Tujuan: {output_dir}")
    print("\nMulai download... (ini mungkin memakan waktu beberapa menit)")
    
    try:
        # download folder
        # gdown akan otomatis menampilkan progress bar
        gdown.download_folder(
            url,
            output=str(output_dir),
            quiet=False,  # tampilkan progress bar
            use_cookies=False
        )
        
        print(f"\nDownload selesai!")
        
        # Verifikasi
        if not verify_download(output_dir, required_files):
            print(f"\nERROR: Model {model_name} tidak lengkap setelah download!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR saat download {model_name}:")
        print(f"   {str(e)}")
        print(f"\nTips:")
        print(f"   1. pastikan link Google Drive sudah di-share dengan akses 'Anyone with the link'")
        print(f"   2. cek koneksi internet")
        print(f"   3. coba jalankan script lagi")
        sys.exit(1)


def main():
    print("="*60)
    print("OPINION INTELLIGENCE ENGINE - MODEL DOWNLOADER")
    print("="*60)
    
    # download sentiment model
    download_model(
        folder_id=SENTIMENT_FOLDER_ID,
        output_dir=SENTIMENT_DIR,
        model_name="Sentiment Classification",
        required_files=REQUIRED_SENTIMENT_FILES
    )
    
    # download summarization model
    download_model(
        folder_id=SUMMARIZATION_FOLDER_ID,
        output_dir=SUMMARIZATION_DIR,
        model_name="Summarization",
        required_files=REQUIRED_SUMMARIZATION_FILES
    )
    
    print(f"\n{'='*60}")
    print("[OK] SEMUA MODEL BERHASIL DIDOWNLOAD!")
    print(f"{'='*60}")
    print(f"\nModel tersimpan di:")
    print(f"  - Sentiment: {SENTIMENT_DIR}")
    print(f"  - Summarization: {SUMMARIZATION_DIR}")


if __name__ == "__main__":
    main()