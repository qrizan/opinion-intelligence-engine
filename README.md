# Opinion Intelligence Engine

Proyek pembelajaran untuk analisis sentimen dan ringkasan teks panjang menggunakan machine learning.

---

## Features

- Analisis sentimen teks panjang (positif atau negatif)
- Menentukan tingkat keyakinan dari analisis
- Memilih bagian teks penting yang paling mewakili sentimen
- Merangkum bagian teks penting menjadi ringkasan singkat

---

## Limitations

- Karena keterbatasan dataset yang sesuai, projek hanya mendukung bahasa Inggris (input teks harus dalam bahasa Inggris)
- Tidak bisa menganalisis teks yang sangat pendek (< 10 karakter)
- Tidak 100% akurat (seperti semua model ML)
- Tidak bisa memahami sarkasme atau konteks sangat kompleks
- Hanya untuk teks, tidak untuk gambar/audio/video

---

## How It Works

Karena model ML biasanya hanya bisa memproses teks pendek (maksimal 512 kata), proyek ini:

1. Memecah teks panjang menjadi bagian-bagian kecil
2. Menganalisis setiap bagian secara terpisah
3. Menggabungkan hasil dari semua bagian
4. Memilih bagian terpenting berdasarkan keyakinan
5. Merangkum bagian terpenting menjadi ringkasan

---

## Project Structure

```
opinion-intelligence-engine/
├── notebook/          # Notebook untuk training
├── api/               # Server (FastAPI + Gradio)
├── src/               # Pipeline logic
├── models/            # Trained models (not in GitHub)
├── download_models.py # Download models from Google Drive
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Using Docker (Recommended)

```bash
docker-compose up -d
```

Tunggu build selesai (~35-50 menit pertama kali), lalu buka `http://localhost:8000`

### Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_models.py
python api/main.py
```

---

## Usage

### Web Interface (Gradio)

**Catatan Penting:** Interface menggunakan bahasa Indonesia untuk kemudahan penggunaan, namun **input teks harus dalam bahasa Inggris** karena model hanya mendukung bahasa Inggris.

**Cara Menggunakan:**
1. Buka `http://localhost:8000`
2. Masukkan teks panjang dalam **bahasa Inggris**
3. Atur jumlah bagian teks penting (1-5) dengan slider
4. Klik tombol "Analisis"
5. Lihat hasil di panel kanan

**Output yang Ditampilkan:**
- **Sentimen Dokumen:** Positif atau Negatif dengan tingkat keyakinan (0-100%)
- **Informasi Teks:** Panjang teks (karakter) dan jumlah bagian yang dibuat
- **Distribusi Sentimen per Bagian:** Persentase bagian positif vs negatif
- **Bagian Teks Penting:** Bagian teks dengan keyakinan tertinggi (ditampilkan sesuai jumlah yang dipilih)
- **Ringkasan:** Ringkasan singkat untuk setiap bagian teks penting

### API

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "max_excerpts": 3}'
```

---

## Notebooks

- **00_setup_environment.ipynb** - Setup environment
- **01_data_exploration.ipynb** - Eksplorasi dataset
- **02_train_sentiment_model.ipynb** - Training model sentiment
- **03_long_text_sentiment_pipeline.ipynb** - Pipeline untuk teks panjang
- **04_train_summarization_model.ipynb** - Training model summarization
- **05_end_to_end_inference_demo.ipynb** - Demo lengkap

---

## Technologies

- PyTorch
- Transformers (HuggingFace)
- FastAPI
- Gradio
- Docker

**Models:**
- DistilBERT (sentiment analysis)
- T5-base (summarization)

---

## Environment Variables (Optional)

Untuk keamanan dan fleksibilitas, beberapa konfigurasi bisa di-set via environment variables:

- `PORT`: Port untuk server (default: 8000)
- `ALLOWED_ORIGINS`: CORS allowed origins, pisahkan dengan koma (default: http://localhost:8000,http://127.0.0.1:8000)
- `SENTIMENT_FOLDER_ID`: Google Drive folder ID untuk model sentiment (default: sudah di-set)
- `SUMMARIZATION_FOLDER_ID`: Google Drive folder ID untuk model summarization (default: sudah di-set)

**Catatan:** Folder IDs default sudah di-set di kode. Hanya perlu set environment variable jika ingin menggunakan folder berbeda.

---

## Notes

- Proyek ini dibuat untuk **pembelajaran**, bukan untuk production
- Model dilatih dengan dataset terbatas
- Kode dibuat sederhana untuk mudah dipahami
- Tidak ada klaim bahwa ini adalah solusi terbaik atau paling akurat
