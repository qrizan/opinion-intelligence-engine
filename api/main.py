# FastAPI + Gradio server untuk Opinion Intelligence Engine.

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gradio as gr
from pydantic import BaseModel
from typing import Optional

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# constants
MAX_TEXT_LENGTH = 100000  # maksimal 100k karakter
MIN_TEXT_LENGTH = 10  # minimal 10 karakter

# import pipeline functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline import OpinionIntelligencePipeline

# initialize FastAPI
app = FastAPI(
    title="Opinion Intelligence Engine",
    description="API untuk analisis sentiment dan summarization teks panjang",
    version="1.0.0"
)

# CORS configuration - lebih aman dengan environment variable
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:8000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# initialize pipeline (lazy load)
pipeline = None

def get_pipeline():
    # lazy load pipeline saat pertama kali digunakan.
    global pipeline
    if pipeline is None:
        pipeline = OpinionIntelligencePipeline()
    return pipeline

# health check endpoint
@app.get("/health")
async def health_check():
    # health check untuk monitoring.
    return {"status": "healthy", "service": "opinion-intelligence-engine"}

# fastAPI endpoint untuk sentiment + summarization
class TextInput(BaseModel):
    text: str
    max_excerpts: Optional[int] = 3

@app.post("/analyze")
async def analyze_text(input_data: TextInput):
    """
    Analisis sentiment dan summarization untuk teks panjang.
    
    Args:
        input_data: TextInput dengan field 'text' dan optional 'max_excerpts'
    
    Returns:
        JSON dengan sentiment, confidence, excerpts, dan summaries
    
    Raises:
        HTTPException: Jika input tidak valid atau terjadi error
    """

    # validasi input
    text = input_data.text.strip() if input_data.text else ""
    
    if not text:
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong")
    
    if len(text) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"Text terlalu pendek (minimal {MIN_TEXT_LENGTH} karakter)"
        )
    
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text terlalu panjang (maksimal {MAX_TEXT_LENGTH:,} karakter)"
        )
    
    if input_data.max_excerpts and (input_data.max_excerpts < 1 or input_data.max_excerpts > 5):
        raise HTTPException(status_code=400, detail="max_excerpts harus antara 1-5")
    
    try:
        logger.info(f"Processing text analysis request (length: {len(text)} chars)")
        pipe = get_pipeline()
        result = pipe.process(text, max_excerpts=input_data.max_excerpts or 3)
        logger.info("Text analysis completed successfully")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat memproses teks. Silakan coba lagi."
        )

# gradio interface - versi lebih informatif untuk user awam
def gradio_interface(text: str, max_excerpts: int = 3, progress=gr.Progress()):
    """
    Gradio interface dengan label bahasa Indonesia yang jelas dan output lebih informatif.
    
    Args:
        text: Teks yang akan dianalisis
        max_excerpts: Jumlah bagian teks penting yang akan ditampilkan (1-5)
        progress: Gradio progress tracker untuk loading indicator
    
    Returns:
        String HTML dengan hasil analisis atau pesan error
    """
    
    # Validasi input
    text = text.strip() if text else ""
    
    if not text:
        return "## Error\n\n**Teks tidak boleh kosong. Silakan masukkan teks untuk dianalisis.**"
    
    if len(text) < MIN_TEXT_LENGTH:
        return f"## Error\n\n**Teks terlalu pendek. Minimal {MIN_TEXT_LENGTH} karakter.**"
    
    if len(text) > MAX_TEXT_LENGTH:
        return f"## Error\n\n**Teks terlalu panjang. Maksimal {MAX_TEXT_LENGTH:,} karakter.**"
    
    if max_excerpts < 1 or max_excerpts > 5:
        return "## Error\n\n**Jumlah bagian teks penting harus antara 1-5.**"
    
    try:
        # Progress indicator
        progress(0.1, desc="Memuat model...")
        logger.info(f"Processing Gradio request (length: {len(text)} chars, max_excerpts: {max_excerpts})")
        pipe = get_pipeline()
        logger.info("Pipeline loaded, starting text processing...")
        
        progress(0.3, desc="Menganalisis teks (ini mungkin memakan waktu beberapa menit)...")
        logger.info("Calling pipe.process()...")
        try:
            result = pipe.process(text, max_excerpts=max_excerpts)
            logger.info("pipe.process() completed successfully")
        except Exception as process_error:
            logger.error(f"Error in pipe.process(): {str(process_error)}", exc_info=True)
            raise
        
        progress(0.8, desc="Menyusun hasil...")
        logger.info("Gradio analysis completed successfully")
        
        # format output lebih informatif untuk user awam
        sentiment_label = 'Positif' if result['document_sentiment'] == 1 else 'Negatif'
        sentiment_color = '#4caf50' if result['document_sentiment'] == 1 else '#ff5722'  # hijau untuk positif, orange untuk negatif
        
        chunk_stats = result['chunk_statistics']
        total_chunks = chunk_stats['total_chunks']
        positive_pct = (chunk_stats['positive_chunks'] / total_chunks * 100) if total_chunks > 0 else 0
        negative_pct = (chunk_stats['negative_chunks'] / total_chunks * 100) if total_chunks > 0 else 0
        
        # Confidence level description untuk user awam
        confidence_pct = result['document_confidence'] * 100
        if confidence_pct >= 80:
            confidence_desc = "Sangat Yakin"
        elif confidence_pct >= 60:
            confidence_desc = "Cukup Yakin"
        elif confidence_pct >= 40:
            confidence_desc = "Agak Yakin"
        else:
            confidence_desc = "Kurang Yakin"
        
        # output dengan penjelasan lebih jelas untuk user awam
        summary_text = f"""
<div style="padding: 0px 20px; border-radius: 8px; margin-bottom: 20px;">
<h2 style="margin: 0; font-size: 24px;">Hasil Analisis Sentimen</h2>
</div>

<div style="padding: 20px; border-radius: 8px; border-left: 4px solid {sentiment_color}; margin-bottom: 20px;">
<h3 style="margin-top: 0; color: {sentiment_color}; font-size: 20px;">
Sentimen Dokumen: <strong>{sentiment_label}</strong>
</h3>
<p style="font-size: 16px; margin: 10px 0;">
<strong>Tingkat Keyakinan:</strong> {result['document_confidence']:.1%} ({confidence_desc})
</p>
<p style="font-size: 14px; margin: 5px 0;">
Model menganalisis seluruh teks dan menentukan bahwa sentimen keseluruhan adalah <strong>{sentiment_label.lower()}</strong> dengan tingkat keyakinan {result['document_confidence']:.1%}. Semakin tinggi persentase, semakin yakin model dengan hasil analisisnya.
</p>
</div>

<div style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
<h3 style="margin-top: 0; font-size: 18px;">Informasi Teks</h3>
<ul style="line-height: 1.8;">
<li><strong>Panjang Teks:</strong> {result['text_length']:,} karakter</li>
<li><strong>Jumlah Bagian yang Dianalisis:</strong> {total_chunks} bagian</li>
</ul>
<p style="font-size: 14px; margin-top: 10px;">
Teks panjang dibagi menjadi {total_chunks} bagian untuk dianalisis secara lebih akurat. Setiap bagian dianalisis terpisah, kemudian hasilnya digabungkan untuk mendapatkan sentimen keseluruhan.
</p>
</div>

<div style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
<h3 style="margin-top: 0; font-size: 18px;">Distribusi Sentimen per Bagian</h3>
<div style="margin: 15px 0;">
<div style="display: flex; align-items: center; margin-bottom: 10px;">
<span style="display: inline-block; width: 20px; height: 20px; background: #4caf50; border-radius: 4px; margin-right: 10px;"></span>
<strong>Bagian Positif:</strong> {chunk_stats['positive_chunks']} bagian ({positive_pct:.1f}%)
</div>
<div style="display: flex; align-items: center;">
<span style="display: inline-block; width: 20px; height: 20px; background: #ff5722; border-radius: 4px; margin-right: 10px;"></span>
<strong>Bagian Negatif:</strong> {chunk_stats['negative_chunks']} bagian ({negative_pct:.1f}%)
</div>
</div>
<p style="font-size: 14px; margin-top: 10px;">
Dari {total_chunks} bagian teks yang dianalisis, {chunk_stats['positive_chunks']} bagian terdeteksi sebagai positif dan {chunk_stats['negative_chunks']} bagian terdeteksi sebagai negatif. Ini membantu memahami bagaimana sentimen tersebar di seluruh teks.
</p>
</div>

<div style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
<h3 style="margin-top: 0; font-size: 18px;">Bagian Teks Penting & Ringkasan</h3>
<p style="font-size: 14px; margin-bottom: 15px;">
Berikut adalah bagian-bagian teks yang paling mewakili sentimen dokumen. Bagian-bagian ini dipilih karena memiliki tingkat keyakinan tertinggi dalam analisis. Setiap bagian juga telah dirangkum untuk memudahkan pemahaman.
</p>
"""
        
        for i, (excerpt, summary) in enumerate(zip(result['key_excerpts'], result['summaries']), 1):
            excerpt_label = 'Positif' if excerpt['label'] == 1 else 'Negatif'
            excerpt_color = '#4caf50' if excerpt['label'] == 1 else '#ff5722'
            
            # Confidence badge
            excerpt_confidence_pct = excerpt['confidence'] * 100
            if excerpt_confidence_pct >= 80:
                confidence_badge = "Sangat Yakin"
            elif excerpt_confidence_pct >= 60:
                confidence_badge = "Cukup Yakin"
            else:
                confidence_badge = "Agak Yakin"
            
            summary_text += f"""
<div style="padding: 20px; border-radius: 8px; border-left: 4px solid {excerpt_color}; margin-bottom: 20px;">
<h4 style="margin-top: 0; color: {excerpt_color}; font-size: 16px;">
Bagian {i} - {excerpt_label} 
<span style="font-size: 12px; font-weight: normal;">(Keyakinan: {excerpt['confidence']:.1%} - {confidence_badge})</span>
</h4>

<div style="padding: 12px; border-radius: 6px; margin: 15px 0;">
<strong>Teks Asli:</strong>
<p style="margin: 8px 0; font-style: italic; line-height: 1.6;">
"{excerpt['text'][:300]}{'...' if len(excerpt['text']) > 300 else ''}"
</p>
</div>

<div style="padding: 12px; border-radius: 6px; border-left: 3px solid #999;">
<strong>Ringkasan:</strong>
<p style="margin: 8px 0; line-height: 1.6;">
{summary}
</p>
</div>
</div>
"""
        
        progress(1.0, desc="Selesai!")
        return summary_text
    except Exception as e:
        logger.error(f"Error in Gradio interface: {str(e)}", exc_info=True)
        return "## Terjadi Kesalahan\n\n**Terjadi kesalahan saat memproses teks. Silakan coba lagi atau pastikan teks yang dimasukkan valid.**"

# Theme default Gradio - simple dan clean
with gr.Blocks(
    title="Opinion Intelligence Engine"
) as gradio_app:
    
    # Header simple
    gr.Markdown("# Opinion Intelligence Engine")
    gr.Markdown("Analisis sentimen dan ringkasan untuk teks panjang")
        
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Masukkan Teks (Bahasa Inggris)",
                placeholder="Masukkan teks panjang untuk analisis sentimen dan ringkasan...\n\nContoh: Review produk, artikel, komentar, dll.",
                lines=15,
                show_label=True,
                info="Teks harus dalam bahasa Inggris. Minimal 10 karakter, maksimal 100,000 karakter."
            )
            max_excerpts = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Jumlah Bagian Teks Penting",
                info="- Berapa banyak bagian teks dengan keyakinan tertinggi yang akan ditampilkan (1-5).\n- Semakin banyak, semakin detail hasilnya."
            )
            with gr.Row():
                submit_btn = gr.Button("Analisis", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("Hapus", variant="secondary", size="lg", scale=1)
        
        with gr.Column(scale=3):
            output = gr.HTML(
                label="Hasil Analisis",
                value="""
                <div style="text-align: center; padding: 40px; color: #666;">
                    <p style="font-size: 18px; margin-bottom: 10px;">Hasil analisis akan muncul di sini</p>
                    <p style="font-size: 14px;">Masukkan teks di sebelah kiri dan klik tombol "Analisis" untuk memulai</p>
                </div>
                """
            )
    
    # Event handlers dengan loading indicator
    submit_btn.click(
        fn=gradio_interface,
        inputs=[text_input, max_excerpts],
        outputs=output,
        show_progress="full"  # Menampilkan progress bar
    )
    
    clear_btn.click(
        fn=lambda: (
            "", 
            3, 
            """
            <div style="text-align: center; padding: 40px; color: #666;">
                <p style="font-size: 18px; margin-bottom: 10px;">Hasil analisis akan muncul di sini</p>
                <p style="font-size: 14px;">Masukkan teks di sebelah kiri dan klik tombol "Analisis" untuk memulai</p>
            </div>
            """
        ),
        outputs=[text_input, max_excerpts, output]
    )

# mount Gradio ke FastAPI
# untuk Gradio 4.0+, gunakan mount_gradio_app
try:
    # API Gradio 4.0+
    if hasattr(gr, 'mount_gradio_app'):
        app = gr.mount_gradio_app(app, gradio_app, path="/")
    else:
        # fallback untuk versi lama - mount sebagai ASGI app
        app.mount("/", gradio_app)
except Exception as e:
    logger.warning(f"Error mounting Gradio app: {e}")
    logger.warning("Gradio akan tetap berjalan, tapi mungkin perlu akses terpisah")

# run server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
