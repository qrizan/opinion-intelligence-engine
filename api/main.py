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
MAX_TEXT_LENGTH = 50000  # Maksimal 50k karakter
MIN_TEXT_LENGTH = 10  # Minimal 10 karakter

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

# gradio interface - versi simple, informatif, tanpa icon
def gradio_interface(text: str, max_excerpts: int = 3):
    """
    Gradio interface dengan label bahasa Indonesia yang jelas.
    
    Args:
        text: Teks yang akan dianalisis
        max_excerpts: Jumlah bagian teks penting yang akan ditampilkan (1-5)
    
    Returns:
        String markdown dengan hasil analisis atau pesan error
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
        logger.info(f"Processing Gradio request (length: {len(text)} chars, max_excerpts: {max_excerpts})")
        pipe = get_pipeline()
        result = pipe.process(text, max_excerpts=max_excerpts)
        logger.info("Gradio analysis completed successfully")
        
        # format output user-friendly tanpa icon
        sentiment_label = 'Positif' if result['document_sentiment'] == 1 else 'Negatif'
        
        chunk_stats = result['chunk_statistics']
        total_chunks = chunk_stats['total_chunks']
        positive_pct = (chunk_stats['positive_chunks'] / total_chunks * 100) if total_chunks > 0 else 0
        negative_pct = (chunk_stats['negative_chunks'] / total_chunks * 100) if total_chunks > 0 else 0
        
        # output dengan penjelasan jelas
        summary_text = f"""
## Hasil Analisis Sentimen

**Sentimen Dokumen:** {sentiment_label}  
**Tingkat Keyakinan:** {result['document_confidence']:.1%}  
*Tingkat keyakinan model dalam menentukan sentimen (0-100%)*

---

## Informasi Teks

- **Panjang Teks:** {result['text_length']:,} karakter
- **Jumlah Bagian:** {total_chunks} bagian  
*Teks panjang dibagi menjadi beberapa bagian untuk analisis yang lebih akurat*

---

## Distribusi Sentimen per Bagian

- **Bagian Positif:** {chunk_stats['positive_chunks']} bagian ({positive_pct:.1f}%)
- **Bagian Negatif:** {chunk_stats['negative_chunks']} bagian ({negative_pct:.1f}%)  
*Setiap bagian teks dianalisis secara terpisah, kemudian digabungkan untuk hasil akhir*

---

## Bagian Teks Penting & Ringkasan

*Bagian teks dengan tingkat keyakinan tertinggi yang paling mewakili sentimen dokumen*

"""
        
        for i, (excerpt, summary) in enumerate(zip(result['key_excerpts'], result['summaries']), 1):
            excerpt_label = 'Positif' if excerpt['label'] == 1 else 'Negatif'
            
            summary_text += f"""
### Bagian {i} - {excerpt_label} (Keyakinan: {excerpt['confidence']:.1%})

**Teks Asli:**
> {excerpt['text'][:200]}{'...' if len(excerpt['text']) > 200 else ''}

**Ringkasan:**
{summary}

---
"""
        
        return summary_text
    except Exception as e:
        logger.error(f"Error in Gradio interface: {str(e)}", exc_info=True)
        return "## Terjadi Kesalahan\n\n**Terjadi kesalahan saat memproses teks. Silakan coba lagi.**"

with gr.Blocks(title="Opinion Intelligence Engine") as gradio_app:
    gr.Markdown("# Opinion Intelligence Engine")
    gr.Markdown("Analisis sentiment dan summarization untuk teks panjang")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Masukkan Teks",
                placeholder="Masukkan teks panjang untuk analisis sentimen dan ringkasan...",
                lines=12,
                show_label=True
            )
            max_excerpts = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Jumlah Bagian Teks Penting",
                info="Berapa banyak bagian teks dengan keyakinan tertinggi yang akan ditampilkan (1-5)"
            )
            submit_btn = gr.Button("Analisis", variant="primary", size="lg")
            clear_btn = gr.Button("Hapus", variant="secondary")
        
        with gr.Column(scale=3):
            output = gr.Markdown(
                label="Hasil Analisis",
                value="**Hasil analisis akan muncul di sini setelah Anda memasukkan teks dan klik tombol Analisis...**"
            )
    
    # Event handlers
    submit_btn.click(
        fn=gradio_interface,
        inputs=[text_input, max_excerpts],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: ("", 3, "**Hasil analisis akan muncul di sini setelah Anda memasukkan teks dan klik tombol Analisis...**"),
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
    uvicorn.run(app, host="0.0.0.0", port=port)