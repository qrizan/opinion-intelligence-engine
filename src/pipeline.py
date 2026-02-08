"""
Pipeline untuk analisis sentiment dan summarization teks panjang.
"""

import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class OpinionIntelligencePipeline:
    """Pipeline untuk analisis sentiment dan summarization teks panjang."""
    
    def __init__(self):
        """Initialize pipeline dengan load model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = Path(__file__).parent.parent
        
        # validasi path model
        self.sentiment_model_path = self.base_dir / "models" / "sentiment"
        self.summarization_model_path = self.base_dir / "models" / "summarization"
        
        if not self.sentiment_model_path.exists():
            raise FileNotFoundError(
                f"Sentiment model tidak ditemukan di {self.sentiment_model_path}\n"
                f"Pastikan model sudah di-download dengan menjalankan: python download_models.py"
            )
        
        if not self.summarization_model_path.exists():
            raise FileNotFoundError(
                f"Summarization model tidak ditemukan di {self.summarization_model_path}\n"
                f"Pastikan model sudah di-download dengan menjalankan: python download_models.py"
            )
        
        # validasi file penting
        required_sentiment_files = ["config.json", "model.safetensors", "tokenizer.json"]
        missing_sentiment = [
            f for f in required_sentiment_files 
            if not (self.sentiment_model_path / f).exists()
        ]
        if missing_sentiment:
            raise FileNotFoundError(
                f"File sentiment model tidak lengkap. Missing: {missing_sentiment}\n"
                f"Pastikan model sudah di-download dengan lengkap."
            )
        
        required_summarization_files = ["config.json", "model.safetensors", "tokenizer.json"]
        missing_summarization = [
            f for f in required_summarization_files 
            if not (self.summarization_model_path / f).exists()
        ]
        if missing_summarization:
            raise FileNotFoundError(
                f"File summarization model tidak lengkap. Missing: {missing_summarization}\n"
                f"Pastikan model sudah di-download dengan lengkap."
            )
        
        # load sentiment model
        try:
            logger.info("Loading sentiment model...")
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                self.sentiment_model_path,
                local_files_only=True
            )
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_path,
                local_files_only=True
            ).to(self.device)
            self.sentiment_model.eval()
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading sentiment model: {str(e)}")
        
        # load summarization model
        try:
            logger.info("Loading summarization model...")
            self.summarization_tokenizer = AutoTokenizer.from_pretrained(
                self.summarization_model_path,
                local_files_only=True
            )
            self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.summarization_model_path,
                local_files_only=True
            ).to(self.device)
            self.summarization_model.eval()
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading summarization model: {str(e)}")
        
        logger.info("Pipeline initialized successfully")
    
    def chunk_text_by_tokens(self, text: str, max_length: int = 512) -> List[str]:
        """
        Chunk teks berdasarkan token length.
        
        Args:
            text: Teks yang akan di-chunk
            max_length: Panjang maksimal token per chunk (default: 512)
        
        Returns:
            List string yang berisi chunks teks
        """

        tokens = self.sentiment_tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), max_length):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = self.sentiment_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def predict_sentiment_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment untuk setiap chunk.
        
        Args:
            chunks: List string yang berisi chunks teks
        
        Returns:
            List dictionary dengan keys: text, label, confidence
        """

        predictions = []
        
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.sentiment_tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.sentiment_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_label].item()
                
                predictions.append({
                    "text": chunk,
                    "label": pred_label,
                    "confidence": confidence
                })
        
        return predictions
    
    def aggregate_chunk_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate chunk predictions menjadi document-level sentiment.
        
        Menggunakan majority voting untuk menentukan label dokumen dan
        menghitung rata-rata confidence untuk label yang menang.
        
        Args:
            predictions: List dictionary dengan keys: text, label, confidence
        
        Returns:
            Dictionary dengan keys: document_sentiment, document_confidence
        """

        labels = [p["label"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        
        # majority voting untuk label
        document_label = max(set(labels), key=labels.count)
        
        # average confidence untuk label yang menang
        winning_confidences = [c for l, c in zip(labels, confidences) if l == document_label]
        document_confidence = np.mean(winning_confidences)
        
        return {
            "document_sentiment": document_label,
            "document_confidence": float(document_confidence)
        }
    
    def select_key_excerpts(self, predictions: List[Dict[str, Any]], max_excerpts: int = 3) -> List[Dict[str, Any]]:
        """
        Pilih excerpt dengan confidence tertinggi.
        
        Args:
            predictions: List dictionary dengan keys: text, label, confidence
            max_excerpts: Jumlah excerpt teratas yang akan dipilih (default: 3)
        
        Returns:
            List dictionary dengan excerpt terpilih, diurutkan berdasarkan confidence tertinggi
        """

        sorted_predictions = sorted(
            predictions,
            key=lambda x: x["confidence"],
            reverse=True
        )
        return sorted_predictions[:max_excerpts]
    
    def summarize_excerpts(self, excerpts: List[Dict[str, Any]]) -> List[str]:
        """
        Summarize setiap excerpt menggunakan model summarization.
        
        Args:
            excerpts: List dictionary dengan keys: text, label, confidence
        
        Returns:
            List string yang berisi ringkasan untuk setiap excerpt
        """

        summaries = []
        
        for excerpt in excerpts:
            text = excerpt["text"]
            inputs = self.summarization_tokenizer(
                f"summarize: {text}",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.summarization_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.summarization_tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            summaries.append(summary)
        
        return summaries
    
    def process(self, text: str, max_excerpts: int = 3) -> Dict[str, Any]:
        """
        Process teks panjang: sentiment analysis + summarization.
        
        Pipeline:
        1. Chunk teks menjadi bagian-bagian
        2. Predict sentiment untuk setiap chunk
        3. Aggregate menjadi document-level sentiment
        4. Select key excerpts berdasarkan confidence
        5. Summarize key excerpts
        
        Args:
            text: Teks panjang yang akan dianalisis
            max_excerpts: Jumlah excerpt teratas yang akan dipilih (default: 3)
        
        Returns:
            Dictionary dengan keys:
                - document_sentiment: 0 (negative) atau 1 (positive)
                - document_confidence: float (0-1)
                - text_length: int (karakter)
                - text_length_tokens: int (token)
                - chunk_statistics: dict dengan statistik chunks
                - key_excerpts: list dict dengan excerpt terpilih
                - summaries: list string dengan ringkasan
        """

        # 1. chunk text
        chunks = self.chunk_text_by_tokens(text)
        
        # 2. predict sentiment per chunk
        chunk_predictions = self.predict_sentiment_chunks(chunks)
        
        # 3. aggregate menjadi document-level
        doc_result = self.aggregate_chunk_predictions(chunk_predictions)
        
        # 4. select key excerpts
        key_excerpts = self.select_key_excerpts(chunk_predictions, max_excerpts)
        
        # 5. summarize excerpts
        summaries = self.summarize_excerpts(key_excerpts)
        
        # statistik chunks
        total_chunks = len(chunks)
        positive_chunks = sum(1 for p in chunk_predictions if p["label"] == 1)
        negative_chunks = sum(1 for p in chunk_predictions if p["label"] == 0)
        avg_confidence = float(np.mean([p["confidence"] for p in chunk_predictions]))
        
        # token count
        token_count = len(self.sentiment_tokenizer.encode(text, add_special_tokens=False))
        
        return {
            "document_sentiment": doc_result["document_sentiment"],
            "document_confidence": doc_result["document_confidence"],
            "text_length": len(text),
            "text_length_tokens": token_count,
            "chunk_statistics": {
                "total_chunks": total_chunks,
                "positive_chunks": positive_chunks,
                "negative_chunks": negative_chunks,
                "avg_confidence": avg_confidence
            },
            "key_excerpts": [
                {
                    "text": ex["text"],
                    "confidence": ex["confidence"],
                    "label": ex["label"]
                }
                for ex in key_excerpts
            ],
            "summaries": summaries
        }