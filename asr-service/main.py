import os
from typing import Dict

import numpy as np
import uvicorn
import whisper
from fastapi import FastAPI, HTTPException, Query, File
from fastapi.responses import JSONResponse
import structlog


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

app = FastAPI(title="ASR Service")


class ASRService:
    """
    class for asr model
    functions - _init_, load_model, transcribe_audio
    """
    def __init__(self, model_size: str = "base"):
        """
        model_size (str): The size of the model. Allowable values: "tiny", "base", "small", "medium", "large"
        """
        self.model_size = model_size
        self.model = None
        
    def load_model(self):
        """
        Load Whisper model
        """
        try:
            self.model = whisper.load_model(self.model_size)
            logger.info("ASR model loaded successfully", model_size=self.model_size)
        except Exception as e:
            logger.error("Failed to load ASR model", error=str(e))
            raise
    
    def transcribe_audio(self, audio_bytes: bytes, sample_rate: int, channels: int) -> Dict:
        """
        Transcribe audio bytes to text
        Parameters: audio_bytes (bytes): Raw audio data in PCM format without header,
                    sample_rate (int): Audio sampling rate,
                    channels (int): Number of audio channels 
        Returns: Transcription result
        """
        if not self.model:
            raise HTTPException(status_code=503, detail="ASR model not loaded")
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data")
        # Check duration (max 15 seconds)
        duration = len(audio_bytes) / (sample_rate * channels * 2)  # 16-bit = 2 bytes
        if duration > 15:
            raise HTTPException(status_code=400, detail="Audio too long (max 15 seconds)")
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # Handle mono/stereo
            if channels == 2:
                audio_array = audio_array.reshape((-1, 2)).mean(axis=1)
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            # Transcribe
            result = self.model.transcribe(audio_array, language="en")
            # Format response
            response = {
                "text": result["text"].strip(),
                "segments": [
                    {
                        "start_ms": int(segment["start"] * 1000),
                        "end_ms": int(segment["end"] * 1000),
                        "text": segment["text"].strip()
                    }
                    for segment in result["segments"]
                ]
            }
            logger.info("Transcription completed", text_length=len(response["text"]))
            return response
        except Exception as e:
            logger.error("Transcription failed", error=str(e))
            raise HTTPException(status_code=500, detail="Transcription failed")


asr_service = ASRService(os.getenv("MODEL_SIZE", "base"))


@app.on_event("startup")
async def startup_event():
    """
    Application launch event.
    Loads the Whisper model when the service starts.
    """
    asr_service.load_model()


@app.post("/api/stt/bytes")
async def stt_from_bytes(
    audio_data: bytes = File(...),
    sr: int = Query(16000, ge=8000, le=48000, description="Sample rate"),
    ch: int = Query(1, ge=1, le=2, description="Channels"),
    lang: str = Query("en", description="Language")
):
    """Endpoint for speech recognition from raw audio data"""
    if lang != "en":
        raise HTTPException(status_code=400, detail="Only English language supported")
    
    try:
        result = asr_service.transcribe_audio(audio_data, sr, ch)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("STT processing error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": asr_service.model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_config=None)
