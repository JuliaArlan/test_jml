import asyncio
import json
import os
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from piper.voice import PiperVoice
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve

import structlog


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
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

app = FastAPI(title="TTS Service")


class TTSService:
    def __init__(self, model_path: str):
        """
        Parameters:
            model_path (str): Path to the Piper TTS model file in ONNX format
        """
        self.model_path = model_path
        self.voice: Optional[PiperVoice] = None
        self.sample_rate = 22050
        
    def load_model(self):
        """
        Load TTS model
        """
        try:
            from piper.voice import PiperVoice
            self.voice = PiperVoice.load(self.model_path)
            logger.info("TTS model loaded successfully", model_path=self.model_path)
        except Exception as e:
            logger.error("Failed to load TTS model", error=str(e))
            raise
    
    async def synthesize_stream(self, text: str, chunk_size: int = 1024):
        """
        Stream synthesized audio in chunks
        Parameters:
            text (str): Text for speech synthesis
            chunk_size (int): Size of audio chunks in bytes
        """
        if not self.voice:
            raise HTTPException(status_code=503, detail="TTS model not loaded")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Empty text")
        
        try:
            # Generate audio using Piper
            audio_array = self.voice.synthesize(text)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            
            # Stream in chunks
            total_size = len(audio_bytes)
            for i in range(0, total_size, chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.001)  # Small delay to simulate streaming
                
        except Exception as e:
            logger.error("TTS synthesis failed", error=str(e))
            raise HTTPException(status_code=500, detail="Synthesis failed")

tts_service = TTSService(os.getenv("MODEL_PATH"))


@app.on_event("startup")
async def startup_event():
    tts_service.load_model()


@app.post("/api/tts")
async def tts_http(text: dict):
    """
    HTTP endpoint for TTS
    Parameters:
        text (dict): Dictionary with text for synthesis
    """
    if "text" not in text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    
    return StreamingResponse(
        tts_service.synthesize_stream(text["text"]),
        media_type="application/octet-stream"
    )


async def websocket_handler(websocket: WebSocketServerProtocol):
    """
    WebSocket handler for TTS
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                text = data.get("text", "").strip()
                
                if not text:
                    await websocket.send(json.dumps({"error": "Empty text"}))
                    continue
                
                # Stream audio chunks
                async for chunk in tts_service.synthesize_stream(text):
                    await websocket.send(chunk)
                
                # Send end signal
                await websocket.send(json.dumps({"type": "end"}))
                logger.info("TTS streaming completed", text_length=len(text))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error("WebSocket TTS error", error=str(e))
                await websocket.send(json.dumps({"error": str(e)}))
                
    except ConnectionClosed:
        logger.info("WebSocket connection closed")


async def websocket_server():
    """
    Start WebSocket server
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8082))
    
    async with serve(websocket_handler, host, port):
        logger.info("WebSocket server started", host=host, port=port)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    # Start both HTTP and WebSocket servers
    import threading
    
    def run_websocket():
        asyncio.run(websocket_server())
    
    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8082, log_config=None)
