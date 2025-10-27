import json
import os

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from websockets.client import connect as ws_connect

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

app = FastAPI(title="Gateway Service")


class GatewayService:
    def __init__(self):
        self.tts_url = os.getenv("TTS_SERVICE_URL", "ws://tts-service:8082/ws/tts")
        self.asr_url = os.getenv("ASR_SERVICE_URL", "http://asr-service:8081/api/stt/bytes")
    
    async def stream_tts_to_client(self, text: str, client_websocket):
        """
        Stream TTS audio to client WebSocket
        Parameters:
        text (str): Text for speech synthesis
        client_websocket: WebSocket connection to the client
        """
        try:
            async with ws_connect(self.tts_url) as tts_ws:
                # Send text to TTS service
                await tts_ws.send(json.dumps({"text": text}))
                logger.info("TTS request sent", text_length=len(text))
                
                # Stream audio chunks to client
                async for message in tts_ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get("type") == "end":
                            await client_websocket.send(json.dumps({"type": "end"}))
                            break
                        elif data.get("error"):
                            await client_websocket.send(json.dumps({"error": data["error"]}))
                            break
                    else:
                        # Binary audio data
                        await client_websocket.send(message)
                        
        except Exception as e:
            logger.error("TTS streaming failed", error=str(e))
            await client_websocket.send(json.dumps({"error": "TTS service unavailable"}))
    
    async def echo_bytes_stream(self, audio_bytes: bytes, sample_rate: int, channels: int):
        """
        Stream echo bytes: ASR → TTS → client
        Parameters:
            audio_bytes (bytes): Raw audio data in PCM format
            sample_rate (int): Audio sampling frequency
            channels (int): Number of audio channels
        """
        try:
            # Step 1: Transcribe with ASR
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.asr_url}?sr={sample_rate}&ch={channels}&lang=en",
                    content=audio_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="ASR service error")
                
                asr_result = response.json()
                text = asr_result.get("text", "")
                logger.info("ASR completed", text=text)
                
                if not text:
                    yield b""  # Empty stream
                    return
                
                # Step 2: Stream TTS for recognized text
                async with ws_connect(self.tts_url) as tts_ws:
                    await tts_ws.send(json.dumps({"text": text}))
                    
                    async for message in tts_ws:
                        if isinstance(message, str):
                            data = json.loads(message)
                            if data.get("type") == "end":
                                break
                            elif data.get("error"):
                                raise HTTPException(status_code=500, detail=data["error"])
                        else:
                            yield message
                            
        except httpx.TimeoutException:
            logger.error("ASR service timeout")
            raise HTTPException(status_code=504, detail="ASR service timeout")
        except Exception as e:
            logger.error("Echo bytes failed", error=str(e))
            raise HTTPException(status_code=500, detail="Echo service failed")


gateway_service = GatewayService()


@app.websocket("/ws/tts")
async def websocket_tts(websocket):
    """WebSocket endpoint for TTS through gateway"""
    await websocket.accept()
    logger.info("Gateway WebSocket connection established")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if "text" in data:
                # Single text
                await gateway_service.stream_tts_to_client(data["text"], websocket)
            elif "segments" in data:
                # Multiple segments
                for segment in data["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        await gateway_service.stream_tts_to_client(text, websocket)
            else:
                await websocket.send(json.dumps({"error": "Invalid message format"}))
                
    except Exception as e:
        logger.error("Gateway WebSocket error", error=str(e))
        await websocket.send(json.dumps({"error": str(e)}))


@app.post("/api/echo-bytes")
async def echo_bytes(
    audio_data: bytes,
    sr: int = Query(16000, description="Sample rate"),
    ch: int = Query(1, description="Channels"),
    fmt: str = Query("s16le", description="Format")
):
    """
    Echo bytes endpoint: ASR → TTS → audio stream
    Parameters:
    audio_data (bytes): Request body with raw audio data
    sr (int): Query parameter, sample rate
    ch (int): Query parameter, number of channels
    fmt (str): Query parameter, audio format
    """
    if fmt != "s16le":
        raise HTTPException(status_code=400, detail="Only s16le format supported")
    
    return StreamingResponse(
        gateway_service.echo_bytes_stream(audio_data, sr, ch),
        media_type="application/octet-stream"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
