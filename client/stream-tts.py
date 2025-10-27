import asyncio
import json
import time
import wave
import websockets
from datetime import datetime

async def stream_tts():
    uri = "ws://localhost:8000/ws/tts"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to gateway WebSocket")
            
            # Send text for TTS
            text = "Hello world! This is a test of streaming text to speech."
            await websocket.send(json.dumps({"text": text}))
            print(f"Sent text: {text}")
            
            # Receive audio stream
            audio_chunks = []
            start_time = time.time()
            
            async for message in websocket:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get("type") == "end":
                        print("Received end signal")
                        break
                    elif data.get("error"):
                        print(f"Error: {data['error']}")
                        break
                else:
                    # Binary audio data
                    receive_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"Received audio chunk: {len(message)} bytes at {receive_time}")
                    audio_chunks.append(message)
            
            end_time = time.time()
            print(f"Streaming completed in {end_time - start_time:.2f} seconds")
            
            # Save to WAV file
            if audio_chunks:
                audio_data = b''.join(audio_chunks)
                with wave.open("out.wav", "wb") as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(22050)  # sample rate
                    wav_file.writeframes(audio_data)
                print(f"Saved audio to out.wav ({len(audio_data)} bytes)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(stream_tts())
