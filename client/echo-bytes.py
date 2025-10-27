import asyncio
import wave
import httpx
import time
from datetime import datetime

async def echo_bytes():
    # Read input WAV file
    try:
        with wave.open("input.wav", "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
        
        print(f"Input WAV: {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")
        
    except FileNotFoundError:
        print("Error: input.wav not found")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return
    
    # Send to echo-bytes endpoint
    url = f"http://localhost:8000/api/echo-bytes?sr={sample_rate}&ch={channels}&fmt=s16le"
    
    try:
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            
            async with client.stream(
                "POST", 
                url, 
                content=frames,
                headers={"Content-Type": "application/octet-stream"},
                timeout=60.0
            ) as response:
                
                if response.status_code != 200:
                    print(f"Error: HTTP {response.status_code}")
                    return
                
                # Receive streaming audio
                audio_chunks = []
                print("Receiving audio stream...")
                
                async for chunk in response.aiter_bytes():
                    receive_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"Received audio chunk: {len(chunk)} bytes at {receive_time}")
                    audio_chunks.append(chunk)
            
            end_time = time.time()
            print(f"Echo completed in {end_time - start_time:.2f} seconds")
            
            # Save output WAV
            if audio_chunks:
                audio_data = b''.join(audio_chunks)
                with wave.open("out_echo.wav", "wb") as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(22050)  # TTS sample rate
                    wav_file.writeframes(audio_data)
                print(f"Saved echo audio to out_echo.wav ({len(audio_data)} bytes)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(echo_bytes())
