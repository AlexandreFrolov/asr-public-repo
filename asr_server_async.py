import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from faster_whisper import WhisperModel
import numpy as np
import torch
import wave
import os
import time
from datetime import datetime
import sys

# Конфигурация
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3

# Определяем режим отладки через переменную окружения или аргумент командной строки
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Если переданы аргументы командной строки
if len(sys.argv) > 1 and "--debug" in sys.argv:
    DEBUG = True

AUDIO_SAVE_DIR = "recordings" if DEBUG else None  # Папка для сохранения файлов только в режиме отладки

app = FastAPI()

# Создаем папку для сохранения аудио только в режиме отладки
if DEBUG and AUDIO_SAVE_DIR and not os.path.exists(AUDIO_SAVE_DIR):
    os.makedirs(AUDIO_SAVE_DIR)

# Проверка GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Starting speech recognition server")
print(f"📡 Using device: {device}")
print(f"🔧 Buffer size: {BUFFER_SECONDS} seconds")
print(f"⚡ Model: large-v3")
print(f"🐛 Debug mode: {'ENABLED' if DEBUG else 'DISABLED'}")
if DEBUG:
    print(f"📁 Audio recordings will be saved to: {AUDIO_SAVE_DIR}")

# Загружаем модель Whisper
model = WhisperModel("large-v3", device=device)

async def transcribe_audio(audio_float: np.ndarray):
    """Запускаем транскрипцию в отдельном потоке"""
    return await asyncio.to_thread(model.transcribe, audio_float, beam_size=5, language="ru")

def save_audio_to_wav(audio_int16: np.ndarray, filename: str):
    """Сохраняет аудио данные в WAV файл (только в режиме отладки)"""
    try:
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # моно
            wav_file.setsampwidth(2)   # 16 бит = 2 байта
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        print(f"💾 Audio saved to {filename}")
        return True
    except Exception as e:
        print(f"⚠ Error saving WAV file: {e}")
        return False

@app.websocket("/ws/asr")
async def asr_ws(ws: WebSocket):
    await ws.accept()
    print("✅ Client connected")

    buffer = np.zeros(0, dtype=np.int16)
    all_audio = np.zeros(0, dtype=np.int16) if DEBUG else None  # Для сохранения всего аудио только в режиме отладки
    
    # Создаем уникальное имя файла для этого подключения только в режиме отладки
    if DEBUG:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        connection_id = id(ws)
        wav_filename = os.path.join(AUDIO_SAVE_DIR, f"audio_{timestamp}_{connection_id}.wav")

    try:
        while True:
            try:
                data = await ws.receive_bytes()
            except WebSocketDisconnect:
                print("❌ Client disconnected")
                break
            except Exception as e:
                print("⚠ Error receiving data:", e)
                continue

            pcm = np.frombuffer(data, dtype=np.int16)
            buffer = np.concatenate([buffer, pcm])
            
            # Сохраняем для записи в файл только в режиме отладки
            if DEBUG and all_audio is not None:
                all_audio = np.concatenate([all_audio, pcm])
            
            # Вывод отладочной информации только в режиме отладки
            if DEBUG:
                print(f"📥 Received chunk: {len(pcm)} samples, buffer total: {len(buffer)} samples")
                print(f"📊 Chunk stats - Min: {np.min(pcm)}, Max: {np.max(pcm)}, Mean: {np.mean(np.abs(pcm)):.1f}")

            # Если накопили ≥ BUFFER_SECONDS, делаем транскрипцию
            if len(buffer) >= SAMPLE_RATE * BUFFER_SECONDS:
                audio_float = buffer.astype(np.float32) / 32768.0
                
                # Дополнительная диагностика амплитуды только в режиме отладки
                if DEBUG:
                    max_amplitude = np.max(np.abs(audio_float))
                    print(f"📈 Buffer amplitude: {max_amplitude:.4f} (target: 0.1-0.9)")
                
                try:
                    segments, info = await transcribe_audio(audio_float)
                    for segment in segments:
                        await ws.send_text(segment.text)
                        # Выводим распознанный текст всегда
                        if DEBUG:
                            # В режиме отладки показываем дополнительную информацию
                            segment_info = f"📝 Segment: {segment.start:.2f}s -> {segment.end:.2f}s"
                            if hasattr(segment, 'avg_logprob'):
                                segment_info += f" | Logprob: {segment.avg_logprob:.2f}"
                            if hasattr(segment, 'no_speech_prob'):
                                segment_info += f" | No speech prob: {segment.no_speech_prob:.2f}"
                            segment_info += f" | Text: '{segment.text}'"
                            print(segment_info)
                        else:
                            # В обычном режиме показываем только текст
                            print(f"📝 '{segment.text}'")
                    
                    # Информация о транскрипции только в режиме отладки
                    if DEBUG and hasattr(info, 'language'):
                        print(f"🌐 Detected language: {info.language} (probability: {info.language_probability:.2f})")
                        
                except Exception as e:
                    print("⚠ Transcription error:", e)

                buffer = np.zeros(0, dtype=np.int16)

    except Exception as e:
        print("❌ Unexpected error:", e)
    finally:
        print("⏹ Connection closed")
        
        # Сохраняем все полученное аудио в WAV файл только в режиме отладки
        if DEBUG and all_audio is not None and len(all_audio) > 0:
            print(f"💿 Total audio received: {len(all_audio)} samples ({len(all_audio)/SAMPLE_RATE:.2f} seconds)")
            
            # Проверяем уровень сигнала
            max_val = np.max(np.abs(all_audio))
            avg_val = np.mean(np.abs(all_audio))
            print(f"📊 Signal stats - Max: {max_val}, Avg: {avg_val:.1f}")
            
            # Предупреждения
            if max_val < 1000:
                print("⚠ WARNING: Signal too weak! Check microphone gain.")
            elif max_val > 30000:
                print("⚠ WARNING: Possible clipping (signal too strong)!")
            
            # Сохраняем в WAV
            success = save_audio_to_wav(all_audio, wav_filename)
            if success:
                print(f"🎵 File saved: {wav_filename}")
        
        try:
            await ws.close()
        except Exception:
            pass

@app.get("/recordings")
async def list_recordings():
    """API endpoint для просмотра сохраненных записей (только в режиме отладки)"""
    if not DEBUG or not os.path.exists(AUDIO_SAVE_DIR):
        return {"recordings": [], "message": "Debug mode is disabled"}
    
    files = []
    for filename in os.listdir(AUDIO_SAVE_DIR):
        if filename.endswith('.wav'):
            filepath = os.path.join(AUDIO_SAVE_DIR, filename)
            size = os.path.getsize(filepath)
            files.append({
                "name": filename,
                "size_bytes": size,
                "size_mb": size / (1024 * 1024),
                "path": filepath
            })
    
    return {"recordings": files}

@app.delete("/recordings/clear")
async def clear_recordings():
    """Очистить все записи (только в режиме отладки)"""
    if not DEBUG:
        return {"cleared": 0, "message": "Debug mode is disabled"}
    
    if os.path.exists(AUDIO_SAVE_DIR):
        count = 0
        for filename in os.listdir(AUDIO_SAVE_DIR):
            if filename.endswith('.wav'):
                os.remove(os.path.join(AUDIO_SAVE_DIR, filename))
                count += 1
        return {"cleared": count, "message": f"Removed {count} recording files"}
    return {"cleared": 0, "message": "No recordings to clear"}

@app.get("/health")
async def health_check():
    """Проверка состояния сервера"""
    return {
        "status": "ok",
        "device": device,
        "model": "large-v3",
        "debug_mode": DEBUG,
        "buffer_seconds": BUFFER_SECONDS
    }

if __name__ == "__main__":
    print("-" * 50)
    print(f"📝 Usage: DEBUG=true python3 {sys.argv[0]} [--debug]")
    print(f"📝 или: python3 {sys.argv[0]} --debug")
    print("-" * 50)
    
    uvicorn.run(
        "asr_server_async:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False  # Выключаем логи доступа для производительности
    )
