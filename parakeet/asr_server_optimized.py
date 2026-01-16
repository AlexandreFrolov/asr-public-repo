# /home/ubuntu/parakeet_v3/asr_server_optimized.py
import asyncio
import sys
import os
import warnings
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

# Подавляем предупреждение о pkg_resources
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

sys.path.insert(0, '/home/ubuntu/parakeet_v3')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import numpy as np
import torch
import time
import logging
from datetime import datetime
from typing import List, Optional
import traceback
import soundfile as sf

# Настройка логирования
log_dir = "/home/ubuntu/parakeet_v3/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/asr_optimized.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
SAMPLE_RATE = 16000
BUFFER_SECONDS = 5  # Увеличили для лучшего качества
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = 2  # Количество потоков для обработки

# Оптимизации для CUDA
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

app = FastAPI()

# Пул потоков для обработки транскрипции
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Загружаем русскую модель
logger.info("="*60)
logger.info("🚀 Запуск оптимизированного ASR сервера")
logger.info("="*60)
logger.info(f"📡 Устройство: {DEVICE}")
logger.info(f"🔧 Размер буфера: {BUFFER_SECONDS} секунд")
logger.info(f"🤖 Модель: stt_ru_conformer_ctc_large")
logger.info(f"👥 Потоков обработки: {MAX_WORKERS}")
logger.info("="*60)

MODEL_LOADED = False
model = None

try:
    import nemo.collections.asr as nemo_asr
    logger.info("✅ NVIDIA NeMo импортирован успешно")
    
    # Загружаем русскую модель Conformer CTC Large
    logger.info("🔄 Загрузка модели stt_ru_conformer_ctc_large...")
    
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="stt_ru_conformer_ctc_large",
        map_location=torch.device(DEVICE)
    )
    
    if DEVICE == "cuda":
        model = model.cuda()
        logger.info(f"✅ Модель загружена на CUDA")
    
    # Переводим модель в режим оценки
    model.eval()
        
    MODEL_LOADED = True
    
    logger.info("✅ Модель загружена успешно")
    
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    logger.error(traceback.format_exc())

def transcribe_audio_file(audio_data: np.ndarray) -> str:
    """Функция для транскрипции аудио (запускается в отдельном потоке)"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Сохраняем аудио как WAV файл
            sf.write(tmp.name, audio_data, SAMPLE_RATE)
            
            # Для CTC моделей используем paths2audio_files
            transcriptions = model.transcribe(
                paths2audio_files=[tmp.name],
                batch_size=1
            )
            
            # Удаляем временный файл
            os.unlink(tmp.name)
        
        if transcriptions and len(transcriptions) > 0:
            return transcriptions[0].strip()
        else:
            return ""
            
    except Exception as e:
        logger.error(f"Ошибка в потоке транскрипции: {e}")
        return ""

@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    
    logger.info(f"✅ Клиент подключился: {client_ip}")
    
    audio_buffer = []
    buffer_start_time = None
    
    try:
        while True:
            data = await websocket.receive_bytes()
            pcm_data = np.frombuffer(data, dtype=np.int16)
            
            if buffer_start_time is None:
                buffer_start_time = time.time()
            
            audio_buffer.append(pcm_data)
            
            total_samples = sum(len(chunk) for chunk in audio_buffer)
            buffer_duration = total_samples / SAMPLE_RATE
            
            if buffer_duration >= BUFFER_SECONDS:
                combined_audio = np.concatenate(audio_buffer)
                audio_float = combined_audio.astype(np.float32) / 32768.0
                
                # Транскрибируем в отдельном потоке
                start_transcribe = time.time()
                
                # Запускаем транскрипцию в пуле потоков
                future = executor.submit(transcribe_audio_file, audio_float)
                text = future.result(timeout=10.0)  # Таймаут 10 секунд
                
                transcribe_time = time.time() - start_transcribe
                
                if text:
                    await websocket.send_text(text)
                    logger.info(f"📝 [{client_ip}] {text} (время обработки: {transcribe_time:.2f}с)")
                else:
                    # Если текст пустой, возможно это тишина
                    signal_level = np.mean(np.abs(audio_float))
                    if signal_level > 0.01:  # Если есть сигнал
                        await websocket.send_text("...")
                        logger.info(f"📝 [{client_ip}] Пустой результат (уровень сигнала: {signal_level:.4f})")
                
                # Очищаем буфер
                audio_buffer = []
                buffer_start_time = None
                
    except WebSocketDisconnect:
        logger.info(f"❌ Клиент отключился: {client_ip}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket [{client_ip}]: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.get("/health")
async def health_check():
    """Проверка состояния сервера"""
    cuda_info = None
    if torch.cuda.is_available():
        cuda_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        }
    
    return {
        "status": "healthy" if MODEL_LOADED else "error",
        "model": "stt_ru_conformer_ctc_large",
        "model_loaded": MODEL_LOADED,
        "device": DEVICE,
        "cuda": cuda_info,
        "sample_rate": SAMPLE_RATE,
        "buffer_seconds": BUFFER_SECONDS,
        "workers": MAX_WORKERS,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Оптимизированный ASR сервер")
    print("="*60)
    print(f"🌐 Host: 0.0.0.0")
    print(f"🔌 Port: 8000")
    print("="*60)
    
    if not MODEL_LOADED:
        print("❌ ОШИБКА: Модель не загружена!")
        sys.exit(1)
    
    print("✅ Модель успешно загружена и готова к работе!")
    print("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False
    )


