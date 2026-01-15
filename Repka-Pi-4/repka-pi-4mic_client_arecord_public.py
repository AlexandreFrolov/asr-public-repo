import asyncio
import websockets
import subprocess
import numpy as np
import time
import queue
import threading
import wave
import os
import sys
import argparse

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='ASR Stream Client')
parser.add_argument('--debug', '-d', action='store_true', 
                    help='Enable debug mode (save audio to file and show detailed logs)')
parser.add_argument('--server', '-s', default="ws://195.209.210.71:8000/ws/asr",
                    help='WebSocket server URL')
args = parser.parse_args()

SERVER_WS = args.server
DEBUG = args.debug
CHUNK = 1600  # samples per chunk

# Для отладки: файл для сохранения аудио
debug_filename = None
debug_wav_file = None

def read_from_arecord(q, stop_event):
    """Читает данные из arecord в отдельном потоке"""
    cmd = [
        "arecord",
        "-D", "plughw:2,0",
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-t", "raw",
        "-q",  # Тихий режим
        "--buffer-size=65536"  # Большой буфер
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if DEBUG:
        print("🎤 arecord started in background thread")
    
    try:
        while not stop_event.is_set():
            # Читаем чанк данных
            data = proc.stdout.read(CHUNK * 2)
            if not data:
                break
            
            # Проверяем размер
            if len(data) == CHUNK * 2:
                q.put(data)
            elif DEBUG:
                print(f"⚠ Incomplete chunk in thread: {len(data)} bytes")
                
    except Exception as e:
        if DEBUG:
            print(f"❌ Thread error: {e}")
    finally:
        proc.terminate()
        proc.wait()
        if DEBUG:
            print("🛑 arecord stopped")

async def stream_audio():
    global debug_wav_file, debug_filename
    
    async with websockets.connect(SERVER_WS) as ws:
        print("✅ Connected to server")
        
        # Инициализация отладки
        if DEBUG:
            timestamp = int(time.time())
            debug_filename = f"client_debug_{timestamp}.wav"
            debug_wav_file = wave.open(debug_filename, 'wb')
            debug_wav_file.setnchannels(1)
            debug_wav_file.setsampwidth(2)
            debug_wav_file.setframerate(16000)
            print(f"📁 Debug mode enabled, saving to: {debug_filename}")
        
        # Создаем очередь для данных
        audio_queue = queue.Queue(maxsize=100)
        stop_event = threading.Event()
        
        # Запускаем поток чтения
        reader_thread = threading.Thread(
            target=read_from_arecord,
            args=(audio_queue, stop_event),
            daemon=True
        )
        reader_thread.start()
        
        # Даем время на запуск
        await asyncio.sleep(0.5)
        
        print("📤 Starting to send audio...")
        bytes_sent = 0
        chunk_count = 0
        start_time = time.time()
        
        # Статистика для отладки
        last_stat_time = start_time
        stat_interval = 5.0  # секунд между выводами статистики
        
        try:
            while True:
                try:
                    # Берем данные из очереди
                    try:
                        data = audio_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Отладка: сохранение в файл
                    if DEBUG and debug_wav_file:
                        debug_wav_file.writeframes(data)
                    
                    # Анализ уровня сигнала (только в режиме отладки)
                    if DEBUG and chunk_count % 50 == 0:
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        max_val = np.max(np.abs(audio_data))
                        avg_val = np.mean(np.abs(audio_data))
                        print(f"📊 Chunk {chunk_count}: max={max_val}, avg={avg_val:.1f}")
                    
                    # Отправляем на сервер
                    await ws.send(data)
                    bytes_sent += len(data)
                    chunk_count += 1
                    
                    # Периодическая статистика (только в режиме отладки)
                    current_time = time.time()
                    if DEBUG and current_time - last_stat_time >= stat_interval:
                        elapsed = current_time - start_time
                        expected = int(16000 * 2 * elapsed)
                        buffer_status = (bytes_sent - expected) / 1024
                        
                        print(f"📈 Progress: {chunk_count} chunks, {bytes_sent/1024:.1f}KB, "
                              f"buffer: {buffer_status:+.1f}KB")
                        print(f"   - Real-time factor: {(bytes_sent/32000)/elapsed:.2f}x")
                        print(f"   - Chunks per second: {chunk_count/elapsed:.1f}")
                        
                        last_stat_time = current_time
                    
                    # Получаем ответ от сервера
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=0.001)
                        print(f"📥 Server: {resp}")
                    except asyncio.TimeoutError:
                        pass
                        
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Завершаем
            stop_event.set()
            reader_thread.join(timeout=1.0)
            
            # Закрываем файл отладки
            if DEBUG and debug_wav_file:
                debug_wav_file.close()
                print(f"💾 Debug audio saved to: {debug_filename}")
            
            # Финальная статистика (показываем всегда)
            elapsed = time.time() - start_time
            print(f"\n📊 Final statistics:")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Audio sent: {bytes_sent/32000:.1f} seconds")
            print(f"  Real-time factor: {(bytes_sent/32000)/elapsed:.2f}x")
            
            # Дополнительная статистика только в режиме отладки
            if DEBUG:
                print(f"  Chunks sent: {chunk_count}")
                print(f"  Bytes sent: {bytes_sent}")
                
                if debug_filename and os.path.exists(debug_filename):
                    file_size = os.path.getsize(debug_filename)
                    expected_header = 44  # Размер заголовка WAV
                    actual_audio_bytes = file_size - expected_header
                    print(f"  Debug file size: {file_size} bytes")
                    print(f"  Audio data in file: {actual_audio_bytes} bytes")
                    
                    if actual_audio_bytes != bytes_sent:
                        print(f"⚠  WARNING: File size mismatch! "
                              f"Expected {bytes_sent} bytes, got {actual_audio_bytes} bytes")

if __name__ == "__main__":
    print(f"🚀 ASR Stream Client")
    print(f"📡 Server: {SERVER_WS}")
    print(f"🔧 Debug mode: {'ENABLED' if DEBUG else 'DISABLED'}")
    print(f"📝 Usage: python3 {sys.argv[0]} [--debug] [--server WS_URL]")
    print("-" * 50)
    
    asyncio.run(stream_audio())
