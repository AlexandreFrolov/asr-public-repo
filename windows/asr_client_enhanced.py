# asr_client_enhanced.py
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import queue
import threading
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--server', default="ws://192.168.0.71:8000/ws/asr")
parser.add_argument('--device', type=int, default=None)
parser.add_argument('--list-devices', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-microphones', action='store_true')
args = parser.parse_args()

SAMPLE_RATE = 16000
CHUNK = 1600

def list_devices_with_details():
    """Показать устройства с подробной информацией"""
    print("📋 Доступные аудиоустройства:")
    print("-" * 60)
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"[{i}] {device['name']}")
            print(f"    Каналы: {device['max_input_channels']}, "
                  f"Частота: {device['default_samplerate']} Hz")
            print(f"    API: {device['hostapi']}")
            print("-" * 30)

def find_best_microphone():
    """Автоматически найти лучший микрофон"""
    devices = sd.query_devices()
    
    # Приоритеты: Intel массивы, затем Realtek, затем остальное
    priority_keywords = [
        ('intel', 3),
        ('array', 2),
        ('realtek', 1),
        ('mic', 1),
        ('microphone', 1)
    ]
    
    best_device = None
    best_score = 0
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            score = 0
            name_lower = device['name'].lower()
            
            for keyword, points in priority_keywords:
                if keyword in name_lower:
                    score += points
            
            # Предпочтение устройствам с 16000 Hz или выше
            if device['default_samplerate'] >= 16000:
                score += 1
            
            if score > best_score:
                best_score = score
                best_device = i
    
    return best_device if best_device is not None else sd.default.device[0]

def test_microphone_quality(device_id):
    """Протестировать качество микрофона"""
    print(f"\n🔊 Тестирование микрофона {device_id}...")
    
    test_duration = 2  # секунды
    audio_data = []
    
    def callback(indata, frames, time_info, status):
        audio_data.append(indata.copy())
    
    try:
        stream = sd.InputStream(
            device=device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=callback,
            blocksize=CHUNK,
            dtype='int16'
        )
        
        with stream:
            time.sleep(test_duration)
        
        if audio_data:
            audio_array = np.concatenate(audio_data)
            max_volume = np.max(np.abs(audio_array))
            avg_volume = np.mean(np.abs(audio_array))
            
            print(f"   Макс. уровень: {max_volume}")
            print(f"   Сред. уровень: {avg_volume:.1f}")
            print(f"   Качество: {'ХОРОШО' if max_volume > 1000 else 'СЛАБО'}")
            
            return max_volume
        else:
            print("   Нет данных")
            return 0
            
    except Exception as e:
        print(f"   Ошибка: {e}")
        return 0

async def stream_audio():
    async with websockets.connect(args.server) as ws:
        print("✅ Подключено к серверу")
        
        audio_queue = queue.Queue(maxsize=100)
        stop_event = threading.Event()
        
        def callback(indata, frames, time_info, status):
            if status and args.debug:
                print(f"Аудио статус: {status}")
            try:
                audio_queue.put(indata.copy())
            except queue.Full:
                if args.debug:
                    print("⚠ Очередь переполнена, пропускаем данные")
        
        # Выбор устройства
        if args.device is None:
            args.device = find_best_microphone()
        
        device_info = sd.query_devices(args.device)
        print(f"🎤 Устройство: {device_info['name']} (ID: {args.device})")
        print(f"📊 Частота: {device_info['default_samplerate']} Hz")
        
        # Проверяем поддержку 16000 Hz
        if device_info['default_samplerate'] < 16000:
            print("⚠ Внимание: устройство может не поддерживать 16000 Hz")
        
        stream = sd.InputStream(
            device=args.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=callback,
            blocksize=CHUNK,
            dtype='int16'
        )
        
        with stream:
            print("🎤 Микрофон запущен")
            print("📤 Отправка аудио на сервер...")
            print("-" * 40)
            
            try:
                chunk_count = 0
                start_time = time.time()
                
                while True:
                    try:
                        # Получаем аудио из очереди
                        audio_data = audio_queue.get(timeout=0.1)
                        audio_bytes = audio_data.tobytes()
                        
                        # Отладочная информация
                        if args.debug and chunk_count % 50 == 0:
                            if len(audio_data) > 0:
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                                max_val = np.max(np.abs(audio_array))
                                avg_val = np.mean(np.abs(audio_array))
                                
                                # Индикатор уровня
                                level = min(int(max_val / 100), 20)
                                level_bar = "█" * level + "░" * (20 - level)
                                
                                print(f"📊 Чанк {chunk_count}: {level_bar} (max={max_val}, avg={avg_val:.1f})")
                        
                        # Отправляем на сервер
                        await ws.send(audio_bytes)
                        chunk_count += 1
                        
                        # Проверяем ответ от сервера
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=0.001)
                            print(f"📥 Сервер: {response}")
                        except asyncio.TimeoutError:
                            pass
                            
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        
            except KeyboardInterrupt:
                print("\n🛑 Остановка...")
            finally:
                stop_event.set()
                
                # Статистика
                elapsed = time.time() - start_time
                print(f"\n📈 Статистика:")
                print(f"   Отправлено чанков: {chunk_count}")
                print(f"   Время работы: {elapsed:.1f} сек")
                print(f"   Скорость: {chunk_count/elapsed:.1f} чанков/сек")

if __name__ == "__main__":
    print(f"🚀 Улучшенный ASR клиент")
    print(f"📡 Сервер: {args.server}")
    
    if args.test_microphones:
        print("\n🧪 Тестирование микрофонов:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"\n[{i}] {device['name']}")
                test_microphone_quality(i)
        exit(0)
    
    if args.list_devices:
        list_devices_with_details()
        
        # Показать рекомендуемое устройство
        best = find_best_microphone()
        if best is not None:
            device_info = sd.query_devices(best)
            print(f"\n💡 Рекомендуемое устройство: {best} - {device_info['name']}")
        exit(0)
    
    print(f"🔧 Режим отладки: {'ВКЛ' if args.debug else 'ВЫКЛ'}")
    print("-" * 50)
    
    try:
        asyncio.run(stream_audio())
    except KeyboardInterrupt:
        print("\n👋 Завершено")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
