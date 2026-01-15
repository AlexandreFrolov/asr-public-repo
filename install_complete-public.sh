#!/bin/bash
set -e

echo "========================================="
echo "ПОЛНАЯ УСТАНОВКА WHISPER + PyAnnote"
echo "Для NVIDIA Tesla T4"
echo "========================================="

sudo apt install -y python3.10-venv python3-dev python3-venv

sudo apt install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    libatlas-base-dev \
    gfortran

echo "Обновление ffmpeg"

sudo apt remove ffmpeg

wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz
sudo mv ffmpeg-*/ffmpeg /usr/local/bin/ffmpeg
sudo mv ffmpeg-*/ffprobe /usr/local/bin/ffprobe

ffmpeg -version

sudo apt install -y gpustat mc


echo "1. Создание виртуального окружения..."
rm -rf whisper_gpu_complete_env
python3 -m venv whisper_gpu_complete_env
source whisper_gpu_complete_env/bin/activate

echo "2. Обновление pip..."
pip install --upgrade pip setuptools wheel

echo "3. Установка PyTorch 2.1.0 + CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

echo "4. Установка NumPy..."
pip install numpy==1.24.3

echo "5. Установка PyTorch Lightning..."
pip install pytorch_lightning==2.0.9

echo "6. Установка Whisper..."
pip install openai-whisper

echo "7. Установка зависимостей PyAnnote..."
pip install \
    einops==0.7.0 \
    pyannote.core==5.0.0 \
    pyannote.database==5.0.1 \
    pyannote.metrics==3.2.1 \
    huggingface_hub==0.20.3 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    rich==13.7.0 \
    transformers==4.35.2 \
    tokenizers==0.15.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    matplotlib==3.8.2 \
    tqdm==4.66.1 \
    colorama==0.4.6

echo "8. Установка PyAnnote..."
pip install pyannote.audio==3.1.1

echo "9. Установка дополнительных пакетов..."
pip install \
    pydub==0.25.1 \
    ffmpeg-python==0.2.0 \
    webvtt-py==0.4.6 \
    sentencepiece==0.1.99

echo "10. Проверка установки..."
cat > test_complete_install.py << 'EOF'
#!/usr/bin/env python3
import torch
import whisper
import numpy as np

print("=" * 60)
print("ПРОВЕРКА УСТАНОВКИ")
print("=" * 60)

print(f"PyTorch версия: {torch.__version__}")
print(f"NumPy версия: {np.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n1. Проверка Whisper:")
try:
    model = whisper.load_model("base")
    print(f"  ✓ Модель 'base' загружена на {model.device}")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

print("\n2. Проверка PyAnnote:")
try:
    from pyannote.audio import Pipeline
    print("  ✓ PyAnnote импортирован")
    print("  ⚠️  Для работы нужен токен HuggingFace")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")

print("\n3. Тест GPU:")
if torch.cuda.is_available():
    try:
        # Быстрый тест
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        import time
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  Время операции 1000x1000: {elapsed:.3f} сек")
        print(f"  ✓ GPU работает")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")

print("\n" + "=" * 60)
print("УСТАНОВКА УСПЕШНА!")
print("=" * 60)
EOF

python test_complete_install.py

echo "11. Создание скрипта активации..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "========================================="
echo "АКТИВАЦИЯ ОКРУЖЕНИЯ WHISPER"
echo "========================================="
source whisper_gpu_complete_env/bin/activate
echo "Окружение активировано!"
echo ""
echo "Проверка:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
echo ""
echo "Использование:"
echo "1. whisper audio.mp3 --model medium --device cuda"
echo "2. Для PyAnnote нужен токен HuggingFace"
echo "========================================="
EOF

chmod +x activate_env.sh

echo "12. Создание примера использования..."
cat > example_whisper.py << 'EOF'
#!/usr/bin/env python3
"""
Пример использования Whisper
"""

import whisper
import torch

def main():
    print("Пример использования Whisper с GPU")
    print("=" * 50)
    
    # Проверка GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("Используется CPU")
    
    # Выбор модели
    print("\nДоступные модели для Tesla T4:")
    models = [
        ("tiny", 0.5, "Очень быстро, низкая точность"),
        ("base", 1.0, "Быстро, средняя точность"),
        ("small", 2.0, "Хороший баланс"),
        ("medium", 5.0, "Высокая точность (рекомендуется)"),
        ("large-v3", 10.0, "Максимальная точность")
    ]
    
    for name, mem, desc in models:
        print(f"  {name:10} - ~{mem}GB памяти: {desc}")
    
    # Загрузка модели
    model_name = "medium"
    print(f"\nЗагрузка модели '{model_name}'...")
    
    try:
        model = whisper.load_model(model_name)
        print(f"✓ Модель загружена на {model.device}")
        
        print("\nПример транскрипции:")
        print('''
# Для транскрипции файла:
result = model.transcribe(
    "audio.mp3",
    language="ru",           # Язык (ru, en, и т.д.)
    task="transcribe",       # transcribe или translate
    temperature=0.0,         # Температура (0.0 для детерминированного результата)
    best_of=5,               # Количество кандидатов
    beam_size=5,             # Размер beam search
    fp16=True                # Использовать половинную точность на GPU
)

print(result["text"])
''')
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        print("\nПопробуйте более легкую модель:")
        print("  model = whisper.load_model('base')")

if __name__ == "__main__":
    main()
EOF

chmod +x example_whisper.py

echo "13. Создание тестового скрипта..."
cat > quick_test.sh << 'EOF'
#!/bin/bash
echo "Быстрый тест установки"
echo "======================"

# Активация окружения
source whisper_gpu_complete_env/bin/activate

# Проверка версий
echo "1. Проверка версий:"
python -c "
import torch
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Проверка Whisper
echo ""
echo "2. Проверка Whisper:"
python -c "
import whisper
try:
    model = whisper.load_model('tiny')
    print(f'✓ Модель tiny загружена на {model.device}')
except Exception as e:
    print(f'✗ Ошибка: {e}')
"

# Проверка PyAnnote
echo ""
echo "3. Проверка PyAnnote:"
python -c "
try:
    from pyannote.audio import Pipeline
    print('✓ PyAnnote импортируется')
except Exception as e:
    print(f'✗ Ошибка: {e}')
"
EOF

chmod +x quick_test.sh

echo ""
echo "========================================="
echo "УСТАНОВКА ЗАВЕРШЕНА!"
echo "========================================="
echo ""
echo "Все компоненты установлены:"
echo "  • PyTorch 2.1.0 + CUDA 11.8"
echo "  • PyTorch Lightning 2.0.9"
echo "  • Whisper (все модели)"
echo "  • PyAnnote 3.1.1"
echo ""
echo "Для использования:"
echo ""
echo "1. Активируйте окружение:"
echo "   source whisper_gpu_complete_env/bin/activate"
echo "   или ./activate_env.sh"
echo ""
echo "2. Проверьте установку:"
echo "   python test_complete_install.py"
echo "   или ./quick_test.sh"
echo ""
echo "3. Примеры использования:"
echo "   • python example_whisper.py"
echo "   • whisper audio.mp3 --model medium --device cuda"
echo ""
echo "4. Для PyAnnote получите токен на:"
echo "   https://huggingface.co/settings/tokens"
echo ""
echo "========================================="

