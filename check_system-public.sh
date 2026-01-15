#!/bin/bash

# Скрипт проверки системы и создания установочного скрипта для Ubuntu 22.04.2
# Автоматически определяет зависимости и создает установщик

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Файлы
CHECK_REPORT="system_check_report.txt"
INSTALL_SCRIPT="install_dependencies_ubuntu_22.04.sh"
PYTHON_SCRIPT="whisper_pyannote_gpu_v2.py"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Очистка предыдущих файлов
rm -f $CHECK_REPORT $INSTALL_SCRIPT

# Начало отчета
echo "=== ОТЧЕТ ПРОВЕРКИ СИСТЕМЫ ===" > $CHECK_REPORT
echo "Дата проверки: $(date)" >> $CHECK_REPORT
echo "=========================================" >> $CHECK_REPORT

print_status "Начинаю проверку системы..."

# =========================================
# 1. Проверка версии Ubuntu
# =========================================
print_status "Проверяю версию Ubuntu..."

if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION="$VERSION_ID"
    UBUNTU_CODENAME="$VERSION_CODENAME"
    UBUNTU_NAME="$PRETTY_NAME"
    
    echo "ОС: $UBUNTU_NAME" >> $CHECK_REPORT
    echo "Версия: $UBUNTU_VERSION" >> $CHECK_REPORT
    echo "Кодовое имя: $UBUNTU_CODENAME" >> $CHECK_REPORT
    
    if [ "$UBUNTU_VERSION" = "22.04" ]; then
        print_success "Ubuntu 22.04 обнаружена"
        echo "Статус: Совместима ✓" >> $CHECK_REPORT
    else
        print_warning "Обнаружена Ubuntu $UBUNTU_VERSION (ожидается 22.04)"
        echo "Статус: Несовместима (требуется 22.04) ⚠" >> $CHECK_REPORT
    fi
else
    print_error "Не удалось определить дистрибутив"
    echo "ОС: Не определено" >> $CHECK_REPORT
    echo "Статус: Ошибка проверки ✗" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 2. Проверка архитектуры процессора
# =========================================
print_status "Проверяю архитектуру процессора..."

ARCH=$(uname -m)
echo "Архитектура: $ARCH" >> $CHECK_REPORT

if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then
    print_success "Архитектура $ARCH поддерживается"
    echo "Статус: Поддерживается ✓" >> $CHECK_REPORT
else
    print_warning "Архитектура $ARCH может иметь ограниченную поддержку"
    echo "Статус: Проверьте совместимость ⚠" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 3. Проверка версии Python
# =========================================
print_status "Проверяю версию Python..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_PATH=$(which python3)
    echo "Python версия: $PYTHON_VERSION" >> $CHECK_REPORT
    echo "Путь: $PYTHON_PATH" >> $CHECK_REPORT
    
    # Проверяем версию Python
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python 3.8+ обнаружен"
        echo "Статус: Совместим ✓" >> $CHECK_REPORT
    else
        print_warning "Python $PYTHON_VERSION обнаружен (рекомендуется 3.8+)"
        echo "Статус: Проверьте совместимость ⚠" >> $CHECK_REPORT
    fi
    
    # Проверяем pip
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | awk '{print $2}')
        echo "Pip версия: $PIP_VERSION" >> $CHECK_REPORT
        echo "Статус pip: Установлен ✓" >> $CHECK_REPORT
    else
        print_warning "pip3 не найден"
        echo "Статус pip: Не установлен ⚠" >> $CHECK_REPORT
    fi
else
    print_error "Python3 не найден"
    echo "Python версия: Не найден" >> $CHECK_REPORT
    echo "Статус: Требуется установка ✗" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 4. Проверка GPU и CUDA
# =========================================
print_status "Проверяю GPU и CUDA..."

# Проверяем наличие nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA драйверы обнаружены"
    echo "NVIDIA драйверы: Обнаружены ✓" >> $CHECK_REPORT
    
    # Получаем информацию о GPU
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ ! -z "$GPU_INFO" ]; then
        GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1)
        DRIVER_VERSION=$(echo $GPU_INFO | cut -d',' -f2)
        GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f3)
        
        echo "Модель GPU: $GPU_NAME" >> $CHECK_REPORT
        echo "Версия драйвера: $DRIVER_VERSION" >> $CHECK_REPORT
        echo "Память GPU: ${GPU_MEMORY}MB" >> $CHECK_REPORT
    fi
    
    # Проверяем CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        echo "CUDA версия: $CUDA_VERSION" >> $CHECK_REPORT
        echo "Статус CUDA: Установлена ✓" >> $CHECK_REPORT
        
        # Проверяем совместимость с PyTorch
        if [ ! -z "$CUDA_VERSION" ]; then
            CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
            if [ "$CUDA_MAJOR" -ge 11 ]; then
                print_success "CUDA 11.x+ обнаружена, совместима с PyTorch"
                echo "Совместимость с PyTorch: Да ✓" >> $CHECK_REPORT
            else
                print_warning "CUDA $CUDA_VERSION обнаружена (рекомендуется 11.8+)"
                echo "Совместимость с PyTorch: Проверьте версию ⚠" >> $CHECK_REPORT
            fi
        fi
    else
        print_warning "CUDA Toolkit не найден"
        echo "CUDA версия: Не найдена" >> $CHECK_REPORT
        echo "Статус CUDA: Требуется установка ⚠" >> $CHECK_REPORT
    fi
    
    # Проверяем cuDNN
    if [ -f /usr/local/cuda/include/cudnn_version.h ] || [ -f /usr/include/cudnn_version.h ]; then
        echo "cuDNN: Обнаружена ✓" >> $CHECK_REPORT
    else
        print_warning "cuDNN не найдена"
        echo "cuDNN: Не найдена ⚠" >> $CHECK_REPORT
    fi
else
    print_warning "NVIDIA GPU не обнаружена или драйверы не установлены"
    echo "NVIDIA GPU: Не обнаружена" >> $CHECK_REPORT
    echo "Статус: Будет использоваться CPU" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 5. Проверка системных пакетов
# =========================================
print_status "Проверяю системные пакеты..."

REQUIRED_PACKAGES=(
    "ffmpeg"
    "git"
    "python3-pip"
    "python3-dev"
    "python3-venv"
    "build-essential"
    "cmake"
    "libssl-dev"
    "zlib1g-dev"
    "libbz2-dev"
    "libreadline-dev"
    "libsqlite3-dev"
    "libncursesw5-dev"
    "xz-utils"
    "tk-dev"
    "libxml2-dev"
    "libxmlsec1-dev"
    "libffi-dev"
    "liblzma-dev"
    "portaudio19-dev"
)

echo "Требуемые системные пакеты:" >> $CHECK_REPORT
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii  $pkg "; then
        echo "  $pkg: Установлен ✓" >> $CHECK_REPORT
    else
        echo "  $pkg: Отсутствует ✗" >> $CHECK_REPORT
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    print_success "Все системные пакеты установлены"
    echo "Статус системных пакетов: Все установлены ✓" >> $CHECK_REPORT
else
    print_warning "Отсутствуют пакеты: ${MISSING_PACKAGES[*]}"
    echo "Статус системных пакетов: Отсутствуют ${#MISSING_PACKAGES[@]} пакетов ⚠" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 6. Проверка Python пакетов
# =========================================
print_status "Проверяю Python пакеты..."

REQUIRED_PYTHON_PACKAGES=(
    "torch"
    "torchaudio"
    "openai-whisper"
    "pyannote.audio"
    "numpy"
    "huggingface-hub"
    "dataclasses"
    "argparse"
    "scipy"
    "librosa"
    "soundfile"
    "pydub"
    "tqdm"
    "colorama"
)

echo "Требуемые Python пакеты:" >> $CHECK_REPORT
MISSING_PYTHON_PACKAGES=()

# Проверяем каждый пакет
for pkg in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'версия не определена'))" 2>/dev/null || echo "версия не определена")
        echo "  $pkg: $VERSION ✓" >> $CHECK_REPORT
    else
        echo "  $pkg: Отсутствует ✗" >> $CHECK_REPORT
        MISSING_PYTHON_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PYTHON_PACKAGES[@]} -eq 0 ]; then
    print_success "Все Python пакеты установлены"
    echo "Статус Python пакетов: Все установлены ✓" >> $CHECK_REPORT
else
    print_warning "Отсутствуют Python пакеты: ${MISSING_PYTHON_PACKAGES[*]}"
    echo "Статус Python пакетов: Отсутствуют ${#MISSING_PYTHON_PACKAGES[@]} пакетов ⚠" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 7. Проверка памяти и диска
# =========================================
print_status "Проверяю системные ресурсы..."

# Память
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -m | awk '/^Mem:/{print $7}')
echo "Общая память: ${TOTAL_MEM}MB" >> $CHECK_REPORT
echo "Доступная память: ${AVAILABLE_MEM}MB" >> $CHECK_REPORT

if [ $TOTAL_MEM -lt 8000 ]; then
    print_warning "Мало оперативной памяти (рекомендуется 8GB+)"
    echo "Статус памяти: Минимум 8GB рекомендуется ⚠" >> $CHECK_REPORT
else
    print_success "Оперативная памяти достаточно"
    echo "Статус памяти: Достаточно ✓" >> $CHECK_REPORT
fi

# Диск
TOTAL_DISK=$(df -h / | awk 'NR==2 {print $2}')
AVAILABLE_DISK=$(df -h / | awk 'NR==2 {print $4}')
echo "Общий диск: $TOTAL_DISK" >> $CHECK_REPORT
echo "Доступно на диске: $AVAILABLE_DISK" >> $CHECK_REPORT

# Место для моделей
if [ -d ~/.cache ]; then
    CACHE_SIZE=$(du -sh ~/.cache 2>/dev/null | cut -f1)
    echo "Размер кэша: $CACHE_SIZE" >> $CHECK_REPORT
fi

echo "" >> $CHECK_REPORT

# =========================================
# 8. Создание скрипта установки
# =========================================
print_status "Создаю скрипт установки..."

cat > $INSTALL_SCRIPT << 'EOF'
#!/bin/bash

# Скрипт автоматической установки зависимостей для Whisper + PyAnnote
# Для Ubuntu 22.04.2
# Создан автоматически на основе проверки системы

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка на sudo
if [ "$EUID" -ne 0 ]; then 
    print_status "Запрос прав sudo для установки системных пакетов..."
    sudo echo "Права получены" || {
        print_error "Не удалось получить права sudo"
        exit 1
    }
fi

# Начало установки
echo "========================================="
echo "Установка зависимостей для Whisper + PyAnnote"
echo "Для Ubuntu 22.04.2"
echo "========================================="

# =========================================
# 1. Обновление системы
# =========================================
print_status "Обновляю систему..."
sudo apt-get update
sudo apt-get upgrade -y

# =========================================
# 2. Установка системных зависимостей
# =========================================
print_status "Устанавливаю системные зависимости..."
sudo apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    portaudio19-dev \
    sox \
    libsox-fmt-all

# =========================================
# 3. Установка CUDA (если есть NVIDIA GPU)
# =========================================
if command -v nvidia-smi &> /dev/null; then
    print_status "Обнаружена NVIDIA GPU, проверяю CUDA..."
    
    if ! command -v nvcc &> /dev/null; then
        print_status "Устанавливаю CUDA Toolkit 11.8..."
        
        # Добавляем репозиторий NVIDIA
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        
        # Устанавливаем CUDA 11.8
        sudo apt-get install -y cuda-toolkit-11-8
        
        # Добавляем CUDA в PATH
        echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
        
        source ~/.bashrc
        print_success "CUDA Toolkit 11.8 установлена"
    else
        print_success "CUDA уже установлена"
    fi
    
    # Установка cuDNN
    print_status "Проверяю cuDNN..."
    if [ ! -f /usr/local/cuda/include/cudnn_version.h ] && [ ! -f /usr/include/cudnn_version.h ]; then
        print_warning "cuDNN не найдена. Для максимальной производительности установите cuDNN вручную:"
        print_warning "Скачайте с https://developer.nvidia.com/cudnn"
        print_warning "И следуйте инструкциям установки"
    else
        print_success "cuDNN обнаружена"
    fi
else
    print_status "NVIDIA GPU не обнаружена, пропускаю установку CUDA"
fi

# =========================================
# 4. Создание виртуального окружения Python
# =========================================
print_status "Создаю виртуальное окружение Python..."
python3 -m venv whisper_env
source whisper_env/bin/activate

# =========================================
# 5. Обновление pip
# =========================================
print_status "Обновляю pip и setuptools..."
pip install --upgrade pip setuptools wheel

# =========================================
# 6. Установка PyTorch с поддержкой CUDA или CPU
# =========================================
if command -v nvidia-smi &> /dev/null && command -v nvcc &> /dev/null; then
    print_status "Устанавливаю PyTorch с поддержкой CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "Устанавливаю PyTorch для CPU..."
    pip install torch torchvision torchaudio
fi

# =========================================
# 7. Установка основных Python пакетов
# =========================================
print_status "Устанавливаю основные зависимости..."
pip install numpy scipy pandas matplotlib tqdm colorama

# =========================================
# 8. Установка Whisper
# =========================================
print_status "Устанавливаю Whisper..."
pip install git+https://github.com/openai/whisper.git

# =========================================
# 9. Установка PyAnnote и зависимостей
# =========================================
print_status "Устанавливаю PyAnnote и аудио-библиотеки..."
pip install pyannote.audio
pip install librosa soundfile pydub ffmpeg-python

# =========================================
# 10. Установка дополнительных зависимостей
# =========================================
print_status "Устанавливаю дополнительные зависимости..."
pip install \
    huggingface-hub \
    transformers \
    dataclasses \
    typing-extensions \
    python-dotenv \
    sentencepiece \
    protobuf \
    onnxruntime

# =========================================
# 11. Проверка установки
# =========================================
print_status "Проверяю установку..."
echo "Проверка PyTorch и CUDA:"
python3 -c "
import torch
print(f'PyTorch версия: {torch.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Количество GPU: {torch.cuda.device_count()}')
    print(f'Имя GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Проверка Whisper:"
python3 -c "
import whisper
print(f'Whisper доступен: {whisper.__version__}')
"

echo ""
echo "Проверка PyAnnote:"
python3 -c "
try:
    from pyannote.audio import Pipeline
    print('PyAnnote доступен')
except ImportError as e:
    print(f'Ошибка импорта PyAnnote: {e}')
"

# =========================================
# 12. Создание скрипта активации
# =========================================
cat > activate_whisper.sh << 'ACTIVATE_EOF'
#!/bin/bash
echo "Активация окружения Whisper..."
source whisper_env/bin/activate
echo "Окружение активировано"
echo ""
echo "Пример использования:"
echo "python whisper_pyannote_gpu_v2.py audio.wav --hf-token YOUR_TOKEN"
echo ""
echo "Для получения токена HuggingFace:"
echo "1. Зарегистрируйтесь на https://huggingface.co"
echo "2. Перейдите в https://huggingface.co/settings/tokens"
echo "3. Создайте новый токен с правами read"
echo ""
echo "Для деактивации окружения выполните: deactivate"
ACTIVATE_EOF

chmod +x activate_whisper.sh

# =========================================
# 13. Завершение
# =========================================
print_success "Установка завершена!"
echo ""
echo "Для активации виртуального окружения выполните:"
echo "source whisper_env/bin/activate"
echo "Или запустите: ./activate_whisper.sh"
echo ""
echo "Для работы с PyAnnote необходим токен HuggingFace:"
echo "1. Зарегистрируйтесь на https://huggingface.co"
echo "2. Создайте токен на https://huggingface.co/settings/tokens"
echo "3. Примите условия использования моделей:"
echo "   - https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""
echo "Пример запуска:"
echo "python whisper_pyannote_gpu_v2.py audio.mp3 --hf-token YOUR_TOKEN"
echo ""
echo "========================================="
EOF

# Делаем скрипт исполняемым
chmod +x $INSTALL_SCRIPT

print_success "Скрипт установки создан: $INSTALL_SCRIPT"

# =========================================
# 9. Создание файла требований для Python
# =========================================
print_status "Создаю requirements.txt..."

cat > requirements.txt << 'REQ_EOF'
# Основные зависимости
torch
torchaudio
torchvision

# Whisper
openai-whisper

# PyAnnote и аудио обработка
pyannote.audio
librosa
soundfile
pydub
ffmpeg-python

# Дополнительные
numpy
scipy
pandas
matplotlib
tqdm
colorama
huggingface-hub
transformers
dataclasses
typing-extensions
python-dotenv
sentencepiece
protobuf
onnxruntime

# Для работы с субтитрами
webvtt-py
REQ_EOF

print_success "Файл требований создан: requirements.txt"

# =========================================
# 10. Итоговый отчет
# =========================================
echo "=== ИТОГИ ПРОВЕРКИ ===" >> $CHECK_REPORT
echo "Созданы файлы:" >> $CHECK_REPORT
echo "1. $CHECK_REPORT - полный отчет о системе" >> $CHECK_REPORT
echo "2. $INSTALL_SCRIPT - скрипт установки зависимостей" >> $CHECK_REPORT
echo "3. requirements.txt - файл Python зависимостей" >> $CHECK_REPORT
echo "" >> $CHECK_REPORT

# Проверяем совместимость
COMPATIBLE=true
if [ "$UBUNTU_VERSION" != "22.04" ]; then
    echo "⚠ Предупреждение: Рекомендуется Ubuntu 22.04" >> $CHECK_REPORT
    COMPATIBLE=false
fi

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "⚠ Отсутствуют системные пакеты" >> $CHECK_REPORT
    COMPATIBLE=false
fi

if [ ${#MISSING_PYTHON_PACKAGES[@]} -gt 0 ]; then
    echo "⚠ Отсутствуют Python пакеты" >> $CHECK_REPORT
    COMPATIBLE=false
fi

if $COMPATIBLE; then
    echo "✅ Система в основном совместима" >> $CHECK_REPORT
    print_success "Система проверена. Запустите ./$INSTALL_SCRIPT для установки зависимостей"
else
    echo "⚠ Система требует доработки" >> $CHECK_REPORT
    print_warning "Обнаружены проблемы. Запустите ./$INSTALL_SCRIPT для исправления"
fi

echo "" >> $CHECK_REPORT
echo "Следующие шаги:" >> $CHECK_REPORT
echo "1. Просмотрите отчет: cat $CHECK_REPORT" >> $CHECK_REPORT
echo "2. Запустите установку: sudo ./$INSTALL_SCRIPT" >> $CHECK_REPORT
echo "3. Активируйте окружение: source whisper_env/bin/activate" >> $CHECK_REPORT
echo "4. Запустите программу: python $PYTHON_SCRIPT" >> $CHECK_REPORT

print_status "Проверка завершена. Отчет сохранен в $CHECK_REPORT"
print_status "Для установки зависимостей выполните: sudo ./$INSTALL_SCRIPT"

# Выводим краткую информацию
echo ""
echo "========================================="
echo "КРАТКИЙ ОТЧЕТ"
echo "========================================="
echo "ОС: $UBUNTU_NAME"
echo "Python: $PYTHON_VERSION"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: Обнаружена ($GPU_NAME)"
else
    echo "GPU: Не обнаружена (будет использоваться CPU)"
fi
echo "Отсутствует пакетов: ${#MISSING_PACKAGES[@]} системных, ${#MISSING_PYTHON_PACKAGES[@]} Python"
echo "========================================="


