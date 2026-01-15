#!/bin/bash

set -e

echo "Установка драйверов NVIDIA для Tesla T4"

echo "Очистка старых драйверов..."
sudo apt purge -y nvidia-* 2>/dev/null || true
sudo apt autoremove -y
sudo apt autoclean

echo "Установка nvidia-driver-535..."
sudo apt update
sudo apt install -y nvidia-driver-535

echo "Установка nvidia-utils-535..."
sudo apt install -y nvidia-utils-535

echo "Проверка установленных пакетов..."
dpkg -l | grep nvidia

echo "Загрузка модулей NVIDIA..."
sudo modprobe nvidia 2>/dev/null || true
sudo nvidia-modprobe 2>/dev/null || true

echo "Проверка nvidia-smi..."
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi найден: $(which nvidia-smi)"
    echo "Пробуем выполнить nvidia-smi..."
    timeout 5 nvidia-smi || echo "nvidia-smi не смог выполниться"
else
    echo "nvidia-smi не найден"
fi

echo ""
echo "=========================================="
echo "Драйверы установлены. Перезагрузите систему:"
echo "sudo reboot"
echo ""
echo "После перезагрузки проверьте:"
echo "nvidia-smi"
echo "=========================================="

