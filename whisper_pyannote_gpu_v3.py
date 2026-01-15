#!/usr/bin/env python3

import torch
import whisper
import os
import sys
import json
import gc
import warnings
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

# PyAnnote импорты
try:
    from pyannote.audio import Pipeline
    from huggingface_hub import HfFolder
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Предупреждение: pyannote.audio не установлен. Установите: pip install pyannote.audio")

# Подавление предупреждений
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# КОНФИГУРАЦИЯ И СТРУКТУРЫ ДАННЫХ
# ============================================================================

@dataclass
class DiarizationConfig:
    """Конфигурация параметров диаризации PyAnnote (совместимая с версией 3.1+)"""
    # Основные параметры
    min_speakers: Optional[int] = None      # Минимальное количество спикеров
    max_speakers: Optional[int] = None      # Максимальное количество спикеров
    
    # Параметры сегментации (VAD)
    segmentation_threshold: float = 0.55    # Порог сегментации (0.5-0.7)
    min_segment_duration: float = 0.1       # Минимальная длительность сегмента
    
    # Параметры кластеризации
    clustering_method: str = "centroid"     # Алгоритм кластеризации
    min_cluster_size: int = 15              # Мин. сегментов для кластера
    clustering_threshold: float = 0.65      # Порог кластеризации (0.5-0.9)
    
    # Дополнительные параметры
    step: float = 0.05                      # Шаг окна обработки (сек)
    
    # Кастомные параметры для разных сценариев
    scenario: str = "meeting"               # meeting, interview, phone, podcast
    
    def get_hyperparameters(self) -> Dict:
        """Возвращает параметры в формате PyAnnote 3.1+"""
        params = {}
        
        # Добавляем только заданные параметры
        if self.min_speakers is not None:
            params["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            params["max_speakers"] = self.max_speakers
        
        # Параметры сегментации (VAD)
        params["segmentation_threshold"] = self.segmentation_threshold
        
        # Параметры кластеризации
        params["clustering"] = {
            "method": self.clustering_method,
            "min_cluster_size": self.min_cluster_size,
            "threshold": self.clustering_threshold,
        }
        
        # Дополнительные параметры
        params["step"] = self.step
        
        return params
    
    @classmethod
    def for_scenario(cls, scenario: str) -> "DiarizationConfig":
        """Предустановки для разных сценариев"""
        presets = {
            "meeting": cls(
                min_speakers=2,
                max_speakers=10,
                segmentation_threshold=0.58,
                min_cluster_size=12,
                clustering_threshold=0.68,
            ),
            "interview": cls(
                min_speakers=1,
                max_speakers=3,
                segmentation_threshold=0.55,
                min_cluster_size=8,
                clustering_threshold=0.70,
            ),
            "phone": cls(
                min_speakers=1,
                max_speakers=2,
                segmentation_threshold=0.52,
                min_cluster_size=20,
                clustering_threshold=0.62,
            ),
            "podcast": cls(
                min_speakers=1,
                max_speakers=4,
                segmentation_threshold=0.60,
                min_cluster_size=10,
                clustering_threshold=0.75,
            ),
            "fast": cls(
                segmentation_threshold=0.60,
                step=0.10,
                min_cluster_size=20,
            ),
        }
        return presets.get(scenario, cls())

@dataclass
class WhisperConfig:
    """Конфигурация параметров Whisper"""
    model_size: str = "base"              # tiny, base, small, medium, large-v3
    language: str = "ru"                  # Язык распознавания
    task: str = "transcribe"              # transcribe или translate
    temperature: float = 0.0              # Температура генерации
    best_of: int = 5                      # Количество кандидатов
    beam_size: int = 5                    # Размер beam search
    patience: float = 1.0                 # Патience для beam search
    word_timestamps: bool = True          # Таймстампы слов
    fp16: bool = True                     # Использовать половинную точность
    initial_prompt: Optional[str] = None  # Промпт для контекста
    compression_ratio_threshold: float = 2.4  # Порог компрессии
    logprob_threshold: float = -1.0       # Порог логарифмической вероятности
    no_speech_threshold: float = 0.6      # Порог отсутствия речи
    
    # Дополнительные параметры
    suppress_tokens: str = "-1"           # Подавляемые токены
    condition_on_previous_text: bool = True  # Учитывать предыдущий текст

@dataclass
class AlignmentConfig:
    """Конфигурация совмещения диаризации и транскрипции"""
    overlap_threshold: float = 0.3        # Мин. пересечение для привязки (30%)
    max_gap: float = 2.0                  # Макс. разрыв для поиска (сек)
    boundary_tolerance: float = 1.0       # Допуск по границам (сек)
    min_segment_duration: float = 0.1     # Мин. длительность сегмента (сек)
    
    # Параметры постобработки
    min_speaker_duration: float = 1.0     # Мин. общая длительность спикера
    merge_same_speaker_gap: float = 1.0   # Макс. разрыв для объединения
    fix_short_segments: bool = True       # Исправлять короткие сегменты
    short_segment_threshold: float = 0.3  # Порог короткого сегмента (сек)

@dataclass 
class ProcessingConfig:
    """Общая конфигурация обработки"""
    device: str = "cuda"                  # Устройство для обработки
    num_workers: int = 1                  # Количество воркеров
    batch_size: int = 16                  # Размер батча
    chunk_duration: int = 600             # Длительность чанка для больших файлов (сек)
    temp_dir: str = "./temp"              # Временная директория
    output_format: str = "txt"            # Формат вывода: txt, json, srt, vtt
    save_intermediate: bool = False       # Сохранять промежуточные результаты

# ============================================================================
# УЛУЧШЕННАЯ ДИАРИЗАЦИЯ (СОВМЕСТИМАЯ С ВЕРСИЕЙ 3.1+)
# ============================================================================

class AdvancedDiarizer:
    """Улучшенный диаризатор с поддержкой разных сценариев"""
    
    def __init__(self, hf_token: str, config: DiarizationConfig, enabled: bool = True):
        """
        Инициализация диаризатора
        
        Args:
            hf_token: Токен HuggingFace
            config: Конфигурация диаризации
            enabled: Включена ли диаризация
        """
        self.enabled = enabled
        if not enabled:
            print("Диаризация отключена через параметры запуска")
            return
        
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio не установлен")
        
        self.config = config
        self.hf_token = hf_token
        
        # Устанавливаем токен
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        try:
            HfFolder.save_token(hf_token)
        except:
            pass
        
        # Инициализация пайплайна
        print(f"Загрузка модели диаризации для сценария '{config.scenario}'...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)
        
        # Пробуем применить параметры
        self._apply_hyperparameters()
        
        print(f"Модель загружена на {self.device}")
    
    def _apply_hyperparameters(self):
        """Применение гиперпараметров с обработкой ошибок"""
        if not self.enabled:
            return
            
        hyperparams = self.config.get_hyperparameters()
        
        if not hyperparams:
            print("Предупреждение: нет гиперпараметров для настройки")
            return
        
        print(f"Применение гиперпараметров: {list(hyperparams.keys())}")
        
        try:
            # Пробуем применить через instantiate
            self.pipeline.instantiate(hyperparams)
            print("✓ Гиперпараметры успешно применены")
        except Exception as e:
            print(f"⚠ Не удалось применить гиперпараметры: {e}")
            print("  Используются параметры по умолчанию")
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Выполнение диаризации аудиофайла
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Список сегментов диаризации
        """
        if not self.enabled:
            print("Диаризация отключена, возвращаем пустой список")
            return []
            
        print(f"Диаризация файла: {os.path.basename(audio_path)}")
        
        try:
            # Применяем дополнительные параметры через kwargs
            diarization_kwargs = {}
            
            if self.config.min_speakers is not None:
                diarization_kwargs["min_speakers"] = self.config.min_speakers
            if self.config.max_speakers is not None:
                diarization_kwargs["max_speakers"] = self.config.max_speakers
            
            # Выполняем диаризацию
            if diarization_kwargs:
                print(f"Дополнительные параметры: {diarization_kwargs}")
                diarization = self.pipeline(audio_path, **diarization_kwargs)
            else:
                diarization = self.pipeline(audio_path)
            
            # Конвертация в удобный формат
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'speaker': speaker,
                    'duration': float(turn.end - turn.start)
                })
            
            # Сортировка по времени начала
            segments.sort(key=lambda x: x['start'])
            
            print(f"✓ Обнаружено {len(segments)} сегментов диаризации")
            
            # Анализ спикеров
            speakers = set(s['speaker'] for s in segments)
            print(f"✓ Обнаружено {len(speakers)} спикеров: {', '.join(sorted(speakers))}")
            
            return segments
            
        except Exception as e:
            print(f"✗ Ошибка диаризации: {e}")
            raise
    
    def analyze_diarization(self, segments: List[Dict]) -> Dict:
        """Анализ результатов диаризации"""
        if not self.enabled:
            return {'diarization_enabled': False}
            
        if not segments:
            return {'diarization_enabled': True, 'segments_found': 0}
        
        # Статистика по спикерам
        speaker_stats = {}
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['duration']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'avg_duration': 0.0,
                }
            
            stats = speaker_stats[speaker]
            stats['total_duration'] += duration
            stats['segment_count'] += 1
        
        # Вычисляем среднюю длительность
        for speaker, stats in speaker_stats.items():
            if stats['segment_count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['segment_count']
        
        # Общая статистика
        total_duration = sum(seg['duration'] for seg in segments)
        avg_segment_duration = total_duration / len(segments) if segments else 0
        
        return {
            'diarization_enabled': True,
            'total_segments': len(segments),
            'total_duration': total_duration,
            'avg_segment_duration': avg_segment_duration,
            'speaker_count': len(speaker_stats),
            'speaker_stats': speaker_stats,
        }

# ============================================================================
# УЛУЧШЕННАЯ ТРАНСКРИПЦИЯ WHISPER
# ============================================================================

class AdvancedWhisperTranscriber:
    """Улучшенный транскрайбер с поддержкой промптов и оптимизаций"""
    
    def __init__(self, config: WhisperConfig):
        """
        Инициализация транскрайбера
        
        Args:
            config: Конфигурация Whisper
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.fp16 else "cpu"
        
        # Создаем директорию для кэша моделей
        os.makedirs("./whisper_models", exist_ok=True)
        
        print(f"Загрузка модели Whisper {config.model_size} на {self.device}...")
        
        # Проверяем доступность памяти для больших моделей
        if config.model_size in ["medium", "large", "large-v2", "large-v3"]:
            if self.device == "cuda":
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                required_memory = 5e9 if config.model_size == "medium" else 10e9
                if free_memory < required_memory:
                    print(f"⚠ Может не хватить памяти для модели {config.model_size}")
                    print(f"  Свободно: {free_memory/1e9:.1f}GB, Требуется: ~{required_memory/1e9:.1f}GB")
        
        # Загружаем модель
        self.model = whisper.load_model(
            config.model_size,
            device=self.device,
            download_root="./whisper_models"
        )
        
        # Оптимизация для GPU
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.empty_cache()
    
    def transcribe(self, audio_path: str, **kwargs) -> List[Dict]:
        """
        Транскрипция аудиофайла
        
        Args:
            audio_path: Путь к аудиофайлу
            **kwargs: Дополнительные параметры
            
        Returns:
            Список сегментов транскрипции
        """
        print(f"Транскрипция файла: {os.path.basename(audio_path)}")
        
        # Объединяем конфигурацию с переданными параметрами
        transcribe_params = {
            "language": self.config.language,
            "task": self.config.task,
            "temperature": self.config.temperature,
            "best_of": self.config.best_of,
            "beam_size": self.config.beam_size,
            "patience": self.config.patience,
            "word_timestamps": self.config.word_timestamps,
            "fp16": self.config.fp16,
            "initial_prompt": self.config.initial_prompt,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "logprob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "condition_on_previous_text": self.config.condition_on_previous_text,
        }
        
        # Добавляем suppress_tokens если задано
        if self.config.suppress_tokens:
            try:
                transcribe_params["suppress_tokens"] = [
                    int(tok) for tok in self.config.suppress_tokens.split(",")
                ]
            except:
                pass
        
        # Обновляем переданными параметрами
        transcribe_params.update(kwargs)
        
        # Выполняем транскрипцию
        print("Запуск распознавания...")
        result = self.model.transcribe(audio_path, **transcribe_params)
        
        # Форматируем результат
        segments = []
        for seg in result['segments']:
            segments.append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'].strip(),
                'confidence': seg.get('avg_logprob', 0.0) if 'avg_logprob' in seg else None,
                'words': seg.get('words', [])
            })
        
        print(f"✓ Распознано {len(segments)} сегментов транскрипции")
        return segments
    
    def cleanup(self):
        """Очистка памяти"""
        if hasattr(self, 'model'):
            del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# ============================================================================
# ИНТЕЛЛЕКТУАЛЬНОЕ СОВМЕЩЕНИЕ И ПОСТОБРАБОТКА
# ============================================================================

class IntelligentAligner:
    """Интеллектуальное совмещение диаризации и транскрипции"""
    
    def __init__(self, config: AlignmentConfig, diarization_enabled: bool = True):
        """
        Инициализация аллайнера
        
        Args:
            config: Конфигурация совмещения
            diarization_enabled: Включена ли диаризация
        """
        self.config = config
        self.diarization_enabled = diarization_enabled
    
    def align(self, whisper_segments: List[Dict], 
              diarization_segments: List[Dict]) -> List[Dict]:
        """
        Совмещение сегментов транскрипции и диаризации
        
        Args:
            whisper_segments: Сегменты транскрипции Whisper
            diarization_segments: Сегменты диаризации
            
        Returns:
            Совмещенные сегменты с информацией о спикере
        """
        print("Совмещение транскрипции и диаризации...")
        
        if not self.diarization_enabled:
            print("Диаризация отключена, присваиваем всем сегментам одного спикера")
            for seg in whisper_segments:
                seg['speaker'] = "SPEAKER_01"
                seg['alignment_method'] = 'diarization_disabled'
            return whisper_segments
            
        if not diarization_segments:
            print("⚠ Нет данных диаризации, используем заглушку")
            for seg in whisper_segments:
                seg['speaker'] = "SPEAKER_01"
                seg['alignment_method'] = 'no_diarization'
            return whisper_segments
        
        # Сортируем сегменты диаризации
        diarization_segments.sort(key=lambda x: x['start'])
        
        aligned_segments = []
        diar_idx = 0
        total_whisper = len(whisper_segments)
        
        for w_idx, w_seg in enumerate(whisper_segments):
            w_start = w_seg['start']
            w_end = w_seg['end']
            w_duration = w_end - w_start
            
            best_match = self._find_best_speaker_match(
                w_start, w_end, w_duration, 
                diarization_segments, diar_idx
            )
            
            # Обновляем индекс для оптимизации
            if best_match['diar_index'] > diar_idx:
                diar_idx = best_match['diar_index']
            
            # Создаем совмещенный сегмент
            aligned_seg = w_seg.copy()
            aligned_seg['speaker'] = best_match['speaker']
            aligned_seg['alignment_method'] = best_match['method']
            aligned_seg['alignment_confidence'] = best_match['confidence']
            
            if 'overlap_ratio' in best_match:
                aligned_seg['overlap_ratio'] = best_match['overlap_ratio']
            
            aligned_segments.append(aligned_seg)
            
            # Прогресс
            if (w_idx + 1) % 20 == 0 or (w_idx + 1) == total_whisper:
                print(f"  Совмещено {w_idx + 1}/{total_whisper} сегментов")
        
        print("✓ Совмещение завершено")
        return aligned_segments
    
    def _find_best_speaker_match(self, w_start: float, w_end: float, w_duration: float,
                                diar_segments: List[Dict], start_idx: int) -> Dict:
        """
        Поиск наилучшего соответствия спикера для сегмента Whisper
        """
        best_match = {
            'speaker': "SPEAKER_01",
            'method': 'fallback',
            'confidence': 0.0,
            'diar_index': start_idx
        }
        
        if not diar_segments:
            return best_match
        
        # Поиск пересечений
        max_overlap_ratio = 0
        best_overlap_idx = start_idx
        
        # Ищем в окрестности текущего индекса
        search_start = max(0, start_idx - 5)
        search_end = min(len(diar_segments), start_idx + 20)
        
        for i in range(search_start, search_end):
            d_seg = diar_segments[i]
            
            # Проверка на возможность пропуска
            if d_seg['end'] < w_start - self.config.max_gap:
                best_match['diar_index'] = i + 1
                continue
            
            if d_seg['start'] > w_end + self.config.max_gap:
                break
            
            # Вычисляем пересечение
            overlap_start = max(w_start, d_seg['start'])
            overlap_end = min(w_end, d_seg['end'])
            
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                overlap_ratio = overlap / w_duration if w_duration > 0 else 0
                
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    best_overlap_idx = i
        
        # Метод 1: Хорошее пересечение
        if max_overlap_ratio >= self.config.overlap_threshold:
            best_diar = diar_segments[best_overlap_idx]
            best_match.update({
                'speaker': best_diar['speaker'],
                'method': 'overlap',
                'confidence': max_overlap_ratio,
                'overlap_ratio': max_overlap_ratio,
                'diar_index': best_overlap_idx
            })
            return best_match
        
        # Метод 2: Проверка близости границ
        if max_overlap_ratio > 0.1 and best_overlap_idx < len(diar_segments):
            best_diar = diar_segments[best_overlap_idx]
            start_diff = abs(w_start - best_diar['start'])
            end_diff = abs(w_end - best_diar['end'])
            
            if start_diff < self.config.boundary_tolerance and end_diff < self.config.boundary_tolerance:
                best_match.update({
                    'speaker': best_diar['speaker'],
                    'method': 'boundary_match',
                    'confidence': 0.7,
                    'diar_index': best_overlap_idx
                })
                return best_match
        
        # Метод 3: Поиск ближайшего сегмента диаризации
        nearest_speaker = "SPEAKER_01"
        min_distance = float('inf')
        nearest_idx = start_idx
        
        for i in range(max(0, start_idx - 3), min(start_idx + 10, len(diar_segments))):
            d_seg = diar_segments[i]
            
            # Расстояние до границ
            dist_to_start = abs(w_start - d_seg['start'])
            dist_to_end = abs(w_start - d_seg['end'])
            distance = min(dist_to_start, dist_to_end)
            
            if distance < min_distance and distance < self.config.max_gap:
                min_distance = distance
                nearest_speaker = d_seg['speaker']
                nearest_idx = i
        
        if nearest_speaker != "SPEAKER_01":
            confidence = max(0.3, 1.0 - (min_distance / self.config.max_gap))
            best_match.update({
                'speaker': nearest_speaker,
                'method': 'nearest',
                'confidence': confidence,
                'diar_index': nearest_idx
            })
        
        return best_match
    
    def postprocess(self, aligned_segments: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Постобработка совмещенных сегментов
        
        Returns:
            (обработанные_сегменты, статистика)
        """
        if not self.diarization_enabled:
            # Если диаризация отключена, пропускаем постобработку
            return aligned_segments, {'diarization_enabled': False}
            
        if not self.config.fix_short_segments or len(aligned_segments) <= 1:
            return aligned_segments, {'diarization_enabled': True}
        
        print("Постобработка сегментов...")
        
        # 1. Исправление коротких сегментов
        processed_segments = aligned_segments.copy()
        
        for i in range(1, len(processed_segments) - 1):
            curr = processed_segments[i]
            prev = processed_segments[i - 1]
            next_seg = processed_segments[i + 1]
            
            curr_duration = curr['end'] - curr['start']
            
            # Если короткий сегмент зажат между одинаковыми спикерами
            if (curr_duration < self.config.short_segment_threshold and
                prev['speaker'] == next_seg['speaker'] and
                curr['speaker'] != prev['speaker']):
                
                # Проверяем расстояние до соседних сегментов
                gap_prev = curr['start'] - prev['end']
                gap_next = next_seg['start'] - curr['end']
                
                if gap_prev < 0.5 and gap_next < 0.5:
                    processed_segments[i]['speaker'] = prev['speaker']
                    processed_segments[i]['postprocessed'] = True
        
        # 2. Объединение последовательных сегментов одного спикера
        merged_segments = []
        current = None
        
        for seg in processed_segments:
            if current is None:
                current = seg.copy()
            elif (seg['speaker'] == current['speaker'] and 
                  seg['start'] - current['end'] < self.config.merge_same_speaker_gap):
                
                # Объединяем
                current['end'] = seg['end']
                current['text'] = current.get('text', '') + ' ' + seg['text'].strip()
                current['merged'] = current.get('merged', 1) + 1
            else:
                merged_segments.append(current)
                current = seg.copy()
        
        if current:
            merged_segments.append(current)
        
        # 3. Переименование спикеров по популярности
        speaker_durations = {}
        for seg in merged_segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # Сортируем спикеров по общей длительности
        sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        speaker_mapping = {}
        
        for i, (old_speaker, _) in enumerate(sorted_speakers):
            new_speaker = f"SPEAKER_{i+1:02d}"
            speaker_mapping[old_speaker] = new_speaker
        
        # Применяем переименование
        for seg in merged_segments:
            seg['speaker'] = speaker_mapping.get(seg['speaker'], seg['speaker'])
        
        # Обновляем статистику с новыми именами
        final_speaker_durations = {}
        for seg in merged_segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            final_speaker_durations[speaker] = final_speaker_durations.get(speaker, 0) + duration
        
        # 4. Фильтрация слишком коротких сегментов
        filtered_segments = [
            seg for seg in merged_segments 
            if (seg['end'] - seg['start']) >= self.config.min_segment_duration
        ]
        
        # Статистика
        stats = {
            'diarization_enabled': True,
            'original_segments': len(aligned_segments),
            'merged_segments': len(merged_segments),
            'final_segments': len(filtered_segments),
            'speaker_count': len(final_speaker_durations),
            'speaker_durations': final_speaker_durations,
            'total_duration': sum(final_speaker_durations.values()),
            'postprocessing_applied': len(aligned_segments) != len(filtered_segments)
        }
        
        print(f"✓ После постобработки: {stats['final_segments']} сегментов, "
              f"{stats['speaker_count']} спикеров")
        
        return filtered_segments, stats

# ============================================================================
# ЭКСПОРТ И ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================

class ResultExporter:
    """Экспорт результатов в различные форматы"""
    
    @staticmethod
    def export_text(aligned_segments: List[Dict], output_path: str, diarization_enabled: bool = True):
        """Экспорт в текстовый файл"""
        print(f"Экспорт в текстовый файл: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if not diarization_enabled:
                f.write(f"Диаризация отключена. Все реплики приписаны одному спикеру.\n\n")
            
            for seg in aligned_segments:
                start_str = f"{seg['start']:.2f}"
                end_str = f"{seg['end']:.2f}"
                f.write(f"[{start_str}-{end_str}] {seg['speaker']}: {seg.get('text', '')}\n")
    
    @staticmethod
    def export_json(aligned_segments: List[Dict], output_path: str, diarization_enabled: bool = True):
        """Экспорт в JSON"""
        print(f"Экспорт в JSON: {output_path}")
        
        output_data = {
            'segments': aligned_segments,
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'segment_count': len(aligned_segments),
                'speakers': list(set(seg['speaker'] for seg in aligned_segments)),
                'diarization_enabled': diarization_enabled
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def export_srt(aligned_segments: List[Dict], output_path: str, diarization_enabled: bool = True):
        """Экспорт в формат SRT (субтитры)"""
        print(f"Экспорт в SRT: {output_path}")
        
        def format_timestamp(seconds: float) -> str:
            """Форматирование времени для SRT"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if not diarization_enabled:
                f.write(f"1\n00:00:00,000 --> 00:00:05,000\nДиаризация отключена\n\n")
            
            for i, seg in enumerate(aligned_segments, 1):
                start_time = format_timestamp(seg['start'])
                end_time = format_timestamp(seg['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{seg['speaker']}] {seg.get('text', '')}\n\n")
    
    @staticmethod
    def export_vtt(aligned_segments: List[Dict], output_path: str, diarization_enabled: bool = True):
        """Экспорт в формат WebVTT"""
        print(f"Экспорт в VTT: {output_path}")
        
        def format_timestamp_vtt(seconds: float) -> str:
            """Форматирование времени для VTT"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            if not diarization_enabled:
                f.write(f"NOTE Диаризация отключена\n\n")
            
            for i, seg in enumerate(aligned_segments, 1):
                start_time = format_timestamp_vtt(seg['start'])
                end_time = format_timestamp_vtt(seg['end'])
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"<v {seg['speaker']}>{seg.get('text', '')}\n\n")
    
    @staticmethod
    def export_all(aligned_segments: List[Dict], base_output_path: str, 
                  stats: Optional[Dict] = None, diarization_enabled: bool = True):
        """Эспорт во все форматы"""
        base_name = os.path.splitext(base_output_path)[0]
        
        # Экспорт в разные форматы
        ResultExporter.export_text(aligned_segments, f"{base_name}.txt", diarization_enabled)
        ResultExporter.export_json(aligned_segments, f"{base_name}.json", diarization_enabled)
        ResultExporter.export_srt(aligned_segments, f"{base_name}.srt", diarization_enabled)
        ResultExporter.export_vtt(aligned_segments, f"{base_name}.vtt", diarization_enabled)
        
        # Сохранение статистики если есть
        if stats:
            stats_path = f"{base_name}_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"Статистика сохранена: {stats_path}")

# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция программы"""
    parser = argparse.ArgumentParser(
        description="Усовершенствованная диаризация и транскрипция аудио",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python whisper_pyannote_gpu_v2.py audio.wav --hf-token YOUR_TOKEN
  python whisper_pyannote_gpu_v2.py audio.wav --hf-token YOUR_TOKEN --whisper-model large-v3 --scenario interview
  python whisper_pyannote_gpu_v2.py audio.wav --hf-token YOUR_TOKEN --output-dir ./results --export-all
  python whisper_pyannote_gpu_v2.py audio.wav --hf-token YOUR_TOKEN --no-diarization
        """
    )
    
    # Обязательные аргументы
    parser.add_argument("audio_file", help="Путь к аудиофайлу")
    parser.add_argument("--hf-token", required=True, help="Токен HuggingFace")
    
    # Параметры диаризации
    parser.add_argument("--no-diarization", action="store_true",
                       help="Отключить диаризацию (все реплики будут от одного спикера)")
    parser.add_argument("--scenario", default="meeting",
                       choices=["meeting", "interview", "phone", "podcast", "fast"],
                       help="Сценарий аудио (влияет на параметры диаризации)")
    parser.add_argument("--min-speakers", type=int, default=None,
                       help="Минимальное количество спикеров")
    parser.add_argument("--max-speakers", type=int, default=None,
                       help="Максимальное количество спикеров")
    
    # Параметры Whisper
    parser.add_argument("--whisper-model", default="base",
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Модель Whisper для распознавания")
    parser.add_argument("--language", default="ru", help="Язык распознавания")
    parser.add_argument("--prompt", help="Промпт для Whisper с контекстом")
    parser.add_argument("--beam-size", type=int, default=5,
                       help="Размер beam search для Whisper")
    
    # Параметры вывода
    parser.add_argument("--output-dir", default="./results",
                       help="Директория для сохранения результатов")
    parser.add_argument("--export-all", action="store_true",
                       help="Экспортировать во все форматы")
    parser.add_argument("--no-postprocess", action="store_true",
                       help="Отключить постобработку")
    
    args = parser.parse_args()
    
    # Проверка файла
    if not os.path.exists(args.audio_file):
        print(f"✗ Ошибка: Файл '{args.audio_file}' не найден")
        sys.exit(1)
    
    # Создание директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("УСОВЕРШЕНСТВОВАННАЯ ДИАРИЗАЦИЯ И ТРАНСКРИПЦИЯ (v2.1)")
    print("=" * 70)
    
    # Определение устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Устройство: {device}")
    print(f"Аудиофайл: {args.audio_file}")
    print(f"Диаризация: {'ВЫКЛЮЧЕНА' if args.no_diarization else 'включена'}")
    if not args.no_diarization:
        print(f"Сценарий: {args.scenario}")
    print(f"Модель Whisper: {args.whisper_model}")
    print()
    
    try:
        # ====================================================================
        # 1. КОНФИГУРАЦИЯ
        # ====================================================================
        print("Этап 1: Конфигурация...")
        
        # Конфигурация диаризации
        diarization_config = DiarizationConfig.for_scenario(args.scenario)
        
        # Переопределение параметров если заданы
        if args.min_speakers is not None:
            diarization_config.min_speakers = args.min_speakers
        if args.max_speakers is not None:
            diarization_config.max_speakers = args.max_speakers
        
        # Конфигурация Whisper
        whisper_config = WhisperConfig(
            model_size=args.whisper_model,
            language=args.language,
            beam_size=args.beam_size,
            initial_prompt=args.prompt,
            fp16=(device == "cuda")
        )
        
        # Конфигурация совмещения
        alignment_config = AlignmentConfig(
            fix_short_segments=not args.no_postprocess
        )
        
        # ====================================================================
        # 2. ДИАРИЗАЦИЯ
        # ====================================================================
        print("\nЭтап 2: Диаризация...")
        
        diarization_segments = []
        diarization_stats = {'diarization_enabled': not args.no_diarization}
        
        if not args.no_diarization:
            try:
                diarizer = AdvancedDiarizer(args.hf_token, diarization_config, enabled=not args.no_diarization)
                diarization_segments = diarizer.diarize(args.audio_file)
                
                # Анализ результатов диаризации
                diarization_stats = diarizer.analyze_diarization(diarization_segments)
            except Exception as e:
                print(f"✗ Критическая ошибка диаризации: {e}")
                print("  Продолжаем без диаризации...")
                diarization_segments = []
                diarization_stats = {'diarization_enabled': False, 'error': str(e)}
        else:
            print("Диаризация отключена через параметры запуска")
        
        # ====================================================================
        # 3. ТРАНСКРИПЦИЯ
        # ====================================================================
        print("\nЭтап 3: Транскрипция...")
        
        transcriber = AdvancedWhisperTranscriber(whisper_config)
        whisper_segments = transcriber.transcribe(args.audio_file)
        
        # Очистка памяти Whisper
        transcriber.cleanup()
        
        # ====================================================================
        # 4. СОВМЕЩЕНИЕ И ПОСТОБРАБОТКА
        # ====================================================================
        print("\nЭтап 4: Совмещение и постобработка...")
        
        aligner = IntelligentAligner(alignment_config, diarization_enabled=not args.no_diarization)
        aligned_segments = aligner.align(whisper_segments, diarization_segments)
        
        # Постобработка
        if not args.no_postprocess and aligned_segments:
            final_segments, postprocess_stats = aligner.postprocess(aligned_segments)
        else:
            final_segments = aligned_segments
            postprocess_stats = {}
        
        # ====================================================================
        # 5. ЭКСПОРТ РЕЗУЛЬТАТОВ
        # ====================================================================
        print("\nЭтап 5: Экспорт результатов...")
        
        # Базовое имя для файлов
        audio_basename = Path(args.audio_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = os.path.join(args.output_dir, f"{audio_basename}_{timestamp}")
        
        # Объединяем статистику
        all_stats = {**diarization_stats, **postprocess_stats}
        
        # Экспорт
        if args.export_all:
            ResultExporter.export_all(final_segments, base_output, all_stats, diarization_enabled=not args.no_diarization)
        else:
            ResultExporter.export_text(final_segments, f"{base_output}.txt", diarization_enabled=not args.no_diarization)
        
        # Вывод статистики
        print("\n" + "=" * 70)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 70)
        
        if final_segments:
            total_duration = max(seg['end'] for seg in final_segments)
            
            print(f"✓ Файл результатов: {base_output}.txt")
            print(f"✓ Общее время аудио: {total_duration:.1f} сек ({total_duration/60:.1f} мин)")
            print(f"✓ Всего сегментов: {len(final_segments)}")
            
            if not args.no_diarization and postprocess_stats and 'speaker_durations' in postprocess_stats:
                speakers = postprocess_stats['speaker_durations']
                print(f"✓ Обнаружено спикеров: {len(speakers)}")
                
                print("\nРаспределение по спикерам:")
                for speaker, duration in sorted(speakers.items(), key=lambda x: x[1], reverse=True):
                    percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                    print(f"  {speaker}: {duration:.1f} сек ({percentage:.1f}%)")
            elif args.no_diarization:
                print("✓ Диаризация отключена, все реплики от одного спикера")
            
            print(f"\n✓ Результаты сохранены в: {args.output_dir}")
        
    except Exception as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Финализация
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()

