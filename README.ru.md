<div align="center">

<a href="README.md">🇬🇧 Read in English</a>

# 🛡️ FedGuard-IIoT

### Коммуникационно-эффективное и устойчивое федеративное обнаружение аномалий  
### для ресурсно-ограниченных IIoT Edge-сетей

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge)](https://github.com/astral-sh/uv)
[![Typst](https://img.shields.io/badge/Typst-article-239DAD?style=for-the-badge)](https://typst.app)

*Sundetkhan Bekzat · Baibolat Bekarys*  
*Introduction to Multi-Agent Systems · Seema Rawat · 2026*

</div>

---

## 📌 О проекте

**FedGuard-IIoT** — исследовательский проект, посвящённый задаче **федеративного обнаружения аномалий** в промышленных IoT-сетях (IIoT). Главная цель — разработать и экспериментально проверить модульную систему, которая одновременно решает три ключевые проблемы реальных edge-развёртываний:

| Проблема | Наше решение |
|----------|-------------|
| 🔗 **Ограниченная полоса uplink** | Top-K sparsification + error-feedback |
| 📊 **Неоднородность данных (non-IID)** | FedProx + персонализированные пороги |
| ☠️ **Злоумышленные клиенты (Byzantine)** | Robust aggregation: RFA / Median / Krum |

> В отличие от классического FedAvg, который уязвим к атакам и плохо работает при неоднородных данных, наша система сохраняет высокое качество детекции аномалий **даже при 30% атакующих клиентах** и **10-кратном сжатии трафика**.

---

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                     IIoT Edge Clients                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Client 0 │  │ Client 1 │  │ Client 2 │  │ Client N │   │
│  │ (Line A) │  │ (SCADA)  │  │(Gateway) │  │  (PLC)   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│  [LSTM-AE Local Training] → [FedProx Loss] → [Top-K Compress]
└───────────────────────┬─────────────────────────────────────┘
                        │  Сжатые обновления Δ̂ₖ
                ┌───────▼────────┐
                │   FL  Сервер   │
                │  ┌──────────┐  │
                │  │  RFA /   │  │
                │  │ Median / │  │  ← Robust Aggregation
                │  │  Krum    │  │
                │  └──────────┘  │
                │  Глобальная θ  │
                └───────┬────────┘
                        │ Broadcast
                   [Edge Inference]
                  Anomaly Score > τ ?
```

---

## 🧠 ML-стек: модели и методы

### 🔵 Модель: LSTM Autoencoder (LSTM-AE)

Основная модель обнаружения аномалий — **реконструкционный автоэнкодер** на основе LSTM. Обучается только на **нормальных данных** (unsupervised). Аномалия = высокая ошибка реконструкции.

```
Вход:  X ∈ ℝ^(W×F)   (окно W=30 шагов, F=6 сенсоров)
       ↓
   LSTM Encoder  →  z ∈ ℝ^16  (латентное представление)
       ↓
   LSTM Decoder  →  X̂ ∈ ℝ^(W×F)
       ↓
Score: s(X) = (1/WF) ‖X - X̂‖²_F    (anomaly score)
Аномалия если s(X) > τ  (τ = 95-й персентиль ошибок на train)
```

**Параметры:** Hidden `H=48` · Latent `L=16` · ~51K параметров · Подходит для edge-устройств

---

### 🟢 Методы федеративного обучения

| Метод | Роль | Ключевой параметр |
|-------|------|------------------|
| **FedAvg** | Базовый агрегатор | — |
| **FedProx** | Стабилизация при non-IID | μ = 0.01 (proximal term) |
| **Top-K sparsification** | Сжатие обновлений | κ = 10% (передаём топ-10% весов) |
| **Error-feedback** | Компенсация ошибки сжатия | Накопленный остаток eₖ |

Каждый раунд: выбираем 60% клиентов → 3 локальных эпохи → сжатие → агрегация → broadcast.

---

### 🔴 Robust Aggregation (защита от атак)

| Агрегатор | Принцип | Устойчивость |
|-----------|---------|-------------|
| **FedAvg** | Среднее арифметическое | ❌ Уязвим |
| **Coord. Median** | Координатная медиана | ✅ Хорошая |
| **Trimmed Mean** | Усечённое среднее (10%) | ✅ Хорошая |
| **Krum** | Выбор ближайшего соседа | ✅ Теоретически строгая |
| **RFA** | Геометрическая медиана (Weiszfeld) | ✅ Наилучшая |

---

### ☠️ Смоделированные атаки

| Атака | Описание |
|-------|----------|
| **Model Poisoning** | Случайный/направленный шум в обновлениях модели |
| **Scaled Poisoning** | Инверсия и масштабирование градиентов |
| **Label Flipping** | Инверсия меток аномалий при локальном обучении |
| **On-Off Attack** | Чередование честных и атакующих раундов для обхода защиты |

---

## 📊 Результаты экспериментов

### Коммуникационная эффективность

| Конфигурация | AUROC | F1 | Трафик (MB) | Экономия |
|--------------|-------|----|-------------|---------|
| FedAvg (baseline) | 0.703 | 0.231 | 22.37 | — |
| FedProx | 0.695 | 0.251 | 22.37 | — |
| **TopK-10%** | 0.690 | 0.248 | **2.24** | **10×** |
| **FedProx + TopK** | 0.701 | 0.229 | **2.24** | **10×** |

### Устойчивость к атакам (model poisoning)

| Агрегатор | α=0% | α=10% | α=20% | α=30% |
|-----------|------|-------|-------|-------|
| FedAvg | 0.77 | 0.51 | 0.50 | 0.51 |
| **Median** | 0.77 | 0.72 | **0.82** | **0.78** |
| RFA | 0.64 | 0.66 | 0.71 | 0.70 |

### On-Off атака vs. постоянное отравление (α=20%)

| | Постоянное отравление | On-Off атака |
|-|-----------------------|-------------|
| FedAvg | 0.49 | 0.49 |
| **Median** | **0.84** | **0.82** |
| RFA | 0.71 | 0.68 |

---

## 🗂️ Структура проекта

```
multi/
│
├── 📄 sn-article.typ          ← Typst-исходник научной статьи
├── 📄 sn-article.pdf          ← Скомпилированная статья (1.4 MB)
├── 📄 run_experiments.py      ← Главный пайплайн (запустить всё)
├── 📄 pyproject.toml          ← uv зависимости
├── 🔧 typst                   ← Бинарник Typst для компиляции PDF
│
├── src/
│   ├── data_gen.py            ← Генератор синтетического IIoT датасета
│   ├── models.py              ← LSTM Autoencoder + Deep SVDD
│   ├── fl_engine.py           ← FL движок (FedAvg/FedProx/TopK/агрегаторы)
│   ├── attacks.py             ← Реализации атак
│   ├── experiments.py         ← 5 экспериментов с метриками
│   └── visualization.py       ← Генерация 8 публикационных графиков
│
├── data/                      ← Синтетический датасет (auto-generated, .npy)
├── figures/                   ← 8 PNG графиков для статьи
├── results/
│   └── experiment_results.json
└── deep-research-report.md    ← Исходный research-report
```

---

## 🚀 Быстрый старт

```bash
# 1. Установить зависимости
uv sync

# 2. Запустить все эксперименты (~26 мин на GPU)
uv run python run_experiments.py

# 3. Скомпилировать PDF статью
./typst compile sn-article.typ sn-article.pdf
```

### Зависимости

```toml
torch >= 2.2.0        # Глубокое обучение
numpy >= 1.26.0       # Численные вычисления
scikit-learn >= 1.4.0 # Метрики (AUROC, F1)
matplotlib >= 3.8.0   # Построение графиков
seaborn >= 0.13.0     # Стилизация
pandas >= 2.2.0       # Работа с данными
scipy >= 1.12.0       # Статистика
tqdm >= 4.66.0        # Прогресс-бары
```

---

## 📐 Датасет

Синтетический **IIoT multi-client benchmark** — 5 клиентов (edge-узлов) с разными параметрами (non-IID):

| Клиент | Частота | Амплитуда | Шум | Роль |
|--------|---------|-----------|-----|------|
| 0 | 0.010 | 1.0 | 0.05 | Factory Line A |
| 1 | 0.018 | 1.4 | 0.08 | SCADA Node |
| 2 | 0.026 | 1.8 | 0.11 | Edge Gateway |
| 3 | 0.034 | 2.2 | 0.14 | Sensor Array |
| 4 | 0.042 | 2.6 | 0.17 | PLC Unit |

**Типы аномалий:** Point spikes · Drift segments · Pattern shifts (~6% данных)

---

## 📖 Публикация

Статья оформлена в стиле **Springer Nature** через [Typst](https://typst.app):

```
Communication-Efficient and Robust Federated Anomaly Detection
for Resource-Constrained IIoT Edge Networks
under Non-IID and Adversarial Settings

Sundetkhan Bekzat, Baibolat Bekarys
Introduction to Multi-Agent Systems · Seema Rawat · 2026
```

---

<div align="center">

Сделано с ❤️ для курса **Introduction to Multi-Agent Systems**

</div>
