# 🎯 Link Prediction in Twitch Social Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.0+-green.svg)](https://catboost.ai/)
[![AUC](https://img.shields.io/badge/AUC-0.9369-brightgreen.svg)]()

Модель машинного обучения для предсказания вероятности подписки между пользователями Twitch.

---

## 📊 Ключевые результаты

| Метрика | Значение |
|---------|----------|
| **AUC-ROC** | **0.9369** |
| **Accuracy** | **87%** |
| **Precision (class 1)** | 0.88 |
| **Recall (class 1)** | 0.85 |
| **Время обучения** | 6.62 сек |
| **Размер выборки** | 74,608 пар |

---

## 🧠 Методология

### 1. Построение графа
- **Узлы**: пользователи Twitch
- **Ребра**: существующие подписки
- **Библиотека**: NetworkX

### 2. Признаки узлов

| Признак | Описание | Источник |
|---------|----------|----------|
| `connections` | Количество подписчиков | NetworkX (degree) |
| `rate` | Важность узла в графе | NetworkX (PageRank) |
| `community` | Сообщество пользователя | Louvain algorithm |
| `days` | Возраст аккаунта | Профиль пользователя |
| `mature` | Взрослый контент (0/1) | Профиль пользователя |
| `partner` | Партнер Twitch (0/1) | Профиль пользователя |
| `views` | Количество просмотров | Профиль пользователя |

### 3. Признаки пар (9 признаков)

| Признак | Описание | Важность |
|---------|----------|----------|
| **common_friends** | Количество общих друзей | **25%** |
| **views_diff** | Разница в просмотрах | **22%** |
| **same_comm** | В одном сообществе | **18%** |
| **rate_diff** | Разница в PageRank | **14%** |
| **connections_diff** | Разница в подписчиках | **10%** |
| **common_interests** | Общие интересы (теги) | **6%** |
| **days_diff** | Разница в возрасте аккаунта | **3%** |
| **both_partner** | Оба партнеры Twitch | **1.5%** |
| **both_mature** | Оба имеют взрослый контент | **0.5%** |

### 4. Обучение модели

**Почему CatBoost?**
- ✅ Лучший AUC (0.9369) среди XGBoost, LightGBM, Random Forest
- ✅ Автоматическая обработка категориальных признаков
- ✅ Устойчивость к переобучению (Ordered Boosting)
- ✅ Быстрое обучение (6.62 сек)

**Гиперпараметры:**
```python
model = CatBoostClassifier(
    iterations=500,       # Количество деревьев
    learning_rate=0.05,   # Скорость обучения
    depth=6,              # Глубина деревьев
    eval_metric='AUC',    # Метрика качества
    random_seed=42        # Воспроизводимость
)
```

---

## 📈 Визуализация важности признаков

![Feature Importance](image.png)

---

## 🚀 Запуск

### 1. Клонировать репозиторий
```bash
git clone https://github.com/Mimi-cyber519/catboost-twitch-link-prediction.git
cd catboost-twitch-link-prediction
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

### 3. Скачать данные
Данные доступны на Kaggle:  
[https://www.kaggle.com/datasets/andreagarritano/twitch-social-networks](https://www.kaggle.com/datasets/andreagarritano/twitch-social-networks)

Положите файлы в папку `data/`:
- `edges.csv`
- `target.csv`
- `features.json`

### 4. Запустить обучение
```bash
link_prediction.py
```

---

## 💡 Выводы

- **Главный фактор** — социальные связи (общие друзья)
- **Популярность** (просмотры, подписчики) — важный предиктор
- **Сообщества** объединяют пользователей
- **Интересы** важны, но данные неполные
- Модель готова к использованию в рекомендательных системах

---

## 🛠 Технологии

- Python 3.8+
- CatBoost
- NetworkX
- Pandas
- Scikit-learn
- Matplotlib

---

## 📁 Структура проекта

```
catboost-twitch-link-prediction/
│
├── README.md                 # Описание проекта
├── requirements.txt          # Зависимости
├── link_prediction.py        # Основной код
│
├── data/                     # Данные (не включены)
│   ├── edges.csv
│   ├── target.csv
│   └── features.json
```

---

## 📝 Лицензия

MIT License

---

⭐ **Если проект был полезен, поставьте звезду!**
