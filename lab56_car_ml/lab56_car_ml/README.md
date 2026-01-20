# Лабораторні роботи 5+6 — Car Evaluation

**ЛР5 (класифікація)** та **ЛР6 (кластеризація)** на датасеті `Car evaluation.csv`.

## Запуск
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.run_all
```

## Результати
Після запуску  `outputs/`:
- `outputs/lab5/` — describe, статистика, scatter-matrix, класифікація + метрики, confusion matrix, вплив масштабування, підбір кращої моделі, власна евклідова відстань + nearest neighbor.
- `outputs/lab6/` — k-means (k=4), 4 графіки попарних ознак + центроїди, таблиці кластерів, підбір k (метрики) + вплив StandardScaler/MinMaxScaler.

> Якщо у файлі зустрічаються значення `Interval[{5, Infinity}]`, код автоматично замінює їх на `5`.
