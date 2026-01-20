# ЛР2 — Комп'ютерне моделювання: затяжний стрибок парашутиста

Проєкт реалізує модель з ЛР1 (нелінійний опір повітря, залежність Fd ~ v^2) у двох фазах:
- фаза 1: без парашута до моменту t_open;
- фаза 2: з парашутом після t_open.

Після запуску автоматично створюються:
- flowchart_algorithm.png — блок-схема алгоритму;
- plot1_speed_vs_time_mass.png — графік v(t) з 5 кривими для різних мас;
- plot2_height_vs_time_radius.png — графік h(t) з 4 кривими для різних радіусів;
- diagram_time_to_const_vs_radius.png — стовпчикова діаграма t_const після розкриття vs R;
- summary_table.csv — таблиця підсумкових результатів;
- verification_terminal_velocity.csv — верифікація: чисельно vs теоретично;
- report_short.txt — короткий текстовий звіт.

## Вимоги
- Python 3.10+ (або 3.11/3.12)

## Як запустити (Windows)
1) Розпакуй ZIP.
2) Відкрий PowerShell у папці проєкту.
3) Створи venv і встанови залежності:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4) Запусти генерацію всіх матеріалів:

```powershell
python -m src.run_all
```

Результати  у папці `outputs`.

## Як запустити (Linux/Mac)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all
```


Базові параметри для експериментів задаються у файлі:
- `src/experiments.py` (змінна `base` і списки `masses`, `radii`).


Використано відносний допуск:
|v - v_term| / v_term <= eps_rel
За замовчуванням eps_rel = 0.01 (1%).
