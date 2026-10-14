# Julia-версия численного эксперимента: дробная SIR-модель (Caputo) и Pluto-блокнот

Зеркальная реализация Python-версии (`code/python/fractional_sir.py`) на Julia.
Вычислительное ядро использует **только стандартную библиотеку Julia**
(Base, LinearAlgebra, Printf, DelimitedFiles, Statistics) — никаких внешних
пакетов, поэтому оно работает даже в средах, где установка пакетов и
прекомпиляция недоступны (ошибка `IOError: pipe_open: bad file descriptor`).

| Файл | Назначение |
|---|---|
| `code/julia/fractional_sir.jl` | ядро: схема L1, спецфункции, чтение данных, метод Нелдера—Мида, CLI |
| `code/julia/wave_identification_pluto.jl` | Pluto-блокнот с визуализацией (для машины, где работает прекомпиляция) |
| `code/julia/check_notebook_core.jl` | проверка вычислительного ядра блокнота без пакетов Plots/DataFrames/CSV/Optim |
| `code/julia/compare_with_python.jl` | построчное сравнение результатов Julia и Python-версии |

---

## 1. Быстрый запуск без установки пакетов (скриптовая версия)

Обязательные флаги: `--startup-file=no --compiled-modules=no`.
Julia-depot удобно держать внутри проекта:

```powershell
$env:JULIA_DEPOT_PATH = "C:\Users\Alex\Rev01\.julia"
```

```powershell
# Проверка схемы L1 на аналитическом решении E_{1/2}(-1) = exp(1)*erfc(1)
julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --verify

# Идентификация по всем 7 территориям (classic + fractional),
# печатает таблицу и сохраняет results/identification_julia.csv
julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --all

# Одна территория (имя в кавычках, пробелы допустимы)
julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --territory "Москва"

# Дополнительные параметры: число стартов Нелдера—Мида и лимит итераций
julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --all --starts 20 --maxiter 1500
```

Что создаётся:

* `results/identification_julia.csv` — параметры и метрики по территориям (`--all`);
* `results/identification_julia_partial.csv` — то же для `--territory`;
* `results/identification_julia_table.txt` — текстовая копия таблицы (UTF-8);
* `results/verify_scheme_julia.txt` — протокол проверки схемы (`--verify` или `--all`).

## 2. Проверка вычислительного ядра Pluto-блокнота

```powershell
julia --startup-file=no --compiled-modules=no code/julia/check_notebook_core.jl
```

Скрипт разбирает блокнот (уникальность UUID ячеек, полнота секции `Cell order`,
вычисление всех md-ячеек) и выполняет те же расчёты, что и код-ячейки блокнота,
но без пакетов `Plots`, `DataFrames`, `CSV`, `Optim`. Протокол сохраняется в
`results/notebook_core_check.txt`.

## 3. Сравнение с Python-версией

Сначала получите свежий прогон Python-версии при тех же настройках
(`observable = "cumulative"`, `tau = 7.0`):

```powershell
& code\python\.venv\Scripts\python.exe code\python\run_identification.py `
    --observable cumulative --out identification_cumulative_tau7.json
```

Затем постройте построчную таблицу сравнения:

```powershell
julia --startup-file=no --compiled-modules=no code/julia/compare_with_python.jl
```

Результат — `results/comparison_julia_python.csv` с относительными расхождениями
по RMSE, ``R^2``, AIC, ``\alpha``, ``\lambda``, ``R_0`` и ``A``.

## 4. Синтаксическая проверка блокнота

```powershell
julia --startup-file=no --compiled-modules=no -e 'Meta.parseall(read("code/julia/wave_identification_pluto.jl", String)); println("PARSE_OK")'
```

## 5. Полноценный запуск Pluto-блокнота (на обычной машине)

Установка пакетов (однократно):

```powershell
julia -e 'using Pkg; Pkg.add(["Pluto","Plots","DataFrames","CSV","Optim"])'
```

Запуск:

```powershell
julia -e 'using Pluto; Pluto.run()'
```

В открывшемся браузере выберите файл
`code/julia/wave_identification_pluto.jl`. Блокнот сам подключает
`fractional_sir.jl` через `include(joinpath(@__DIR__, "fractional_sir.jl"))`,
поэтому посторонние пакеты для расчётов не нужны — они требуются только для
графиков и таблиц.

> **Примечание.** Если запускать блокнот в среде, где прекомпиляция ломается,
> расчётные ячейки всё равно выполняются (они зависят только от stdlib), а
> ячейки с `Plots`/`DataFrames`/`CSV`/`Optim` — нет. Для проверки без пакетов
> используйте `check_notebook_core.jl` (п. 2).

## 6. Содержание блокнота

1. Производная Капуто и вывод схемы L1, веса ``b_k = (k+1)^{1-\alpha} - k^{1-\alpha}``.
2. Проверка схемы на аналитическом решении ``E_{1/2}(-1) = e\,\mathrm{erfc}(1)``,
   таблица ошибок и график сходимости (log–log, эталонный наклон −1).
3. Семейство решений дробной SIR при ``\alpha \in \{0.3, 0.5, 0.7, 0.9, 1.0\}``
   при прочих равных — демонстрация роли порядка дробной производной.
4. Загрузка `data/processed/waves.csv` (через `CSV`/`DataFrames` и через
   `DelimitedFiles` в ядре) и графики реальных волн.
5. Идентификация параметров выбранной территории и наложение подгонки на данные.
6. Дополнительное уточнение найденного минимума пакетом `Optim.jl` (перекрёстная
   проверка оптимизатора) и таблица сравнения классической и дробной моделей.
7. Сводный график ``\alpha`` и RMSE по территориям, выводы.

## 7. Согласование с Python-версией

Формулы, границы параметров, схема L1 и метрики полностью совпадают с
`code/python/fractional_sir.py`. Проверено прямым сравнением: при
`observable = "cumulative"`, `tau = 7.0` Julia и Python дают одинаковые
``\lambda``, ``R_0``, ``A``, RMSE, ``R^2`` и AIC по всем семи территориям
(см. отчёт). Проверка схемы (`--verify`) воспроизводит вывод
`code/python/verify_scheme.py` с той же точностью.

**Важное расхождение с файлами, лежавшими в проекте до этой работы.**
Файлы `results/identification_cumulative.csv` / `.json`, которые были в проекте
на момент начала работы, получены более ранней версией Python-скрипта:
в них ``\tau = 14`` суток (видно по ``\gamma = 1/14 = 0.0714286``), а ``k`` в AIC
равно 4 (classic) и 5 (fractional). Текущий код и постановка задачи фиксируют
``\tau = 7`` суток и ``k = 3/4``, поэтому числа отличаются, например для Москвы
(классическая модель): RMSE 5666.55 (старый вариант, ``\tau=14``) против
6135.64 (актуальный расчёт, ``\tau=7``).
Параметризация ``\tau=14`` сохранилась в нетронутом
`results/identification_daily_ma7.csv` (там же ``k = 2/3`` и другая
параметризация свободных параметров).
Свежий Python-прогон с ``\tau=7`` лежит в
`results/identification_cumulative_tau7.csv` — он совпадает с Julia.
