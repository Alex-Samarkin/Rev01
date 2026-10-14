# COVID-19 по субъектам РФ: чтение timeseries.json и экспорт отдельных графиков.
# Запуск: julia 03_timeseries_regions.jl
# Ячейки # %% можно последовательно выполнять в Julia VS Code.

# %% Окружение и зависимости
import Pkg
Pkg.activate(joinpath(@__DIR__, "@SciTools"))
# Установка зависимостей при первой настройке: Pkg.instantiate().
using JSON, DataFrames, CSV, Dates, Plots

# %% Настройки
# Название должно точно совпадать с ключом timeseries.json.
# Доступные города: Москва, Санкт-Петербург, Севастополь.
# Все доступные субъекты в текущем timeseries.json:
#   Москва
#   Московская область
#   Санкт-Петербург
#   Свердловская область
#   Самарская область
#   Республика Саха (Якутия)
#   Нижегородская область
#   Кировская область
#   Калининградская область
#   Республика Татарстан
#   Тульская область
#   Красноярский край
#   Краснодарский край
#   Новосибирская область
#   Хабаровский край
#   Тюменская область
#   Тамбовская область
#   Пермский край
#   Липецкая область
#   Тверская область
#   Удмуртская Республика
#   Кабардино-Балкарская Республика
#   Воронежская область
#   Кемеровская область
#   Архангельская область
#   Рязанская область
#   Ханты-Мансийский АО
#   Пензенская область
#   Республика Коми
#   Мурманская область
#   Ленинградская область
#   Ставропольский край
#   Республика Крым
#   Калужская область
#   Курганская область
#   Брянская область
#   Саратовская область
#   Ярославская область
#   Забайкальский край
#   Ивановская область
#   Белгородская область
#   Ульяновская область
#   Новгородская область
#   Челябинская область
#   Томская область
#   Оренбургская область
#   Республика Хакасия
#   Республика Башкортостан
#   Приморский край
#   Орловская область
#   Чеченская Республика
#   Волгоградская область
#   Псковская область
#   Ростовская область
#   Республика Бурятия
#   Республика Мордовия
#   Республика Дагестан
#   Сахалинская область
#   Костромская область
#   Смоленская область
#   Республика Адыгея
#   Омская область
#   Амурская область
#   Иркутская область
#   Севастополь
#   Вологодская область
#   Владимирская область
#   Республика Марий Эл
#   Алтайский край
#   Республика Калмыкия
#   Астраханская область
#   Магаданская область
#   Курская область
#   Республика Северная Осетия — Алания
#   Ямало-Ненецкий автономный округ
#   Еврейская автономная область
#   Республика Ингушетия
#   Камчатский край
#   Республика Карелия
#   Карачаево-Черкесская Республика
#   Республика Тыва
#   Ненецкий автономный округ
#   Чукотский автономный округ
#   Республика Алтай
#   Чувашская Республика
const SUBJECT = "Москва"
const JSON_PATH = joinpath(@__DIR__, "timeseries.json")
const OUTPUT_DIR = joinpath(@__DIR__, "figures", "timeseries", SUBJECT)

# %% Чтение JSON и формирование общей таблицы
# JSON содержит объект {"название субъекта": [наблюдения, ...]}.
# confirmed, recovered, deaths — накопленные числа, а не суточные приращения.
# Числа в исходном файле записаны строками; преобразуем их в Int.
function read_timeseries(path::AbstractString)
    isfile(path) || error("JSON-файл не найден: $path")
    data = JSON.parsefile(path)
    data isa AbstractDict || error("Ожидается JSON-объект с субъектами РФ в ключах")

    table = DataFrame(
        субъект = String[],
        дата = Date[],
        заболевших = Int[],
        выздоровевших = Int[],
        умерших = Int[],
    )
    for (subject, records) in pairs(data)
        records isa AbstractVector || error("Для субъекта $subject ожидается массив наблюдений")
        for record in records
            # При некорректной записи прекращаем обработку с указанием субъекта.
            # Нулевые значения сохраняем; пропуски не подменяем нулями.
            row = try
                (
                    String(subject),
                    Date(record["date"], dateformat"yyyy-mm-dd"),
                    parse(Int, string(record["confirmed"])),
                    parse(Int, string(record["recovered"])),
                    parse(Int, string(record["deaths"])),
                )
            catch err
                error("Некорректное наблюдение для $subject: $record; $(sprint(showerror, err))")
            end
            all(x -> x >= 0, row[3:5]) || error("Отрицательное число случаев для $subject, $(row[2])")
            push!(table, row)
        end
    end
    sort!(table, [:субъект, :дата])
    # Дубли даты для одного субъекта делают временной ряд неоднозначным.
    nrow(unique(table, [:субъект, :дата])) == nrow(table) ||
        error("Обнаружены повторяющиеся пары субъект–дата")
    return table
end

timeseries_table = read_timeseries(JSON_PATH)
println("Общая таблица: $(nrow(timeseries_table)) строк")
display(first(timeseries_table, 10))

# %% Выбор субъекта и сохранение таблиц
available_subjects = sort(unique(timeseries_table.субъект))
SUBJECT in available_subjects || error(
    "Субъект '$SUBJECT' отсутствует. Доступны: $(join(available_subjects, ", "))"
)
subject_table = filter(:субъект => ==(SUBJECT), timeseries_table)
println("$SUBJECT: $(nrow(subject_table)) наблюдений, $(first(subject_table.дата)) — $(last(subject_table.дата))")
display(first(subject_table, 10))

mkpath(OUTPUT_DIR)
# Общая таблица содержит все субъекты; выбранная — только SUBJECT.
# UTF-8 BOM упрощает отображение русских заголовков в Excel.
CSV.write(joinpath(dirname(OUTPUT_DIR), "all_subjects.csv"), timeseries_table; bom = true)
CSV.write(joinpath(OUTPUT_DIR, "timeseries.csv"), subject_table; bom = true)

# %% Первые разности: количество за один день
# Разности считаем отдельно внутри каждого субъекта: ΔC(t) = C(t) − C(t−1).
# Первое наблюдение и даты после пропуска получают missing: по накопленным
# значениям за несколько дней нельзя восстановить число за конкретный день.
# Отрицательные разности сохраняем: это могут быть корректировки источника.
function daily_differences(table)
    daily = DataFrame(
        субъект = String[], дата = Date[],
        заболевших = Union{Missing, Int}[],
        выздоровевших = Union{Missing, Int}[],
        умерших = Union{Missing, Int}[],
    )
    for series in groupby(sort(table, [:субъект, :дата]), :субъект)
        for i in 1:nrow(series)
            consecutive = i > 1 && series.дата[i] - series.дата[i - 1] == Day(1)
            changes = map((:заболевших, :выздоровевших, :умерших)) do column
                consecutive ? series[i, column] - series[i - 1, column] : missing
            end
            push!(daily, (series.субъект[i], series.дата[i], changes...))
        end
    end
    return daily
end

daily_table = daily_differences(timeseries_table)
subject_daily_table = filter(:субъект => ==(SUBJECT), daily_table)
println("$SUBJECT: первые разности за один день (missing — нет предыдущего дня)")
show(stdout, MIME"text/plain"(), subject_daily_table; allrows = true, allcols = true)
println()
CSV.write(joinpath(dirname(OUTPUT_DIR), "all_subjects_daily.csv"), daily_table; bom = true)
CSV.write(joinpath(OUTPUT_DIR, "timeseries_daily.csv"), subject_daily_table; bom = true)

# %% Три отдельных графика: PNG и SVG
# Контракт графиков: накопленное число людей по датам для выбранного субъекта.
# Линейная шкала начинается с нуля; без сглаживания и пересчёта в новые случаи.
# Каждый показатель имеет отдельный график, подписи дат и источник данных.
function save_timeseries_plot(table, column::Symbol, label::String, filename::String; daily = false)
    dates = table.дата
    # NaN разрывает линию на датах, где суточный прирост неизвестен.
    values = Float64.(coalesce.(table[!, column], NaN))
    finite_values = filter(isfinite, values)
    isempty(finite_values) && error("Нет значений для графика $label")
    lower = min(0.0, minimum(finite_values) * 1.05)
    upper = max(1.0, maximum(finite_values) * 1.05)
    units = daily ? "За один день, человек" : "Накопленно, человек"
    # Не более семи подписей оси X; точки располагаются по реальным датам.
    tick_indices = unique(round.(Int, range(1, length(dates); length = min(7, length(dates)))))
    tick_dates = dates[tick_indices]
    tick_labels = Dates.format.(tick_dates, dateformat"dd.mm.yyyy")
    figure = plot(
        dates, values;
        title = "$(first(table.субъект)): $label\n$units • источник: timeseries.json",
        xlabel = "Дата",
        ylabel = daily ? "Человек за день (первая разность)" : "Человек (накопленно)",
        xticks = (tick_dates, tick_labels),
        xrotation = 30,
        ylims = (lower, upper),
        color = "#4263EB",
        linewidth = 2,
        legend = false,
        grid = true,
        gridalpha = 0.15,
        framestyle = :box,
        size = (1100, 650),
        dpi = 180,
        fontfamily = "DejaVu Sans",
        titlefontsize = 12,
        guidefontsize = 11,
        tickfontsize = 9,
        margin = 8Plots.mm,
    )
    for extension in ("png", "svg")
        path = joinpath(OUTPUT_DIR, "$filename.$extension")
        savefig(figure, path)
        println("Сохранён график: $path")
    end
    display(figure)
    return figure
end

confirmed_plot = save_timeseries_plot(subject_table, :заболевших, "заболевшие", "confirmed")
recovered_plot = save_timeseries_plot(subject_table, :выздоровевших, "выздоровевшие", "recovered")
deaths_plot = save_timeseries_plot(subject_table, :умерших, "умершие", "deaths")

# %% Суточные приросты: три дополнительных графика PNG и SVG
confirmed_daily_plot = save_timeseries_plot(subject_daily_table, :заболевших, "новые заболевшие", "confirmed_daily"; daily = true)
recovered_daily_plot = save_timeseries_plot(subject_daily_table, :выздоровевших, "новые выздоровевшие", "recovered_daily"; daily = true)
deaths_daily_plot = save_timeseries_plot(subject_daily_table, :умерших, "новые умершие", "deaths_daily"; daily = true)
