### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 5a1c0001-0a1b-11f1-a001-5a9c2f1e7b01
begin
    import Pkg
    Pkg.activate("@SciTools")
    Pkg.add(["DataFrames", "Plots"])
end

# ╔═╡ 5a1c0002-0a1b-11f1-a001-5a9c2f1e7b01
begin
    using Dates, DataFrames, Plots
end

# ╔═╡ 5a1c0003-0a1b-11f1-a001-5a9c2f1e7b01
md"""
# Доминирующие штаммы SARS-CoV-2 в России, 2020–2023

Блокнот содержит справочник доминирующих вариантов вируса на территории РФ за период пандемии
(от объявления пандемии ВОЗ 11.03.2020 до окончания режима ЧС ВОЗ 05.05.2023) и функцию
`strain_at(дата)`, которая возвращает показатели доминирующего штамма на выбранный день.
Далее для каждого показателя строится отдельный график за всё время.

**Показатели:**

| Показатель | Что это |
|---|---|
| `share_pct` | доля доминирующей линии среди секвенированных образцов в РФ, % |
| `R` | оценка репродуктивного числа (R₀ или Rₑ — см. примечания, методики различаются) |
| `incubation_days` | средний инкубационный период, сутки |

> **Важно о данных.** Это сводка по публикациям, а не официальный ряд Роспотребнадзора.
> Границы периодов приблизительные. Там, где в найденных источниках нет значения,
> стоит `NaN` (на графике разрыв), а не придуманная цифра. Подробности и источники — внизу.
"""

# ╔═╡ 5a1c0004-0a1b-11f1-a001-5a9c2f1e7b01
begin
    pandemic_start = Date(2020, 3, 11)   # объявление пандемии ВОЗ
    pandemic_end = Date(2023, 5, 5)      # отмена ЧС международного значения ВОЗ

    # share_anchors — опорные точки доли (дата, доля в %), между ними линейная интерполяция;
    # вне диапазона опорных точек доля не определена (NaN).
    strains = [
        (
            name = "Ранние линии (B.1 / B.1.1)", short = "B.1/B.1.1",
            pango = "B.1, B.1.1", who = "—",
            start = Date(2020, 3, 11), stop = Date(2020, 7, 31),
            R = 2.79, incubation_days = 6.57,
            share_anchors = Tuple{Date, Float64}[],
            confidence = "низкая",
            note = "Исходный (предковый) тип; R₀ = 2.79 — оценка для предкового штамма. " *
                   "Инкубация 6.57 сут — сводная оценка метаанализа по всем штаммам. " *
                   "Числовой доли для РФ в найденных источниках нет."
        ),
        (
            name = "B.1.1.317 и подлиния B.1.1.397", short = "B.1.1.317/397",
            pango = "B.1.1.317, B.1.1.397", who = "—",
            start = Date(2020, 8, 1), stop = Date(2021, 5, 14),
            R = NaN, incubation_days = 6.57,
            share_anchors = Tuple{Date, Float64}[],
            confidence = "низкая",
            note = "Линии, редкие в других странах, но распространившиеся в РФ в этот период " *
                   "(исследование геномов февраль 2020 – март 2021). Преимущество в " *
                   "трансмиссивности выведено косвенно; оценки R для них в источниках нет. " *
                   "Точная дата смены линий и доли не приведены."
        ),
        (
            name = "Дельта (AY.122)", short = "Дельта",
            pango = "B.1.617.2 / AY.*, в т.ч. AY.122", who = "Delta",
            start = Date(2021, 5, 15), stop = Date(2021, 12, 31),
            R = 5.08, incubation_days = 4.41,
            share_anchors = [
                (Date(2021, 5, 20), 66.0),    # CoVariants, вторая половина мая 2021
                (Date(2021, 6, 30), 90.0),    # «свыше 90 %» с июня; нижняя граница
                (Date(2021, 12, 31), 90.0),   # AY.122 > 90 % в период май–декабрь 2021
            ],
            confidence = "средняя",
            note = "Рост с 1 % в апреле до >90 % в июне 2021 (по геномным данным); Роспотребнадзор " *
                   "в июне называл 41 % — расхождение источников. Доли 90 % — нижняя граница " *
                   "(«свыше 90 %»). R₀ = 5.08 — среднее по обзору (диапазон 3.2–8)."
        ),
        (
            name = "Омикрон BA.1 / BA.1.1", short = "BA.1.1",
            pango = "B.1.1.529 → BA.1, BA.1.1", who = "Omicron",
            start = Date(2022, 1, 1), stop = Date(2022, 2, 28),
            R = 3.22, incubation_days = 3.42,
            share_anchors = [
                (Date(2022, 1, 15), 62.0),    # BA.1.1 среди образцов января 2022
                (Date(2022, 2, 15), 58.0),    # февраль 2022
            ],
            confidence = "средняя",
            note = "Доли — для подлинии BA.1.1 среди секвенированных образцов (VGARus). " *
                   "R — эффективное репродуктивное число Rₑ (объединённая оценка), не R₀. " *
                   "Инкубация 3.42 сут — для Омикрона в целом."
        ),
        (
            name = "Омикрон BA.2", short = "BA.2",
            pango = "BA.2", who = "Omicron",
            start = Date(2022, 3, 1), stop = Date(2022, 6, 30),
            R = 5.04, incubation_days = 3.42,
            share_anchors = [
                (Date(2022, 3, 15), 53.0),    # март 2022
                (Date(2022, 6, 15), 74.0),    # июнь 2022
            ],
            confidence = "средняя",
            note = "Доля BA.2 росла с 53 % (март) до 74 % (июнь 2022). R — Rₑ (объединённая оценка)."
        ),
        (
            name = "Омикрон BA.5 (BA.5.2) и потомки", short = "BA.5",
            pango = "BA.5, BA.5.2", who = "Omicron",
            start = Date(2022, 7, 1), stop = Date(2023, 5, 5),
            R = 5.22, incubation_days = 3.42,
            share_anchors = [
                (Date(2022, 9, 15), 81.0),    # BA.5.2 в сентябре 2022
                (Date(2022, 10, 15), 78.0),   # BA.5.2 в октябре 2022
            ],
            confidence = "средняя до ноября 2022, низкая для 2023",
            note = "BA.5.2 занимал 78–81 % в августе–октябре 2022; с ноября его доля снижалась в " *
                   "пользу рекомбинантов XBB*. В январе 2023 Роспотребнадзор всё ещё называл BA.5 " *
                   "доминирующим и сообщал о первом случае XBB.1.5; даты полной смены в найденных " *
                   "источниках нет, поэтому 2023 год отнесён к этому периоду условно."
        ),
    ]

    strains_table = DataFrame(
        period_start = [s.start for s in strains],
        period_end = [s.stop for s in strains],
        strain = [s.name for s in strains],
        who = [s.who for s in strains],
        R = [s.R for s in strains],
        incubation_days = [s.incubation_days for s in strains],
        confidence = [s.confidence for s in strains],
    )
end

# ╔═╡ 5a1c0005-0a1b-11f1-a001-5a9c2f1e7b01
begin
    """Линейная интерполяция доли между опорными точками; вне диапазона — NaN."""
    function interp_share(anchors, d::Date)
        isempty(anchors) && return NaN
        (d < anchors[1][1] || d > anchors[end][1]) && return NaN
        for k in 1:(length(anchors) - 1)
            d0, v0 = anchors[k]
            d1, v1 = anchors[k + 1]
            if d0 <= d <= d1
                t = Dates.value(d - d0) / max(Dates.value(d1 - d0), 1)
                return v0 + t * (v1 - v0)
            end
        end
        return anchors[end][2]   # единственная опорная точка и d совпадает с ней
    end

    """
        strain_at(d::Date)

    Показатели доминирующего в РФ штамма на дату `d`
    (`pandemic_start` ≤ d ≤ `pandemic_end`). Неизвестные значения — `NaN`.
    """
    function strain_at(d::Date)
        k = findfirst(s -> s.start <= d <= s.stop, strains)
        k === nothing && throw(ArgumentError(
            "Дата $d вне периода пандемии $pandemic_start – $pandemic_end"))
        s = strains[k]
        (
            date = d,
            strain = s.name,
            short = s.short,
            pango = s.pango,
            who = s.who,
            share_pct = interp_share(s.share_anchors, d),
            R = s.R,
            incubation_days = s.incubation_days,
            confidence = s.confidence,
            note = s.note,
        )
    end
end

# ╔═╡ 5a1c0006-0a1b-11f1-a001-5a9c2f1e7b01
md"""
## Запрос показателей на дату

Измените `query_date` — функция `strain_at` вернёт показатели доминирующего штамма.
"""

# ╔═╡ 5a1c0007-0a1b-11f1-a001-5a9c2f1e7b01
begin
    query_date = Date(2022, 4, 15)
    strain_at(query_date)
end

# ╔═╡ 5a1c0008-0a1b-11f1-a001-5a9c2f1e7b01
md"""
## Справочник периодов
"""

# ╔═╡ 5a1c0009-0a1b-11f1-a001-5a9c2f1e7b01
strains_table

# ╔═╡ 5a1c0010-0a1b-11f1-a001-5a9c2f1e7b01
begin
    all_dates = collect(pandemic_start:Day(1):pandemic_end)
    timeline = DataFrame(strain_at.(all_dates))
    first(timeline[:, Not([:note, :pango])], 3)
end

# ╔═╡ 5a1c0011-0a1b-11f1-a001-5a9c2f1e7b01
begin
    """График одного показателя за всё время пандемии: границы периодов и подписи штаммов."""
    function plot_indicator(col::Symbol; ylabel, title, ylims_ = nothing, scatter_anchors = false)
        y = timeline[!, col]
        finite = filter(isfinite, y)
        top = isempty(finite) ? 1.0 : maximum(finite)
        ceil_top = isnothing(ylims_) ? 1.15 * top : ylims_[2]

        p = plot(
            timeline.date, y,
            lw = 3, label = "", color = :steelblue,
            xlabel = "Дата", ylabel = ylabel, title = title,
            ylims = isnothing(ylims_) ? (0.0, ceil_top) : ylims_,
            size = (900, 380), legend = false, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
        )
        for s in strains
            vline!([s.start], color = :gray, ls = :dash, alpha = 0.5, label = "")
            mid = s.start + (s.stop - s.start) ÷ 2
            annotate!(mid, 0.96 * ceil_top, text(s.short, 8, :center, :gray30))
        end
        if scatter_anchors
            for s in strains, (d, v) in s.share_anchors
                scatter!([d], [v], color = :crimson, markersize = 5, label = "")
            end
        end
        # Периоды без данных (NaN) подписываем
        if any(isnan, y)
            nan_idx = findall(isnan, y)
            annotate!(timeline.date[nan_idx[1]] + Day(30), 0.45 * ceil_top,
                text("нет данных", 9, :left, :gray50))
        end
        p
    end
end

# ╔═╡ 5a1c0012-0a1b-11f1-a001-5a9c2f1e7b01
md"""
## График 1. Доля доминирующей линии среди секвенированных образцов

Красные точки — опорные значения из источников, между ними линейная интерполяция.
Для 2020 – апреля 2021 и для 2023 года числовых долей в найденных источниках нет.
"""

# ╔═╡ 5a1c0013-0a1b-11f1-a001-5a9c2f1e7b01
plot_indicator(
    :share_pct;
    ylabel = "Доля, %", title = "Доля доминирующей линии (РФ)",
    ylims_ = (0.0, 105.0), scatter_anchors = true
)

# ╔═╡ 5a1c0014-0a1b-11f1-a001-5a9c2f1e7b01
md"""
## График 2. Репродуктивное число доминирующего штамма

Для Омикрона приведены эффективные числа Rₑ (объединённая оценка), для остальных — R₀.
Прямое сравнение уровней некорректно из-за разных методик. Для B.1.1.317/397 оценки нет.
"""

# ╔═╡ 5a1c0015-0a1b-11f1-a001-5a9c2f1e7b01
plot_indicator(:R; ylabel = "R₀ / Rₑ", title = "Репродуктивное число доминирующего штамма")

# ╔═╡ 5a1c0016-0a1b-11f1-a001-5a9c2f1e7b01
md"""
## График 3. Средний инкубационный период

Сводная оценка метаанализа: убывает от предкового типа (≈6.6 сут) к Омикрону (≈3.4 сут).
Значения Дельты и Омикрона — по метаанализу по штаммам; значимых различий между
подлиниями Омикрона не найдено, поэтому для всех BA.* одно значение.
"""

# ╔═╡ 5a1c0017-0a1b-11f1-a001-5a9c2f1e7b01
plot_indicator(
    :incubation_days;
    ylabel = "Сутки", title = "Средний инкубационный период"
)

# ╔═╡ 5a1c0018-0a1b-11f1-a001-5a9c2f1e7b01
md"""
## Примечания к данным

- Нет ряда по тяжести течения (летальность, госпитализации) по штаммам — в найденных
  источниках не нашлось сопоставимых оценок для РФ, поэтому показатель не включён.
- R₀ для предкового штамма (2.79) и Дельты (5.08) — обзор Liu & Rocklöv (2021). Значения
  Омикрона — объединённые оценки Rₑ; выведенные оттуда R₀ ≈ 11–18 получены умножением на
  относительную трансмиссивность и здесь не используются.
- Инкубационный период — метаанализ по штаммам: Дельта 4.41, Омикрон 3.42, сводная
  оценка по всем штаммам 6.57 сут (для предкового типа и линий 2020 года использована как
  ориентир).
- Границы периодов — по месяцам публикаций, не по точным датам смены доминирования.
- 2023 год: данные по долям отсутствуют; смена BA.5 → XBB* для РФ в источниках не
  датирована.

### Источники

- Spread of endemic SARS-CoV-2 lineages in Russia (PLOS ONE): <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0270717>
- The rise and spread of the SARS-CoV-2 AY.122 lineage in Russia: <https://www.medrxiv.org/content/10.1101/2021.12.02.21267168.full.pdf>
- Real-time PCR screening of Omicron sublineages, VGARus (Genes, 2023): <https://doi.org/10.3390/genes14061218>
- Дельта-вариант SARS-CoV-2 (Википедия): <https://ru.wikipedia.org/wiki/%D0%94%D0%B5%D0%BB%D1%8C%D1%82%D0%B0-%D0%B2%D0%B0%D1%80%D0%B8%D0%B0%D0%BD%D1%82_SARS-CoV-2>
- Омикрон-вариант SARS-CoV-2 (Википедия): <https://ru.wikipedia.org/wiki/%D0%9E%D0%BC%D0%B8%D0%BA%D1%80%D0%BE%D0%BD-%D1%88%D1%82%D0%B0%D0%BC%D0%BC_SARS-CoV-2>
- Incubation period by SARS-CoV-2 strain, systematic review (JAMA Netw Open 2022): <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9396366/>
- Первый случай XBB.1.5 в РФ (Интерфакс, январь 2023): <https://www.interfax.ru/russia/880505>
- Liu & Rocklöv, reproduction number of Delta: <https://insight.jci.org/references/scholar/4300/B7>
"""

# ╔═╡ Cell order:
# ╠═5a1c0001-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0002-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0003-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0004-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0005-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0006-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0007-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0008-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0009-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0010-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0011-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0012-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0013-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0014-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0015-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0016-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0017-0a1b-11f1-a001-5a9c2f1e7b01
# ╠═5a1c0018-0a1b-11f1-a001-5a9c2f1e7b01
