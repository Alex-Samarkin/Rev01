# -*- coding: utf-8 -*-
# check_notebook_core.jl — проверка вычислительного ядра Pluto-блокнота
# wave_identification_pluto.jl без установки пакетов Plots / DataFrames / CSV / Optim.
#
# Что проверяется:
#   1) структура блокнота: уникальность UUID ячеек, полнота секции Cell order,
#      корректность всех md-ячеек (в них нет «сломанных» escape-последовательностей);
#   2) те же вычисления, что в код-ячейках блокнота (веса L1, сходимость схемы,
#      семейство решений при разных alpha, чтение waves.csv, идентификация
#      параметров Москвы и сводка alpha по всем территориям).
#
# Запуск:
#     julia --startup-file=no --compiled-modules=no code/julia/check_notebook_core.jl
using Markdown
using Printf
using Statistics

include(joinpath(@__DIR__, "fractional_sir.jl"))
using .FractionalSIR

const NOTEBOOK = joinpath(@__DIR__, "wave_identification_pluto.jl")
const CELL_RE = r"^# ╔═╡ ([0-9a-fA-F-]{36})\s*$"m

println("="^78)
println("1. ПРОВЕРКА СТРУКТУРЫ БЛОКНОТА")
println("="^78)

src = read(NOTEBOOK, String)
lines = split(src, "\n")
println("первая строка: ", strip(lines[1]))
@assert strip(lines[1]) == "### A Pluto.jl notebook ###" "неверный заголовок Pluto-блокнота"

# Разбор ячеек
cells = Tuple{String,String}[]           # (uuid, тело ячейки)
ms = collect(eachmatch(CELL_RE, src))
offsets = [m.offset for m in ms]
uuids = [String(m.captures[1]) for m in ms]
for (k, m) in enumerate(ms)
    body_start = m.offset + sizeof(m.match)
    body_end = k < length(offsets) ? offsets[k + 1] - 1 : sizeof(src)
    body = src[body_start:body_end]
    body = replace(body, r"\n# ╔═╡ Cell order:[\s\S]*$" => "\n")
    push!(cells, (uuids[k], body))
end
println("найдено ячеек: ", length(cells))
println("уникальных UUID: ", length(unique(uuids)))
@assert length(unique(uuids)) == length(uuids) "дублируются UUID ячеек!"

order_section = occursin("# ╔═╡ Cell order:", src)
@assert order_section "нет секции Cell order"
ordered = [String(m.captures[1]) for m in eachmatch(r"^# [╠╟][═─]([0-9a-fA-F-]{36})\s*$"m, src)]
println("UUID в секции Cell order: ", length(ordered))
@assert Set(ordered) == Set(uuids) "Cell order не совпадает с набором ячеек"

function evaluate_md_cells(cells)
    n_md = 0
    n_code = 0
    for (u, body) in cells
        t = strip(body)
        if startswith(t, "md\"")
            n_md += 1
            # Реально вычисляем md-ячейку: так ловятся некорректные
            # escape-последовательности и незакрытые строки.
            include_string(Main, t, "notebook_cell_$u")
        elseif !isempty(t)
            n_code += 1
        end
    end
    return n_md, n_code
end

n_md, n_code = evaluate_md_cells(cells)
println("md-ячеек: $n_md (все успешно вычислены), код-ячеек: $n_code")

println()
println("="^78)
println("2. ВЫЧИСЛИТЕЛЬНОЕ ЯДРО БЛОКНОТА (без Plots/DataFrames/CSV/Optim)")
println("="^78)

# --- ячейка 4: веса L1 и телескопическая сумма --------------------------------
let
    α = 0.5
    b = FractionalSIR.l1_weights(α, 8)
    m = 5000
    s = sum(FractionalSIR.l1_weights(α, m))
    @printf("веса L1 при alpha=0.5: b_0..b_3 = %.6f %.6f %.6f %.6f  (b_0 = 1)\n",
            b[1], b[2], b[3], b[4])
    @printf("sum b_k = %.10f,  (m+1)^(1-alpha) = %.10f,  |разность| = %.2e\n",
            s, (m + 1)^(1 - α), abs(s - (m + 1)^(1 - α)))
end

# --- ячейки 8-9: сходимость схемы L1 к E_{1/2}(-1) ----------------------------
let
    exact = exp(1.0) * FractionalSIR.erfc_fn(1.0)
    ns = [100, 200, 400, 800, 1600, 3200]
    errs = [abs(FractionalSIR.l1_scalar(0.5, 1.0, 1.0, n)[end] - exact) for n in ns]
    @printf("E_{1/2}(-1) = %.10f\n", exact)
    for (n, e) in zip(ns, errs)
        @printf("  n = %5d, |err| = %.3e\n", n, e)
    end
    ord = [log(errs[i] / errs[i + 1]) / log(2.0) for i in 1:(length(ns) - 1)]
    @printf("наблюдаемый порядок: %s (теория 1.0)\n",
            join([@sprintf("%.3f", o) for o in ord], " "))
end

# --- ячейки 12-13: семейство решений при разных alpha -------------------------
let
    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]
    n_days = 120
    A0, λ0, τ0 = 100_000.0, 0.2, 7.0
    println("семейство решений (lambda = 0.2, tau = 7, A = 1e5, i0 = 1):")
    for a in alphas
        β, γ, R0 = FractionalSIR.rates_from_lambda_tau(λ0, τ0, a)
        C = FractionalSIR.simulate_sir(a, β, γ, A0, n_days; i0 = 1.0, steps_per_day = 10)
        @printf("  alpha = %.1f: R0 = %.3f, beta = %.5f, gamma = %.5f, C(120 сут) = %.1f\n",
                a, R0, β, γ, C[end])
    end
end

# --- ячейка 15: чтение waves.csv ---------------------------------------------
waves = FractionalSIR.load_waves()
println("прочитано территорий: ", length(waves))
for w in waves
    @printf("  %-24s %-8s N = %12.0f, n = %3d, C_end = %12.1f, пик MA7 = %10.1f\n",
            w.territory, w.country, w.population, length(w.dates),
            w.cumulative[end], maximum(w.daily_ma7))
end

# --- ячейки 19-20: идентификация выбранной территории ------------------------
let
    name = "Москва"
    w = FractionalSIR.wave_by_name(waves, name)
    fc = FractionalSIR.fit_wave(w, :classic)
    ff = FractionalSIR.fit_wave(w, :fractional)
    for f in (fc, ff)
        @printf("%-24s %-11s alpha = %.4f, lambda = %.4f, R0 = %.3f, A = %.1f, i0 = %.1f, RMSE = %.4f, R2 = %.5f, AIC = %.3f\n",
                f.territory, string(f.model), f.params["alpha"], f.params["lam"],
                f.params["R0"], f.params["A"], f.params["i0"], f.rmse, f.r2, f.aic)
    end
    wts = FractionalSIR.aic_weights([fc.aic, ff.aic])
    @printf("вес Акаике: classic %.4f / fractional %.4f;  RMSE-выигрыш %.1f %%\n",
            wts[1], wts[2], 100 * (1 - ff.rmse / fc.rmse))
end

# --- ячейка 26: сводка alpha по территориям (n_starts = 12, как в блокноте) ---
let
    println("сводка по территориям (n_starts = 12):")
    for w in waves
        f = FractionalSIR.fit_wave(w, :fractional; n_starts = 12)
        c = FractionalSIR.fit_wave(w, :classic; n_starts = 12)
        @printf("  %-24s alpha = %.4f, R0 = %.3f, RMSE frac = %8.2f, RMSE classic = %8.2f\n",
                w.territory, f.params["alpha"], f.params["R0"], f.rmse, c.rmse)
    end
end

println()
println("ЯДРО БЛОКНОТА: ПРОВЕРКА ЗАВЕРШЕНА УСПЕШНО")
