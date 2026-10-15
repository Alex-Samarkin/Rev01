### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 98642bf0-c7d1-11f1-b154-d54c439aaaa5
begin
    import Pkg
    Pkg.activate("@SciTools")
    Pkg.add([
        "CSV", "DataFrames", "Dates", "Plots", "Optim",
        "OrdinaryDiffEq", "Statistics", "Random"
    ])
    # Свежие версии FractionalDiffEq не прекомпилируются (overwriting genfun) — закрепляем 0.5.1
    Pkg.add(name = "FractionalDiffEq", version = "0.5.1")
end

# ╔═╡ 9864ef40-c7d1-11f1-afd9-6f2a9cb58f53
begin
    using CSV, DataFrames, Dates, Plots, Optim, Statistics, Random
    using FractionalDiffEq, OrdinaryDiffEq
    import OrdinaryDiffEq: solve   # снимает неоднозначность имени `solve`
end

# ╔═╡ 9864ef40-c7d1-11f1-b173-2bd778287a61
md"""
# Идентификация параметров волны заболеваемости: Caputo-дробная SEIR-модель

**Данные:** еженедельные данные по Москве, опубликованные на HDX.  
Первичный источник: правительство Москвы.  
Страница набора данных: <https://data.humdata.org/dataset/covid-19-cases-data-in-moscow>

Скачайте файл `covid_data_moscow_region_weekly_2021CSV` и укажите путь к нему ниже.
"""

# ╔═╡ 9864ef40-c7d1-11f1-b2aa-2973f7b8242c
begin
    # data_path = "covid_data_moscow_region_weekly_2021.csv"
    # data_path = "csv_moskov.csv"
    # Волны по Москве (CSV, см. папку data/):
    data_path = "data/moscow_wave1_autumn_winter_2020.csv"  # 2020-09-01 → 2021-03-13
    # data_path = "data/moscow_wave2_delta_2021.csv"          # 2021-05-25 → 2021-09-01
    # data_path = "data/moscow_wave3_autumn_2021.csv"         # 2021-09-01 → 2022-01-03

    if !isfile(data_path)
        @warn "Файл данных не найден. Скачайте CSV со страницы HDX и укажите корректный путь."
    end
end

# ╔═╡ 9864ef40-c7d1-11f1-9eb6-d3742a8962ac
begin
    raw = CSV.read(data_path, DataFrame)
    println(names(raw))
    first(raw, 5)
end

# ╔═╡ 9864ef40-c7d1-11f1-aa57-2b324000dab3
begin
    date_candidates = ["date", "Date", "дата", "week", "Week"]
    cases_candidates = [
        "cases", "Cases", "new_cases", "new_cases_weekly",
        "заболевших", "случаев", "cases_weekly", "sick"
    ]
    
    date_col = first(filter(x -> x in names(raw), date_candidates))
    cases_col = first(filter(x -> x in names(raw), cases_candidates))
    
    df = dropmissing(select(raw, [date_col, cases_col]))
    rename!(df, [date_col => :date, cases_col => :cases])
    df.date = Date.(df.date,DateFormat("dd.mm.yyyy"))
    sort!(df, :date)
    
    first(df, 10)
end

# ╔═╡ 9864ef40-c7d1-11f1-bce3-3b91b8657128
md"""
## Подготовка ряда

Столбец `sick` — **накопленное** число случаев, поэтому интенсивность считается
по приращениям с учётом фактического шага между датами (1 день для суточных данных, 7 — для недельных):
```math

y_i=\frac{C_i - C_{i-1}}{t_i - t_{i-1}}.

```
"""

# ╔═╡ 9864ef40-c7d1-11f1-bcd9-ed23986dbbeb
begin
    # Границы окна должны попадать в диапазон данных выбранного файла
    # (csv_moskov.csv: 06.01.2021–04.02.2021; wave1: 01.09.2020–13.03.2021)
    # wave_start = Date(2021, 1, 1)
    # wave_end = Date(2021, 1, 31)
    # Осень–зима 2020/21 (data/moscow_wave1_autumn_winter_2020.csv)
    wave_start = Date(2020, 9, 1)
    wave_end = Date(2021, 1, 31)

    in_window = filter(row -> wave_start <= row.date <= wave_end, df)
    nrow(in_window) >= 3 || error("В окне [$wave_start, $wave_end] меньше 3 точек данных")
    
    step_days = Dates.value.(diff(in_window.date))
    wave = in_window[2:end, :]
    wave.daily_rate = diff(Float64.(in_window.cases)) ./ step_days
    # wave.smoothed = [
    #     mean(wave.daily_rate[max(1, i - 1):min(end, i + 1)])
    #     for i in eachindex(wave.daily_rate)
    # ]
    # Центрированное среднее за 7 суток (убирает недельную периодичность отчётности)
    wave.smoothed = [
        mean(wave.daily_rate[max(1, i - 3):min(end, i + 3)])
        for i in eachindex(wave.daily_rate)
    ]
    
    plot(
        wave.date, wave.daily_rate,
        label = "Новые случаи в сутки",
        xlabel = "Дата",
        ylabel = "Случаев в сутки",
        legend = :topleft
    )
    plot!(wave.date, wave.smoothed, lw = 3, label = "Сглаженный ряд")
end

# ╔═╡ 9864ef40-c7d1-11f1-b5f6-cd63c50bdf37
md"""
## Модель

```math

\begin{aligned}
{}^C D_t^\alpha S &= -\beta S I/N,\\
{}^C D_t^\alpha E &= \beta S I/N-\sigma E,\\
{}^C D_t^\alpha I &= \sigma E-\gamma I,\\
{}^C D_t^\alpha R &= \gamma I.
\end{aligned}

```
Наблюдаемая переменная:

```math
\hat y(t)=q\sigma E(t).
```

Начальное состояние привязано к первому наблюдению:
``E_0 = y_1/(q\sigma)``, ``I_0 = E_0\sigma/\gamma``.
"""

# ╔═╡ 9864ef40-c7d1-11f1-864a-672cbf0f8644
begin
    N = 12_678_000.0
    
    # Время в сутках от первой даты окна (а не номер наблюдения)
    observation_times = Float64.(Dates.value.(wave.date .- wave.date[1]))
    tspan = (0.0, observation_times[end])
    
    function initial_state(sigma, gamma, scale)
        E0 = wave.smoothed[1] / (scale * sigma)
        I0 = E0 * sigma / gamma
        [N - E0 - I0, E0, I0, 0.0]
    end
end

# ╔═╡ 9864ef40-c7d1-11f1-8f0b-2bed60fb58d0
begin
    function seir_rhs(u, p, t)
        beta, sigma, gamma, scale = p
        S, E, I, R = u
        [
            -beta * S * I / N,
            beta * S * I / N - sigma * E,
            sigma * E - gamma * I,
            gamma * I
        ]
    end
##
#    function fractional_seir_rhs(u, p, t)
#        alpha, beta, sigma, gamma, scale = p
#        S, E, I, R = u
#        [
#            -beta * S * I / N,
#            beta * S * I / N - sigma * E,
#            sigma * E - gamma * I,
#            gamma * I
#        ]
#    end
##

    function make_fractional_rhs(beta, sigma, gamma, scale)
        return function (du, u, p, t)
            S, E, I, R = u

            du[1] = -beta * S * I / N
            du[2] =  beta * S * I / N - sigma * E
            du[3] =  sigma * E - gamma * I
            du[4] =  gamma * I

            return nothing
        end
    end
    
end

#

# ╔═╡ 9864ef40-c7d1-11f1-a386-213273edd593
begin
    observed = Float64.(wave.smoothed)
    
    function solve_classical(p)
        beta, sigma, gamma, scale = p
        u0 = initial_state(sigma, gamma, scale)
        prob = ODEProblem(seir_rhs, u0, tspan, p)
        sol = solve(
            prob, Tsit5();
            saveat = observation_times,
            abstol = 1e-8,
            reltol = 1e-8
        )
        length(sol.t) == length(observation_times) || error("ODE solver failed: $(sol.retcode)")
        scale * sigma .* sol[2, :]
    end
    
    function solve_fractional(p; dt = 0.25)
        alpha, beta, sigma, gamma, scale = p
        u0 = initial_state(sigma, gamma, scale)
    
        rhs = make_fractional_rhs(beta, sigma, gamma, scale)
    
        # Для системы порядок задаётся вектором по числу уравнений:
        # скалярный alpha при length(u0) = 4 PECE воспринимает как задачу размерности 1
        prob = FODEProblem(rhs, fill(alpha, length(u0)), u0, tspan)
    
        sol = solve(prob, PECE(); dt = dt)
    
        obs_indices = [argmin(abs.(sol.t .- t)) for t in observation_times]
    
        E = [sol.u[i][2] for i in obs_indices]
        all(isfinite, E) || error("FODE solution is not finite")
        return scale * sigma .* E
    end
end

# ╔═╡ 9864ef40-c7d1-11f1-aa3e-bb61602cee0f
begin
    # Параметры вне разумных границ отвергаются (иначе Нелдер–Мид уходит в вырожденные решения)
    in_bounds(beta, sigma, gamma, scale) =
        0.01 <= beta <= 5 && 0.02 <= sigma <= 1 && 0.02 <= gamma <= 1 && 1e-3 <= scale <= 1
    
    function loss_classical(theta)
        p = exp.(theta)
        in_bounds(p...) || return 1e30
        pred = try
            solve_classical(p)
        catch err
            @debug "loss_classical failed" p err
            return 1e30
        end
        length(pred) == length(observed) || return 1e30
        mean((pred .- observed).^2)
    end
    
    function loss_fractional(theta)
        beta, sigma, gamma, scale = exp.(theta[1:4])
        alpha = 1 / (1 + exp(-theta[5]))
        in_bounds(beta, sigma, gamma, scale) || return 1e30
        p = [alpha, beta, sigma, gamma, scale]
    
        pred = try
            solve_fractional(p)
        catch err
            @debug "loss_fractional failed" p err
            return 1e30
        end
        length(pred) == length(observed) || return 1e30
    
        mean((pred .- observed).^2)
    end
end

# ╔═╡ 9864ef40-c7d1-11f1-930c-2d821e3c113f
begin
    # --- Старый вариант: подгонка всех параметров из одной стартовой точки ---
    # theta0_classical = log.([0.35, 1/5, 1/10, 0.5])
    # theta0_fractional = vcat(log.([0.35, 1/5, 1/10, 0.5]), [2.0])  # alpha0 ≈ 0.88
    #
    # result_classical = optimize(
    #     loss_classical,
    #     theta0_classical,
    #     NelderMead(),
    #     Optim.Options(iterations = 2000)
    # )
    #
    # result_fractional = optimize(
    #     loss_fractional,
    #     theta0_fractional,
    #     NelderMead(),
    #     Optim.Options(iterations = 3000)
    # )

    # --- Новый вариант: sigma и gamma фиксированы (литература), мульти-старт ---
    sigma_fixed = 1 / 5.5     # инкубация ≈ 5.5 суток
    gamma_fixed = 1 / 10.0    # длительность заразности ≈ 10 суток
    n_starts = 12

    # Подгоняем theta = [log beta, log q] (классическая) и [log beta, log q, logit alpha] (дробная)
    loss_classical_fixed(th) =
        loss_classical([th[1], log(sigma_fixed), log(gamma_fixed), th[2]])
    loss_fractional_fixed(th) =
        loss_fractional([th[1], log(sigma_fixed), log(gamma_fixed), th[2], th[3]])

    Random.seed!(2024)
    starts_beta = exp.(range(log(0.15), log(1.5), length = n_starts))
    starts_scale = exp.(range(log(0.02), log(0.5), length = n_starts))[randperm(n_starts)]
    starts_alpha = range(0.5, 0.95, length = n_starts)[randperm(n_starts)]

    function best_of(loss, starts; iterations)
        results = [
            optimize(loss, s, NelderMead(), Optim.Options(iterations = iterations))
            for s in starts
        ]
        results[argmin(Optim.minimum.(results))], Optim.minimum.(results)
    end

    best_classical, losses_classical = best_of(
        loss_classical_fixed,
        [[log(b), log(q)] for (b, q) in zip(starts_beta, starts_scale)];
        iterations = 1000
    )
    best_fractional, losses_fractional = best_of(
        loss_fractional_fixed,
        [[log(b), log(q), log(a / (1 - a))] for (b, q, a) in zip(starts_beta, starts_scale, starts_alpha)];
        iterations = 1000
    )

    # Совместимость с ячейками ниже: восстанавливаем полный вектор theta
    result_classical = best_classical
    result_fractional = best_fractional
    theta_classical_full = [Optim.minimizer(best_classical)[1], log(sigma_fixed), log(gamma_fixed), Optim.minimizer(best_classical)[2]]
    theta_fractional_full = [Optim.minimizer(best_fractional)[1], log(sigma_fixed), log(gamma_fixed),
                             Optim.minimizer(best_fractional)[2], Optim.minimizer(best_fractional)[3]]

    result_classical, result_fractional
end

# ╔═╡ 9864ef40-c7d1-11f1-a56f-659763d49c26
begin
    # theta_classical = Optim.minimizer(result_classical)
    # theta_fractional = Optim.minimizer(result_fractional)
    theta_classical = theta_classical_full
    theta_fractional = theta_fractional_full
    
    beta_c, sigma_c, gamma_c, scale_c = exp.(theta_classical)
    beta_f, sigma_f, gamma_f, scale_f = exp.(theta_fractional[1:4])
    alpha_f = 1 / (1 + exp(-theta_fractional[5]))
    
    pred_classical = solve_classical([beta_c, sigma_c, gamma_c, scale_c])
    pred_fractional = solve_fractional([alpha_f, beta_f, sigma_f, gamma_f, scale_f])
    
    DataFrame(
        model = ["classical SEIR", "fractional Caputo SEIR"],
        alpha = [1.0, alpha_f],
        beta = [beta_c, beta_f],
        sigma = [sigma_c, sigma_f],
        gamma = [gamma_c, gamma_f],
        detection_scale = [scale_c, scale_f],
        RMSE = [
            sqrt(mean((pred_classical .- observed).^2)),
            sqrt(mean((pred_fractional .- observed).^2))
        ],
        MAE = [
            mean(abs.(pred_classical .- observed)),
            mean(abs.(pred_fractional .- observed))
        ]
    )
end

# ╔═╡ 9864ef40-c7d1-11f1-9cd7-7ddeb47e4ea7
begin
    plot(
        wave.date, observed,
        label = "Данные",
        xlabel = "Дата",
        ylabel = "Новых случаев в сутки",
        legend = :topleft,
        lw = 2
    )
    plot!(wave.date, pred_classical, label = "Классическая SEIR", lw = 2, ls = :dash)
    plot!(wave.date, pred_fractional, label = "Caputo-дробная SEIR", lw = 2, ls = :dot)
end

# ╔═╡ 9864ef40-c7d1-11f1-a001-5a9c2f1e7b01
begin
    # Остатки (модель − данные) и проверка, не упёрлись ли параметры в границы in_bounds
    plot(
        wave.date, pred_classical .- observed,
        label = "Классическая SEIR", lw = 2,
        xlabel = "Дата", ylabel = "Остаток, случаев в сутки", legend = :topleft
    )
    plot!(wave.date, pred_fractional .- observed, label = "Caputo-дробная SEIR", lw = 2, ls = :dash)
    hline!([0.0], color = :black, label = "")
end

# ╔═╡ 9864ef40-c7d1-11f1-a002-5a9c2f1e7b02
begin
    bound_check = DataFrame(
        parameter = ["beta", "sigma", "gamma", "scale"],
        classical = [beta_c, sigma_c, gamma_c, scale_c],
        fractional = [beta_f, sigma_f, gamma_f, scale_f],
        lower = [0.01, 0.02, 0.02, 1e-3],
        upper = [5.0, 1.0, 1.0, 1.0],
    )
    bound_check.classical_at_bound = [
        !(1.01 * l < v < 0.99 * u) for (v, l, u) in zip(bound_check.classical, bound_check.lower, bound_check.upper)
    ]
    bound_check.fractional_at_bound = [
        !(1.01 * l < v < 0.99 * u) for (v, l, u) in zip(bound_check.fractional, bound_check.lower, bound_check.upper)
    ]
    peak_day = DataFrame(
        series = ["Данные", "Классическая", "Дробная"],
        peak_day = [argmax(observed), argmax(pred_classical), argmax(pred_fractional)] .- 1,
        peak_value = [maximum(observed), maximum(pred_classical), maximum(pred_fractional)],
    )
    bound_check, peak_day
end

# ╔═╡ 9864ef40-c7d1-11f1-a003-5a9c2f1e7b03
begin
    # Устойчивость мульти-старта: итоговый лосс (RMSE) по каждой из стартовых точек.
    # Если все точки лежат на одном уровне, минимум единственный и фит устойчив.
    plot(
        1:n_starts, sqrt.(losses_classical),
        seriestype = :scatter, label = "Классическая SEIR",
        xlabel = "Номер стартовой точки", ylabel = "RMSE после оптимизации",
        legend = :topright, markersize = 6
    )
    plot!(
        1:n_starts, sqrt.(losses_fractional),
        seriestype = :scatter, label = "Caputo-дробная SEIR", markershape = :diamond, markersize = 6
    )
end

# ╔═╡ 9864ef40-c7d1-11f1-a004-5a9c2f1e7b04
begin
    # Предсказание против данных: точки у диагонали — хорошее соответствие
    lim = (0.0, 1.05 * maximum(observed))
    plot(
        observed, pred_classical,
        seriestype = :scatter, label = "Классическая SEIR",
        xlabel = "Данные, случаев в сутки", ylabel = "Модель, случаев в сутки",
        xlims = lim, ylims = lim, legend = :topleft, alpha = 0.6
    )
    plot!(observed, pred_fractional, seriestype = :scatter, label = "Caputo-дробная SEIR", alpha = 0.6)
    plot!([0.0, lim[2]], [0.0, lim[2]], color = :black, ls = :dash, label = "y = x")
end

# ╔═╡ 9864ef40-c7d1-11f1-937e-5569d2afbbe8
md"""
## Диагностика

1. Повторите оптимизацию из разных начальных точек.
2. Проверьте устойчивость к изменению шага численного метода.
3. Оцените интервалы неопределенности бутстреп-методом.
4. Проверьте прогноз на 7–14 дней вне обучающего окна.
5. Проанализируйте чувствительность к выбору границ волны.
"""

# ╔═╡ Cell order:
# ╠═98642bf0-c7d1-11f1-b154-d54c439aaaa5
# ╠═9864ef40-c7d1-11f1-afd9-6f2a9cb58f53
# ╠═9864ef40-c7d1-11f1-b173-2bd778287a61
# ╠═9864ef40-c7d1-11f1-b2aa-2973f7b8242c
# ╠═9864ef40-c7d1-11f1-9eb6-d3742a8962ac
# ╠═9864ef40-c7d1-11f1-aa57-2b324000dab3
# ╠═9864ef40-c7d1-11f1-bce3-3b91b8657128
# ╠═9864ef40-c7d1-11f1-bcd9-ed23986dbbeb
# ╠═9864ef40-c7d1-11f1-b5f6-cd63c50bdf37
# ╠═9864ef40-c7d1-11f1-864a-672cbf0f8644
# ╠═9864ef40-c7d1-11f1-8f0b-2bed60fb58d0
# ╠═9864ef40-c7d1-11f1-a386-213273edd593
# ╠═9864ef40-c7d1-11f1-aa3e-bb61602cee0f
# ╠═9864ef40-c7d1-11f1-930c-2d821e3c113f
# ╠═9864ef40-c7d1-11f1-a56f-659763d49c26
# ╠═9864ef40-c7d1-11f1-9cd7-7ddeb47e4ea7
# ╠═9864ef40-c7d1-11f1-a001-5a9c2f1e7b01
# ╠═9864ef40-c7d1-11f1-a002-5a9c2f1e7b02
# ╠═9864ef40-c7d1-11f1-a003-5a9c2f1e7b03
# ╠═9864ef40-c7d1-11f1-a004-5a9c2f1e7b04
# ╠═9864ef40-c7d1-11f1-937e-5569d2afbbe8