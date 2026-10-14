### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ Cell 0001
begin
    import Pkg
    Pkg.activate("@SciTools")
    Pkg.add([
        "CSV", "DataFrames", "Dates", "Plots", "Optim",
        "FractionalDiffEq", "Statistics", "Random"
    ])
end

##
# ╔═╡ Cell 0002
begin
    using CSV, DataFrames, Dates, Plots, Optim, Statistics, Random
    using FractionalDiffEq
end

##
# ╔═╡ Cell 0003
md"""
# Идентификация параметров волны заболеваемости: Caputo-дробная SEIR-модель

**Данные:** еженедельные данные по Москве, опубликованные на HDX.  
Первичный источник: правительство Москвы.  
Страница набора данных: <https://data.humdata.org/dataset/covid-19-cases-data-in-moscow>

Скачайте файл `covid_data_moscow_region_weekly_2021CSV` и укажите путь к нему ниже.
"""
##
# ╔═╡ Cell 0004
begin
    # data_path = "covid_data_moscow_region_weekly_2021.csv"
    data_path = "csv_moskov.csv"

    if !isfile(data_path)
        @warn "Файл данных не найден. Скачайте CSV со страницы HDX и укажите корректный путь."
    end
end
##
# ╔═╡ Cell 0005
begin
    raw = CSV.read(data_path, DataFrame)
    println(names(raw))
    first(raw, 5)
end

##
# ╔═╡ Cell 0006
begin
    date_candidates = ["date", "Date", "дата", "week", "Week"]
    cases_candidates = [
        "cases", "Cases", "new_cases", "new_cases_weekly",
        "заболевших", "случаев", "cases_weekly"
    ]

    date_col = first(filter(x -> x in names(raw), date_candidates))
    cases_col = first(filter(x -> x in names(raw), cases_candidates))

    df = dropmissing(select(raw, [date_col, cases_col]))
    rename!(df, [date_col => :date, cases_col => :cases])
    df.date = Date.(df.date)
    sort!(df, :date)

    first(df, 10)
end


### ╔═╡ Cell 0007
md"""
## Подготовка ряда

Еженедельные случаи пересчитываются в среднесуточную интенсивность:

\[
y_i=\frac{\Delta C_i}{7}.
\]
"""

# ╔═╡ Cell 0008
begin
    wave_start = Date(2020, 9, 1)
    wave_end = Date(2021, 1, 31)

    wave = filter(row -> wave_start <= row.date <= wave_end, df)
    wave.daily_rate = wave.cases ./ 7.0
    wave.smoothed = [
        mean(wave.daily_rate[max(1, i - 1):min(end, i + 1)])
        for i in eachindex(wave.daily_rate)
    ]

    plot(
        wave.date, wave.daily_rate,
        label = "Недельные случаи / 7",
        xlabel = "Дата",
        ylabel = "Случаев в сутки",
        legend = :topleft
    )
    plot!(wave.date, wave.smoothed, lw = 3, label = "Сглаженный ряд")
end

### ╔═╡ Cell 0009
md"""
## Модель

\[
\begin{aligned}
{}^C D_t^\alpha S &= -\beta S I/N,\\
{}^C D_t^\alpha E &= \beta S I/N-\sigma E,\\
{}^C D_t^\alpha I &= \sigma E-\gamma I,\\
{}^C D_t^\alpha R &= \gamma I.
\end{aligned}
\]

Наблюдаемая переменная:

\[
\hat y(t)=q\sigma E(t).
\]
"""

##
# ╔═╡ Cell 0010
begin
    N = 12_678_000.0
    E0 = 500.0
    I0 = 500.0
    R0 = 0.0
    S0 = N - E0 - I0 - R0

    u0 = [S0, E0, I0, R0]
    tspan = (0.0, Float64(length(wave.smoothed) - 1))
end

##
# ╔═╡ Cell 0011
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

    function fractional_seir_rhs(u, p, t)
        alpha, beta, sigma, gamma, scale = p
        S, E, I, R = u
        [
            -beta * S * I / N,
            beta * S * I / N - sigma * E,
            sigma * E - gamma * I,
            gamma * I
        ]
    end
end

##
# ╔═╡ Cell 0012
begin
    observation_times = collect(tspan[1]:1.0:tspan[2])
    observed = Float64.(wave.smoothed)

    function solve_classical(p)
        prob = ODEProblem(seir_rhs, u0, tspan, p)
        sol = solve(
            prob, Tsit5();
            saveat = observation_times,
            abstol = 1e-8,
            reltol = 1e-8
        )
        p[4] .* sol[2, :]
    end

    function solve_fractional(p)
        alpha, beta, sigma, gamma, scale = p
        prob = FODEProblem(fractional_seir_rhs, alpha, u0, tspan, p[2:end])
        sol = solve(prob, PECE(); dt = 0.25, saveat = observation_times)
        scale .* sol[2, :]
    end
end

##
# ╔═╡ Cell 0013
begin
    function loss_classical(theta)
        p = exp.(theta)
        pred = try
            solve_classical(p)
        catch
            return 1e30
        end
        mean((pred .- observed).^2)
    end

    function loss_fractional(theta)
        beta, sigma, gamma, scale = exp.(theta[1:4])
        alpha = 1 / (1 + exp(-theta[5]))
        p = [alpha, beta, sigma, gamma, scale]

        pred = try
            solve_fractional(p)
        catch
            return 1e30
        end

        mean((pred .- observed).^2)
    end
end

##
# ╔═╡ Cell 0014
begin
    theta0_classical = log.([0.35, 1/5, 1/10, 0.5])
    theta0_fractional = vcat(log.([0.35, 1/5, 1/10, 0.5]), [0.0])

    result_classical = optimize(
        loss_classical,
        theta0_classical,
        NelderMead(),
        Optim.Options(iterations = 2000)
    )

    result_fractional = optimize(
        loss_fractional,
        theta0_fractional,
        NelderMead(),
        Optim.Options(iterations = 3000)
    )

    result_classical, result_fractional
end

##
# ╔═╡ Cell 0015
begin
    theta_classical = Optim.minimizer(result_classical)
    theta_fractional = Optim.minimizer(result_fractional)

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

##
# ╔═╡ Cell 0016
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


# ╔═╡ Cell 0017
md"""
## Диагностика

1. Повторите оптимизацию из разных начальных точек.
2. Проверьте устойчивость к изменению шага численного метода.
3. Оцените интервалы неопределенности бутстреп-методом.
4. Проверьте прогноз на 7–14 дней вне обучающего окна.
5. Проанализируйте чувствительность к выбору границ волны.
"""