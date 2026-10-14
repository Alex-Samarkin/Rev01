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
        "FractionalDiffEq", "OrdinaryDiffEq", "Statistics", "Random"
    ])
end

#

# ╔═╡ 9864ef40-c7d1-11f1-afd9-6f2a9cb58f53
begin
    using CSV, DataFrames, Dates, Plots, Optim, Statistics, Random
    using FractionalDiffEq, OrdinaryDiffEq
end

#

# ╔═╡ 9864ef40-c7d1-11f1-b173-2bd778287a61
md"""
# Идентификация параметров волны заболеваемости: Caputo-дробная SEIR-модель

**Данные:** еженедельные данные по Москве, опубликованные на HDX.  
Первичный источник: правительство Москвы.  
Страница набора данных: <https://data.humdata.org/dataset/covid-19-cases-data-in-moscow>

Скачайте файл `covid_data_moscow_region_weekly_2021CSV` и укажите путь к нему ниже.
"""
#

# ╔═╡ 9864ef40-c7d1-11f1-b2aa-2973f7b8242c
begin
    # data_path = "covid_data_moscow_region_weekly_2021.csv"
    data_path = "csv_moskov.csv"

    if !isfile(data_path)
        @warn "Файл данных не найден. Скачайте CSV со страницы HDX и укажите корректный путь."
    end
end
#

# ╔═╡ 9864ef40-c7d1-11f1-9eb6-d3742a8962ac
begin
    raw = CSV.read(data_path, DataFrame)
    println(names(raw))
    first(raw, 5)
end

#

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

Еженедельные случаи пересчитываются в среднесуточную интенсивность:
```math

y_i=\frac{\Delta C_i}{7}.

```
"""

# ╔═╡ 9864ef40-c7d1-11f1-bcd9-ed23986dbbeb
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

"""

#

# ╔═╡ 9864ef40-c7d1-11f1-864a-672cbf0f8644
begin
    N = 12_678_000.0
    E0 = 500.0
    I0 = 500.0
    R0 = 0.0
    S0 = N - E0 - I0 - R0

    u0 = [S0, E0, I0, R0]
    tspan = (0.0, Float64(length(wave.smoothed) - 1))
end

#

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

    rhs = make_fractional_rhs(beta, sigma, gamma, scale)

    prob = FODEProblem(rhs, alpha, u0, tspan)

    sol = solve(prob, PECE(); dt = 0.25)

    # Плотная сетка решения
    solution_times = collect(tspan[1]:0.25:tspan[2])

    obs_indices = [
        argmin(abs.(solution_times .- t))
        for t in observation_times
    ]

    return scale .* [sol.u[i][2] for i in obs_indices]
    end
end

#

# ╔═╡ 9864ef40-c7d1-11f1-aa3e-bb61602cee0f
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

#

# ╔═╡ 9864ef40-c7d1-11f1-930c-2d821e3c113f
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

#

# ╔═╡ 9864ef40-c7d1-11f1-a56f-659763d49c26
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

#

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


# ╔═╡ 9864ef40-c7d1-11f1-937e-5569d2afbbe8
md"""
## Диагностика

1. Повторите оптимизацию из разных начальных точек.
2. Проверьте устойчивость к изменению шага численного метода.
3. Оцените интервалы неопределенности бутстреп-методом.
4. Проверьте прогноз на 7–14 дней вне обучающего окна.
5. Проанализируйте чувствительность к выбору границ волны.
"

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
# ╠═9864ef40-c7d1-11f1-937e-5569d2afbbe8
