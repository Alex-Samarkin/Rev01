# Классическая и Caputo-дробная SEIR: идентификация параметров для Москвы.
# Ячейки # %% выполняйте по порядку в Julia VS Code.
# Полный запуск: julia 02_moscow_fractional_seir_pluto.jl

# %% Окружение SciTools
import Pkg
# Активируем существующее окружение рядом со скриптом, независимо от cwd.
Pkg.activate(joinpath(@__DIR__, "@SciTools"))
# При первой настройке выполните Pkg.instantiate() отдельно.
# Исходный блокнот использовал FractionalDiffEq 0.5.1; версии зависимостей
# теперь определяются Project.toml и Manifest.toml окружения.

# %% Зависимости
using CSV, DataFrames, Dates, Plots, Optim, Statistics
using FractionalDiffEq, OrdinaryDiffEq
import OrdinaryDiffEq: solve   # снимает неоднозначность имени `solve`

# %% Задача и источник данных
# # Идентификация параметров волны заболеваемости: Caputo-дробная SEIR-модель
# 
# **Данные:** еженедельные данные по Москве, опубликованные на HDX.  
# Первичный источник: правительство Москвы.  
# Страница набора данных: <https://data.humdata.org/dataset/covid-19-cases-data-in-moscow>
# 
# Скачайте файл `covid_data_moscow_region_weekly_2021CSV` и укажите путь к нему ниже.

# %% Путь к данным
# data_path = "covid_data_moscow_region_weekly_2021.csv"
data_path = joinpath(@__DIR__, "csv_moskov.csv")

if !isfile(data_path)
    error("Файл данных не найден: $data_path. Скачайте CSV со страницы HDX и укажите корректный путь.")
end

# %% Чтение CSV
raw = CSV.read(data_path, DataFrame)
println(names(raw))
display(first(raw, 5))

# %% Подготовка столбцов и дат
# Приводим имена столбцов к date и cases, исключаем пропуски и сортируем даты.
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

display(first(df, 10))

# %% Подготовка ряда
# ## Подготовка ряда
# 
# Столбец `sick` — **накопленное** число случаев, поэтому интенсивность считается
# по приращениям с учётом фактического шага между датами (1 день для суточных данных, 7 — для недельных):
# ```math
# 
# y_i=\frac{C_i - C_{i-1}}{t_i - t_{i-1}}.
# 
# ```

# %% Выбор окна и сглаживание
# Приращения накопленных случаев делим на фактический интервал в днях.
# Интенсивность относим к правой границе интервала. Сглаживание использует
# три соседних наблюдения; на краях окно укорочено.
# В окно включаем только доступные наблюдения из CSV.
wave_start = Date(2021, 1, 1)
wave_end = Date(2021, 1, 31)

in_window = filter(row -> wave_start <= row.date <= wave_end, df)
nrow(in_window) >= 3 || error("В окне [$wave_start, $wave_end] меньше 3 точек данных")

step_days = Dates.value.(diff(in_window.date))
wave = in_window[2:end, :]
wave.daily_rate = diff(Float64.(in_window.cases)) ./ step_days
wave.smoothed = [
    mean(wave.daily_rate[max(1, i - 1):min(end, i + 1)])
    for i in eachindex(wave.daily_rate)
]

data_plot = plot(
    wave.date, wave.daily_rate,
    label = "Новые случаи в сутки",
    xlabel = "Дата",
    ylabel = "Случаев в сутки",
    legend = :topleft
)
plot!(wave.date, wave.smoothed, lw = 3, label = "Сглаженный ряд")
display(data_plot)

# %% Уравнения модели
# ## Модель
# 
# ```math
# 
# \begin{aligned}
# {}^C D_t^\alpha S &= -\beta S I/N,\\
# {}^C D_t^\alpha E &= \beta S I/N-\sigma E,\\
# {}^C D_t^\alpha I &= \sigma E-\gamma I,\\
# {}^C D_t^\alpha R &= \gamma I.
# \end{aligned}
# 
# ```
# Наблюдаемая переменная:
# 
# ```math
# \hat y(t)=q\sigma E(t).
# ```
# 
# Начальное состояние привязано к первому наблюдению:
# ``E_0 = y_1/(q\sigma)``, ``I_0 = E_0\sigma/\gamma``.

# %% Время и начальное состояние
# Время измеряем в сутках, сохраняя реальные интервалы между наблюдениями.
# q = scale — доля регистрации; E₀ задаём из y₁ = q σ E₀.
# I₀ выбираем из равенства σ E₀ = γ I₀, а R₀ принимаем равным нулю.
N = 12_678_000.0

# Время в сутках от первой даты окна (а не номер наблюдения)
observation_times = Float64.(Dates.value.(wave.date .- wave.date[1]))
tspan = (0.0, observation_times[end])

function initial_state(sigma, gamma, scale)
    E0 = wave.smoothed[1] / (scale * sigma)
    I0 = E0 * sigma / gamma
    [N - E0 - I0, E0, I0, 0.0]
end

# %% Правые части SEIR
# S — восприимчивые, E — латентно инфицированные, I — заразные, R — выбывшие.
# β — передача инфекции, σ — переход E → I, γ — переход I → R.
# Сумма правых частей равна нулю: численность населения сохраняется.
# scale влияет на наблюдения и начальное состояние, но не на правую часть.
# Дробный решатель использует in-place интерфейс: результат записывается в du.
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


# %% Численное решение
# Tsit5 решает классическую систему с адаптивным шагом.
# PECE решает систему с производной Капуто, учитывающей память о предыстории.
# dt — шаг равномерной сетки в сутках. Наблюдения сопоставляем ближайшим
# узлам этой сетки без интерполяции; проверьте чувствительность к выбору dt.
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

# %% Целевые функции
# Минимизируем среднюю квадратичную ошибку по сглаженной интенсивности.
# exp обеспечивает положительность β, σ, γ, q; логистическое преобразование
# задаёт 0 < α < 1. Ошибки решателя и недопустимые параметры получают штраф.
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

# %% Оптимизация
# Nelder–Mead не требует градиентов. Начальная точка влияет на результат;
# лимит итераций не гарантирует сходимость или достижение глобального минимума.
theta0_classical = log.([0.35, 1/5, 1/10, 0.5])
theta0_fractional = vcat(log.([0.35, 1/5, 1/10, 0.5]), [2.0])  # alpha0 ≈ 0.88

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

display(result_classical)
display(result_fractional)

# %% Параметры и ошибки
# Восстанавливаем параметры из переменных оптимизатора.
# RMSE сильнее учитывает большие ошибки, MAE — средняя абсолютная ошибка.
# Это ошибки подгонки на обучающем окне, а не оценка качества прогноза.
theta_classical = Optim.minimizer(result_classical)
theta_fractional = Optim.minimizer(result_fractional)

beta_c, sigma_c, gamma_c, scale_c = exp.(theta_classical)
beta_f, sigma_f, gamma_f, scale_f = exp.(theta_fractional[1:4])
alpha_f = 1 / (1 + exp(-theta_fractional[5]))

pred_classical = solve_classical([beta_c, sigma_c, gamma_c, scale_c])
pred_fractional = solve_fractional([alpha_f, beta_f, sigma_f, gamma_f, scale_f])

fit_summary = DataFrame(
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
display(fit_summary)

# %% Сравнение моделей
comparison_plot = plot(
    wave.date, observed,
    label = "Данные",
    xlabel = "Дата",
    ylabel = "Новых случаев в сутки",
    legend = :topleft,
    lw = 2
)
plot!(wave.date, pred_classical, label = "Классическая SEIR", lw = 2, ls = :dash)
plot!(wave.date, pred_fractional, label = "Caputo-дробная SEIR", lw = 2, ls = :dot)
display(comparison_plot)

# %% Диагностика
# ## Диагностика
# 
# 1. Повторите оптимизацию из разных начальных точек.
# 2. Проверьте устойчивость к изменению шага численного метода.
# 3. Оцените интервалы неопределенности бутстреп-методом.
# 4. Проверьте прогноз на 7–14 дней вне обучающего окна.
# 5. Проанализируйте чувствительность к выбору границ волны.
