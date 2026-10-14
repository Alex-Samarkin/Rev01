# -*- coding: utf-8 -*-
"""
fractional_sir.jl — зеркальная (Julia) реализация численного эксперимента по
идентификации параметров волн COVID-19 дробной SIR-моделью с производной Капуто.

Реализация использует ТОЛЬКО стандартную библиотеку Julia
(Base, LinearAlgebra, Printf, DelimitedFiles, Statistics) — внешних пакетов нет,
поэтому модуль запускается в средах, где установка пакетов невозможна:

    julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --all
    julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --verify
    julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl --territory "Москва"

Модель (Caputo-SIR, alpha in (0, 1]):

    D^alpha S = -beta * S * I / A
    D^alpha I =  beta * S * I / A - gamma * I
    S(0) = A - i0,   I(0) = i0
    наблюдаемое:  C(t) = A - S(t)   (накопленное число случаев внутри окна волны)

Параметризация (эпидемиологически интерпретируемая):

    tau    — длительность инфекционного периода, ФИКСИРОВАНА = 7.0 суток;
    gamma  = tau^(-alpha);
    beta   = gamma + lambda^alpha,   lambda — начальная скорость роста (сут^-1);
    R0     = beta / gamma = 1 + (lambda * tau)^alpha   (при alpha = 1: R0 = 1 + lambda*tau).

Численная схема — явная схема L1 для производной Капуто:

    y_n = y_{n-1} + Gamma(2-alpha) * dt^alpha * f(y_{n-1})
          - sum_{k=1}^{n-1} b_k * (y_{n-k} - y_{n-k-1}),
    b_k = (k+1)^(1-alpha) - k^(1-alpha),  b_0 = 1,  dt = 0.1 суток.

Скрипт согласован по формулам с эталонной Python-реализацией
code/python/fractional_sir.py (отличие только в оптимизаторе: вместо
scipy.optimize.least_squares используется собственный многомерный метод
Нелдера—Мида с мультистартом).
"""
module FractionalSIR

using LinearAlgebra
using Printf
using DelimitedFiles
using Statistics

export l1_weights, simulate_sir, rates_from_lambda_tau, l1_scalar
export load_waves, wave_by_name, Wave, FitResult, fit_wave
export verify_scheme, aic_weights, nelder_mead, gamma_fn, erfc_fn, mittag_leffler

# ------------------------------------------------------------------- пути ---
const ROOT        = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_PATH   = joinpath(ROOT, "data", "processed", "waves.csv")
const RESULTS_DIR = joinpath(ROOT, "results")

# -------------------------------------------------------------- константы ---
const STEPS_PER_DAY = 10                       # dt = 0.1 суток
const LAMBDA_LO, LAMBDA_HI = 0.02, 0.90        # сут^-1
const TAU_FIXED = 7.0                          # длительность инфекционного периода, сутки
const ALPHA_LO, ALPHA_HI = 0.10, 1.00
const I0_LO, I0_HI = 0.5, 100000.0
const A_REPORT_MIN = 0.10                      # A <= 10 * C_end
const A_POP_MAX    = 12.0                      # A <= 12 * население
const BETA_MAX     = 6.0                       # техническое ограничение устойчивости
const INVALID_RSS  = 1.0e24                    # штраф за недопустимые параметры
const SEED_DEFAULT = 20260115

# ======================================================= спецфункции (Base) ===
# В Julia 1.12 нет Base.gamma / Base.erfc (они вынесены в SpecialFunctions),
# а внешние пакеты запрещены, поэтому реализуем их сами.

const _LANCZOS_G = 7.0
const _LANCZOS_C = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                    771.32342877765313, -176.61502916214059, 12.507343278686905,
                    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)

"""Γ(x) — аппроксимация Ланцоша (g = 7, n = 9), ~1e-15 относительной точности."""
function gamma_fn(x::Float64)
    if x < 0.5
        return pi / (sin(pi * x) * gamma_fn(1.0 - x))
    end
    z = x - 1.0
    a = _LANCZOS_C[1]
    @inbounds for i in 2:9
        a += _LANCZOS_C[i] / (z + (i - 1))
    end
    t = z + _LANCZOS_G + 0.5
    return sqrt(2.0 * pi) * t^(z + 0.5) * exp(-t) * a
end
gamma_fn(x::Real) = gamma_fn(Float64(x))

"""erf(x) степенным рядом: erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1)/(n!(2n+1))."""
function erf_series(x::Float64)
    x2 = x * x
    term = x
    s = term
    n = 1
    while n < 400
        term *= -x2 * (2n - 1) / (n * (2n + 1))
        s += term
        abs(term) <= 1e-18 * abs(s) && break
        n += 1
    end
    return 2.0 / sqrt(pi) * s
end

"""erfc(x) для x > 2 — цепная дробь Лапласа (модифицированный алгоритм Ленца)."""
function erfc_cf(x::Float64)
    tiny = 1.0e-300
    f = x
    C = f
    D = 0.0
    for k in 1:300
        a = 0.5 * k
        b = x
        D = b + a * D
        D == 0.0 && (D = tiny)
        C = b + a / C
        C == 0.0 && (C = tiny)
        D = 1.0 / D
        delta = C * D
        f *= delta
        abs(delta - 1.0) < 1e-16 && break
    end
    return exp(-x * x) / (sqrt(pi) * f)
end

"""erfc(x) — дополнительная функция ошибок (точность ~1e-15 для |x| <= ~5)."""
function erfc_fn(x::Float64)
    if x < 0.0
        return 2.0 - erfc_fn(-x)
    elseif x <= 2.0
        return 1.0 - erf_series(x)
    else
        return erfc_cf(x)
    end
end
erfc_fn(x::Real) = erfc_fn(Float64(x))

"""E_alpha(z) = sum_{k>=0} z^k / Gamma(alpha*k + 1) — ряд Миттаг-Леффлера."""
function mittag_leffler(alpha::Real, z::Real; terms::Int = 300)
    a = Float64(alpha); zz = Float64(z)
    s = 0.0
    for k in 0:terms
        s += zz^k / gamma_fn(a * k + 1.0)
    end
    return s
end

# ============================================================= схема L1 ======
"""
    l1_weights(alpha, n) -> Vector{Float64}

Весовые коэффициенты схемы L1: b_k = (k+1)^(1-alpha) - k^(1-alpha), k = 0..n.
b[1] = b_0 = 1.
"""
function l1_weights(alpha::Real, n::Int)
    a = Float64(alpha)
    b = Vector{Float64}(undef, n + 1)
    @inbounds for k in 0:n
        b[k + 1] = (k + 1.0)^(1.0 - a) - Float64(k)^(1.0 - a)
    end
    return b
end

"""
    simulate_sir(alpha, beta, gamma, A, n_days; i0 = 1.0, steps_per_day = 10)
        -> Vector{Float64}

Явная схема L1 для дробной SIR-модели. Возвращает C(t_j) = A - S(t_j)
для j = 0..n_days-1 (шаг вывода — 1 сутки). Полностью повторяет
Python-функцию simulate_sir из code/python/fractional_sir.py.
"""
function simulate_sir(alpha::Real, beta::Real, gamma::Real, A::Real, n_days::Int;
                      i0::Real = 1.0, steps_per_day::Int = STEPS_PER_DAY)
    a = Float64(alpha); bta = Float64(beta); gma = Float64(gamma)
    Aa = Float64(A); i0a = Float64(i0)

    n_steps = (n_days - 1) * steps_per_day
    dt = 1.0 / steps_per_day
    b = l1_weights(a, n_steps)
    G = gamma_fn(2.0 - a) * dt^a

    S  = Vector{Float64}(undef, n_steps + 1)
    I  = Vector{Float64}(undef, n_steps + 1)
    dS = zeros(Float64, n_steps + 1)
    dI = zeros(Float64, n_steps + 1)
    S[1] = Aa - i0a
    I[1] = i0a

    @inbounds for n in 1:n_steps
        p = n                       # индекс предыдущего слоя (1-based)
        hS = 0.0
        hI = 0.0
        for k in 1:(n - 1)
            bk = b[k + 1]
            hS += bk * dS[n - k + 1]
            hI += bk * dI[n - k + 1]
        end
        inf_ = bta * (S[p] / Aa) * I[p]
        S[n + 1] = S[p] + G * (-inf_) - hS
        I[n + 1] = I[p] + G * (inf_ - gma * I[p]) - hI
        dS[n + 1] = S[n + 1] - S[p]
        dI[n + 1] = I[n + 1] - I[p]
    end

    out = Vector{Float64}(undef, n_days)
    @inbounds for j in 0:(n_days - 1)
        out[j + 1] = Aa - S[j * steps_per_day + 1]
    end
    return out
end

"""
    rates_from_lambda_tau(lambda, tau, alpha) -> (beta, gamma, R0)

gamma = tau^(-alpha); beta = gamma + lambda^alpha; R0 = 1 + (lambda*tau)^alpha.
"""
function rates_from_lambda_tau(lam::Real, tau::Real, alpha::Real)
    a = Float64(alpha)
    gamma = Float64(tau)^(-a)
    beta = gamma + Float64(lam)^a
    R0 = 1.0 + (Float64(lam) * Float64(tau))^a
    return beta, gamma, R0
end

"""
    l1_scalar(alpha, lam, t_end, n_steps; y0 = 1.0, implicit = false)
        -> Vector{Float64}

Схема L1 для скалярной задачи D^alpha y = -lam*y, y(0) = y0 (используется
в verify_scheme). implicit = true — неявный вариант (для сравнения порядка).
"""
function l1_scalar(alpha::Real, lam::Real, t_end::Real, n_steps::Int;
                   y0::Real = 1.0, implicit::Bool = false)
    a = Float64(alpha); l = Float64(lam); T = Float64(t_end)
    dt = T / n_steps
    b = l1_weights(a, n_steps)
    G = gamma_fn(2.0 - a) * dt^a
    y  = Vector{Float64}(undef, n_steps + 1)
    dy = zeros(Float64, n_steps + 1)
    y[1] = Float64(y0)
    @inbounds for n in 1:n_steps
        h = 0.0
        for k in 1:(n - 1)
            h += b[k + 1] * dy[n - k + 1]
        end
        if implicit
            y[n + 1] = (y[n] - h) / (1.0 + G * l)
        else
            y[n + 1] = y[n] + G * (-l * y[n]) - h
        end
        dy[n + 1] = y[n + 1] - y[n]
    end
    return y
end

# ============================================================== данные =======
"""Одна волна (территория) из waves.csv."""
struct Wave
    territory::String
    country::String
    unit_type::String
    population::Float64
    dates::Vector{String}
    cumulative::Vector{Float64}      # cumulative_rel — основная наблюдаемая величина
    daily::Vector{Float64}
    daily_ma7::Vector{Float64}
end

npoints(w::Wave) = length(w.cumulative)

"""
    load_waves(; path = DATA_PATH) -> Vector{Wave}

Читает long-таблицу waves.csv через DelimitedFiles.readdlm (заголовок
пропускается) и выделяет волны территорий в порядке первого появления
(порядок такой же, как pandas.groupby(sort=False) в Python-версии).
"""
function load_waves(; path::AbstractString = DATA_PATH)
    raw = readdlm(path, ',')
    header = [String(x) for x in raw[1, :]]
    col = Dict{String,Int}()
    for (j, h) in enumerate(header)
        col[h] = j
    end
    for need in ("territory", "country", "unit_type", "population", "date",
                 "cumulative", "cumulative_rel", "daily", "daily_ma7")
        haskey(col, need) || error("в waves.csv нет колонки $need")
    end

    order = String[]
    acc = Dict{String,Vector{Any}}()
    for i in 2:size(raw, 1)
        terr = strip(String(raw[i, col["territory"]]))
        if !haskey(acc, terr)
            acc[terr] = Any[]
            push!(order, terr)
        end
        push!(acc[terr], raw[i, :])
    end

    waves = Wave[]
    for terr in order
        rows = acc[terr]
        # сортировка по дате (ISO-формат => лексикографически = хронологически)
        rows = rows[sortperm([strip(String(r[col["date"]])) for r in rows])]
        push!(waves, Wave(
            terr,
            strip(String(rows[1][col["country"]])),
            strip(String(rows[1][col["unit_type"]])),
            Float64(rows[1][col["population"]]),
            [strip(String(r[col["date"]])) for r in rows],
            [Float64(r[col["cumulative_rel"]]) for r in rows],
            [Float64(r[col["daily"]]) for r in rows],
            [Float64(r[col["daily_ma7"]]) for r in rows],
        ))
    end
    return waves
end

wave_by_name(waves::Vector{Wave}, name::AbstractString) = begin
    idx = findfirst(w -> w.territory == String(name), waves)
    idx === nothing && error("территория не найдена: $name")
    waves[idx]
end

# ================================================== преобразование границ ====
sigmoid(z::Float64) = z >= 0 ? 1.0 / (1.0 + exp(-z)) : (e = exp(z); e / (1.0 + e))
logit(u::Float64) = log(u / (1.0 - u))

function from_unit(u::Float64, lo::Float64, hi::Float64, log_scale::Bool)
    return log_scale ? lo * (hi / lo)^u : lo + (hi - lo) * u
end

function to_unit(v::Float64, lo::Float64, hi::Float64, log_scale::Bool)
    if log_scale
        v = clamp(v, lo * 1.0001, hi * 0.9999)
        return log(v / lo) / log(hi / lo)
    else
        v = clamp(v, lo + 1e-9, hi - 1e-9)
        return (v - lo) / (hi - lo)
    end
end

"""Границы эффективной численности A: [C_end, min(10*C_end, 12*N)]."""
function a_limits(w::Wave)
    c_end = w.cumulative[end]
    return c_end, min(c_end / A_REPORT_MIN, A_POP_MAX * w.population)
end

"""Описание свободных параметров модели: (имя, lo, hi, log_scale)."""
function param_spec(model::Symbol, w::Wave)
    a_lo, a_hi = a_limits(w)
    if model === :classic
        return [("lam", LAMBDA_LO, LAMBDA_HI, true),
                ("i0", I0_LO, I0_HI, true),
                ("A", a_lo, a_hi, false)]
    elseif model === :fractional
        return [("lam", LAMBDA_LO, LAMBDA_HI, true),
                ("alpha", ALPHA_LO, ALPHA_HI, false),
                ("i0", I0_LO, I0_HI, true),
                ("A", a_lo, a_hi, false)]
    else
        error("неизвестная модель: $model (ожидается :classic или :fractional)")
    end
end

param_values_to_x(vals::Vector{Float64}, spec) =
    [logit(to_unit(vals[i], spec[i][2], spec[i][3], spec[i][4])) for i in eachindex(spec)]

function x_to_params(x::Vector{Float64}, spec, tau::Float64)
    p = Dict{String,Float64}()
    for i in eachindex(spec)
        nm, lo, hi, lg = spec[i]
        p[nm] = from_unit(sigmoid(x[i]), lo, hi, lg)
    end
    alpha = get(p, "alpha", 1.0)
    beta, gamma, R0 = rates_from_lambda_tau(p["lam"], tau, alpha)
    p["alpha"] = alpha
    p["beta"] = beta
    p["gamma"] = gamma
    p["R0"] = R0
    p["tau"] = tau
    return p
end

# ============================================================ Нелдер—Мид ====
"""
    nelder_mead(f, x0; step = 0.5, maxiter = 1500, ftol = 1e-12, xtol = 1e-9)
        -> (xbest, fbest, nfev, converged)

Классический многомерный метод Нелдера—Мида (отражение, растяжение, сжатие,
редукция) с собственным симплексом. Реализация не использует внешних пакетов.
"""
function nelder_mead(f::Function, x0::Vector{Float64};
                     step::Float64 = 0.5, maxiter::Int = 1500,
                     ftol::Float64 = 1e-12, xtol::Float64 = 1e-9)
    n = length(x0)
    nfev = 0
    simplex = Vector{Vector{Float64}}(undef, n + 1)
    fvals = Vector{Float64}(undef, n + 1)

    function feval!(v::Vector{Float64})
        nfev += 1
        return Float64(f(v))
    end

    simplex[1] = copy(x0)
    fvals[1] = feval!(simplex[1])
    for i in 1:n
        xi = copy(x0)
        xi[i] += step
        simplex[i + 1] = xi
        fvals[i + 1] = feval!(xi)
    end

    alpha_r, gamma_e, rho_c, sigma_s = 1.0, 2.0, 0.5, 0.5
    converged = false
    for _ in 1:maxiter
        ord = sortperm(fvals)
        simplex = simplex[ord]
        fvals = fvals[ord]

        fspread = fvals[end] - fvals[1]
        xspread = 0.0
        for i in 1:n
            for j in 1:n
                xspread = max(xspread, abs(simplex[i + 1][j] - simplex[1][j]))
            end
        end
        if fspread <= ftol * (abs(fvals[1]) + ftol) && xspread <= xtol
            converged = true
            break
        end

        centroid = zeros(Float64, n)
        for i in 1:n, j in 1:n
            centroid[j] += simplex[i][j] / n
        end
        xworst = simplex[end]
        fworst = fvals[end]

        xr = [centroid[j] + alpha_r * (centroid[j] - xworst[j]) for j in 1:n]
        fr = feval!(xr)

        if fr < fvals[1]
            xe = [centroid[j] + gamma_e * (centroid[j] - xworst[j]) for j in 1:n]
            fe = feval!(xe)
            if fe < fr
                simplex[end] = xe; fvals[end] = fe
            else
                simplex[end] = xr; fvals[end] = fr
            end
        elseif fr < fvals[n]
            simplex[end] = xr; fvals[end] = fr
        else
            if fr < fworst
                xc = [centroid[j] + rho_c * (xr[j] - centroid[j]) for j in 1:n]
                fc = feval!(xc)
                if fc <= fr
                    simplex[end] = xc; fvals[end] = fc
                else
                    for i in 2:(n + 1)
                        for j in 1:n
                            simplex[i][j] = simplex[1][j] + sigma_s * (simplex[i][j] - simplex[1][j])
                        end
                        fvals[i] = feval!(simplex[i])
                    end
                end
            else
                xc = [centroid[j] + rho_c * (xworst[j] - centroid[j]) for j in 1:n]
                fc = feval!(xc)
                if fc < fworst
                    simplex[end] = xc; fvals[end] = fc
                else
                    for i in 2:(n + 1)
                        for j in 1:n
                            simplex[i][j] = simplex[1][j] + sigma_s * (simplex[i][j] - simplex[1][j])
                        end
                        fvals[i] = feval!(simplex[i])
                    end
                end
            end
        end
    end

    k = argmin(fvals)
    return simplex[k], fvals[k], nfev, converged
end

# ========================================================= идентификация ====
"""Результат идентификации одной волны одной моделью."""
struct FitResult
    territory::String
    model::Symbol
    params::Dict{String,Float64}
    rss::Float64
    rmse::Float64
    r2::Float64
    aic::Float64
    bic::Float64
    n::Int
    k::Int
    nfev::Int
    seconds::Float64
    infected_fraction::Float64   # C_end(наблюдаемое) / A
    curve::Vector{Float64}       # C(t_j), j = 0..n-1
    daily::Vector{Float64}       # diff(C)
end

"""
    fit_wave(territory, model; tau = 7.0, n_starts = 20, maxiter = 1500,
             steps_per_day = 10, verbose = false, seed = 20260115) -> FitResult

МНК-идентификация параметров волны (модель :classic или :fractional) методом
Нелдера—Мида с мультистартом. Критерий — сумма квадратов невязок по
наблюдённой кривой cumulative_rel. Границы параметров удерживаются логистическим
преобразованием (лог-шкала для lambda и i0).
"""
function fit_wave(territory, model::Symbol;
                  tau::Float64 = TAU_FIXED,
                  n_starts::Int = 20,
                  maxiter::Int = 1500,
                  steps_per_day::Int = STEPS_PER_DAY,
                  verbose::Bool = false,
                  seed::Int = SEED_DEFAULT,
                  refine::Int = 2)
    w = territory isa Wave ? territory : wave_by_name(load_waves(), territory)
    spec = param_spec(model, w)
    target = w.cumulative
    n = length(target)

    function rss_of(x::Vector{Float64})
        all(isfinite, x) || return INVALID_RSS
        p = x_to_params(x, spec, tau)
        if !isfinite(p["beta"]) || p["beta"] > BETA_MAX || !(p["gamma"] > 0) || !(p["A"] > 0)
            return INVALID_RSS
        end
        C = try
            simulate_sir(p["alpha"], p["beta"], p["gamma"], p["A"], n;
                         i0 = p["i0"], steps_per_day = steps_per_day)
        catch
            return INVALID_RSS
        end
        all(isfinite, C) || return INVALID_RSS
        r = 0.0
        @inbounds for j in 1:n
            d = C[j] - target[j]
            r += d * d
        end
        return isfinite(r) ? r : INVALID_RSS
    end

    a_lo, a_hi = a_limits(w)
    grid_lam  = [0.04, 0.07, 0.12, 0.20, 0.35]
    grid_i0   = [1.0, 30.0, 300.0, 3000.0]
    grid_afr  = [0.03, 0.15, 0.50]
    grid_alp  = [0.35, 0.60, 0.85]
    starts = Vector{Float64}[]
    for (i, lv) in enumerate(grid_lam), (j, i0v) in enumerate(grid_i0)
        afr = grid_afr[(i + j) % 3 + 1]
        Av = a_lo + (a_hi - a_lo) * afr
        if model === :classic
            push!(starts, param_values_to_x([lv, i0v, Av], spec))
        else
            push!(starts, param_values_to_x([lv, grid_alp[(i + j) % 3 + 1], i0v, Av], spec))
        end
    end
    # детерминированный «случайный» добор (без Random: внешних зависимостей нет)
    rng = seed
    nextrand() = (rng = (1103515245 * rng + 12345) % 2147483648; rng / 2147483648.0)
    while length(starts) < n_starts
        lv = 0.03 + 0.47 * nextrand()
        i0v = exp(log(1.0) + (log(3000.0) - log(1.0)) * nextrand())
        Av = a_lo + (a_hi - a_lo) * (0.02 + 0.78 * nextrand())
        if model === :classic
            push!(starts, param_values_to_x([lv, i0v, Av], spec))
        else
            push!(starts, param_values_to_x([lv, 0.15 + 0.80 * nextrand(), i0v, Av], spec))
        end
    end

    t0 = time()
    nfev_total = 0
    best_x = starts[1]
    best_f = INVALID_RSS
    for (s, x0) in enumerate(starts)
        xb, fb, nf, _ = nelder_mead(rss_of, x0; step = 0.6, maxiter = maxiter)
        nfev_total += nf
        if fb < best_f
            best_f = fb
            best_x = xb
        end
        verbose && @printf("    старт %2d/%d: RSS = %.6g\n", s, length(starts), fb)
    end

    # уточняющие перезапуски от лучшей точки (маленький симплекс)
    for r in 1:refine
        xb, fb, nf, _ = nelder_mead(rss_of, best_x; step = 0.6 / (3.0^r),
                                    maxiter = maxiter)
        nfev_total += nf
        if fb < best_f
            best_f = fb
            best_x = xb
        end
    end
    elapsed = time() - t0

    p = x_to_params(best_x, spec, tau)
    C = simulate_sir(p["alpha"], p["beta"], p["gamma"], p["A"], n;
                     i0 = p["i0"], steps_per_day = steps_per_day)
    res = C .- target
    rss = sum(abs2, res)
    rmse = sqrt(rss / n)
    tbar = mean(target)
    r2 = 1.0 - rss / sum(abs2, target .- tbar)
    kpar = length(spec)
    aic = n * log(rss / n) + 2 * kpar
    bic = n * log(rss / n) + kpar * log(n)

    return FitResult(w.territory, model, p, rss, rmse, r2, aic, bic, n, kpar,
                     nfev_total, elapsed, target[end] / p["A"], C, diff(C))
end

"""Вес Акаике: относительная поддержка моделей."""
function aic_weights(aics)
    a = Float64.(collect(aics))
    m = minimum(a)
    wts = exp.(-0.5 .* (a .- m))
    return wts ./ sum(wts)
end

# ============================================================ верификация ====
"""
    verify_scheme(; verbose = true) -> NamedTuple

Проверка схемы L1 на задаче D^alpha y = -lambda*y, y(0) = 1 против
аналитического решения. Контрольная точка: alpha = 1/2, lambda = 1, t = 1,
      y(1) = E_{1/2}(-1) = exp(1) * erfc(1) = 0.4275850...
Дополнительно печатается наблюдаемый порядок сходимости по dt.
"""
function verify_scheme(; verbose::Bool = true, io::IO = stdout)
    exact12 = exp(1.0) * erfc_fn(1.0)
    lines = String[]

    function emit(s)
        push!(lines, s)
        verbose && println(io, s)
    end

    emit("="^78)
    emit("ВЕРИФИКАЦИЯ СХЕМЫ L1 (D^alpha y = -lambda*y, y(0)=1)")
    emit("="^78)
    emit(@sprintf("Аналитическая точка: E_{1/2}(-1) = exp(1)*erfc(1) = %.10f", exact12))
    emit(@sprintf("(проверка собственной erfc: erfc(1) = %.16f, эталон 0.15729920705028513)",
                  erfc_fn(1.0)))
    emit("")

    emit("Проверка 1: alpha = 1/2, lambda = 1, t = 1 (явная схема L1)")
    ys = Float64[]
    for n in (100, 1000, 10000)
        y = l1_scalar(0.5, 1.0, 1.0, n)
        push!(ys, y[end])
        emit(@sprintf("  n=%6d  y(1)=%.10f  точное=%.10f  |err|=%.3e",
                      n, y[end], exact12, abs(y[end] - exact12)))
    end

    emit("")
    emit("Проверка 2: alpha = 1, lambda = 1, t = 1 (должно совпасть с exp(-1))")
    exact_l = exp(-1.0)
    for n in (100, 1000)
        y = l1_scalar(1.0, 1.0, 1.0, n)
        emit(@sprintf("  n=%6d  y(1)=%.10f  точное=%.10f  |err|=%.3e",
                      n, y[end], exact_l, abs(y[end] - exact_l)))
    end

    emit("")
    emit("Проверка 3: наблюдаемый порядок сходимости по dt (alpha = 0.5, t = 1)")
    ns = (100, 200, 400, 800, 1600)
    errs_ex = [abs(l1_scalar(0.5, 1.0, 1.0, n)[end] - exact12) for n in ns]
    errs_im = [abs(l1_scalar(0.5, 1.0, 1.0, n; implicit = true)[end] - exact12) for n in ns]
    ord_ex = [log(errs_ex[i] / errs_ex[i + 1]) / log(2.0) for i in 1:(length(ns) - 1)]
    ord_im = [log(errs_im[i] / errs_im[i + 1]) / log(2.0) for i in 1:(length(ns) - 1)]
    emit("  явная   : ошибки " * join([@sprintf("%.3e", e) for e in errs_ex], " "))
    emit("            порядки " * join([@sprintf("%.3f", o) for o in ord_ex], " ") *
         "   (теория: 1.0)")
    emit("  неявная : ошибки " * join([@sprintf("%.3e", e) for e in errs_im], " "))
    emit("            порядки " * join([@sprintf("%.3f", o) for o in ord_im], " ") *
         "   (теория: 2-alpha = 1.5)")

    emit("")
    emit("Проверка 4: сравнение с рядом Миттаг-Леффлера, t = 5, alpha = 0.5, lambda = 1")
    ml = mittag_leffler(0.5, -sqrt(5.0))
    y_ex = l1_scalar(0.5, 1.0, 5.0, 20000)[end]
    y_im = l1_scalar(0.5, 1.0, 5.0, 20000; implicit = true)[end]
    emit(@sprintf("  ряд   E_{0.5}(-sqrt(5)) = %.10f", ml))
    emit(@sprintf("  явная   L1 = %.10f   |err| = %.3e", y_ex, abs(y_ex - ml)))
    emit(@sprintf("  неявная L1 = %.10f   |err| = %.3e", y_im, abs(y_im - ml)))
    emit("="^78)

    return (exact = exact12, y_n100 = ys[1], y_n1000 = ys[2], y_n10000 = ys[3],
            errs_explicit = errs_ex, errs_implicit = errs_im,
            order_explicit = ord_ex, order_implicit = ord_im,
            mittag_leffler_t5 = ml, lines = lines)
end

# ================================================================== CLI =====
function _print_table(results::Vector{FitResult}, io::IO = stdout)
    hdr = @sprintf("%-24s %-11s %7s %8s %6s %8s %12s %11s %9s %9s %8s",
                   "territory", "model", "alpha", "lambda", "R0", "tau",
                   "A", "RMSE", "R2", "AIC", "сек")
    println(io, hdr)
    println(io, "-"^length(hdr))
    for r in results
        p = r.params
        println(io, @sprintf("%-24s %-11s %7.4f %8.4f %6.3f %8.2f %12.1f %11.2f %9.5f %9.2f %8.1f",
                             r.territory, string(r.model), p["alpha"], p["lam"], p["R0"],
                             p["tau"], p["A"], r.rmse, r.r2, r.aic, r.seconds))
    end
end

function _write_results_csv(results::Vector{FitResult}, path::AbstractString)
    mkpath(dirname(path))
    cols = ["territory", "model", "alpha", "lambda", "tau", "R0", "beta", "gamma", "A", "i0",
            "infected_fraction", "rss", "rmse", "r2", "aic", "bic", "n", "k",
            "nfev", "seconds"]
    M = Matrix{Any}(undef, length(results) + 1, length(cols))
    for (j, c) in enumerate(cols)
        M[1, j] = c
    end
    for (i, r) in enumerate(results)
        p = r.params
        vals = Any[r.territory, string(r.model), p["alpha"], p["lam"], p["tau"], p["R0"],
                   p["beta"], p["gamma"], p["A"], p["i0"],
                   r.infected_fraction, r.rss, r.rmse, r.r2, r.aic, r.bic, r.n, r.k,
                   r.nfev, r.seconds]
        for (j, v) in enumerate(vals)
            M[i + 1, j] = v
        end
    end
    open(path, "w") do fio
        writedlm(fio, M, ',')
    end
    return path
end

function _print_summary(results::Vector{FitResult}, io::IO = stdout)
    println(io, "")
    println(io, "=== Сравнение классической и дробной моделей ===")
    terr = unique([r.territory for r in results])
    for t in terr
        cl = findfirst(r -> r.territory == t && r.model === :classic, results)
        fr = findfirst(r -> r.territory == t && r.model === :fractional, results)
        (cl === nothing || fr === nothing) && continue
        a, b = results[cl], results[fr]
        wts = aic_weights([a.aic, b.aic])
        println(io, @sprintf("%-24s RMSE %9.1f -> %9.1f (%5.1f%%)   R2 %7.4f -> %7.4f   dAIC %8.1f   вес %.4f/%.4f",
                             t, a.rmse, b.rmse, 100 * (1 - b.rmse / a.rmse),
                             a.r2, b.r2, a.aic - b.aic, wts[1], wts[2]))
    end
end

function main(argv::Vector{String} = ARGS)
    do_verify = "--verify" in argv
    do_all = "--all" in argv
    targs = String[]
    n_starts = 20
    maxiter = 1500
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--territory" && i < length(argv)
            push!(targs, argv[i + 1]); i += 1
        elseif a == "--starts" && i < length(argv)
            n_starts = parse(Int, argv[i + 1]); i += 1
        elseif a == "--maxiter" && i < length(argv)
            maxiter = parse(Int, argv[i + 1]); i += 1
        end
        i += 1
    end

    if !(do_verify || do_all || !isempty(targs))
        println("Использование: julia --startup-file=no --compiled-modules=no code/julia/fractional_sir.jl [--all] [--verify] [--territory \"Имя\"] [--starts N] [--maxiter N]")
        return
    end

    if do_verify || do_all
        res = verify_scheme()
        mkpath(RESULTS_DIR)
        open(joinpath(RESULTS_DIR, "verify_scheme_julia.txt"), "w") do fio
            for l in res.lines
                println(fio, l)
            end
        end
        do_verify && !do_all && isempty(targs) && return
    end

    waves = load_waves()
    selected = do_all ? waves : [wave_by_name(waves, t) for t in targs]
    isempty(selected) && (selected = waves)

    println("\nИдентификация параметров: " * string(length(selected)) * " территорий(и)")
    results = FitResult[]
    for w in selected
        println(@sprintf("\n=== %s (n=%d, население %.0f, C_end = %.0f) ===",
                         w.territory, npoints(w), w.population, w.cumulative[end]))
        flush(stdout)
        for model in (:classic, :fractional)
            r = fit_wave(w, model; n_starts = n_starts, maxiter = maxiter)
            push!(results, r)
            p = r.params
            println(@sprintf("  %-11s alpha=%.4f  lambda=%.4f сут^-1  tau=%.2f сут  R0=%.3f  A=%.0f  i0=%.1f  RMSE=%9.2f  R2=%.5f  AIC=%8.2f  (%.1f с, %d вычисл. схемы)",
                             string(model), p["alpha"], p["lam"], p["tau"], p["R0"], p["A"],
                             p["i0"], r.rmse, r.r2, r.aic, r.seconds, r.nfev))
            flush(stdout)
        end
    end

    println("")
    _print_table(results)
    _print_summary(results)

    out = do_all ? joinpath(RESULTS_DIR, "identification_julia.csv") :
                   joinpath(RESULTS_DIR, "identification_julia_partial.csv")
    _write_results_csv(results, out)
    println("\nРезультаты сохранены: " * out)

    # текстовая копия сводки (UTF-8) для дальнейшего анализа
    open(joinpath(RESULTS_DIR, "identification_julia_table.txt"), "w") do fio
        _print_table(results, fio)
        _print_summary(results, fio)
    end
    return results
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    FractionalSIR.main()
end
