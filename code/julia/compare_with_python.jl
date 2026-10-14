# -*- coding: utf-8 -*-
# compare_with_python.jl — построчное сравнение результатов Julia и Python.
#
# Читает results/identification_julia.csv (эта реализация) и
# results/identification_cumulative_tau7.csv (прогон эталонной Python-версии
# code/python/fractional_sir.py при observable = "cumulative", tau = 7.0)
# и пишет results/comparison_julia_python.csv.
#
# Запуск:
#   julia --startup-file=no --compiled-modules=no code/julia/compare_with_python.jl

using DelimitedFiles
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const JULIA_CSV = joinpath(ROOT, "results", "identification_julia.csv")
const PY_CSV = joinpath(ROOT, "results", "identification_cumulative_tau7.csv")
const OUT_CSV = joinpath(ROOT, "results", "comparison_julia_python.csv")

function read_table(path)
    raw = readdlm(path, ',')
    header = [String(x) for x in raw[1, :]]
    idx = Dict(h => j for (j, h) in enumerate(header))
    rows = Dict{String,Dict{String,Float64}}()
    for i in 2:size(raw, 1)
        terr = strip(String(raw[i, idx["territory"]]))
        model = strip(String(raw[i, idx["model"]]))
        d = Dict{String,Float64}()
        for h in ("alpha", "lambda", "R0", "A", "rmse", "r2", "aic", "bic")
            if haskey(idx, h)
                v = raw[i, idx[h]]
                d[h] = v isa Number ? Float64(v) : NaN
            end
        end
        rows["$terr|$model"] = d
    end
    return rows
end

const jl = read_table(JULIA_CSV)
const py = read_table(PY_CSV)

cols = ["territory", "model",
        "julia_rmse", "python_rmse", "rel_diff_rmse_pct",
        "julia_r2", "python_r2",
        "julia_aic", "python_aic",
        "julia_alpha", "python_alpha",
        "julia_lambda", "python_lambda",
        "julia_R0", "python_R0",
        "julia_A", "python_A",
        "max_rel_diff_pct"]

keys_sorted = sort(collect(keys(jl)))
M = Matrix{Any}(undef, length(keys_sorted) + 1, length(cols))
for (j, c) in enumerate(cols)
    M[1, j] = c
end

for (i, key) in enumerate(keys_sorted)
    a = jl[key]
    b = get(py, key, Dict{String,Float64}())
    terr, model = split(key, "|")
    rel(f, x, y, idx) = begin
        xv = get(x, f, NaN); yv = get(y, f, NaN)
        (isnan(xv) || isnan(yv)) && return NaN
        abs(yv) < 1e-300 && return NaN
        return abs(xv - yv) / abs(yv) * 100
    end
    diffs = [rel(f, a, b, 0) for f in ("rmse", "r2", "aic", "alpha", "lambda", "R0", "A")]
    diffs = filter(!isnan, diffs)
    vals = Any[terr, model,
               get(a, "rmse", NaN), get(b, "rmse", NaN), rel("rmse", a, b, 0),
               get(a, "r2", NaN), get(b, "r2", NaN),
               get(a, "aic", NaN), get(b, "aic", NaN),
               get(a, "alpha", NaN), get(b, "alpha", NaN),
               get(a, "lambda", NaN), get(b, "lambda", NaN),
               get(a, "R0", NaN), get(b, "R0", NaN),
               get(a, "A", NaN), get(b, "A", NaN),
               isempty(diffs) ? NaN : maximum(diffs)]
    for (j, v) in enumerate(vals)
        M[i + 1, j] = v
    end
    @printf("%-24s %-11s RMSE %10.4f / %10.4f  (отн. %.2e %%)  max расхождение %.2e %%\n",
            terr, model, get(a, "rmse", NaN), get(b, "rmse", NaN),
            rel("rmse", a, b, 0), vals[end])
end

open(OUT_CSV, "w") do fio
    writedlm(fio, M, ',')
end
println("\nТаблица сравнения: ", OUT_CSV)
