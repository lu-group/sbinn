#import Pkg; Pkg.add("DifferentialEquations")
#import Pkg; Pkg.add("ForwardDiff")
#import Pkg; Pkg.add("DiffResults")
#import Pkg; Pkg.add("DiffEqSensitivity")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("CSV")
#import Pkg; Pkg.add("Dierckx")
#import Pkg; Pkg.add("StatsPlots")
#import Pkg; Pkg.add("LaTeXStrings")
#import Pkg; Pkg.add("Plots")

using DifferentialEquations
using ForwardDiff
using DiffResults
using DiffEqSensitivity
using Statistics
using Random
using LinearAlgebra
using DataFrames
using CSV
using Dierckx
using Plots
using StatsPlots
using LaTeXStrings

function f_apop(dx, x, p, t)
    x1, x2, x3, x4, x5, x6 = x
    E, tp, ti, td, k, Rm, a1, C1, C2, C4, C5, Ub, U0, Um, Rg, alpha, beta = p
    Vp = 3
    Vi = 11
    Vg = 10
    C3 = 100/100
    meal_t = 300, 650, 1100
    meal_q = 60e3, 40e3, 50e3
    f1 = Rm / (1 + exp(x3 / (Vg * C1) - a1))
    f2 = Ub * (1 - exp(-x3 / (Vg * C2)))
    kappa = (1 / Vi + 1 / (E * ti)) / C4
    f3 = (U0 + Um / (1 + (kappa * x2)^(-beta))) / (Vg * C3)
    f4 = Rg / (1 + exp(alpha * (1 - x6 / (Vp * C5))))
    dt1 = t - meal_t[1]
    dt2 = t - meal_t[2]
    dt3 = t - meal_t[3]
    IG1 = 0.5 * meal_q[1] * k * exp(-k * dt1) * (sign(dt1) + 1)
    IG2 = 0.5 * meal_q[2] * k * exp(-k * dt2) * (sign(dt2) + 1)
    IG3 = 0.5 * meal_q[3] * k * exp(-k * dt3) * (sign(dt3) + 1)
    IG = IG1 + IG2 + IG3
    tmp = E * (x1 / Vp - x2 / Vi)
    dx[1] = (f1 - tmp - x1 / tp)
    dx[2] = (tmp - x2 / ti)
    dx[3] = (f4 + IG - f2 - f3 * x3)
    dx[4] = (x1 - x4) / td
    dx[5] = (x4 - x5) / td
    dx[6] = (x5 - x6) / td
end

cscale = 1; tscale = 1.;
p = [0.200853979,5.986360211,101.2036365,11.97776077,0.00833154,208.6221286,6.592132806,301.2623884,37.65196466,78.75855574,25.93801618,71.32611075,4.063671595,88.97766871,179.85672,7.536059566,1.783340558]

x0 = [36., 44., 11000., 0., 0., 0.] / cscale
tspan = (0.0, 1800.0)
prob_apop = ODELocalSensitivityProblem(f_apop, x0, tspan, p)

sol_apop = solve(prob_apop, alg_hints=[:stiff], saveat=0.1)
x_apop, dp_apop = extract_local_sensitivities(sol_apop)

lab = [L"E", L"tp", L"ti", L"td", L"k", L"Rm", L"a1", L"C1", L"C2", L"C4", L"C5", L"Ub", L"U0", L"Um", L"Rg", L"alpha", L"beta"]
σ = 0.01 * std(x_apop, dims=2)
cov_ϵ = σ[4]
dp = dp_apop
cols = 4:4

#plot(sol_apop.t, x_apop[4,:], lw=2)

Nt = length(dp[1][1,:])
Nstate = length(dp[1][:,1])
Nparam = length(dp[:,1])
F = zeros(Float64, Nparam, Nparam)

perm = vcat(1, sort(rand(2:Nt-1, Nt÷5)), Nt)

for i in perm
    S = reshape(dp[1][:,i], (Nstate,1))
    for j = 2:Nparam
        S = hcat(S, reshape(dp[j][:,i], (Nstate,1)))
    end
    global F += S[cols,:]' * inv(cov_ϵ) * S[cols,:]
end

C = inv(F)
R = ones(size(C))
R = [C[i,j]/sqrt(C[i,i]*C[j,j]) for i = 1:size(C)[1], j = 1:size(C)[1]]
#heatmap(R, xlims=(0.5,size(R)[1]+0.5), aspect_ratio = 1, color = :inferno, clims = (-1, 1),
#        xticks = (1:1:size(C)[1], lab), xtickfont = font(14, "Times"),
#        yticks = (1:1:size(C)[1], lab), ytickfont = font(14, "Times"), fmt = :pdf, dpi=300)
#savefig("correlation_matrix")

abs.(R) .> 0.99

lowerbound = sqrt.(diag(inv(F))) / tscale
lowerbound[1:3:7] = lowerbound[1:3:7] / cscale
for i = 1:length(lab)
    println(lab[i], '\t', lowerbound[i])
end

for i = 1:Nparam
    println(eigvals(F)[i])
    println(eigvecs(F)[:,i])
    println('\n')
end

#bar(eigvecs(F)[:,1:9], ylabel = "FIM null eigenvector coefficients", ytickfont = font(12, "Times"),
#    xticks = (1:1:size(C)[1], lab), xtickfont = font(12, "Courier"),
#    legendfontsize = 10, label = [L"null_1" L"null_2" L"null_3" L"null_4" L"null_5" L"null_6" L"null_7" L"null_8" L"null_9"], fmt = :png,
#    legend=:topright, dpi=300)
#savefig("nulleigen_apop")
