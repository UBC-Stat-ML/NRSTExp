###############################################################################
# Titanic stan
###############################################################################

using NRST
using NRSTExp
using NRSTExp.ExamplesGallery
using SplittableRandoms
using StanSample

# define and tune an NRSTSampler as template
tm  = Titanic()
rng = SplittableRandom(11035)
ns, TE, Λ = NRSTSampler(
    tm,
    rng,
    use_mean   = true,
    maxcor     = 0.9,
    γ          = 8.0,
    xpl_smooth_λ = 0.0001
);
res=parallel_run(ns,rng,NRST.NRSTTrace(ns),TE=TE,δ=0.5);
sin_res = NRST.inference_on_V(res,h=sin)
res.trVs[end]
using Plots
σs = [first(pars) for pars in ns.np.xplpars]
plot(σs)
plot(ns.np.nexpls)

#######################################
# run stan
#######################################

# model definition
model_str = """
data {
  int<lower=0> n;            // number of data items
  int<lower=0> p;            // number of predictors
  matrix[n, p] Q;            // scaled Q in the QR decomposition of X
  matrix[p, p] Rinv;         // inverse of scaled R
  vector[p] Ritm;            // Rinv' * m, where m is the vector of column means of X
  int<lower=0,upper=1> y[n]; // response
}
parameters {
  real alpha;                // intercept
  vector[p] theta;           // coefficients on Q
  real<lower=0> sigma;       // std dev of coefficients
}
transformed parameters {
  real alpha_m;                       // mean effect
  alpha_m = dot_product(Ritm, theta);
  vector[p] beta;                     // coefficients on X
  beta = Rinv * theta;
}
model {
  sigma ~ exponential(1);
  alpha ~ cauchy(0,sigma);
  beta  ~ cauchy(0,sigma);                              // note: can put prior on a transformed param because the trans is linear so no need to account for the logabsdetjac (a constant)
  y     ~ bernoulli_logit(alpha + alpha_m + Q * theta); // likelihood
}
""";

# construct model, data, and sample
sm = SampleModel("Titanic", model_str);
data = Dict(
    "n"    => tm.n,
    "p"    => tm.p,
    "Q"    => tm.Q,
    "Rinv" => tm.Rinv,
    "Ritm" => tm.Ritm,
    "y"    => 1*tm.y
)
rc = stan_sample(sm; data, num_chains=1, num_samples=4000);#sig_figs=18
res_sum = read_summary(sm)
res_sam = read_samples(sm, :table);

# parse samples and compute V
function get_var_samples(st, vn)
  vns  = String.(keys(st))
  syms = Symbol.(vns[sort(findall(Base.Fix2(contains, Regex(vn*"(\$|\\.)")), vns))]) # regex: name either ends after vn, or it has a "." (for vectors)
  collect(hcat((st[syms[i]] for i in eachindex(syms))...)')
end
θs = get_var_samples(res_sam, "theta")
αs = get_var_samples(res_sam, "alpha")
Vs = [NRST.V(tm, [1;α;θ]) for (α,θ) in zip(eachcol(αs),eachcol(θs))]

using StatsPlots
density(res.trVs[end], label="NRST")
density!(Vs, label="Stan")
using mcmcse, Statistics
x = sin.(Vs)
λ  = var(x)   # estimate variance under stationary distribution 
σ² = mcvar(x) # estimate asymptotic variance of the MCMC estimate 
hw = sqrt(σ² / length(x))
(sin_res[!,"C.I. High"]-sin_res[!,"C.I. Low"])/2
xess = length(x)*λ/σ²
