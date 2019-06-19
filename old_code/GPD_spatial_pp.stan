functions {
  // Define log probability mass function
  real GPD_lpdf(real y, real shape, real scale) {
    real lpdf;
    lpdf = - log(scale) + (-1/shape - 1) * log(1 + shape * y/scale);
    return(lpdf);
  }
}
data {
  int<lower = 0> N; // Number of observations
  int<lower = 0> N_loc; // Number of locations
  int<lower = 0> N_z; // Number of sub-locations
  real y[N]; // Vector response
  int loc[N]; // Location of each point
  vector[2] X[N_loc]; // Spatial locations
  vector[2] Z[N_z]; // Spatial locations
}
parameters {
  real shape; // GPD shape parameter for ecah location
  real int_s; // Scale intercept
  real<lower=0> rho_s; // GP scale for scale!
  real<lower=0> alpha_s; // GP standard deviation for scale
  vector[N_z] log_scale_m; // log_scale effect
}
transformed parameters {
  vector[N_loc] log_scale; // log_scale effect
  vector[N_loc] scale; // GPD scale parameter for each location
  vector[N_z] mu_s = rep_vector(int_s, N_z); // Mean parameter for scale
  matrix[N_z, N_z] Sigma_s = cov_exp_quad(Z, alpha_s, rho_s);
  matrix[N_loc, N_z] Sigma_cross_s = cov_exp_quad(X, Z, alpha_s, rho_s);
  for (n in 1:N_z) 
    Sigma_s[n, n] = Sigma_s[n, n] + 1e-12; // Add in to ensure positive definiteness
  log_scale = Sigma_cross_s * inverse(Sigma_s) * log_scale_m;
  scale = exp(log_scale);
}
model {
  // Declare objects only used here
  matrix[N_z, N_z] Sigma_chol_s;
  
  // Priors - mostly vague
  shape ~ normal(0, 1); 
  int_s ~ normal(0, 1); 
  rho_s ~ inv_gamma(5, 5);
  alpha_s ~ normal(0, 1);
  
  // Cholesky decomposition of variance matrix
  Sigma_chol_s = cholesky_decompose(Sigma_s);
  
  // Spatial effect
  log_scale_m ~ multi_normal_cholesky(mu_s, Sigma_chol_s);
  
  // Likelihood
  for (n in 1:N){
    target += GPD_lpdf(y[n] | shape, scale[loc[n]]);
  }
}
