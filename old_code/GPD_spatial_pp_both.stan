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
  real int_sc; // Scale intercept
  real<lower=0> rho_sc; // GP scale for scale!
  real<lower=0> alpha_sc; // GP standard deviation for scale
  vector[N_z] log_scale_m; // log_scale effect
  real int_sh; // Shape intercept
  real<lower=0> rho_sh; // GP scale for shape
  real<lower=0> alpha_sh; // GP standard deviation for shape
  vector[N_z] shape_m; // shape effect
}
transformed parameters {
  vector[N_loc] log_scale; // log_scale effect
  vector[N_loc] scale; // GPD scale parameter for each location
  vector[N_z] mu_sc = rep_vector(int_sc, N_z); // Mean parameter for scale
  matrix[N_z, N_z] Sigma_sc = cov_exp_quad(Z, alpha_sc, rho_sc);
  matrix[N_loc, N_z] Sigma_cross_sc = cov_exp_quad(X, Z, alpha_sc, rho_sc);
  vector[N_loc] shape; // Shape effect
  vector[N_z] mu_sh = rep_vector(int_sh, N_z); // Mean parameter for scale
  matrix[N_z, N_z] Sigma_sh = cov_exp_quad(Z, alpha_sh, rho_sh);
  matrix[N_loc, N_z] Sigma_cross_sh = cov_exp_quad(X, Z, alpha_sh, rho_sh);
  
  for (n in 1:N_z) 
    Sigma_sc[n, n] = Sigma_sc[n, n] + 1e-12; // Add in to ensure positive definiteness
  log_scale = Sigma_cross_sc * inverse(Sigma_sc) * log_scale_m;
  scale = exp(log_scale);
  
  for (n in 1:N_z) 
    Sigma_sh[n, n] = Sigma_sh[n, n] + 1e-12; // Add in to ensure positive definiteness
  shape = Sigma_cross_sh * inverse(Sigma_sh) * shape_m;
}
model {
  // Declare objects only used here
  matrix[N_z, N_z] Sigma_chol_sc;
  matrix[N_z, N_z] Sigma_chol_sh;
  
  // Priors 
  int_sc ~ normal(0, 1); 
  rho_sc ~ inv_gamma(5, 5);
  alpha_sc ~ normal(0, 1);
  int_sh ~ normal(0, 1); 
  rho_sh ~ inv_gamma(5, 5);
  alpha_sh ~ normal(0, 1);
  
  // Cholesky decomposition of variance matrix
  Sigma_chol_sc = cholesky_decompose(Sigma_sc);
  Sigma_chol_sh = cholesky_decompose(Sigma_sh);
  
  // Spatial effect
  log_scale_m ~ multi_normal_cholesky(mu_sc, Sigma_chol_sc);
  shape_m ~ multi_normal_cholesky(mu_sh, Sigma_chol_sh);
  
  // Likelihood
  for (n in 1:N){
    target += GPD_lpdf(y[n] | shape[loc[n]], scale[loc[n]]);
  }
}
