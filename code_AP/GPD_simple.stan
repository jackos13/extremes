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
  real y[N]; // Vector response
}
parameters {
  real shape; // GPD shape parameter
  real scale; // GPD scale parameter
}
model {
  // Priors - mostly vague
  shape ~ normal(0, 10); 
  scale ~ normal(0, 10); 

  // Likelihood
  for (n in 1:N){
    target += GPD_lpdf(y[n] | shape, scale);
  }
}
