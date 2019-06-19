# Run a very basic Generalised Pareto distribution model in Stan

# Boiler plate code. Clear workspace and load in packages
rm(list = ls())

# Install devtools nad auf if not already installed
if(!require(devtools)) install.packages('devtools')
if(!require(auf)) devtools::install_github('andrewcparnell/auf')

library(auf)
packages('tidyverse', 'boot', 'rstan', 'gridExtra',
         'ggpubr', 'bayesplot', 'mgcv', 'loo')
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Set the seed for repeatable results
set.seed(123)

# Load in the data
exc = readRDS(file = 'data_master.rds')

# Compile the simple GPD stan model
gpd4 = stanc(file = 'code_AP/GPD_spatial_pp_both.stan')
gpd4_run = stan_model(stanc_ret = gpd4) # Compile Stan code

# Prep data
y = exc$exceedance[,1]
loc = exc$exceedance[,2]
X = exc$locations
#Z = X[seq(1,nrow(X), by = 2),] # Every other location
#plot(X); points(Z, pch = 19)
N = length(y)
N_loc = nrow(X)
N_z = 10 # Choose a random 10 rows
Z = X[sample(1:nrow(X), N_z),] 

# Run it
fit4 = sampling(gpd4_run,
                data = list(y = y,
                            loc = loc,
                            X = X,
                            Z = Z,
                            N_z = N_z,
                            N_loc = N_loc,
                            N = N))

# Plot it
plot(fit4, pars = c('int_sc', 'rho_sc', 'alpha_sc',
                    'int_sh', 'rho_sh', 'alpha_sh'))

# Print it
print(fit4, pars = c('int_sc', 'rho_sc', 'alpha_sc',
                     'int_sh', 'rho_sh', 'alpha_sh'))

# Compare it to truth
truth = readRDS(file = 'truth_master.rds')
shape_stan = colMeans(rstan::extract(fit4, pars = 'shape')$shape)
plot(shape_stan, truth$shape)
abline(a=0, b = 1)
scale_stan = colMeans(rstan::extract(fit4, pars = 'scale')$scale)
plot(scale_stan, exp(truth$log_scale))
abline(a=0, b = 1)
