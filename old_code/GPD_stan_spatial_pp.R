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

# Load in the data
exc = readRDS(file = 'data_master.rds')

# Compile the simple GPD stan model
gpd3 = stanc(file = 'code_AP/GPD_spatial_pp.stan')
gpd3_run = stan_model(stanc_ret = gpd3) # Compile Stan code

# Prep data
y = exc$exceedance[,1]
loc = exc$exceedance[,2]
# x1 = exc$locations[,1]
# x2 = exc$locations[,2]
X = exc$locations
Z = X[seq(1,nrow(X), by = 2),] # Every other locaiton
#plot(X); points(Z, pch = 19)
N = length(y)
N_loc = nrow(X)
N_z = nrow(Z)

# Run it
fit3 = sampling(gpd3_run,
                data = list(y = y,
                            loc = loc,
                            X = X,
                            Z = Z,
                            N_z = N_z,
                            N_loc = N_loc,
                            N = N))

# Plot it
plot(fit3, pars = c('shape', 'int_s', 'rho_s', 'alpha_s'))

# Print it
print(fit3, pars = c('shape', 'int_s', 'rho_s', 'alpha_s'))
