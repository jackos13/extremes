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
gpd2 = stanc(file = 'code_AP/GPD_spatial.stan')
gpd2_run = stan_model(stanc_ret = gpd2) # Compile Stan code

# Prep data
y = exc$exceedance[,1]
loc = exc$exceedance[,2]
# x1 = exc$locations[,1]
# x2 = exc$locations[,2]
X = exc$locations
N = length(y)
N_loc = nrow(X)

# Run it
fit2 = sampling(gpd2_run,
                data = list(y = y,
                            loc = loc,
                            X = X,
                            N_loc = N_loc,
                            N = N))

# Plot it
plot(fit2, pars = c('shape', 'int_s', 'rho_s', 'alpha_s'))

# Print it
print(fit2, pars = c('shape', 'int_s', 'rho_s', 'alpha_s'))
