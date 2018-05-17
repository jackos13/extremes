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
gpd1 = stanc(file = 'code_AP/GPD_simple.stan')
gpd1_run = stan_model(stanc_ret = gpd1) # Compile Stan code

# Prep data
y = exc$exceedance[,1]
N = length(y)

# Run it
fit1 = sampling(gpd1_run,
                data = list(y = y,
                            N = N))

# Plot it
plot(fit1)

# Print it
print(fit1)
