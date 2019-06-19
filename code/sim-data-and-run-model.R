############################################################################################
################################# Run Model ################################################
############################################################################################

# Clear workspace
rm(list = ls())

# Required packages:
if (!require("mgcv")) install.packages("mgcv")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("fExtremes")) install.packages("fExtremes")
if (!require("Rcpp")) install.packages("Rcpp")
if (!require("RcppArmadillo")) install.packages("RcppArmadillo")

library(mgcv)
library(tidyverse)
library(fExtremes)
library(Rcpp)
library(RcppArmadillo)

# Set the seed for reproducible results
set.seed(13)

# Get the address:
address <- getwd()

# Source the Rcpp code:
Rcpp::sourceCpp(paste0(address,'/pp-script.cpp'))

# Source the R script with functions to prepare the data:
source(paste0(address,'/prepare-data.R'))

# Set grid dimension (increase grid_len for a larger grid):
grid_len <- 6
x1 <- seq(-1, 1, length.out = grid_len)
x <- as.matrix(expand.grid(x1, x1))

# Spatial dimension (number of gridpoints):
n <- nrow(x)

# A vector of indexes saying which points are kept in the subgrid
# Run the commented line to keep all points (subgrid = full grid)
# Change k - here, it's every 2nd point. 3 = every 3rd point etc.
# xx <- rep(1, nrow(x))
k <- 2
xx <- rep(c(rep(0, k-1), 1), ceiling(n/k))[1:n]

###############################################################
# Setting x to be coords.n and z to be coords.m:
coords.n <- as.matrix(x)
coords.m <- z <- coords.n[xx==1,]
# Get the dimension of the subgrid:
m <- nrow(coords.m)

# Set hyperparameters (adjust as desired):
alpha.phi <- c(0, -0.1, 0.1)
beta.phi <- diag(c(10, 10))
nu.phi <- 2.5
varsigma2.phi <- 1
tau2.phi <- 0.1

alpha.xi <- c(0, 0.1, -0.1)
beta.xi <- diag(c(10, 10))
nu.xi <- 2.5
varsigma2.xi <- 1
tau2.xi <- 0.1

# Calculate Sigma and Psi:
dist.list <- dist_mat_diff(as.matrix(x), as.matrix(x))
Sigma <- matrix(NA, nrow=nrow(x), ncol=nrow(x))
Psi <- matrix(NA, nrow=nrow(x), ncol=nrow(x))

# Filling these one-by-one:
for(i in 1:nrow(Sigma)) {
  for(j in 1:ncol(Sigma)) {
    Sigma[i,j] <- matern_cpp(varsigma2.phi, nu.phi, dist.list[[i]][j,], solve(beta.phi), tau2.phi)
    Psi[i,j] <- matern_cpp(varsigma2.xi, nu.xi, dist.list[[i]][j,], solve(beta.xi), tau2.xi)
  }
}

# Getting the mean of the GP:
X <- cbind(1, x)
mu.phi <- as.matrix(X) %*% alpha.phi
mu.xi <- as.matrix(X) %*% alpha.xi

# Simulating the surfaces phi and xi:
phi <- t(rmvn(1, t(mu.phi), Sigma))
xi <- t(rmvn(1, t(mu.xi), Psi))

# Simulate some random values from the GPD based on these
# 50-100 excesses (random amounts) at each gridpoint:
data <- vector('list', length=nrow(x))
n1 <- 50
n2 <- 100

for(i in 1:length(data)) {
  data[[i]] <- rgpd(sample(seq(n1, n2)), xi[i], 0, exp(phi[i]))
}
data <- lapply(data, as.vector)

# Save the data set at this point so that we have everything to hand
# The list of exceedances
# The locations
# The sub-locations (optional, part of the modelling choice)
data_master = list()
data_master$exceedance <- data
data_master$locations <- coords.n
data_master$sublocations <- coords.m
saveRDS(data_master, file = 'data_master.rds')
# stop()

# The data is now ready to put into the MCMC function.
# Now to set everything else up:
# It needs the following:
# List data, List big_beta, mat big_beta_mat, List start, int iterations, mat covar1,
# mat covar2, mat bigcovar1, mat bigcovar2, mat distance_vectors_n, mat distance_vectors_m,
# int dimension, List step, List prior, int burnin, int nth, int n_dim, int m_dim, vec ind_m

# A list of all possible numbers in the beta matrices:
beta.vec <- c(0, 0.05, 1, 10)

# Getting the dimension of beta, which is essentially
# the number of spatial coordinates - lat, lon, (elevation possibly too)
spatial.dim <- ncol(x)

# Using a function to calculate all possible matrices beta,
# keeping only those which are symmetric and positive definite, and returning them
# as a list and as a matrix:
list1 <- get_betas(beta.vec, spatial.dim)
big_beta <- list1[[1]]
big_beta_mat <- list1[[2]]

# Now making the matrix of covariates X,
# where here it is assumed that scale
# and shape get the same X
# This is easily modified if necessary
bigcov1 <- as.matrix(cbind(1, x))
bigcov2 <- bigcov1

# And the matrix of covariates Z for the reduced grid:
# (again, keeping it the same for now)
cov1 <- as.matrix(cbind(1, x[xx==1,]))
cov2 <- cov1
dim.cov1 <- ncol(cov1)
dim.cov2 <- ncol(cov2)

###############################################################
# Set the prior parameters
# (a few of these are now redundant, but the ordering is needed, so they're left in for ease
# e.g., some are from when parameters were continuous then moved to discrete etc.)
# Adjust these as desired
prior <- list('alpha_scale_hyper_mean'  = c(rep(0, dim.cov1)),
              'alpha_scale_hyper_sd'    = diag(rep(2, dim.cov1)),
              'beta_scale_prior'        = 0, # redundant
              'scale_hyper_mean'        = 0, # redundant
              'scale_hyper_sd'          = 5, # redundant
              
              'alpha_shape_hyper_mean'  = c(rep(0, dim.cov2)),
              'alpha_shape_hyper_sd'    = diag(rep(2, dim.cov2)),
              'beta_shape_prior'        = 0, # redundant
              'shape_hyper_mean'        = 0, # redundant
              'shape_hyper_sd'          = 5, # redundant
              
              'sigsq_scale_hyper_mean'  = 0,
              'sigsq_scale_hyper_sd'    = 1,
              'tausq_scale_hyper_mean'  = -2.3,   # This is ~log(0.1)
              'tausq_scale_hyper_sd'    = 0.5,
              'nu_scale_hyper_mean'     = 0, # redundant
              'nu_scale_hyper_sd'       = 2, # redundant
              
              'sigsq_shape_hyper_mean'  = 0,
              'sigsq_shape_hyper_sd'    = 1,
              'tausq_shape_hyper_mean'  = -2.3,   # This is ~log(0.1)
              'tausq_shape_hyper_sd'    = 0.5,
              'nu_shape_hyper_mean'     = 0, # redundant
              'nu_shape_hyper_sd'       = 2, # redundant
              
              'nu_scale_hyper_discrete' = c(0.5, 2.5),
              'nu_shape_hyper_discrete' = c(0.5, 2.5))

###############################################################
# Define the starting values for the algorithm:
# (These are flexible to changes)
start.phi <- c(rep(0.01, n))
start.xi <- c(rep(0.01, n))

start.phi.m <- start.phi[xx==1]
start.xi.m <- start.xi[xx==1]

start <- list('lscale'              = start.phi.m,
              'alpha_scale'         = runif(dim.cov1),
              'beta_scale'          = big_beta[[1]],  
              'sigsq_scale'         = 1,                    
              'tausq_scale'         = 0,
              
              'shape'              = start.xi.m,
              'alpha_shape'         = runif(dim.cov1),
              'beta_shape'          = big_beta[[1]],
              'sigsq_shape'         = 1,                  
              'tausq_shape'         = 0,                
              
              'nlscale'             = start.phi,
              'nshape'              = start.xi,
              
              "nu_scale" = 0.5,
              "nu_shape" = 0.5)

###############################################################
# Define the steps used. These need to be adjusted to 'tune' the algorithm:
step <- list('scale_step'      = 0.02,
             'alpha_scale_step' = c(1.4, 0.4, 0.4),
             'sigsq_scale_step' = 1,
             'tausq_scale_step' = 0.8,
             
             'shape_step'      = 0.02,
             'alpha_shape_step' = c(1.2, 0.4, 0.4),
             'sigsq_shape_step' = 0.7,
             'tausq_shape_step' = 0.8,
             
             'blank1' = 0, # redundant
             'blank2' = 0, # redundant
             
             'nu_scale_step' = 0.12,
             'nu_shape_step' = 0.12)

###############################################################
# Final pieces to define:
# Creating the cubes necessary for the discrete updates of beta and nu:
# (have a folder called cubes in your working directory)
addresses <- c(paste0(address, '/cubes/cube1.cube'),
               paste0(address, '/cubes/cube2.cube'),
               paste0(address, '/cubes/big_cube1.cube'),
               paste0(address, '/cubes/big_cube2.cube'))
# save_cubes is in the R code in the prepare-data.R script
save_cubes(addresses, coords.m, coords.n, big_beta, prior, start)

# ind_m is used to update across the grid in a random order, for each iteration i
ind_m <- 0:(m-1)

# Setting the number of iterations, the burnin and the n^th value to save
iterations <- 1e4
burnin <- 1e3
nth <- 1e1

# This can be needed on larger grids to ensure
# stability of the algorithm - not needed for small grids
# start <- find.init(start, it.max=1000, addresses, m, n, data,
#                   bigcov1, bigcov2, cov1, cov2)

# Now to reset the tausq parameters after the cubes have been created:
start$tausq_scale <- 0.01; start$tausq_shape <- 0.01

###############################################################
# Run the code:
chainC <- mcmcC(data, big_beta, big_beta_mat, start,
                iterations, cov1, cov2,
                bigcov1, bigcov2,
                coords.n, coords.m, spatial.dim, step, prior,
                burnin, nth, n, m, ind_m)

# save(chainC, file="chain.RData")
# break()
# load("chain.RData")

# 1e4 is probably enough to see if the steps need adjusting
chainC$acc_scale; chainC$acc_shape
chainC$acc_alpha_scale; chainC$acc_alpha_shape
chainC$acc_sigsq_scale; chainC$acc_sigsq_shape
chainC$acc_tausq_scale; chainC$acc_tausq_shape
# 1e5 should be enough to converge

##########################################################################################
# Analysis plots:
##########################################################################################

# In all plots below, where vertical/horizontal lines
# are overlain, red is the known value and black is the posterior mean 

# Plotting the alpha_scale (phi) parameters:
par(mfrow=c(1, ncol(chainC$alpha_scale)))
for(i in 1:ncol(chainC$alpha_scale)) {
  plot(chainC$alpha_scale[,i], type="l")
  abline(h=alpha.phi[i], col="red")
  abline(h=mean(chainC$alpha_scale[,i]), col="black")
}

# Plotting the alpha_shape (xi) parameters:
par(mfrow=c(1, ncol(chainC$alpha_shape)))
for(i in 1:ncol(chainC$alpha_shape)) {
  plot(chainC$alpha_shape[,i], type="l")
  abline(h=alpha.xi[i], col="red")
  abline(h=mean(chainC$alpha_shape[,i]), col="black")
}

# Printing the table and actual beta_scale values from the matrix:
for(i in 1:ncol(chainC$beta_scale)) {
  print(table(chainC$beta_scale[,i]))
  print(paste0('True value of beta.phi = ', beta.phi[i]))
}

# Printing the table and actual beta_shape values from the matrix:
for(i in 1:ncol(chainC$beta_shape)) {
  print(table(chainC$beta_shape[,i]))
  print(paste0('True value = ', beta.xi[i]))
}

# Histogram for varsigma^2 for the scale:
par(mfrow=c(1, 1))
hist(chainC$sigsq_scale)
abline(v=varsigma2.phi, col="red")
abline(v=mean(chainC$sigsq_scale), col="black")

# Histogram for varsigma^2 for the shape:
hist(chainC$sigsq_shape)
abline(v=varsigma2.xi, col="red")
abline(v=mean(chainC$sigsq_shape), col="black")

# Histogram for tau^2 for the scale:
hist(chainC$tausq_scale)
abline(v=tau2.phi, col="red")
abline(v=mean(chainC$tausq_scale), col="black")

# Histogram for tau^2 for the shape:
hist(chainC$tausq_shape)
abline(v=tau2.xi, col="red")
abline(v=mean(chainC$tausq_shape), col="black")

# Table for nu for the scale:
table(chainC$nu_scale)
print(paste0('True value = ', nu.phi))

# Table for nu for the shape:
table(chainC$nu_shape)
print(paste0('True value = ', nu.xi))

##########################################################################################
# End
##########################################################################################
