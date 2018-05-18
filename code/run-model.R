############################################################################################
################################# Run Model ################################################
############################################################################################

# Clear workspace
rm(list = ls())

# Required packages:
if (!require("rpgm")) install.packages("rpgm")
library(rpgm)
library(mgcv)
library(tidyverse)

# Set the seed so we get the same data set every time
set.seed(123)

# Get the address:
address <- getwd()

# Source the Rcpp code:
Rcpp::sourceCpp(paste0(address,'/code/pp-script.cpp'))

# Source the R script:
source(paste0(address,'/code/prepare-data.R'))

# Set grid:
grid_len = 6
x1 <- seq(-1, 1, length.out = grid_len)
x <- as.matrix(expand.grid(x1, x1))

# Spatial dimension:
n <- nrow(x)

# Keep both grids the same for now:
z <- x
m <- nrow(z)

# A vector of indexes saying which points are kept in the subgrid
# (Keeping all for now)
xx <- rep(1, nrow(x))

###############################################################
# Setting x to be coords.n and z to be coords.m to use my old notation later on:
coords.n <- as.matrix(x)
coords.m <- as.matrix(z)

# Set hyperparameters:
alpha.phi <- c(1, 0, 0)
beta.phi <- diag(2)
nu.phi <- 4.5
varsigma2.phi <- 1
tau2.phi <- 0.1

alpha.xi <- c(0, 0, 0)
beta.xi <- diag(2)
nu.xi <- 4.5
varsigma2.xi <- 1
tau2.xi <- 0.1

# Calculate Sigma and Psi:
dist.list <- dist_mat_diff(as.matrix(z), as.matrix(z))
Sigma <- matrix(NA, nrow=nrow(z), ncol=nrow(z))
Psi <- matrix(NA, nrow=nrow(z), ncol=nrow(z))

# Filling these one-by-one:
for(i in 1:nrow(Sigma)) {
  for(j in 1:ncol(Sigma)) {
    Sigma[i,j] <- matern_cpp(varsigma2.phi, nu.phi, dist.list[[i]][j,], solve(beta.phi), tau2.phi)
    Psi[i,j] <- matern_cpp(varsigma2.xi, nu.xi, dist.list[[i]][j,], solve(beta.xi), tau2.xi)
  }
}

# Getting the mean of the GP:
Z <- cbind(1, z)
mu.phi <- as.matrix(Z) %*% alpha.phi
mu.xi <- as.matrix(Z) %*% alpha.xi

# Simulating phi and xi:
phi <- t(rmvn(1, t(mu.phi), Sigma))
xi <- t(rmvn(1, t(mu.xi), Psi))

# Simulate some random values from the GPD based on these
# 50-100 excesses (random amounts) at each gridpoint:
data <- vector('list', length=nrow(z))
n1 <- 50
n2 <- 100

for(i in 1:length(data)) {
  data[[i]] <- rgpd(sample(seq(n1, n2)), xi[i], 0, exp(phi[i]))
}

# Save the data set at this point so that we have everything to hand
# The list of exceedances as a data frame with location
# The locations 
# The sub-locations (optional, part of the modelling choice)

#FIrst convert the data into a proper data frame 
data2 = unlist(data)
data_locs = rep(1:nrow(x), 
                times = unlist(lapply(data, length)))
data_master = list()
data_master$exceedance = data.frame(exceedance = data2, 
                                    location = data_locs)
data_master$locations = coords.n
data_master$sublocations = coords.m
saveRDS(data_master, file = 'data_master.rds')
#stop()

# data is now ready to put into the MCMC function.
# Now to set everything else up:
# It needs the following:
# List data, List big_beta, mat big_beta_mat, List start, int iterations, mat covar1,
# mat covar2, mat bigcovar1, mat bigcovar2, mat distance_vectors_n, mat distance_vectors_m,
# int dimension, List step, List prior, int burnin, int nth, int n_dim, int m_dim, vec ind_m

# A list of all possible numbers in the beta matrices:
beta.vec <- c(0.01, 0.1, 1, 10, 100, 1000)

# Getting the dimension of beta, which is essentially
# the number of spatial coordinates - lat, lon, (elevation maybe)
# should really only be 3 or 4 at its max
spatial.dim <- ncol(z)

# Using a function to calculate all possible matrices beta,
# keep only those which are symmetric and positive definite, and returning them
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
# (again, I'm keeping it the same for now)
cov1 <- as.matrix(cbind(1, z))
cov2 <- cov1
dim.cov1 <- ncol(cov1)
dim.cov2 <- ncol(cov2)

###############################################################
# Set the prior parameters
# (a few of these are now redundant, but the ordering is needed, so I've left them in....
# e.g., some are from when parameters were continuous then moved to discrete etc.)
prior <- list('alpha_scale_hyper_mean'  = c(rep(0, dim.cov1)),
              'alpha_scale_hyper_sd'    = diag(rep(10, dim.cov1)),
              'beta_scale_prior'        = 0, #beta.vec,
              'scale_hyper_mean'        = 0, 
              'scale_hyper_sd'          = 5,
              
              'alpha_shape_hyper_mean'  = c(rep(0, dim.cov2)),
              'alpha_shape_hyper_sd'    = diag(rep(10, dim.cov2)),
              'beta_shape_prior'        = 0, #beta.vec,
              'shape_hyper_mean'        = 0,
              'shape_hyper_sd'          = 5,
              
              'sigsq_scale_hyper_mean'  = 0,
              'sigsq_scale_hyper_sd'    = 1,
              'tausq_scale_hyper_mean'  = -2.3,   # This is ~log(0.1)
              'tausq_scale_hyper_sd'    = 5,
              'nu_scale_hyper_mean'     = 0,
              'nu_scale_hyper_sd'       = 10,
              
              'sigsq_shape_hyper_mean'  = 0,
              'sigsq_shape_hyper_sd'    = 1,
              'tausq_shape_hyper_mean'  = -2.3,   # This is ~log(0.1)
              'tausq_shape_hyper_sd'    = 5,
              'nu_shape_hyper_mean'     = 0,
              'nu_shape_hyper_sd'       = 10,
              
              'nu_scale_hyper_discrete' = seq(0.5, 8.5, by=2),
              'nu_shape_hyper_discrete' = seq(0.5, 8.5, by=2))

###############################################################
# Define the starting values for the algorithm:
# (should be flexible to changes here)
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
              'alpha_shape'         = c(rep(0, dim.cov1)),
              'beta_shape'          = big_beta[[1]],
              'sigsq_shape'         = 1,                  
              'tausq_shape'         = 0,                
              
              'nlscale'             = start.phi,
              'nshape'              = start.xi,
              
              "nu_scale" = 0.5,
              "nu_shape" = 0.5)

###############################################################
# Setting x to be coords.n and z to be coords.m to use my old notation:
coords.n <- as.matrix(x)
coords.m <- as.matrix(z)

###############################################################
# Define the steps used. These need to be adjusted to 'tune' the algorithm:
step <- list('scale_step'      = 0.013,
             'alpha_scale_step' = c(1.4, 0.4, 0.4),
             'sigsq_scale_step' = 1,
             'tausq_scale_step' = 0.8,
             
             'shape_step'      = 0.013,
             'alpha_shape_step' = c(1.2, 0.4, 0.4),
             'sigsq_shape_step' = 0.7,
             'tausq_shape_step' = 0.8,
             
             'blank1' = 0,
             'blank2' = 0,
             
             'nu_scale_step' = 0.12,
             'nu_shape_step' = 0.12)

###############################################################
# Final pieces to define:
# Creating the cubes necessary for the discrete updates of beta and nu:
addresses <- c(paste0(address, '/cube1.cube'),
               paste0(address, '/cube2.cube'),
               paste0(address, '/big_cube1.cube'),
               paste0(address, '/big_cube2.cube'))
save_cubes(addresses) # save_cubes is in the R code in the prepare-data.R script

# ind_m is used to randomly update across the grid, for each iteration i
ind_m <- 0:(m-1)

# Setting the number of iterations, the burnin and the n^th value to save
iterations <- 1e4
burnin <- 1e3
nth <- 1e1

###############################################################
# Run the code:
chainC <- mcmcC(data, big_beta, big_beta_mat, start,
                iterations, cov1, cov2,
                bigcov1, bigcov2,
                coords.n, coords.m, spatial.dim, step, prior,
                burnin, nth, n, m, ind_m)

# The code takes about 4 minutes for 1e4, 40 minutes for 1e5 etc.
# 1e4 is probably enough to see if the steps need adjusting
attach(chainC)
acc_scale; acc_shape
acc_alpha_scale; acc_alpha_shape
acc_sigsq_scale; acc_sigsq_shape
acc_tausq_scale; acc_tausq_shape
# Then 1e5 (40 minutes) should be enough to converge

##########################################################################################
# Analysis plots:
##########################################################################################
attach(chainC)

# In all plots below, where vertical/horizontal lines
# are overlain, red is the known value and black is the posterior mean 

# Quick check of the difference between the known values of the scale and the shape
# on the grid, and the posterior values:
round(apply(scale, 2, mean) - exp(phi), 1)
round(apply(shape, 2, mean) - xi, 1)

plot(apply(scale, 2, mean),exp(phi))
abline(a=0, b=1)
plot(apply(shape, 2, mean),xi)
abline(a=0, b=1)

truth = list(log_scale = phi, shape = xi)
saveRDS(truth, file = 'truth_master.rds')

# Some plots below - values of the scale and the shape at 4 random points on the grid:
# red line - true value
# black line - posterior mean
ind <- sample(1:n, 4)
par(mfrow=c(2, 2))

# scale first:
for(i in 1:4) {
  plot(scale[,ind[i]], type="l", xlab=paste0("Gridpoint:", ind[i]))
  abline(h=exp(phi[ind[i]]), col="red")
  abline(h=mean(scale[,ind[i]]), col="black")
}

# shape now:
for(i in 1:4) {
  plot(shape[,ind[i]], type="l", xlab=paste0("Gridpoint:", ind[i]))
  abline(h=xi[ind[i]], col="red")
  abline(h=mean(shape[,ind[i]]), col="black")
}

# Plotting the alpha_scale (phi) parameters:
par(mfrow=c(1, ncol(alpha_scale)))
for(i in 1:ncol(alpha_scale)) {
  plot(alpha_scale[,i], type="l")
  abline(h=alpha.phi[i], col="red")
  abline(h=mean(alpha_scale[,i]), col="black")
}

# Plotting the alpha_shape (xi) parameters:
par(mfrow=c(1, ncol(alpha_shape)))
for(i in 1:ncol(alpha_shape)) {
  plot(alpha_shape[,i], type="l")
  abline(h=alpha.xi[i], col="red")
  abline(h=mean(alpha_shape[,i]), col="black")
}

# Printing the table and actual beta_scale values from the matrix:
for(i in 1:ncol(beta_scale)) {
  print(table(beta_scale[,i]))
  print(paste0('True value of beta.phi = ', beta.phi[i]))
}

# Printing the table and actual beta_shape values from the matrix:
for(i in 1:ncol(beta_shape)) {
  print(table(beta_shape[,i]))
  print(paste0('True value = ', beta.xi[i]))
}

# Histogram for varsigma^2 for the scale:
par(mfrow=c(1, 1))
hist(sigsq_scale)
abline(v=varsigma2.phi, col="red")
abline(v=mean(sigsq_scale), col="black")

# Histogram for varsigma^2 for the shape:
hist(sigsq_shape)
abline(v=varsigma2.xi, col="red")
abline(v=mean(sigsq_shape), col="black")

# Histogram for tau^2 for the scale:
hist(tausq_scale)
abline(v=tau2.phi, col="red")
abline(v=mean(tausq_scale), col="black")

# Histogram for tau^2 for the shape:
hist(tausq_shape)
abline(v=tau2.xi, col="red")
abline(v=mean(tausq_shape), col="black")

# Table for nu for the scale:
table(nu_scale)
print(paste0('True value = ', nu.phi))

# Table for nu for the shape:
table(nu_shape)
print(paste0('True value = ', nu.xi))

##########################################################################################
# End
##########################################################################################
