########################################################################
# Functions to prepare to run the PP GPD:
########################################################################

########################################################################
# Getting all potential betas
get_betas <- function(beta.vec, spatial.dim) {
  # Getting the number of unique elements in the symmetric
  # matrix, using n(n+1)/2
  beta.sym.dim <- (spatial.dim) * (spatial.dim + 1) / 2
  
  # Filling a list with repeated beta.vec (a roundabout way, but it works)
  beta <- vector("list", length = beta.sym.dim)
  for(i in 1:beta.sym.dim) {
    beta[[i]] <- beta.vec
  }
  
  # This needs to be the number of elements in the matrix, eventually -
  # 3 for a 2x2 matrix (since it's symmetric)
  # 6 for a 3x3 matrix (since it's symmetric) etc.
  big.beta <-  as.matrix(expand.grid(beta))
  
  # Now to reflect it, and generate all possible matrices using those values:
  big.list <- vector("list", nrow(big.beta))
  temp <- diag(spatial.dim)
  ind <- lower.tri(temp, diag=T)
  for (i in 1:length(big.list)) {
    
    temp[ind] <- big.beta[i,]
    temp[upper.tri(temp)] <- t(temp)[upper.tri(temp)]
    big.list[[i]] <- temp
    
  }
  
  # Using my own C++ function to test for positive definiteness:
  list1_beta <- lapply(big.list, pos_def_testC)
  
  # Only keeping those which are positive definite:
  big_beta <- big.list[unlist(list1_beta)]
  
  # Turning the list into a matrix - each row is the components of a single beta matrix:
  big_beta_mat <- do.call(rbind, lapply(big_beta, as.vector))
  
  return(list(big_beta, big_beta_mat))
}

########################################################################
# Getting the big cubes
# Create the cubes (if needed):
save_cubes <- function(addresses, coords.m, coords.n, big_beta, prior, start) {
  
  # Creating the scale and shape covariances:
  dist <- dist_mat_diff(coords.m, coords.m)
  save_big_cube(dist, big_beta, prior$nu_scale_hyper_discrete,
                start$sigsq_scale, start$tausq_scale, addresses[1])
  save_big_cube(dist, big_beta, prior$nu_shape_hyper_discrete,
                start$sigsq_shape, start$tausq_shape, addresses[2])
  
  # Creating the big nxm cubes:
  big_dist <- dist_mat_diff(coords.n, coords.m)
  save_big_cube(big_dist, big_beta, prior$nu_scale_hyper_discrete,
                start$sigsq_scale, start$tausq_scale, addresses[3])
  save_big_cube(big_dist, big_beta, prior$nu_shape_hyper_discrete,
                start$sigsq_shape, start$tausq_shape, addresses[4])

}

########################################################################
# Find the initial values for the algorithm:
find.init <- function(start, it.max, addresses, m, n, data,
                      bigcov1, bigcov2, cov1, cov2) {
  
  # Step a:
  # Read in the cubes created above:
  cube_list <- retrieve_cube(addresses)
  names(cube_list) <- c('cube_covar_scale',
                        'cube_covar_shape',
                        'cube_nxm_scale',
                        'cube_nxm_shape')
  
  # Step b:
  # Packages needed:
  library(mvtnorm)
  library(boot)
  library(fields)
  library(ismev)
  
  # Step c:
  # Getting the log-scale first:
  # Defining everything needed:
  alpha1 <- start$alpha_scale
  n_x_m1 <- cube_list$cube_nxm_scale[,,1]
  Sigma1 <- cube_list$cube_covar_scale[,,1]
  ldett1 <- log(det(Sigma1))
  data1 <- data
  shape <- rep(0, length(data))
  
  # I want to maximise post_scaleC
  post_scale_negll <- function(mlscale) {
    
    lscale <- bigcov1 %*% alpha1 + n_x_m1 %*% solve(Sigma1, mlscale - cov1 %*% alpha1)
    
    # Adding in this to allow the shape to be 0:
    count <- 0
    for(i in 1:n) {
      count <- count + sum(texmex::dgpd(data[[i]], exp(lscale)[i], shape[i], 0, log.d = TRUE))
    }
    
    b <- gp_mvnC2(mlscale, Sigma1, alpha1, cov1, ldett1)
    
    return(-(count + b))
  }
  
  # Now optimise it:
  answer <- nlminb(start = rep(0.1, m), 
                   objective = post_scale_negll,
                   control = list(iter.max = it.max))
  
  # Now to find the full vector from this:
  new.mlscale <- answer$par
  new.lscale <- bigcov1 %*% alpha1 + n_x_m1 %*% solve(Sigma1, new.mlscale - cov1 %*% alpha1)
  
  # Step d:
  # Now to find a corresponding one for the shape:
  ### Now to find values for the shape:
  alpha2 <- start$alpha_shape
  n_x_m2 <- cube_list$cube_nxm_shape[,,1]
  Sigma2 <- cube_list$cube_covar_shape[,,1]
  ldett2 <- log(det(Sigma2))
  
  # I want to maximise post_shapeC
  post_shape_negll <- function(mshape) {
    
    shape <- bigcov2 %*% alpha2 + n_x_m2 %*% solve(Sigma2, mshape - cov2 %*% alpha2)
    
    # Adding in this to allow the shape to be 0:
    count <- 0
    for(i in 1:n) {
      count <- count + sum(texmex::dgpd(data[[i]], exp(new.lscale)[i], shape[i], 0, log.d = TRUE))
    }
    
    b <- gp_mvnC2(mshape, Sigma2, alpha2, cov2, ldett2)
    
    return(-(count + b))
  }
  
  # Now optimise it:
  answer2 <- nlminb(start = rep(1, m), 
                    objective = post_shape_negll,
                    control = list(iter.max = it.max))
  
  # Now to find the full vector from this:
  new.mshape <- answer2$par
  new.shape <- bigcov2 %*% alpha2 + n_x_m2 %*% solve(Sigma2, new.mshape - cov2 %*% alpha2)
  
  # Step e:
  # Now to replace the parts of start:
  start$lscale <- new.mlscale
  start$shape <- new.mshape
  start$nlscale <- new.lscale
  start$nshape <- new.shape
  
  return(start)
}

########################################################################
# Functions to prepare to run the PP Binomial:
########################################################################

########################################################################
# Getting all potential betas
get_betas2 <- function(beta.vec, spatial.dim) {
  # Getting the number of unique elements in the symmetric
  # matrix, using n(n+1)/2
  beta.sym.dim <- (spatial.dim) * (spatial.dim + 1) / 2
  
  # Filling a list with repeated beta.vec (a roundabout way, but it works)
  beta <- vector("list", length = beta.sym.dim)
  for(i in 1:beta.sym.dim) {
    beta[[i]] <- beta.vec
  }
  
  # This needs to be the number of elements in the matrix, eventually -
  # 3 for a 2x2 matrix (since it's symmetric)
  # 6 for a 3x3 matrix (since it's symmetric) etc.
  big.beta <-  as.matrix(expand.grid(beta))
  
  # Now to reflect it, and generate all possible matrices using those values:
  big.list <- vector("list", nrow(big.beta))
  temp <- diag(spatial.dim)
  ind <- lower.tri(temp, diag=T)
  for (i in 1:length(big.list)) {
    
    temp[ind] <- big.beta[i,]
    temp[upper.tri(temp)] <- t(temp)[upper.tri(temp)]
    big.list[[i]] <- temp
    
  }
  
  # Using my own C++ function to test for positive definiteness:
  list1_beta <- lapply(big.list, pos_def_testC)
  
  # Only keeping those which are positive definite:
  big_beta <- big.list[unlist(list1_beta)]
  
  # Turning the list into a matrix - each row is the components of a single beta matrix:
  big_beta_mat <- do.call(rbind, lapply(big_beta, as.vector))
  
  return(list(big_beta, big_beta_mat))
}

########################################################################
# Getting the big cubes
# Create the cubes (if needed):
save_cubes2 <- function(addresses, coords.m, coords.n, big_beta, prior, start) {
  
  # Creating the zeta covariance:
  dist <- dist_mat_diff(coords.m, coords.m)
  save_big_cube(dist, big_beta, prior$nu_zeta_hyper_discrete,
                start$sigsq_zeta, start$tausq_zeta, addresses[1])
  
  # Creating the big nxm cube:
  big_dist <- dist_mat_diff(coords.n, coords.m)
  save_big_cube(big_dist, big_beta, prior$nu_zeta_hyper_discrete,
                start$sigsq_zeta, start$tausq_zeta, addresses[2])
  
  # Now to reset the tausq parameter after the cubes have been created:
  start$tausq_zeta <- 0.01
  
}

########################################################################
# Find the initial values for the algorithm:
find.init2 <- function(start, it.max=init.it.max, addresses, m, n, data, vec_counts, vec_total_obs, 
                      bigcov1, cov1) {
  
  # Step a:
  # Read in the cubes created above:
  cube_list <- retrieve_cube(addresses)
  names(cube_list) <- c('cube_covar_zeta',
                        'cube_nxm_zeta')
  
  # Step b:
  # Packages needed:
  library(mvtnorm)
  library(boot)
  library(fields)
  library(ismev)
  
  # Step c:
  # Getting zeta first:
  # Defining everything needed:
  alpha1 <- start$alpha_zeta
  n_x_m1 <- cube_list$cube_nxm_zeta[,,1]
  Sigma1 <- cube_list$cube_covar_zeta[,,1]
  ldett1 <- log(det(Sigma1))
  data1 <- data
  
  # I want to maximise post_scaleC
  post_zeta_negll <- function(mzeta) {
    
    zeta <- bigcov1 %*% alpha1 + n_x_m1 %*% solve(Sigma1, mzeta - cov1 %*% alpha1)
    
    # Adding in this to allow the shape to be 0:
    a <- sum(dbinom(x=vec_counts, size=vec_total_obs, prob=(exp(zeta)/(1 + exp(zeta))), log = TRUE))
    b <- gp_mvnC2(mzeta, Sigma1, alpha1, cov1, ldett1)
    
    return(-(a + b))
  }
  
  # Now optimise it:
  answer <- nlminb(start = rep(0.1, m), 
                   objective = post_zeta_negll,
                   control = list(iter.max = it.max))
  
  # Now to find the full vector from this:
  new.mzeta <- answer$par
  new.zeta <- bigcov1 %*% alpha1 + n_x_m1 %*% solve(Sigma1, new.mzeta - cov1 %*% alpha1)
  
  # Step e:
  # Now to replace the parts of start:
  start$zeta <- new.mzeta
  start$nzeta <- new.zeta
  
  return(start)
}
