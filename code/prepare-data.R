############################################################################################
########################### Functions to prep the data #####################################
############################################################################################

########################### Getting all potential betas
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

########################### Getting the big cubes
# Create the cubes (if needed):
save_cubes <- function(addresses) {
  
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
  
  # Now to reset the tausq parameters after the cubes have been created:
  start$tausq_scale <- 0.01; start$tausq_shape <- 0.01
  
}
