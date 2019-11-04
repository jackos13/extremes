////////////////////////////////////////////////////////////////////////////////////
/////// MCMC Predictive Processes script ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//////// Cpp Script - ////////////
//////// This script is for the binomial parameter, zeta -
//////// It will contain the code for the MCMC process to target
//////// posterior distributions of zeta, given the data (counts, and total number of obs. at each point)
// This script moves from a squared exponential covariance function to a Matern one
// It implements the Predictive Processes piece in full

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculating the distance list:
// This function takes two matrices and returns a list.
// n is the number of points
// s is the dimension of that point: 3 for x, y and z, 3 for x, y and t etc.
// 4 for x, y, z and t... All will work with this code.
// Each component returned is an n x m matrix showing the distance
// between its points and all the points in the other matrix
// e.g. element 1 in the list has its first row as the distance between
// point 1 in the first matrix and point 1 in the second,
// the second row as the distance between point 1 in the first and point 2 in the second etc.
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
List dist_mat_diff(mat x, mat y) {
  int n = x.n_rows;
  int m = y.n_rows;
  int s = x.n_cols;
  List out(n);
  mat temp(m, s);
  
  for (int i=0; i < n; ++i) {
    for (int j=0; j < m; ++j) {
      temp.row(j) = x.row(i) - y.row(j);
    }
    out[i] = temp;
  }
  
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Formula 2 - for the Matern covariance:
// This works in two parts - first, the matern function is evaluated on a single
// entry of the matrix (which in fact needs to be converted to a single number
// using the d^T * invbeta * d piece)
// Calculating the var/covar matrix using the distance, beta (which must be the same
// size as the dimension of the gridpoints), sigsq and tausq as inputs:
// This function calculates the big covar matrix Sigma needed for all the GPs -
// Sigma = sigsq * con * dist^nu * besselK_nu(dist) + tausq * Identity
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
double matern_cpp(double sigsq, double nu, vec d, mat invbeta, double tausq) {
  
  // If the points are the same, sigsq + tausq should be returned:
  if (sum(abs(d))==0) {
    return(sigsq + tausq);
  }
  
  // Otherwise:
  double con;
  double dist = pow(as_scalar(d.t()*invbeta*d), 0.5);
  con = pow(2,nu-1) * tgamma(nu);
  con = 1/con;
  return(sigsq * con * pow(dist, nu) * boost::math::cyl_bessel_k(nu, dist));
  //return(dist);
}

// Next piece: the function which works on a list, in order to fill a large matrix:
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
mat get_mat_sigma_matern_called(List dist, mat beta, double sigsq, double tausq, double nu) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the dimension of the coordinates:
  int k = beta.n_cols;
  
  // Declaring the variables used to fill the matrix sig:
  mat sig = zeros(n, n);
  mat tmpi(n, k);
  mat invb = inv(beta);
  
  // Looping through the list taking each matrix in turn from dist[[i]]:
  for (int i=0; i < n; ++i) {
    tmpi = as<mat>(dist[i]);
    
    // Running over j <= i since the matrix is symmetric, in order to calculate
    // the covariance:
    for (int j=0; j <= i; ++j) {
      sig(i, j) = matern_cpp(sigsq, nu, tmpi.row(j).t(), invb, tausq);
      sig(j, i) = sig(i, j);
    }
  }
  return(sig);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Formula 3 - for the Matern covariance:
// This function returns the covariance matrix - the Matern is built into it:
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
mat get_mat_sigma_matern(List dist, mat beta, double sigsq, double tausq, double nu) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the dimension of the coordinates:
  int k = beta.n_cols;
  
  // Declaring the variables used to fill the matrix sig:
  mat sig = zeros(n, n);
  mat tmpi(n, k);
  mat invb = inv(beta);
  
  // The pieces needed to evaluate the Matern covariance function:
  vec d(k);
  double con;
  con = pow(2,nu-1) * tgamma(nu);
  con = sigsq/con;
  double dist1;
  
  // Looping through the list taking each matrix in turn from dist[[i]]:
  for (int i=0; i < n; ++i) {
    tmpi = as<mat>(dist[i]);
    
    // Running over j <= i since the matrix is symmetric, in order to calculate
    // the covariance:
    for (int j=0; j <= i; ++j) {
      d = tmpi.row(j).t();
      
      // If the points are the same, sigsq + tausq should be returned:
      if (sum(abs(d))==0) {
        sig(i, j) = sigsq + tausq;
      } else {
        // Evaluating the Matern:
        dist1 = pow(as_scalar(d.t()*invb*d), 0.5);
        sig(i,j) =  con * pow(dist1, nu) * boost::math::cyl_bessel_k(nu, dist1);
        sig(j, i) = sig(i, j);
      }
    }
  }
  return(sig);
}

///////////////////////////////////////////////////////////////////////////////////
// This is the function to calculate the n_x_m matrix needed in order to
// calculate the n-dim vector from the m-dim vector and other
// parameters - essentially it's the Predictive Processes component:
///////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
mat get_mat_m_to_n(List dist, double sigsq, double tausq, mat beta, double nu) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the dimension of the coordinates:
  int k = beta.n_cols;
  
  // Getting the size of the reduced grid m:
  int m = as<mat>(dist[1]).n_rows;
  
  // Declaring the variables used to fill the matrix sig:
  mat n_x_m = zeros(n, m);
  mat tmpi(m, k);
  mat invb = inv(beta);
  
  // Looping through the list taking each matrix in turn from dist[[i]]:
  for (int i=0; i < n; ++i) {
    tmpi = as<mat>(dist[i]);
    
    // Running over j <= i since the matrix is symmetric, in order to calculate
    // the covariance:
    for (int j=0; j < m; ++j) {
      n_x_m(i, j) = matern_cpp(sigsq, nu, tmpi.row(j).t(), invb, tausq);
    }
  }
  return(n_x_m);
}


///////////////////////////////////////////////////////////////////////////////////
// This is the second function to calculate the n_x_m matrix needed in order to
// calculate the n-dim vector from the m-dim vector and other
// parameters - essentially it's the Predictive Processes component
// Here, the Matern is built into the function:
///////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
mat get_mat_m_to_n_matern(List dist, double sigsq, double tausq, mat beta, double nu) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the dimension of the coordinates:
  int k = beta.n_cols;
  
  // Getting the size of the reduced grid m:
  int m = as<mat>(dist[1]).n_rows;
  
  // Declaring the variables used to fill the matrix sig:
  mat n_x_m = zeros(n, m);
  mat tmpi(m, k);
  mat invb = inv(beta);
  
  // Declaring the variables needed for the Matern component:
  vec d(k);
  double con;
  con = pow(2,nu-1) * tgamma(nu);
  con = sigsq/con;
  
  double dist1;
  
  // Looping through the list taking each matrix in turn from dist[[i]]:
  for (int i=0; i < n; ++i) {
    tmpi = as<mat>(dist[i]);
    
    for (int j=0; j < m; ++j) {
      d = tmpi.row(j).t();
      
      // If the points are the same, sigsq + tausq should be returned:
      if (sum(abs(d))==0) {
        n_x_m(i, j) = sigsq + tausq;
      } else {
        dist1 = pow(as_scalar(d.t()*invb*d), 0.5);
        n_x_m(i,j) = con * pow(dist1, nu) * boost::math::cyl_bessel_k(nu, dist1);
      }
    }
  }
  return(n_x_m);
}

///////////////////////////////////////////////////////////////////////////////////
// This is the third function to calculate the n_x_m matrix needed in order to
// calculate the n-dim vector from the m-dim vector and other
// parameters - essentially it's the Predictive Processes component
// Here, the exponential covariance is built into the function:
///////////////////////////////////////////////////////////////////////////////////
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

// [[Rcpp::export]]
mat get_mat_m_to_n_exp(List dist, double sigsq, double tausq, mat beta, double nu) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the dimension of the coordinates:
  int k = beta.n_cols;
  
  // Getting the size of the reduced grid m:
  int m = as<mat>(dist[1]).n_rows;
  
  // Declaring the variables used to fill the matrix sig:
  mat n_x_m = zeros(n, m);
  mat tmpi(m, k);
  mat invb = inv(beta);
  
  // Declaring the variables needed for the covariance piece:
  vec d(k);
  
  // Looping through the list taking each matrix in turn from dist[[i]]:
  for (int i=0; i < n; ++i) {
    
    tmpi = as<mat>(dist[i]);
    
    for (int j=0; j < m; ++j) {
      d = tmpi.row(j).t();
      
      // If the points are the same, sigsq + tausq should be returned:
      if (sum(abs(d))==0) {
        n_x_m(i, j) = sigsq + tausq;
      } else {
        n_x_m(i,j) = sigsq * exp( - as_scalar( tmpi.row(j) * invb * tmpi.row(j).t() ) );
      }
    }
  }
  return(n_x_m);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// In this function, I create the massive array (cube) over both all values of beta and all values of nu:
// It returns the cube into the function, which is then accessed by .slice() as appropriate:
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
cube get_big_cube(List dist, List big_beta, vec nu, double sigsq, double tausq) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the number of possible betas:
  int b = big_beta.size();
  
  // Getting the number of possible nu's:
  int nu1 = nu.size();
  
  // Getting the size of the reduced grid m:
  int m = as<mat>(dist[1]).n_rows;
  
  // Creating the large array needed:
  cube out = zeros(n, m, b * nu1);
  
  // Getting the dimension of the coordinates:
  int k = as<mat>(big_beta[1]).n_cols;
  
  // Creating the vectors etc. needed to use in the loops
  mat beta(k, k);
  mat invb(k, k);
  mat n_x_m = zeros(n, m);
  mat tmpi(m, k);
  double tempnu;
  
  for (int p=0; p < nu1; ++p) {
    
    tempnu = nu[p];
    
    for (int l=0; l < b; ++l) {
      beta = as<mat>(big_beta[l]);
      invb = inv(beta);
      
      // Looping through the list taking each matrix in turn from dist[[i]]:
      for (int i=0; i < n; ++i) {
        
        tmpi = as<mat>(dist[i]);
        
        // Running over all entries in this matrix from a particular n point to all
        // the m points:
        for (int j=0; j < m; ++j) {
          n_x_m(i, j) = matern_cpp(sigsq, tempnu, tmpi.row(j).t(), invb, tausq);
        }
      }
      out.slice(p*b + l) = n_x_m;
    }
  }
  return(out);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ################################################################################
// ############################# Data layer #######################################
// ################################################################################
// The full Binomial, works on a list:
// This function, on the log scale, takes in a vector of counts for each gridpoint, and a vector of total
// obs. corresponding to those (allowing for the fact that there may be different length time series
// at different grid points) and a vector of probabilities at each point too.
// Note that the factorial component is not needed, since it appears identically in the
// denominator and the numerator, and thus will cancel each time  
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double dbinomC(vec counts, vec total_obs, vec zeta) {
  
  // Converting back into a vector of probs. between 0 and 1
  vec prob = exp(zeta) / (1 + exp(zeta));
  
  // This is the (1 - p_i) piece:
  vec inv_prob = 1 - prob;
  
  // This is the k_i * log(p_i) piece:
  vec a = counts % log(prob);
  
  // This is the (n_i - k_i) * log(1 - p_i) piece:
  vec b = (total_obs - counts) % log(inv_prob);
  
  // Finally to sum the two vectors, and sum all entries:
  double out = sum(a + b);
  
  return out;
}


// ################################################################################
// ############################# The latent process layer #########################
// ################################################################################
// The MVN in log form, for proportions: mvnC
// This gives the same value os dmvnorm(x, mu, sigma, log=T)
// when the (-k/2 * log(2 * pi)) factor is included at the end. This will be above and below, and will cancel
#define _USE_MATH_DEFINES
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// Now for the MVN:
// [[Rcpp::export]]
double mvnC(vec x, vec mu, mat sigma) {
  
  double x1;
  double sign = 1;
  log_det(x1, sign, sigma); // This gets the log of the determinant of the covar matrix sigma.
  
  vec p = x - mu;
  double c = arma::as_scalar(p.t() * solve(sigma, p));
  double d = 0.5 * c;
  return (-0.5) * x1 - d;
}


// ################################################################################
// The MVN in log form and using the precision, for proportions: mvnC2 
// Here, the precision is used - where it's calculated as the inverse of sigma directly withing
// the MCMC code. This formula agrees with the one above (luckily!)
#define _USE_MATH_DEFINES
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// Now for the MVN:
// [[Rcpp::export]]
double mvnC2(vec x, vec mu, mat sigma, double dett) {
  
  vec p = x - mu;
  double c = arma::as_scalar(p.t() * solve(sigma, p));
  double d = 0.5 * c;
  
  return (-0.5) * dett - d;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Using mvnC for the GP distribution:
// This is the final component needed for layer 2 - the mvn applied from the previous formulae, but using
// the covariates too in order to calculate mu:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double gp_mvnC(vec x, mat sigma, vec alpha, mat covar, double dett) {
  
  vec mean = covar * alpha;
  
  double out = mvnC2(x, mean, sigma, dett);
  //double out = mvnC(x, mean, sigma);
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Using mvnC for the GP distribution:
// This is the final component needed for layer 2 - the mvn applied from the previous formulae, but using
// the covariates too in order to calculate mu:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double gp_mvnC2(vec x, mat sigma, vec alpha, mat covar, double ldett) {
  
  return - (0.5 * ldett) - (0.5 * arma::as_scalar((x - covar*alpha).t() * solve(sigma, x - covar*alpha)));
  
}

// #############################################################################################################
// ############################# Priors on hyperparameters (layer 3) ###########################################
// #############################################################################################################
// Evaluating the normal density component-wise:
// This takes a vector x, a mean vector mu, and a standard deviation vector s
// All must be the same length
// It then calculates pointwise densities for each element of the vector,
// gets the log, then sums the result to give a single number.
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double dnormvecC(vec x, vec m, vec s) {
  int n = x.size();
  
  vec out(n);
  
  double c = 1/sqrt(2*M_PI);
  
  vec f = (x - m) / s;
  vec ff = f % f;
  out = c* (1/s) % exp ( - 0.5 * ff);
  vec ret = log(out);
  
  return sum(ret);
}


// ################################################################################
// Evaluating the normal density for scalars:
// This straightforward function evaluates the normal density for a single scalar, given a mean
// and an sd. It then takes the log of this, and returns the value.
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double dnormscalarC(double x, double m, double s) {
  
  double out;
  
  double c = 1/sqrt(2*M_PI);
  
  double f = pow((x - m) / s, 2);
  out = c * (1/s) * exp ( - 0.5 * f);
  
  return log(out);
}


// ################################################################################
// A function to test for positive definitness:
// This function gets the det(big matrix) and saves it. It then sheds the final row and column and repeats the process
// At the end, a check is done that all entries are positive, as this is a sufficient requirement for a
// matrix to be positive definite. 
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
bool pos_def_testC(mat m) {
  int n = m.n_rows;
  vec v(n);
  
  for (int j=0; j<n; ++j) {
    v(j) = det(m); //round(det(m)); This was causing trouble at some point....
    m.shed_row(n-j-1);
    m.shed_col(n-j-1);
  }
  
  bool test = all(v>0);  
  return test;
}


// ###############################################################################################
// ############################################ Posteriors #######################################
// ###############################################################################################

// In this section, the necessary components for each posterior density are calculated
// These are in log form (or more precisely, the functions which they call are)
// Then within the MCMC code, the relevant fractions are evaluated as exp ( log (top) - log (bottom) )
// in order to get the ratio needed

// ##########################################################################################
// Evaluating the full posterior conditional of zeta:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_zetaC(vec counts, vec total_obs, vec nzeta, vec mzeta, vec alpha, mat covar,
                  double dett_zeta, mat sigma) {
  
  double a = dbinomC(counts, total_obs, nzeta);
  double b = gp_mvnC(mzeta, sigma, alpha, covar, dett_zeta);
  return a + b;
}


// ################################################################################
// Hyper-parameter inference:
// For these, we only deal with the m-dimensional reduced case now...:

// Evaluating the full posterior conditional of the alpha_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_alpha_zetaC(vec alpha_zeta, vec mzeta, mat covar, mat sigma,
                        vec alpha_zeta_hyper_mean, mat alpha_zeta_hyper_sd, double dett_zeta) {
  
  double a = gp_mvnC2(mzeta, sigma, alpha_zeta, covar, dett_zeta); // Layer2
  double b = mvnC(alpha_zeta, alpha_zeta_hyper_mean, alpha_zeta_hyper_sd) ; //Layer3
  return a + b;
}  

// ################################################################################
// Evaluating the full posterior conditional of the beta_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_beta_zetaC(mat sigma, vec mzeta,
                       vec alpha_zeta, mat covar, double dett_zeta) {
  
  double a = gp_mvnC2(mzeta, sigma, alpha_zeta, covar, dett_zeta); // Layer2
  //double b =  ; //Layer3 - blank for now
  return a;
}  

// ################################################################################
// Evaluating the full posterior conditional of the sigsq_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_sigsq_zetaC(mat sigma, vec mzeta, vec alpha,
                        mat covar, double sigsq, double dett_zeta,
                        double sigsq_zeta_mean, double sigsq_zeta_sd) {
  
  double a = gp_mvnC2(mzeta, sigma, alpha, covar, dett_zeta); // Layer2
  double b =  dnormscalarC(log(sigsq), sigsq_zeta_mean, sigsq_zeta_sd); //Layer3
  
  return a + b;
}

// ################################################################################
// Evaluating the full posterior conditional of the tausq_zeta parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_tausq_zetaC(mat sigma, vec mzeta, vec alpha,
                        mat covar, double tausq, double dett_zeta,
                        double tausq_zeta_mean, double tausq_zeta_sd) {
  
  double a = gp_mvnC2(mzeta, sigma, alpha, covar, dett_zeta); // Layer2
  double b = dnormscalarC(log(tausq), tausq_zeta_mean, tausq_zeta_sd); //Layer3
  
  return a + b;
}

// ################################################################################
// Evaluating the full posterior conditional of the nu_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_nu_zetaC(mat sigma, vec mzeta, vec alpha,
                     mat covar, double nu, double dett_zeta,
                     double nu_zeta_mean, double nu_zeta_sd) {
  
  double a = gp_mvnC2(mzeta, sigma, alpha, covar, dett_zeta); // Layer2
  //double b = dnormscalarC(log(nu), nu_zeta_mean, nu_zeta_sd); //Layer3
  
  return a; // + b;
}

// // ################################################################################
// // ############################ GP Simplifications ################################
// // ################################################################################
// // Alternatives to the GP here, for comparison: Assuming that zeta is a constant value:
// 
// // Evaluating the full posterior conditional of zeta, where it is assumed constant:
// // (Can include a prior on this too, by uncommenting below)
// #include <math.h>
// #include <RcppArmadillo.h>
// // [[Rcpp::depends(RcppArmadillo)]]
// using namespace std;
// using namespace arma;
// 
// #include <Rcpp.h>
// using namespace Rcpp;
// // [[Rcpp::export]]
// double post_zetaC2(vec counts, vec total_obs, vec zeta,
//                    int l, double prior_mean, double prior_sd) {
//   
//   double a = dbinomC(counts[l], total_obs[l], zeta[l]);
//   double b = dnormscalarC(zeta[l], prior_mean, prior_sd);
//   
//   return a + b;
// }


// ################################################################################
// ############################# Random updates ###################################
// ################################################################################
// This function takes a mean and an SD and returns a random value drawn from that Normal:

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double rnormscalarC(double m, double s) {
  vec x = rnorm(1) * s + m;
  return as_scalar(x);
}

// ################################################################################
// ############################# Acceptance Rates #################################
// ################################################################################

// This code is to keep track of the acceptance rates within the MCMC process
// It compares vectors entry by entry

#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp ;

// [[Rcpp::export]]
LogicalVector keep_acc(vec x, vec y) {
  
  int n = x.size();
  
  LogicalVector r(n);
  for( int i=0; i<n; i++){
    r[i] = (x[i] == y[i]);
  }
  
  return(r);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Cube code here:
////////////////////////////////////////////////////////////////////////////////////////////////////
// In this function, I create the massive cube over both all values of beta and all values of nu:
// I also save it, for loading later:
// 1 is for the scale, 2 is for the shape, 3 is for the zeta parameter,
// to allow nu and beta to be parameter specific:
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
int save_big_cube(List dist, List big_beta, vec nu, double sigsq, double tausq, string address) {
  
  // Getting the length of the list, which is the number of points
  // on the grid:
  int n = dist.size();
  
  // Getting the number of possible betas:
  int b = big_beta.size();
  
  // Getting the number of possible nu's:
  int nu1 = nu.size();
  
  // Getting the size of the reduced grid m:
  int m = as<mat>(dist[1]).n_rows;
  
  // Creating the large array needed:
  cube out = zeros(n, m, b * nu1);
  
  // Getting the dimension of the coordinates:
  int k = as<mat>(big_beta[1]).n_cols;
  
  // Creating the vectors etc. needed to use in the loops
  mat beta(k, k);
  mat invb(k, k);
  mat n_x_m = zeros(n, m);
  mat tmpi(m, k);
  double tempnu;
  
  for (int p=0; p < nu1; ++p) {
    
    tempnu = nu[p];
    
    for (int l=0; l < b; ++l) {
      beta = as<mat>(big_beta[l]);
      invb = inv(beta);
      
      // Looping through the list taking each matrix in turn from dist[[i]]:
      for (int i=0; i < n; ++i) {
        
        tmpi = as<mat>(dist[i]);
        
        // Running over all entries in this matrix from a particular n point to all
        // the m points:
        for (int j=0; j < m; ++j) {
          n_x_m(i, j) = matern_cpp(sigsq, tempnu, tmpi.row(j).t(), invb, tausq);
        }
      }
      out.slice(p*b + l) = n_x_m;
    }
  }
  
  out.save(address);
  
  return(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Code to load the cube into R here:
////////////////////////////////////////////////////////////////////////////////////////////////////
// In this function, I simply return the cube to R. It seems a roundabout way,
// but the easiest way
// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;
using boost::math::cyl_bessel_k;

// [[Rcpp::export]]
List retrieve_cube(StringVector address1) {
  
  // The List to return:
  List out(2);
  
  // Creating the 2 variables:
  cube cube_covar_zeta; cube cube_nxm_zeta;
  
  // Loading the two with the given names:
  string str1; string str2;
  str1 = address1[0]; str2 = address1[1];
  cube_covar_zeta.load(str1);
  cube_nxm_zeta.load(str2);
  
  // Writing them to the list:
  out[0] = cube_covar_zeta;
  out[1] = cube_nxm_zeta;
  
  return(out);
}


// ################################################################################
// *******************************************************************************
// ################################################################################
// ############################# MCMC process #####################################
// ################################################################################
// *******************************************************************************
// ################################################################################

// Ok, here's the big code that runs the MCMC process:

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;
#include <limits>

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List mcmc_binom_C(vec counts,
                  List big_beta,
                  mat big_beta_mat,
                  vec total_obs,
                  List start,
                  int iterations,
                  mat covar1,
                  mat bigcovar1,
                  mat distance_vectors_n,
                  mat distance_vectors_m,
                  int dimension,
                  List step,
                  List prior,
                  int burnin,
                  int nth,
                  int n_dim,
                  int m_dim,
                  vec ind_m) {
  
  // start is the same size as a single iteration of the process:
  List out(4 * start.size());
  
  // Creating the distance list, unchanged throughout
  List dist = dist_mat_diff(distance_vectors_m, distance_vectors_m);
  
  // Creating the distance list between the n grid and those points on the m grid, unchanged throughout:
  List dist_n_to_m = dist_mat_diff(distance_vectors_n, distance_vectors_m);
  
  //// Forming the matrices for zeta, to be filled during the MCMC step:
  // The zeta parameter:
  vec start0 = start[0];
  mat mzeta;
  mzeta.insert_rows(0, start0.t());
  
  // The alpha_zeta coefficients:
  vec start1 = start[1];
  mat alpha_zeta;
  alpha_zeta.insert_rows(0, start1.t());
  
  // The beta_zeta matrix:
  vec start2 = start[2];
  mat beta_zeta;
  beta_zeta.insert_rows(0, start2.t());
  
  // The sigsq_zeta coefficient:
  vec start5 = start[3];
  double s5 = as_scalar(start5);
  NumericVector sigsq_zeta;
  sigsq_zeta.push_back(s5);
  
  // The tausq_zeta coefficient:
  vec start6 = start[4];
  double s6 = as_scalar(start6);
  NumericVector tausq_zeta;
  tausq_zeta.push_back(s6);
  
  // Now to extract the longer, larger zeta (n-dim):
  vec start7 = start[5];
  mat nzeta;
  nzeta.insert_rows(0, start7.t());
  
  // Now, as it was added later, the nu parameter:
  vec start14 = start[6];
  double s14 = as_scalar(start14);
  NumericVector nu_zeta;
  nu_zeta.push_back(s14);
  
  // Setting the starting values for zeta and the zeta hyper-parameters:
  vec prop_mzeta = mzeta.row(0).t();
  vec prop_alpha_zeta = alpha_zeta.row(0).t();
  mat prop_beta_zeta(dimension, dimension);
  prop_beta_zeta = reshape(big_beta_mat.row(0), dimension, dimension);
  //  for (int i = 0; i < dimension; ++i) {
  //    prop_beta_zeta.row(i) = beta_zeta.submat(0, i*dimension, 0, (i+1)*dimension-1);
  //  } // A complicated way of getting at the elements needed
  double prop_sigsq_zeta = as_scalar(sigsq_zeta(0));
  double prop_tausq_zeta = as_scalar(tausq_zeta(0));
  double prop_nu_zeta    = as_scalar(nu_zeta(0));
  vec prop_nzeta = nzeta.row(0).t();
  
  // Now creating vectors to keep track of the acceptance rates
  int n1 = prop_mzeta.size(); int n2 = prop_alpha_zeta.size();
  int n3 = prop_beta_zeta.size();
  Rcpp::NumericVector tempr(n1); Rcpp::NumericVector temps(n2);
  Rcpp::NumericVector tempt(n3);
  
  // These are to be filled in:
  Rcpp::NumericVector acc_zeta(n1);
  Rcpp::NumericVector acc_alpha_zeta(n2);
  //vec acc_beta_zeta(n3);
  double acc_beta_zeta  = 0;
  double acc_sigsq_zeta = 0;
  double acc_tausq_zeta = 0;
  double acc_nu_zeta    = 0;
  bool b1 = 0; bool b2 = 0; bool b6 = 0; bool b10 = 0;
  //Rcpp::NumericMatrix temp_acc_beta_zeta(sqrt(n3), sqrt(n3));
  
  ////////////////////////////////////////////////////////////////////////////////////////
  // Picking out the steps for the zeta updates:
  // (These are essentially the standard deviations, inputted at the beginning. Adjusting these
  // will affect the acceptance rate. Too big, and not enough values are accepted.
  // Too small, and too many values are accepted, leading to slow mixing (exploration) of
  // the posterior space)
  vec step_zeta = step[0]; vec step_alpha_zeta = step[1]; vec step_sigsq_zeta1 = step[2];
  double step_sigsq_zeta = as_scalar(step_sigsq_zeta1); vec step_tausq_zeta1 = step[3];
  double step_tausq_zeta = as_scalar(step_tausq_zeta1); vec step_nu_zeta1 = step[4];
  
  // Picking out the zeta hyper parameters:
  // These are the priors on the layer 3 parameters, and the end of our hierarchy:
  vec alpha_zeta_hyper_mean = prior[0]; mat alpha_zeta_hyper_sd = prior[1];
  vec beta_zeta_hyper_prior = prior[2];
  // Now for sigsq, tausq and nu:
  vec sigsq_zeta_mean1  = prior[3]; vec sigsq_zeta_sd1   = prior[4];
  vec tausq_zeta_mean1  = prior[5]; vec tausq_zeta_sd1   = prior[6];
  vec nu_zeta_mean1     = prior[7]; vec nu_zeta_sd1      = prior[8];
  
  double sigsq_zeta_mean  = as_scalar(sigsq_zeta_mean1);
  double sigsq_zeta_sd    = as_scalar(sigsq_zeta_sd1);
  double tausq_zeta_mean  = as_scalar(tausq_zeta_mean1);
  double tausq_zeta_sd    = as_scalar(tausq_zeta_sd1);
  double nu_zeta_mean     = as_scalar(nu_zeta_mean1);
  double nu_zeta_sd       = as_scalar(nu_zeta_sd1);
  
  // Or in the option of a discrete update for nu - zeta:
  vec nu_zeta_hyper_discrete = prior[9];
  int len_nu_zeta = nu_zeta_hyper_discrete.size();
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // The n x m x (beta x nu) cubes to be created
  // and to be accessed to update the predicted process of the vector n
  // based on the current proposed values of the vector
  cube cube_covar_zeta; cube cube_nxm_zeta;
  
  cube_covar_zeta.load("cubes/binom_cube1.cube");
  cube_nxm_zeta.load("cubes/binom_big_cube1.cube");
  
  // Selecting the first of these to input into the MCMC process - this should be fixed later to select
  // the one that corresponds to the starting values of the parameters:
  // Calculating Sigma_zeta for input to the first step of the MCMC process:
  mat Sigma_zeta = cube_covar_zeta.slice(0);
  mat Sigma_new_zeta = Sigma_zeta;
  
  ////////////////////////////////////////////////////////////////////////////////////////
  // doubles needed for the log det. step, all of which are calculated in vectors below:
  double dett_zeta; double sign = 1;
  double dettnew_zeta = dett_zeta;
  
  // Defining the items to be used in the zeta update:
  // These will essentially be the last items in the chain - vectors and scalars
  // that are rewritten at
  // each line, and copied into the big matrix if they are accepted:
  vec newprop_mzeta;
  vec newprop_alpha_zeta;
  mat newprop_beta_zeta(dimension, dimension);
  double temp_sigsq; double newprop_sigsq_zeta;
  double temp_tausq; double newprop_tausq_zeta;
  double newprop_nu_zeta;
  vec newprop_nzeta=prop_nzeta;
  
  // Doubles, vecs and ints to be used within the loops:
  double rndm;
  double probab=0;
  vec rnd;
  int m;
  
  // The n x m matrices to be created and rewritten
  // and to be used to update the predicted process of the vector n
  // based on the current proposed values of the vector m
  // Need to change from selecting 0 to the particular entry needed:
  mat n_x_m_zeta = cube_nxm_zeta.slice(0);
  
  // Declaring new ints:
  int numbeta = big_beta_mat.n_rows;
  int c_beta_zeta = 0; int c_nu_zeta = 0;
  int c_beta_zeta_new = 0; int c_nu_zeta_new = 0;
  
  // Integer for the index of vectors:
  int ind1 = 0;
  ind1 = (numbeta) * c_nu_zeta + c_beta_zeta;
  
  // Saving these at each iteration:
  NumericVector cube_zeta_counter;
  cube_zeta_counter.push_back(ind1);
  
  // Fix:
  prop_tausq_zeta = 0.01;
  // It's because to make the cube easier, I set tausq to be 0 initially -
  // but once selected, I want the first one to be invertible, so it's better to add a small tausq
  
  // Creating the vector for random indices:
  vec rand_ind; vec prob1;
  rand_ind.set_size(m_dim);
  prob1.fill(0.5);
  
  ////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// MCMC piece ///////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  // Now for the MCMC step, now that everything has been set up:
  for(int i = 0; i < iterations; ++i) {
    
    // At the i^th step, all will be accepted or rejected with this probability:
    rnd = runif(1, 0, 1);
    rndm = as_scalar(rnd);
    
    // Now to update the values of nzeta depending on the current value
    // of mzeta, and the matrices etc.:
    
    // ///////////////////////////////////////////////////////////////////////////////////
    // ///////////////////////// Zeta updates now: //////////////////////////////////////
    // //////////////////////////////////////////////////////////////////////////////////
    // // Updating the zeta values:
    // ////////////////////////////////////////////////////////////////////////////////////
    // // Setting things up:
    // ///////////////////////////////////////////////////////////////////////////////////
    newprop_mzeta = prop_mzeta;
    
    // Getting the correct Sigma_zeta for the zeta update:
    Sigma_zeta = prop_sigsq_zeta * cube_covar_zeta.slice(ind1);
    Sigma_zeta.replace(prop_sigsq_zeta, prop_sigsq_zeta + prop_tausq_zeta);
    
    // Getting the correct n x m matrix once for the zeta update:
    n_x_m_zeta = prop_sigsq_zeta * cube_nxm_zeta.slice(ind1);
    n_x_m_zeta.replace(prop_sigsq_zeta, prop_sigsq_zeta + prop_tausq_zeta);
    
    // Getting the relevant determinant:
    log_det(dett_zeta, sign, Sigma_zeta);
    // ///////////////////////////////////////////////////////////////////////////////////
    // // Now getting the value on the larger grid using the kriging equations:
    prop_nzeta = bigcovar1 * prop_alpha_zeta + n_x_m_zeta * solve(Sigma_zeta,
                                                                  prop_mzeta - covar1 * prop_alpha_zeta);
    
    newprop_nzeta = prop_nzeta;
    
    // Getting the necessary indices to update in a random order:
    rand_ind = Rcpp::RcppArmadillo::sample(ind_m, m_dim, 0, prob1);
    
    for (unsigned int l = 0; l < prop_mzeta.size(); ++l) {
      
      // Updating each of the m entries one by one:
      newprop_mzeta[rand_ind[l]] = rnormscalarC(prop_mzeta[rand_ind[l]], step_zeta(0));
      
      // Now to update the values of nzeta depending on the current value
      // of newprop_mzeta, and the matrices etc. (the PP part):
      newprop_nzeta = bigcovar1 * prop_alpha_zeta + n_x_m_zeta * solve(Sigma_zeta,
                                                                       newprop_mzeta - covar1 * prop_alpha_zeta);
      
      // Calculating the MCMC fraction:
      probab = exp( post_zetaC(counts, total_obs, newprop_nzeta, newprop_mzeta,
                               prop_alpha_zeta, covar1, dett_zeta, Sigma_zeta) -
                                 post_zetaC(counts, total_obs, prop_nzeta, prop_mzeta,
                                            prop_alpha_zeta, covar1, dett_zeta, Sigma_zeta) );
      
      if (rndm < probab) {
        prop_nzeta = newprop_nzeta;
        prop_mzeta = newprop_mzeta;
        tempr[rand_ind[l]] = 1;
      } else {
        newprop_nzeta = prop_nzeta;
        newprop_mzeta = prop_mzeta;
        tempr[rand_ind[l]] = 0;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Updating the alpha_zeta parameter:
    ////////////////////////////////////////////////////////////////////////////////////
    newprop_alpha_zeta = prop_alpha_zeta;
    for (unsigned int l = 0; l < prop_alpha_zeta.size(); ++l) {
      
      newprop_alpha_zeta[l] = rnormscalarC(prop_alpha_zeta[l], step_alpha_zeta[l]);
      
      // Now to update the values of nzeta depending on the current value
      // of newprop_mzeta, and the new alpha hyper-parameters and the matrices etc. (the PP part):
      newprop_nzeta = bigcovar1 * newprop_alpha_zeta + n_x_m_zeta *
        solve(Sigma_zeta, prop_mzeta - covar1 * newprop_alpha_zeta);
      
      probab = exp (post_alpha_zetaC(newprop_alpha_zeta, prop_mzeta, covar1, Sigma_zeta,
                                     alpha_zeta_hyper_mean, alpha_zeta_hyper_sd, dett_zeta) -
                                       post_alpha_zetaC(prop_alpha_zeta, prop_mzeta, covar1, Sigma_zeta,
                                                        alpha_zeta_hyper_mean, alpha_zeta_hyper_sd, dett_zeta) ) ;
      
      //  Acceptance step:
      if (rndm < probab) {
        prop_alpha_zeta = newprop_alpha_zeta;
        prop_nzeta = newprop_nzeta;
        temps[l] = 1;
      } else {
        newprop_alpha_zeta = prop_alpha_zeta;
        newprop_nzeta = prop_nzeta;
        temps[l] = 0;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Updating the beta_zeta parameter:
    ////////////////////////////////////////////////////////////////////////////////////
    // A random number between 0 and (number of beta matrices - 1):
    c_beta_zeta_new = rand() % numbeta;
    
    // Getting the new index for the cubes, based on this random number:
    ind1 = (numbeta) * c_nu_zeta + c_beta_zeta_new;
    
    // Selecting the entries of beta:
    // Reshaping the vector into a matrix:
    newprop_beta_zeta = reshape(big_beta_mat.row(c_beta_zeta_new), dimension, dimension);
    
    // Selecting the relevant covariance matrix:
    Sigma_new_zeta = prop_sigsq_zeta * cube_covar_zeta.slice(ind1);
    Sigma_new_zeta.replace(prop_sigsq_zeta, prop_sigsq_zeta + prop_tausq_zeta);
    // Getting the relevant determinant:
    log_det(dettnew_zeta, sign, Sigma_new_zeta);
    
    // Calculating the ratio with the new Sigma and beta vs. the old ones:
    probab = exp (post_beta_zetaC(Sigma_new_zeta, prop_mzeta,
                                  prop_alpha_zeta, covar1, dettnew_zeta) -
                                    post_beta_zetaC(Sigma_zeta, prop_mzeta,
                                                    prop_alpha_zeta, covar1, dett_zeta) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_beta_zeta = newprop_beta_zeta;
      Sigma_zeta = Sigma_new_zeta;
      dett_zeta = dettnew_zeta;
      c_beta_zeta = c_beta_zeta_new;
      b6 = 1;
    } else {
      newprop_beta_zeta = prop_beta_zeta;
      Sigma_new_zeta = Sigma_zeta;
      dettnew_zeta = dett_zeta;
      c_beta_zeta_new = c_beta_zeta;
      b6 = 0;
    }
    
    // Recalculating the index:
    ind1 = (numbeta) * c_nu_zeta + c_beta_zeta;
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Updating the sigsq_zeta parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    temp_sigsq = rnormscalarC(log(prop_sigsq_zeta), step_sigsq_zeta);
    newprop_sigsq_zeta = exp(temp_sigsq);
    
    // Updating Sigma - Selecting the relevant covariance matrix:
    Sigma_new_zeta = newprop_sigsq_zeta * cube_covar_zeta.slice(ind1);
    Sigma_new_zeta.replace(newprop_sigsq_zeta, newprop_sigsq_zeta + prop_tausq_zeta);
    // Getting the relevant determinant:
    log_det(dettnew_zeta, sign, Sigma_new_zeta);
    
    // Calculating probab:
    probab = exp( post_sigsq_zetaC(Sigma_new_zeta, prop_mzeta, prop_alpha_zeta,
                                   covar1, newprop_sigsq_zeta, dettnew_zeta,
                                   sigsq_zeta_mean, sigsq_zeta_sd) -
                                     post_sigsq_zetaC(Sigma_zeta, prop_mzeta, prop_alpha_zeta,
                                                      covar1, prop_sigsq_zeta, dett_zeta,
                                                      sigsq_zeta_mean, sigsq_zeta_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_sigsq_zeta = newprop_sigsq_zeta;
      Sigma_zeta = Sigma_new_zeta;
      dett_zeta = dettnew_zeta;
      b1 = 1;
    } else {
      newprop_sigsq_zeta = prop_sigsq_zeta;
      Sigma_new_zeta = Sigma_zeta;
      dettnew_zeta = dett_zeta;
      b1 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Updating the tausq_zeta parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    temp_tausq = rnormscalarC(log(prop_tausq_zeta), step_tausq_zeta);
    newprop_tausq_zeta = exp(temp_tausq);
    
    // Updating Sigma - Selecting the relevant covariance matrix (I don't think the first line
    // is needed here, but will leave it in for now!):
    Sigma_new_zeta = prop_sigsq_zeta * cube_covar_zeta.slice(ind1);
    Sigma_new_zeta.replace(prop_sigsq_zeta, prop_sigsq_zeta + newprop_tausq_zeta);
    // Getting the relevant determinant:
    log_det(dettnew_zeta, sign, Sigma_new_zeta);
    
    probab = exp( post_tausq_zetaC(Sigma_new_zeta, prop_mzeta, prop_alpha_zeta,
                                   covar1, newprop_tausq_zeta, dettnew_zeta,
                                   tausq_zeta_mean, tausq_zeta_sd) -
                                     post_tausq_zetaC(Sigma_zeta, prop_mzeta, prop_alpha_zeta,
                                                      covar1, prop_tausq_zeta, dett_zeta,
                                                      tausq_zeta_mean, tausq_zeta_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_tausq_zeta = newprop_tausq_zeta;
      Sigma_zeta = Sigma_new_zeta;
      dett_zeta = dettnew_zeta;
      b2 = 1;
    } else {
      newprop_tausq_zeta = prop_tausq_zeta;
      Sigma_new_zeta = Sigma_zeta;
      dettnew_zeta = dett_zeta;
      b2 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Updating the nu_zeta parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    // A random number between 0 and the (number of possible nu's - 1):
    c_nu_zeta_new = rand() % len_nu_zeta;
    newprop_nu_zeta = nu_zeta_hyper_discrete(c_nu_zeta_new);
    
    // Getting the new index for the cubes, based on this random number:
    ind1 = (numbeta) * c_nu_zeta_new + c_beta_zeta;
    
    // Selecting the relevant covariance matrix:
    Sigma_new_zeta = prop_sigsq_zeta * cube_covar_zeta.slice(ind1);
    Sigma_new_zeta.replace(prop_sigsq_zeta, prop_sigsq_zeta + prop_tausq_zeta);
    // Getting the relevant determinant:
    log_det(dettnew_zeta, sign, Sigma_new_zeta);
    
    // Calculating the ratio with the new Sigma and nu vs. the old ones:
    probab = exp( post_nu_zetaC(Sigma_new_zeta, prop_mzeta, prop_alpha_zeta,
                                covar1, newprop_nu_zeta, dettnew_zeta,
                                nu_zeta_mean, nu_zeta_sd) -
                                  post_nu_zetaC(Sigma_zeta, prop_mzeta, prop_alpha_zeta,
                                                covar1, prop_nu_zeta, dett_zeta,
                                                nu_zeta_mean, nu_zeta_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_nu_zeta = newprop_nu_zeta;
      Sigma_zeta = Sigma_new_zeta;
      dett_zeta = dettnew_zeta;
      c_nu_zeta = c_nu_zeta_new;
      b10 = 1;
    } else {
      newprop_nu_zeta = prop_nu_zeta;
      Sigma_new_zeta = Sigma_zeta;
      dettnew_zeta = dett_zeta;
      c_nu_zeta_new = c_nu_zeta;
      b10 = 0;
    }
    
    // Recalculating the index:
    ind1 = (numbeta) * c_nu_zeta + c_beta_zeta;
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// Now to calculate the acceptance rates: ///////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    acc_zeta = acc_zeta + tempr;
    acc_alpha_zeta = acc_alpha_zeta + temps;
    acc_beta_zeta = acc_beta_zeta + b6;
    acc_sigsq_zeta = acc_sigsq_zeta + b1;
    acc_tausq_zeta = acc_tausq_zeta + b2;
    acc_nu_zeta = acc_nu_zeta + b10;
    
    /////////////////// Saving these at selected iterations: //////////////////////////////////////
    
    // Save if we're beyond the burn-in period *and* it's every nth iteration:
    if(i > burnin && i % nth == 0) {
      m = mzeta.n_rows;
      
      // Printing the value of i
      Rprintf("%d \n", i);
      
      // Writing all the values at each selected cycle for zeta:
      mzeta.insert_rows(m, prop_mzeta.t());
      alpha_zeta.insert_rows(m, prop_alpha_zeta.t());
      beta_zeta.insert_rows(m, vectorise(prop_beta_zeta).t());
      sigsq_zeta.push_back(prop_sigsq_zeta);
      tausq_zeta.push_back(prop_tausq_zeta);
      nu_zeta.push_back(prop_nu_zeta);
      nzeta.insert_rows(m, prop_nzeta.t());
      
      // Checking the values of the cube used for the zeta update:
      cube_zeta_counter.push_back(ind1);
      
    }
  }
  
  //////////////////////////// Getting ready to output results: //////////////////////
  // Writing each element of the list:
  out[0] = mzeta;
  out[1] = alpha_zeta;
  out[2] = beta_zeta;
  out[3] = sigsq_zeta;
  out[4] = tausq_zeta;
  out[5] = nu_zeta;
  out[6] = nzeta;
  
  out[7] = acc_zeta/iterations;
  out[8] = acc_alpha_zeta/iterations;
  out[9] = acc_beta_zeta/iterations;
  out[10] = acc_sigsq_zeta/iterations;
  out[11] = acc_tausq_zeta/iterations;
  out[12] = acc_nu_zeta/iterations;
  
  // Covar matrices
  out[13] = covar1;
  out[14] = bigcovar1;
  
  // Cube things:
  out[15] = cube_covar_zeta;
  out[16] = cube_nxm_zeta;
  out[17] = cube_zeta_counter;
  
  // Misc:
  out[18] = big_beta_mat;
  out[19] = prior;
  
  // Delete this after debugging is finished:
  out[20] = n_x_m_zeta;
  out[21] = ind1;
  
  // Creating the names for all the elements of the output list:
  int g1 = out.size();
  CharacterVector names(g1);
  names[0] = "mzeta";
  names[1] = "alpha_zeta";
  names[2] = "beta_zeta";
  names[3] = "sigsq_zeta";
  names[4] = "tausq_zeta";
  names[5] = "nu_zeta";
  names[6] = "nzeta";
  
  names[7] = "acc_zeta";
  names[8] = "acc_alpha_zeta";
  names[9] = "acc_beta_zeta";
  names[10] = "acc_sigsq_zeta";
  names[11] = "acc_tausq_zeta";
  names[12] = "acc_nu_zeta";
  
  names[13] = "covar";
  names[14] = "bigcovar";
  
  names[15] = "cube_covar_zeta";
  names[16] = "cube_nxm_zeta";
  names[17] = "cube_zeta_counter";
  
  names[18] = "big_beta_mat";
  names[19] = "prior";
  
  names[20] = "n_x_m_zeta";
  names[21] = "ind1";
  
  out.attr("names") = names;
  
  return out;
}

