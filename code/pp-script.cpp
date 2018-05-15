////////////////////////////////////////////////////////////////////////////////////
/////// MCMC Predictive Processes script ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// This script uses a Matern covariance function
// and implements the Predictive Processes piece in full

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
// Formula: for the Matern covariance:
// This works by evaluating the Matern on a single
// entry of the matrix (which in fact needs to be converted to a single number
// using the d^T * invbeta * d piece)
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// cube function:
////////////////////////////////////////////////////////////////////////////////////////////////////
// In this function, I create the big array (cube) over both all values of beta and all values of nu:
// It returns the cube into the function, which is then accessed by .slice() as appropriate:
// It seems clunky, but is only created once - so any speed-up wouldn't be worthwhile
// I also save it, for loading later:
// 1 is for the scale, 2 is for the shape, to allow nu and beta to be parameter specific:
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

// ################################################################################
// ############################# Data layer #######################################
// ################################################################################
// ################################################################################
// The full GPD, works on a list:
// This function, on the log scale, takes in a list of excesses for each gridpoint,
// and a single value of the threshold, the scale and the shape for each gridpoint
//(that is, a vector of values),
// and then evaluates the gpd at each. These are then summed up (log(product) = sum(logs) etc.)
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double dgpdC(List exc, vec lscale, vec shape, int n) {
  
  vec out(n);
  
  // Exponentiating the log of the scale:
  vec scale = exp(lscale);
  
  for(int i = 0; i < n; i++) {
    out[i] = sum(- log(scale[i]) + (-1/shape[i] - 1) * log(1 + shape[i] * (as<vec>(exc[i])/scale[i] )));
    //  out[i] = - log (scale[i]) - (as<vec>(exc[i]) - mu[i])/scale[i];
  }
  
  return sum(out);
}


// ################################################################################
// ############################# The latent process layer #########################
// ################################################################################
// The MVN for the GP layer:
// This is the main component needed for layer 2 - the mvn applied using
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
// The MVN in log form, for proportions: mvnC
// This gives the same value os dmvnorm(x, mu, sigma, log=T)
// when the (-k/2 * log(2 * pi)) factor is included at the end.
// This will be above and below, and will cancel
// This function is only needed in the update for alpha_scale and alpha_shape
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

///////////////////////////////////////////////////////////////////////////////////
// Evaluating the normal density for scalars:
// This straightforward function evaluates the normal density for a single scalar, given a mean
// and an sd. It then takes the log of this, and returns the value.
// (Why did I include the 2*pi? Come back to this. I assume it's not needed.)
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

// #############################################################################################################
// ############################# Additional functions needed ###########################################
// #############################################################################################################
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
// Then within the MCMC code, the relevant fractions are evaluated
// as exp ( log (top) - log (bottom) )
// in order to get the ratio needed

// ###############################################################################
// Evaluating the full posterior coditional of the scale:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_scaleC(List data, vec lscale, vec shape,
                   vec alpha, mat covar, vec mlscale, mat sigma, double dett_scale, int n_dim) {
  
  double a = dgpdC(data, lscale, shape, n_dim); // Layer 1
  double b = gp_mvnC2(mlscale, sigma, alpha, covar, dett_scale); // Layer 2
  return a + b;
}

// ################################################################################
// Evaluating the full posterior conditional of the shape:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_shapeC(List data, vec lscale, vec shape,
                   vec alpha, mat covar, vec mshape,
                   mat sigma_shape, double dett_shape, int n_dim) {
  
  double a = dgpdC(data, lscale, shape, n_dim); // Layer 1
  double b = gp_mvnC2(mshape, sigma_shape, alpha, covar, dett_shape); // Layer 2
  return a + b;
}

// ################################################################################
// Hyper-parameter inference:
// For these, we only deal with the m-dimensional reduced case now...:

// Evaluating the full posterior conditional of the alpha_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_alpha_scaleC(List data, vec lscale, vec shape,
                         vec alpha_scale, vec mlscale, mat covar, mat sigma,
                         vec alpha_scale_hyper_mean, mat alpha_scale_hyper_sd,
                         double dett_scale) {
  
  double a = gp_mvnC2(mlscale, sigma, alpha_scale, covar, dett_scale); // Layer2
  double b = mvnC(alpha_scale, alpha_scale_hyper_mean, alpha_scale_hyper_sd) ; //Layer3
  
  return a + b;
}  

// ################################################################################
// Evaluating the full posterior conditional of the beta_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_beta_scaleC(mat sigma, mat beta_scale, vec mlscale,
                        vec alpha_scale, mat covar, double dett_scale) {
  
  double a = gp_mvnC2(mlscale, sigma, alpha_scale, covar, dett_scale); // Layer2
  
  return a;
}  


// ################################################################################
// Evaluating the full posterior conditional of the sigsq_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_sigsq_scaleC(mat sigma, vec mlscale, vec alpha,
                         mat covar, double sigsq, double dett_scale,
                         double sigsq_scale_mean, double sigsq_scale_sd) {
  
  double a = gp_mvnC2(mlscale, sigma, alpha, covar, dett_scale); // Layer2
  double b =  dnormscalarC(log(sigsq), sigsq_scale_mean, sigsq_scale_sd); //Layer3
  
  return a + b;
}


// ################################################################################
// Evaluating the full posterior conditional of the tausq_scale parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_tausq_scaleC(mat sigma, vec mlscale, vec alpha,
                         mat covar, double tausq, double dett_scale,
                         double tausq_scale_mean, double tausq_scale_sd) {
  
  double a = gp_mvnC2(mlscale, sigma, alpha, covar, dett_scale); // Layer2
  double b = dnormscalarC(log(tausq), tausq_scale_mean, tausq_scale_sd); //Layer3
  
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
double post_nu_scaleC(mat sigma, vec mlscale, vec alpha,
                      mat covar, double nu, double dett_scale,
                      double nu_scale_mean, double nu_scale_sd) {
  
  double a = gp_mvnC2(mlscale, sigma, alpha, covar, dett_scale); // Layer2
  
  return a;
}


// ################################################################################
// Evaluating the full posterior conditional of the alpha_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_alpha_shapeC(vec alpha_shape, vec mshape, mat covar, mat sigma_shape,
                         vec alpha_shape_hyper_mean, mat alpha_shape_hyper_sd,
                         double dett_shape) {
  
  double a = gp_mvnC2(mshape, sigma_shape, alpha_shape, covar, dett_shape); // Layer2
  double b = mvnC(alpha_shape, alpha_shape_hyper_mean, alpha_shape_hyper_sd) ; //Layer3
  
  return a + b;
}  


// ################################################################################
// Evaluating the full posterior conditional of the beta_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::export]]
double post_beta_shapeC(mat sigma_shape, mat beta_shape, vec mshape, vec alpha_shape,
                        mat covar, double dett_shape) {
  
  double a = gp_mvnC2(mshape, sigma_shape, alpha_shape, covar, dett_shape); // Layer2
  
  return a;
}  


// ################################################################################
// Evaluating the full posterior conditional of the sigsq_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_sigsq_shapeC(mat sigma_shape, vec mshape, vec alpha, mat covar,
                         double sigsq, double dett_shape,
                         double sigsq_shape_mean, double sigsq_shape_sd) {
  
  double a = gp_mvnC2(mshape, sigma_shape, alpha, covar, dett_shape); // Layer2
  double b = dnormscalarC(log(sigsq), sigsq_shape_mean, sigsq_shape_sd); //Layer3
  
  return a + b;
}


// ################################################################################
// Evaluating the full posterior conditional of the tausq_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_tausq_shapeC(mat sigma_shape, vec mshape, vec alpha, mat covar,
                         double tausq, double dett_shape,
                         double tausq_shape_mean, double tausq_shape_sd) {
  
  double a = gp_mvnC2(mshape, sigma_shape, alpha, covar, dett_shape); // Layer2
  double b = dnormscalarC(log(tausq), tausq_shape_mean, tausq_shape_sd); //Layer3
  
  return a + b;
}

// ################################################################################
// Evaluating the full posterior conditional of the nu_shape parameters:
#include <math.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace std;
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double post_nu_shapeC(mat sigma_shape, vec mshape, vec alpha, mat covar,
                      double nu, double dett_shape,
                      double nu_shape_mean, double nu_shape_sd) {
  
  double a = gp_mvnC2(mshape, sigma_shape, alpha, covar, dett_shape); // Layer2
  
  return a;
}

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
List mcmcC(List data,
           List big_beta,
           mat big_beta_mat,
           List start,
           int iterations,
           mat covar1,
           mat covar2,
           mat bigcovar1,
           mat bigcovar2,
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
  
  // Creating a list with enough components to store everything needed at the output:
  // start is the same size as a single iteration of the process:
  List out(3 * start.size());
  
  // Creating the distance list, unchanged throughout
  List dist = dist_mat_diff(distance_vectors_m, distance_vectors_m);
  
  // Creating the distance list between the n grid and those points on the m grid, unchanged throughout:
  List dist_n_to_m = dist_mat_diff(distance_vectors_n, distance_vectors_m);
  
  //// Forming the matrices for the scale, to be filled during the MCMC step:
  // The log(scale) parameter:
  vec start0 = start[0];
  mat mlscale;
  mlscale.insert_rows(0, start0.t());
  
  // The alpha_scale coefficients:
  vec start1 = start[1];
  mat alpha_scale;
  alpha_scale.insert_rows(0, start1.t());
  
  // The beta_scale matrix:
  vec start2 = start[2];
  mat beta_scale;
  beta_scale.insert_rows(0, start2.t());
  
  // The sigsq_scale coefficient:
  vec start5 = start[3];
  double s5 = as_scalar(start5);
  NumericVector sigsq_scale;
  sigsq_scale.push_back(s5);
  
  // The tausq_scale coefficient:
  vec start6 = start[4];
  double s6 = as_scalar(start6);
  NumericVector tausq_scale;
  tausq_scale.push_back(s6);
  
  //// Next, forming the matrices for the shape, to be filled during the MCMC step:
  // The log(shape) parameter:
  vec start7 = start[5];
  mat mshape;
  mshape.insert_rows(0, start7.t());
  
  // The alpha_shape coefficients:
  vec start8 = start[6];
  mat alpha_shape;
  alpha_shape.insert_rows(0, start8.t());
  
  // The beta_shape matrix:
  vec start9 = start[7];
  mat beta_shape; 
  beta_shape.insert_rows(0, start9.t());
  
  // The sigsq_shape coefficient:
  vec start12 = start[8];
  double s12 = as_scalar(start12);
  NumericVector sigsq_shape;
  sigsq_shape.push_back(s12);
  
  // The tausq_shape coefficient:
  vec start13 = start[9];
  double s13 = as_scalar(start13);
  NumericVector tausq_shape;
  tausq_shape.push_back(s13);
  
  // Now to extract the longer, larger scales and shapes (n-dim):
  vec start10 = start[10];
  mat nlscale;
  nlscale.insert_rows(0, start10.t());
  
  vec start11 = start[11];
  mat nshape;
  nshape.insert_rows(0, start11.t());
  
  // Now, as it was added later, the nu parameters:
  vec start14 = start[12];
  double s14 = as_scalar(start14);
  NumericVector nu_scale;
  nu_scale.push_back(s14);
  
  vec start15 = start[13];
  double s15 = as_scalar(start15);
  NumericVector nu_shape;
  nu_shape.push_back(s15);
  
  // Setting the starting values for the scale and scale hyper-parameters:
  vec prop_mlscale = mlscale.row(0).t();
  vec prop_alpha_scale = alpha_scale.row(0).t();
  mat prop_beta_scale(dimension, dimension);
  prop_beta_scale = reshape(big_beta_mat.row(0), dimension, dimension);
  double prop_sigsq_scale = as_scalar(sigsq_scale(0));
  double prop_tausq_scale = as_scalar(tausq_scale(0));
  double prop_nu_scale    = as_scalar(nu_scale(0));
  vec prop_nlscale = nlscale.row(0).t();
  
  // Setting the starting values for the shape and shape hyper-parameters:
  vec prop_mshape = mshape.row(0).t();
  vec prop_alpha_shape = alpha_shape.row(0).t();
  mat prop_beta_shape(dimension, dimension);
  prop_beta_shape = reshape(big_beta_mat.row(0), dimension, dimension);
  double prop_sigsq_shape = as_scalar(sigsq_shape(0));
  double prop_tausq_shape = as_scalar(tausq_shape(0));
  double prop_nu_shape    = as_scalar(nu_shape(0));
  vec prop_nshape = nshape.row(0).t();
  
  // Now creating vectors to keep track of the acceptance rates
  int n1 = prop_mlscale.size(); int n2 = prop_alpha_scale.size();
  int n3 = prop_beta_scale.size();
  Rcpp::NumericVector tempr(n1); Rcpp::NumericVector temps(n2);
  Rcpp::NumericVector tempt(n3);
  
  int m1 = prop_mshape.size(); int m2 = prop_alpha_shape.size();
  int m3 = prop_beta_shape.size();
  Rcpp::NumericVector tempu(m1); Rcpp::NumericVector tempv(m2);
  Rcpp::NumericVector tempw(m3);
  
  // These are to be filled in:
  Rcpp::NumericVector acc_lscale(n1);
  Rcpp::NumericVector acc_alpha_scale(n2);
  double acc_beta_scale  = 0;
  double acc_sigsq_scale = 0;
  double acc_tausq_scale = 0;
  double acc_nu_scale    = 0;
  bool b1 = 0; bool b2 = 0; bool b6 = 0; bool b10 = 0;
  
  Rcpp::NumericVector acc_shape(m1);
  Rcpp::NumericVector acc_alpha_shape(m2);
  double acc_beta_shape  = 0;
  double acc_sigsq_shape = 0;
  double acc_tausq_shape = 0;
  double acc_nu_shape    = 0;
  bool b3 = 0; bool b4 = 0; bool b7 = 0; bool b11 = 0;
  
  ////////////////////////////////////////////////////////////////////////////////////////
  // Picking out the steps for the scale updates:
  // (These are essentially the standard deviations, inputted at the beginning. Adjusting these
  // will affect the acceptance rate. Too big, and not enough values are accepted.
  // Too small, and too many values are accepted, leading to slow mixing (exploration) of
  // the posterior space)
  vec step_scale = step[0]; vec step_alpha_scale = step[1]; vec step_sigsq_scale1 = step[2];
  double step_sigsq_scale = as_scalar(step_sigsq_scale1); vec step_tausq_scale1 = step[3];
  double step_tausq_scale = as_scalar(step_tausq_scale1); vec step_nu_scale1 = step[10];
  
  // Picking out the steps for the shape updates:
  // As above. The single values need to be taken as a vector, then set to a scalar:
  vec step_shape = step[4]; vec step_alpha_shape = step[5]; vec step_sigsq_shape1 = step[6];
  double step_sigsq_shape = as_scalar(step_sigsq_shape1); vec step_tausq_shape1 = step[7];
  double step_tausq_shape = as_scalar(step_tausq_shape1); vec step_nu_shape1 = step[11];
  
  // Picking out the scale hyper parameters:
  // These are the priors on the layer 3 parameters, and the end of our hierarchy:
  vec alpha_scale_hyper_mean = prior[0]; mat alpha_scale_hyper_sd = prior[1];
  vec beta_scale_hyper_prior = prior[2];
  // Now for sigsq, tausq and nu:
  vec sigsq_scale_mean1  = prior[10]; vec sigsq_scale_sd1   = prior[11];
  vec tausq_scale_mean1  = prior[12]; vec tausq_scale_sd1   = prior[13];
  vec nu_scale_mean1     = prior[14]; vec nu_scale_sd1      = prior[15];
  
  double sigsq_scale_mean  = as_scalar(sigsq_scale_mean1);
  double sigsq_scale_sd    = as_scalar(sigsq_scale_sd1);
  double tausq_scale_mean  = as_scalar(tausq_scale_mean1);
  double tausq_scale_sd    = as_scalar(tausq_scale_sd1);
  double nu_scale_mean     = as_scalar(nu_scale_mean1);
  double nu_scale_sd       = as_scalar(nu_scale_sd1);
  
  // Or in the option of a discrete update for nu - scale:
  vec nu_scale_hyper_discrete = prior[22];
  int len_nu_scale = nu_scale_hyper_discrete.size();
  
  // Picking out the shape hyper parameters:
  vec alpha_shape_hyper_mean = prior[5]; mat alpha_shape_hyper_sd = prior[6];
  vec beta_shape_hyper_prior = prior[7];
  // Now for sigsq, tausq and nu:
  vec sigsq_shape_mean1 = prior[16]; vec sigsq_shape_sd1 = prior[17];
  vec tausq_shape_mean1 = prior[18]; vec tausq_shape_sd1 = prior[19];
  vec nu_shape_mean1    = prior[20]; vec nu_shape_sd1    = prior[21];
  
  double sigsq_shape_mean = as_scalar(sigsq_shape_mean1);
  double sigsq_shape_sd   = as_scalar(sigsq_shape_sd1);
  double tausq_shape_mean = as_scalar(tausq_shape_mean1);
  double tausq_shape_sd   = as_scalar(tausq_shape_sd1);
  double nu_shape_mean    = as_scalar(nu_shape_mean1);
  double nu_shape_sd      = as_scalar(nu_shape_sd1);
  
  // Or in the option of a discrete update for nu - shape:
  vec nu_shape_hyper_discrete = prior[23];
  int len_nu_shape = nu_shape_hyper_discrete.size();
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // The n x m x (beta x nu) cubes to be created
  // and to be accessed to update the predicted process of the vector n
  // based on the current proposed values of the vector
  cube cube_covar_scale; cube cube_covar_shape; cube cube_nxm_scale; cube cube_nxm_shape;
  
  cube_covar_scale.load("cube1.cube");
  cube_covar_shape.load("cube2.cube");
  cube_nxm_scale.load("big_cube1.cube");
  cube_nxm_shape.load("big_cube2.cube");
  
  // Selecting the first of these to input into the MCMC process
  // This means beta and nu should have their starting value as the first one
  // in their vector/matrix
  // this should be fixed later to select the correct one if I want
  // to start beta and nu at different values. I then need to start off with
  // the one that corresponds to the starting values of the parameters:
  // Creating Sigma_scale for input to the first step of the MCMC process:
  mat Sigma_scale = cube_covar_scale.slice(0);
  mat Sigma_new_scale = Sigma_scale;
  
  // Creating Sigma_shape for input to the first step of the MCMC process:
  mat Sigma_shape = cube_covar_shape.slice(0);
  mat Sigma_new_shape = Sigma_shape;
  
  ////////////////////////////////////////////////////////////////////////////////////////
  // doubles needed for the log det. step, all of which are calculated in vectors below:
  double dett_scale; double sign = 1;
  double dett_shape;
  double dettnew_scale = dett_scale;
  double dettnew_shape = dett_shape;
  
  ////////////////////////////////////////////////////////////////////////////////////////
  // Defining the items to be used in the scale update:
  // These will essentially be the last items in the chain - vectors and scalars
  // that are rewritten at
  // each line, and copied into the big matrix if they are accepted:
  vec newprop_mlscale;
  vec newprop_alpha_scale;
  mat newprop_beta_scale(dimension, dimension);
  double temp_sigsq; double newprop_sigsq_scale;
  double temp_tausq; double newprop_tausq_scale;
  double newprop_nu_scale;
  vec newprop_nlscale=prop_nlscale;
  
  // Defining the items to be used in the shape update:
  vec newprop_mshape;
  vec newprop_alpha_shape;
  mat newprop_beta_shape;
  double newprop_sigsq_shape;
  double newprop_tausq_shape;
  double newprop_nu_shape;
  vec newprop_nshape=prop_nshape;
  
  // Doubles, vecs and ints to be used within the loops:
  double rndm;
  double probab=0;
  vec rnd;
  int m;
  
  // The n x m matrices to be created and rewritten
  // and to be used to update the predicted process of the vector n
  // based on the current proposed values of the vector m
  mat n_x_m_scale = cube_nxm_scale.slice(0);
  mat n_x_m_shape = cube_nxm_shape.slice(0);
  
  // Declaring new ints:
  int numbeta = big_beta_mat.n_rows;
  int c_beta_scale = 0; int c_nu_scale = 0;
  int c_beta_scale_new = 0; int c_nu_scale_new = 0;
  int c_beta_shape = 0; int c_nu_shape = 0;
  int c_beta_shape_new = 0; int c_nu_shape_new = 0;
  
  // Integer for the index of vectors:
  int ind1 = 0; int ind2 = 0;
  ind1 = (numbeta) * c_nu_scale + c_beta_scale;
  ind2 = (numbeta) * c_nu_shape + c_beta_shape;
  
  // Saving these at each iteration:
  NumericVector cube_scale_counter;
  cube_scale_counter.push_back(ind1);
  
  NumericVector cube_shape_counter;
  cube_shape_counter.push_back(ind2);
  
  // A quick fix (it's not a big deal)
  prop_tausq_scale = 0.01; prop_tausq_shape = 0.01;
  // It's because to make the cube work, I neeed to set tausq to be 0 initially -
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
    
    // Now to update the values of nlscale and nshape depending on the current value
    // of mlscale and mshape, and the matrices etc.:
    
    // ///////////////////////////////////////////////////////////////////////////////////
    // ///////////////////////// Scale updates now: //////////////////////////////////////
    // //////////////////////////////////////////////////////////////////////////////////
    // // Updating the lscale values:
    // ////////////////////////////////////////////////////////////////////////////////////
    // // Setting things up:
    // ///////////////////////////////////////////////////////////////////////////////////
    newprop_mlscale = prop_mlscale;
    
    // Getting the correct Sigma_scale for the scale update:
    Sigma_scale = prop_sigsq_scale * cube_covar_scale.slice(ind1);
    Sigma_scale.replace(prop_sigsq_scale, prop_sigsq_scale + prop_tausq_scale);
    
    // Getting the correct n x m matrix once for the scale update:
    n_x_m_scale = prop_sigsq_scale * cube_nxm_scale.slice(ind1);
    n_x_m_scale.replace(prop_sigsq_scale, prop_sigsq_scale + prop_tausq_scale);
    
    // Getting the relevant determinant:
    log_det(dett_scale, sign, Sigma_scale);
    ///////////////////////////////////////////////////////////////////////////////////
    // Now getting the value on the larger grid using the kriging equations:
    prop_nlscale = bigcovar1 * prop_alpha_scale +
      n_x_m_scale * solve(Sigma_scale, prop_mlscale - covar1 * prop_alpha_scale);
    
    newprop_nlscale = prop_nlscale;
    
    // Getting the necessary indices (updating in a random order across the grid):
    rand_ind = Rcpp::RcppArmadillo::sample(ind_m, m_dim, 0, prob1);
    
    for (unsigned int l = 0; l < prop_mlscale.size(); ++l) {
      
      // Updating each of the m entries one by one:
      newprop_mlscale[rand_ind[l]] = rnormscalarC(prop_mlscale[rand_ind[l]], step_scale(0));
      
      // Now to update the values of nlscale depending on the current value
      // of newprop_mlscale, and the matrices etc. (the PP part):
      newprop_nlscale = bigcovar1 * prop_alpha_scale +
        n_x_m_scale * solve(Sigma_scale, newprop_mlscale - covar1 * prop_alpha_scale);
      
      // Calculating the MCMC fraction:
      probab = exp( post_scaleC(data, newprop_nlscale, prop_nshape, prop_alpha_scale,
                                covar1, newprop_mlscale, Sigma_scale, dett_scale, n_dim) -
                                  post_scaleC(data, prop_nlscale, prop_nshape, prop_alpha_scale,
                                              covar1, prop_mlscale, Sigma_scale, dett_scale, n_dim) );
      
      if (rndm < probab) {
        prop_nlscale = newprop_nlscale;
        prop_mlscale = newprop_mlscale;
        tempr[rand_ind[l]] = 1;
      } else {
        newprop_nlscale = prop_nlscale;
        newprop_mlscale = prop_mlscale;
        tempr[rand_ind[l]] = 0;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Updating the alpha_scale parameter:
    ////////////////////////////////////////////////////////////////////////////////////
    newprop_alpha_scale = prop_alpha_scale;
    for (unsigned int l = 0; l < prop_alpha_scale.size(); ++l) {
      
      newprop_alpha_scale[l] = rnormscalarC(prop_alpha_scale[l], step_alpha_scale[l]);
      
      // Now to update the values of nlscale depending on the current value
      // of newprop_mlscale, and the new alpha hyper-parameters and the matrices etc. (the PP part):
      newprop_nlscale = bigcovar1 * newprop_alpha_scale + n_x_m_scale *
        solve(Sigma_scale, prop_mlscale - covar1 * newprop_alpha_scale);
      
      probab = exp (post_alpha_scaleC(data, newprop_nlscale, prop_nshape,
                                      newprop_alpha_scale, prop_mlscale, covar1, Sigma_scale,
                                      alpha_scale_hyper_mean, alpha_scale_hyper_sd, dett_scale) -
                                        post_alpha_scaleC(data, prop_nlscale, prop_nshape,
                                                          prop_alpha_scale, prop_mlscale, covar1, Sigma_scale,
                                                          alpha_scale_hyper_mean, alpha_scale_hyper_sd, dett_scale) ) ;
      
      //  Come back to this:
      if (rndm < probab) {
        prop_alpha_scale = newprop_alpha_scale;
        prop_nlscale = newprop_nlscale;
        temps[l] = 1;
      } else {
        newprop_alpha_scale = prop_alpha_scale;
        newprop_nlscale = prop_nlscale;
        temps[l] = 0;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Updating the beta_scale parameter:
    ////////////////////////////////////////////////////////////////////////////////////
    // A random number between 0 and (number of beta matrices - 1):
    c_beta_scale_new = rand() % numbeta;
    
    // Getting the new index for the cubes, based on this random number:
    ind1 = (numbeta) * c_nu_scale + c_beta_scale_new;
    
    // Selecting the entries of beta:
    // Reshaping the vector into a matrix:
    newprop_beta_scale = reshape(big_beta_mat.row(c_beta_scale_new), dimension, dimension);
    
    // Selecting the relevant covariance matrix:
    Sigma_new_scale = prop_sigsq_scale * cube_covar_scale.slice(ind1);
    Sigma_new_scale.replace(prop_sigsq_scale, prop_sigsq_scale + prop_tausq_scale);
    // Getting the relevant determinant:
    log_det(dettnew_scale, sign, Sigma_new_scale);
    
    // Calculating the ratio with the new Sigma and beta vs. the old ones:
    probab = exp (post_beta_scaleC(Sigma_new_scale, newprop_beta_scale, prop_mlscale,
                                   prop_alpha_scale, covar1, dettnew_scale) -
                                     post_beta_scaleC(Sigma_scale, prop_beta_scale, prop_mlscale,
                                                      prop_alpha_scale, covar1, dett_scale) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_beta_scale = newprop_beta_scale;
      Sigma_scale = Sigma_new_scale;
      dett_scale = dettnew_scale;
      c_beta_scale = c_beta_scale_new;
      b6 = 1;
    } else {
      newprop_beta_scale = prop_beta_scale;
      Sigma_new_scale = Sigma_scale;
      dettnew_scale = dett_scale;
      c_beta_scale_new = c_beta_scale;
      b6 = 0;
    }
    
    // Recalculating the index:
    ind1 = (numbeta) * c_nu_scale + c_beta_scale;
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Updating the sigsq_scale parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    temp_sigsq = rnormscalarC(log(prop_sigsq_scale), step_sigsq_scale);
    newprop_sigsq_scale = exp(temp_sigsq);
    
    // Updating Sigma - Selecting the relevant covariance matrix:
    Sigma_new_scale = newprop_sigsq_scale * cube_covar_scale.slice(ind1);
    Sigma_new_scale.replace(newprop_sigsq_scale, newprop_sigsq_scale + prop_tausq_scale);
    // Getting the relevant determinant:
    log_det(dettnew_scale, sign, Sigma_new_scale);
    
    // Calculating probab:
    probab = exp( post_sigsq_scaleC(Sigma_new_scale, prop_mlscale, prop_alpha_scale,
                                    covar1, newprop_sigsq_scale, dettnew_scale,
                                    sigsq_scale_mean, sigsq_scale_sd) -
                                      post_sigsq_scaleC(Sigma_scale, prop_mlscale, prop_alpha_scale,
                                                        covar1, prop_sigsq_scale, dett_scale,
                                                        sigsq_scale_mean, sigsq_scale_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_sigsq_scale = newprop_sigsq_scale;
      Sigma_scale = Sigma_new_scale;
      dett_scale = dettnew_scale;
      b1 = 1;
    } else {
      newprop_sigsq_scale = prop_sigsq_scale;
      Sigma_new_scale = Sigma_scale;
      dettnew_scale = dett_scale;
      b1 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Updating the tausq_scale parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    temp_tausq = rnormscalarC(log(prop_tausq_scale), step_tausq_scale);
    newprop_tausq_scale = exp(temp_tausq);
    
    // Updating Sigma - Selecting the relevant covariance matrix (I don't think the first line
    // is needed here, but will leave it in for now!):
    Sigma_new_scale = prop_sigsq_scale * cube_covar_scale.slice(ind1);
    Sigma_new_scale.replace(prop_sigsq_scale, prop_sigsq_scale + newprop_tausq_scale);
    // Getting the relevant determinant:
    log_det(dettnew_scale, sign, Sigma_new_scale);
    
    probab = exp( post_tausq_scaleC(Sigma_new_scale, prop_mlscale, prop_alpha_scale,
                                    covar1, newprop_tausq_scale, dettnew_scale,
                                    tausq_scale_mean, tausq_scale_sd) -
                                      post_tausq_scaleC(Sigma_scale, prop_mlscale, prop_alpha_scale,
                                                        covar1, prop_tausq_scale, dett_scale,
                                                        tausq_scale_mean, tausq_scale_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_tausq_scale = newprop_tausq_scale;
      Sigma_scale = Sigma_new_scale;
      dett_scale = dettnew_scale;
      b2 = 1;
    } else {
      newprop_tausq_scale = prop_tausq_scale;
      Sigma_new_scale = Sigma_scale;
      dettnew_scale = dett_scale;
      b2 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Updating the nu_scale parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    // A random number between 0 and the (number of possible nu's - 1):
    c_nu_scale_new = rand() % len_nu_scale;
    newprop_nu_scale = nu_scale_hyper_discrete(c_nu_scale_new);
    
    // Getting the new index for the cubes, based on this random number:
    ind1 = (numbeta) * c_nu_scale_new + c_beta_scale;
    
    // Selecting the relevant covariance matrix:
    Sigma_new_scale = prop_sigsq_scale * cube_covar_scale.slice(ind1);
    Sigma_new_scale.replace(prop_sigsq_scale, prop_sigsq_scale + prop_tausq_scale);
    // Getting the relevant determinant:
    log_det(dettnew_scale, sign, Sigma_new_scale);
    
    // Calculating the ratio with the new Sigma and nu vs. the old ones:
    probab = exp( post_nu_scaleC(Sigma_new_scale, prop_mlscale, prop_alpha_scale,
                                 covar1, newprop_nu_scale, dettnew_scale,
                                 nu_scale_mean, nu_scale_sd) -
                                   post_nu_scaleC(Sigma_scale, prop_mlscale, prop_alpha_scale,
                                                  covar1, prop_nu_scale, dett_scale,
                                                  nu_scale_mean, nu_scale_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_nu_scale = newprop_nu_scale;
      Sigma_scale = Sigma_new_scale;
      dett_scale = dettnew_scale;
      c_nu_scale = c_nu_scale_new;
      b10 = 1;
    } else {
      newprop_nu_scale = prop_nu_scale;
      Sigma_new_scale = Sigma_scale;
      dettnew_scale = dett_scale;
      c_nu_scale_new = c_nu_scale;
      b10 = 0;
    }
    
    // Recalculating the index:
    ind1 = (numbeta) * c_nu_scale + c_beta_scale;
    
    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Shape updates now: //////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    // Updating the shape parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    newprop_mshape = prop_mshape;
    
    // Getting the correct Sigma_shape for the shape update:
    Sigma_shape = prop_sigsq_shape * cube_covar_shape.slice(ind2);
    Sigma_shape.replace(prop_sigsq_shape, prop_sigsq_shape + prop_tausq_shape);
    
    // Getting the correct n x m matrix once for the shape update:
    n_x_m_shape = prop_sigsq_shape * cube_nxm_shape.slice(ind2);
    n_x_m_shape.replace(prop_sigsq_shape, prop_sigsq_shape + prop_tausq_shape);
    
    // Getting the relevant determinant:
    log_det(dett_shape, sign, Sigma_shape);
    
    ///////////////////////////////////////////////////////////////////////////////////
    // Now getting the value on the larger grid using the kriging equations:
    
    prop_nshape = bigcovar2 * prop_alpha_shape + n_x_m_shape *
      solve(Sigma_shape, prop_mshape - covar2 * prop_alpha_shape);
    
    newprop_nshape = prop_nshape;
    
    // Getting the necessary indices (updating in a random order):
    rand_ind = Rcpp::RcppArmadillo::sample(ind_m, m_dim, 0, prob1);
    
    for (unsigned int l = 0; l < prop_mshape.size(); ++l) {
      
      // Updating each of the m entries one by one:
      newprop_mshape[rand_ind[l]] = rnormscalarC(prop_mshape[rand_ind[l]], step_shape(0));
      
      // Now to update the values of nshape depending on the current value
      // of newprop_mshape, and the matrices etc. (the PP part)
      newprop_nshape = bigcovar2 * prop_alpha_shape + n_x_m_shape *
        solve(Sigma_shape, newprop_mshape - covar2 * prop_alpha_shape);
      
      // Calculating the MCMC fraction:
      probab = exp( post_shapeC(data, prop_nlscale, newprop_nshape, prop_alpha_shape,
                                covar2, newprop_mshape, Sigma_shape, dett_shape, n_dim) -
                                  post_shapeC(data, prop_nlscale, prop_nshape, prop_alpha_shape,
                                              covar2, prop_mshape, Sigma_shape, dett_shape, n_dim) );
      
      //  Come back to this:
      if (rndm < probab) {
        prop_mshape = newprop_mshape;
        prop_nshape = newprop_nshape;
        tempu[rand_ind[l]] = 1;
      } else {
        newprop_mshape = prop_mshape;
        newprop_nshape = prop_nshape;
        tempu[rand_ind[l]] = 0;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Updating the alpha_shape parameter:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    newprop_alpha_shape = prop_alpha_shape;
    for (unsigned int l = 0; l < prop_alpha_shape.size(); ++l) {
      
      newprop_alpha_shape[l] = rnormscalarC(prop_alpha_shape[l], step_alpha_shape[l]);
      
      probab = exp (post_alpha_shapeC(newprop_alpha_shape, prop_mshape, covar2, Sigma_shape,
                                      alpha_shape_hyper_mean, alpha_shape_hyper_sd, dett_shape) -
                                        post_alpha_shapeC(prop_alpha_shape, prop_mshape, covar2, Sigma_shape,
                                                          alpha_shape_hyper_mean, alpha_shape_hyper_sd, dett_shape) ) ;
      
      if (rndm < probab) {
        prop_alpha_shape = newprop_alpha_shape;
        tempv[l] = 1;
      } else {
        newprop_alpha_shape = prop_alpha_shape;
        tempv[l] = 0;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Updating the beta_shape parameter:
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // A random number between 0 and (number of beta matrices - 1):
    c_beta_shape_new = rand() % numbeta;
    
    // Getting the new index for the cubes, based on this random number:
    ind2 = (numbeta) * c_nu_shape + c_beta_shape_new;
    
    // Selecting the entries of beta:
    // Reshaping the vector into a matrix:
    newprop_beta_shape = reshape(big_beta_mat.row(c_beta_shape_new), dimension, dimension);
    
    // Selecting the relevant covariance matrix:
    Sigma_new_shape = prop_sigsq_shape * cube_covar_shape.slice(ind2);
    Sigma_new_shape.replace(prop_sigsq_shape, prop_sigsq_shape + prop_tausq_shape);
    // Getting the relevant determinant:
    log_det(dettnew_shape, sign, Sigma_new_shape);
    
    // Calculating the ratio with the new Sigma and beta vs. the old ones:
    probab = exp (post_beta_shapeC(Sigma_new_shape, newprop_beta_shape, prop_mshape,
                                   prop_alpha_shape, covar2, dettnew_shape) -
                                     post_beta_shapeC(Sigma_shape, prop_beta_shape, prop_mshape,
                                                      prop_alpha_shape, covar2, dett_shape) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_beta_shape = newprop_beta_shape;
      Sigma_shape = Sigma_new_shape;
      dett_shape = dettnew_shape;
      c_beta_shape = c_beta_shape_new;
      b7 = 1;
    } else {
      newprop_beta_shape = prop_beta_shape;
      Sigma_new_shape = Sigma_shape;
      dettnew_shape = dett_shape;
      c_beta_shape_new = c_beta_shape;
      b7 = 0;
    }
    
    // Recalculating the index:
    ind2 = (numbeta) * c_nu_shape + c_beta_shape;
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Updating the sigsq_shape parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    temp_sigsq = rnormscalarC(log(prop_sigsq_shape), step_sigsq_shape);
    newprop_sigsq_shape = exp(temp_sigsq);
    
    // Updating Sigma - Selecting the relevant covariance matrix:
    Sigma_new_shape = newprop_sigsq_shape * cube_covar_shape.slice(ind2);
    Sigma_new_shape.replace(newprop_sigsq_shape, newprop_sigsq_shape + prop_tausq_shape);
    // Getting the relevant determinant:
    log_det(dettnew_shape, sign, Sigma_new_shape);
    
    // Calculating probab:
    probab = exp( post_sigsq_shapeC(Sigma_new_shape, prop_mshape, prop_alpha_shape,
                                    covar1, newprop_sigsq_shape, dettnew_shape,
                                    sigsq_shape_mean, sigsq_shape_sd) -
                                      post_sigsq_shapeC(Sigma_shape, prop_mshape, prop_alpha_shape,
                                                        covar1, prop_sigsq_shape, dett_shape,
                                                        sigsq_shape_mean, sigsq_shape_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_sigsq_shape = newprop_sigsq_shape;
      Sigma_shape = Sigma_new_shape;
      dett_shape = dettnew_shape;
      b3 = 1;
    } else {
      newprop_sigsq_shape = prop_sigsq_shape;
      Sigma_new_shape = Sigma_shape;
      dettnew_shape = dett_shape;
      b3 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Updating the tausq_shape parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    temp_tausq = rnormscalarC(log(prop_tausq_shape), step_tausq_shape);
    newprop_tausq_shape = exp(temp_tausq);
    
    // Updating Sigma - Selecting the relevant covariance matrix (I don't think the first line
    // is needed here, but will leave it in for now!):
    Sigma_new_shape = prop_sigsq_shape * cube_covar_shape.slice(ind2);
    Sigma_new_shape.replace(prop_sigsq_shape, prop_sigsq_shape + newprop_tausq_shape);
    // Getting the relevant determinant:
    log_det(dettnew_shape, sign, Sigma_new_shape);
    
    probab = exp( post_tausq_shapeC(Sigma_new_shape, prop_mshape, prop_alpha_shape,
                                    covar1, newprop_tausq_shape, dettnew_shape,
                                    tausq_shape_mean, tausq_shape_sd) -
                                      post_tausq_shapeC(Sigma_shape, prop_mshape, prop_alpha_shape,
                                                        covar1, prop_tausq_shape, dett_shape,
                                                        tausq_shape_mean, tausq_shape_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_tausq_shape = newprop_tausq_shape;
      Sigma_shape = Sigma_new_shape;
      dett_shape = dettnew_shape;
      b4 = 1;
    } else {
      newprop_tausq_shape = prop_tausq_shape;
      Sigma_new_shape = Sigma_shape;
      dettnew_shape = dett_shape;
      b4 = 0;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Updating the nu_shape parameter now:
    ////////////////////////////////////////////////////////////////////////////////////
    // A random number between 0 and the (number of possible nu's - 1):
    c_nu_shape_new = rand() % len_nu_shape;
    newprop_nu_shape = nu_shape_hyper_discrete(c_nu_shape_new);
    
    // Getting the new index for the cubes, based on this random number:
    ind2 = (numbeta) * c_nu_shape_new + c_beta_shape;
    
    // Selecting the relevant covariance matrix:
    Sigma_new_shape = prop_sigsq_shape * cube_covar_shape.slice(ind2);
    Sigma_new_shape.replace(prop_sigsq_shape, prop_sigsq_shape + prop_tausq_shape);
    // Getting the relevant determinant:
    log_det(dettnew_shape, sign, Sigma_new_shape);
    
    // Calculating the ratio with the new Sigma and nu vs. the old ones:
    probab = exp( post_nu_shapeC(Sigma_new_shape, prop_mshape, prop_alpha_shape,
                                 covar1, newprop_nu_shape, dettnew_shape,
                                 nu_shape_mean, nu_shape_sd) -
                                   post_nu_shapeC(Sigma_shape, prop_mshape, prop_alpha_shape,
                                                  covar1, prop_nu_shape, dett_shape,
                                                  nu_shape_mean, nu_shape_sd) );
    
    // Accept or reject step:
    if (rndm < probab) {
      prop_nu_shape = newprop_nu_shape;
      Sigma_shape = Sigma_new_shape;
      dett_shape = dettnew_shape;
      c_nu_shape = c_nu_shape_new;
      b11 = 1;
    } else {
      newprop_nu_shape = prop_nu_shape;
      Sigma_new_shape = Sigma_shape;
      dettnew_shape = dett_shape;
      c_nu_shape_new = c_nu_shape;
      b11 = 0;
    }
    
    // Recalculating the index:
    ind2 = (numbeta) * c_nu_shape + c_beta_shape;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// Now to calculate the acceptance rates: ///////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    acc_lscale = acc_lscale + tempr;
    acc_alpha_scale = acc_alpha_scale + temps;
    acc_beta_scale = acc_beta_scale + b6;
    acc_sigsq_scale = acc_sigsq_scale + b1;
    acc_tausq_scale = acc_tausq_scale + b2;
    acc_nu_scale = acc_nu_scale + b10;
    
    acc_shape = acc_shape + tempu;
    acc_alpha_shape = acc_alpha_shape + tempv;
    acc_beta_shape = acc_beta_shape + b7;
    acc_sigsq_shape = acc_sigsq_shape + b3;
    acc_tausq_shape = acc_tausq_shape + b4;
    acc_nu_shape = acc_nu_shape + b11;
    
    /////////////////// Saving these at selected iterations: //////////////////////////////////////
    
    // Save if we're beyond the burn-in period *and* it's every nth iteration:
    if(i > burnin && i % nth == 0) {
      m = mlscale.n_rows;
      
      // Writing all the values at each selected cycle for the scale:
      mlscale.insert_rows(m, prop_mlscale.t());
      alpha_scale.insert_rows(m, prop_alpha_scale.t());
      beta_scale.insert_rows(m, vectorise(prop_beta_scale).t());
      sigsq_scale.push_back(prop_sigsq_scale);
      tausq_scale.push_back(prop_tausq_scale);
      nu_scale.push_back(prop_nu_scale);
      nlscale.insert_rows(m, prop_nlscale.t());
      
      // Checking the values of the cube used for the scale update:
      cube_scale_counter.push_back(ind1);
      
      // Writing all the values at each selected cycle for the shape:
      mshape.insert_rows(m, prop_mshape.t());
      alpha_shape.insert_rows(m, prop_alpha_shape.t());
      beta_shape.insert_rows(m, vectorise(prop_beta_shape).t());
      sigsq_shape.push_back(prop_sigsq_shape);
      tausq_shape.push_back(prop_tausq_shape);
      nu_shape.push_back(prop_nu_shape);
      nshape.insert_rows(m, prop_nshape.t());
      
      // Checking the values of the cube used for the shape update:
      cube_shape_counter.push_back(ind2);
      
    }
    
  }
  
  //////////////////////////// Getting ready to output results: //////////////////////
  // Writing each element of the list:
  out[0] = exp(mlscale);
  out[1] = alpha_scale;
  out[2] = beta_scale;
  out[3] = sigsq_scale;
  out[4] = tausq_scale;
  
  out[5] = mshape;
  out[6] = alpha_shape;
  out[7] = beta_shape;
  out[8] = sigsq_shape;
  out[9] = tausq_shape;
  
  out[10] = acc_lscale/iterations;
  out[11] = acc_alpha_scale/iterations;
  out[12] = acc_beta_scale/iterations;
  out[13] = acc_sigsq_scale/iterations;
  out[14] = acc_tausq_scale/iterations;
  
  out[15] = acc_shape/iterations;
  out[16] = acc_alpha_shape/iterations;
  out[17] = acc_beta_shape/iterations;
  out[18] = acc_sigsq_shape/iterations;
  out[19] = acc_tausq_shape/iterations;
  
  out[20] = exp(nlscale);
  out[21] = nshape;
  
  out[22] = covar1;
  out[23] = covar2;
  
  // nu:
  out[24] = nu_scale;
  out[25] = nu_shape;
  
  out[26] = acc_nu_scale/iterations;
  out[27] = acc_nu_shape/iterations;
  
  out[28] = cube_covar_scale; // 0;
  out[29] = cube_covar_shape; // 0;
  
  out[30] = cube_nxm_scale; // 0;
  out[31] = cube_nxm_shape; // 0;
  
  out[32] = cube_scale_counter;
  out[33] = cube_shape_counter;
  
  out[34] = big_beta_mat;
  out[35] = prior;
  
  out[36] = bigcovar1;
  out[37] = bigcovar2;
  
  // Creating the names for all the elements of the output list:
  int g1 = out.size();
  CharacterVector names(g1);
  names[0] = "scale";
  names[1] = "alpha_scale";
  names[2] = "beta_scale";
  names[3] = "sigsq_scale";
  names[4] = "tausq_scale";
  
  names[5] = "shape";
  names[6] = "alpha_shape";
  names[7] = "beta_shape";
  names[8] = "sigsq_shape";
  names[9] = "tausq_shape";
  
  names[10] = "acc_scale";
  names[11] = "acc_alpha_scale";
  names[12] = "acc_beta_scale";
  names[13] = "acc_sigsq_scale";
  names[14] = "acc_tausq_scale";
  
  names[15] = "acc_shape";
  names[16] = "acc_alpha_shape";
  names[17] = "acc_beta_shape";
  names[18] = "acc_sigsq_shape";
  names[19] = "acc_tausq_shape";
  
  names[20] = "nscale";
  names[21] = "nshape";
  
  names[22] = "covar1";
  names[23] = "covar2";
  
  names[24] = "nu_scale";
  names[25] = "nu_shape";
  
  names[26] = "acc_nu_scale";
  names[27] = "acc_nu_shape";
  
  names[28] = "cube_covar_scale";
  names[29] = "cube_covar_shape";
  
  names[30] = "cube_nxm_scale";
  names[31] = "cube_nxm_shape";
  
  names[32] = "cube_scale_index";
  names[33] = "cube_shape_index";
  
  names[34] = "betas_all";
  names[35] = "prior";
  
  names[36] = "bigcovar1";
  names[37] = "bigcovar2";
  
  names[38] = "blank";
  names[39] = "blank1";
  names[40] = "blank2";
  names[41] = "blank3";
  
  out.attr("names") = names;
  
  return out;
}

///////////////////////////////////////////////////////////////////////////////////////////
