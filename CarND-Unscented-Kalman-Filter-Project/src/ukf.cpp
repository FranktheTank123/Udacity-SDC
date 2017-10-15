#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //create sigma point matrix
  MatrixXd Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  //create sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  // Predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // R matrices
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
  0, std_radphi_*std_radphi_, 0,
  0, 0, std_radrd_*std_radrd_;

  R_laser_ = MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_, 0,
  0, std_laspy_*std_laspy_;

  // Weights of sigma points
  weights_ = VectorXd(n_sig_);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (auto i=1; i<n_sig_; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */

double NormalizeAngle(double phi){
  return atan2(sin(phi), cos(phi));
}

void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

  if (!is_initialized_) {
    // first measurement
    cout << "UKF: " << endl;

    double px, py, vx, vy;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rhodot = measurement_pack.raw_measurements_[2];
      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = rhodot * cos(phi);
      vy = rhodot * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
      vx = 0;
      vy = 0;
    }

    x_ << px, py, sqrt(pow(vx, 2) + pow(vy, 2)), 0, 0;
    cout << "init x_: " << x_ << endl;

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  cout << "Start predicting" << endl;
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  cout << "dt: " << dt << endl;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Generate Augmented sigma points
  AugmentedSigmaPoints();
  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates;
    UpdateRadar(measurement_pack);
  } else {
    // Laser updates
    UpdateLidar(measurement_pack);
  }

  // print NIS
  cout << "NIS_radar_ = " << NIS_radar_  << endl;
  cout << "NIS_laser_ = " << NIS_laser_  << endl;

}

void UKF::AugmentedSigmaPoints() {

  //create augmented mean and Cov
  VectorXd x_aug = VectorXd(7);
  MatrixXd P_aug = MatrixXd(7, 7);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //augmented sigma points
  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);
  Xsig_aug_.col(0)  = x_aug;
  for (auto i = 0; i< n_aug_; i++) {
    Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

void UKF::PredictSigmaPoints(double dt) {
  //predict sigma points
  for (auto i = 0; i< n_sig_; i++){
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    //predicted state values
    double px_p, py_p;

    //edge case
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * dt) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * dt));
    }
    else {
      px_p = p_x + v * dt * cos(yaw);
      py_p = p_y + v * dt * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * dt;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * dt * dt * cos(yaw);
    py_p = py_p + 0.5 * nu_a * dt * dt * sin(yaw);
    v_p = v_p + nu_a * dt;

    yaw_p = yaw_p + 0.5 * nu_yawdd * dt * dt;
    yawd_p = yawd_p + nu_yawdd * dt;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {

  //create vector and covariance for predicted state
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predicted state mean
  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P.fill(0.0);
  for (auto i = 0; i < n_sig_; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);

    x_diff(3) = NormalizeAngle(x_diff(3));
    P += weights_(i) * x_diff * x_diff.transpose() ;
  }
  P_ = P;

  std::cout << "Predicted state:\n"<< x_ << std::endl;
  std::cout << "Predicted covariance matrix:\n" << P_ << std::endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

void UKF::PredictRadarMeasurement() {
  //set measurement dimension
  int n_z = 3;

  //sigma points -> measurement space
  z_pred_ = VectorXd(n_z);
  Zsig_ = MatrixXd(n_z, n_sig_);
  S_ = MatrixXd(n_z, n_z);

  for (auto i = 0; i < n_sig_; i++){
    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    Zsig_(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig_(1, i) = atan2(p_y, p_x);
    Zsig_(2, i) = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / sqrt(p_x * p_x + p_y * p_y);
  }

  //mean predicted measurement
  z_pred_ = Zsig_ * weights_;

  // predicted measurement covariance
  //measurement covariance matrix S
  S_.fill(0.0);
  for (auto i = 0; i < n_sig_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    z_diff(1) = NormalizeAngle(z_diff(1));
    S_ += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S_ += R_radar_;
}

void UKF::PredictLidarMeasurement() {

  //set measurement dimension
  int n_z = 2;

  // mean predicted measurement
  z_pred_ = VectorXd(n_z);
  Zsig_ = MatrixXd(n_z, n_sig_);
  S_ = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (auto i = 0; i < n_sig_; i++) {
    Zsig_(0,i) =  Xsig_pred_(0,i);
    Zsig_(1,i) = Xsig_pred_(1,i);
  }

  //mean predicted measurement
  z_pred_ = Zsig_ * weights_;

  //measurement covariance matrix S
  S_.fill(0.0);
  for (auto i = 0; i < n_sig_; i++) {
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    S_ += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S_ += R_laser_;
}

void UKF::UpdateState(int n_z, bool RADER) {
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);

  for (auto i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    if (RADER) z_diff(1) = NormalizeAngle(z_diff(1));
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    if (RADER) x_diff(3) = NormalizeAngle(x_diff(3));
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  //residual
  VectorXd z_diff = z_ - z_pred_;

  if (RADER) z_diff(1) = NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K*S_*K.transpose();

  // Calculate NIS
  double NIS = z_diff.transpose() * S_.inverse() * z_diff;
  if (RADER) NIS_radar_ = NIS; else NIS_laser_ = NIS;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  z_ = VectorXd(2);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  PredictLidarMeasurement();
  UpdateState(2, false);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  z_ = VectorXd(3);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];
  PredictRadarMeasurement();
  UpdateState(3, true);
}