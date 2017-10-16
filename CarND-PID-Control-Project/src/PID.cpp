#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
  sum_cte = prev_cte = 0;
}

void PID::UpdateError(double cte) {
  sum_cte += cte;
  p_error = - Kp * cte;
  i_error = - Ki * sum_cte;
  d_error = - Kd * (cte - prev_cte);
  prev_cte = cte;
}

double PID::TotalError() {
  return p_error + i_error + d_error;
}
