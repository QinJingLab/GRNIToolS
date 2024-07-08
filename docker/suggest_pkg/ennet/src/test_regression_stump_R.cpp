/*
 *   This is an implementation of ENNET algorithm for Gene Regulatory Network
 *   inference from mRNA expression data, in form of an R package.
 *   Copyright (C) 2013  Janusz Slawek
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program, see LICENSE.
 */

#include "regression_stump.h"

extern "C" {

/*
 * This is a simple R interface
 * input variables are passed as constant pointers
 * results are returned as non-constant pointers
 */
void test_regression_stump_R(const int *N_test, const int *P_test,
		const int *P_train, const double *x_test, const double *y_test,
		const int *M_test, const int *M_train, const double *nu,
		const double *f0, const int *featI, const double * featT,
		const double *gamma_l, const double *gamma_r, double *loss, double *p) {

	/*
	 * R use:
	 * int*           - as.logical()
	 * int*           - as.integer()
	 * double*        - as.double()
	 * Rcomplex*      - as.complex()
	 * char**         - as.character()
	 * unsigned char* - as.raw()
	 */

	/*
	 * First step is to recreate Model from raw arrays
	 */
	Model m(*M_train, *P_test);
	m.setNu(*nu);
	m.setF0(*f0);
	for (int i = 0; i < *M_train; i++) {
		m.setFeatSplitI(i, featI[i]);
		m.setFeatSplitT(i, featT[i]);
		m.setGammaL(i, gamma_l[i]);
		m.setGammaR(i, gamma_r[i]);
		//leave importance blank
	}

	/*
	 * Now calculate a prediction
	 */
	Prediction prediction = test_regression_stump(m, *N_test, x_test, y_test,
			*M_test);

	for (int i = 0; i < *M_test; i++) {
		loss[i] = prediction.getLoss(i);
	}
	for (int i = 0; i < *N_test; i++) {
		p[i] = prediction.getPrediction(i);
	}
}

}

