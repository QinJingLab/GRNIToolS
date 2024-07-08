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
void train_regression_stump_R(const int *N_train, const int *P_test,
		const double *x_train, const double *y_train, const double *s_f,
		const double *s_s, const int *M_train, const double *nu, double *I,
		double *f0, int *featI, double * featT, double *gamma_l,
		double *gamma_r) {

	/*
	 * R use:
	 * int*           - as.logical()
	 * int*           - as.integer()
	 * double*        - as.double()
	 * Rcomplex*      - as.complex()
	 * char**         - as.character()
	 * unsigned char* - as.raw()
	 */

	Model m = train_regression_stump(*N_train, *P_test, x_train, y_train, *s_f,
			*s_s, *M_train, *nu);
	for (int i = 0; i < *P_test; i++) {
		I[i] = m.getImportance(i);
	}
	*f0 = m.getF0();
	for (int i = 0; i < *M_train; i++) {
		featI[i] = (int) m.getFeatSplitI(i);
		featT[i] = m.getFeatSplitT(i);
		gamma_l[i] = m.getGammaL(i);
		gamma_r[i] = m.getGammaR(i);
	}
}

}
