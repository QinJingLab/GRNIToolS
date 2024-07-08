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

#include <cstdlib>
#include "regression_stump.h"

const Prediction test_regression_stump(const Model &m, const int N_test,
		const double *x, const double *y, const int M_test) {
	/*
	 * allocate memory for arrays ...
	 */

	double *prediction = (double *) calloc(N_test, sizeof(double));

	/*
	 * ... and scalars ...
	 */
	int tree;
	int row;
	double total_loss;

	/*
	 * ... and result
	 */
	Prediction r(M_test, N_test);

	/*
	 * Initialize prediction with f0
	 */
	for (row = 0; row < N_test; row++) {
		prediction[row] = m.getF0();
	}
	/*
	 * update the prediction for prescribed number of trees
	 */

	for (tree = 0; tree < M_test; tree++) {
		total_loss = 0;
		for (row = 0; row < N_test; row++) {
			if (x[m.getFeatSplitI(tree) * N_test + row]
					< m.getFeatSplitT(tree)) {
				prediction[row] += m.getNu() * m.getGammaL(tree);
			} else {
				prediction[row] += m.getNu() * m.getGammaR(tree);
			}
			total_loss += (y[row] - prediction[row])
					* (y[row] - prediction[row]);
		}
		r.setLoss(tree, total_loss);
	}
	for (row = 0; row < N_test; row++) {
		r.setPrediction(row, prediction[row]);
	}

	free(prediction);

	return r;
}
