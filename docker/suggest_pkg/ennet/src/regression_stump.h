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

#ifndef REGRESSION_STUMP_H_
#define REGRESSION_STUMP_H_

#include "Model.h"
#include "Prediction.h"

const Model train_regression_stump(const int N_train, const int P_train,
		const double *x, const double *y, const double col_sampling_rate,
		const double row_sampling_rate, const int M_train, const double nu);

const Prediction test_regression_stump(const Model &m, const int N_test,
		const double *x, const double *y, const int M_test);

#endif /* REGRESSION_STUMP_H_ */
