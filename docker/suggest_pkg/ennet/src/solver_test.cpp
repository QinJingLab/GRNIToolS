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

#include <iostream>
#include <cstdlib>
#include "regression_stump.h"

#define N 100
#define P 5

using namespace std;

int main() {

	// dummy test
	double x[N * P];
	for (int i = 0; i < N * P; i++) {
		x[i] = rand() % 1000;
	}
	double y[N];
	for (int i = 0; i < N; i++) {
		y[i] = rand() % 1000;
	}

	train_regression_stump(N, P, x, y, 0.5, 0.5, 1, 0.01);
	cout << "done" << endl;
}
