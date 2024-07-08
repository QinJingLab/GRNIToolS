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

#ifndef PREDICTION_H_
#define PREDICTION_H_

#include <vector>

using namespace std;

class Prediction {
public:
	Prediction(size_t M, size_t N) {
		this->M = M;
		this->N = N;
		loss.resize(M);
		prediction.resize(N);
	}

	virtual ~Prediction() {
	}

	size_t getM() const {
		return M;
	}

	const double getLoss(size_t tree) const {
		return loss[tree];
	}

	void setLoss(size_t tree, double l) {
		loss[tree] = l;
	}

	const double getPrediction(size_t row) const {
		return prediction[row];
	}

	void setPrediction(size_t row, double p) {
		prediction[row] = p;
	}

private:
	size_t M;
	size_t N;
	vector<double> prediction;
	vector<double> loss;
};

#endif /* PREDICTION_H_ */
