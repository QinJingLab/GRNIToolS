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
#include <cmath>
#include <ctime>
#include <cfloat>
#include "regression_stump.h"

//#include <R.h>

inline int compare(const void *a, const void *b) {
	double result = **(double **) a - **(double**) b;
	if (result > 0)
		return 1;
	if (result < 0)
		return -1;
	return 0;
}

const Model train_regression_stump(const int N, const int P, const double *x,
		const double *y, const double col_sampling_rate,
		const double row_sampling_rate, const int M, const double nu) {
	/*
	 * reset random seed
	 */
	srand(time(NULL));

	/*
	 * allocate memory for arrays ...
	 */
	const double **x_to_sort = (const double **) calloc(N,
			sizeof(const double*));
	int *x_sorted_index = (int *) calloc(N * P, sizeof(int));
	double *F = (double*) calloc(N, sizeof(double));
	double *h = (double *) calloc(N, sizeof(double));
	double *r = (double *) calloc(N, sizeof(double));
	bool *col_w = (bool *) calloc(P, sizeof(bool));
	double *row_w = (double *) calloc(N, sizeof(double));
	int *active_sorted_rows = (int *) calloc(N, sizeof(int));
	double *I = (double *) calloc(P, sizeof(double));

	/*
	 * ... and scalars ...
	 */
	int P_unique_inbag;
	int P_already_bagged;
	int N_of_selections;
	int N_active_rows;
	double total_w;
	int row;
	int next_row;
	int col;
	int iteration;
	int s;
	int s_i;

	double local_importance;
	double local_threshold;
	double best_inbag_importance;
	double best_inbag_threshold = 0.0;
	int best_inbag_column = 0;
	double sum_r;
	double ib_w_l;
	double ib_w_r;
	double ib_sum_l;
	double ib_sum_r;
	double ib_gamma_l;
	double ib_gamma_r;
	double best_gamma_l;
	double best_gamma_r;
	const double * origin;
	double total_importance;

	/*
	 * ... and vectors for returned object
	 */
	Model result = Model(M, P);

	/*
	 * Initialize importance I with zeros
	 */
	for (col = 0; col < P; col++) {
		I[col] = 0;
	}

	/*
	 * get indices of sorted columns of x
	 */
	for (col = 0; col < P; col++) {
		for (row = 0; row < N; row++) {
			x_to_sort[row] = x + col * N + row;
		}
		origin = x_to_sort[0];
		qsort(x_to_sort, N, sizeof(double*), compare);
		for (row = 0; row < N; row++) {
			x_sorted_index[col * N + row] = x_to_sort[row] - origin;
		}
	}

	/*
	 * calculate initial prediction
	 * arithmetic mean of y
	 */
	double avg_y = 0;
	for (row = 0; row < N; row++) {
		avg_y += y[row];
	}
	avg_y /= N;
	for (row = 0; row < N; row++) {
		F[row] = avg_y;
	}

	/*
	 * Start the main part of the algorithm
	 */
	for (iteration = 0; iteration < M; iteration++) {
		/*
		 * calculate working response for all observations
		 * according to squared-error loss function
		 */
		for (row = 0; row < N; row++) {
			/*
			 * L(y,f) = 0.5*(f-y)^2
			 * -dL/df = y-f
			 */
			r[row] = y[row] - F[row];
		}

		/*
		 * select sample of columns without replacement
		 * no replicates
		 */
		if (col_sampling_rate > 0 && col_sampling_rate <= 1) {
			P_unique_inbag = ceil(col_sampling_rate * P);
			P_already_bagged = 0;
			for (s = 0; s < P; s++) {
				if ((1.0 * rand() / RAND_MAX) * (P - s)
						< P_unique_inbag - P_already_bagged) {
					col_w[s] = true;
					P_already_bagged++;
				} else {
					col_w[s] = false;
				}
				if (P_already_bagged >= P_unique_inbag) {
					s++;
					break;
				}
			}
			// the remainder is not in the bag
			for (; s < P; s++) {
				col_w[s] = false;
			}
		}

		/*
		 * select sample of rows with replacement
		 * replicates are marked with row_w = 2, 3, etc
		 */
		if (row_sampling_rate > 0) {
			N_of_selections = ceil(row_sampling_rate * N);
			for (s = 0; s < N; s++) {
				row_w[s] = 0.0;
			}
			for (s = 0; s < N_of_selections; s++) {
				row_w[rand() % N] += 1.0;
			}
		}

		/*
		 * Calculate INBAG sum of all pseudo-residuals
		 */
		sum_r = 0.0;
		total_w = 0.0;
		for (row = 0; row < N; row++) {
			sum_r += row_w[row] * r[row];
			total_w += row_w[row];
		}

		/*
		 * go through all the columns in sample
		 * begin the search
		 */

		best_inbag_importance = 0.0;
		best_gamma_l = 0.0;
		best_gamma_r = 0.0;

		for (col = 0; col < P; col++) {
			if (col_w[col]) {
				/*
				 * Which rows are active
				 * with respect to the order
				 * of x
				 *
				 * active_rows = rows_w[x_index]
				 *
				 */
				s_i = 0;
				for (s = 0; s < N; s++) {
					if (row_w[x_sorted_index[col * N + s]] > 0.0) {
						active_sorted_rows[s_i] = s;
						s_i++;
					}
				}
				N_active_rows = s_i;

				/*
				 * Initialize variables
				 */
				ib_sum_l = 0.0;
				ib_sum_r = sum_r;
				ib_w_l = 0.0;
				ib_w_r = total_w;

				/*
				 * go through all the thresholds
				 */
				for (s = 0; s < N_active_rows - 1; s++) {
					row = x_sorted_index[col * N + active_sorted_rows[s]];
					next_row = x_sorted_index[col * N
							+ active_sorted_rows[s + 1]];

					ib_sum_l += row_w[row] * r[row];
					ib_sum_r -= row_w[row] * r[row];
					ib_w_l += row_w[row];
					ib_w_r -= row_w[row];

					/*
					 * test if it's a valid split point
					 * OK: 1 2 | 3 4
					 * WRONG: 1 1 | 1 3
					 */
					if (x[col * N + row] < x[col * N + next_row]) {
						/*
						 * it's a valid split
						 */
						local_threshold = (x[col * N + row]
								+ x[col * N + next_row]) / 2.0;
						ib_gamma_l = ib_sum_l / ib_w_l;
						ib_gamma_r = ib_sum_r / ib_w_r;
						local_importance = (ib_w_l * ib_w_r) / total_w
								* (ib_gamma_l - ib_gamma_r)
								* (ib_gamma_l - ib_gamma_r);

						if (local_importance > best_inbag_importance) {
							best_inbag_column = col;
							best_inbag_importance = local_importance;
							best_inbag_threshold = local_threshold;
							best_gamma_l = ib_gamma_l;
							best_gamma_r = ib_gamma_r;
						}
					}
				}
			}
		}
		if (best_inbag_importance > 0.0) {
			/*
			 * Update relative influence
			 */
			I[best_inbag_column] += best_inbag_importance;

			/*
			 * update f using all observations
			 * nu * h(x)
			 */
			for (row = 0; row < N; row++) {
				if (x[best_inbag_column * N + row] < best_inbag_threshold) {
					F[row] += nu * best_gamma_l;
				} else {
					F[row] += nu * best_gamma_r;
				}
			}
			/*
			 * Prepare result
			 * part 1
			 */
			result.setFeatSplitI(iteration, best_inbag_column);
			result.setFeatSplitT(iteration, best_inbag_threshold);
			result.setGammaL(iteration, best_gamma_l);
			result.setGammaR(iteration, best_gamma_r);
		}
	}

	/*
	 * Scale importance
	 */
	total_importance = 0.0;
	for (col = 0; col < P; col++) {
		total_importance += I[col];
	}
	for (col = 0; col < P; col++) {
		I[col] = I[col] / total_importance;
	}

	/*
	 * Preapre result
	 * part 2
	 */
	result.setF0(avg_y);
	result.setNu(nu);
	for (col = 0; col < P; col++) {
		result.setImportance(col, I[col]);
	}

	/*
	 * Deallocate all variables allocated dynamically
	 */
	free(x_to_sort);
	free(x_sorted_index);
	free(F);
	free(h);
	free(r);
	free(col_w);
	free(row_w);
	free(active_sorted_rows);
	free(I);

	/*
	 * Return a copy of model
	 */
	return result;
}

