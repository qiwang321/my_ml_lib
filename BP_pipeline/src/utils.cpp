/*
 * utils.cpp
 *
 *  Created on: Mar 31, 2013
 *      Author: qwang37
 */
#include <math.h>

//compute the sigmoid of a layer
void sigm(float* res, float* b, float* W, float* x, int n, int m) {
	for (int j = 0; j < n; j++) {
		res[j] = -b[j];
		for (int i = 0; i < m; i++)
			res[j] = res[j] - W[j*m + i] * x[i];
		res[j] = 1 / (1 + exp(res[j]));
	}
}

void back_delta(float *res, float *delta_u, float *W, float *act, int n, int m) {
	for (int i = 0; i < m; i++) {
		res[i] = 0;
		for (int j = 0; j < n; j++)
			res[i] = res[i] + delta_u[j] * W[j*m + i];
		res[i] = res[i] * act[i] * (1.0 - act[i]);
	}
}


