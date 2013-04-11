/*
 * utils.h
 *
 *  Created on: Mar 31, 2013
 *      Author: qwang37
 */

#ifndef UTILS_H_
#define UTILS_H_

void sigm(float* res, float* b, float* W, float* x, int n, int m);
void back_delta(float *res, float *delta_u, float *W, float *act, int n, int m);

#endif /* UTILS_H_ */
