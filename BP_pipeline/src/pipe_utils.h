/*
 * pipe_utils.h
 *
 *  Created on: Mar 30, 2013
 *      Author: qiwang321
 */

#ifndef PIPE_UTILS_H_
#define PIPE_UTILS_H_

void pipe_send_f(int layer_ind, float *data, float *label, bool flag);
void pipe_send_b(int layer_ind, float *delta, bool flag);
void *pipe_stage(void *arg);

#endif /* PIPE_UTILS_H_ */
