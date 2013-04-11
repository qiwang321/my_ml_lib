/*
 * global.h
 *
 *  Created on: Mar 19, 2013
 *      Author: qiwang321
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

// global parameters
#define NUM_LAYER  4
#define NODES_INPUT  784
#define TRAIN_TIME 80.0

typedef struct {
	int layer; //specify the layer of a thread
} arg;

#endif /* GLOBAL_H_ */
