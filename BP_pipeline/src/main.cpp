/*
 * main.cpp
 *
 *  Created on: Mar 28, 2013
 *      Author: qwang37
 */

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <limits>
#include "global.h"
#include "pipe_utils.h"
#include "utils.h"

using namespace std;

stage_t stages[NUM_LAYER+1];

pthread_t threads_f[NUM_LAYER+1];
pthread_t threads_b[NUM_LAYER+1];

float *weights[NUM_LAYER+1]; // weights at each layer
float *b[NUM_LAYER+1]; // biase values at each layer
float *inc_w[NUM_LAYER+1]; // weights at each layer
float *inc_b[NUM_LAYER+1]; // weights at each layer
int layer_nodes[NUM_LAYER+1]; // number of nodes at each layer
FIFO *layer_data[NUM_LAYER+1]; // updata queue at each layer
float *layer_delta[NUM_LAYER+1]; // downdata at each layer
float *output; // activation at the output layer
float *layer_label[NUM_LAYER+1]; // label data
bool layer_flag_f[NUM_LAYER+1]; // signal to stop threads running
bool layer_flag_b[NUM_LAYER+1];

float yita_w = 0.5, yita_b = 0.5, mu = 0.0, reg = 0.0000;
/*
typedef struct pipe { //pipeline
	pthread_mutex_t m;
	stage_t *head, *tail; //first/last stage of the pipeline
	int stages; // number of stages of the pipeline
	//int active; // number of active data elements in the pipeline
} pipe_t;
 */

//const int N_STAGES = 10;

int main (int argc, char *argv[]) {
	// create stages+1 stages
	int dim = 2, dim_t = 1;
	layer_nodes[0] = dim;
	layer_nodes[1] = 100;
	layer_nodes[2] = dim_t;
	//layer_nodes[3] = 1;

	int i, j, k, t;

	for (k = 1; k <= NUM_LAYER; k++) {/*
		pthread_mutex_init(&stages[k].m_f, NULL);
		pthread_mutex_init(&stages[k].m_b, NULL);
		pthread_mutex_init(&stages[k].m_data, NULL);
		pthread_mutex_init(&stages[k].m_delta, NULL);
		pthread_cond_init(&stages[k].avail_f, NULL);
		pthread_cond_init(&stages[k].avail_b, NULL);
		pthread_cond_init(&stages[k].ready_f, NULL);
		pthread_cond_init(&stages[k].ready_b, NULL);
	 */
		stages[k].data_ready_b = false;
		stages[k].data_ready_f = false;

		// initialize weights, biases and flag variables
		weights[k] = (float*) malloc (layer_nodes[k-1] * layer_nodes[k] * sizeof(float));
		inc_w[k] = (float*) malloc (layer_nodes[k-1] * layer_nodes[k] * sizeof(float));
		for (i = 0; i < layer_nodes[k-1] * layer_nodes[k]; i++) {
			weights[k][i] = 1 * ((float)rand() / RAND_MAX * 2 - 1);
			inc_w[k][i] = 1 * ((float)rand() / RAND_MAX * 2 - 1);
		}

		b[k] = (float*) malloc (layer_nodes[k] * sizeof(float));
		inc_b[k] = (float*) malloc (layer_nodes[k] * sizeof(float));
		for (i = 0; i < layer_nodes[k]; i++) {
			b[k][i] = 1 * ((float)rand() / RAND_MAX * 2 - 1);
			inc_b[k][i] = 1 * ((float)rand() / RAND_MAX * 2 - 1);
		}

		layer_flag_f[k] = false;
		layer_flag_b[k] = false;

		// initialize the layer data storage
		layer_data[k] = new FIFO(2 * (NUM_LAYER-k+1));
		//layer_delta[k] = (float*) malloc (k * sizeof(float)); // no need to allocate memory since the layer delta only stores a pointer
	}

	//create two threads for each stage
	layer_arg *arg;
	for (k = 1; k <= NUM_LAYER; k++) {
		arg = new layer_arg;
		arg->layer_ind = k;
		arg->f = true;
		pthread_create(&threads_f[k], NULL, pipe_stage, (void*)arg);

		arg = new layer_arg;
		arg->layer_ind = k;
		arg->f = false;
		pthread_create(&threads_b[k], NULL, pipe_stage, (void*)arg);
	}

	// data
	int train_num = 4;
	float train_data[train_num][dim];
	float train_labels[train_num][dim_t];
	float dmax[dim], dmin[dim]; //for normalization
	float tmax[dim_t], tmin[dim_t];

	for (j = 0; j < dim; j++) {
		dmax[j] = std::numeric_limits<float>::min();
		dmin[j] = std::numeric_limits<float>::max();
	}
	for (j = 0; j < dim_t; j++) {
		tmax[j] = std::numeric_limits<float>::min();
		tmin[j] = std::numeric_limits<float>::max();
	}


	ifstream train_d("xor");
	ifstream train_t("xor_target");

	for (t = 0; t < train_num; t++) {
		for (j = 0; j < dim; j++) {
			train_d >> train_data[t][j];
			if (train_data[t][j] > dmax[j])
				dmax[j] = train_data[t][j];
			if (train_data[t][j] < dmin[j])
				dmin[j] = train_data[t][j];
		}
		for (j = 0; j < dim_t; j++) {
			train_t >> train_labels[t][j];
			if (train_labels[t][j] > tmax[j])
				tmax[j] = train_labels[t][j];
			if (train_labels[t][j] < tmin[j])
				tmin[j] = train_labels[t][j];
		}
	}
	train_d.close();
	train_t.close();

	/*
	// normalization
	for (t = 0; t < train_num; t++) {
		for (j = 0; j < dim; j++)
			train_data[t][j] = (train_data[t][j] - dmin[j]) / (dmax[j] - dmin[j]);
		for (j = 0; j < dim_t; j++)
			train_labels[t][j] = (train_labels[t][j] - tmin[j]) / (tmax[j] - tmin[j]);
	}
	*/

	// feed data
	for (int epoch = 0; epoch < 2000; epoch++) {
		for (t = 0; t < train_num; t++) {
			pipe_send_f(1, train_data[t], train_labels[t], false);
			cout << "fed data " << t+1 << endl;
		}
	}

	// send shutdown signal
	pipe_send_f(1, NULL, NULL, true);

	cout << "training threads shut down!\n";


	// test phase
	int test_num = 4;
	float test_data[test_num][dim];
	float test_labels[test_num][dim_t];
	ifstream test_d("xor");
	ifstream test_t("xor_target");

	// read and normalization
	for (t = 0; t < test_num; t++) {
		for (j = 0; j < dim; j++) {
			test_d >> test_data[t][j];
			//test_data[t][j] = (test_data[t][j] - dmin[j]) / (dmax[j] - dmin[j]);
		}
		for (j = 0; j < dim_t; j++) {
			test_t >> test_labels[t][j];
			//test_labels[t][j] = (test_labels[t][j] - tmin[j]) / (tmax[j] - tmin[j]);
		}
	}
	test_d.close();
	test_t.close();

	ofstream out("xor_out");
	out << "test_out\ttarget\n";

	float *test_out[test_num];
	float *act[NUM_LAYER];
	for (k = 1; k < NUM_LAYER; k++)
		act[k] = (float*) malloc (layer_nodes[k] * sizeof(float));
	for (t = 0; t < test_num; t++) {
		act[0] = test_data[t];
		for (k = 1; k < NUM_LAYER; k++) {
			sigm(act[k], b[k], weights[k], act[k-1], layer_nodes[k], layer_nodes[k-1]);
		}
		test_out[t] = (float*) malloc (layer_nodes[NUM_LAYER] * sizeof(float));
		sigm(test_out[t], b[NUM_LAYER], weights[NUM_LAYER], act[NUM_LAYER-1],
				layer_nodes[NUM_LAYER], layer_nodes[NUM_LAYER-1]);
		for (j = 0; j < layer_nodes[NUM_LAYER]; j++)
			out << test_out[t][j] << " ";
		out << "\t";
		for (j = 0; j < layer_nodes[NUM_LAYER]; j++)
			out << test_labels[t][j] << " ";
		out << "\n";
	}
	out.close();

	cout << "test done!" << endl;
}
