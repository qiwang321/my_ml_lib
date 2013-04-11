#include <pthread.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <string.h>
#include "util.h"
#include "global.h"
using namespace std;

// global variables
extern float* sample_mem[NUM_LAYER + 1]; //space storing the MCMC samples
extern float* weights[NUM_LAYER + 1]; //space storing the updating weights (first is not used)
extern float* bh[NUM_LAYER + 1]; // hidden layer biases (rbm)
extern float* bv[NUM_LAYER + 1]; // visible layer biases (rbm)
extern int nodes_layer[NUM_LAYER + 1]; // number of nodes in each layer

extern pthread_mutex_t mutex_data[NUM_LAYER + 1];
// extern pthread_mutex_t mutex_print;

extern time_t time_start1; //starter of timer

extern float yita_w, yita_bv, yita_bh, mu, reg, yita_wt, yita_bvt, yita_bht; // learning rate

extern float* data0, * data1, * data2;
extern long len0, len1, len2; // number of records in each data memory



// work done by each thread (read from memory version)
void *work1(void *a) {
	//implementation of the CD algorithm
	int layer_ind = ((arg*)a)->layer; //identify the layer index

	if (layer_ind == 0) { // input layer
		int offset;

		while (difftime(time(NULL), time_start1) <= TRAIN_TIME) {
			if ((float) rand()/RAND_MAX < 1.0 / 3) {
				offset = (rand() % len0) * nodes_layer[0];
				pthread_mutex_lock(&mutex_data[0]);
				{
					for (int j = 0; j < nodes_layer[0]; j++)
						sample_mem[0][j] = (float) data0[offset + j];
				}
				pthread_mutex_unlock(&mutex_data[0]);
			}
			else if ((float) rand()/RAND_MAX > 2.0 / 3) {
				offset = (rand() % len1) * nodes_layer[0];
				pthread_mutex_lock(&mutex_data[0]);
				{
					for (int j = 0; j < nodes_layer[0]; j++)
						sample_mem[0][j] = (float) data1[offset + j];
				}
				pthread_mutex_unlock(&mutex_data[0]);
			}
			else {
				offset = (rand() % len2) * nodes_layer[0];

				pthread_mutex_lock(&mutex_data[0]);
				{
					for (int j = 0; j < nodes_layer[0]; j++)
						sample_mem[0][j] = (float) data2[offset + j];
				}
				pthread_mutex_unlock(&mutex_data[0]);
			}


			// print the layer input data (just for testing)

		}
	}

	else if (layer_ind != NUM_LAYER) { // normal layer
		float* x0 = (float*) malloc(nodes_layer[layer_ind - 1] * sizeof(float)); // data
		float* h0 = (float*) malloc(nodes_layer[layer_ind] * sizeof(float));  // hidden
		float* x1 = (float*) malloc(nodes_layer[layer_ind - 1] * sizeof(float));
		float* h1 = (float*) malloc(nodes_layer[layer_ind] * sizeof(float));
		float* inc_w = (float *)  malloc(nodes_layer[layer_ind-1]
		                                             * nodes_layer[layer_ind] * sizeof(float)); // previous increase of weights
		float* inc_bv = (float *)  malloc(nodes_layer[layer_ind-1] * sizeof(float));
		float* inc_bh = (float *)  malloc(nodes_layer[layer_ind] * sizeof(float));
		memset(inc_w, 0, nodes_layer[layer_ind-1] * nodes_layer[layer_ind] * sizeof(float));
		memset(inc_bv, 0, nodes_layer[layer_ind-1] * sizeof(float));
		memset(inc_bh, 0, nodes_layer[layer_ind] * sizeof(float));

		while (difftime(time(NULL), time_start1) <= TRAIN_TIME) {

			//copy data
			pthread_mutex_lock(&mutex_data[layer_ind - 1]);
			{
				for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
					x0[i] = sample_mem[layer_ind - 1][i];
			}
			pthread_mutex_unlock(&mutex_data[layer_ind - 1]);

			//perform real computation
			sigm(h0, bh[layer_ind], weights[layer_ind], x0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

			//write data
			pthread_mutex_lock(&mutex_data[layer_ind]);
			{
				for (int j = 0; j < nodes_layer[layer_ind]; j++)
					sample_mem[layer_ind][j] = h0[j];
			}
			pthread_mutex_unlock(&mutex_data[layer_ind]);

			for (int i = 0; i < nodes_layer[layer_ind]; i++) {
				if ((float)rand()/RAND_MAX < h0[i])
					h0[i] = 1;
				else
					h0[i] = 0;
			}


			sigm(x1, bv[layer_ind], weights[layer_ind], h0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

			sigm(h1, bh[layer_ind], weights[layer_ind], x1,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
					inc_w[j*nodes_layer[layer_ind-1] + i] = mu * inc_w[j*nodes_layer[layer_ind-1] + i]
					                                                   + yita_w * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
					weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
							weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
							                   +inc_w[j*nodes_layer[layer_ind-1] + i];
				}

			for (int j = 0; j < nodes_layer[layer_ind]; j++) {
				inc_bh[j] = mu * inc_bh[j] + yita_bh*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
				bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[j];
			}

			for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
				inc_bv[i] = mu * inc_bv[i] + yita_bv*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
				bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[i];
			}


			// print the layer input data (just for testing)
		}
	}

	else { // top layer
		float* x0 = (float*) malloc(nodes_layer[layer_ind - 1] * sizeof(float)); // data
		float* h0 = (float*) malloc(nodes_layer[layer_ind] * sizeof(float));  // hidden
		float* x1 = (float*) malloc(nodes_layer[layer_ind - 1] * sizeof(float));
		float* h1 = (float*) malloc(nodes_layer[layer_ind] * sizeof(float));
		float* inc_w = (float *)  malloc(nodes_layer[layer_ind-1]
		                                             * nodes_layer[layer_ind] * sizeof(float)); // previous increase of weights
		float* inc_bv = (float *)  malloc(nodes_layer[layer_ind-1] * sizeof(float));
		float* inc_bh = (float *)  malloc(nodes_layer[layer_ind] * sizeof(float));
		memset(inc_w, 0, nodes_layer[layer_ind-1] * nodes_layer[layer_ind] * sizeof(float));
		memset(inc_bv, 0, nodes_layer[layer_ind-1] * sizeof(float));
		memset(inc_bh, 0, nodes_layer[layer_ind] * sizeof(float));

		while (difftime(time(NULL), time_start1) <= TRAIN_TIME) {

			//copy data
			pthread_mutex_lock(&mutex_data[layer_ind - 1]);
			{
				for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
					x0[i] = sample_mem[layer_ind - 1][i];
			}
			pthread_mutex_unlock(&mutex_data[layer_ind - 1]);

			//perform real computation
			for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
				h0[j] = bh[NUM_LAYER][j];
				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
					h0[j] = h0[j] + weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * x0[i];
			}

			//write data
			pthread_mutex_lock(&mutex_data[layer_ind]);
			{
				for (int j = 0; j < nodes_layer[layer_ind]; j++)
					sample_mem[layer_ind][j] = h0[j];
			}
			pthread_mutex_unlock(&mutex_data[layer_ind]);


			sigm(x1, bv[layer_ind], weights[NUM_LAYER], h0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

			for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
				h1[j] = bh[NUM_LAYER][j];
				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
					h1[j] = h1[j] + weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * x1[i];
			}

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
					inc_w[j*nodes_layer[layer_ind-1] + i] = mu * inc_w[j*nodes_layer[layer_ind-1] + i]
					                                                   + yita_wt * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
					weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
							weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
							                   +inc_w[j*nodes_layer[layer_ind-1] + i];
				}

			for (int j = 0; j < nodes_layer[layer_ind]; j++) {
				inc_bh[j] = mu * inc_bh[j] + yita_bht*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
				bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[j];
			}

			for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
				inc_bv[i] = mu * inc_bv[i] + yita_bvt*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
				bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[i];
			}


			// print the layer input data (just for testing)
		}
	}

	//free(x0); free(x1); free(h0); free(h1);
	pthread_exit((void*) a);
}

