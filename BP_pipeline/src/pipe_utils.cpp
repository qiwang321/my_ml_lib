/*
 * pipe_utils.cpp
 *
 *  Created on: Mar 30, 2013
 *      Author: qiwang321
 */
#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include "global.h"
#include "utils.h"

using namespace std;

extern stage_t stages[NUM_LAYER+1];

extern float *weights[NUM_LAYER+1]; // weights at each layer
extern float *b[NUM_LAYER+1]; // bias values at each layer
extern float *inc_w[NUM_LAYER+1]; // weights at each layer
extern float *inc_b[NUM_LAYER+1]; // weights at each layer
extern int layer_nodes[NUM_LAYER+1]; // number of nodes at each layer
extern FIFO *layer_data[NUM_LAYER+1]; // updata queue at each layer
extern float *layer_delta[NUM_LAYER+1]; // downdata at each layer
extern float *layer_label[NUM_LAYER+1]; // label data
extern bool layer_flag_f[NUM_LAYER+1]; // signal to stop threads running
extern bool layer_flag_b[NUM_LAYER+1];

extern float yita_w, yita_b, mu, reg;
pthread_mutex_t m_print;

void pipe_send_f(int layer_ind, float *data, float *label, bool flag) {
	//parameter: targe layer, passed data and label
	pthread_mutex_lock(&stages[layer_ind].m_f);
	{
		while (stages[layer_ind].data_ready_f)
			pthread_cond_wait(&stages[layer_ind].ready_f, &stages[layer_ind].m_f);
		// copy data
		pthread_mutex_lock(&stages[layer_ind].m_data);
		{
			layer_data[layer_ind]->enqueue(new element(data));
			layer_label[layer_ind] = label;
			layer_flag_f[layer_ind] = flag;
		}
		pthread_mutex_unlock(&stages[layer_ind].m_data);

		pthread_mutex_lock(&m_print);
		{
			cout << "layer " << layer_ind << " received forward data" << endl;
		}
		pthread_mutex_unlock(&m_print);


		stages[layer_ind].data_ready_f = true;
		pthread_cond_signal(&stages[layer_ind].avail_f);
	}
	pthread_mutex_unlock(&stages[layer_ind].m_f);

}

void pipe_send_b(int layer_ind, float *delta, bool flag) {
	//parameter: targe layer, passed delta
	pthread_mutex_lock(&stages[layer_ind].m_b);
	{
		while (stages[layer_ind].data_ready_b)
			pthread_cond_wait(&stages[layer_ind].ready_b, &stages[layer_ind].m_b);
		// copy data
		pthread_mutex_lock(&stages[layer_ind].m_delta);
		{
			free(layer_delta[layer_ind]);
			layer_delta[layer_ind] = delta;
			layer_flag_b[layer_ind] = flag;
		}
		pthread_mutex_unlock(&stages[layer_ind].m_delta);

		pthread_mutex_lock(&m_print);
		{
			cout << "layer " << layer_ind << " received backward data" << endl;
		}
		pthread_mutex_unlock(&m_print);

		stages[layer_ind].data_ready_b = true;
		pthread_cond_signal(&stages[layer_ind].avail_b);
	}
	pthread_mutex_unlock(&stages[layer_ind].m_b);

}


void *pipe_stage(void *arg) {
	int layer_ind = ((layer_arg*) arg)->layer_ind;
	bool f = ((layer_arg*) arg)->f; // forward?
	if (f) {
		if (layer_ind != NUM_LAYER) {
			for ( ; ; ) {
				pthread_mutex_lock(&stages[layer_ind].m_f);
				{
					while (!stages[layer_ind].data_ready_f) { //wait for data
						pthread_cond_wait(&stages[layer_ind].avail_f, &stages[layer_ind].m_f);
					}

					// the shutdown signal
					if (layer_flag_f[layer_ind]) {
						pipe_send_f(layer_ind + 1, NULL, NULL, true);
						cout << "layer " << layer_ind << " forward thread exited" << endl;
						pthread_exit(arg);
					}

					// process data element and forward to next stage :
					float *act = (float*) malloc (layer_nodes[layer_ind] * sizeof(float)); // buffer for activation output for each layer
					pthread_mutex_lock(&stages[layer_ind].m_data); // lock the current layer data for computation
					{
						sigm(act, b[layer_ind], weights[layer_ind], layer_data[layer_ind]->head->data, layer_nodes[layer_ind], layer_nodes[layer_ind-1]);
					}
					pthread_mutex_unlock(&stages[layer_ind].m_data);

					pthread_mutex_lock(&m_print);
					{
						cout << "layer " << layer_ind << " sent forward data" << endl;
					}
					pthread_mutex_unlock(&m_print);

					pipe_send_f(layer_ind + 1, act, layer_label[layer_ind], false);
					stages[layer_ind].data_ready_f = false;
					pthread_cond_signal(&stages[layer_ind].ready_f);
				}
				pthread_mutex_unlock(&stages[layer_ind].m_f);
			}
		}

		else { // deal with the top layer
			for ( ; ; ) {
				pthread_mutex_lock(&stages[layer_ind].m_f);
				{
					while (!stages[NUM_LAYER].data_ready_f) { //wait for data
						pthread_cond_wait(&stages[NUM_LAYER].avail_f, &stages[NUM_LAYER].m_f);
					}

					// the shutdown signal
					if (layer_flag_f[NUM_LAYER]) {
						pipe_send_b(NUM_LAYER, NULL, true);
						cout << "layer " << layer_ind << " forward thread exited" << endl;
						pthread_exit(arg);
					}

					// process data element and forward to next stage :
					float *act = (float*) malloc (layer_nodes[NUM_LAYER] * sizeof(float)); // buffer for activation output for each layer
					pthread_mutex_lock(&stages[NUM_LAYER].m_data); // lock the current layer data for computation
					{
						sigm(act, b[NUM_LAYER], weights[NUM_LAYER], layer_data[NUM_LAYER]->head->data, layer_nodes[NUM_LAYER], layer_nodes[NUM_LAYER-1]);
					}
					pthread_mutex_unlock(&stages[NUM_LAYER].m_data);

					pthread_mutex_lock(&stages[NUM_LAYER].m_delta);
					{
						for (int j = 0; j < layer_nodes[NUM_LAYER]; j++)
							act[j] = (layer_label[NUM_LAYER][j] - act[j]) * act[j] * (1 - act[j]);
					}
					pthread_mutex_unlock(&stages[NUM_LAYER].m_delta);

					pthread_mutex_lock(&m_print);
					{
						cout << "layer " << layer_ind << " sent backward data" << endl;
					}
					pthread_mutex_unlock(&m_print);

					pipe_send_b(NUM_LAYER, act, false);
					stages[layer_ind].data_ready_f = false;
					pthread_cond_signal(&stages[layer_ind].ready_f);
				}
				pthread_mutex_unlock(&stages[layer_ind].m_f);
			}
		}
	}

	else {
		if (layer_ind > 1) {
			for ( ; ; ) {
				pthread_mutex_lock(&stages[layer_ind].m_b);
				{
					while (!stages[layer_ind].data_ready_b) { //wait for data
						pthread_cond_wait(&stages[layer_ind].avail_b, &stages[layer_ind].m_b);
					}


					// shutdown signal
					if (layer_flag_b[layer_ind]) {
						pipe_send_b(layer_ind - 1, NULL, true);
						cout << "layer " << layer_ind << " backward thread exited" << endl;
						pthread_exit(arg);
					}

					// process data element and forward to next stage :
					float *delta = (float*) malloc (layer_nodes[layer_ind-1] * sizeof(float)); // buffer for delta value backpropagated to each layer
					pthread_mutex_lock(&stages[layer_ind].m_data);
					pthread_mutex_lock(&stages[layer_ind].m_delta); // lock the current layer data for computation
					{
						back_delta(delta, layer_delta[layer_ind], weights[layer_ind], layer_data[layer_ind]->tail->data,
								layer_nodes[layer_ind], layer_nodes[layer_ind-1]);
					}
					pthread_mutex_unlock(&stages[layer_ind].m_delta);
					pthread_mutex_unlock(&stages[layer_ind].m_data);

					pthread_mutex_lock(&m_print);
					{
						cout << "layer " << layer_ind << " sent backward data" << endl;
					}
					pthread_mutex_unlock(&m_print);


					//update weights of current layer
					pthread_mutex_lock(&stages[layer_ind].m_data);
					pthread_mutex_lock(&stages[layer_ind].m_delta);
					{
						for (int i = 0; i < layer_nodes[layer_ind-1]; i++)
							for (int j = 0; j < layer_nodes[layer_ind]; j++) {
								inc_w[layer_ind][j*layer_nodes[layer_ind-1]+i] = mu * inc_w[layer_ind][j*layer_nodes[layer_ind-1]+i] +
										yita_w * (layer_delta[layer_ind][j] * layer_data[layer_ind]->tail->data[i] -
												reg * weights[layer_ind][j*layer_nodes[layer_ind-1]+i]);
								weights[layer_ind][j*layer_nodes[layer_ind-1]+i] = weights[layer_ind][j*layer_nodes[layer_ind-1]+i] +
										inc_w[layer_ind][j*layer_nodes[layer_ind-1]+i];
							}
						for (int j = 0; j < layer_nodes[layer_ind]; j++) {
							inc_b[layer_ind][j] = mu * inc_b[layer_ind][j] + yita_b * (layer_delta[layer_ind][j] - reg * b[layer_ind][j]);
							b[layer_ind][j] = b[layer_ind][j] + inc_b[layer_ind][j];
						}
						//delete(layer_data[layer_ind]->dequeue()); // delete used data
						layer_data[layer_ind]->dequeue();
						pthread_mutex_lock(&m_print);
						cout << "called dequeue() at layer " << layer_ind << endl;
						pthread_mutex_unlock(&m_print);
					}
					pthread_mutex_unlock(&stages[layer_ind].m_delta);
					pthread_mutex_unlock(&stages[layer_ind].m_data);

					pipe_send_b(layer_ind - 1, delta, false);
					stages[layer_ind].data_ready_b = false;
					pthread_cond_signal(&stages[layer_ind].ready_b);
				}
				pthread_mutex_unlock(&stages[layer_ind].m_b);
			}
		}

		else {
			for ( ; ; ) {
				pthread_mutex_lock(&stages[layer_ind].m_b);
				{
					while (!stages[layer_ind].data_ready_b) { //wait for data
						pthread_cond_wait(&stages[layer_ind].avail_b, &stages[layer_ind].m_b);
					}




					// shutdown signal
					if (layer_flag_b[layer_ind]) {
						pipe_send_b(layer_ind - 1, NULL, true);
						cout << "layer " << layer_ind << " backward thread exited" << endl;
						pthread_exit(arg);
					}



					// just update weights of current layer
					pthread_mutex_lock(&stages[layer_ind].m_data);
					pthread_mutex_lock(&stages[layer_ind].m_delta);
					{
						for (int i = 0; i < layer_nodes[layer_ind-1]; i++)
							for (int j = 0; j < layer_nodes[layer_ind]; j++) {
								inc_w[layer_ind][j*layer_nodes[layer_ind-1]+i] = mu * inc_w[layer_ind][j*layer_nodes[layer_ind-1]+i] +
										yita_w * (layer_delta[layer_ind][j] * layer_data[layer_ind]->tail->data[i] -
											reg * weights[layer_ind][j*layer_nodes[layer_ind-1]+i]);
								weights[layer_ind][j*layer_nodes[layer_ind-1]+i] = weights[layer_ind][j*layer_nodes[layer_ind-1]+i] +
										inc_w[layer_ind][j*layer_nodes[layer_ind-1]+i];
							}
						for (int j = 0; j < layer_nodes[layer_ind]; j++) {
							inc_b[layer_ind][j] = mu * inc_b[layer_ind][j] + yita_b * (layer_delta[layer_ind][j] - reg * b[layer_ind][j]);
							b[layer_ind][j] = b[layer_ind][j] + inc_b[layer_ind][j];
						}
						//delete(layer_data[layer_ind]->dequeue()); // delete used data
						layer_data[layer_ind]->dequeue();
						pthread_mutex_lock(&m_print);
						cout << "called dequeue() at layer " << layer_ind << endl;
						pthread_mutex_unlock(&m_print);
					}
					pthread_mutex_unlock(&stages[layer_ind].m_delta);
					pthread_mutex_unlock(&stages[layer_ind].m_data);

					stages[layer_ind].data_ready_b = false;
					pthread_cond_signal(&stages[layer_ind].ready_b);

				}
				pthread_mutex_unlock(&stages[layer_ind].m_b);
			}
		}
	}
}









