/*
 * main.cpp
 *
 *  Created on: Mar 4, 2013
 *      Author: qiwang321
 */

#include "global.h"
#include "util.h"
#include "work.h"

#include <pthread.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
using namespace std;

// global variables
float* sample_mem[NUM_LAYER + 1]; //space storing the MCMC samples
float* b_sample_mem[2*NUM_LAYER + 1]; //layer sample for the back-propagation stage
float* weights[NUM_LAYER + 1]; //space storing the updating weights (first is not used)
float* bh[NUM_LAYER + 1]; // hidden layer biases (rbm)
float* bv[NUM_LAYER + 1]; // visible layer biases (rbm)
int nodes_layer[NUM_LAYER + 1]; // number of nodes in each layer

float* b_weights[2*NUM_LAYER + 1]; // weights at back-propagation stage
float* b_b[2*NUM_LAYER + 1]; // biases at back-propagation stage

time_t time_start1; //starter of timer

float yita_w = 0.05, yita_bv = 0.05, yita_bh = 0.05,
		yita_wt = 5e-4, yita_bvt = 5e-4, yita_bht = 5e-4; // learning rates
float mu = 0.5, reg = 0.0002;

pthread_mutex_t mutex_data[NUM_LAYER + 1];
//pthread_mutex_t mutex_print;

float* data0, * data1, * data2, * data3; // data memory

long len0, len1, len2, len3; // number of records in each data memory

arg layer_arg[NUM_LAYER + 1];

int train_len[10] = {5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949};
int test_len[10] = {980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009};

int main() {
	srand(time(NULL));

	//ifstream in0("data0"); // data source
	//ifstream in1("data1");
	const char* file0 = "data/train0";
	const char* file1 = "data/train1";
	const char* file2 = "data/train2";
	const char* file3 = "data/train3";

	len0 = train_len[0];
	len1 = train_len[1];
	len2 = train_len[2];
	len3 = train_len[3];
	data0 = read_data_si2(file0, len0);
	data1 = read_data_si2(file1, len1);
	data2 = read_data_si2(file2, len2);
	data3 = read_data_si2(file3, len3);
	//in0.close();
	//in1.close();

	void* status;

	//initialize number of nodes in each layer
	//for (int k = 1; k < NUM_LAYER; k++) {
	//nodes_layer[k] = 100;
	//}
	nodes_layer[1]=1000;
	nodes_layer[2]=500;
	nodes_layer[3]=250;
	nodes_layer[0] = NODES_INPUT; //feature length
	nodes_layer[NUM_LAYER] = 2; // OUTPUT layer

	//initialize lockers
	for (int k = 0; k < NUM_LAYER + 1; k++) {
		pthread_mutex_init(&mutex_data[k], NULL);
	}

	//initialize layer arguments (identify workers of different layers)
	for (int k = 0; k < NUM_LAYER + 1; k++) {
		layer_arg[k].layer = k;
	}

	// Initialize the  memory for MCMC samples
	for (int k = 0; k < NUM_LAYER + 1; k++) {
		sample_mem[k] = (float*) malloc(nodes_layer[k] * sizeof(float));

		for (int j = 0; j < nodes_layer[k]; j++)
			sample_mem[k][j] = (float)rand()/RAND_MAX;
	}

	// Initialize the  memory for weight parameters
	for (int k = 1; k < NUM_LAYER + 1; k++) {
		weights[k] = (float *)  malloc(nodes_layer[k-1] * nodes_layer[k] * sizeof(float));
		bh[k] = (float *) malloc((nodes_layer[k]) * sizeof(float));
		bv[k] = (float *) malloc(nodes_layer[k-1] * sizeof(float));

		for (int j = 0; j < nodes_layer[k-1] * nodes_layer[k]; j++)
			weights[k][j] = 0.1 * ((float)rand()/RAND_MAX * 2 - 1);
		for (int j = 0; j < nodes_layer[k-1]; j++)
			//bv[k][j] = 1.0 / nodes_layer[k-1] * ((float)rand()/RAND_MAX * 2 - 1);
			bv[k][j] = 0.0;
		for (int j = 0; j < nodes_layer[k]; j++)
			//bh[k][j] = 1.0 / nodes_layer[k] * ((float)rand()/RAND_MAX * 2 - 1);
			bh[k][j] = 0.0;
	}

	pthread_t thread[NUM_LAYER + 1];
	time_start1 = time(NULL);
	for (int i = 0; i < NUM_LAYER + 1; i++) {
		pthread_create(&thread[i], NULL, work1, (void *)&layer_arg[i]);
	}


	for (int i = 0; i < NUM_LAYER + 1; i++) {
		pthread_join(thread[i], &status);
	}
	cout << "training phase 1 completed successfully!" << endl;


	//test phase
	long lent0 = test_len[0], lent1 = test_len[1], lent2 = test_len[2], lent3 = test_len[3];
	//long lent0 = len0, lent1 = len1;
	float* test_records0 = read_data_si2("data/test0", lent0);
	float* test_records1 = read_data_si2("data/test1", lent1);
	float* test_records2 = read_data_si2("data/test2", lent2);
	float* test_records3 = read_data_si2("data/test3", lent3);


	// test examples of class 0.
	ofstream out0("out0123_0");
	for (int t = 0; t < lent0; t++) {
		for (int j = 0; j < NODES_INPUT; j++)
			sample_mem[0][j] = (float) test_records0[j + NODES_INPUT * t];
		//  	for (int j = 0; j < NODES_INPUT; j++)
		//  		cout << sample_mem[0][j] << " ";
		//  	cout << endl;
		for (int k = 1; k < NUM_LAYER; k++)
			sigm(sample_mem[k], bh[k], weights[k], sample_mem[k-1],
					nodes_layer[k], nodes_layer[k-1], true);
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
			sample_mem[NUM_LAYER][j] = -bh[NUM_LAYER][j];
			for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
				sample_mem[NUM_LAYER][j] = sample_mem[NUM_LAYER][j]
				                                                 - weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * sample_mem[NUM_LAYER-1][i];
		}
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++)
			out0 << sample_mem[NUM_LAYER][j] << " ";
		out0 << endl;
	}
	out0.close();


	// test examples of class 1.
	ofstream out1("out0123_1");
	for (int t = 0; t < lent1; t++) {
		for (int j = 0; j < NODES_INPUT; j++) {
			sample_mem[0][j] = test_records1[j + NODES_INPUT * t];
		}
		for (int k = 1; k < NUM_LAYER; k++)
			sigm(sample_mem[k], bh[k], weights[k], sample_mem[k-1],
					nodes_layer[k], nodes_layer[k-1], true);
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
			sample_mem[NUM_LAYER][j] = -bh[NUM_LAYER][j];
			for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
				sample_mem[NUM_LAYER][j] = sample_mem[NUM_LAYER][j]
				                                                 - weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * sample_mem[NUM_LAYER-1][i];
		}
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++)
			out1 << sample_mem[NUM_LAYER][j] << " ";
		out1 << endl;
	}
	out1.close();

	// test examples of class 2.
	ofstream out2("out0123_2");
	for (int t = 0; t < lent2; t++) {
		for (int j = 0; j < NODES_INPUT; j++) {
			sample_mem[0][j] = test_records2[j + NODES_INPUT * t];
		}
		for (int k = 1; k < NUM_LAYER; k++)
			sigm(sample_mem[k], bh[k], weights[k], sample_mem[k-1],
					nodes_layer[k], nodes_layer[k-1], true);
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
			sample_mem[NUM_LAYER][j] = -bh[NUM_LAYER][j];
			for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
				sample_mem[NUM_LAYER][j] = sample_mem[NUM_LAYER][j]
				                                                 - weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * sample_mem[NUM_LAYER-1][i];
		}
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++)
			out2 << sample_mem[NUM_LAYER][j] << " ";
		out2 << endl;
	}
	out2.close();

	// test examples of class 3.
	ofstream out3("out0123_3");
	for (int t = 0; t < lent3; t++) {
		for (int j = 0; j < NODES_INPUT; j++) {
			sample_mem[0][j] = test_records3[j + NODES_INPUT * t];
		}
		for (int k = 1; k < NUM_LAYER; k++)
			sigm(sample_mem[k], bh[k], weights[k], sample_mem[k-1],
					nodes_layer[k], nodes_layer[k-1], true);
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
			sample_mem[NUM_LAYER][j] = -bh[NUM_LAYER][j];
			for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
				sample_mem[NUM_LAYER][j] = sample_mem[NUM_LAYER][j]
				                                                 - weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * sample_mem[NUM_LAYER-1][i];
		}
		for (int j = 0; j < nodes_layer[NUM_LAYER]; j++)
			out3 << sample_mem[NUM_LAYER][j] << " ";
		out3 << endl;
	}
	out3.close();

	cout << "testing completed successfully!" << endl;
}



