/*
 * util.h
 *
 *  Created on: Mar 19, 2013
 *      Author: qiwang321
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <fstream>
#include <iostream>
using namespace std;

/* function for computing sigmoid function
 * b: bias
 * w: weight vector
 * x: data vector
 */

//compute the sigmoid function
void sigm(float* res, float* b, float* W, float* x, int n, int m, bool dir);

//sample a Bernoulli r.v.
int binrand(float p);

//read random record from stream: length is the length of the file
//acquired by the calling function.
float* read_randln(ifstream& in, long length);

// read data from memory
int* read_data_si(ifstream& in, long* length);
int* read_data_si1(const char* file, long* length);
float* read_data_si2(const char* file, long length);
float dist(float* x1, float* x2, int len);

#endif
