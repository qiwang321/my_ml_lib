/*
 * util.cpp
 *
 *  Created on: Mar 4, 2013
 *      Author: qiwang321
 */

#include "global.h"
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;


//#define NODES_INPUT  4

/* function for computing sigmoid function
 * b: bias
 * w: weight vector
 * x: data vector
 */

//compute the sigmoid of a layer
void sigm(float* res, float* b, float* W, float* x, int n, int m, bool dir) {
	if (dir == true) { //up

		for (int j = 0; j < n; j++) {
			res[j] = -b[j];
			for (int i = 0; i < m; i++)
				res[j] = res[j] - W[j*m + i] * x[i];
			res[j] = 1 / (1 + exp(res[j]));
		}

	}

	else { //down

		for (int i = 0; i < m; i++){
			res[i] = -b[i];
			for (int j = 0; j < n; j++)
				res[i] = res[i] - W[j*m + i] * x[j];
			res[i] = 1 / (1 + exp(res[i]));
		}

	}

}

//sample a Bernoulli r.v.
int binrand(float p) {
	srand(time(NULL));
	float t = rand()/((float) RAND_MAX);
	return t < p ? 1:0;
}

//read random record from stream: length is the length of the file
//acquired by the calling function.
float* read_randln(ifstream& in, long length) {

	char* buffer = (char *) malloc(4096*sizeof(char)); //string buffer
	float* record = (float *) malloc(NODES_INPUT * sizeof(float));
	//srand(time(NULL));
  long pos = rand() % length; // random position
  in.seekg(pos, ios::beg); // move pointer to the new position
  in.getline(buffer, 4096); // skip the incomplete line
  in.getline(buffer, 4096); // read the record

  while (buffer[0] == 0) { //resample
  	pos = rand() % length;
  	in.clear(); //clear stream states
  	in.seekg(pos, ios::beg);
  	in.getline(buffer, 4096);
  	in.getline(buffer, 4096); // read the record
  }



  //parsing: current only be able to read records of aligned lengths
  //x1 x2 x3\n
  char* word = buffer;// a single word to be parsed
  char* p = buffer; // probing pointer
  int i = 1; // index of the component
  sscanf(word, "%f", &record[0]); // get the first component

  while (*p != 0) {
  	if (*p == ' ' || *p == '\t') {
  		word = p+1;
  		sscanf(word, "%f", &record[i]);
  		i += 1;
  	}
  	p += 1;
  }

  return record;

}

// read data from a single index to memory
int* read_data_si(ifstream& in, long* length) {
	char* buffer = (char *) malloc(2000*sizeof(char)); // buffer for a record
	int* records = NULL;
	int count = 0; //count of total number of records

	//char* data = NULL;
	/*
	while (!in.eof()) {
		in.getline(buffer, 2000);
		count++;
		records =  (int*) realloc(records, NODES_INPUT * count * sizeof(int));

		char* word; // word to store a single value
		char* p = buffer; // probing pointer
		int i = 0; // index of the component

		while (1) {
			while ((*p < '0' || *p > '9') && *p != 0) p++;
			word = p;
			sscanf(word, "%d", records + NODES_INPUT * (count-1) + i);
			i++;
			if (*p == 0) break;
			while (*p >= '0' && *p <= '9') p++;
		}
	}*/



	*length = count; //length of the record
	return records;
}

int* read_data_si1(const char* file, long* length) {
	ifstream in(file);
	int count = 0;
	char* buffer = (char*) malloc(2000*sizeof(char));
	for (count = 0; !in.eof(); count++)
		in.getline(buffer, 2000);
	in.close();
	int* records = (int*) malloc(count * NODES_INPUT * sizeof(int));
	in.open(file, ifstream::in);
	for (int i = 0; i < count-2; i++)
		for (int j = 0; j < NODES_INPUT; j++)
			in >> records[i * NODES_INPUT + j];

	*length = count-2;
	return records;
}

// read data with known number of records
float* read_data_si2(const char* file, long length) {
	ifstream in(file);
	float tmp;
	float* records = (float*) malloc(NODES_INPUT * length * sizeof(float));
	for (int i = 0; i < length; i++)
		for (int j = 0; j < NODES_INPUT; j++) {
			in >> tmp;
			records[i*NODES_INPUT + j] = tmp / 255.0;
		}
	in.close();
	return records;
}

// compute the 2-norm distance between two vectors
float dist(float* x1, float* x2, int len) {
	float sum = 0;
	for (int i = 0; i < len; i++)
		sum += pow(x1[i]-x2[i], 2);
	return sqrt(sum);
}

// pipelined back propagation
