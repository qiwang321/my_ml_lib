/*
 * global.h
 *
 *  Created on: Mar 30, 2013
 *      Author: qiwang321
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef GLOBAL_H_
#define GLOBAL_H_

#define NUM_LAYER 2 // total number of layers

class element { // linklist node on layer data
public:
	float *data; //data array
	element *next; // pointer to next
	element *prev; // pointer to previous

	element() {
		data = NULL;
		next = NULL;
		prev = NULL;
	}

	element(float *d) {
		data = d;
		next = NULL;
		prev = NULL;
	}

	~element() {
		free(data);
	}

};

class FIFO { // queue for updata
public:
	int capacity; // queue capacity
	int count; // current number of element in the queue
	element *head;
	element *tail;

	FIFO() {
		capacity = 0;
		count = 0;
		head = NULL;
		tail = NULL;
	}

	FIFO(int c) {
		capacity = c;
		count = 0;
		head = NULL;
		tail = NULL;
	}

	void enqueue (element *e) {
		if (count == capacity) {
			printf("error: enqueuing full queue!\n");
			return;
		}
		e->next = head;
		if (count != 0)
			head->prev = e;
		else
			tail = e;
		head = e;
		count++;
	}

	element *dequeue () {
		if (count == 0) {
			printf("error: dequeuing empty queue!\n");
			return NULL;
		}
		element *tmp = tail;
		if (count == 1) {
			head = NULL;
			tail = NULL;
		}
		else {
			tail = tmp->prev;
			tail->next = NULL;
		}
		count--;
		return tmp;
	}

	~FIFO () {
		while (head) {
			tail = head;
			delete(tail);
			head = head->next;
		}
	}

};

typedef struct stage { // pipeline stage
	pthread_mutex_t m_f;
	pthread_mutex_t m_b;
	pthread_mutex_t m_data;
	pthread_mutex_t m_delta;
	pthread_cond_t avail_f; // input data available for this stage (forward)?
	pthread_cond_t avail_b; // backward
	pthread_cond_t ready_f; // stage ready to receive new data (backward)?
	pthread_cond_t ready_b; // backward
	bool data_ready_f; // !=0, if other data is currently computed
	bool data_ready_b;
	pthread_t thread_f; // Forward Thread ID
	pthread_t thread_b; // backward thread id
	int layer_ind; // index of layer

} stage_t;

typedef struct layer_arg {
	int layer_ind;
	bool f;
} layer_arg;




#endif /* GLOBAL_H_ */
