#include "req_queue.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern pthread_mutex_t req_queue_lock;
extern pthread_cond_t req_queue_full;
extern pthread_cond_t req_queue_empty;

req_queue queue = {.size = 0, .front = NULL, .back = NULL};

/*
 * Pushes the given request into the requests queue
 */
void req_queue_push(request* req) {
    pthread_mutex_lock(&req_queue_lock);
    while (queue.size == MAX_SIZE) {
        pthread_cond_wait(&req_queue_full, &req_queue_lock);
    }
    if (queue.back == NULL) {
        queue.front = queue.back = req;
    } else {
        queue.back->next = req;
        queue.back = queue.back->next;
    }
    queue.size++;
    pthread_cond_signal(&req_queue_empty);
    pthread_mutex_unlock(&req_queue_lock);
}

/*
 * Pops and returns a request from the requests queue
 */
request* req_queue_pop() {
    pthread_mutex_lock(&req_queue_lock);
    while (queue.size == 0) {
        pthread_cond_wait(&req_queue_empty, &req_queue_lock);
    }
    request* req = queue.front;
    queue.front = queue.front->next;
    if (req == queue.back) {
        queue.back = 0;
    }
    queue.size--;
    pthread_cond_signal(&req_queue_full);
    pthread_mutex_unlock(&req_queue_lock);
    return req;
}

/*
 * Builds a requests queue node with the given file descriptor
 */
request* build_request(const int fd) {
    request* req = calloc(1, sizeof(request));
    req->fd = fd;
    req->next = NULL;
    return req;
}
