#include "sync.h"

#include <pthread.h>

pthread_mutex_t req_queue_lock;
pthread_cond_t req_queue_full;
pthread_cond_t req_queue_empty;

/*
 * Initializes variables for thread sync
 */
void initialize_syncs() {
    pthread_mutex_init(&req_queue_lock, NULL);
    pthread_cond_init(&req_queue_full, NULL);
    pthread_cond_init(&req_queue_empty, NULL);
}

/*
 * Destroys variables for thread sync
 */
void destroy_syncs() {
    pthread_mutex_destroy(&req_queue_lock);
    pthread_cond_destroy(&req_queue_full);
    pthread_cond_destroy(&req_queue_empty);
}
