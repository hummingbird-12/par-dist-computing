#include "req_queue.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Builds a requests queue node with the given file descriptor
 */
request* build_request(const int fd) {
    request* req = calloc(1, sizeof(request));
    req->fd = fd;
    req->next = NULL;
    return req;
}
