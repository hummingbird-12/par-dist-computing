#ifndef _REQ_QUEUE_H_INCLUDED_
#define _REQ_QUEUE_H_INCLUDED_

#include <stdbool.h>
#include <stdio.h>

#define MAX_SIZE 1024

typedef struct _REQUEST {
    int fd;
    struct _REQUEST* next;
} request;

typedef struct _REQ_QUEUE {
    int size;
    request* front;
    request* back;
} req_queue;

request* build_request(const int fd);

#endif /* _REQ_QUEUE_H_INCLUDED_ */
