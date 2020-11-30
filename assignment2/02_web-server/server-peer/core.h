#ifndef _CORE_H_INCLUDED_
#define _CORE_H_INCLUDED_

#define MAX_POOL 1024

typedef struct _EPOLL_ARGS {
    int socket_fd;
    int epoll_fd;
} epoll_args;

void error(const char* format, ...);

#endif /* _CORE_H_INCLUDED_ */
