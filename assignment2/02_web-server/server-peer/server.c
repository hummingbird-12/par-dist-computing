#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "core.h"
#include "req_queue.h"
#include "worker.h"

#define BUF_SIZE 1024

struct epoll_event ep_events[MAX_POOL];

int main(int argc, char** argv) {
    int port;
    int sock_fd;
    int sock_opt;
    struct sockaddr_in s_address;

    int ep_fd;
    struct epoll_event ep_event;
    epoll_args args;

    pthread_t* thread_ids;
    int workers_cnt;

    // Parse arguments
    if (argc != 3) {
        error("Usage: %s PORT WORKER_THREADS\n", argv[0]);
        exit(1);
    }
    port = atoi(argv[1]);
    workers_cnt = atoi(argv[2]);

    // Open socket
    if ((sock_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        error("Failed to open socket\n");
        exit(1);
    }

    // Allow socket reuse on termination
    sock_opt = 1;
    setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, (const void*) &sock_opt,
               sizeof(sock_opt));

    // Bind port to socket
    memset(&s_address, 0, sizeof(s_address));
    s_address.sin_family = AF_INET;
    s_address.sin_addr.s_addr = htonl(INADDR_ANY);
    s_address.sin_port = htons((unsigned short) port);
    if (bind(sock_fd, (struct sockaddr*) &s_address, sizeof(s_address)) == -1) {
        error("Failed to band port to socket\n");
        exit(1);
    }

    // Allow upto 50 queued requests (socket-level queue)
    if (listen(sock_fd, 50) == -1) {
        error("Failed to listen socket\n");
        exit(1);
    }

    // Create epoll
    if ((ep_fd = epoll_create(MAX_POOL)) == -1) {
        error("Failed to create epoll\n");
        exit(1);
    }

    // Register socket fd in epoll
    ep_event.events = EPOLLIN | EPOLLET;
    ep_event.data.fd = sock_fd;
    if (epoll_ctl(ep_fd, EPOLL_CTL_ADD, sock_fd, &ep_event) == -1) {
        error("Failed to register to epoll\n");
        close(ep_fd);
        exit(1);
    }

    // Create worker threads
    args.epoll_fd = ep_fd;
    args.socket_fd = sock_fd;
    thread_ids = (pthread_t*) malloc(sizeof(pthread_t) * workers_cnt);
    for (int i = 0; i < workers_cnt; i++) {
        pthread_create(thread_ids + i, NULL, worker_request_q_poll, &args);
    }

    for (int i = 0; i < workers_cnt; i++) {
        pthread_join(thread_ids[i], NULL);
    }
    free(thread_ids);

    return 0;
}

void error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}
