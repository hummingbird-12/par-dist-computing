#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "error.h"
#include "req_queue.h"
#include "sync.h"
#include "worker.h"

#define BUF_SIZE 1024

extern req_queue queue;

int main(int argc, char** argv) {
    int port;
    int sock_fd;
    int sock_opt;
    struct sockaddr_in s_address;
    struct sockaddr_in c_address;
    int c_len;
    int con_fd;

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

    initialize_syncs();
    for (int i = 0; i < workers_cnt; i++) {
        pthread_t tid;
        pthread_create(&tid, NULL, worker_request_q_poll, NULL);
    }

    // Connection loop
    c_len = sizeof(c_address);
    while (true) {
        // Wait and accept incoming connection
        if ((con_fd = accept(sock_fd, (struct sockaddr*) &c_address,
                             (socklen_t*) &c_len)) == -1) {
            error("Failed to accept socket connection\n");
            exit(1);
        }

        // Push to requests queue
        req_queue_push(build_request(con_fd));
    }

    destroy_syncs();

    return 0;
}

void error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}
