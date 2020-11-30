#include "worker.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core.h"

#define BUF_SIZE 1024

extern struct epoll_event ep_events[MAX_POOL];

static void handle_request(request* req);
static void print_header(FILE* stream, const char* code, const char* status,
                         const char* type);

/*
 * Entrance function for worker threads
 */
void* worker_request_q_poll(void* param) {
    epoll_args* args = (epoll_args*) param;
    int ep_fd = args->epoll_fd;
    int sock_fd = args->socket_fd;
    int con_fd;
    int ep_timeout;
    int ep_event_cnt;
    int c_len;
    struct sockaddr_in c_address;

    // Wait for epoll
    ep_timeout = -1;
    while (true) {
        if ((ep_event_cnt =
                 epoll_wait(ep_fd, ep_events, MAX_POOL, ep_timeout)) == -1) {
            error("Failed epoll wait\n");
            exit(1);
        }

        for (int i = 0; i < ep_event_cnt; i++) {
            if (ep_events[i].data.fd == sock_fd) {
                if ((con_fd = accept(sock_fd, (struct sockaddr*) &c_address,
                                     (socklen_t*) &c_len)) == -1) {
                    error("Failed to accept socket connection\n");
                    exit(1);
                }

                handle_request(build_request(con_fd));
            }
        }
    }
}

/*
 * Handler for each HTTP request
 */
void handle_request(request* req) {
    int con_fd = req->fd;
    FILE* con_stream;
    char req_buffer[BUF_SIZE];
    char req_method[BUF_SIZE];
    char req_uri[BUF_SIZE];
    char req_version[BUF_SIZE];
    char req_path[BUF_SIZE];
    struct stat stat_buf;
    char* file_map;

    // Open connection as a stream
    if ((con_stream = fdopen(con_fd, "r+")) == NULL) {
        error("Failed to read socket connection\n");
        exit(1);
    }

    // Read HTTP request
    fgets(req_buffer, BUF_SIZE, con_stream);
    sscanf(req_buffer, "%s %s %s\n", req_method, req_uri, req_version);
    strcpy(req_path, ".");
    strcat(req_path, req_uri);

    // Ignore HTTP headers
    fgets(req_buffer, BUF_SIZE, con_stream);
    while (strcmp(req_buffer, "\r\n")) {
        fgets(req_buffer, BUF_SIZE, con_stream);
    }

    // Check for a valid GET request
    if (strcasecmp(req_method, "GET") != 0) {
        error("Received a non-GET request\n");
        print_header(con_stream, "501", "Not Implemented", "text/html");
        fprintf(con_stream, "501 Not Implemented");
        fclose(con_stream);
        close(con_fd);
        return;
    }

    // Check existence of requested file
    if (stat(req_path, &stat_buf) == -1) {
        error("Requested file not found: %s\n", req_path);
        print_header(con_stream, "404", "Not Found", "text/html");
        fprintf(con_stream, "404 Not Found\n");
        fclose(con_stream);
        close(con_fd);
        return;
    }

    // Print response header
    print_header(con_stream, "200", "OK", "text/html");
    fflush(con_stream);

    // Map file to memory and give reponse
    int fd = open(req_path, O_RDONLY);
    file_map = mmap(0, stat_buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    fwrite(file_map, 1, stat_buf.st_size, con_stream);
    munmap(file_map, stat_buf.st_size);

    // Clean up connection
    fclose(con_stream);
    close(con_fd);
}

/*
 * Prints the HTTP response header
 */
static void print_header(FILE* stream, const char* code, const char* status,
                         const char* type) {
    fprintf(stream, "HTTP/1.1 %s %s\n", code, status);
    fprintf(stream, "Content-type: %s\n", type);
    fprintf(stream, "\n");
}
