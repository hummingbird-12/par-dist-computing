#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define BUF_SIZE 512

typedef struct _ARGUMENTS {
    char req_files[BUF_SIZE][BUF_SIZE];
    char server_addr[BUF_SIZE];
    int server_port;
    int requests_cnt;
    int files_cnt;
    int id;
} sim_args;

static void error(const char* format, ...);

void* request_files(void* param);
void request(const int thread_id, const int iter, const char* addr,
             const int port, const char* path);

int main(int argc, char** argv) {
    sim_args args;
    sim_args* thread_args;
    FILE* req_files_fp;
    pthread_t* thread_ids;
    int threads_cnt;

    // Parse arguments
    if (argc != 6) {
        error(
            "Usage: %s SERVER_ADDR SERVER_PORT THREADS_NUM REQ_PER_THREAD "
            "REQ_FILES\n",
            argv[0]);
        exit(1);
    }
    strcpy(args.server_addr, argv[1]);
    args.server_port = atoi(argv[2]);
    threads_cnt = atoi(argv[3]);
    args.requests_cnt = atoi(argv[4]);
    if ((req_files_fp = fopen(argv[5], "r")) == NULL) {
        error("Failed to open request files list\n");
        exit(1);
    }

    // Read file list available for request
    int i = 0;
    while (fgets(args.req_files[i], BUF_SIZE, req_files_fp) != NULL) {
        args.req_files[i][strlen(args.req_files[i]) - 1] = '\0';
        i++;
    }
    args.files_cnt = i;

    thread_ids = (pthread_t*) malloc(sizeof(pthread_t) * threads_cnt);
    thread_args = (sim_args*) malloc(sizeof(sim_args) * threads_cnt);
    for (int i = 0; i < threads_cnt; i++) {
        thread_args[i] = args;
        thread_args[i].id = i;
        pthread_create(thread_ids + i, NULL, request_files, &thread_args[i]);
    }

    for (int i = 0; i < threads_cnt; i++) {
        pthread_join(thread_ids[i], NULL);
    }
    free(thread_ids);
    free(thread_args);

    return 0;
}

void* request_files(void* param) {
    sim_args* args = (sim_args*) param;

    for (int i = 0; i < args->requests_cnt; i++) {
        const int file_idx = (i + args->id) % args->files_cnt;
        request(args->id, i, args->server_addr, args->server_port,
                args->req_files[file_idx]);
        sleep(1);
    }

    pthread_exit(0);
    return (void*) 0;
}

void request(const int thread_id, const int iter, const char* addr,
             const int port, const char* path) {
    int sock_fd;
    int sock_opt;
    struct sockaddr_in s_address;
    char request[BUF_SIZE] = {'\0'};
    char response[BUF_SIZE] = {'\0'};
    char res_prot[BUF_SIZE];
    int res_code;
    char res_status[BUF_SIZE];
    int req_size;
    int res_size;
    int bytes_read;

    // Open socket
    if ((sock_fd = socket(PF_INET, SOCK_STREAM, 0)) == -1) {
        error("Failed to open socket\n");
        return;
    }

    // Allow socket reuse on termination
    sock_opt = 1;
    setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, (const void*) &sock_opt,
               sizeof(sock_opt));

    // Bind port to socket
    memset(&s_address, 0, sizeof(s_address));
    s_address.sin_family = AF_INET;
    s_address.sin_addr.s_addr = inet_addr(addr);
    s_address.sin_port = htons((unsigned short) port);

    // Connect to socket
    if (connect(sock_fd, (struct sockaddr*) &s_address, sizeof(s_address)) ==
        -1) {
        error("Failed socket connection\n");
        return;
    }

    // Send request
    req_size = sprintf(request, "GET %s HTTP/1.0\n\r\n", path);
    if (write(sock_fd, request, req_size) == -1) {
        error("Failed to write to socket\n");
        return;
    }

    // Read HTTP response
    res_size = read(sock_fd, response, BUF_SIZE);
    if (sscanf(response, "%s %d %s\n", res_prot, &res_code, res_status) != 3) {
        error("Failed to parse response\n");
        return;
    }

    // Unhealthy response
    if (res_code != 200) {
        error("Received as response: %d %s\n", res_code, res_status);
        return;
    }

    // Read the rest of the response
    while ((bytes_read = read(sock_fd, response, BUF_SIZE)) > 0) {
        res_size += bytes_read;
    }

    printf("[Thread %2d] File #%d (%s) - Received %d bytes\n", thread_id, iter,
           path, res_size);
}

void error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}
