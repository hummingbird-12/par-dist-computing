#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calc.h"

#define BUF_SIZE 1024

int errno;

// Operators stack
char c_stack[BUF_SIZE];
int c_top = -1;

// Operands stack
int i_stack[BUF_SIZE];
int i_top = -1;

// Stack operations
static inline void c_push(const char e);
static inline char c_pop();
static inline void i_push(const int e);
static inline int i_pop();

/*
 * Returns the operator's precedence
 */
static int precedence(const char op);
/*
 * Returns whether it is an operator
 */
static bool is_op(const char op);
/*
 * Evaluates the expression
 */
static int evaluate(const char* exp, int* result, CLIENT* clnt);
/*
 * Process an operation
 */
static int process_op(CLIENT* clnt);
/*
 * Fallback function for failure
 */
static void call_failed(CLIENT* clnt);

int main (int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s host\n", argv[0]);
        errno = EINVAL;
        perror("Error parsing arguments");
        exit(1);
    }

    CLIENT *clnt = clnt_create(argv[1], CALC_PROG, CALC_VERS, "udp");
    if (clnt == NULL) {
        clnt_pcreateerror(argv[1]);
        exit(1);
    }

    char* in = (char*) calloc(BUF_SIZE, sizeof(char));
    int result;

    // Input command
    while (fgets(in, BUF_SIZE, stdin)) {
        in[strlen(in) - 1] = '\0';
        if (strcmp(in, "exit") == 0) {
            break;
        }
        if (strcmp(in, "test") != 0) {
            printf("Unkown command: %s\n", in);
            continue;
        }

        // Input expression
        fgets(in, BUF_SIZE, stdin);
        in[strlen(in) - 1] = '\0';
        if (strlen(in) == 0) {
            printf("Invalid expression\n");
            continue;
        }

        // Evaluate expression
        if (evaluate(in, &result, clnt) == 0) {
            printf("The answer is %d\n", result);
        }
    }

    printf("\nProgram exit\n");
    clnt_destroy(clnt);
    free(in);

    return 0;
}

static inline void c_push(const char e) {
    c_stack[++c_top] = e;
}

static inline char c_pop() {
    return c_stack[c_top--];
}

static inline void i_push(const int e) {
    i_stack[++i_top] = e;
}

static inline int i_pop() {
    return i_stack[i_top--];
}

static int precedence(const char op) {
    switch (op) {
        case '^':
            return 3;
            break;
        case '*':
        case '/':
            return 2;
            break;
        case '+':
        case '-':
            return 1;
            break;
    }
}

static bool is_op(const char op) {
    switch (op) {
        case '^':
        case '*':
        case '/':
        case '+':
        case '-':
            return true;
            break;
        default:
            return false;
            break;
    }
}

static int evaluate(const char* exp, int* result, CLIENT* clnt) {
    const int n = strlen(exp);
    i_top = c_top = -1;

    for (int i = 0; i < n; i++) {
        char c = exp[i];

        if (!is_op(c)) {
            // NOT an operator

            if (!isdigit(c)) {
                printf("Invalid expression (illegal character)\n");
                return 1;
            }

            // Parse operand and push to stack
            int num = 0;
            while (i < n && isdigit(exp[i])) {
                num = num * 10 + exp[i++] - '0';
            }
            i--;
            i_push(num);
        }
        else {
            // Operator

            // Change the power operator "**" to '^'
            if (c == '*' && exp[i + 1] == '*') {
                c = '^';
                i++;
            }

            // Process operations according to precedence
            while (c_top > -1 && precedence(c_stack[c_top]) >= precedence(c)) {
                if (process_op(clnt) != 0) {
                    return 1;
                }
            }
            c_push(c);
        }
    }

    // Process remaining operations
    while (c_top > -1) {
        if (process_op(clnt) != 0) {
            return 1;
        }
    }
    *result = i_stack[i_top];

    return 0;
}

static int process_op(CLIENT* clnt) {
    if (i_top < 1) {
        printf("Invalid expression\n");
        return 1;
    }

    int* result;
    struct expression ex;

    // Obtain operands
    ex.op2 = i_pop();
    ex.op1 = i_pop();

    // Process operation with RPC
    switch(c_stack[c_top]) {
        case '+':
            if ((result = addition_1(&ex, clnt)) == NULL) {
                call_failed(clnt);
                return 1;
            }
            break;
        case '-':
            ex.op2 *= -1;
            if ((result = addition_1(&ex, clnt)) == NULL) {
                call_failed(clnt);
                return 1;
            }
            break;
        case '*':
            if ((result = product_1(&ex, clnt)) == NULL) {
                call_failed(clnt);
                return 1;
            }
            break;
        case '/':
            if (ex.op2 == 0) {
                printf("Invalid expression (division by zero)\n");
                return 1;
            }
            if ((result = division_1(&ex, clnt)) == NULL) {
                call_failed(clnt);
                return 1;
            }
            break;
        case '^':
            if ((result = power_1(&ex, clnt)) == NULL) {
                call_failed(clnt);
                return 1;
            }
            break;
    }
    i_push(*result);
    c_pop();

    return 0;
}

static void call_failed(CLIENT* clnt) {
    clnt_perror(clnt, "Call failed\n");
}
