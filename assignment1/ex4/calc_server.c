#include "calc.h"

int* addition_1_svc(expression* argp, struct svc_req* rqstp) {
	static int result;

        result = argp->op1 + argp->op2;

	return &result;
}

int* product_1_svc(expression* argp, struct svc_req* rqstp) {
	static int result;

	result = argp->op1 * argp->op2;

	return &result;
}

int* division_1_svc(expression* argp, struct svc_req* rqstp) {
	static int result;

        result = argp->op1 / argp->op2;

	return &result;
}

int* power_1_svc(expression* argp, struct svc_req* rqstp) {
	static int result;

        result = 1;
        for (int i = 0; i < argp->op2; i++) {
            result *= argp->op1;
        }

	return &result;
}
