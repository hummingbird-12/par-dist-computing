struct expression {
    int op1;
    int op2;
};

program CALC_PROG {
    version CALC_VERS {
        int ADDITION(expression*) = 1;
        int PRODUCT(expression*) = 2;
        int DIVISION(expression*) = 3;
        int POWER(expression*) = 4;
    } = 1;
} = 0x20000000;
