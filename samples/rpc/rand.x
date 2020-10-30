program RAND_PROG {
    version RAND_VERS {
        void INITIALIZE_RANDOM(long) = 1;
        double GET_NEXT_RANDOM(void) = 2;
    } = 1;
} = 0x31111111;
