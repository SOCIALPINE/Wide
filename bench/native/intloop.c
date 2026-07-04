// bench: 10M-iteration integer while loop — C reference (self-timed).
#include <stdio.h>
#include <time.h>

int main(void) {
    clock_t t0 = clock();
    volatile long long total = 0; /* volatile: keep the compiler from folding the loop away */
    long long i = 0;
    while (i < 10000000) {
        total = total + i;
        i = i + 1;
    }
    double ms = (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;
    printf("%lld\n", (long long)total);
    fprintf(stderr, "time: %.3f ms\n", ms);
    return 0;
}
