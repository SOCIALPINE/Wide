// bench: 10M-iteration f64 while loop — C reference (self-timed).
#include <stdio.h>
#include <time.h>

int main(void) {
    clock_t t0 = clock();
    volatile double total = 0.0;
    double i = 0.0;
    while (i < 10000000.0) {
        total = total + i;
        i = i + 1.0;
    }
    double ms = (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;
    printf("%.0f\n", (double)total);
    fprintf(stderr, "time: %.3f ms\n", ms);
    return 0;
}
