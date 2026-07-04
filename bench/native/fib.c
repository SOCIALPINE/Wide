// bench: recursive fib(30) — C reference (self-timed like `wide --time`).
#include <stdio.h>
#include <time.h>

long long fib(long long n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

int main(void) {
    clock_t t0 = clock();
    long long r = fib(30);
    double ms = (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;
    printf("%lld\n", r);
    fprintf(stderr, "time: %.3f ms\n", ms);
    return 0;
}
