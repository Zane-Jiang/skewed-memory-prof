#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define ARRAY_SIZE (1024 * 1024 * 256) // 1GB for int32_t

int main() {
    int32_t *array = (int32_t *)malloc(sizeof(int32_t) * ARRAY_SIZE);
    if (!array) {
        perror("malloc failed");
        return 1;
    }
    
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        array[i] = i;
    }
    volatile int64_t sum = 0;

    #define DUMMY_SIZE (1024 * 1024 * 128) // 512MB
    {
        char *dummy = (char *)malloc(DUMMY_SIZE);
        if (dummy) {
            for (size_t i = 0; i < DUMMY_SIZE; ++i) {
                dummy[i] += i;
            }
            free(dummy);
        }
    }

    clock_t start = clock();
    
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        sum += array[i];
    }
    clock_t end = clock();
    printf("Sum: %ld\n", sum);
    printf("Time: %.3f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(array);
    return 0;
}
