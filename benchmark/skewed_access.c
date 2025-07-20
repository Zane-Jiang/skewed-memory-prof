#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define NUM_BLOCKS 4096
#define BLOCK_SIZE 256
#define HOT_RATIO 0.1 // 10% 热块
#define HOT_ACCESS 10000
#define COLD_ACCESS 100
//模拟内存访问倾斜现象
int main() {
    uint8_t **blocks = (uint8_t **)malloc(sizeof(uint8_t *) * NUM_BLOCKS);
    if (!blocks) {
        perror("malloc failed");
        return 1;
    }
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        blocks[i] = (uint8_t *)malloc(BLOCK_SIZE);
        if (!blocks[i]) {
            perror("malloc block failed");
            return 1;
        }
        for (size_t j = 0; j < BLOCK_SIZE; ++j) {
            blocks[i][j] = (uint8_t)(i + j);
        }
    }
    size_t num_hot = (size_t)(NUM_BLOCKS * HOT_RATIO);
    volatile uint64_t sum = 0;
    clock_t start = clock();
    // 热块高频访问
    for (size_t i = 0; i < num_hot; ++i) {
        for (int k = 0; k < HOT_ACCESS; ++k) {
            for (size_t j = 0; j < BLOCK_SIZE; ++j) {
                sum += blocks[i][j];
            }
        }
    }
    // 冷块低频访问
    for (size_t i = num_hot; i < NUM_BLOCKS; ++i) {
        for (int k = 0; k < COLD_ACCESS; ++k) {
            for (size_t j = 0; j < BLOCK_SIZE; ++j) {
                sum += blocks[i][j];
            }
        }
    }
    clock_t end = clock();
    printf("Sum: %lu\n", sum);
    printf("Time: %.3f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        free(blocks[i]);
    }
    free(blocks);
    return 0;
}
