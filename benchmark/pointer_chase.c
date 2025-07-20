#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define NODES (1024 )

typedef struct Node {
    struct Node *next;
    int32_t value;
} Node;

int main() {
    Node **nodes = (Node **)malloc(sizeof(Node *) * NODES);
    if (!nodes) {
        perror("malloc failed");
        return 1;
    }
    for (size_t i = 0; i < NODES; ++i) {
        nodes[i] = (Node *)malloc(sizeof(Node));
        if (!nodes[i]) {
            perror("malloc node failed");
            return 1;
        }
        nodes[i]->value = i;
    }
    for (size_t i = NODES - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        Node *tmp = nodes[i];
        nodes[i] = nodes[j];
        nodes[j] = tmp;
    }
    for (size_t i = 0; i < NODES - 1; ++i) {
        nodes[i]->next = nodes[i + 1];
    }
    nodes[NODES - 1]->next = NULL;

    volatile int64_t sum = 0;
    Node *cur = nodes[0];
    clock_t start = clock();
    while (cur) {
        sum += cur->value;
        cur = cur->next;
    }
    clock_t end = clock();
    printf("Sum: %ld\n", sum);
    printf("Time: %.3f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    for (size_t i = 0; i < NODES; ++i) {
        free(nodes[i]);
    }
    free(nodes);
    return 0;
}
