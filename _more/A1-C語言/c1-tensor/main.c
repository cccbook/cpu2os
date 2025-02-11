#include <stdio.h>
#include "tensor.h"

int main() {
    // Test create_tensor and print_tensor
    int shape[] = {2, 3, 4};
    int ndim = 3;
    Tensor *tensor = create_tensor(shape, ndim);

    if (tensor == NULL) {
        printf("Failed to create tensor.\n");
        return 1;
    }

    printf("Created Tensor:\n");
    print_tensor(tensor);
    printf("\n");

    // Test initialize_tensor and print_tensor
    initialize_tensor(tensor, 1.5);

    printf("Initialized Tensor:\n");
    print_tensor(tensor);
    printf("\n");

    // Test free_tensor
    free_tensor(tensor);

    // Test tensor_add
    // int shape[] = {2, 3, 4};
    // int ndim = 3;
    Tensor *tensor1 = create_tensor(shape, ndim);
    Tensor *tensor2 = create_tensor(shape, ndim);
    Tensor *result = create_tensor(shape, ndim);

    initialize_tensor(tensor1, 1.0);
    random_tensor(tensor2);
    // initialize_tensor(tensor2, 2.0);

    printf("Tensor 1:\n");
    print_tensor(tensor1);
    printf("\n");

    printf("Tensor 2:\n");
    print_tensor(tensor2);
    printf("\n");

    tensor_add(result, tensor1, tensor2);

    printf("Result Tensor (Tensor 1 + Tensor 2):\n");
    print_tensor(result);

    // Cleanup
    free_tensor(tensor1);
    free_tensor(tensor2);
    free_tensor(result);
    return 0;
}
