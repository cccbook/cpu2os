#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"

// Function to create a tensor
Tensor *create_tensor(int *shape, int ndim) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        return NULL; // Allocation failure
    }

    tensor->shape = (int *)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) {
        free(tensor);
        return NULL; // Allocation failure
    }

    tensor->ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        tensor->shape[i] = shape[i];
    }

    tensor->data = (double *)malloc(product_of_dimensions(shape, ndim) * sizeof(double));
    if (tensor->data == NULL) {
        free(tensor->shape);
        free(tensor);
        return NULL; // Allocation failure
    }

    return tensor;
}

// Function to free memory allocated for a tensor
void free_tensor(Tensor *tensor) {
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}

// Function to initialize tensor with a specific value
void initialize_tensor(Tensor *tensor, double value) {
    int total_elements = product_of_dimensions(tensor->shape, tensor->ndim);
    for (int i = 0; i < total_elements; ++i) {
        tensor->data[i] = value;
    }
}

// Function to print a tensor
void print_tensor(Tensor *tensor) {
    printf("Tensor Shape: (");
    for (int i = 0; i < tensor->ndim; ++i) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf(")\n");

    printf("Tensor Data:\n");
    int *indices = (int *)malloc(tensor->ndim * sizeof(int));
    for (int i = 0; i < tensor->ndim; ++i) {
        indices[i] = 0;
    }

    for (int i = 0; i < product_of_dimensions(tensor->shape, tensor->ndim); ++i) {
        printf("%.2f ", tensor->data[i]);

        // Update indices for next element
        for (int j = tensor->ndim - 1; j >= 0; --j) {
            indices[j]++;
            if (indices[j] < tensor->shape[j]) {
                break;
            }
            indices[j] = 0;
        }

        if (i % tensor->shape[tensor->ndim - 1] == tensor->shape[tensor->ndim - 1] - 1) {
            printf("\n");
        }
    }

    free(indices);
}

// Function to calculate the product of dimensions in a shape
int product_of_dimensions(int *shape, int ndim) {
    int product = 1;
    for (int i = 0; i < ndim; ++i) {
        product *= shape[i];
    }
    return product;
}

void tensor_add(Tensor *result, Tensor *tensor1, Tensor *tensor2) {
    // Check if shapes are compatible for addition
    for (int i = 0; i < result->ndim; ++i) {
        if (result->shape[i] != tensor1->shape[i] || result->shape[i] != tensor2->shape[i]) {
            printf("Error: Incompatible shapes for addition.\n");
            return;
        }
    }

    // Perform tensor addition
    int total_elements = product_of_dimensions(result->shape, result->ndim);
    for (int i = 0; i < total_elements; ++i) {
        result->data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

void random_tensor(Tensor *tensor) {
    int total_elements = product_of_dimensions(tensor->shape, tensor->ndim);
    srand(time(NULL));

    for (int i = 0; i < total_elements; ++i) {
        tensor->data[i] = (double)rand() / RAND_MAX; // Generates random values between 0 and 1
    }
}
