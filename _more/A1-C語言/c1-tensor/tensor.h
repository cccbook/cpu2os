#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int *shape;     // Array to store the shape of the tensor
    int ndim;       // Number of dimensions
    double *data;   // Pointer to the data array
} Tensor;

// Function prototypes
Tensor *create_tensor(int *shape, int ndim);
void free_tensor(Tensor *tensor);
void initialize_tensor(Tensor *tensor, double value);
void print_tensor(Tensor *tensor);
int product_of_dimensions(int *shape, int ndim);
void tensor_add(Tensor *result, Tensor *tensor1, Tensor *tensor2);
void random_tensor(Tensor *tensor);

#endif /* TENSOR_H */
