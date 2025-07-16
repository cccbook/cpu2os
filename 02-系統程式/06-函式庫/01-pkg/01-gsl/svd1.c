// https://stackoverflow.com/questions/66536134/gsl-gsl-linalg-sv-decomp-returning-u-and-v-reversed
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

void pretty_print(const gsl_matrix * M)
{
  // Get the dimension of the matrix.
  int rows = M->size1;
  int cols = M->size2;
  // Now print out the data in a square format.
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      printf("%f ", gsl_matrix_get(M, i, j));
    }
    printf("\n");
  }
  printf("\n");
}

void pretty_print_vector(const gsl_vector * M)
{
  int cols = M->size;
    for(int j = 0; j < cols; j++){
      printf("%f ", gsl_vector_get(M, j));
    }
  printf("\n");
}

int
main()
{
  const size_t M = 4;
  const size_t N = 5;
  double A_data[] = {
    1.0, 0.0, 0.0, 0.0, 2.0,
    0.0, 0.0, 3.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0, 0.0 };
  gsl_matrix_view A = gsl_matrix_view_array(A_data, 4, 5);
  gsl_matrix * B = gsl_matrix_alloc(N, M);
  gsl_matrix * V = gsl_matrix_alloc(M, M);
  gsl_vector * S = gsl_vector_alloc(M);
  gsl_vector * work = gsl_vector_alloc(M);

  gsl_matrix_transpose_memcpy(B, &A.matrix);

  gsl_linalg_SV_decomp(B, V, S, work);

  printf("U:\n");
  pretty_print(V);

  printf("S:\n");
  pretty_print_vector(S);

  printf("V:\n");
  pretty_print(B);

  gsl_matrix_free(B);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work);

  return 0;
}