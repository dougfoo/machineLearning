#include <stdio.h>

void printMatrix(int m[3][3]) 
{
    int c, d = 0;
    for (c = 0; c < 3; c++) {
      for (d = 0; d < 3; d++)
        printf("%d\t", m[c][d]);
      printf("\n");
    }
    printf("\n");
}

int main()
{
  int c, d, k, sum = 0;

  int first[3][3] =  {{ 0,1,2},{3,4,5},{6,7,8 }};
  int second[3][3] =  {{ 5,5,5},{-5,-5,-5},{5,5,5 }};
  int multiply[3][3];

    for (c = 0; c < 3; c++) {
      for (d = 0; d < 3; d++) {
        for (k = 0; k < 3; k++) {
          sum = sum + first[c][k]*second[k][d];
        }
 
        multiply[c][d] = sum;
        sum = 0;
      }
    }
 
    printf("Product of the matrices:\n");
    printMatrix(first);
    printMatrix(second);
    printMatrix(multiply);
}

