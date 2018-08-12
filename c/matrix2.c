#include <stdio.h>

int main()
{
  short c, d, k, sum = 0;

  int first[3][3] =  {{ 1,2,3},{4,5,6},{7,8,9 }};
  int second[3][3] =  {{ 2,3,4},{5,6,7},{8,9,0 }};
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
}

