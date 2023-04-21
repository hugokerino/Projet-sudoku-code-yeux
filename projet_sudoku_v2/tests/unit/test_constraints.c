#include "pyas/resolve.h"
#include "stdlib.h"
#include <pyas/resolve.h>
#include <pyas/sudoku.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
  char* path_to_sudoku = "tests/integration/reading01/sudoku4.txt";
  sudoku_t* psudoku = calloc(1, sizeof(sudoku_t));
  *psudoku = read_sudoku(path_to_sudoku);
  print_sudoku(*psudoku);

  int i, j;
  int** tab_of_constraints = malloc(9 * sizeof(int*));
  for (i = 0; i < 9; i++) {
    tab_of_constraints[i] = malloc(9 * sizeof(int));
  } //allocation dynamique du tableau des contraintes !

  for (i = 0; i < 9; i++) {
    for (j = 0; j < 9; j++) {
      set_possibilities(psudoku, i, j);
      tab_of_constraints[i][j] = count_contraints(psudoku, i, j); //stockage des possibilités dans chaque cellule et creation du tableau de contraintes
      printf("%d", tab_of_constraints[i][j]);
    }
    printf("\n");
  }

  return 0;
}
