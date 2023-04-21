#include <stdlib.h>
#include <stdio.h>
#include "pyas/sudoku.h"

void test_sudoku_validity(sudoku_t sudoku, int expected_result) {
  if (is_valid_sudoku(sudoku) == expected_result) {
    printf("Test passed!\n");
  } else {
    printf("Test failed!\n");
  }
}

//Si on part d'un sudoku vide, cela devrait fonctionner : tester la fonction de resolution !
/*
int main() {
sudoku_t sudoku = generate_random_sudoku();
print_sudoku(sudoku);
sudoku.cells[0][0].number = 9;
print_sudoku(sudoku);
int test = is_valid_sudoku(sudoku);
printf("Sudoku is valid ? %d\n", test);
printf("%d", EXIT_SUCCESS);
return 0;
}*/

/*
int main() { //TEST RESOLUTION 001
  sudoku_t sudoku = create_sudoku();
  print_sudoku(sudoku);
  resolve(&sudoku, 1);
  print_sudoku(sudoku);
  if (is_valid_sudoku(sudoku)) {
    printf("Resolution valide !\n");
  }
  else {
    printf("##### ERREUR #####  Résolution non valide !\n");
  }
  return 0;
}
*/


int main ( int argc, char *argv[] ) {
  if ( argc != 4 ) {
    fprintf( stderr, "Usage :\n\t%s [path to sudoku] [path to resolved sudoku] [method for solving]\n", argv[ 0 ] );
    exit( EXIT_FAILURE );
  }

  char* path_to_sudoku = argv[1];
  char* path_to_resolved_sudoku = argv[2];
  char*  method = argv[3];
  int method_number = (int)method[0] - (int)('0');

  sudoku_t sudoku = read_sudoku(path_to_sudoku);
  print_sudoku(sudoku);
  resolve(&sudoku, method_number);
  print_sudoku(sudoku);

  if (is_final_sudoku(sudoku)) {
    write_sudoku(sudoku, path_to_resolved_sudoku);
  }
  else {
    printf("Impossible to write invalid sudoku\n");
  }
  return EXIT_SUCCESS;
}
