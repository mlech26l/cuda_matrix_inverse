/* Matrix Inversion 
 * Group F: M. Lechner, P. Knöbel, J. Lövhall
 *
 * Main entry point
*/

#include "includes.h"

static void print_usage(char *progname)
{
  fprintf(stderr, "\nUsage: %s [-d deviceID] [-g Use gauss] [-c Use cofactors] [-n size]\n",
          progname);
}

int main(int argc, char **argv)
{
  printf("CUDA Matrix inversion program - by Jakob and Mathias\n\n");
  int opt;
  int deviceID=0;
  int n=100;
  int use_gauss=0;
  int use_cofactors=0;

  while ((opt = getopt(argc, argv, "gcn:d:")) != -1) {
          switch (opt) {
          case 'n':
              n = atoi(optarg);
              break;
          case 'd':
              deviceID = atoi(optarg);
              break;
          case 'g':
              use_gauss=1;
              break;
          case 'c':
              use_cofactors=1;
              break;
          default: /* '?' */
              print_usage(argv[0]);
              exit(EXIT_FAILURE);
          }
      }
  if(n<= 0)
  {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  device_query(deviceID);

  if(use_gauss)
  {
    test_gauss(n);
  }
  else if(use_cofactors)
  {
    test_cofactors(n);
  }
  else
  {
    test_lu_decomposition(n);
  }
  
	exit(EXIT_SUCCESS);
}
