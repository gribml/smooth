//============================================================================
// Name        : ACA2-2013.cpp
// Author      : George Rokos
// Description : 2nd Assessed Coursework for ACA 2013
//============================================================================

#include <cstdlib>
#include <iostream>
#include <sys/time.h>

#include "Mesh.hpp"
#include "Smooth.hpp"

double get_wtime(){
    struct timeval tic;

    long seconds, useconds;

    gettimeofday(&tic, NULL);

    seconds  = tic.tv_sec;
    useconds = tic.tv_usec;

    return seconds + useconds*1e-06;
}

int main(int argc, char **argv){
  if(argc!=2){
    std::cerr << "Usage: " << argv[0] << " mesh_file" << std::endl;
  }

  Mesh *mesh = new Mesh(argv[1]);

  Quality q = mesh->get_mesh_quality();

  std::cout << "Initial quality:\n"
            << "Quality mean:  " << q.mean << std::endl
            << "Quality min:   " << q.min << std::endl;

  double time = get_wtime();
  smooth(mesh, 200);
  double time_smooth = get_wtime() - time;

  q = mesh->get_mesh_quality();

  std::cout<<"After smoothing:\n"
           << "Quality mean:  " << q.mean << std::endl
           << "Quality min:   " << q.min << std::endl;

  if((q.mean>0.90)&&(q.min>0.55))
    std::cout << "Test passed"<< std::endl;
  else
    std::cout << "Test failed"<< std::endl;

  std::cout<<"BENCHMARK: " << time_smooth << "s" << std::endl;

  delete mesh;

  return EXIT_SUCCESS;
}
