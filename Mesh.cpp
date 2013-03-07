//============================================================================
// Name        : Mesh.cpp
// Author      : George Rokos
// Description : Mesh implementation
//============================================================================

#include <algorithm>
#include <cassert>
#include <iostream>
#include <cmath>
#include <set>
#include <vector>

#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#include "Mesh.hpp"

Mesh::Mesh(const char *filename){
  // Check whether the provided file exists.
  ifstream ifile(filename);
  if(!ifile){
    std::cerr << "File " << filename << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  vtkXMLUnstructuredGridReader *reader = vtkXMLUnstructuredGridReader::New();
  reader->SetFileName(filename);
  reader->Update();

  vtkUnstructuredGrid *ug = reader->GetOutput();

  NNodes = ug->GetNumberOfPoints();
  NElements = ug->GetNumberOfCells();

  // Get the coordinates of each mesh vertex. There is no z coordinate in 2D,
  // but VTK treats 2D and 3D meshes uniformly, so we have to provide memory
  // for z as well (r[2] will always be zero and we ignore it).
  for(size_t i=0;i<NNodes;i++){
    double r[3];
    ug->GetPoints()->GetPoint(i, r);
    coords.push_back(r[0]);
    coords.push_back(r[1]);
  }
  assert(coords.size() == 2*NNodes);

  // Get the metric at each vertex.
  for(size_t i=0;i<NNodes;i++){
    double *tensor = ug->GetPointData()->GetArray("Metric")->GetTuple4(i);
    metric.push_back(tensor[0]);
    metric.push_back(tensor[1]);
    assert(tensor[1] == tensor[2]);
    metric.push_back(tensor[3]);
  }
  assert(metric.size() == 3*NNodes);

  // Get the 3 vertices comprising each element.
  for(size_t i=0;i<NElements;i++){
    vtkCell *cell = ug->GetCell(i);
    for(int j=0;j<3;j++){
      ENList.push_back(cell->GetPointId(j));
    }
  }
  assert(ENList.size() == 3*NElements);

  reader->Delete();

  create_adjacency();
  find_surface();
  set_orientation();
}

void Mesh::create_adjacency(){
  NNList.resize(NNodes);
  NEList.resize(NNodes);

  for(size_t eid=0; eid<NElements; ++eid){
    // Get a pointer to the three vertices comprising element eid.
    const size_t *n = &ENList[3*eid];

    // For each vertex, add the other two vertices to its node-node adjacency
    // list and element eid to its node-element adjacency list.
    for(size_t i=0; i<3; ++i){
      NNList[n[i]].push_back(n[(i+1)%3]);
      NNList[n[i]].push_back(n[(i+2)%3]);

      NEList[n[i]].insert(eid);
    }
  }
}

bool Mesh::isSurfaceNode(size_t vid) const{
  return NEList[vid].size() < NNList[vid].size();
}

bool Mesh::isCornerNode(size_t vid) const{
  return fabs(normals[2*vid])==1.0 && fabs(normals[2*vid+1]==1.0);
}

void Mesh::find_surface(){
  // Initialise all normal vectors to (0.0,0.0).
  normals.resize(2*NNodes, 0.0);

  // If an edge is on the surface, then it belongs to only 1 element. We
  // traverse all edges (vid0,vid1) and for each edge we find the intersection
  // of NEList[vid0] and NEList[vid1]. If the intersection size is 1, then
  // this edge belongs to only one element, so it lies on the mesh surface.
  for(size_t vid=0; vid<NNodes; ++vid){
    for(std::vector<size_t>::const_iterator it=NNList[vid].begin();
      it!=NNList[vid].end(); ++it){
      // In order to avoid processing an edge twice, one in the
      // form of (vid0,vid1) and one in the form of (vid1,vid0),
      // an edge is processed only of vid0 < vid1.
      if(vid > *it)
        continue;

      std::set<size_t> intersection;
      std::set_intersection(NEList[vid].begin(), NEList[vid].end(),
          NEList[*it].begin(), NEList[*it].end(),
          std::inserter(intersection, intersection.begin()));

      if(intersection.size()==1){ // We have found a surface edge
        double x=coords[2*vid], y=coords[2*vid+1];

        // Find which surface vid and *it belong to and set the corresponding
        // coordinate of the normal vector to ±1.0. The other coordinate is
        // intentionally left intact. This way, the normal vector for corner
        // vertices will be at the end (±1.0,±1.0), which enables us to detect
        // that they are corner vertices and are not allowed to be smoothed.
        if(fabs(y-1.0) < 1E-12){// vid is on the top surface
          normals[2*vid+1] = 1.0;
          normals[2*(*it)+1] = 1.0;
        }
        else if(fabs(y) < 1E-12){// vid is on the bottom surface
          normals[2*vid+1] = -1.0;
          normals[2*(*it)+1] = -1.0;
        }
        else if(fabs(x-1.0) < 1E-12){// vid is on the right surface
          normals[2*vid] = 1.0;
          normals[2*(*it)] = 1.0;
        }
        else if(fabs(x) < 1E-12){// vid is on the left surface
          normals[2*vid] = -1.0;
          normals[2*(*it)] = -1.0;
        }
        else{
          std::cerr << "Invalid surface vertex coordinates" << std::endl;
        }
      }
    }
  }
}

/* Computing the area of an element as the inner product of two element edges
 * depends on the order in which the three vertices comprising the element have
 * been stored in ENList. Using the right-hand rule for the cross product of
 * two 2D vectors, we can find whether the three vertices define a positive or
 * negative element. If the order in ENList suggests a clockwise traversal of
 * the element, the cross product faces the negative z-axis, so the element is
 * negative and we will have to correct the calculation of its area by
 * multiplying the result by -1.0. During smoothing, after a vertex is
 * relocated, if the area of an adjacent element is found to be negative it
 * means that the element has been inverted and the new location of the vertex
 * should be discarded. The orientation is the same for all mesh elements, so
 * it is enough to calculate it for one mesh element only.
 */
void Mesh::set_orientation(){
  // Find the orientation for the first element
  const size_t *n = &ENList[0];

  // Pointers to the coordinates of each vertex
  const double *c0 = &coords[2*n[0]];
  const double *c1 = &coords[2*n[1]];
  const double *c2 = &coords[2*n[2]];

  double x1 = (c0[0] - c1[0]);
  double y1 = (c0[1] - c1[1]);

  double x2 = (c0[0] - c2[0]);
  double y2 = (c0[1] - c2[1]);

  double A = x1*y2 - x2*y1;

  if(A<0)
    orientation = -1;
  else
    orientation = 1;
}

/* Element area in physical (Euclidean) space. Recall that the area of a
 * triangle ABC is calculated as area=0.5*(AB⋅AC), i.e. half the inner product
 * of two of the element's edges (e.g. AB and AC). The result is corrected by
 * the orientation factor ±1.0, so that the area is always a positive number.
 */
double Mesh::element_area(size_t eid) const{
  const size_t *n = &ENList[3*eid];

  // Pointers to the coordinates of each vertex
  const double *c0 = &coords[2*n[0]];
  const double *c1 = &coords[2*n[1]];
  const double *c2 = &coords[2*n[2]];

  return orientation * 0.5 *
            ( (c0[1] - c2[1]) * (c0[0] - c1[0]) -
              (c0[1] - c1[1]) * (c0[0] - c2[0]) );
}

/* This function evaluates the quality of an element, based on the 2D quality
 * functional proposed by Lipnikov et. al.. The description for the functional
 * is taken from: Yu. V. Vasileskii and K. N. Lipnikov, An Adaptive Algorithm
 * for Quasioptimal Mesh Generation, Computational Mathematics and Mathematical
 * Physics, Vol. 39, No. 9, 1999, pp. 1468 - 1486.
 */
double Mesh::element_quality(size_t eid) const{
  const size_t *n = &ENList[3*eid];

  // Pointers to the coordinates of each vertex
  const double *c0 = &coords[2*n[0]];
  const double *c1 = &coords[2*n[1]];
  const double *c2 = &coords[2*n[2]];

  // Pointers to the metric tensor at each vertex
  const double *m0 = &metric[3*n[0]];
  const double *m1 = &metric[3*n[1]];
  const double *m2 = &metric[3*n[2]];

  // Metric tensor averaged over the element
  double m00 = (m0[0] + m1[0] + m2[0])/3;
  double m01 = (m0[1] + m1[1] + m2[1])/3;
  double m11 = (m0[2] + m1[2] + m2[2])/3;

  // l is the length of the perimeter, measured in metric space
  double l =
    sqrt((c0[1] - c1[1])*((c0[1] - c1[1])*m11 + (c0[0] - c1[0])*m01) +
         (c0[0] - c1[0])*((c0[1] - c1[1])*m01 + (c0[0] - c1[0])*m00))+
    sqrt((c0[1] - c2[1])*((c0[1] - c2[1])*m11 + (c0[0] - c2[0])*m01) +
         (c0[0] - c2[0])*((c0[1] - c2[1])*m01 + (c0[0] - c2[0])*m00))+
    sqrt((c2[1] - c1[1])*((c2[1] - c1[1])*m11 + (c2[0] - c1[0])*m01) +
         (c2[0] - c1[0])*((c2[1] - c1[1])*m01 + (c2[0] - c1[0])*m00));

  // Area in physical space
  double a = element_area(eid);

  // Area in metric space
  double a_m = a*sqrt(m00*m11 - m01*m01);

  // Function
  double f = std::min(l/3.0, 3.0/l);
  double F = pow(f * (2.0 - f), 3.0);

  // This is the 2D Lipnikov functional.
  double quality = 12.0 * sqrt(3.0) * a_m * F / (l*l);

  return quality;
}

// Finds the mean quality, averaged over all mesh elements,
// and the quality of the worst element.
Quality Mesh::get_mesh_quality() const{
  Quality q;

  double mean_q = 0.0;
  double min_q = 1.0;

  for(size_t i=0;i<NElements;i++){
    double ele_q = element_quality(i);

    mean_q += ele_q;
    min_q = std::min(min_q, ele_q);
  }

  q.mean = mean_q/NElements;
  q.min = min_q;

  return q;
}
