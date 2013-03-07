#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

bool isSurfaceNode(size_t vid, __global uint *NEListOffs, __global uint *NNListOffs) {
  return (NEListOffs[vid + 1] - NEListOffs[vid]) < (NNListOffs[vid + 1] - NNListOffs[vid]);
}

bool isCornerNode(size_t vid, __global double *normals) {
  return fabs(normals[2 * vid]) == 1.0 && fabs(normals[2 * vid + 1]) == 1.0;
}

double element_area(size_t eid, __global uint *ENList, __global double *coords, int orientation) {
  const __global uint *n    = &ENList[3 * eid];
  const __global double *c0 = &coords[2 * n[0]];
  const __global double *c1 = &coords[2 * n[1]];
  const __global double *c2 = &coords[2 * n[2]];

  return 0.5 * orientation * ((c0[1] - c2[1]) * (c0[0] - c1[0]) -
                              (c0[1] - c1[1]) * (c0[0] - c2[0]));

}

double element_quality(size_t eid, __global uint *ENList, __global double *metric, __global double *coords, int orientation) {
  const __global uint *n = &ENList[3 * eid];

  const __global double *c0 = &coords[2 * n[0]];
  const __global double *c1 = &coords[2 * n[1]];
  const __global double *c2 = &coords[2 * n[2]];

  const __global double *m0 = &metric[3 * n[0]];
  const __global double *m1 = &metric[3 * n[1]];
  const __global double *m2 = &metric[3 * n[2]];

  // Metric tensor averaged over the element
  double m00 = (m0[0] + m1[0] + m2[0]) / 3;
  double m01 = (m0[1] + m1[1] + m2[1]) / 3;
  double m11 = (m0[2] + m1[2] + m2[2]) / 3;

  // l is the length of the perimeter, measured in metric space
  double l =
    sqrt((c0[1] - c1[1]) * ((c0[1] - c1[1]) * m11 + (c0[0] - c1[0]) * m01)  +
         (c0[0] - c1[0]) * ((c0[1] - c1[1]) * m01 + (c0[0] - c1[0]) * m00)) +
    sqrt((c0[1] - c2[1]) * ((c0[1] - c2[1]) * m11 + (c0[0] - c2[0]) * m01)  +
         (c0[0] - c2[0]) * ((c0[1] - c2[1]) * m01 + (c0[0] - c2[0]) * m00)) +
    sqrt((c2[1] - c1[1]) * ((c2[1] - c1[1]) * m11 + (c2[0] - c1[0]) * m01)  +
         (c2[0] - c1[0]) * ((c2[1] - c1[1]) * m01 + (c2[0] - c1[0]) * m00));

  // Area in physical space
  double a = element_area(eid, ENList, coords, orientation);

  // Area in metric space
  double a_m = a * sqrt(m00 * m11 - m01 * m01);

  // Function
  double f = l < 3.0 ? l / 3.0 : 3.0 / l;
  double F = f * (2.0 - f);
  F = F * F * F;

  // This is the 2D Lipnikov functional.
  return 12.0 * sqrt(3.0) * a_m * F / (l * l);
}

__kernel void smooth
(__global uint *NEList, __global uint *NNList,
 __global uint *NEListOffs, __global uint *NNListOffs,
 __global uint *ENList,
 __global double *metric, __global double *coords,
 __global double *normals, __global uint *colorIdxs,
 const uint orientation, const uint NNodes,
 const uint colorOffset) {
  int gid = get_global_id(0);

  if(gid >= NNodes)
    return;

  uint vid = colorIdxs[colorOffset + gid];

  if(isCornerNode(vid, normals))
    return;

  double worst_q = 1.0;

  for(int i = NEListOffs[vid]; i < NEListOffs[vid + 1]; ++i) {
    uint eid = NEList[i];
    double qual = element_quality(eid, ENList, metric, coords, orientation);
    worst_q = min(worst_q, qual);
  }

  const __global double *m0 = &metric[3 * vid];
  double x0 = coords[2 * vid];
  double y0 = coords[2 * vid + 1];

  double A[4] = {0.0, 0.0, 0.0, 0.0};
  double q[2] = {0.0, 0.0};

  for(int i = NNListOffs[vid]; i < NNListOffs[vid + 1]; ++i) {
    size_t il = NNList[i];

    const __global double *m1 = &metric[3 * il];

    const double ml00 = 0.5 * (m0[0] + m1[0]);
    const double ml01 = 0.5 * (m0[1] + m1[1]);
    const double ml11 = 0.5 * (m0[2] + m0[2]);

    double x = coords[2 * il] - x0;
    double y = coords[2 * il + 1] - y0;

    q[0] += (ml00 * x + ml01 * y);
    q[1] += (ml01 * x + ml11 * y);

    A[0] += ml00;
    A[1] += ml01;
    A[3] += ml11;
  }

  A[2] = A[1];
  double p[2];

  p[1] = (q[1] - A[2] / A[0]) / (A[3] - A[1] * A[2] / A[0]);
  p[0] = (q[0] - A[1] * p[1]) / A[0];

  if(isSurfaceNode(vid, NEListOffs, NNListOffs)) {
    p[0] -= p[0] * fabs(normals[2 * vid]);
    p[1] -= p[1] * fabs(normals[2 * vid + 1]);
  }

  //TODO: optimization: check if p.x and p.y are actually nonzero
  coords[2 * vid]     += p[0];
  coords[2 * vid + 1] += p[1];

  double new_worst_q = 1.0;

  for(int i = NEListOffs[vid]; i < NEListOffs[vid + 1]; ++i) {
    uint eid = NEList[i];
    double qual = element_quality(eid, ENList, metric, coords, orientation);
    new_worst_q = min(new_worst_q, qual);
  }

  if (new_worst_q < worst_q) {
    coords[2 * vid]     -= p[0];
    coords[2 * vid + 1] -= p[1];
  }
}
