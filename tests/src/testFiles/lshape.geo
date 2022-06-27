// Gmsh project created on Wed Jun 15 08:50:43 2022
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 0.5, 0, 1.0};
//+
Point(4) = {0.5, 0.5, 0, 1.0};
//+
Point(5) = {0.5, 1, 0, 1.0};
//+
Point(6) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 1};
//+
Curve Loop(1) = {5, 6, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve("homogenous_dirichlet_bc", 7) = {6, 1};
//+
Physical Curve("inhomogenous_dirichlet_bc", 8) = {4, 3};
//+
Physical Curve("neumann_bc", 9) = {5, 2};
//+
Transfinite Curve {6} = 8 Using Progression 1;
//+
Transfinite Curve {4} = 4 Using Progression 1;
//+
Transfinite Curve {2} = 4 Using Progression 1;
//+
Transfinite Curve {5} = 4 Using Progression 1;
//+
Transfinite Curve {3} = 4 Using Progression 1;
//+
Transfinite Curve {1} = 8 Using Progression 1;
//+
Recombine Surface {1};
