Point(1) = {0, 0, 0, 1.0};
Point(2) = {10, 0, 0, 1.0};
Point(3) = {10, 2, 0, 1.0};
Point(4) = {10, 10, 0, 1.0};
Point(5) = {0, 10, 0, 1.0};
Point(6) = {0, 8, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {6, 3};
Line Loop(8) = {1, 2, -7, 6};
Plane Surface(9) = {8};
Line Loop(10) = {4, 5, 7, 3};
Plane Surface(11) = {10};
Physical Line(12) = {1};
Physical Line(13) = {2};
Physical Line(14) = {3};
Physical Line(15) = {4};
Physical Line(16) = {5};
Physical Line(17) = {6};
Physical Line(18) = {7};
Physical Surface(19) = {9};
Physical Surface(20) = {11};
