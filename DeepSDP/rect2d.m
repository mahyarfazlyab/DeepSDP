function X = rect2d(x_min,x_max)

x1 = linspace(x_min(1),x_max(1),50);
x2 = linspace(x_min(2),x_max(2),50);

[X1,X2] = meshgrid(x1,x2);

X(1,:) = X1(:);
X(2,:) = X2(:);



end