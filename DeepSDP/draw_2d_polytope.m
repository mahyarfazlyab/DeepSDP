function h = draw_2d_polytope(X,Y,color,name)

m = length(X);


for i=1:m-1
    plot([X(i),X(i+1)],[Y(i),Y(i+1)],color,'LineWidth',2);hold on;
end
h = plot([X(m),X(1)],[Y(m),Y(1)],color,'LineWidth',2,'DisplayName',name);

%legend(h(m));

end