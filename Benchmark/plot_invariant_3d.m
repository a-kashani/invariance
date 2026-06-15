function plot_invariant_3d(example, g, V_func, center_bounds, Oinf)
% PLOT_INVARIANT_3D  Isosurface + constraint box visualization for 3D systems.

if nargin < 5, Oinf = []; end

n_plot3 = 60;
xlo = center_bounds(1,1); xhi = center_bounds(1,2);
ylo = center_bounds(2,1); yhi = center_bounds(2,2);
zlo = center_bounds(3,1); zhi = center_bounds(3,2);
[xx, yy, zz] = meshgrid(linspace(xlo,xhi,n_plot3), linspace(ylo,yhi,n_plot3), linspace(zlo,zhi,n_plot3));
x_grid = [xx(:)'; yy(:)'; zz(:)'];
V_grid = reshape(V_func(x_grid), size(xx));

figure
set(gcf, 'Position', [100, 100, 560, 440])
hold on

p  = isosurface(xx, yy, zz, V_grid, 1);
hp = patch(p, 'FaceColor', [0.85,0.33,0.10], 'FaceAlpha', 1.0, 'EdgeColor', 'none');
isonormals(xx, yy, zz, V_grid, hp);

cb = [0.10 0.35 0.90]; lw = 1.5;
plot3([xlo xhi xhi xlo xlo],[ylo ylo yhi yhi ylo],[zlo zlo zlo zlo zlo],'-','Color',cb,'LineWidth',lw);
plot3([xlo xhi xhi xlo xlo],[ylo ylo yhi yhi ylo],[zhi zhi zhi zhi zhi],'-','Color',cb,'LineWidth',lw);
plot3([xlo xlo],[ylo ylo],[zlo zhi],'-','Color',cb,'LineWidth',lw);
plot3([xhi xhi],[ylo ylo],[zlo zhi],'-','Color',cb,'LineWidth',lw);
plot3([xhi xhi],[yhi yhi],[zlo zhi],'-','Color',cb,'LineWidth',lw);
plot3([xlo xlo],[yhi yhi],[zlo zhi],'-','Color',cb,'LineWidth',lw);
hX3 = plot3(nan,nan,nan,'-','Color',cb,'LineWidth',lw);

if ~isempty(Oinf)
    K = convhull(Oinf);
    trisurf(K, Oinf(:,1), Oinf(:,2), Oinf(:,3), ...
            'FaceColor','cyan','FaceAlpha',0.3,'EdgeColor','none');
end
legend([hX3, hp], {'Constraint $\mathcal{X}$','Data-driven $\mathcal{O}$'}, ...
       'Interpreter','latex','FontSize',14,'EdgeColor','none','Color','none', ...
       'Orientation','horizontal','Location','north');

lighting gouraud; camlight; axis equal; view(3)
xlabel('$x_1$','Interpreter','latex','FontSize',20)
ylabel('$x_2$','Interpreter','latex','FontSize',20)
zlabel('$x_3$','Interpreter','latex','FontSize',20)
title(strrep(example,'_','\_'), 'FontSize', 14)
end
