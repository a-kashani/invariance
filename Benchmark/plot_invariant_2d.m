function plot_invariant_2d(example, g, V_func, center_bounds, Oinf)
% PLOT_INVARIANT_2D  RGB image + contour visualization for 2D systems.

if nargin < 5, Oinf = []; end

col_X  = [0.35, 0.65, 1.00];
col_O  = [0.85, 0.33, 0.10];
n_plot = 300;

xlo = center_bounds(1,1); xhi = center_bounds(1,2);
ylo = center_bounds(2,1); yhi = center_bounds(2,2);
[xx, yy] = meshgrid(linspace(xlo,xhi,n_plot), linspace(ylo,yhi,n_plot));
x_grid = [xx(:)'; yy(:)'];
V_grid = reshape(V_func(x_grid), size(xx));
g_grid = reshape(g(x_grid),      size(xx));

img  = ones(n_plot, n_plot, 3);
in_X = g_grid <= 1;
in_O = V_grid <= 1;
for c = 1:3
    ch = img(:,:,c);
    ch(in_X) = col_X(c);
    ch(in_O) = col_O(c);
    img(:,:,c) = ch;
end

figure
set(gcf, 'Position', [100, 100, 560, 440])
image(linspace(xlo,xhi,n_plot), linspace(ylo,yhi,n_plot), img)
set(gca, 'YDir', 'normal')
hold on
contour(xx, yy, g_grid, [1 1], 'Color', [0.10 0.35 0.90], 'LineWidth', 2);
contour(xx, yy, V_grid, [1 1], 'Color', [0.60 0.10 0.00], 'LineWidth', 1.5);

hX = fill(nan, nan, col_X, 'EdgeColor', 'none');
hO = fill(nan, nan, col_O, 'EdgeColor', 'none');
if ~isempty(Oinf)
    K    = convhull(Oinf(:,1:2));
    hInf = patch(Oinf(K,1), Oinf(K,2), 'cyan', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    legend([hX, hO, hInf], {'Constraint $\mathcal{X}$','Data-driven $\mathcal{O}$','Model-based $\mathcal{O}_\infty$'}, ...
           'Interpreter','latex','FontSize',14,'EdgeColor','none','Color','none', ...
           'Orientation','horizontal','Location','north');
else
    legend([hX, hO], {'Constraint $\mathcal{X}$','Data-driven $\mathcal{O}$'}, ...
           'Interpreter','latex','FontSize',14,'EdgeColor','none','Color','none', ...
           'Orientation','horizontal','Location','north');
end
yl = ylim; ylim([yl(1), yl(2) + 0.12*(yl(2)-yl(1))]);
xlabel('$x_1$','Interpreter','latex','FontSize',20)
ylabel('$x_2$','Interpreter','latex','FontSize',20)
title(strrep(example,'_','\_'), 'FontSize', 14)
end
