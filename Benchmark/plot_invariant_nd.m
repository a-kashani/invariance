function plot_invariant_nd(example, n, z, g, V_func, center_bounds)
% PLOT_INVARIANT_ND  Pairwise scatter subplots for n >= 4 systems.
%   Each subplot projects data onto one (xi, xj) pair.
%   Blue = in X \ O,  Orange = in invariant set O.

col_X  = [0.35, 0.65, 1.00];
col_O  = [0.85, 0.33, 0.10];
pairs  = nchoosek(1:n, 2);
np     = size(pairs, 1);
V_data = V_func(z);
in_O   = V_data <= 1;

switch n
    case 4,  nr = 2; nc = 3; fig_w = 900;  fig_h = 560;
    case 5,  nr = 2; nc = 5; fig_w = 1200; fig_h = 500;
    otherwise
        fprintf('Visualization skipped for %dD system (%s).\n', n, example);
        return
end

figure
set(gcf, 'Position', [100, 100, fig_w, fig_h])

% For moon_lander pre-compute constraint image in the x1-x2 plane
x1v = []; x2v = []; img_c = [];
if strcmp(example, 'moon_lander')
    npc  = 300;
    x1v  = linspace(center_bounds(1,1), center_bounds(1,2), npc);
    x2v  = linspace(center_bounds(2,1), center_bounds(2,2), npc);
    [X1g, X2g] = meshgrid(x1v, x2v);
    xc   = [X1g(:)'; X2g(:)'; zeros(2, npc^2); ones(1, npc^2)];
    in_Xc = reshape(g(xc) <= 1, npc, npc);
    img_c = ones(npc, npc, 3);
    for c = 1:3
        ch = img_c(:,:,c); ch(in_Xc) = col_X(c); img_c(:,:,c) = ch;
    end
end

for k = 1:np
    ii = pairs(k,1); jj = pairs(k,2);
    subplot(nr, nc, k); hold on; box on
    if strcmp(example, 'moon_lander') && ii == 1 && jj == 2
        image(x1v, x2v, img_c); set(gca, 'YDir', 'normal')
    end
    scatter(z(ii,~in_O), z(jj,~in_O), 4, col_X, 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(z(ii, in_O), z(jj, in_O), 8, col_O, 'filled');
    xlabel(['$x_{' num2str(ii) '}$'], 'Interpreter','latex','FontSize',12)
    ylabel(['$x_{' num2str(jj) '}$'], 'Interpreter','latex','FontSize',12)
end

hX = fill(nan, nan, col_X, 'EdgeColor','none');
hO = fill(nan, nan, col_O, 'EdgeColor','none');
legend([hX, hO], {'Constraint $\mathcal{X}$','Data-driven $\mathcal{O}$'}, ...
       'Interpreter','latex','FontSize',12,'EdgeColor','none','Color','none', ...
       'Orientation','horizontal','Location','south');
sgtitle(strrep(example,'_','\_'), 'FontSize', 14)
end
