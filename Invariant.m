% Author: Ali Kashani
% Visualization of Invariant Sets with Increasing Sample Resolution
% This script plots several approximations of an invariant set using 
% different discretization densities.
%

%% 
clear
close all
clc

% Generate approximations of the invariant set with varying sample sizes
% Each call returns a polygon defined by vertices arranged in a matrix.
O1 = invariant_set(0.4, 20,  'triangle');  % Coarse sampling
O2 = invariant_set(0.2, 60,  'triangle');  % Medium sampling
O3 = invariant_set(0.1, 100, 'triangle');  % Fine sampling
O4 = invariant_set(0.1, 150, 'triangle');  % Very fine sampling

%% Plotting
figure
hold on
set(gcf, 'Position', [338, 341, 700, 420], 'Color', 'white');  % Set figure size and background

% Plot constraint set \mathcal{X} as a light gray square
x = [1.01 1.01 -1.01 -1.01];
y = [1.01 -1.01 -1.01 1.01];
patch(x, y, 'k', 'FaceColor', [0.9, 0.9, 0.9], 'EdgeColor', 'k', 'LineWidth', 2, 'LineStyle', '-');

% Plot the true invariant set (must be defined elsewhere in workspace as Oinf)
load Model_PI_set.mat
patch(Oinf(:,1), Oinf(:,2), 'b', 'LineStyle', 'none', 'LineWidth', 2, 'FaceColor', [0.47, 0.67, 0.19]);

% Overlay approximated invariant sets with distinct styles and colors
patch(O1(1,2:end), O1(2,2:end), [0.64, 0.08, 0.18], 'EdgeColor', [0.64, 0.08, 0.18], 'LineStyle', '-',   'LineWidth', 2, 'FaceAlpha', 0);
patch(O2(1,2:end), O2(2,2:end), [0.49, 0.18, 0.56], 'EdgeColor', [0.49, 0.18, 0.56], 'LineStyle', '-.',  'LineWidth', 2, 'FaceAlpha', 0);
patch(O3(1,2:end), O3(2,2:end), [0.93, 0.69, 0.13], 'EdgeColor', [0.93, 0.69, 0.13], 'LineStyle', '--',  'LineWidth', 2, 'FaceAlpha', 0);
patch(O4(1,2:end), O4(2,2:end), [0.85, 0.33, 0.10], 'EdgeColor', [0.85, 0.33, 0.10], 'LineStyle', ':',   'LineWidth', 2, 'FaceAlpha', 0);

% Axis labels using LaTeX formatting
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 20);

% Legend describing each set
legend("$\mathcal{X}$", "$\mathcal{O}_\infty$", ...
       "$n_\theta=6^2, n_s=20^2$", ...
       "$n_\theta=11^2, n_s=60^2$", ...
       "$n_\theta=21^2, n_s=100^2$", ...
       "$n_\theta=21^2, n_s=150^2$", ...
       'Interpreter', 'latex', 'FontSize', 20, 'NumColumns', 3, ...
       'EdgeColor', 'none', 'Color', 'none', ...
       'Position', [0.0154, 0.8894, 0.9506, 0.044]);

% Adjust axis settings
set(gca, 'Position', [0.09, 0.18, 0.92, 0.67], ...
         'FontSize', 20, 'XTick', -1:1, 'YTick', -1:1);
xlim([-1.1, 1.1]);
ylim([-1.1, 1.1]);
