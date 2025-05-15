%% ========================================================================
% Author: Ali Kashani
% Description:
% This script computes data-driven approximations of invariant sets using
% both probabilistic and deterministic guarantees. The approach is based on 
% radial basis function (RBF) representations and linear programming.

%% preparing the workspace
clear
close all
clc

%% Control Parameters
spacing = 0.2;                 % center spacing of RBFs in each dimension
Lyapunov_discount = 0.1;       %  0 < Lyapunov discount factor < 1
RBFtype = 'bump';              % Kernel type for RBF
Guarantee = 'deterministic';   % 'deterministic' or 'probabilistic'
delta = 0;                     % Data Density 
L_phi = 1; L_f = 1.1; L_g = 1; % Lipschitz constants

%% Loading data and defining the constraint set
load data.mat                  % Loads samples for the dynamics z_p = f(z), where z=(x,u). Oinf is admissible set
num_samples = size(z,2);       % Number of samples

% Constraint Set Definition: g(x) <= 1 defines the constaint set X
g = @(x) max(abs(x(1:2,:)));   % Unit box constraint over the states x=z(1:2)\in X
g_value = g(z);                % value of g(x) on samples


%% Generate Basis Functions
spacing3D = spacing * [1; 1; 1];
[c1, c2, c3] = meshgrid(-1:spacing3D(1):1, -1:spacing3D(2):1, -1:spacing3D(3):1); % Create RBF centers
centers = [c1(:)'; c2(:)'; c3(:)'];  % Flatten to center matrix

% Evaluate RBFs at data points
lifted_z = rbf(z, centers, RBFtype, spacing3D);      
lifted_z_p = rbf(z_p, centers, RBFtype, spacing3D);

% Augment basis with constraint and bias term
lifted_z = [g(z); lifted_z; ones(1, num_samples)];
lifted_z_p = [g(z_p); lifted_z_p; ones(1, num_samples)];
lift_dim = size(lifted_z,1);      % Dimension of augmented feature vector


%% Setup and Solve Linear Program
    disp("Computing Control Barrier Function V(x)")
    fprintf("Solving linear program with %d parameters and %d constraints \n", 2*lift_dim, num_samples);
    disp("...")
tic
switch Guarantee
    case "probabilistic"
        % Initialize LP matrices
        A = zeros(2*num_samples, lift_dim);        % LP inequality Ax <= b
        b = zeros(2*num_samples, 1);
        
        for i = 1:num_samples
            if g(z_p(:,i)) <= 1                    % if state successor x_p in X
                % x_p inside the constraint set
                A(i,:) = lifted_z_p(:,i)' - (1 - Lyapunov_discount) * lifted_z(:,i)';
                b(i) = Lyapunov_discount;
            else
                % x_p outside the constraint set
                A(i,:) = -(1 - Lyapunov_discount) * lifted_z(:,i)';
                b(i) = -g(z_p(:,i)) + Lyapunov_discount;
            end
        end
        
        % Add constraint V(x) ≥ g(x)
        A(num_samples+1:end,:) = -lifted_z';
        b(num_samples+1:end) = -g_value';

        % LP objective: minimize sum of V(x)
        cost_coef = sum(lifted_z, 2)';
        theta = linprog(cost_coef, A, b);  % Solve LP

    case 'deterministic'
        % Initialize LP matrices
        A = zeros(num_samples, 2*lift_dim);     % LP inequality Ax <= b
        b = zeros(num_samples, 1);

        for i = 1:num_samples
            if g(z_p(:,i)) <= 1          % if state successor x_p in X
                % x_p inside the constraint set
                A(i,1:lift_dim) = -(1 - Lyapunov_discount) * lifted_z(:,i)' + lifted_z_p(:,i)' + L_phi * delta * (1 - Lyapunov_discount + L_f);
                A(i,lift_dim+1:end) = (1 - Lyapunov_discount) * lifted_z(:,i)' - lifted_z_p(:,i)' + L_phi * delta * (1 - Lyapunov_discount + L_f);
                b(i) = Lyapunov_discount;
            else
                % x_p outside the constraint set
                A(i,1:lift_dim) = -(1 - Lyapunov_discount) * lifted_z(:,i)' + L_phi * delta * (1 - Lyapunov_discount);
                A(i,lift_dim+1:end) = (1 - Lyapunov_discount) * lifted_z(:,i)' + L_phi * delta * (1 - Lyapunov_discount);
                b(i) = -g(z_p(:,i)) + Lyapunov_discount - delta * L_g * L_f;
            end
        end

        % Objective: minimize a measure of V + regularization 
        measure_V = sum(lifted_z, 2)'; 
        cost_coef = [measure_V + abs(measure_V), -measure_V + abs(measure_V)];
        theta_pn = linprog(cost_coef, A, b, [], [], zeros(2*lift_dim,1));    % theta_pn=[theta_p; theta_n]
        theta = theta_pn(1:lift_dim) - theta_pn(lift_dim+1:end);              % theta= theta_p-theta_n
end
toc
%% Compute Value Function V(z)
V = (theta' * lifted_z) .* (g_value <= 1) + g_value .* (g_value > 1);


%% Visualization
figure
hold on

% Draw convex hull of the learned set
K = convhull(z(:, V <= 1)');
patch('Faces', K, 'Vertices', z(:, V <= 1)', ...
      'FaceColor', [0.85, 0.33, 0.10], 'FaceAlpha', 1, 'EdgeColor', 'none');

% Plot model-based O_inf
K = convhull(1*Oinf);
trisurf(K, Oinf(:,1), Oinf(:,2), Oinf(:,3), 'FaceColor', 'cyan', 'FaceAlpha', 0.3);
axis equal
view(3)

xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 20)
zlabel('$u$', 'Interpreter', 'latex', 'FontSize', 20)

legend("Data-driven $\mathcal{O}$", "Model-based $\mathcal{O}_\infty$", ...
    'Interpreter', 'latex', 'FontSize', 14, 'NumColumns', 3, ...
    'EdgeColor', 'none', 'Color', 'none', ...
    'Position', [0.0154, 0.8894, 0.9506, 0.1204]);

set(gcf, 'Position', [340, 350, 420, 462]);
set(gca, 'Position', [0.128, 0.129, 0.867, 0.775], ...
    'FontSize', 16, 'XTick', -1:1, 'YTick', -1:1, 'ZTick', -1:1);
