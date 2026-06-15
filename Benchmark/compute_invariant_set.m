function [theta, centers, spacing_vec, t_rbf, t_lp] = compute_invariant_set( ...
    z, z_p, g, n_centers, RBFtype, Guarantee, Lyapunov_discount, delta, L_phi, L_f, L_g, lp_timeout_min)
% COMPUTE_INVARIANT_SET  Fit a barrier function V = theta' * phi(x) via LP.
%
%   Inputs:
%     z, z_p          [n x N] state samples and one-step successors (pre-filtered to X)
%     g               constraint function handle: g(x) <= 1 defines X
%     n_centers       number of RBF centers per dimension
%     RBFtype         RBF kernel string (e.g. 'bump', 'thinplate')
%     Guarantee       'probabilistic' or 'deterministic'
%     Lyapunov_discount, delta, L_phi, L_f, L_g  — LP parameters
%
%   Outputs:
%     theta       [lift_dim x 1] barrier function coefficients
%     centers     [n x M] RBF center locations
%     spacing_vec [n x 1] spacing between centers per dimension
%     t_rbf, t_lp timing in seconds

n           = size(z, 1);
num_samples = size(z, 2);

%% RBF center grid
tic
center_bounds = [min(z,[],2), max(z,[],2)];
grid_vecs     = cell(n, 1);
for i = 1:n
    grid_vecs{i} = linspace(center_bounds(i,1), center_bounds(i,2), n_centers);
end
grids = cell(n, 1);
[grids{:}] = ndgrid(grid_vecs{:});
centers = zeros(n, numel(grids{1}));
for i = 1:n
    centers(i,:) = grids{i}(:)';
end
centers     = centers(:, g(centers) <= 1);
spacing_vec = (center_bounds(:,2) - center_bounds(:,1)) / (n_centers - 1);

%% Lift data
g_value    = g(z);
lifted_z   = [g(z);   rbf(z,   centers, RBFtype, spacing_vec); ones(1, num_samples)];
lifted_z_p = [g(z_p); rbf(z_p, centers, RBFtype, spacing_vec); ones(1, num_samples)];
lift_dim   = size(lifted_z, 1);
t_rbf      = toc;
fprintf('RBF lifting:   %.2f s  (%d centers, %d features, %d samples)\n', ...
        t_rbf, size(centers,2), lift_dim, num_samples);

%% Solve LP
fprintf('LP: %d vars, %d constraints  [%s]\n', 2*lift_dim, num_samples, Guarantee);
if nargin < 12 || isempty(lp_timeout_min) || lp_timeout_min <= 0
    opts = optimoptions('linprog', 'Display', 'off');
else
    opts = optimoptions('linprog', 'MaxTime', lp_timeout_min * 60, 'Display', 'off');
end
tic
switch lower(Guarantee)
    case 'probabilistic'
        A = zeros(2*num_samples, lift_dim);
        b = zeros(2*num_samples, 1);
        for i = 1:num_samples
            if g(z_p(:,i)) <= 1
                A(i,:) = lifted_z_p(:,i)' - (1-Lyapunov_discount)*lifted_z(:,i)';
                b(i)   = Lyapunov_discount;
            else
                A(i,:) = -(1-Lyapunov_discount)*lifted_z(:,i)';
                b(i)   = -g(z_p(:,i)) + Lyapunov_discount;
            end
        end
        A(num_samples+1:end,:) = -lifted_z';
        b(num_samples+1:end)   = -g_value';
        [theta, ~, exitflag] = linprog(sum(lifted_z,2)', A, b, [], [], [], [], opts);
        if exitflag <= 0
            warning('LP did not converge (exitflag=%d). Skipping run.', exitflag);
            theta = []; t_lp = toc; return
        end

    case 'deterministic'
        A = zeros(num_samples, 2*lift_dim);
        b = zeros(num_samples, 1);
        for i = 1:num_samples
            if g(z_p(:,i)) <= 1
                A(i,1:lift_dim)     = -(1-Lyapunov_discount)*lifted_z(:,i)' + lifted_z_p(:,i)' ...
                                       + L_phi*delta*(1-Lyapunov_discount+L_f);
                A(i,lift_dim+1:end) =  (1-Lyapunov_discount)*lifted_z(:,i)' - lifted_z_p(:,i)' ...
                                       + L_phi*delta*(1-Lyapunov_discount+L_f);
                b(i) = Lyapunov_discount;
            else
                A(i,1:lift_dim)     = -(1-Lyapunov_discount)*lifted_z(:,i)' + L_phi*delta*(1-Lyapunov_discount);
                A(i,lift_dim+1:end) =  (1-Lyapunov_discount)*lifted_z(:,i)' + L_phi*delta*(1-Lyapunov_discount);
                b(i) = -g(z_p(:,i)) + Lyapunov_discount - delta*L_g*L_f;
            end
        end
        mV = sum(lifted_z, 2)';
        [theta_pn, ~, exitflag] = linprog([mV+abs(mV), -mV+abs(mV)], A, b, [], [], zeros(2*lift_dim,1), [], opts);
        if exitflag <= 0
            warning('LP did not converge (exitflag=%d). Skipping run.', exitflag);
            theta = []; t_lp = toc; return
        end
        theta = theta_pn(1:lift_dim) - theta_pn(lift_dim+1:end);
end
t_lp = toc;
fprintf('LP solve:      %.2f s\n', t_lp);
end
