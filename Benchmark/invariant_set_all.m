%% ========================================================================
% Author: Ali Kashani
% Description:
% Data-driven invariant set computation using RBF + LP.
% Calls: get_constraint, compute_invariant_set, plot_invariant_2d/3d/nd
%
% Set any of the sweep parameters below to a vector to run all combinations:
%   example_id        — scalar or vector,  e.g. 1:10
%   Guarantee_id      — scalar or vector,  e.g. 1:2
%   Lyapunov_discount — scalar or vector,  e.g. [0, 0.1, 0.2, 0.3, 0.5]
%   RBFtype_id        — scalar or vector,  e.g. [3, 4]
%   n_centers         — scalar or vector,  e.g. [15, 31, 51]
% All five can be combined; total runs = product of their lengths.

%% Workspace
clear; close all; clc
t_total = tic;

%% Sweep Parameters  (set vectors to run multiple configurations)
example_id        = [1];                        % scalar or vector — see list below
Guarantee_id      = [1];                        % scalar or vector — 1: probabilistic, 2: deterministic
Lyapunov_discount = [0.1];                      % scalar or vector
RBFtype_id        = [4];                        % scalar or vector — see list below
n_centers         = [21];                       % scalar or vector — RBF centers per dimension

%% Fixed Parameters
lp_timeout_min = 1;   % LP wall-clock timeout in minutes (0 = no limit)
delta = 0;
L_phi = 1; L_f = 1.1; L_g = 1;

%% Lookup tables
examples   = {'linear','soft_landing','soft_landing_relaxed','pendulum','van_der_pol', ...
              'inverted_pendulum','duffing','julia','bicycle','power', ...
              'lorenz','cornhole','double_pendulum','moon_lander'};
RBFtypes   = {'pyramid','triangle','thinplate','gauss','invquad','invmultquad','polyharmonic','bump'};
%            1: pyramid  2: triangle  3: thinplate  4: gauss  5: invquad
%            6: invmultquad  7: polyharmonic  8: bump
Guarantees = {'probabilistic','deterministic'};

n_runs = numel(example_id) * numel(Guarantee_id) * numel(Lyapunov_discount) ...
       * numel(RBFtype_id) * numel(n_centers);
fprintf('=== Sweep: %d run(s) total ===\n\n', n_runs);
run_idx = 0;

%% Main sweep loop
for ex_id = example_id(:)'
for gu_id = Guarantee_id(:)'
for ld    = Lyapunov_discount(:)'
for rb_id = RBFtype_id(:)'
for nc    = n_centers(:)'

run_idx   = run_idx + 1;
example   = examples{ex_id};
Guarantee = Guarantees{gu_id};
RBFtype   = RBFtypes{rb_id};

fprintf('\n[%d/%d] %s | %s | ld=%.2f | %s | nc=%d\n', ...
        run_idx, n_runs, example, Guarantee, ld, RBFtype, nc);
fprintf('%s\n', repmat('-', 1, 60));
t_run = tic;

%% Load data
data_file = example;
if strcmp(example, 'soft_landing_relaxed'), data_file = 'soft_landing'; end
load(fullfile('data_grid', [data_file, '.mat']))
n = size(z, 1);
if ~exist('Oinf', 'var'), Oinf = []; end

%% Constraint set
g = get_constraint(example);

%% Filter samples outside X
g_value   = g(z);
in_X_mask = g_value <= 1;
z_filt    = z(:, in_X_mask);
z_p_filt  = z_p(:, in_X_mask);
fprintf('Samples in X: %d / %d (%.1f%%)\n', size(z_filt,2), numel(in_X_mask), 100*mean(in_X_mask));

%% Compute invariant set
[theta, centers, spacing_vec, t_rbf, t_lp] = compute_invariant_set( ...
    z_filt, z_p_filt, g, nc, RBFtype, Guarantee, ...
    ld, delta, L_phi, L_f, L_g, lp_timeout_min);

if isempty(theta)
    fprintf('  --> LP failed or timed out — skipping this run.\n');
    continue
end

%% Barrier function handle
V_func = @(x) (theta' * [g(x); rbf(x, centers, RBFtype, spacing_vec); ones(1,size(x,2))]) ...
               .* (g(x) <= 1) + g(x) .* (g(x) > 1);

center_bounds = [min(z_filt,[],2), max(z_filt,[],2)];
V_data = V_func(z_filt);
fprintf('States in learned set: %d / %d (%.1f%%)\n', sum(V_data<=1), size(z_filt,2), 100*mean(V_data<=1));

%% Visualize
tic
if n == 2
    plot_invariant_2d(example, g, V_func, center_bounds, Oinf);
elseif n == 3
    plot_invariant_3d(example, g, V_func, center_bounds, Oinf);
else
    plot_invariant_nd(example, n, z_filt, g, V_func, center_bounds);
end
t_plot = toc;
fprintf('Visualization: %.2f s\n', t_plot);

%% Save
results_dir = fullfile('results', example);
if ~exist(results_dir, 'dir'), mkdir(results_dir); end
base_name = sprintf('%s_%s_%s_ld%.2f_nc%d', example, RBFtype, Guarantee, ld, nc);

if n <= 5
    fig_base = fullfile(results_dir, base_name);
    savefig(gcf, [fig_base, '.fig']);
    exportgraphics(gcf, [fig_base, '.png'], 'Resolution', 300);
end

save(fullfile(results_dir, [base_name, '_barrier.mat']), ...
    'example', 'theta', 'centers', 'spacing_vec', 'RBFtype', ...
    'Guarantee', 'ld', 'nc', 'n', 'delta', 'L_phi', 'L_f', 'L_g');

fprintf('Saved to %s  (%.2f s)\n', results_dir, toc(t_run));

end % n_centers
end % RBFtype_id
end % Lyapunov_discount
end % Guarantee_id
end % example_id

fprintf('\n=== All runs complete. Total: %.2f s ===\n', toc(t_total));
