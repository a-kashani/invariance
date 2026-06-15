function gen_data(example, bounds, npts, dt, filename)
% GEN_DATA  Generate grid-sampled state-transition data for invariant set benchmarks.
%
%   GEN_DATA(EXAMPLE, BOUNDS, NPTS, DT) generates a uniform grid of
%   state points z and their one-step successors z_p = f(z), then saves
%   them to <example>.mat in the current directory.
%
%   GEN_DATA(EXAMPLE, BOUNDS, NPTS, DT, FILENAME) saves to FILENAME instead.
%
%   Inputs:
%     EXAMPLE   - string identifying the benchmark (see list below)
%     BOUNDS    - [n x 2] matrix: [lo_1, hi_1; ...; lo_n, hi_n]
%     NPTS      - scalar or [n x 1] vector: number of grid points per dimension
%     DT        - sampling time (used for continuous-time systems)
%     FILENAME  - output .mat filename (default: '<example>.mat')
%
%   Saved variables:
%     z   - [n x N] state samples
%     z_p - [n x N] one-step successors  z_p(:,i) = f(z(:,i))
%
%   Supported examples (string must match exactly, case-insensitive):
%     'linear'              Under-damped 2nd-order linear system
%     'soft_landing'        Soft-landing with critically-damped controller
%     'cornhole'            Ballistic bean-bag (linear, 4D, non-convex constraints)
%     'moon_lander'         Multi-site lunar lander (switched, 5D)
%     'pendulum'            Nonlinear pendulum (2D)
%     'inverted_pendulum'   Inverted pendulum with linear controller (2D)
%     'double_pendulum'     Double pendulum (Euler-Lagrange, 4D)
%     'julia'               Julia recursion (discrete-time map, 2D)
%     'van_der_pol'         Van der Pol oscillator (2D)
%     'duffing'             Duffing oscillator (2D)
%     'lorenz'              Lorenz attractor (3D)
%     'bicycle'             Kinematic bicycle lane keeping (2D)
%     'power'               Two-machine power system (2D)
%
%   Example usage:
%     % Van der Pol: 61x61 grid
%     gen_data('van_der_pol', [-3,3;-3,3], 61, 0.1)
%
%     % Lorenz: 21x31x26 grid (different points per dim)
%     gen_data('lorenz', [-20,20;-30,30;0,50], [21;31;26], 0.01)
%
%     % Julia recursion (discrete-time, dt is ignored)
%     gen_data('julia', [-1,1;-1,1], 101, [])

if nargin < 5 || isempty(filename)
    filename = fullfile('data_grid', [example, '.mat']);
end
if ~exist('data_grid', 'dir'), mkdir('data_grid'); end

n = size(bounds, 1);
if isscalar(npts)
    npts = repmat(npts, n, 1);
end
npts = round(npts(:));

%% Build uniform grid

grid_vecs = cell(n, 1);
for i = 1:n
    grid_vecs{i} = linspace(bounds(i,1), bounds(i,2), npts(i));
end

% ndgrid gives [n1 x n2 x ... x nk] arrays, one per dimension
grids = cell(n, 1);
[grids{:}] = ndgrid(grid_vecs{:});
N = numel(grids{1});
z = zeros(n, N);
for i = 1:n
    z(i,:) = grids{i}(:)';
end

fprintf('Example: %s  |  state dim: %d  |  grid points: %d\n', example, n, N);

%% Select dynamics

[f, is_discrete] = get_dynamics(example);

%% Compute one-step successors

z_p = zeros(n, N);
if is_discrete
    for i = 1:N
        z_p(:,i) = f(z(:,i));
    end
else
    if isempty(dt) || dt <= 0
        error('dt must be a positive scalar for continuous-time systems.');
    end
    % Vectorized ODE integration — process each point independently
    for i = 1:N
        [~, xT] = ode45(@(t,x) f(x), [0, dt], z(:,i));
        z_p(:,i) = xT(end,:)';
    end
end

%% Save

save(filename, 'z', 'z_p');
fprintf('Saved %d sample pairs to %s\n', N, filename);
end


%% =========================================================================
%  Dynamics selector
%% =========================================================================

function [f, is_discrete] = get_dynamics(example)
is_discrete = false;

switch lower(strtrim(example))

    % ------------------------------------------------------------------
    case 'linear'
        % Under-damped 2nd-order: wn=5, zeta=0.1
        % Constraint: |x1| <= 1
        wn = 5; zeta = 0.1;
        A = [0, 1; -wn^2, -2*zeta*wn];
        f = @(x) A * x;

    % ------------------------------------------------------------------
    case 'soft_landing'
        % Critically-damped closed-loop: k1=1, k2=2
        % Constraint: -gamma_lo*x1 <= x2 <= -gamma_hi*x1
        k1 = 1; k2 = 2;
        A = [0, 1; -k1, -k2];
        f = @(x) A * x;

    % ------------------------------------------------------------------
    case 'cornhole'
        % Ballistic double-integrator with gravity (4D)
        % States: x1=horiz pos, x2=vert pos, x3=horiz vel, x4=vert vel
        % Constraint: non-convex — bean-bag passes through cornhole
        g_acc = 9.81;
        A = [0, 0, 1, 0;
             0, 0, 0, 1;
             0, 0, 0, 0;
             0, 0, 0, 0];
        d = [0; 0; 0; -g_acc];
        f = @(x) A * x + d;

    % ------------------------------------------------------------------
    case 'moon_lander'
        % Switched PD-controlled multi-site lunar lander (5D)
        % States: x1=vert pos, x2=horiz pos, x3=vert vel, x4=horiz vel, x5=fuel
        % Landing sites r_a, r_b at horizontal positions -2 and +2
        kp_v = 2; kd_v = 1;   % vertical PD gains (critically damped)
        kp_h = 1; kd_h = 1;   % horizontal PD gains (under-damped)
        r_a  = -2; r_b = 2;   % landing site horizontal positions
        f = @(x) moon_lander_f(x, kp_v, kd_v, kp_h, kd_h, r_a, r_b);

    % ------------------------------------------------------------------
    case 'pendulum'
        % Nonlinear pendulum: wn=5, zeta=0.1 (linearizes to 'linear')
        % Constraint: |x1| <= 1
        wn = 5; zeta = 0.1;
        f = @(x) [x(2); -2*zeta*wn*x(2) - wn^2*sin(x(1))];

    % ------------------------------------------------------------------
    case 'inverted_pendulum'
        % Inverted pendulum + linear stabilizing controller
        % No explicit state constraint (CAPI set = region of attraction)
        k1 = 4.4142; k2 = 2.3163;
        f = @(x) [x(2); sin(x(1)) - x(2) - k1*x(1) - k2*x(2)];

    % ------------------------------------------------------------------
    case 'double_pendulum'
        % Double pendulum (Euler-Lagrange, 4D), equal masses and lengths
        % States: x1=theta1, x2=theta2, x3=dtheta1, x4=dtheta2
        % Constraint: wall collisions  |l*sin(x1)| <= 1, |l*sin(x1)+l*sin(x2)| <= 1
        m_dp = 1; l_dp = 1; g_dp = 9.81;
        f = @(x) double_pendulum_f(x, m_dp, l_dp, g_dp);

    % ------------------------------------------------------------------
    case 'julia'
        % Julia recursion: discrete-time map (dt is ignored)
        % Constraint: ||x||_2 <= 1
        xi1 = -0.7; xi2 = 0.2;
        f = @(x) [x(1)^2 - x(2)^2 + xi1; 2*x(1)*x(2) + xi2];
        is_discrete = true;

    % ------------------------------------------------------------------
    case 'van_der_pol'
        % Van der Pol oscillator: mu=0.1
        % Constraint: ||x||_inf <= 3
        mu = 0.1;
        f = @(x) [x(2); mu*(1 - x(1)^2)*x(2) - x(1)];

    % ------------------------------------------------------------------
    case 'duffing'
        % Undamped Duffing oscillator (conservative Hamiltonian)
        % Two stable equilibria at (+/-1, 0), one unstable at (0,0)
        % Constraint: |x2| <= 0.7
        f = @(x) [x(2); x(1) - x(1)^3];

    % ------------------------------------------------------------------
    case 'lorenz'
        % Lorenz attractor (3D chaotic)
        % Constraint: asymmetric box (see doc.tex)
        sigma = 10; rho = 28; beta = 8/3;
        f = @(x) [sigma*(x(2) - x(1));
                  x(1)*(rho - x(3)) - x(2);
                  x(1)*x(2) - beta*x(3)];

    % ------------------------------------------------------------------
    case 'bicycle'
        % Kinematic bicycle lane keeping (2D)
        % States: x1=lateral pos, x2=heading angle
        % Constraint: |x1| <= 2  (4m lane)
        v = 20; l = 1.6; k1 = 0.3034; k2 = 0.6172;
        f = @(x) [v * sin(x(2) - k1*x(1) - k2*x(2));
                  (v/l) * sin(-k1*x(1) - k2*x(2))];

    % ------------------------------------------------------------------
    case 'power'
        % Two-machine power system (2D)
        % States: x1=relative phase, x2=relative frequency
        % Constraint: |x1| <= 3, |x2| <= 3
        f = @(x) [x(2); -0.5*x(2) - sin(x(1) + pi/3) + sin(pi/3)];

    % ------------------------------------------------------------------
    otherwise
        error('gen_data: unknown example "%s".\nSupported: linear, soft_landing, cornhole, moon_lander, pendulum, inverted_pendulum, double_pendulum, julia, van_der_pol, duffing, lorenz, bicycle, power', example);
end
end


%% =========================================================================
%  Local helper: moon lander dynamics
%% =========================================================================

function dxdt = moon_lander_f(x, kp_v, kd_v, kp_h, kd_h, r_a, r_b)
% States: x1=vert pos, x2=horiz pos, x3=vert vel, x4=horiz vel, x5=fuel
y1  = x(1); y2  = x(2);
dy1 = x(3); dy2 = x(4);

u1 = -kp_v * y1 - kd_v * dy1;   % vertical thrust (critically damped)

% Horizontal thrust switches to the nearer landing site
if abs(y2 - r_a) <= abs(y2 - r_b)
    u2 = -kp_h * (y2 - r_a) - kd_h * dy2;
else
    u2 = -kp_h * (y2 - r_b) - kd_h * dy2;
end

dxdt = [dy1; dy2; u1; u2; -(abs(u1) + abs(u2))];
end


%% =========================================================================
%  Local helper: double pendulum Euler-Lagrange dynamics
%% =========================================================================

function dxdt = double_pendulum_f(x, m, l, g)
% States: x = [theta1; theta2; dtheta1; dtheta2]
th1  = x(1); th2  = x(2);
dth1 = x(3); dth2 = x(4);

M = [2*m*l^2,                    2*m*l^2*cos(th1 - th2);
     2*m*l^2*cos(th1 - th2),     m*l^2               ];

C_mat = [0,                                  -m*l^2*sin(th1 - th2)*dth2;
          m*l^2*sin(th1 - th2)*dth1,          0                        ];

G_vec = [m*g*l*sin(th1); m*g*l*sin(th2)];

ddth = -M \ (C_mat * [dth1; dth2] + G_vec);

dxdt = [dth1; dth2; ddth(1); ddth(2)];
end
