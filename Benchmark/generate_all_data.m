%% generate_all_data.m
% Generates grid-sampled state-transition data (z, z_p) for all benchmarks
% defined in doc.tex by calling gen_data.m.
%
% Bounds are set to cover the constraint set X with a small margin.
% Sampling time (dt) is chosen per system based on its time-scale.
% Spacing is coarser for higher-dimensional systems to limit grid size.
%
% Each example is saved to <example>.mat in the current directory.

clear; clc

%% =========================================================================
%  2D Benchmarks
%% =========================================================================

% -- Linear System --
% X = {x : |x1| <= 1}  (wn=5, zeta=0.1)
gen_data('linear', ...
    [-1.1,  1.1; ...   % x1: position (constraint |x1| <= 1)
     -8,    8  ], ...  % x2: velocity
    51, 0.02)

% -- Soft-Landing --
% X = {x : -0.1*x1 <= x2 <= -0.5*x1}  (x1<=0, x2>=0)
gen_data('soft_landing', ...
    [-5,   0.1; ...    % x1: position (x1 <= 0 in constraint set)
     -0.1, 3  ], ...   % x2: velocity (x2 >= 0 in constraint set)
    51, 0.01)

% -- Pendulum --
% X = {x : |x1| <= 1}  (wn=5, zeta=0.1, nonlinear sin term)
gen_data('pendulum', ...
    [-1.1, 1.1; ...    % x1: angle (constraint |x1| <= 1 rad)
     -8,   8  ], ...   % x2: angular velocity
    51, 0.02)

% -- Inverted Pendulum --
% X = R^2  (CAPI set = region of attraction; k1=4.4142, k2=2.3163)
gen_data('inverted_pendulum', ...
    [-2.5, 2.5; ...    % x1: angle (covers region of attraction)
     -5,   5  ], ...   % x2: angular velocity
    51, 0.05)

% -- Van der Pol --
% X = {x : ||x||_inf <= 3}  (mu=0.1)
gen_data('van_der_pol', ...
    [-3, 3; ...        % x1 (constraint -3 <= x1 <= 3)
     -3, 3], ...       % x2 (constraint -3 <= x2 <= 3)
    51, 0.1)

% -- Duffing Oscillator --
% X = {x : |x2| <= 0.7}  (two stable equilibria at (+-1, 0))
gen_data('duffing', ...
    [-2,  2  ; ...     % x1: displacement (covers both equilibria +/-1)
     -0.8, 0.8], ...   % x2: velocity (constraint |x2| <= 0.7)
    51, 0.1)

% -- Julia Recursion (discrete-time) --
% X = {x : ||x||_2 <= 1}  (xi1=-0.7, xi2=0.2)
gen_data('julia', ...
    [-1, 1; ...        % x1 (unit disk)
     -1, 1], ...       % x2
    81, [])           % dt unused for discrete-time map

% -- Kinematic Bicycle Lane Keeping --
% X = {x : |x1| <= 2}  (v=20, l=1.6)
gen_data('bicycle', ...
    [-2, 2; ...    % x1: lateral position (constraint |x1| <= 2 m)
     -0.5, 0.5], ...   % x2: heading angle (radians)
    51, 0.05)

% -- Two-Machine Power System --
% X = {x : |x1| <= 3, |x2| <= 3}
gen_data('power', ...
    [-3, 3; ...        % x1: relative phase (constraint |x1| <= 3)
     -3, 3], ...       % x2: relative frequency (constraint |x2| <= 3)
    61, 0.05)

%% =========================================================================
%  3D Benchmarks
%% =========================================================================

% -- Lorenz Attractor --
% X = {x : -20<=x1<=20, -30<=x2<=30, 0<=x3<=50}  (sigma=10, rho=28, beta=8/3)
% Small dt needed due to fast chaotic dynamics
gen_data('lorenz', ...
    [-20, 20; ...      % x1 (constraint |x1| <= 20)
     -30, 30; ...      % x2 (constraint |x2| <= 30)
       0, 50], ...     % x3 (constraint 0 <= x3 <= 50)
    21, 0.01)

%% =========================================================================
%  4D Benchmarks
%% =========================================================================

% -- Cornhole (ballistic double integrator) --
% X = non-convex (bag passes through cornhole hole)
% Coarser grid due to 4D — covers plausible throw trajectories
gen_data('cornhole', ...
    [-4,  4; ...       % x1: horizontal position
     -4,  4; ...       % x2: vertical position
     -12, 12; ...      % x3: horizontal velocity
     -12, 12], ...     % x4: vertical velocity (positive = upward)
    8, 0.05)

% -- Double Pendulum --
% X = {x: |l*sin(x1)|<=1, |l*sin(x1)+l*sin(x2)|<=1}  (l=1, m=1)
% Coarser grid due to 4D and slow ode45 per point
gen_data('double_pendulum', ...
    [-pi, pi; ...      % x1: joint 1 angle
     -pi, pi; ...      % x2: joint 2 angle
     -5,  5; ...       % x3: joint 1 angular velocity
     -5,  5], ...      % x4: joint 2 angular velocity
    8, 0.05)

%% =========================================================================
%  5D Benchmarks
%% =========================================================================

% -- Multi-Site Moon Lander --
% X = {x: |x2-r_a|<=|x1| OR |x2-r_b|<=|x1|, x5>=0}
% r_a=-2, r_b=2 (horizontal landing sites)
% Coarser grid due to 5D
gen_data('moon_lander', ...
    [ 0,  5; ...       % x1: vertical position (>= 0, above ground)
     -5,  5; ...       % x2: horizontal position
     -4,  4; ...       % x3: vertical velocity
     -4,  4; ...       % x4: horizontal velocity
      0, 10], ...      % x5: fuel remaining (>= 0)
    8, 0.05)

disp('Done. All data files generated.')
