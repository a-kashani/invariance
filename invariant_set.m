% Author: Ali Kashani

% Computes an approximation of an invariant set using Radial Basis Functions (RBFs)
% over a uniformly sampled grid.
%
% Syntax:
%   O = invariant_set(center_spacing, data_density, RBFtype)
%
% Inputs:
%   center_spacing : Scalar specifying the spacing between RBF centers.
%   data_density   : Scalar indicating the number of sample points used per dimension.
%                    The total number of samples is proportional to data_density^2.
%   RBFtype        : String specifying the type of radial basis function to use.
%                    Examples: 'gauss', 'triangle', 'polyharmonic', 'invquad',  etc.
%                    see rbf(.) for more details.
%
% Output:
%   O              : Vertices defining the polygonal approximation
%                    of the invariant set.
%


function O = invariant_set(center_spacing,data_density, RBFtype)

% parameters
Lyapunov_discount=0.1;    %0<=gamma<=1       % from the invariance condition   
num_samples= data_density^2;     
data_spacing=0.01*2*sqrt(2)/data_density;
L_phi=0.5;  L_f=1.4;   L_g=1;    % Lipschitz constants

%% create system for simulation
% system parameters
zeta = 0.1;
wn   = 1;
dt   = 0.3;

% continuous-time dynamics
Ac = [0,1;-wn^2,-2*zeta*wn];
sysc = ss(Ac,zeros(2,1),zeros(1,2),0);

% discrete-time dynamics
sys = c2d(sysc,dt);
Ad = sys.A;

f = @(x) Ad*x;               %Dynamics x^+=f(x)
g = @(x) max(abs(x));        %Unit box constraint set X={x: g(x)<=1}  

%% sampling
% Uniform
[x1,x2] = meshgrid(linspace(-1,1,data_density));
x  = [x1(:)';x2(:)'];
xp = f(x);

%% Basis function approach
% centers of basis functions 
[c1,c2] = meshgrid(-1:center_spacing:1);
c = [c1(:)';c2(:)'];          % vectorizing the meshgrid centers

phi= rbf(x,c,RBFtype, [center_spacing;center_spacing]);                       % Spline basis
phip= rbf(xp,c,RBFtype, [center_spacing;center_spacing]);

phi=[g(x); phi;ones(1,num_samples)];                             % adding g and a constant to phi for better approximation. could be removed
phip=[g(xp); phip; ones(1,num_samples)];
M = size(phi,1);

% lp parameters
A = zeros(num_samples,2*M);
b = zeros(num_samples,1);

for i=1:num_samples
    if g(xp(:,i))<=1
        A(i,1:M)= -(1-Lyapunov_discount)*phi(:,i)' + phip(:,i)' + L_phi*data_spacing*(1-Lyapunov_discount+L_f);
        A(i,M+1:end)= (1-Lyapunov_discount)*phi(:,i)' - phip(:,i)' + L_phi*data_spacing*(1-Lyapunov_discount+L_f);
        b(i)= Lyapunov_discount;
    else
        A(i,1:M)= -(1-Lyapunov_discount)*phi(:,i)'+L_phi*data_spacing*(1-Lyapunov_discount);
        A(i,M+1:end)= (1-Lyapunov_discount)*phi(:,i)'+ L_phi*data_spacing*(1-Lyapunov_discount);
        b(i)= - g(xp(:,i)) +Lyapunov_discount-data_spacing*L_g*L_f;
    end
end

f_0=sum(phi,2)';
f = [f_0+abs(f_0), -f_0+ abs(f_0)];
qpn = linprog(f,A,b,[],[],zeros(2*M,1));

q=qpn(1:M)-qpn(M+1:end);

% plot barrier function
    Vf= @(x) [g(x); rbf(x,c,RBFtype, [center_spacing;center_spacing]); 1]'*q.*(g(x)<=1)+ g(x).*(g(x)>1);
    [xx, yy] = meshgrid(linspace(-1.01,1.01,201)); 
    Z= arrayfun(@(x, y) -Vf([x; y]), xx, yy);
    level = -1;  % Define the sublevel you want to plot
    O= contour(xx, yy, Z, [level level]);
    close
