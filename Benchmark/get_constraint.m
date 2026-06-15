function g = get_constraint(example)
% GET_CONSTRAINT  Return the constraint function g for the given benchmark.
%   g(x) <= 1  defines the constraint set X.
%   x is [n x N], g returns [1 x N].

switch lower(example)
    case {'linear', 'pendulum'}
        g = @(x) abs(x(1,:));
    case 'van_der_pol'
        g = @(x) max(abs(x)) / 3;
    case 'duffing'
        g = @(x) abs(x(2,:)) / 0.7;
    case 'julia'
        g = @(x) sum(x.^2, 1);
    case 'bicycle'
        g = @(x) abs(x(1,:)) / 2;
    case 'power'
        g = @(x) max(abs(x)) / 3;
    case 'lorenz'
        g = @(x) max(abs(x - [0;0;25]) ./ [20;30;25]);
    case 'double_pendulum'
        g = @(x) max(abs(sin(x(1,:))), abs(sin(x(1,:)) + sin(x(2,:))));
    case 'inverted_pendulum'
        g = @(x) max(abs(x) ./ [2.5; 5]);
    case {'soft_landing', 'soft_landing_relaxed'}
        % X = {x : x1 <= 0.01, x2 >= 0}
        g = @(x) max(x(1,:) + 0.99, -x(2,:) + 1);
    case 'cornhole'
        board_tol = 0.2;
        in_X = @(x) ~(abs(5*x(2,:) - x(1,:)) < board_tol & abs(5*x(1,:) + x(2,:)) > 7);
        g    = @(x) 1.1 * ~in_X(x);
    case 'moon_lander'
        r_a = -2; r_b = 2;
        in_X = @(x) (abs(x(2,:)-r_a) <= abs(x(1,:)) | abs(x(2,:)-r_b) <= abs(x(1,:))) ...
                    & (x(5,:) >= 0);
        g    = @(x) 1.1 * ~in_X(x);
    otherwise
        error('get_constraint: no g defined for "%s"', example);
end
end
