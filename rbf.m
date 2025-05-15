% X: n x N matrix of N data points in n-dimensional space
% C: RBF center(s), either an n x 1 vector (single center) or an n x K matrix (K centers)
%     - If C is n x K, the output will be stacked over the K centers, resulting in a K x N matrix
% eps: (optional) kernel width (standard deviation) for Gaussian RBFs
% k: (optional) polyharmonic coefficient for polyharmonic RBFs

% Note: All RBFs are symmetric. Evaluating multiple centers against a single point 
% is equivalent to evaluating a single center against multiple points.

% WARNING: This function may not work correctly in 1D.
% Quick workaround: append a row of zeros to lift the input to 2D.

%%
function out = rbf(X, Center, RBFtype, Center_spacing, width, polyharmonic_power)

RBFtype = lower(RBFtype);  % Make RBF type case-insensitive

% Default width if not provided
if(~exist('width','var') || isempty(width))
    width = 1;
end

% Default polyharmonic power if not provided
if(~exist('polyharmonic_power','var') || isempty(polyharmonic_power))
    polyharmonic_power = 1;
end

% Default center spacing: difference between first two centers
if(~exist('Center_spacing','var') || isempty(Center_spacing))
    Center_spacing = Center(:,2) - Center(:,1);
end

% Preallocate output: one row per center, one column per input point
out = zeros(size(Center,2), size(X,2));

% Loop over each center
for i = 1:size(Center,2)
    Center_i = Center(:,i);  % Current center

    switch RBFtype
        case 'relu'
            % Piecewise linear: active only if X is inside [Center_i, Center_i + spacing]
            out_i = sum((((X - Center_i) ./ Center_spacing) .* ...
                     (sum((X - Center_i >= 0) & (X - Center_i <= Center_spacing)) >= size(Center_spacing,1))));

        case 'pyramid'
            % Simple L1-shaped peak
            r1 = sum(abs((X - Center_i) ./ Center_spacing));
            out_i = (1 - r1) .* (r1 <= 1);

        case 'triangle'
            % Triangle-shaped bump with constraints for different axis-aligned orientations
            r1 = (X - Center_i) ./ Center_spacing;
            out_i = (1 - max(abs(r1))) .* (max(abs(r1)) <= 1) .* ...
                    (abs(sum(sign(r1))) == size(r1,1)) + ...
                    (1 - sum(abs(r1))) .* (sum(abs(r1)) <= 1) .* ...
                    (abs(sum(sign(r1))) < size(r1,1));

        case 'thinplate'
            % Thin-plate spline: r^2 * log(r)
            r_squared = sum((X - Center_i).^2);
            out_i = r_squared .* log(sqrt(r_squared));
            out_i(isnan(out_i)) = 0;  % Handle log(0)

        case 'gauss'
            % Gaussian RBF: exp(-r^2 / width^2)
            r_squared = sum((X - Center_i).^2);
            out_i = exp(-r_squared / width^2);

        case 'invquad'
            % Inverse quadratic: 1 / (1 + r^2 / width^2)
            r_squared = sum((X - Center_i).^2);
            out_i = 1 ./ (1 + r_squared / width^2);

        case 'invmultquad'
            % Inverse multiquadratic: 1 / sqrt(1 + r^2 / width^2)
            r_squared = sum((X - Center_i).^2);
            out_i = 1 ./ sqrt(1 + r_squared / width^2);

        case 'polyharmonic'
            % Polyharmonic spline: r^k * log(r), with user-defined k
            r_squared = sum((X - Center_i).^2);
            out_i = r_squared.^(polyharmonic_power / 2) .* log(sqrt(r_squared));
            out_i(isnan(out_i)) = 0;  % Handle log(0)

        case 'bump'
            % Compactly supported smooth bump: exp(-1 / (1 - r^2)) inside unit ball
            r_squared = sum(((X - Center_i) ./ Center_spacing).^2);
            out_i = exp(-1 ./ (1 - 0.1 * r_squared)) .* (r_squared <= 10);
            out_i(isnan(out_i)) = 0;  % Handle division by zero

        otherwise
            error('RBF type not recognized')
    end

    % Assign result for current center
    out(i,:) = out_i;
end
