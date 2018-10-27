%%% Optional exercise %%%

% load data
load('ex5data1');

m = size(X, 1);

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

lambda = 0.01; 


i = 12; % num examples
n = 50; % num repetition

error_train = zeros(i, 1);
error_val = zeros(i, 1);
for r = 1:n
    % select random examples
    index = randsample(1:m, i);
    X_poly_example = X_poly(index, :);
    X_poly_val_example = X_poly_val(index, :);
    
    y_example = y(index);
    yval_example = yval(index);
    
    theta = trainLinearReg(X_poly_example, y_example, lambda);
    
    [t, v] = learningCurve(X_poly_example, y_example, X_poly_val_example, yval_example, lambda);
    error_train = error_train + t;
    error_val = error_val + v;
    
end

error_train = error_train / n;
error_val = error_val / n;

plot(1:i, error_train, 1:i, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
error_val = error_val / n;
