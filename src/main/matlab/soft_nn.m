function [y_pred] = soft_nn(x, y, xn, h)
% Version of knn where each point's contribution is weighted 
% proportionally to its distance from the target point we want to classify
% Arguments:    1. Training Predictors
%               2. Training Classes
%               3. Single Test data point Predictors
%               4. Gaussian Smoothness Parameter
% Returns:  Class Prediction

N = size(x,1);
vote1 = 0;
vote0 = 0;
for n = 1:N
    xi = x(n,:);
    % Calculate Euclidean distance
    u = sqrt(sum((xi - xn).^2));
    % Calculate contribution weight
    K = (1/sqrt(2*pi))*exp(- u.^2/(2*h^2));
    if (y(n) == 0)
        vote0 = vote0 + (1/(N*h))*K;
    else
        vote1 = vote1 + (1/(N*h))*K;
    end
end

if (vote1 >= vote0)
    y_pred = 1;
else
    y_pred = 0;
end
    
end