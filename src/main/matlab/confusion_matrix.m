function [ C ] = confusion_matrix( y_true, y_pred, nClasses )
%Creates a confusion matrix from the predicted y values and ground truth y
% Arguments:    1. ground truth y values
%               2. predicted y values
%               3. number of classes
% returns: confuse_matrix

nSamples = length(y_true);
C = zeros(nClasses,nClasses);  % Initialize matrix of zeros 

for i = 1:nSamples
    true = y_true(i) + 1;
    predict = y_pred(i) + 1;        % Need +1 b/c matlab indices start at 1, not 0
    C(true,predict) = C(true,predict) + 1;
end

end