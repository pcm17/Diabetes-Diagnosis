%%% Experiments with a kernel density estimator for diabetes classification
%%% *************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the train and test data (both are normalized)
load pima_train.txt;
load pima_test.txt;
load pima.txt

tr_data = pima_train;
test_data = pima_test;
data = pima;

data_col= size(tr_data,2);
n_features = data_col - 1;

% create x
x = tr_data(:,1:n_features);
%x = normalize(x);
%create y vector
y=tr_data(:,data_col);

%% builds x for the the test set
x_test = test_data(:,1:n_features);
% noramlize x
%x_test = normalize(x_test);
%%% builds y vector for the test set
y_test = test_data(:,data_col);

%% 
numReps = 10;
numTestPoints = size(x_test,1);
accuracy = zeros(numReps,1);
hvals = linspace(0.01,1,numReps);
for n = 1:numReps
    h = hvals(n);
    for i = 1:numTestPoints
        y_pred(i,1) = soft_nn(x, y, x_test(i,:), h);
    end
    
    % Class Error
    error= sum(round(y_pred)~=y_test)/size(y_test,1);
    accuracy(n) = 1-error;

end

figure, plot(hvals,accuracy,'k-');
xlabel('Kernel Smoothing value(h)');
ylabel('Classification Accuracy');
title('Kernel Smoothing value(h) vs Classification Accuracy');

