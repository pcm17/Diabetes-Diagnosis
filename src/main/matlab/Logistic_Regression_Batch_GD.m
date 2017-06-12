%%% Experiments with logistic regression w/ batch gradient descent and 
%%% sigmoidal function for diabetes classification
%%% *************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_data = load('pima_train.txt');
test_data = load('pima_test.txt');
pgraph = init_progress_graph;       % Initialize graph
num_attributes = size(train_data,2) - 1; 
K = 10;             % number of steps
num_classes = 2;    % number of classes

X_train = train_data(:,1:num_attributes);
X_test = test_data(:,1:num_attributes);
y_train = train_data(:,num_attributes+1);
y_test = test_data(:,num_attributes+1);

X_train_norm = normalize(X_train);  % Normalize training predictors
X_test_norm = normalize(X_test);    % Normalize test predictors
    
%% Write noramlized data to file
trainFileID = fopen('pima_train_norm.txt', 'w');
fprintf(trainFileID, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n\n', [X_train_norm y_train]');

trainFileID = fopen('pima_test_norm.txt', 'w');
fprintf(trainFileID, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n\n', [X_test_norm y_test]');
%% Initialize variables
col_ones = ones(size(X_train_norm, 1), 1);
X_train_norm = horzcat(col_ones, X_train_norm);     % Add a column of ones on the left to X_train_norm
X_train = horzcat(col_ones, X_train);               % Add a column of ones on the left to X_train

col_ones = ones(size(X_test_norm, 1), 1);
X_test_norm = horzcat(col_ones, X_test_norm);        % Add a column of ones on the left to X_test_norm
X_test = horzcat(col_ones, X_test);                  % Add a column of ones on the left to X_test

W_LR = ones(size(X_train_norm, 2), 1);               % Initialize W to 1 to start with 
%% Run Logistic regression for K iterations
for k = 1:K    
    W_LR = Log_regression(X_train_norm, y_train, W_LR, k);
    
    if mod(k,25) == 0   % Plot progress
        % Make predictions for training data
        y_train_pred_LR = sigmoid(X_train_norm*W_LR);
        % Make predictions for test data
        y_test_pred_LR = sigmoid(X_test_norm*W_LR);
        
        % Calculate errors and add them to the graph
        traine_LR=sum(abs(y_train-round(y_train_pred_LR)))/size(y_train,1);
        teste_LR=sum(abs(y_test-round(y_test_pred_LR)))/size(y_test,1);
        pgraph = add_to_progress_graph(pgraph, k, traine_LR, teste_LR);
    end
end

% Make predictions for training data
y_train_pred_LR = sigmoid(X_train_norm*W_LR);
% Make predictions for test data
y_test_pred_LR = sigmoid(X_test_norm*W_LR);

% Quantize predictions to be either 1 or 0
y_test_pred_LR = round(y_test_pred_LR);
y_train_pred_LR =round(y_train_pred_LR);

traine_LR=sum(y_train~=y_train_pred_LR)/size(y_train,1);
teste_LR=sum(y_test~=y_test_pred_LR)/size(y_test,1);

%%
% Create confusion matrices
confuse_train = confusion_matrix(y_train, y_train_pred_LR, num_classes);
confuse_test = confusion_matrix(y_test, y_test_pred_LR, num_classes);

% Calculated sensitivity and specificity
sens = confuse_test(1,1) / (confuse_test(1,1) + confuse_test(2,1));
spec = confuse_test(2,2) / (confuse_test(2,2) + confuse_test(1,2));

% Display results
fprintf(['\nIterations = ' num2str(K) '\nLearning Rate = 1 / (i^1.75)\n\n']);
fprintf('Training Misclassification Error = %.4f\nTest Misclassification Error = %.4f\n\n',traine_LR, teste_LR);
fprintf('Training Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_train);
fprintf('Test Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_test);
fprintf('Sensitivity = %.4f\nSpecificity = %.4f\n\n', sens, spec);


function W = Log_regression(X, Y, W, i)
% Logistic regression using batch gradient descend
% inputs: 1. the input data
%         2. the output data
%         3. weights to update
%         4. iteration number
% returns: updated weights
%%% annealed learning: 1/i^1.75

    sum_err = 0;                    %%% initialize batch error function gradient
    
    for row = 1:1:size(X, 1)
        x = X(row,:)';
        y = Y(row,:);
        f = 1/(1 + exp(-(W'*x)));
        err = (y - f) * x;          % error (on-line gradient)
        sum_err = sum_err + err;    % update batch error function gradient
    end
    alpha = 1/(i^1.75);
    W = W + (alpha * sum_err);
end


function [ z ] = sigmoid( z )
% Applies the sigmoid function to input
% Arguments:    1. Raw values
%
% Returns: Sigmoided values
for i = 1:size(z,1)
    z(i) = 1 / (1 + exp(-z(i)));
end
end

