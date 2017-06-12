%%% Experiments with Naive Bayes classifier for diabetes classification
%%% ****************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load diabetes data
train_data = load('pima_train.txt');
test_data = load('pima_test.txt');
num_features = size(train_data,2) - 1;
X_train = train_data(:,1:num_features);
X_test = test_data(:,1:num_features);
y_train = train_data(:,num_features+1);
y_test = test_data(:,num_features+1);
class_1_train_data = train_data(train_data(:,num_features+1) == 1,1:num_features);
class_0_train_data = train_data(train_data(:,num_features+1) == 0,1:num_features);

% Compute Parameter Estimates
[exp_0_1_mu, exp_1_1_mu, norm_0_2_mu, norm_0_2_sigma, norm_1_2_mu, ... 
    norm_1_2_sigma, norm_0_3_mu, norm_0_3_sigma, norm_1_3_mu, norm_1_3_sigma, ...
    norm_0_4_mu, norm_0_4_sigma, norm_1_4_mu,norm_1_4_sigma, exp_0_5_mu, ...
    exp_1_5_mu, norm_0_6_mu, norm_0_6_sigma, norm_1_6_mu,norm_1_6_sigma, ...
    exp_0_7_mu, exp_1_7_mu, exp_0_8_mu, exp_1_8_mu, prior_y1, prior_y0] ...
    = Compute_NB_Parameter_Estimates (class_0_train_data, class_1_train_data);
% Predict class
y_train_pred_NB = predict_NB(X_train, exp_0_1_mu, exp_1_1_mu, norm_0_2_mu, norm_0_2_sigma, norm_1_2_mu, norm_1_2_sigma, norm_0_3_mu, norm_0_3_sigma,norm_1_3_mu, norm_1_3_sigma, norm_0_4_mu, norm_0_4_sigma, norm_1_4_mu, norm_1_4_sigma, exp_0_5_mu, exp_1_5_mu, norm_0_6_mu, norm_0_6_sigma, norm_1_6_mu, norm_1_6_sigma,exp_0_7_mu, exp_1_7_mu, exp_0_8_mu, exp_1_8_mu, prior_y0, prior_y1);
y_test_pred_NB = predict_NB(X_test, exp_0_1_mu, exp_1_1_mu, norm_0_2_mu, norm_0_2_sigma, norm_1_2_mu, norm_1_2_sigma, norm_0_3_mu, norm_0_3_sigma,norm_1_3_mu, norm_1_3_sigma, norm_0_4_mu, norm_0_4_sigma, norm_1_4_mu, norm_1_4_sigma, exp_0_5_mu, exp_1_5_mu, norm_0_6_mu, norm_0_6_sigma, norm_1_6_mu, norm_1_6_sigma,exp_0_7_mu, exp_1_7_mu, exp_0_8_mu, exp_1_8_mu, prior_y0, prior_y1);

% Calculate Mean Misclassification Error
traine_NB = immse(y_train_pred_NB,y_train);
teste_NB = immse(y_test_pred_NB,y_test);

% Compute confusion matrix
confuse_train = confusion_matrix(y_train, y_train_pred_NB, 2);
confuse_test = confusion_matrix(y_test, y_test_pred_NB, 2);

% Compute sensitivity and specificity
sens = confuse_test(1,1) / (confuse_test(1,1) + confuse_test(2,1));
spec = confuse_test(2,2) / (confuse_test(2,2) + confuse_test(1,2));

% Display results
fprintf('Training Misclassification Error = %.4f\nTest Misclassification Error = %.4f\n\n',traine_NB, teste_NB);
fprintf( 'Training Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_train);
fprintf( 'Test Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_test);
fprintf( 'Sensitivity = %.4f\nSpecificity = %.4f\n\n', sens, spec);

function [y_pred] = predict_NB(X, exp_0_1_mu, exp_1_1_mu, norm_0_2_mu, norm_0_2_sigma, norm_1_2_mu, norm_1_2_sigma, norm_0_3_mu, norm_0_3_sigma,norm_1_3_mu, norm_1_3_sigma, norm_0_4_mu, norm_0_4_sigma, norm_1_4_mu, norm_1_4_sigma, exp_0_5_mu, exp_1_5_mu, norm_0_6_mu, norm_0_6_sigma, norm_1_6_mu, norm_1_6_sigma,exp_0_7_mu, exp_1_7_mu, exp_0_8_mu, exp_1_8_mu, prior_y0, prior_y1)
% Makes class prediction based on model parameters

pd10 = exppdf(X(:,1),exp_0_1_mu); 
pd11 = exppdf(X(:,1),exp_1_1_mu); 

pd20 = normpdf(X(:,2),norm_0_2_mu, norm_0_2_sigma); 
pd21 = normpdf(X(:,2),norm_1_2_mu, norm_1_2_sigma);

pd30 = normpdf(X(:,3),norm_0_3_mu, norm_0_3_sigma); 
pd31 = normpdf(X(:,3),norm_1_3_mu, norm_1_3_sigma);

pd40 = normpdf(X(:,4),norm_0_4_mu, norm_0_4_sigma); 
pd41 = normpdf(X(:,4),norm_1_4_mu, norm_1_4_sigma);

pd50 = exppdf(X(:,5),exp_0_5_mu); 
pd51 = exppdf(X(:,5),exp_1_5_mu); 

pd60 = normpdf(X(:,6),norm_0_6_mu, norm_0_6_sigma); 
pd61 = normpdf(X(:,6),norm_1_6_mu, norm_1_6_sigma);

pd70 = exppdf(X(:,7),exp_0_7_mu); 
pd71 = exppdf(X(:,7),exp_1_7_mu); 

pd80 = exppdf(X(:,8),exp_0_8_mu); 
pd81 = exppdf(X(:,8),exp_1_8_mu); 

y_pred = zeros(size(X,1),1);     % Initialize prediction array

for i = 1:size(X,1)
    prob0 = log(pd10(i))+log(pd20(i))+log(pd30(i))+log(pd40(i))+log(pd50(i))+log(pd60(i))+log(pd70(i))+log(pd80(i)) + log(prior_y0);
    prob1 = log(pd11(i))+log(pd21(i))+log(pd31(i))+log(pd41(i))+log(pd51(i))+log(pd61(i))+log(pd71(i))+log(pd81(i)) + log(prior_y1);
    %norm_term = prob0 + prob1;
    
    if prob1 > prob0
        y_pred(i,1) = 1;
    end
end

end
