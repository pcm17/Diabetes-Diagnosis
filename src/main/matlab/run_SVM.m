%%% Experiments using linear SVM for diabetes classification
%%% *************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('resources/pima_train.txt');
load('resources/pima_test.txt');
num_attributes = size(pima_train,2) - 1; 
num_classes = 2;                                    

X_train = pima_train(:,1:num_attributes);
X_test = pima_test(:,1:num_attributes);
X_train_norm = normalize(X_train);
X_test_norm = normalize(X_test);

y_train = pima_train(:,num_attributes+1);
y_test = pima_test(:,num_attributes+1);

y_test_pred = zeros(length(y_test),1);
y_train_pred = zeros(length(y_train),1);

% Calculate support vectors and bias term
[W_SVM, b_SVM] = svml(X_train_norm, y_train, 10);

y_train_pred_SVM = X_train_norm*W_SVM + b_SVM;
y_test_pred_SVM = X_test_norm*W_SVM + b_SVM;

y_test_pred(y_test_pred_SVM < 0) = 0; y_test_pred(y_test_pred_SVM > 0) = 1;
y_train_pred(y_train_pred_SVM < 0) = 0; y_train_pred(y_train_pred_SVM > 0) = 1; 

traine_SVM = 100*immse(y_train, y_train_pred);
teste_SVM = 100*immse(y_test, y_test_pred);

[ confuse_train ] = confusion_matrix( y_train, y_train_pred, 2 );
[ confuse_test ] = confusion_matrix( y_test, y_test_pred, 2 );

sens = confuse_test(1,1) / (confuse_test(1,1) + confuse_test(2,1));
spec = confuse_test(2,2) / (confuse_test(2,2) + confuse_test(1,2));

fprintf('Training Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_train);
fprintf('Test Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_test);
fprintf('Training Accuracy = %.2f\nTest Accuracy = %.2f\n\n',100-traine_SVM, 100-teste_SVM);
fprintf('Sensitivity = %.4f\nSpecificity = %.4f\n\n', sens, spec);
