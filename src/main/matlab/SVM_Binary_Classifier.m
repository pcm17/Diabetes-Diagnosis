%%% Experiments using linear SVM for diabetes classification
%%% *************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_data = load('pima_train.txt');
test_data = load('pima_test.txt');
num_attributes = size(train_data,2) - 1; 
num_classes = 2;                                    

X_train = train_data(:,1:num_attributes);
X_test = test_data(:,1:num_attributes);
X_train_norm = normalize(X_train);
X_test_norm = normalize(X_test);
y_train = train_data(:,num_attributes+1);
y_test = test_data(:,num_attributes+1);

% Calculate support vectors and bias term
[W_SVM, b_SVM] = svml(X_train_norm, y_train, 1000);

y_train_pred_SVM = X_train_norm*W_SVM + b_SVM;
y_test_pred_SVM = X_test_norm*W_SVM + b_SVM;

y_test_pred_SVM(y_test_pred_SVM < 0) = 0; y_test_pred_SVM(y_test_pred_SVM > 0) = 1;
y_train_pred_SVM(y_train_pred_SVM < 0) = 0; y_train_pred_SVM(y_train_pred_SVM > 0) = 1; 

traine_SVM = immse(y_train, y_train_pred_SVM);
teste_SVM = immse(y_test, y_test_pred_SVM);

[ confuse_train ] = confusion_matrix( y_train, y_train_pred_SVM, 2 );
[ confuse_test ] = confusion_matrix( y_test, y_test_pred_SVM, 2 );

sens = confuse_test(1,1) / (confuse_test(1,1) + confuse_test(2,1));
spec = confuse_test(2,2) / (confuse_test(2,2) + confuse_test(1,2));

fprintf('Iterations = 2000\tLearning Rate = 2 / sqrt(i)\n\n');
fprintf('Training Misclassification Error = %.4f\nTest Misclassification Error = %.4f\n\n',traine_SVM, teste_SVM);
fprintf('Training Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_train);
fprintf('Test Confusion matrix:\n[%d\t%d]\n[%d\t%d]\n\n',confuse_test);
fprintf('Sensitivity = %.4f\nSpecificity = %.4f\n\n', sens, spec);
