%%% Experiments with diabetes classification using a KNN Classifier
%%% *************************************************************
%%% Peter McCloskey
%%% CS1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% load the train and test data (both are normalized)
load resources/pima_train.txt;
load resources/pima_test.txt;
tr_data = pima_train;
test_data = pima_test;

data_col= size(tr_data,2);
n_features = data_col - 1;

%%% create x
x_tr = tr_data(:,1:n_features);
x_tr = normalize(x_tr);
%% create y vector
y_tr = tr_data(:,data_col);
 
%% build x for the the test set
x_test = test_data(:,1:n_features);
x_test = normalize(x_test);
%myxtest = (x_test - min(x_test))./(max(x_test)-min(x_test)); 
%% build y vector for the test set
y_test=test_data(:,data_col);

%%%% classify examples in the test set using the KNN classifier using the
%%%% Euclidean metric
N = 20;
neighbors = linspace(1,N,N);
accuracy_test = zeros(1,N);
accuracy_tr = zeros(1,N);
for n =1:N
    mdl=fitcknn(x_tr,y_tr,'NumNeighbors', neighbors(n),'NSMethod','euclidean');
    
    y_pred = predict(mdl, x_test);
    error= sum(abs(y_pred ~= y_test))/length(y_pred);
    accuracy_test(n)=1-error;
end

figure,plot(neighbors, accuracy_test);
title('Data Normalized');
xlabel('Number of Neighbors');
ylabel('Classification Accuracy');
top_acc_test = max(accuracy_test);
top_num_neighbors_test = find(accuracy_test == top_acc_test);
fprintf('Top Test Accuracy: %.2f\tw/ %d neighbors\n',top_acc_test*100, top_num_neighbors_test(1));
