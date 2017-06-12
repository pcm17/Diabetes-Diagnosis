%%% Experiments comparing diabetes classification using a NN w/ sigmoidal
%%% function and 1,2,5, or 10 layers
%%% for diabetes classification
%%% *************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% load the train and test data (both are normalized)
load resources/pima_train.txt;
load resources/pima_test.txt;

tr_data = pima_train;
test_data = pima_test;
data_col= size(tr_data,2);
n_features = data_col - 1;

%%% create x
x = tr_data(:,1:n_features);
% create y vector
y=tr_data(:,data_col);
 
% builds x for the the test set
x_test = test_data(:,1:n_features);
% builds y vector for the test set
y_test=test_data(:,data_col);
% defines the number of hidden layers
layers = [1 2 5 10 20];
nLoops=length(layers);
nReps = 5;
%% Loops nLoops*nReps times to experiment with different numbers of hidden layers

for i = 1:nLoops
    for j = 1:nReps
    %%% training of the neural net

    %%% builds a layered neural network with a sigmoidal output function
    %%% to be traind with the gradient method
    net=patternnet(layers(i));
    %view(net
    %% Set the parameters of the NN model
    net.trainParam.epochs = 2000;
    net.trainParam.show = 20;
    net.trainParam.max_fail=5;
    %%% define the training function
    net.trainFcn='trainlm';
    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 90/100;
    net.divideParam.valRatio = 5/100;
    net.divideParam.testRatio = 5/100;

    [net, tr] = train(net,x',y');    


    %% runs learned network on inputs in x (training set)
    res=net(x');
    %%% mean classification error on the training data
    class_error_train(i,j)=sum(y~=round(res)')/size(res,2);
    %%% 'Mean squared error (mse) on the training data'
    mse_error_train(i,j) = perform(net,y',res);


    %% runs learned network on inputs in x (testing set)
    res_test = net(x_test');
    %%% mean classification error on the testing data
    class_error_test(i,j)=sum(y_test~=round(res_test)')/size(res_test,2);
    %%% 'Mean squared error (mse) on the testing data'
    mse_error_test(i,j) = perform(net,y_test',res_test);
    end
end

avg_acc_tr = 100*(1-mean(class_error_train'));
avg_acc_test = 100*(1-mean(class_error_test'));

figure,plot(layers, avg_acc_test, layers, avg_acc_tr);
legend('Test','Train');
title(['Accuracy for ', num2str(layers), ' hidden layers']);





