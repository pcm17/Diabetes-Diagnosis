%%% Experiments with a decision tree for diabetes classification
%%% *************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Machine Learning, University of Pittsburgh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% load the train and test data (both are normalized)
load pima_train.txt;
load pima_test.txt;
tr_data = pima_train;
test_data = pima_test;

data_col= size(tr_data,2);
n_features = data_col - 1;

%%% create x
x = tr_data(:,1:n_features);
%% create y vector
y=tr_data(:,data_col);
 
%% builds x for the the test set
x_test = test_data(:,1:n_features);
%% builds y vector for the test set
y_test=test_data(:,data_col);

%%%% new tree with restrictions on the tree size, parent size, and leaf sizes 
new_tree=fitctree(x,y, 'NumVariablesToSample',10,'MinParentSize',20,'MinLeafSize',16,'splitcriterion','gdi');
y_pred=predict(new_tree,x_test);
error = sum(y_pred~=y_test)/size(y_test,1);
fprintf('\nDecision Tree Error = %.2f\n',error*100);


%plot(leafs,error);
%xlabel('NumVariablesToSample');
%ylabel('Error');
    %%% show the tree logic
%view(new_tree);
%%% show the graphics of the tree
%view(new_tree,'Mode','graph');
%%% evaluate the tree on the test data



