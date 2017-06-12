function [exp_0_1_mu, exp_1_1_mu, norm_0_2_mu, norm_0_2_sigma, norm_1_2_mu, ... 
    norm_1_2_sigma, norm_0_3_mu, norm_0_3_sigma, norm_1_3_mu, norm_1_3_sigma, ...
    norm_0_4_mu, norm_0_4_sigma, norm_1_4_mu,norm_1_4_sigma, exp_0_5_mu, ...
    exp_1_5_mu, norm_0_6_mu, norm_0_6_sigma, norm_1_6_mu,norm_1_6_sigma, ...
    exp_0_7_mu, exp_1_7_mu, exp_0_8_mu, exp_1_8_mu, prior_y1, prior_y0] ...
    = Compute_NB_Parameter_Estimates (class_0_train_data, class_1_train_data)
%%% fit the exponential class-conditional for input attribute 1 
[exp_0_1_mu] = expfit(class_0_train_data(:,1));
[exp_1_1_mu] = expfit(class_1_train_data(:,1));

%%%% fit the class-conditional of the second attribute with normal distribution
[norm_0_2_mu,norm_0_2_sigma] = normfit(class_0_train_data(:,2));
[norm_1_2_mu,norm_1_2_sigma] = normfit(class_1_train_data(:,2));

%%%% fit the class-conditional of the third attribute with normal distribution
[norm_0_3_mu,norm_0_3_sigma] = normfit(class_0_train_data(:,3));
[norm_1_3_mu,norm_1_3_sigma] = normfit(class_1_train_data(:,3));

%%%% fit the class-conditional of the fourth attribute with normal distribution
[norm_0_4_mu,norm_0_4_sigma] = normfit(class_0_train_data(:,4));
[norm_1_4_mu,norm_1_4_sigma] = normfit(class_1_train_data(:,4));

%%% fit the exponential class-conditional for input attribute 5
[exp_0_5_mu] = expfit(class_0_train_data(:,5));
[exp_1_5_mu] = expfit(class_1_train_data(:,5));

%%%% fit the class-conditional of the sixth attribute with normal distribution
[norm_0_6_mu,norm_0_6_sigma] = normfit(class_0_train_data(:,6));
[norm_1_6_mu,norm_1_6_sigma] = normfit(class_1_train_data(:,6));

%%% fit the exponential class-conditional for input attribute 7
[exp_0_7_mu] = expfit(class_0_train_data(:,7));
[exp_1_7_mu] = expfit(class_1_train_data(:,7));

%%% fit the exponential class-conditional for input attribute 8
[exp_0_8_mu] = expfit(class_0_train_data(:,8));
[exp_1_8_mu] = expfit(class_1_train_data(:,8));

% Compute priors
N1 = size(class_0_train_data,1);
N2 = size(class_1_train_data,1);

prior_y1 = N2/(N1 + N2);
prior_y0 = 1 - prior_y1;
end