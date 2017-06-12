function [ x ] = normalize( x )
% Normalizes the input using mean and standard deviation
% Arguments:    1. Raw data
%
% Returns:  Normalized data
   mu = zeros(1,size(x,2));
   stddev = zeros(1, size(x,2));
   
   for i = 1:size(mu,2)
       mu(1,i) = mean(x(:,i));
       stddev(1,i) = std(x(:,i));
       x(:,i) = (x(:,i)-mu(1,i))/stddev(1,i);
   end
end

