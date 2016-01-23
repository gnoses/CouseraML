function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

sample = [0.01 0.03 0.1 0.3 1 3 10 30];
maxacc = [0,0,0];
param = [2,2];

for i=1:size(sample,2)
    for j=1:size(sample,2)
        model= svmTrain(X, y, sample(i), @(x1, x2) gaussianKernel(x1, x2, sample(j)));
        
        figure(1);
        pred = svmPredict(model, X);
        acc = mean(double(pred == y)) * 100;
%         visualizeBoundary(X, y, model);        
        str = sprintf('C=%f, sigma=%f, acc=%f', sample(i), sample(j), acc);
        title(str);
        
        figure(2);
        pred = svmPredict(model, Xval);
        acc2 = mean(double(pred == yval)) * 100;
        
        str = sprintf('C=%f, sigma=%f, acc=%f', sample(i), sample(j), acc);
        title(str);
        
        str = sprintf('acc train=%f, test=%f', acc, acc2);
        disp(str);
        
        if maxacc(1) < acc + acc2
            maxacc(1) = acc + acc2;
            maxacc(2) = acc;
            maxacc(3) = acc2;
            param(1) = sample(i);
            param(2) = sample(j);
        end
        ;
%         pause;
    end
end

% 0.01, 0.03 -> 995 % , 94 %
str = sprintf('Max acc train=%f, test=%f', maxacc(2), maxacc(3));
model= svmTrain(X, y, param(1), @(x1, x2) gaussianKernel(x1, x2, param(2)));

pred = svmPredict(model, X);
acc = mean(double(pred == y)) * 100;
figure(1);
visualizeBoundary(X, y, model);        

pred = svmPredict(model, Xval);
acc2 = mean(double(pred == yval)) * 100;
figure(2);
visualizeBoundary(Xval, yval, model);        

C = param(1);
sigma = param(2);
% =========================================================================

end
