function [accuratePredictions]=  question1MultiLabel(kernelFunc)
load('Data for Assignment 5 (Mini Project 3)\Multi Label Scene Data\X_train');
load('Data for Assignment 5 (Mini Project 3)\Multi Label Scene Data\X_test');
load('Data for Assignment 5 (Mini Project 3)\Multi Label Scene Data\y_train');
load('Data for Assignment 5 (Mini Project 3)\Multi Label Scene Data\y_test');

[x,numClasses]=size(y_train);

svmModels={};

for i=1:numClasses
    y= (y_train(:,i));
    if(strcmp(kernelFunc,'polynomial'))
        svmModels{i}=fitcsvm(X_train,y,'KernelFunction',kernelFunc,'PolynomialOrder',2);
    else
        svmModels{i}=fitcsvm(X_train,y,'KernelFunction',kernelFunc,'KernelScale','Auto');
    end
end
labelMatrixSVM=[];
for i=1:numClasses
    labelSVM = predict(svmModels{i},X_test);
    labelMatrixSVM=horzcat(labelMatrixSVM,labelSVM);
end

n=length(labelMatrixSVM);
accuratePredictions=0;
for i=1:n
    accuracy=sum(y_test(i,:)& labelMatrixSVM(i,:))/sum(y_test(i,:) | labelMatrixSVM(i,:));
    accuratePredictions=accuratePredictions+accuracy*100;
end
accuratePredictions=accuratePredictions/n;
end