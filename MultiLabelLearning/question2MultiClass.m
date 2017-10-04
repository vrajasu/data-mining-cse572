function [accKNN,accSVM,accANN,accEnsemble]=  question2MultiClass()
load('Data for Assignment 5 (Mini Project 3)\Handwritten Digits\X_train');
load('Data for Assignment 5 (Mini Project 3)\Handwritten Digits\X_test');
load('Data for Assignment 5 (Mini Project 3)\Handwritten Digits\y_train');
load('Data for Assignment 5 (Mini Project 3)\Handwritten Digits\y_test');

Model_KNN = fitcknn(X_train,y_train,'NumNeighbors',7,'Distance','euclidean');

[label,z] = predict(Model_KNN,X_test);

[count,y]=size(label);  
labelMatrixKNN=[];
accuratePredictionsKNN=0;
for i=1:count
    if(label(i)==y_test(i))
        accuratePredictionsKNN=accuratePredictionsKNN+1;
    end
    row=zeros(1,10);
    row(label(i))=1;
    labelMatrixKNN=vertcat(labelMatrixKNN,row);
end
accKNN=(accuratePredictionsKNN/count)*100;

%% SVM

uniqueClass=unique(y_train);
numClasses=length(uniqueClass);

svmModels={};

for i=1:numClasses
    y= (y_train==uniqueClass(i));
    svmModels{i}=fitcsvm(X_train,y,'KernelFunction','polynomial','PolynomialOrder',2);
end

scoreMatrixSVM=[];
for i=1:numClasses
    [labelSVM,scoreSVM] = predict(svmModels{i},X_test);
    scoreMatrixSVM=horzcat(scoreMatrixSVM,scoreSVM(:,2));
end

n=length(scoreMatrixSVM);
svmLabels=[];
labelMatrixSVM=[];

for i=1:n
    [mx,idx]=max(scoreMatrixSVM(i,:));
    svmLabels(i)=uniqueClass(idx);
    labelRow=zeros(1,numClasses);
    labelRow(uniqueClass(idx))=1;
    labelMatrixSVM=vertcat(labelMatrixSVM,labelRow);
end
svmLabels=transpose(svmLabels);

accuratePredictionsSVM=0;
for i=1:n
    if(svmLabels(i)==y_test(i))
        accuratePredictionsSVM=accuratePredictionsSVM+1;
    end
end
accSVM=(accuratePredictionsSVM/n)*100;

%ANN
uniqueClasses=unique(y_train);
uniqueCount=length(uniqueClasses);
classMat=zeros(uniqueCount,length(X_train));
for i=1:uniqueCount
    c1=(y_train==uniqueClasses(i));
    classMat(i,:)=c1;
end
 
net = feedforwardnet(25);
net.trainParam.showWindow=0;
X_trainTranspose=transpose(X_train);
[net,tr] = train(net,X_trainTranspose,classMat);

y=net(transpose(X_test));

testInd=vec2ind(y);
accuratePredictionsANN=0;
n=length(X_test);
labelMatrixANN=[];
for i=1:n
    if(testInd(i)==y_test(i))
        accuratePredictionsANN=accuratePredictionsANN+1;
    end
    labelrow=zeros(1,uniqueCount);
    labelrow(testInd(i))=1;
    labelMatrixANN=vertcat(labelMatrixANN,labelrow);
end
accANN=(accuratePredictionsANN/n)*100;

% %%ensemble classification
ensembleLabelMatrix=labelMatrixKNN+labelMatrixSVM+labelMatrixANN;
ensembleLabels=zeros(length(ensembleLabelMatrix),1);
for i=1:length(ensembleLabelMatrix)
    labelrow=ensembleLabelMatrix(i,:);
    [mx,idx]=max(labelrow);
    ensembleLabels(i)=uniqueClasses(idx);
end
accuratePredictionEnsemble=0;
for i=1:length(ensembleLabels)
    if(ensembleLabels(i)==y_test(i))
        accuratePredictionEnsemble=accuratePredictionEnsemble+1;
    end
end
accuratePredictionEnsemble=accuratePredictionEnsemble/length(ensembleLabels);
accEnsemble=accuratePredictionEnsemble*100;

end