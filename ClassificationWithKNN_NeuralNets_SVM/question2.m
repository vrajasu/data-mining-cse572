function [knnAccuracy,SVMAccuracy,ANNAccuracy] = question2()
clear();
load('VidTIMIT\X_train.mat');
load('VidTIMIT\y_train.mat');
load('VidTIMIT\X_test.mat');
load('VidTIMIT\y_test.mat');

Model_KNN = fitcknn(X_train,y_train,'NumNeighbors',5,'Distance','euclidean');

[label,score,cost] = predict(Model_KNN,X_test);

[count,y]=size(label);

accuratePredictionsKNN=0;
for i=1:count
    if(label(i)==y_test(i))
        accuratePredictionsKNN=accuratePredictionsKNN+1;
    end
end
knnAccuracy=(accuratePredictionsKNN/count)*100;

%%%% SVM
uniqueClass=unique(y_train);
numClasses=length(uniqueClass);

svmModels={};
for i=1:numClasses
    y= (y_train==uniqueClass(i));
    svmModels{i}=fitcsvm(X_train,y,'KernelFunction','polynomial','PolynomialOrder',2);
end

scoreMatrixSVM=[];
for i=1:numClasses
    [labelSVM,scoreSVM,costSVM] = predict(svmModels{i},X_test);
    scoreMatrixSVM=horzcat(scoreMatrixSVM,scoreSVM(:,2));
end


n=length(scoreMatrixSVM);

svmLabels=[];
for i=1:n
    [mx,idx]=max(scoreMatrixSVM(i,:));
    svmLabels(i)=uniqueClass(idx);
end
svmLabels=transpose(svmLabels);

accuratePredictionsSVM=0;
for i=1:count
    if(svmLabels(i)==y_test(i))
        accuratePredictionsSVM=accuratePredictionsSVM+1;
    end
end
SVMAccuracy=(accuratePredictionsSVM/count)*100;

%% neural networks

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
count=length(X_test);
for i=1:count
    if(testInd(i)==y_test(i))
        accuratePredictionsANN=accuratePredictionsANN+1;
    end
end
ANNAccuracy=(accuratePredictionsANN/count)*100;

end