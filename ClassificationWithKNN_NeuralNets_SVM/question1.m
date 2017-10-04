function [knnAccuracy,SVMAccuracy] = question1()

clear();

file_trainData = 'Human Activity Recognition\X_train.txt';
delimiterIn = ' ';
trainingData = importdata(file_trainData,delimiterIn);

file_trainClass = 'Human Activity Recognition\y_train.txt';
trainingClass=importdata(file_trainClass);

file_testData = 'Human Activity Recognition\X_test.txt';
delimiterIn = ' ';
testData = importdata(file_testData,delimiterIn);

file_testClass = 'Human Activity Recognition\y_test.txt';
testClass=importdata(file_testClass);

Model_KNN = fitcknn(trainingData,trainingClass,'NumNeighbors',5,'Distance','euclidean');

[label,score,cost] = predict(Model_KNN,testData);

[count,y]=size(label);  

accuratePredictionsKNN=0;
for i=1:count
    if(label(i)==testClass(i))
        accuratePredictionsKNN=accuratePredictionsKNN+1;
    end
end
knnAccuracy=(accuratePredictionsKNN/count)*100;


%%% SVM begins here 
uniqueClass=unique(trainingClass);
numClasses=length(uniqueClass);

svmModels={};

for i=1:numClasses
    y= (trainingClass==uniqueClass(i));
    svmModels{i}=fitcsvm(trainingData,y,'KernelFunction','polynomial','PolynomialOrder',2);
end

scoreMatrixSVM=[];
for i=1:numClasses
    [labelSVM,scoreSVM,costSVM] = predict(svmModels{i},testData);
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
    if(svmLabels(i)==testClass(i))
        accuratePredictionsSVM=accuratePredictionsSVM+1;
    end
end
SVMAccuracy=(accuratePredictionsSVM/count)*100;

end
