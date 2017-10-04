function [accuracy] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels)

classes=unique(trainingLabels);
num_classes=length(classes);

k=10;
n=50;
[y,idx] = datasample(unlabeledMatrix,k);
accuracy=zeros(n,1);
sizetestingMatrix=length(testingMatrix);
for i=1:n
    sizeunLabeled=length(unlabeledMatrix);
    [trained_weights] = train_LR_Classifier(trainingMatrix,trainingLabels,num_classes);
    testOutputLabels=zeros(length(testingMatrix),1);
    for j=1:sizetestingMatrix
        [probabilityVector] = test_LR_Classifier(testingMatrix(j,:),trained_weights,num_classes);
        [mx,idx]=max(probabilityVector);
        testOutputLabels(j)=classes(idx);
    end
    accuratePredictions=0;
    for z=1:sizetestingMatrix
        if(testOutputLabels(z)==testingLabels(z))
            accuratePredictions=accuratePredictions+1;
        end
    end
    accuracy(i)=(accuratePredictions/sizetestingMatrix)*100;
    
    for j=1:sizeunLabeled
        test_LR_Classifier(unlabeledMatrix(j,:),trained_weights,num_classes);
    end
        [y,idx]=datasample(unlabeledMatrix,k);
        trainingMatrix=vertcat(trainingMatrix,y);
        trainingLabels=vertcat(trainingLabels,unlabeledLabels(idx));
        unlabeledMatrix(idx,:)=[];
        unlabeledLabels(idx)=[];
end
end