function [accuracy] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels)

classes=unique(trainingLabels);
num_classes=length(classes);

k=10;
n=50;
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
    
    entropyMatrix=zeros(sizeunLabeled,1);
    for j=1:sizeunLabeled
        [probabilityVector]=test_LR_Classifier(unlabeledMatrix(j,:),trained_weights,num_classes);
        entropy=sum(-probabilityVector.*log2(probabilityVector));
        entropyMatrix(j)=entropy;
    end
    
    [ent,index]=sort(entropyMatrix,'descend');
    trainingMatrix=vertcat(trainingMatrix,unlabeledMatrix(index(1:k),:));
    trainingLabels=vertcat(trainingLabels,unlabeledLabels(index(1:k)));
    unlabeledMatrix(index(1:k),:)=[];
    unlabeledLabels(index(1:k))=[];
end
end