%%questions kmeans function
clear();
[reported_avg] = question1Knn();
n_cluster=[3 5 7];
for i=1:length(reported_avg)
    x=sprintf('Avg. SSE values(10 initializations) for k=%d = %f\n',n_cluster(i),reported_avg(i));
    disp(x);
end

%%%%% question2 - MindReading
load('testingLabels_MindReading1');
load('testingMatrix_MindReading1');
load('trainingLabels_MindReading_1');
load('trainingMatrix_MindReading1');
load('unlabeledLabels_MindReading_1');
load('unlabeledMatrix_MindReading1');

[accuracyMindRandom1] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);
[accuracyMindUncertainity1] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);

load('testingLabels_MindReading2');
load('testingMatrix_MindReading2');
load('trainingLabels_MindReading_2');
load('trainingMatrix_MindReading2');
load('unlabeledLabels_MindReading_2');
load('unlabeledMatrix_MindReading2');

[accuracyMindRandom2] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);
[accuracyMindUncertainity2] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);

load('testingLabels_MindReading3');
load('testingMatrix_MindReading3');
load('trainingLabels_MindReading_3');
load('trainingMatrix_MindReading3');
load('unlabeledLabels_MindReading_3');
load('unlabeledMatrix_MindReading3');

[accuracyMindRandom3] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);
[accuracyMindUncertainity3] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);

averageAccuracyMindRandom= (accuracyMindRandom1+accuracyMindRandom2+accuracyMindRandom3)/3;
averageAccuracyMindUncertainity=(accuracyMindUncertainity1+accuracyMindUncertainity2+accuracyMindUncertainity3)/3;
plot(averageAccuracyMindRandom,'DisplayName','averageAccuracyMindRandom');hold all;plot(averageAccuracyMindUncertainity,'DisplayName','averageAccuracyMindUncertainity');hold off;

%%%%%% MMI Dataset
load('testingLabels_1');
load('testingMatrix_1');
load('trainingLabels_1');
load('trainingMatrix_1');
load('unlabeledLabels_1');
load('unlabeledMatrix_1');

[accuracyMMIRandom1] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);
[accuracyMMIUncertainity1] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);

load('testingLabels_2');
load('testingMatrix_2');
load('trainingLabels_2');
load('trainingMatrix_2');
load('unlabeledLabels_2');
load('unlabeledMatrix_2');

[accuracyMMIRandom2] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);
[accuracyMMIUncertainity2] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);

load('testingLabels_3');
load('testingMatrix_3');
load('trainingLabels_3');
load('trainingMatrix_3');
load('unlabeledLabels_3');
load('unlabeledMatrix_3');

[accuracyMMIRandom3] = question2Random(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);
[accuracyMMIUncertainity3] = question2Uncertainty(trainingMatrix,trainingLabels,testingMatrix,testingLabels,unlabeledMatrix,unlabeledLabels);

averageAccuracyMMIRandom= (accuracyMMIRandom1+accuracyMMIRandom2+accuracyMMIRandom3)/3;
averageAccuracyMMIUncertainity=(accuracyMMIUncertainity1+accuracyMMIUncertainity2+accuracyMMIUncertainity3)/3;
plot(averageAccuracyMMIRandom,'DisplayName','averageAccuracyMMIRandom');hold all;plot(averageAccuracyMMIUncertainity,'DisplayName','averageAccuracyMMIUncertainity');hold off;


