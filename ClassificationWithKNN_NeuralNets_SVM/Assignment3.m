[knnAccuracy1,SVMAccuracy1]=question1();

[knnAccuracy2,SVMAccuracy2,ANNAccuracy2]=question2();

x=sprintf('Problem1 - Accuracy for KNN = %f\n',knnAccuracy1);
disp(x);
x=sprintf('Problem1 - Accuracy for SVM = %f\n\n',SVMAccuracy1);
disp(x);
x=sprintf('Problem2 - Accuracy for KNN = %f\n',knnAccuracy2);
disp(x);
x=sprintf('Problem2 - Accuracy for ANN = %f\n',ANNAccuracy2);
disp(x);
x=sprintf('Problem2 - Accuracy for SVM = %f\n',SVMAccuracy2);
disp(x);