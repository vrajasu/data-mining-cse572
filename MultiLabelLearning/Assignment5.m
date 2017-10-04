%%question 1- Multi label classification using SVM

kernelFunction='polynomial';
[accPolynomial]=question1MultiLabel(kernelFunction);
x=sprintf('Problem 1 - Accuracy for SVM Polynomial = %f\n',accPolynomial);
disp(x);

kernelFunction='gaussian';
[accGaussian]=question1MultiLabel(kernelFunction);
x=sprintf('Problem 1 - Accuracy for SVM Gaussian = %f\n',accGaussian);
disp(x);

%%question 2- Multi class classification using ensemble methods
[accKNN,accSVM,accANN,accEnsemble]=  question2MultiClass();
x=sprintf('Problem2 - Accuracy for KNN = %f\n',accKNN);
disp(x);
x=sprintf('Problem2 - Accuracy for ANN = %f\n',accSVM);
disp(x);
x=sprintf('Problem2 - Accuracy for SVM = %f\n',accANN);
disp(x);
x=sprintf('Problem2 - Accuracy for Ensemble = %f\n',accEnsemble);
disp(x);