load('mnist_all.mat');
warning('off', 'Octave:possible-matlab-short-circuit-operator');

%Training
trainingData = [ones(size(train0,1),1) train0 ones(size(train0,1),1).*10; ones(size(train1,1),1) train1 ones(size(train1,1),1);ones(size(train2,1),1) train2 (ones(size(train2,1),1).*2);ones(size(train3,1),1) train3 (ones(size(train3,1),1).*3);ones(size(train4,1),1) train4 (ones(size(train4,1),1).*4);ones(size(train5,1),1) train5 (ones(size(train5,1),1).*5);ones(size(train6,1),1) train6 (ones(size(train6,1),1).*6);ones(size(train7,1),1) train7 (ones(size(train7,1),1).*7);ones(size(train8,1),1) train8 (ones(size(train8,1),1).*8);ones(size(train9,1),1) train9 (ones(size(train9,1),1).*9)];

m = size(trainingData,1);

trainingData = double(trainingData(randperm(m),:));

lambda = 0.1;

[theta] = DoTraining(trainingData(:,1:785),trainingData(:,786),lambda); 

%Testing
testData = [ones(size(test0,1),1) test0 ones(size(test0,1),1).*10;ones(size(test1,1),1) test1 ones(size(test1,1),1);ones(size(test2,1),1) test2 (ones(size(test2,1),1).*2);ones(size(test3,1),1) test3 (ones(size(test3,1),1).*3);ones(size(test4,1),1) test4 (ones(size(test4,1),1).*4);ones(size(test5,1),1) test5 (ones(size(test5,1),1).*5);ones(size(test6,1),1) test6 (ones(size(test6,1),1).*6);ones(size(test7,1),1) test7 (ones(size(test7,1),1).*7);ones(size(test8,1),1) test8 (ones(size(test8,1),1).*8);ones(size(test9,1),1) test9 (ones(size(test9,1),1).*9)];
n = size(testData,1);
testData = double(testData(randperm(n),:));

p = Predict(theta,testData(:,1:785));

fprintf('\nAccuracy on test set: %f\n', mean(double(p == testData(:,786))) * 100);
