function returnTheta = DoTraining(X,y,lambda)

m = size(X,1);
n = size(X,2);

theta = zeros(785,1);

options = optimset('GradObj', 'on', 'MaxIter', 50);


for i = 1:10
  [returnTheta(i,:),costs] = fmincg(@(t)(Jtheta(t, X, (y==i), lambda)), theta, options);
end

end