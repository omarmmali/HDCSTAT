function [cost,grad] = Jtheta(theta,X,y,lambda)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

  m = size(X,1);
  
  htheta = Sigmoid(X*theta);
  
  grad = zeros(size(theta));

  thetatmp =  theta;
  thetatmp(1) = 0;  

  cost = (-1/m) * sum(y'*log(htheta) + (1-y)'*log(1-htheta)) +((lambda/(2*m)) * (thetatmp'*thetatmp));
  
  grad = (((htheta-y)'*X)/m) + (lambda/m * thetatmp');
  
  grad = grad(:);
end

