function Theta = UpdateTheta( Theta,Alpha ,X,Y )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
  htheta = Sigmoid(X*Theta);
  for i = 1:size(X,1)
    Theta(i) = 1/m * sum((htheta-y).*X(:,i));
  end
end
