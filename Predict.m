
function p = Predict (theta, X)
  m = size(X,1);
  [max,p]=max(X*theta',[],2);
endfunction
