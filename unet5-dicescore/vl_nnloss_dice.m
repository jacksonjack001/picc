function y = vl_nnloss_dice(x,c,dzdy,varargin)
%VL_NNLOSS CNN categorical or attribute loss.
%   Y = VL_NNLOSS(X, C) computes the loss incurred by the prediction
%   scores X given the categorical labels C.
%
%   The prediction scores X are organised as a field of prediction
%   vectors, represented by a H x W x D x N array. The first two
%   dimensions, H and W, are spatial and correspond to the height and
%   width of the field; the third dimension D is the number of
%   categories or classes; finally, the dimension N is the number of
%   data items (images) packed in the array.
%
%   While often one has H = W = 1, the case W, H > 1 is useful in
%   dense labelling problems such as image segmentation. In the latter
%   case, the loss is summed across pixels (contributions can be
%   weighed using the `InstanceWeights` option described below).
%
%   The array C contains the categorical labels. In the simplest case,
%   C is an array of integers in the range [1, D] with N elements
%   specifying one label for each of the N images. If H, W > 1, the
%   same label is implicitly applied to all spatial locations.
%
%   In the second form, C has dimension H x W x 1 x N and specifies a
%   categorical label for each spatial location.
%
%   In the third form, C has dimension H x W x D x N and specifies
%   attributes rather than categories. Here elements in C are either
%   +1 or -1 and C, where +1 denotes that an attribute is present and
%   -1 that it is not. The key difference is that multiple attributes
%   can be active at the same time, while categories are mutually
%   exclusive. By default, the loss is *summed* across attributes
%   (unless otherwise specified using the `InstanceWeights` option
%   described below).
%
%   DZDX = VL_NNLOSS(X, C, DZDY) computes the derivative of the block
%   projected onto the output derivative DZDY. DZDX and DZDY have the
%   same dimensions as X and Y respectively.
%
%   VL_NNLOSS() supports several loss functions, which can be selected
%   by using the option `type` described below. When each scalar c in
%   C is interpreted as a categorical label (first two forms above),
%   the following losses can be used:
%
%   Classification error:: `classerror`
%     L(X,c) = (argmax_q X(q) ~= c). Note that the classification
%     error derivative is flat; therefore this loss is useful for
%     assessment, but not for training a model.
%
%   Top-K classification error:: `topkerror`
%     L(X,c) = (rank X(c) in X <= K). The top rank is the one with
%     highest score. For K=1, this is the same as the
%     classification error. K is controlled by the `topK` option.
%
%   Log loss:: `log`
%     L(X,c) = - log(X(c)). This function assumes that X(c) is the
%     predicted probability of class c (hence the vector X must be non
%     negative and sum to one).
%
%   Softmax log loss (multinomial logistic loss):: `softmaxlog`
%     L(X,c) = - log(P(c)) where P(c) = exp(X(c)) / sum_q exp(X(q)).
%     This is the same as the `log` loss, but renormalizes the
%     predictions using the softmax function.
%
%   Multiclass hinge loss:: `mhinge`
%     L(X,c) = max{0, 1 - X(c)}. This function assumes that X(c) is
%     the score margin for class c against the other classes.  See
%     also the `mmhinge` loss below.
%
%   Multiclass structured hinge loss:: `mshinge`
%     L(X,c) = max{0, 1 - M(c)} where M(c) = X(c) - max_{q ~= c}
%     X(q). This is the same as the `mhinge` loss, but computes the
%     margin between the prediction scores first. This is also known
%     the Crammer-Singer loss, an example of a structured prediction
%     loss.
%
%   When C is a vector of binary attribures c in (+1,-1), each scalar
%   prediction score x is interpreted as voting for the presence or
%   absence of a particular attribute. The following losses can be
%   used:
%
%   Binary classification error:: `binaryerror`
%     L(x,c) = (sign(x - t) ~= c). t is a threshold that can be
%     specified using the `threshold` option and defaults to zero. If
%     x is a probability, it should be set to 0.5.
%
%   Binary log loss:: `binarylog`
%     L(x,c) = - log(c(x-0.5) + 0.5). x is assumed to be the
%     probability that the attribute is active (c=+1). Hence x must be
%     a number in the range [0,1]. This is the binary version of the
%     `log` loss.
%
%   Logistic log loss:: `logisticlog`
%     L(x,c) = log(1 + exp(- cx)). This is the same as the `binarylog`
%     loss, but implicitly normalizes the score x into a probability
%     using the logistic (sigmoid) function: p = sigmoid(x) = 1 / (1 +
%     exp(-x)). This is also equivalent to `softmaxlog` loss where
%     class c=+1 is assigned score x and class c=-1 is assigned score
%     0.
%
%   Hinge loss:: `hinge`
%     L(x,c) = max{0, 1 - cx}. This is the standard hinge loss for
%     binary classification. This is equivalent to the `mshinge` loss
%     if class c=+1 is assigned score x and class c=-1 is assigned
%     score 0.
%
%   VL_NNLOSS(...,'OPT', VALUE, ...) supports these additionals
%   options:
%
%   InstanceWeights:: []
%     Allows to weight the loss as L'(x,c) = WGT L(x,c), where WGT is
%     a per-instance weight extracted from the array
%     `InstanceWeights`. For categorical losses, this is either a H x
%     W x 1 or a H x W x 1 x N array. For attribute losses, this is
%     either a H x W x D or a H x W x D x N array.
%
%   TopK:: 5
%     Top-K value for the top-K error. Note that K should not
%     exceed the number of labels.
%
%   See also: VL_NNSOFTMAX().

% Copyright (C) 2014-15 Andrea Vedaldi.
% Copyright (C) 2016 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.instanceWeights = [] ;
opts.classWeights = [] ;
opts.threshold = 0 ;
opts.loss = 'dice-score' ;
% opts.loss = 'softmaxlog' ;
opts.topK = 5 ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

inputSize = [size(x,1) size(x,2) size(x,3) size(x,4)] ;

% Form 1: C has one label per image. In this case, get C in form 2 or
% form 3.
c = gather(c) ;
if numel(c) == inputSize(4)
  c = reshape(c, [1 1 1 inputSize(4)]) ;
  c = repmat(c, inputSize(1:2)) ;
end

hasIgnoreLabel = any(c(:) == 0);

% --------------------------------------------------------------------
% Spatial weighting
% --------------------------------------------------------------------

% work around a bug in MATLAB, where native cast() would slow
% progressively
if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(labelSize(1:2), inputSize(1:2))) ;
assert(labelSize(4) == inputSize(4)) ;
instanceWeights = [] ;
switch lower(opts.loss)
  case {'classerror', 'topkerror', 'log', 'softmaxlog', 'mhinge', 'mshinge', 'dice-score'}
    % there must be one categorical label per prediction vector
    assert(labelSize(3) == 1) ;

    if hasIgnoreLabel
      % null labels denote instances that should be skipped
      instanceWeights = cast(c(:,:,1,:) ~= 0) ;
    end

  case {'binaryerror', 'binarylog', 'logistic', 'hinge'}

    % there must be one categorical label per prediction scalar
    assert(labelSize(3) == inputSize(3)) ;

    if hasIgnoreLabel
      % null labels denote instances that should be skipped
      instanceWeights = cast(c ~= 0) ;
    end

  otherwise
    error('Unknown loss ''%s''.', opts.loss) ;
end

if ~isempty(opts.instanceWeights)
  % important: this code needs to broadcast opts.instanceWeights to
  % an array of the same size as c
  if isempty(instanceWeights)
    instanceWeights = bsxfun(@times, onesLike(c), opts.instanceWeights) ;
  else
    instanceWeights = bsxfun(@times, instanceWeights, opts.instanceWeights);
  end
end

% class weights!! add by lz
if ~isempty(opts.classWeights)
    table_class=tabulate(c(:));
    weights=table_class(:,3)/100;
%     weights=[1,0.000001]';
    instanceWeights = bsxfun(@times, onesLike(c), opts.instanceWeights) ;
    C=length(weights);
    for i=1:C
        instanceWeights(c==i)=instanceWeights(c==i)/weights(i);
    end
 
end

% --------------------------------------------------------------------
% Do the work
% --------------------------------------------------------------------

switch lower(opts.loss)
  case {'log', 'softmaxlog', 'mhinge', 'mshinge', 'dice-score'}
    % from category labels to indexes
    numPixelsPerImage = prod(inputSize(1:2)) ;
    numPixels = numPixelsPerImage * inputSize(4) ;
    imageVolume = numPixelsPerImage * inputSize(3) ;

    n = reshape(0:numPixels-1,labelSize) ;
    offset = 1 + mod(n, numPixelsPerImage) + ...
             imageVolume * fix(n / numPixelsPerImage) ;
    ci = offset + numPixelsPerImage * max(c - 1,0) ;
end

if nargin <= 2 || isempty(dzdy)
  switch lower(opts.loss)
    case 'classerror'
      [~,chat] = max(x,[],3) ;
      t = cast(c ~= chat) ;
    case 'topkerror'
      [~,predictions] = sort(x,3,'descend') ;
      t = 1 - sum(bsxfun(@eq, c, predictions(:,:,1:opts.topK,:)), 3) ;
    case 'log'
      t = - log(x(ci)) ;
    case 'softmaxlog'
      Xmax = max(x,[],3) ;
      ex = exp(bsxfun(@minus, x, Xmax)) ;
      t = Xmax + log(sum(ex,3)) - x(ci) ;
    case 'mhinge'
      t = max(0, 1 - x(ci)) ;
    case 'mshinge'
      Q = x ;
      Q(ci) = -inf ;
      t = max(0, 1 - x(ci) + max(Q,[],3)) ;
    case 'binaryerror'
      t = cast(sign(x - opts.threshold) ~= c) ;
    case 'binarylog'
      t = -log(c.*(x-0.5) + 0.5) ;
    case 'logistic'
      %t = log(1 + exp(-c.*X)) ;
      a = -c.*x ;
      b = max(0, a) ;
      t = b + log(exp(-b) + exp(a-b)) ;
    case 'hinge'
      t = max(0, 1 - c.*x) ;
    case 'dice-score'
      % here we only care about batchsize==1
      xc=gather(x);
      [~,P]=max(xc,[],3);
%       G=c;
%       confusion = accumarray([P(:),G(:)],1,[2 2]);
%       a=confusion(1,1);b=confusion(1,2);cc=confusion(2,1);d=confusion(2,2);
%       2*a/(2*a+cc+b)
%       2*d/(2*d+b+cc)
      ones1=ones(size(P));
      a=zeros(size(P));a(P==1)=ones1(P==1);
      b=zeros(size(P));b(P==2)=ones1(P==2);    
      P=cat(3,a,b);
      ones1=ones(size(c));
      a=zeros(size(c));a(c==1)=ones1(c==1);
      b=zeros(size(c));b(c==2)=ones1(c==2);    
      G=cat(3,a,b);
      t=sum(sum(2*P.*G))./sum(sum(P.*P+G.*G));

%       xC=xc(1:3,1:3,:);
      
%       sP=P(1:3,1:3,:);
%       sG=G(1:3,1:3,:);
%       2*sum(sum(sP.*sG))./sum(sum(sP.*sP+sG.*sG))
%       
%       [~,Pt]=max(xc(1:3,1:3,:),[],3);
%       Gt=c(1:3,1:3,:);
%       confusion = accumarray([Gt(:), Pt(:)],1,[2 2]);
%       a=confusion(1,1);b=confusion(1,2);cc=confusion(2,1);d=confusion(2,2);
%       2*a/(2*a+cc+b)
%       2*d/(2*d+b+cc)
      
  end
  if ~isempty(instanceWeights)
    y =  sum( t(:) );%instanceWeights(:)' *
  else
    y = sum(t(:));
  end
else
  if ~isempty(instanceWeights)
    dzdy = dzdy * instanceWeights ;
  end
  switch lower(opts.loss)
    case {'classerror', 'topkerror'}
      y = zerosLike(x) ;
    case 'log'
      y = zerosLike(x) ;
      y(ci) = - dzdy ./ max(x(ci), 1e-8) ;
    case 'softmaxlog'
      Xmax = max(x,[],3) ;
      ex = exp(bsxfun(@minus, x, Xmax)) ;
      y = bsxfun(@rdivide, ex, sum(ex,3)) ;
      y(ci) = y(ci) - 1 ;
      y = bsxfun(@times, dzdy, y) ;
    case 'mhinge'
      y = zerosLike(x) ;
      y(ci) = - dzdy .* (x(ci) < 1) ;
    case 'mshinge'
      Q = x ;
      Q(ci) = -inf ;
      [~, q] = max(Q,[],3) ;
      qi = offset + numPixelsPerImage * (q - 1) ;
      W = dzdy .* (x(ci) - x(qi) < 1) ;
      y = zerosLike(x) ;
      y(ci) = - W ;
      y(qi) = + W ;
    case 'binaryerror'
      y = zerosLike(x) ;
    case 'binarylog'
      y = - dzdy ./ (x + (c-1)*0.5) ;
    case 'logistic'
      % t = exp(-Y.*X) / (1 + exp(-Y.*X)) .* (-Y)
      % t = 1 / (1 + exp(Y.*X)) .* (-Y)
      y = - dzdy .* c ./ (1 + exp(c.*x)) ;
    case 'hinge'
      y = - dzdy .* c .* (c.*x < 1) ;
    case 'dice-score'
      xc=gather(x);
      [~,P]=max(xc,[],3);
%       G=c;
%       confusion = accumarray([P(:),G(:)],1,[2 2])
      ones1=ones(size(P));
      a=zeros(size(P));a(P==1)=ones1(P==1);
      b=zeros(size(P));b(P==2)=ones1(P==2);    
      P=cat(3,a,b);
      
      ones1=ones(size(c));
      a=zeros(size(c));a(c==1)=ones1(c==1);
      b=zeros(size(c));b(c==2)=ones1(c==2);    
      G=cat(3,a,b);
      
      
%       sP=P(1:3,1:3,:);
%       sG=G(1:3,1:3,:);
%       P=sP;G=sG;
      
      
      Numerator=2*P.*G;       numerator=sum(sum(Numerator));
      Denominator=(P.*P+G.*G);denominator=sum(sum(Denominator));
      tm=bsxfun(@times, G, denominator) ;
      tz=bsxfun(@times, P, numerator);
      
      all_dP=bsxfun(@times, 2./denominator.^2,tm-tz);
      
      % only update the true label location!!!
      dPi=G>0;% in general, dont need just convert to int type
      dP=zeros(size(x));
      dP(dPi)=all_dP(dPi);
      
      y=-bsxfun(@times, dzdy, dP);
  end
end

function y = zerosLike(x)
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),classUnderlying(x)) ;
else
  y = zeros(size(x),'like',x) ;
end

function y = onesLike(x)
if isa(x,'gpuArray')
  y = gpuArray.ones(size(x),classUnderlying(x)) ;
else
  y = ones(size(x),'like',x) ;
end
