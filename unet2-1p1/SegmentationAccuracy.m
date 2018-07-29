classdef SegmentationAccuracy < dagnn.Loss

  properties
    confusion = {0,0,0,0,0,0}
  end

  methods
    function outputs = forward(obj, inputs, params)
        
      labels = gather(inputs{2}) ;
      T{1}=inputs{1};L{1}=labels;
      for i=2:5
          inputs{i+1}=gather(inputs{i+1});
          L{i}=imresize(labels,size(inputs{i+1}(:,:,1)),'nearest');
          for j=1:2
            T{i}(:,:,j)=imresize(inputs{i+1}(:,:,j),size(labels));
          end
      end
      W=[0.6,0.4,0,0,0];
      TT=W(1)*T{1}+W(2)*T{2}+W(3)*T{3}+W(4)*T{4}+W(5)*T{5};
      T{6}=TT;L{6}=L{1};
      
      C={};
      for  i=1:6
          [~, predictions] = max(T{i}, [], 3) ;
          % compute statistics only on accumulated pixels
          ok = L{1} > 0 ;
          numPixels = sum(ok(:)) ;
          obj.confusion{i} = obj.confusion{i} + accumarray([L{1}(ok),predictions(ok)],1,[2 2]) ;
          % compute various statistics of the confusion matrix
          pos = sum(obj.confusion{i},2) ;
          res = sum(obj.confusion{i},1)' ;
          tp = diag(obj.confusion{i}) ;
          mIou(i,1) = mean(tp ./ max(1, pos + res - tp)) ;
          if i==6
              pixelAccuracy = sum(tp) / max(1,sum(obj.confusion{i}(:))) ;
              meanAccuracy = mean(tp ./ max(1, pos)) ;
          end
      end
      
      obj.average=[mIou;meanAccuracy;pixelAccuracy];
      obj.numAveraged = obj.numAveraged + numPixels ;
      outputs{1} = obj.average ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = [] ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.confusion = {0,0,0,0,0,0} ;
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function obj = SegmentationAccuracy(varargin)
      obj.load(varargin) ;
    end
  end
end
