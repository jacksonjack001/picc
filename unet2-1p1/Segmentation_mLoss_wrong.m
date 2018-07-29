classdef Segmentation_mLoss < dagnn.Loss
    properties
        my_average=[0;0;0;0;0] ;% cumsum the epoch
        ep_average=[0;0;0;0;0] ;% calculate average every epoch
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            mass = sum(sum(inputs{2} > 0,2),1) + 1 ;% add 0 to avoid divide 0
            outputs{1}(1) = vl_nnloss(inputs{1}, inputs{2}, [], ...
                'loss', obj.loss, ...
                'instanceWeights', 1./mass) ;
            n = obj.numAveraged ;% last time number
            m = n + size(inputs{1},4) ;% this time batch size
            obj.my_average(1) = (n * obj.my_average(1) + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
            
            % here obj.loss equals 'softmaxlog'
            mconv={};msize={};mlabel={};mmass={};mot={};
            for i=1:4
                mconv{i}=inputs{i+2};msize{i}=size(mconv{i});
                mlabel{i}=imresize(inputs{2}, msize{i}(1:2), 'nearest');
                mmass{i} = sum(sum(mlabel{i} > 0, 2),1) + 1 ;
                mot{i}=vl_nnloss(mconv{i}, mlabel{i}, [], 'loss', obj.loss, 'instanceWeights', 1./mmass{i});
                
                %outputs{1}=outputs{1}+mot{i};??
                
                obj.my_average(i+1) = (n * obj.my_average(i+1) + double(gather(mot{i}))) / m ;
            end
            obj.average=obj.my_average;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            mass = sum(sum(inputs{2} > 0,2),1) + 1 ;
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', 1./mass) ;
            
            % here obj.loss equals 'softmaxlog'
            mconv={};msize={};mlabel={};mmass={};mder={};
            for i=1:4
                mconv{i}=inputs{i+2};msize{i}=size(mconv{i});
                mlabel{i}=imresize(inputs{2}, msize{i}(1:2), 'nearest');
                mmass{i} = sum(sum(mlabel{i} > 0, 2),1) + 1 ;
                
                mder{i}=vl_nnloss(mconv{i}, mlabel{i}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', 1./mmass{i});
                derInputs{i+2}=mder{i};
                
            end
            derInputs{2} = [] ;
            derParams = {} ;
        end
        function reset(obj)
            obj.ep_average = [0;0;0;0;0] ;
        end
        function obj = Segmentation_mLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
