classdef Segmentation_mLoss < dagnn.Loss

    methods
        function outputs = forward(obj, inputs, params)
            mass = sum(sum(inputs{2} > 0,2),1) + 1 ;% add 0 to avoid divide 0
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], ...
                'loss', obj.loss, ...
                'instanceWeights', 1./mass) ;
            
            fid=fopen('aa.txt','a');
            fprintf(fid, '%3d  ', outputs{1}) ;

%             subplot(1,2,1);imshow(inputs{1}(:,:,1,:),[]);
%             subplot(1,2,2);imshow(inputs{1}(:,:,2,:),[]);
%             pause(0.5)
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
            
            % here obj.loss equals 'softmaxlog'
            mconv={};msize={};mlabel={};mmass={};mot={};
            for i=1:4
                mconv{i}=inputs{i+2};msize{i}=size(mconv{i});
                mlabel{i}=imresize(inputs{2}, msize{i}(1:2), 'nearest');
                mmass{i} = sum(sum(mlabel{i} > 0, 2),1) + 1 ;
                numMaps=msize{i}(3);
                N=2;
                classNum=floor(numMaps/N);
                for j=1:N-1
                    rdi{j}=1+(j-1)*classNum:j*classNum;
                end
                rdi{N}=(N-1)*classNum+1:numMaps;
                conv=[];
                for j=1:N
                    conv(:,:,j,:)=gather(mean(mconv{i}(:,:,rdi{j},:),3));
                end
                
%                 subplot(1,2,1);imshow(conv(:,:,1,:),[]);
%                 subplot(1,2,2);imshow(conv(:,:,2,:),[]);
%                 pause(0.5)
                
                mot{i}=vl_nnloss(conv, mlabel{i}, [], 'loss', obj.loss, 'instanceWeights', 1./mmass{i});
                fprintf(fid, '%3d  ', mot{i}) ;

                outputs{1}=outputs{1}+mot{i};
                n = obj.numAveraged ;
                % in val we get error!! why?? batchsize are not all the
                % same
                if length(size(msize{i}))<4
                    msize{i}(4)=1;
                end
                m = n + 0*msize{i}(4) ;
                obj.average = (n * obj.average + 0*(2^i)*double(gather(mot{i}))) / m ;
                obj.numAveraged = m ;
            end
            fprintf(fid, '\n') ;
            fclose(fid);
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
                
                numMaps=msize{i}(3);
                N=2;
                classNum=floor(numMaps/N);
                for j=1:N-1
                    rdi{j}=1+(j-1)*classNum:j*classNum;
                end
                rdi{N}=(N-1)*classNum+1:numMaps;
                
                % merge 196 channel into  21 categories!
                conv=[];
                for j=1:N
                    conv(:,:,j,:)=gather(mean(mconv{i}(:,:,rdi{j},:),3));
                end
                
                % interp 21 losses back to 196 channels!
                mder{i}=vl_nnloss(conv, mlabel{i}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', 1./mmass{i});
                der={};T1=[];
                for j=1:N
                    der{j}=repmat(mder{i}(:,:,j,:),[1,1,length(rdi{j}),1]);
                    T1=cat(3,T1,der{j});
                end
                % derivation for all inputs except label, prediction has been devrived in the first time 
                derInputs{i+2}=zeros(size(inputs{i+2}));
                derInputs{i+2}=[];
                
                % combine derivation from former layer to formulate in the
                % last layer according to SMAP
                mder_size=size(mder{i});der_size=size(derInputs{1});
                mder_size(4)=1;
                T=[];
                for l=1:mder_size(3)
                    for h=1:mder_size(4)
                        temp=mder{i}(:,:,l,h);
                        temp1=imresize(temp, der_size(1:2), 'nearest');
                        T(:,:,l,h)=temp1;
                    end
                end
                derInputs{1}=derInputs{1}+0*1*(2^(i))*T;
                
            end
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = Segmentation_mLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
