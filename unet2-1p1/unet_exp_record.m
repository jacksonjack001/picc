opts.expDir = 'exp_60/' ;
opts.dataDir = 'data/picc' ;
warning('off');
for kk=60
    opts.modelPath = [opts.expDir,'net-epoch-',num2str(kk),'.mat'] ;
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
    opts.gpus = [1] ;
    
    if ~isempty(opts.gpus)
        gpuDevice(opts.gpus(1));
    end
    % Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
    % segmentations
    if exist(opts.imdbPath)
        imdb = load(opts.imdbPath) ;
    end
    % Get validation subset
    val = find(imdb.images.set == 1 & imdb.images.segmentation) ;
    
    
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net.net) ;
    net.mode = 'train' ;
    for name = {'objective', 'accuracy'}
        net.removeLayer(name) ;
    end
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    net.vars(predVar).precious=1;
    
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;
    
    
    if ~isempty(opts.gpus)
        gpuDevice(opts.gpus(1)) ;
        net.move('gpu') ;
    end
    % net.mode = 'test' ;
    
    
    confusion = zeros(2) ;
    for i =[78]
        imId = val(i) ;
        imname = imdb.images.name{imId} ;
        rgbPath = sprintf(imdb.paths.image, imname) ;
        labelsPath = sprintf(imdb.paths.classSegmentation, imname) ;
        
        % Load an image and gt segmentation
        rgb = vl_imreadjpeg({rgbPath}) ;
        rgb_real = imread(rgbPath);
        
        rgb = rgb{1} ;
        anno = imread(labelsPath) ;
        anno = anno/255;
        %     anno(anno==0)=1;
        %     anno(anno==255)=0;
        
        lb = single(anno) ;
        lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg
        
        % Subtract the mean (color)
        im = bsxfun(@minus, single(rgb(:,:,1)), net.meta.normalization.averageImage(:,:,1)) ;
        im = cat(3,im,im,im);
        % Soome networks requires the image to be a multiple of 32 pixels
        if imageNeedsToBeMultiple
            sz = [size(im,1), size(im,2)] ;
            sz_ = round(sz / 32)*32 ;
            im_ = imresize(im, sz_) ;
        else
            im_ = im ;
        end
        
        if ~isempty(opts.gpus)
            im_ = gpuArray(im_) ;
        end
        
        net.eval({inputVar, im_}) ;
        scores_ = gather(net.vars(predVar).value) ;
        [~,pred_] = max(scores_,[],3) ;
        
        if imageNeedsToBeMultiple
            pred = imresize(pred_, sz, 'method', 'nearest') ;
        else
            pred = pred_ ;
        end
        
        
        ok = lb > 0 ;
        confusion = confusion + accumarray([lb(ok),pred(ok)],1,[2 2]) ;
        % Save segmentation
        save_pred_epoch= [opts.expDir, '/save_pred_epoch/',imname];
        if ~exist(save_pred_epoch)
            mkdir(save_pred_epoch);
        end
        
        
        imPath = fullfile(save_pred_epoch, [num2str(kk), '.png']) ;
%         subplot(1,2,1);imshow([scores_(:,:,1)],[]);
%         subplot(1,2,2);
        imshow([scores_(:,:,2)],[]);
        title(num2str(kk));
%         pause(0.5);
        saveas(gca,imPath);
    end
    
end

