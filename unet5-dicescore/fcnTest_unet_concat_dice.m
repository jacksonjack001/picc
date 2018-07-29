function info = fcnTest_unet_concat_dice(varargin)
% run matconvnet/matlab/vl_setupnn ;
% addpath matconvnet/examples ;
% experiment and data paths

opts.expDir = './dice-score' ;
opts.dataDir = '../data/picc' ;
opts.modelPath = 'dice-score/net-epoch-100.mat' ;

% % r=12
% opts.expDir = 'data/picc/fcn-32s/fcn-32-01combine-255-25mm' ;
% opts.modelPath = 'data/picc/fcn-32s/fcn-32-01combine-255-25mm/net-epoch-427.mat' ;


[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.gpus = [1] ;
opts = vl_argparse(opts, varargin) ;


if ~isempty(opts.gpus)
    gpuDevice(opts.gpus(1));
end
% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
end
% Get validation subset
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;


net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;
% net.mode = 'train' ;
for name = {'objective', 'accuracy'}
    net.removeLayer(name) ;
end
net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
predVar = net.getVarIndex('softmax_fn') ;
net.vars(predVar).precious=1;

inputVar = 'input' ;
imageNeedsToBeMultiple = true ;


if ~isempty(opts.gpus)
    gpuDevice(opts.gpus(1)) ;
    net.move('gpu') ;
end
% net.mode = 'test' ;


confusion = zeros(2) ;
for i =1:numel(val)
    imId = val(i) ;
    name = imdb.images.name{imId} ;
    rgbPath = sprintf(imdb.paths.image, name) ;
    rgbPath = ['../',rgbPath];
    labelsPath = sprintf(imdb.paths.classSegmentation, name) ;
    labelsPath = ['../',labelsPath];
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
    
%     aa=scores_(:,:,1,:);aa=zscore(aa(:));aa=reshape(aa,1024,1024);
%     bb=scores_(:,:,2,:);bb=zscore(bb(:));bb=reshape(bb,1024,1024);
%     scores_ = cat(3,aa,bb) ;
    
    [~,pred_] = max(scores_,[],3) ;
    
    if imageNeedsToBeMultiple
        pred = imresize(pred_, sz, 'method', 'nearest') ;
    else
        pred = pred_ ;
    end
    
    
    ok = lb > 0 ;
    confusion = confusion + accumarray([lb(ok),pred(ok)],1,[2 2]) ;
    
    % Plots
    clear info ;
    [info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion) ;
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * info.iu) ;
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*info.miu, 100*info.pacc, 100*info.macc) ;
    
    figure(1) ; clf;
    imagesc(normalizeConfusion(rot90(confusion,3))) ;
    axis image ; set(gca,'ydir','normal') ;
    colormap(jet) ;colorbar
    drawnow ;
    
    % Save segmentation
    save_pred_color = ['./save_pred_color_val/'];
    save_pred_3_2 = ['./save_pred_3_view_val/'];
    if ~exist(save_pred_color)||~exist(save_pred_3_2)
        mkdir(save_pred_color);
        mkdir(save_pred_3_2);
    end
    
    % Print segmentation
    figure(100) ;clf ;
    imPath = fullfile(save_pred_3_2, [name '.png']) ;
    displayImage(rgb_real, lb, pred, imPath) ;
    drawnow ;
    
    imPath = fullfile(save_pred_color, [name '.png']) ;
    imwrite(pred,labelColors(),imPath,'png');
end

% Save results
% resPath = [opts.expDir, '/resPath/'];
% if ~exist(resPath)
%     mkdir(resPath);
% end

function nconfusion = normalizeConfusion(confusion)
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;

function displayImage(im, lb, pred, imPath)
subplot(2,2,1) ;
imshow(im,[]) ;
% axis image ;
title('source image') ;

subplot(2,2,2) ;
imshow(uint8(lb-1),[]) ;
% axis image ;
title('ground truth')

% cmap = labelColors() ;
subplot(2,2,3) ;
imshow(uint8(pred-1),[]) ;
% axis image ;
title('predicted') ;
% colormap(cmap) ;

subplot(2,2,4);
im=cat(3,im,im,im);
for i=1:1024
    for j=1:1024
        if pred(i,j)==2;
            im(i,j,1)=0;
        end
    end
end
% im(pred>1,1)=0;  out of memory
imshow(im)
title('merged')

saveas(gcf, imPath);

function cmap = labelColors()
N=2;
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
