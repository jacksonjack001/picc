function y = getBatch_Unet_noCrop(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [1024,1024] ;
opts.label_imageSize=[1024, 1024];
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.rgbMean = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.labelStride = 1 ;
opts.labelOffset = 1 ;
opts.classWeights = ones(1,2,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.useGpu = false ;
opts = vl_argparse(opts, varargin);

if opts.prefetch
    ims = [] ;
    labels = [] ;
    return ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
    opts.rgbMean = single([128;128;128]) ;
end
if ~isempty(opts.rgbMean)
    opts.rgbMean = reshape(opts.rgbMean, [1 1 3]) ;
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
    numel(images)*opts.numAugments, 'single') ;
labels = ones(opts.imageSize(1), opts.imageSize(2), 1, numel(images)*opts.numAugments, 'single') ;

classWeights = [opts.classWeights(:)'] ;
im = cell(1,numel(images)) ;
si = 1 ;

for i=1:numel(images)
    % acquire image
    if isempty(im{i})
        rgbPath = sprintf(imdb.paths.image, imdb.images.name{images(i)}) ;
        s1=imdb.paths.classSegmentation;
        s2=strrep(s1,'\','/');
        labelsPath = sprintf(s2, imdb.images.name{images(i)}) ;
        rgb = double(imread(rgbPath) );
%         rgb = rgb{1} ;
        anno = imread(labelsPath) ;
        anno=anno/255;
        %         anno(anno==0)=1;
        %         anno(anno==255)=0;
    else
        rgb = im{i} ;
    end
    if size(rgb,3) == 1
        rgb = cat(3, rgb, rgb, rgb) ;
    end
    
    
    for ai = 1:opts.numAugments
        
        if ~isempty(opts.rgbMean)
            ims(:,:,:,si) = bsxfun(@minus, rgb, opts.rgbMean) ;
        else
            ims(:,:,:,si) = rgb ;
        end
        
        tlabels = anno ;
        tlabels = single(tlabels) ;
        tlabels = mod(tlabels + 1, 256) ; % 0 = ignore, 1 = bkg
        labels(:,:,1,si) = tlabels ;
        si = si + 1 ;
    end
end
if opts.useGpu
    ims = gpuArray(ims) ;
end
y = {'input', ims, 'label', labels} ;

% aa=cat(3,labels,labels,labels);
% aa(:,:,1)=anno;
% imshow(aa,[])
