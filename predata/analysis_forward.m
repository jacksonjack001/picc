opts.expDir = 'data/picc/fcn-32s' ;
opts.dataDir = 'data/picc' ;
for epoch=[140:5:500];
    epoch
    opts.modelPath = ['data/picc/fcn-32s/net-epoch-',num2str(epoch),'.mat' ];
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
    opts.gpus = [1] ;
    if ~isempty(opts.gpus)
        gpuDevice(opts.gpus(1));
    end
    
    imdb = load(opts.imdbPath) ;
    % Get validation subset
    val = find(imdb.images.set == 1 & imdb.images.segmentation) ;
    
    
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net.net) ;
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    inputVar = 'input' ;
    
    
    if ~isempty(opts.gpus)
        gpuDevice(opts.gpus(1)) ;
        net.move('gpu') ;
    end
    net.mode = 'train' ;
    
    
    numGpus = 0 ;
    confusion = zeros(2) ;
    si=1;
    for i =[1]
        imId = val(i) ;
        name = imdb.images.name{imId} ;
        rgbPath = sprintf(imdb.paths.image, name) ;
        labelsPath = sprintf(imdb.paths.classSegmentation, name) ;
        
        % Load an image and gt segmentation
        rgb = vl_imreadjpeg({rgbPath}) ;
        rgb_real = imread(rgbPath);
        rgb = rgb{1} ;
        
        anno = imread(labelsPath) ;
        anno = anno;
        anno(anno==0)=1;
        anno(anno==255)=0;
        lb = single(anno) ;
        lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg
        im = single(rgb(:,:,1));
        im = bsxfun(@minus, single(rgb(:,:,1)), net.meta.normalization.averageImage(:,:,1)) ;
        im = cat(3,im,im,im);
        ims(:,:,:,si)=im;
        
        if ~isempty(opts.gpus)
            img = gpuArray(ims) ;
        end
        labels(:,:,:,si)=lb;
        si=si+1;
        
    end
    inputs= {'input', img, 'label', labels} ;
    bridge_ov=[35,37,44,46,53 ...
        ,55,62,64,71,72 ...
        ];
    bridge_ov=[35,44,53,62,71,72];
    for i=bridge_ov
        net.vars(i).precious=1;
    end
    net.eval(inputs) ;
    
    in=1;T={};
    for i=bridge_ov
        T{in}=gather(net.vars(i).value);
        size(T{in});
        in=in+1;
    end
    
    
    for i=1:in-1
        i
        predir_show=['unet',num2str(epoch),'/featuremaps_show',num2str(i)];
        predir_save=['unet',num2str(epoch),'/featuremaps_save',num2str(i)];
        predir_mat=['unet',num2str(epoch)];
    
%         mkdir(predir_show);mkdir(predir_save);
        mkdir(predir_mat);
        tt=T{i}(:,:,:,1);
        n=size(tt,3);
%         for j=1:n
%             imwrite(tt(:,:,j),[predir_save,'/',num2str(j),'.png']);
%             imshow(tt(:,:,j),[]);
%             saveas(gcf,[predir_show,'/',num2str(j),'.png'])
%         end
        save([predir_mat,'/',num2str(i),'.mat'],'tt');
    end
    
end

% % 1: block average, all the same
% sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
% stats = struct() ;
% for k = 1:numel(sel)
%     stats.(net.layers(sel(k)).outputs{1}) = net.layers(sel(k)).block.average ;
% end
% stats.error