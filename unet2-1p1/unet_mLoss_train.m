function unet_mLoss_train(varargin)
warning('off');

opts.expDir = 'exp/' ;
opts.dataDir = 'data/picc/' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.numFetchThreads = 1; % not used yet
opts.train = struct('gpus', [1]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

trainOpts.batchSize = 1;
trainOpts.numSubBatches = 1 ;
trainOpts.continue = true ;
trainOpts.prefetch = true ;
trainOpts.expDir = opts.expDir ;
trainOpts.learningRate = [0.1* ones(1,40) 0.01 * ones(1,40) 0.001 * ones(1,23)];
trainOpts.numEpochs = numel(trainOpts.learningRate) ;

imdb = load(opts.imdbPath) ;
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;
stats = load(opts.imdbStatsPath) ;

net = unetmLoss() ;
net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% Setup data fetching options
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
info = cnn_train_dag1(net, imdb, getBatchWrapper(bopts), ...
    trainOpts, 'train', train, ...
    'val', val, opts.train) ;

function fn = getBatchWrapper(opts)
fn = @(imdb,batch) getBatch_Unet_noCrop(imdb,batch,opts,'prefetch',nargout==0) ;