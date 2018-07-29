function unet_concat_train_noCrop(varargin)
%FNCTRAIN Train FCN model using MatConvNet
warning('off');
% run matconvnet/matlab/vl_setupnn ;
% addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = './dice-score' ;
opts.dataDir = '../data/picc/' ;
opts.modelType = 'fcn32s' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
trainOpts.batchSize = 1;


opts.numFetchThreads = 2; % not used yet
% training options (SGD)
opts.train = struct('gpus', [1]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

trainOpts.numSubBatches = 1 ;
%so every batch has two pictures
trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.prefetch = true ;
trainOpts.expDir = opts.expDir ;

trainOpts.learningRate = [0.1 * ones(1,40) 0.01 * ones(1,40) 0.001 * ones(1,20)];
trainOpts.numEpochs = numel(trainOpts.learningRate) ;

% Get PASCAL VOC 12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = vocSetup1('dataDir', opts.dataDir) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get training and test/validation subsets
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% Get dataset statistics
if exist(opts.imdbStatsPath)
  stats = load(opts.imdbStatsPath) ;
else
  stats = getDatasetStatistics1(imdb) ;
  save(opts.imdbStatsPath, '-struct', 'stats') ;
end

% Get initial model from VGG-VD-16
net = unet_concat_BN_noCrop_dice();
net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ; 
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,2,'single') ;
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
info = cnn_train_dag1(net, imdb, getBatchWrapper(bopts), ...
                     trainOpts, ....
                     'train', train, ...
                     'val', val, ...
                     opts.train) ;

function fn = getBatchWrapper(opts)
fn = @(imdb,batch) getBatch_Unet_noCrop(imdb,batch,opts,'prefetch',nargout==0) ;