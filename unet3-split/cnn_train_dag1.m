function [net,stats] = cnn_train_dag1(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.expDir = fullfile('data','exp') ;
opts.logfile = 'log.txt';
opts.continue = true ;
opts.batchSize = 1 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end


logfile=fullfile(opts.expDir,opts.logfile);
fid=fopen(logfile,'w');
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n') ;
    end
end

state.getBatch = getBatch ;
stats = [] ;



modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
momentum = [];

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    fprintf(fid,'%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net, stats, momentum] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs
    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.
    
    rng(epoch + opts.randomSeed) ;
    %   prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.
    
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    state.val = opts.val ;
    %     state.val = opts.val(randperm(numel(opts.val))) ;
    
    state.imdb = imdb ;
    state.momentum = momentum;
    
    if numel(opts.gpus) <= 1
        [stats.train(epoch),prof, momentum] = process_epoch(net, state, opts, 'train',fid) ;
        stats.val(epoch) = process_epoch(net, state, opts, 'val',fid) ;
        
        if opts.profile
            profview(0,prof) ;
            keyboard ;
        end
    end
    
    % save
    if ~evaluateMode
        saveState(modelPath(epoch), net, stats, momentum) ;
    end
    
    if opts.plotStatistics
        switchFigure(1) ; clf ;
        
        subplot(1,2,1)
        accTrain = [stats.train.accuracy] ;
        accVal=[stats.val.accuracy];
        plot(1:epoch,accTrain([1,2,6],:),'-o',1:epoch,accVal([1,2,6],:),'-.d')
        xlabel('epoch') ;title('accuracy') ; grid on;
        legend('Tscale1','Tscale2','Tcombine','Vscale1','Vscale2','Vcombine')
        ylim([0,1])
        
        subplot(1,2,2)
        accTrain = [stats.train.objective ];
        accVal=[stats.val.objective];
        plot(1:epoch,accTrain([1,2],:),'-o',1:epoch,accVal([1,2],:),'-.d')
        xlabel('epoch') ;title('objective') ;grid on ;
        legend('Tscale1','Tscale2','Vscale1','Vscale2')

        
        drawnow ;
        print(1, modelFigPath, '-dpdf') ;
    end
end

function [stats, prof, momentum] = process_epoch(net, state, opts, mode,fid)

% initialize empty momentum
if strcmp(mode,'train')
    if isempty(state.momentum)
        state.momentum = num2cell(zeros(1, numel(net.params))) ;
    end
end
% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

% profile
if opts.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, state.epoch, fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    fprintf(fid, '%s: epoch %02d: %3d/%3d:', mode, state.epoch, fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        inputs = state.getBatch(state.imdb, batch) ;
        
        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            state.getBatch(state.imdb, nextBatch) ;
        end
        
        if strcmp(mode, 'train')
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;
            net.eval(inputs, opts.derOutputs) ;
        else
            net.mode = 'train' ;
            net.eval(inputs) ;
        end
    end
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
        %store momentum
        if numGpus >= 1
            state.momentum = cellfun(@gather,state.momentum,'UniformOutput',false) ;
        end
        momentum= state.momentum;
    end
    
    % get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = opts.extractStatsFn(net) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == opts.batchSize + 1
        % compensate for the first iteration, which is an outlier
        adjustTime = 2*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    fprintf(fid, ' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(fid, ' %s:', f) ;
        
        fprintf(' %.5f', stats.(f)) ;
        fprintf(fid, ' %.5f', stats.(f)) ;
        
    end
    fprintf('\n') ;
    fprintf(fid, '\n') ;
    
end

if ~isempty(mmap)
    unmap_gradients(mmap) ;
end

if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile off ;
    else
        prof = mpiprofile('info');
        mpiprofile off ;
    end
else
    prof = [] ;
end

net.reset() ;
net.move('cpu') ;

function state = accumulate_gradients(state, net, opts, batchSize, mmap)
numGpus = numel(opts.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)
    
    % accumualte gradients from multiple labs (GPUs) if needed
    if numGpus > 1
        tag = net.params(p).name ;
        for g = otherGpus
            tmp = gpuArray(mmap.Data(g).(tag)) ;
            net.params(p).der = net.params(p).der + tmp ;
        end
    end
    
    switch net.params(p).trainMethod
        
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate ;
            net.params(p).value = ...
                (1 - thisLR) * net.params(p).value + ...
                (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
            
        case 'gradient'
            thisDecay = opts.weightDecay * net.params(p).weightDecay ;
            thisLR = state.learningRate * net.params(p).learningRate ;
            if ~isempty(net.params(p).der)
                state.momentum{p} = opts.momentum * state.momentum{p} ...
                    - thisDecay * net.params(p).value ...
                    - (1 / batchSize) * net.params(p).der ;
            else
                state.momentum{p} = opts.momentum * state.momentum{p} ...
                    - thisDecay * net.params(p).value;
            end
            net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;
            
        case 'otherwise'
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
    end
end

function mmap = map_gradients(fname, net, numGpus)
format = {} ;
for i=1:numel(net.params)
    format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
    f = fopen(fname,'wb') ;
    for g=1:numGpus
        for i=1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
        end
    end
    fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

function write_gradients(mmap, net)
for i=1:numel(net.params)
    mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

function unmap_gradients(mmap)

function stats = accumulateStats(stats_)

for s = {'train', 'val'}
    s = char(s) ;
    total = 0 ;
    % initialize stats stucture with same fields and same order as
    % stats_{1}
    stats__ = stats_{1} ;
    names = fieldnames(stats__.(s))' ;
    values = zeros(1, numel(names)) ;
    fields = cat(1, names, num2cell(values)) ;
    stats.(s) = struct(fields{:}) ;
    
    for g = 1:numel(stats_)
        stats__ = stats_{g} ;
        num__ = stats__.(s).num ;
        total = total + num__ ;
        
        for f = setdiff(fieldnames(stats__.(s))', 'num')
            f = char(f) ;
            stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;
            
            if g == numel(stats_)
                stats.(s).(f) = stats.(s).(f) / total ;
            end
        end
    end
    stats.(s).num = total ;
end

function stats = extractStats(net)
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
    stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

function saveState(fileName, net, stats, momentum)
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats', 'momentum') ;

function [net, stats, momentum] = loadState(fileName)
load(fileName, 'net', 'stats', 'momentum') ;
net = dagnn.DagNN.loadobj(net) ;

function epoch = findLastCheckpoint(modelDir)
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

function switchFigure(n)
if get(0,'CurrentFigure') ~= n
    try
        set(0,'CurrentFigure',n) ;
    catch
        figure(n) ;
    end
end

function prepareGPUs(opts, cold)
numGpus = numel(opts.gpus) ;
if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate') ;
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(pool) ;
    end
    pool = gcp('nocreate') ;
    if isempty(pool)
        parpool('local', numGpus) ;
        cold = true ;
    end
    if exist(opts.memoryMapFile)
        delete(opts.memoryMapFile) ;
    end
    
end
if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename)
    if numGpus == 1
        gpuDevice(opts.gpus)
    else
        spmd, gpuDevice(opts.gpus(labindex)), end
    end
end
