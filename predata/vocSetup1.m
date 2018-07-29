function imdb = vocSetup(varargin)
opts.dataDir = fullfile('data','picc') ;
opts.includeTest = false ;
opts = vl_argparse(opts, varargin) ;

% Source images and classes
imdb.paths.image = esc(fullfile(opts.dataDir, '/CXR/', '%s.jpg')) ;
imdb.sets.id = uint8([1 2 3]) ;
imdb.sets.name = {'train', 'val', 'test'} ;
imdb.classes.id = uint8(1:2) ;
imdb.classes.name = {'ROI','background'} ;% -------------------------------------------------------------------------

imdb.classes.images = cell(1,2) ;
imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.set = [] ;
index = containers.Map() ;

% Source segmentations
  n = numel(imdb.images.id) ;
  imdb.paths.objectSegmentation = fullfile(opts.dataDir, 'mask', '%s.png') ;
  imdb.paths.classSegmentation = esc(fullfile(opts.dataDir, 'mask', '%s.png')) ;
  imdb.images.segmentation = false(1, n) ;
  [imdb, index] = addSegmentationSet(opts, imdb, index, 'dg_train', 1) ;
  [imdb, index] = addSegmentationSet(opts, imdb, index, 'dg_val', 2) ;
  if opts.includeTest, [imdb, index] = addSegmentationSet(opts, imdb, index, 'test', 3) ; end


%set 1 train, 2val
%segmentaion  to or not
function [imdb, index] = addSegmentationSet(opts, imdb, index, setName, setCode)
segAnnoPath = fullfile(opts.dataDir, 'ImageSets', [setName '.txt']) ;
fprintf('%s: reading %s\n', mfilename, segAnnoPath) ;
segNames = textread(segAnnoPath, '%s') ;
j = numel(imdb.images.id) ;
for i=1:length(segNames)
  if index.isKey(segNames{i})
    k = index(segNames{i}) ;
    imdb.images.segmentation(k) = true ;
    imdb.images.set(k) = setCode ;%
  else
    j = j + 1 ;
    index(segNames{i}) = j ;
    imdb.images.id(j) = j ;
    imdb.images.set(j) = setCode ;
    imdb.images.name{j} = segNames{i} ;
    imdb.images.classification(j) = false ;
    imdb.images.segmentation(j) = true ;
  end
end

% a=dir('picc/ROI');
% for i=3:numel(a)
%     str1=a(i).name;
%     str2=[str1(1:end-8),'.jpg'];
%     copyfile(['picc/ROI/',str1],['picc/ROI1/',str2]);    
% end
% mkdir picc/ROI1


function str=esc(str)
% -------------------------------------------------------------------------
str = strrep(str, '\', '\\') ;


