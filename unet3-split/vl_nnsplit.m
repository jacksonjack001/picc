function y = vl_nnsplit(inputs, dim, numClass, dzdy, varargin)
%VL_NNCONCAT CNN concatenate multiple inputs.
%  Y = VL_NNCONCAT(INPUTS, DIM) concatenates the inputs in the cell
%  array INPUTS along dimension DIM generating an output Y.
%
%  DZDINPUTS = VL_NNCONCAT(INPUTS, DIM, DZDY) computes the derivatives
%  of the block projected onto DZDY. DZDINPUTS has one element for
%  each element of INPUTS, each of which is an array that has the same
%  dimensions of the corresponding array in INPUTS.

opts.inputSizes = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if nargin < 2, dim = 3; end;
if nargin < 3, numClass = 2; end;
if nargin < 4, dzdy = []; end;

N=numClass;
if isempty(dzdy)
    numMaps=size(inputs{1},3);
    classNum=floor(numMaps/N);
    y = cell(1, N) ;
    s.type = '()' ;
    s.subs = {':', ':', ':', ':'} ;
    for i = 1:N
        start = 1+(i-1)*classNum ;stop=i*classNum;
        if i==N; stop=numMaps;end;
        s.subs{dim} = start:stop;
        y{i} = subsref(inputs{1},s) ;
    end
else
    y{1} = cat(dim, dzdy{:});
end
