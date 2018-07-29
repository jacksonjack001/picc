% ep=103;epoch=ep;
ep=45;epoch=ep;
% modelPath = (sprintf('data/nin/net-epoch-%d.mat', 103));
modelPath = (sprintf('dice-score/net-epoch-%d.mat', ep));
% modelFigPath = fullfile('data/nin/net-train.pdf') ;
modelFigPath = fullfile('dice-score/net-train.pdf') ;


load(modelPath, 'net', 'state', 'stats') ;
%
% $$ $ $ $e^{\pi i} + 1 = 0$ $ $ $$
%

n=1;
if get(0,'CurrentFigure') ~= n
    try
        set(0,'CurrentFigure',n) ;
    catch
        figure(n) ;
    end
end
clf ;
plots = setdiff(cat(2, fieldnames(stats.train)', fieldnames(stats.val)'), {'num', 'time', 'top5err'}) ;
% plots = fliplr(plots);
for p = plots
    p = char(p) ;
    values = zeros(0, epoch) ;
    leg = {} ;
    for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
            tmp = [stats.(f).(p)] ;
            if strcmp(p,'accuracy')
                values(end+1,:) = gather(tmp(3,:)') ;
                
            else
                values(end+1,:) = gather(tmp(1,:)') ;
            end
            leg{end+1} = f ;
        end
    end
    subplot(1,numel(plots),find(strcmp(p,plots))) ;
    plot(1:epoch, values','o-') ;
    xlabel('epoch') ;
    if strcmp(p,'top1err')
        title('error') ;
    else
        title(p);
    end
    legend(leg{:}) ;
    if strcmp(p,'accuracy')
        ylim([0,1])
    end
    grid on ;
end
drawnow ;
print(1, modelFigPath, '-dpdf') ;

