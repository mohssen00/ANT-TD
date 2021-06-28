function [ mlmi ] = fastMlMi( X, labels )

fNum = size(X, 2);
lablesNum = size(labels, 2);
mlmi = zeros(fNum, lablesNum);
% labels(labels == 0) = -1;

parfor f = 1:fNum
    for l = 1:lablesNum
        mlmi(f, l) = mutualinfo(X(:,f), labels(:, l));
    end
end

end

