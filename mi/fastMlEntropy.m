function [ mlent ] = fastMlEntropy( X, labels )

fNum = size(X, 2);
lablesNum = size(labels, 2);
mlent = zeros(fNum, lablesNum);
% labels(labels == 0) = -1;

parfor f = 1:fNum
    for l = 1:lablesNum
        mlent(f, l) = condentropy(X(:,f), labels(:, l));
    end
end

mlent(isnan(mlent)) = 0;

end

