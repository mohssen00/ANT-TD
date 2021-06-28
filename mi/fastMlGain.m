function [ gains ] = fastMlGain( X, labels )

fNum = size(X, 2);
lablesNum = size(labels, 2);
gains = zeros(fNum, lablesNum);
labels(labels == 0) = -1;

parfor f = 1:fNum
    fE = fastEntropy(X(:, f));
    for l = 1:lablesNum
        gains(f, l) = fE - condentropy(X(:,f), labels(:, l));
    end
end

end

