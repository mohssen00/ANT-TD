function [pheromones, V] = ANT_TD2(X, pheromones, V, nCycle, decayRate, fCorr, flCorr, flCorr2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%       X                           - n × d matrix, d dimensional training set with n patterns
%       nCycle                      - maximum number of cycles that algorithm repeated.
%       decayRate                   - define the decay rate of the pheromone on each feature.
%       featureCorrelation          - correlation matrix of pair features.
%       featureLabelCorrelation     - correlation of features and class labels
%       heuristicMethod             - the method for calculation heuristic
% Output:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, nFeautures] = size(X);

% nAnt - define the number of agents (number of ants). 
nAnt = 50;


% NF - the number of features selected by each agent in each cycle.

minNf = floor(nFeautures / 8);
maxNf = floor(nFeautures / 6);

% NF = floor(nFeautures / 4);

for c=1:nCycle
    % tic;
    fprintf('cycle : %i \n', c);
    randomAnts = randperm(nFeautures, nAnt);
    FC = zeros(nAnt, nFeautures);
    temp = [];
    for ant=1:nAnt
        NF = randi([minNf, maxNf]);
        % fprintf(' ant : %i / %i \n', ant, nAnt);
        [temp(ant, :), FC(ant, :)] = moveAnt(randomAnts(ant), V, pheromones, ...
            NF, nFeautures, fCorr, flCorr, flCorr2);
        
    end 
    
    V = mean(temp, 1);
    FC = sum(FC);
    pheromones = ( (1 - decayRate) * pheromones) + ( FC ./ sum(FC) );
    
%     subplot(2,1,1);
%     plot(1:nFeautures, pheromones);
%     pause(.01);
%     
%     subplot(2,1,2);
%     plot(1:nFeautures, V);
%     pause(.01);
    
    % toc;
end
end

function [V, featureCounter] = moveAnt (ant, V, pheromones, NF, nFeautures, fCorr, flCorr, flCorr2)    

featureCounter = zeros(1, nFeautures);
visitedFeatures = zeros(1, nFeautures);
visitedFeatures(ant) = 1;

explore = 0;
exploit = 0;
counter = 0;
yes = 0;

while counter < NF
    exploreExploitCoeff = 1 * (1 - counter / NF) ^ 0.7;
%     exploreExploitCoeff = 0.7;
    
    if rand() > exploreExploitCoeff
        exploit = exploit + 1;
        next = heuristic(visitedFeatures, V, pheromones);
        if (next == ant)
            yes = yes + 1;
        end
    else 
        explore = explore + 1;
        next = probability(visitedFeatures, V, pheromones);
        if (next == ant)
            yes = yes + 1;
        end
    end
    
    featureCounter(next) = featureCounter(next) + 1;
    counter = counter + 1;
    visitedFeatures(next) = 1;
    r = reward(ant, next, visitedFeatures, fCorr, flCorr, flCorr2);
    V(ant) = V(ant) + ( 0.5 * ( r + (0.8 * V(next)) - V(ant) ) );
    ant = next;
    
end

end

function [next] = heuristic (visitedFeatures, V, pheromones)
r = (V .* pheromones) .* ~visitedFeatures;
[~, next] = max(r);
end

function [next] = probability(visitedFeatures, V, pheromones)
r = (V .* pheromones) .* ~visitedFeatures;
prob = (r) ./ sum(r);
cs = cumsum(prob);
next = find(cs>rand,1);
end

function r = reward(current, next, visited, fCorr, flCorr, flCorr2)
sim = fCorr(current, next);
% sim = mean(fCorr([find(visited == 1), state], :));
r = ( (1/1+sim) .* flCorr(next) );
end



