clc;
clear;

addpath('./files')
addpath('./ACO');
addpath('./mi');
addpath('./MLKNN');

datasets = {'chess', 'coffee', 'cooking'};

for d=1:length(datasets)
   
    dataset_name = datasets{d};
    path = strcat('./matfile/', dataset_name);
    load(path, 'dataset', 'targets');
    
%     dataset = discretize(dataset, targets, 20);
    
    fRange = 10:10:100;
    bucketNum = length(fRange);

    iters = 20;
    % 1.accuracy, 2.HammingLoss, 3.RankingLoss, 4.OneError, 5.Coverage, 6.Average_Precision, 7.time
    aco1_acc = zeros(iters, bucketNum);
    aco1_hamming = zeros(iters, bucketNum);
    aco1_rankingloss = zeros(iters, bucketNum);
    aco1_one_error = zeros(iters, bucketNum);
    aco1_coverage = zeros(iters, bucketNum);
    aco1_average_precision = zeros(iters, bucketNum);
    aco1_time = zeros(1, iters);

    data = cell(iters, 4);
    aco_top = cell(iters, 1);
    
    decayRate = 0.2;
    nCycle = 25;
    % pre-proccesing step : remove zero-value feature column
    % zeroF = find(~any(dataset));
    % dataset(:, zeroF) = [];
    
    for i=1:iters
        disp(i);
%         tic;
        nLabels = size(targets, 2);
        [dataTrain, dataTrainLabels, dataTest, dataTestLabels] = splitData(dataset, nLabels);

        
%         z = find(all(dataTrainLabels==0));
%         dataTrainLabels(:, z)=[];
%         dataTestLabels(:, z) = [];

        nLabels = size(dataTrainLabels, 2);
        [iNum, fNum] = size(dataTrain);

        data{i, 1} = dataTrain;
        data{i, 2} = dataTrainLabels;
        data{i, 3} = dataTest;
        data{i, 4} = dataTestLabels;
        start=tic;
        mlCosine = abs( 1 - pdist2(dataTrain', dataTrainLabels', 'cosine') );
        mlCorr = abs( 1 - pdist2(dataTrain', dataTrainLabels', 'correlation') );
        mlmi = fastMlMi(dataTrain, dataTrainLabels);
%         mlIG = mlGain(dataTrain, dataTrainLabels);    
%         mlEnt = mlEntropy(dataTrain, dataTrainLabels);
%         mlsu = mlSu(dataTrain, dataTrainLabels);

        % calculate features correlation
        fCorr = abs( 1 - pdist2(dataTrain', dataTrain', 'correlation') );
        fCorr(fCorr == 0) = 0.0001;
       
        flCorr = minmax(max(mlCosine, [], 2))';
        flCorr2 = minmax(max(mlCorr, [], 2))';
        
%         flCorr2 = (max(gains, [], 2))';
        
        initialPheromone = minmax(max(mlmi, [], 2))';
        initialV = minmax(max(mlCosine, [], 2))';   
        
        fCorr(isnan(fCorr)) = 1;
        initialV(isnan(initialV)) = 0;
        flCorr(isnan(flCorr)) = 0;
        flCorr2(isnan(flCorr2)) = 0;
        
        
        [ph1, V] = ANT_TD2(dataTrain, initialPheromone, initialV, nCycle, decayRate, fCorr, flCorr, flCorr2);

%         [~, aa(1, :)] = sort(-ph1);
%         [~, aa(2, :)] = sort(-V);
%         find(aa(1,:) == aa(2,:))

        phV = [minmax(ph1); minmax(V)];
        [ph_val1, ph_idx1] = sort(-ph1);
        aco_top{i} = ph_idx1;
        aco1_time(1, i) = toc(start);
        
        parfor j=1:bucketNum
            disp(j);

            % 1.accuracy, 2.HammingLoss, 3.RankingLoss, 4.OneError, 5.Coverage, 6.Average_Precision, 7.time
            out1 = MLKNN(dataTrain(:, ph_idx1(1:fRange(j))), dataTrainLabels, dataTest(:, ph_idx1(1:fRange(j))), dataTestLabels);

            aco1_acc(i, j) = out1{1,1};
            aco1_hamming(i, j) = out1{1,2};
            aco1_rankingloss(i, j) = out1{1,3};
            aco1_one_error(i, j) = out1{1,4};
            aco1_coverage(i, j) = out1{1,5};
            aco1_average_precision(i, j) = out1{1,6};

        end


%         toc;
    end
    
end
