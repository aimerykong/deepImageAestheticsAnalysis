close all
clear
clc

%% overall score
load('./AADBinfo.mat');
score = [testScore, trainScore];
figure;
hist(score, 10);
xlabel('score');
ylabel('#images');



%% attributes
attNameList = {'BalacingElements', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur', 'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor' };
folderpath = './AADB_prepare';
attScore = cell(11,1);
for attID = 1:length(attNameList)
    fprintf('attribute-%s\n', attNameList{attID});
    attScore{attID} = [];
    filename = fullfile(folderpath, ['imgListTrainRegression_' attNameList{attID} '.txt']);
    numImage = numel(textread(filename,'%1c%*[^\n]'));
    scoreTMP = zeros(1,numImage);
    fn = fopen( filename, 'r' );
    for i = 1:numImage
        tline = fgets(fn);
        C = strsplit(tline, ' ');
%         imgName = C{1};
        imgLabel = str2double(C{2});
%         nameList{i} = imgName;
        scoreTMP(i) = imgLabel;
    end
    fclose(fn);
    attScore{attID} = [attScore{attID}, scoreTMP];
    
    filename = fullfile(folderpath, ['imgListTestRegression_' attNameList{attID} '.txt']);
    numImage = numel(textread(filename,'%1c%*[^\n]'));
    scoreTMP = zeros(1,numImage);
    fn = fopen( filename, 'r' );
    for i = 1:numImage
        tline = fgets(fn);
        C = strsplit(tline, ' ');
%         imgName = C{1};
        imgLabel = str2double(C{2});
%         nameList{i} = imgName;
        scoreTMP(i) = imgLabel;
    end
    fclose(fn);
    attScore{attID} = [attScore{attID}, scoreTMP];
end

%% show attribute distribution

A = cell2mat(attScore);
countA = zeros(11,3);
for attID = 1:length(attNameList)
    a = find(A(attID,:)==1);
    countA(attID, 1) = length(a);
    a = find(A(attID,:)==2);
    countA(attID, 2) = length(a);
    a = find(A(attID,:)==3);
    countA(attID, 3) = length(a);
end

figure;
h = bar(countA);
set(gca,'XTickLabel',{'BalacingElements'; 'ColorHarmony'; 'Content'; 'DoF'; 'Light'; 'MotionBlur'; 'Object'; 'Repetition'; 'RuleOfThirds'; 'Symmetry'; 'VividColor' })
l = {'negative', 'null', 'positive'};
legend(h,l);




