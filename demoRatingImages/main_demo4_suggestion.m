% demo - load a trained model, rate the images and show the attributes
%
% The open-source caffe package is used to train the model. Here, to
% demonstrate our model, we use the matlab interface to load the model
% (caffe version).
% 
% The code and model can only be used for non-commercial cases. 
% 
% Shu Kong
% skong2@ics.uci.edu

clc
close all

addpath(genpath('./matlab'));
caffe.set_mode_cpu();

fprintf('set path...\n');

%% load model, mean image, etc.
fprintf('initialization...\n');

weightsName = 'wholeNetInitial';
model = ['./', weightsName, '.prototxt']; 
weights = ['./', weightsName, '.caffemodel'];
if ~exist('net', 'var')
    net = caffe.Net(model, weights, 'test');
end
meanImg = './mean_AADB_regression_warp256.binaryproto';
meanImg = caffe.io.read_mean(meanImg);

%% input info
imgName = 'tmp1.jpg';

%% read all testing images 
imSize = 227;
imOrg = imread(['./' imgName]);

im = imOrg;
if numel(size(im)) == 2
    im = repmat(im, [1,1,3]);
end
im = imresize(im, [imSize imSize]);
im = im(:, :, [3, 2, 1]); % convert from RGB to BGR
im = permute(im, [2, 1, 3]); % permute width and height
im = single(im); % convert to single precision
im = single(im) - meanImg(15:241,15:241,:);    % subtract mean


predAttMat = -ones(12, 1);

res = net.forward( {im} ); 

predAttMat(1) = net.blobs('fc9_BalancingElement').get_data();
predAttMat(2) = net.blobs('fc9_ColorHarmony').get_data();
predAttMat(3) = net.blobs('fc9_Content').get_data();
predAttMat(4) = net.blobs('fc9_DoF').get_data();
predAttMat(5) = net.blobs('fc9_Light').get_data();
predAttMat(6) = net.blobs('fc9_MotionBlur').get_data();
predAttMat(7) = net.blobs('fc9_Object').get_data();
predAttMat(8) = net.blobs('fc9_Repetition').get_data();
predAttMat(9) = net.blobs('fc9_RuleOfThirds').get_data();
predAttMat(10) = net.blobs('fc9_Symmetry').get_data();
predAttMat(11) = net.blobs('fc9_VividColor').get_data();
predAttMat(12) = net.blobs('fc11_score').get_data();

%% output information
curfigure = figure;
set(curfigure, 'Position', [10 10 1700 1000]);

maxSz = 800;
a = size(imOrg);
if a(1)>maxSz || a(2)>maxSz
    a = maxSz/max(a);
    imOrg = imresize(imOrg,a);
end
subplot(2,2,1);
imshow(imOrg);

text_str = cell(12,2);
scoreList = zeros(12,1);

score = predAttMat(12);
if score > 1
    score = 1;
elseif score < 0
    score = 0;
end
fprintf('\naesthetic score: %.4f', score);
outstrScore = sprintf('Aesthetic rating');
text_str{1,1} = outstrScore;
scoreList(1) = score;

if score >= 0.60
    fprintf('this is a high-aesthetic image.\n');
%     outstrScore = strcat(outstrScore, ' this is a high-aesthetic image.');
elseif score < 0.6 && score>=0.4
    fprintf('this is a so-so image.\n');
%     outstrScore = strcat(outstrScore, ' this is a so-so image.');
elseif score < 0.4
    fprintf('this is a low-aesthetic image.\n');
%     outstrScore = strcat(outstrScore, ' this is a low-aesthetic image.');
end


score = predAttMat(1);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tBalancingElement: %.4f -- ', score);
outstrScore = sprintf('BalancingElement');
text_str{2,1} = outstrScore;
scoreList(2) = score;
if score >= 0.10
    fprintf('balanced elements.\n');
%     outstrScore = strcat(outstrScore, ' balanced elements.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('unbalanced elements.\n');
%     outstrScore = strcat(outstrScore, ' unbalanced elements');
end


score = predAttMat(2);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tColorHarmony: %.4f -- ', score);
outstrScore = sprintf('ColorHarmony');
scoreList(3) = score;
text_str{3,1} = outstrScore;
if score >= 0.10
    fprintf('color harmony.\n');
%     outstrScore = strcat(outstrScore, ' color harmony.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('bad color combination.\n');
%     outstrScore = strcat(outstrScore, ' bad color combination.');
end


score = predAttMat(3);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tContent: %.4f -- ', score);
outstrScore = sprintf('Content');
text_str{4,1} = outstrScore;
scoreList(4) = score;
if score >= 0.10
    fprintf('having interesting content.\n');
%     outstrScore = strcat(outstrScore, ' having interesting content.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('boring content.\n');
%     outstrScore = strcat(outstrScore, ' boring content.');
end

score = predAttMat(4);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tDoF: %.4f -- ', score);
outstrScore = sprintf('DoF');
text_str{5,1} = outstrScore;

scoreList(5) = score;
if score >= 0.10
    fprintf('having depth of field.\n');
%     outstrScore = strcat(outstrScore, ' having depth of field.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('Out of Focus on Foreground.\n');
%     outstrScore = strcat(outstrScore, ' Out of Focus on Foreground.');
end

score = predAttMat(5);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tLight: %.4f -- ', score);
outstrScore = sprintf('Light');
text_str{6,1} = outstrScore;
scoreList(6) = score;
if score >= 0.10
    fprintf('having interesting lighting.\n');
%     outstrScore = strcat(outstrScore, ' having interesting lighting.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('bad lighting.\n');
%     outstrScore = strcat(outstrScore, ' bad lighting.');
end

score = predAttMat(6);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tMotionBlur: %.4f -- ', score);
outstrScore = sprintf('MotionBlur');
text_str{7,1} = outstrScore;
scoreList(7) = score;
if score >= 0.10
    fprintf('having motion blur.\n');
%     outstrScore = strcat(outstrScore, ' having motion blur.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('undesired motion blur (camera shaking).\n');
%     outstrScore = strcat(outstrScore, ' undesired motion blur (camera shaking).');
end

score = predAttMat(7);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tObject: %.4f -- ', score);
outstrScore = sprintf('Object');
text_str{8,1} = outstrScore;
scoreList(8) = score;
if score >= 0.10
    fprintf('having clear/emphasized object.\n');
%     outstrScore = strcat(outstrScore, ' having clear/emphasized object.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('no object emphasis.\n');
%     outstrScore = strcat(outstrScore, ' no object emphasis.');
end

score = predAttMat(8);
score = max([score,0]);
score = min([score,1]);
fprintf('\tRepetition: %.4f -- ', score);
outstrScore = sprintf('Repetition');
text_str{9,1} = outstrScore;
scoreList(9) = score;
if score >= 0.10
    fprintf('having repeated pattern.\n');
%     outstrScore = strcat(outstrScore, ' having repeated pattern.');
elseif score < 0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
end

score = predAttMat(9);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tRuleOfThirds: %.4f -- ', score);
outstrScore = sprintf('RuleOfThirds');
text_str{10,1} = outstrScore;
scoreList(10) = score;
if score >= 0.10
    fprintf('having good rule of thirds.\n');
%     outstrScore = strcat(outstrScore, ' having good rule of thirds.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('bad component placement.\n');
%     outstrScore = strcat(outstrScore, ' bad component placement.');
end

score = predAttMat(10);
score = max([score,0]);
score = min([score,1]);
fprintf('\tSymmetry: %.4f -- ', score);
outstrScore = sprintf('Symmetry');
text_str{11,1} = outstrScore;
scoreList(11) = score;
if score >= 0.10
    fprintf('having symmetry pattern.\n');
%     outstrScore = strcat(outstrScore, ' having symmetry pattern.');
elseif score < 0.1 
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
end

score = predAttMat(11);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tVividColor: %.4f -- ', score);
outstrScore = sprintf('VividColor');
text_str{12,1} = outstrScore;
scoreList(12) = score;
if score >= 0.10
    fprintf('having vivid color.\n');
%     outstrScore = strcat(outstrScore, ' having vivid color.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('dull/boring color.\n');
%     outstrScore = strcat(outstrScore, ' dull/boring color.');
end

%% caption
position = zeros(12, 2);
i = 1;
position(i, 1) = size(imOrg,2)+10;
position(i, 2) = position(i, 2)+(i-1)*40;
text( position(i, 1), position(i, 2), text_str{i, 1}, 'FontSize',15, 'FontWeight', 'bold');

position(i, 1) = size(imOrg,2)+500;
position(i, 2) = position(i, 2)+(i-1)*40;
goldenColor =  [0,0,154]/255;
if scoreList(1) >= 0.8
    text( position(i, 1), position(i, 2), '★ ★ ★ ★ ★', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor); 
elseif scoreList(1) < 0.8 && scoreList(1) >= 0.6
    text( position(i, 1), position(i, 2), '★ ★ ★ ★ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
elseif scoreList(1) < 0.6 && scoreList(1) >= 0.4
    text( position(i, 1), position(i, 2), '★ ★ ★ ☆ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
elseif scoreList(1) < 0.4 && scoreList(1) >= 0.2
    text( position(i, 1), position(i, 2), '★ ★ ☆ ☆ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
else
    text( position(i, 1), position(i, 2), '★ ☆ ☆ ☆ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
end
   

for i = 2:12
    position(i, 1) = size(imOrg,2)+100;
    position(i, 2) = position(i, 2)+(i-1)*40;
    text( position(i, 1), position(i, 2), text_str{i, 1}, 'FontSize',15, 'FontWeight', 'bold');
    
    
    position(i, 1) = size(imOrg,2)+600;
    if scoreList(i) >= 0.1
        text( position(i, 1), position(i, 2), '☑', 'FontSize',20, 'FontWeight', 'bold', 'Color', [24,206,19]/255);
    elseif scoreList(i) < 0.1 && scoreList(i) > -0.1
        text( position(i, 1), position(i, 2), '☐', 'FontSize',20, 'FontWeight', 'bold', 'Color', [128,128,128]/255);
    else
        text( position(i, 1), position(i, 2), '☒', 'FontSize',20, 'FontWeight', 'bold', 'Color', [250,62,4]/255);
    end
end


%% comparison
subplot(2,1,2);
imgName = 'tmp2.jpg';

%% read all testing images 
imSize = 227;
imOrg = imread(['./' imgName]);

im = imOrg;
if numel(size(im)) == 2
    im = repmat(im, [1,1,3]);
end
im = imresize(im, [imSize imSize]);
im = im(:, :, [3, 2, 1]); % convert from RGB to BGR
im = permute(im, [2, 1, 3]); % permute width and height
im = single(im); % convert to single precision
im = single(im) - meanImg(15:241,15:241,:);    % subtract mean


predAttMat = -ones(12, 1);

res = net.forward( {im} ); 

predAttMat(1) = net.blobs('fc9_BalancingElement').get_data();
predAttMat(2) = net.blobs('fc9_ColorHarmony').get_data();
predAttMat(3) = net.blobs('fc9_Content').get_data();
predAttMat(4) = net.blobs('fc9_DoF').get_data();
predAttMat(5) = net.blobs('fc9_Light').get_data();
predAttMat(6) = net.blobs('fc9_MotionBlur').get_data();
predAttMat(7) = net.blobs('fc9_Object').get_data();
predAttMat(8) = net.blobs('fc9_Repetition').get_data();
predAttMat(9) = net.blobs('fc9_RuleOfThirds').get_data();
predAttMat(10) = net.blobs('fc9_Symmetry').get_data();
predAttMat(11) = net.blobs('fc9_VividColor').get_data();
predAttMat(12) = net.blobs('fc11_score').get_data();

%% output information

maxSz = 800;
a = size(imOrg);
if a(1)>maxSz || a(2)>maxSz
    a = maxSz/max(a);
    imOrg = imresize(imOrg,a);
end
subplot(2,2,3);
imshow(imOrg);

text_str = cell(12,2);
scoreList = zeros(12,1);

score = predAttMat(12);
if score > 1
    score = 1;
elseif score < 0
    score = 0;
end
fprintf('\naesthetic score: %.4f', score);
outstrScore = sprintf('Aesthetic rating');
text_str{1,1} = outstrScore;
scoreList(1) = score;

if score >= 0.60
    fprintf('this is a high-aesthetic image.\n');
%     outstrScore = strcat(outstrScore, ' this is a high-aesthetic image.');
elseif score < 0.6 && score>=0.4
    fprintf('this is a so-so image.\n');
%     outstrScore = strcat(outstrScore, ' this is a so-so image.');
elseif score < 0.4
    fprintf('this is a low-aesthetic image.\n');
%     outstrScore = strcat(outstrScore, ' this is a low-aesthetic image.');
end


score = predAttMat(1);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tBalancingElement: %.4f -- ', score);
outstrScore = sprintf('BalancingElement');
text_str{2,1} = outstrScore;
scoreList(2) = score;
if score >= 0.10
    fprintf('balanced elements.\n');
%     outstrScore = strcat(outstrScore, ' balanced elements.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('unbalanced elements.\n');
%     outstrScore = strcat(outstrScore, ' unbalanced elements');
end


score = predAttMat(2);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tColorHarmony: %.4f -- ', score);
outstrScore = sprintf('ColorHarmony');
scoreList(3) = score;
text_str{3,1} = outstrScore;
if score >= 0.10
    fprintf('color harmony.\n');
%     outstrScore = strcat(outstrScore, ' color harmony.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('bad color combination.\n');
%     outstrScore = strcat(outstrScore, ' bad color combination.');
end


score = predAttMat(3);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tContent: %.4f -- ', score);
outstrScore = sprintf('Content');
text_str{4,1} = outstrScore;
scoreList(4) = score;
if score >= 0.10
    fprintf('having interesting content.\n');
%     outstrScore = strcat(outstrScore, ' having interesting content.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('boring content.\n');
%     outstrScore = strcat(outstrScore, ' boring content.');
end

score = predAttMat(4);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tDoF: %.4f -- ', score);
outstrScore = sprintf('DoF');
text_str{5,1} = outstrScore;

scoreList(5) = score;
if score >= 0.10
    fprintf('having depth of field.\n');
%     outstrScore = strcat(outstrScore, ' having depth of field.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('Out of Focus on Foreground.\n');
%     outstrScore = strcat(outstrScore, ' Out of Focus on Foreground.');
end

score = predAttMat(5);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tLight: %.4f -- ', score);
outstrScore = sprintf('Light');
text_str{6,1} = outstrScore;
scoreList(6) = score;
if score >= 0.10
    fprintf('having interesting lighting.\n');
%     outstrScore = strcat(outstrScore, ' having interesting lighting.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('bad lighting.\n');
%     outstrScore = strcat(outstrScore, ' bad lighting.');
end

score = predAttMat(6);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tMotionBlur: %.4f -- ', score);
outstrScore = sprintf('MotionBlur');
text_str{7,1} = outstrScore;
scoreList(7) = score;
if score >= 0.10
    fprintf('having motion blur.\n');
%     outstrScore = strcat(outstrScore, ' having motion blur.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('undesired motion blur (camera shaking).\n');
%     outstrScore = strcat(outstrScore, ' undesired motion blur (camera shaking).');
end

score = predAttMat(7);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tObject: %.4f -- ', score);
outstrScore = sprintf('Object');
text_str{8,1} = outstrScore;
scoreList(8) = score;
if score >= 0.10
    fprintf('having clear/emphasized object.\n');
%     outstrScore = strcat(outstrScore, ' having clear/emphasized object.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('no object emphasis.\n');
%     outstrScore = strcat(outstrScore, ' no object emphasis.');
end

score = predAttMat(8);
score = max([score,0]);
score = min([score,1]);
fprintf('\tRepetition: %.4f -- ', score);
outstrScore = sprintf('Repetition');
text_str{9,1} = outstrScore;
scoreList(9) = score;
if score >= 0.10
    fprintf('having repeated pattern.\n');
%     outstrScore = strcat(outstrScore, ' having repeated pattern.');
elseif score < 0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
end

score = predAttMat(9);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tRuleOfThirds: %.4f -- ', score);
outstrScore = sprintf('RuleOfThirds');
text_str{10,1} = outstrScore;
scoreList(10) = score;
if score >= 0.10
    fprintf('having good rule of thirds.\n');
%     outstrScore = strcat(outstrScore, ' having good rule of thirds.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('bad component placement.\n');
%     outstrScore = strcat(outstrScore, ' bad component placement.');
end

score = predAttMat(10);
score = max([score,0]);
score = min([score,1]);
fprintf('\tSymmetry: %.4f -- ', score);
outstrScore = sprintf('Symmetry');
text_str{11,1} = outstrScore;
scoreList(11) = score;
if score >= 0.10
    fprintf('having symmetry pattern.\n');
%     outstrScore = strcat(outstrScore, ' having symmetry pattern.');
elseif score < 0.1 
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
end

score = predAttMat(11);
score = max([score,-1]);
score = min([score,1]);
fprintf('\tVividColor: %.4f -- ', score);
outstrScore = sprintf('VividColor');
text_str{12,1} = outstrScore;
scoreList(12) = score;
if score >= 0.10
    fprintf('having vivid color.\n');
%     outstrScore = strcat(outstrScore, ' having vivid color.');
elseif score < 0.1 && score> -0.1
    fprintf('nothing to say.\n');
%     outstrScore = strcat(outstrScore, ' no suggestion');
elseif score <= -0.1
    fprintf('dull/boring color.\n');
%     outstrScore = strcat(outstrScore, ' dull/boring color.');
end

%% caption
position = zeros(12, 2);
i = 1;
position(i, 1) = size(imOrg,2)+10;
position(i, 2) = position(i, 2)+(i-1)*40;
text( position(i, 1), position(i, 2), text_str{i, 1}, 'FontSize',15, 'FontWeight', 'bold');

position(i, 1) = size(imOrg,2)+500;
position(i, 2) = position(i, 2)+(i-1)*40;
goldenColor =  [0,0,154]/255;
if scoreList(1) >= 0.8
    text( position(i, 1), position(i, 2), '★ ★ ★ ★ ★', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor); 
elseif scoreList(1) < 0.8 && scoreList(1) >= 0.6
    text( position(i, 1), position(i, 2), '★ ★ ★ ★ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
elseif scoreList(1) < 0.6 && scoreList(1) >= 0.4
    text( position(i, 1), position(i, 2), '★ ★ ★ ☆ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
elseif scoreList(1) < 0.4 && scoreList(1) >= 0.2
    text( position(i, 1), position(i, 2), '★ ★ ☆ ☆ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
else
    text( position(i, 1), position(i, 2), '★ ☆ ☆ ☆ ☆', 'FontSize',20, 'FontWeight', 'bold', 'Color', goldenColor);
end
   

for i = 2:12
    position(i, 1) = size(imOrg,2)+100;
    position(i, 2) = position(i, 2)+(i-1)*40;
    text( position(i, 1), position(i, 2), text_str{i, 1}, 'FontSize',15, 'FontWeight', 'bold');
    
    
    position(i, 1) = size(imOrg,2)+600;
    if scoreList(i) >= 0.1
        text( position(i, 1), position(i, 2), '☑', 'FontSize',20, 'FontWeight', 'bold', 'Color', [24,206,19]/255);
    elseif scoreList(i) < 0.1 && scoreList(i) > -0.1
        text( position(i, 1), position(i, 2), '☐', 'FontSize',20, 'FontWeight', 'bold', 'Color', [128,128,128]/255);
    else
        text( position(i, 1), position(i, 2), '☒', 'FontSize',20, 'FontWeight', 'bold', 'Color', [250,62,4]/255);
    end
end
