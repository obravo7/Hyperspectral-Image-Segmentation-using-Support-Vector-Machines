% Remote sensing part 3: Hyperspectral Imaging
clear all; clc 
%% load data
 fname = ['indian_pines_corrected'];
 newimage = load(fname);
 pines = newimage.indian_pines_corrected;
%% PCA of Hyperspectral Image
%reshape the data for PCA
X = reshape(pines,size(pines,1)*size(pines,2),200);
coeff = pca(X);
Ptransformed = X*coeff;
%take the first 4 principle components
Ipc1 = reshape(Ptransformed(:,1),size(pines,1),size(pines,2));
Ipc2 = reshape(Ptransformed(:,2),size(pines,1),size(pines,2));
Ipc3 = reshape(Ptransformed(:,3),size(pines,1),size(pines,2));
Ipc4 = reshape(Ptransformed(:,3),size(pines,1),size(pines,2));
figure, imshow(Ipc1,[]); 
figure, imshow(Ipc2,[]);
figure, imshow(Ipc3,[]);
figure, imshow(Ipc3,[]);
redChannel = uint8(255 * mat2gray(Ipc1));
greenChannel = uint8(255 * mat2gray(Ipc2));
blueChannel = uint8(255 * mat2gray(Ipc3));
rgbImage = cat(3, redChannel, greenChannel, blueChannel);
figure, imshow(rgbImage);
%% SVM classifier
%  Segment the data for SVM classification
fnamegt = 'indian_pines_gt';
newimg = load(fnamegt);
pines_gt = newimg.indian_pines_gt; %ground truth image 

% Select the 9 landcover classes containing the most number of pixels.
% Select 200 pixels for each class at random
% Store the values that we don't select as well as the labels
numop = 200; %number of pixels to select 

%corn notil
corn_notil = find(pines_gt == 2);         
corn_n = corn_notil(randperm(length(corn_notil), numop));
ind_2 = pines_gt(corn_n);
test_cn = setdiff(corn_notil, corn_n);
label_cn = pines_gt(test_cn);
%corn mintill
corn_mintill = find(pines_gt == 3);
corn_m = corn_mintill(randperm(length(corn_mintill), numop));
ind_3 = pines_gt(corn_m);
test_cm = setdiff(corn_mintill, corn_m);
label_cm = pines_gt(test_cm);
%green pasture
grass_pasture = find(pines_gt == 5);
grass_p = grass_pasture(randperm(length(grass_pasture), numop));
ind_5 = pines_gt(grass_p);
test_gp = setdiff(grass_pasture, grass_p);
label_gp = pines_gt(test_gp);
%grass trees
grass_trees = find(pines_gt == 6);   
grass_t = grass_trees(randperm(length(grass_trees), numop));
ind_6 = pines_gt(grass_t);
test_gt = setdiff(grass_trees, grass_t);
label_gt = pines_gt(test_gt);
%hay windrowed
hay_wind = find(pines_gt == 8);       
hay_w = hay_wind(randperm(length(hay_wind), numop));
ind_8 = pines_gt(hay_w);
test_hw = setdiff(hay_wind, hay_w);
label_hw = pines_gt(test_hw);
%soybean notil
soybean_notil = find(pines_gt == 10);
soybean_n = soybean_notil(randperm(length(soybean_notil), numop));
ind_10 = pines_gt(soybean_n);
test_sn = setdiff(soybean_notil, soybean_n);
label_sn = pines_gt(test_sn);
%soybean mintill
soybean_mintill = find(pines_gt == 11);
soybean_m = soybean_mintill(randperm(length(soybean_mintill), numop));
ind_11 = pines_gt(soybean_m);
test_sm = setdiff(soybean_mintill, soybean_m);
label_sm = pines_gt(test_sm);
%soybean clean
soybean_clean = find(pines_gt == 12);
soybean_c = soybean_clean(randperm(length(soybean_clean), numop));
ind_12 = pines_gt(soybean_c);
test_sc = setdiff(soybean_clean, soybean_c);
label_sc = pines_gt(test_sc);
%wheat
wheat = find(pines_gt == 13);
wht = wheat(randperm(length(wheat), numop));
ind_13 = pines_gt(wht);
test_wh = setdiff(wheat, wht);
label_wh = pines_gt(test_wh);
%% Training data for SVM
% concatinated indexes are our training label
training_label = [ind_2;ind_3;ind_5;ind_6;ind_8;ind_10;ind_11;ind_12;ind_13];

% indexes 
index = [2; 3; 5; 6; 8; 10; 11; 12; 13];
% training data: (this is probably the most inefficient way to do this but I ran out of time)
c_n = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(corn_n);
    c_n = [c_n b];
end

c_m = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(corn_m);
    c_m = [c_m b];
end
g_p = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(grass_p);
    g_p = [g_p b];
end
g_t = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(grass_t);
    g_t = [g_t b];
end
h_w = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(hay_w);
    h_w = [h_w b];
end
s_n = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(soybean_n);
    s_n = [s_n b];
end
s_m = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(soybean_m);
    s_m = [s_m b];
end
s_c = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(soybean_c);
    s_c = [s_c b];
end
w_h = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(wht);
    w_h = [w_h b];
end
% training instance matrix
training_instance_matrix = [c_n;c_m;g_p;g_t;h_w;s_n;s_m;s_c;w_h];
clear c_n c_m 
% create our model
model = svmtrain(training_label, training_instance_matrix,'-t 1 -h 0'); %'-t 1'
%model = svmtrain(training_label, training_instance_matrix,'-c 1 -g 0.07 -b 1');
%% Test our model 

% testing labels
testing_label_matrix = [label_cn;label_cm;label_gp;label_gt;label_hw;label_sn;label_sm;label_sc;label_wh];

% setting the testing data 

tc_n = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_cn);
    tc_n = [tc_n b];
end

tc_m = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_cm);
    tc_m = [tc_m b];
end
tg_p = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_gp);
    tg_p = [tg_p b];
end
tg_t = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_gt);
    tg_t = [tg_t b];
end
th_w = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_hw);
    th_w = [th_w b];
end
ts_n = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_sn);
    ts_n = [ts_n b];
end
ts_m = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_sm);
    ts_m = [ts_m b];
end
ts_c = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_sc);
    ts_c = [ts_c b];
end
tw_h = [];
for i = 1: 200
    a = pines(:,:,i);
    b = a(test_wh);
    tw_h = [tw_h b];
end
% Testing matrix
testing_instance_matrix = [tc_n;tc_m;tg_p;tg_t;th_w;ts_n;ts_m;ts_c;tw_h];

% test the model 
[predicted_label, accuracy, prob_estimates] = svmpredict(testing_label_matrix, testing_instance_matrix, model);
