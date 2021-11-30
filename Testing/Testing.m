%% Detect the spike length in pixels



% Close all windows; Clear the Workspace; Clear the Command Window
close all; clear; clc;

% Define directory
PhotoDir = 'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\IN';
% PhotoDir = pwd;


% Image path
mypic = fullfile(PhotoDir, '318_18.tif'); 

% Read image
RGB = imread(mypic);
% imshow(RGB);

% Crop image
RGB = imcrop( RGB, [0.5 10.5 5031 6933] );

% Threshold based on the red channel
BW1 = RGB(:,:,1) > 30;
% imshow(mask1);
BW1 = bwareafilt(BW1, [10000, inf]);
% imshow(BW1);

% Redefine RGB
RGB_1 = bsxfun(@times, RGB, cast(BW1, 'like', RGB));
% imshow(RGB_1);

% RGB to gray
IG = rgb2gray(RGB_1);
% imshow(IG);

% Convert gray image to double
IG = im2double(IG);

% % Convert gray to binary
% BW1 = imbinarize(IG);
% % imshow(BW1);
% % Label large objects
% BW2 = bwareafilt(BW1, [10000, inf]);
% % imshow(BW2);

% Label spikes with 8-connectivity
[Spk, num_spks] = bwlabel(BW1, 8);

% Get current file's name as string
% a = char(filenames(i));

% Add labels to BW1
s = regionprops(Spk, 'Centroid');
imshow(BW1)
hold on
for k = 1:numel(s)
    c = s(k).Centroid;
    text(c(1), c(2), sprintf('%d', k), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle',...
        'Color','red','FontSize',14);
end
hold off


% Test image
BW = Spk == 1;
imshow(BW);

% Create a structural element(s)
se1 =  strel('disk', 10);

% Erosion
bw1 = imerode(BW, se1);
imshow(bw1);

% Dilation
bw1 = imdilate(BW, se1);
imshow(bw1);

% Opening
bw1 = imopen(BW, se1);
imshow(bw1);

% Closing
bw1 = imclose(BW, se1);
imshow(bw1);

% Internal boundary
bw1 = imdilate(BW, se1) & ~imerode(BW, se1);
imshow(bw1);

% External boundary
bw1 = imdilate(BW, se1) & ~BW;
imshow(bw1);

% Morphological gradient
bw1 = imsubtract(BW, imerode(BW, se1));
imshow(bw1);

% Thinning
bw1 = bwmorph(BW, 'thin');
imshow(bw1);

% thickening
bw1 = bwmorph(BW, 'thicken');
imshow(bw1);

% Skeletonization
bw1 = bwmorph(BW, 'skel', 10);
imshow(bw1);




% Let's see how these operations work on a gray image
% Apply mask to gray image (Gray Spike)
GS = bsxfun(@times, IG, cast(BW, 'like', IG));
imshow(GS);

% Create a structural element(s)
se1 = strel('arbitrary',eye(7)) ;

% Erosion
gray1 = imerode(GS, se1);
imshow(gray1);

% Dilation
gray1 = imdilate(GS, se1);
imshow(gray1);

% Opening
gray1 = imopen(GS, se1);
imshow(gray1);

% Closing
gray1 = imclose(GS, se1);
imshow(gray1);

% Morphological operation on these gray images are not much different from
% the performance on binary iamges. Perhaps i should try to increase the
% IG contrast


%% Histogram Stretching

% Step 1: Get the image

% Step 2: Use histograms to explore un-enanced image
figure;
imhist(IG.*255);

% Step 3: Use correlation to explore un-enhanced truecolor composite
r = RGB_1(:,:,1);
g = RGB_1(:,:,2);
b = RGB_1(:,:,3);
figure
plot3(r(:),g(:),b(:), '.')
grid('on')
xlabel('Red channel')
ylabel('Green channel')
zlabel('Blue channel')
title('Scatterplot of the Visible Bands')

% Step 4: Enhance color with histogram stretching
Stretched_RGB = imadjust(RGB_1, stretchlim(RGB_1));
figure
imshow(Stretched_RGB)
title('True Color Composite after Contrast Stretch')


% Step 5: Check histogram following the Ccontrast stretch
figure
imhist(Stretched_RGB(:,:,1));

% Step 6: Enhance truecolor composite with a decorrelation stretch
decorrstr_RGB = decorrstretch(RGB_1, 'Tol', 0.01);
figure
imshow(decorrstr_RGB)




%% Gaussian filter to detect florets

GF1 = imgaussfilt(double(BW),15);
% imshow(GF1);



%% Watershed?
% Gray spike
GS = bsxfun(@times, IG, cast(BW, 'like', IG));
imshow(GS);

bw1 = adapthisteq(GF1);
bw1 = imbinarize(bw1);
% imshow(bw1);
bw2_perim = bwperim(bw1);

overlay1 = imoverlay(GS, bw2_perim, [.3 1 .3]);
figure; imshow(overlay1);

mask_em = imextendedmax(double(GF1), 0.1);
% imshow(mask);
% imshow(bw1);
overlay1 = imoverlay(GS, bw2_perim | mask_em, [.3 1 .3]);
figure; imshow(overlay1);

GS_c = imcomplement(GS);
figure; imshow(GS_c);

I_mod = imimposemin(GS_c, ~bw1 | mask_em);
figure; imshow(I_mod);

L = watershed(I_mod);
figure; imshow(label2rgb(L));

% Watershed finally worked!
% This video helped: https://www.youtube.com/watch?v=Tf5buFFgnSU&ab_channel=AnselmGriffin


%%

bw2 = edge(bw1);
imshow(bw2);

% J = imtophat(bw1,strel('disk', 43));
J = imopen(bw1,strel('disk', 5));
imshow(J);



%% Image segmentation

% Create a structural element(s)
se1 = strel('disk', 10) ;

% Dilation
bw1 = imdilate(BW, se1);
imshow(bw1);


bw2 = bwmorph(bw1,'thin', 5);
imshow(bw2);

bw2 = bwskel(bw1);
imshow(bw2);

figure; imshow(labeloverlay(IG,bw2,'Transparency',0))

% approach: Detect branch and endpoint and get distances.
% From: https://www.mathworks.com/matlabcentral/answers/67123-how-to-find-length-of-branch-in-a-skeleton-image

It = bwmorph(bw1,'thin','inf');
imshow(It);
B =  bwmorph(It,'branchpoints');
[i,j] = find(bwmorph(It,'endpoints'));
D = bwdistgeodesic(It,find(B),'quasi');
imshow(BW6);
for n = 1:numel(i)
    text(j(n),i(n),[num2str(D(i(n),j(n)))],'color','g');
end



% Skeletonization
bw1 = bwmorph(bw1, 'skel', 10);
imshow(bw1);






%% Detecting features on gray images

% Apply mask to rgb spike (for later)
RGB_S = bsxfun(@times, RGB, cast(BW, 'like', RGB));
imshow(RGB_S);

% Work on gray image
ref_pts = detectSURFFeatures(GS);
[ref_features, ref_validPts] = extractFeatures(GS, ref_pts);

figure; imshow(RGB_S);
hold on; plot(ref_pts.selectStrongest(50));

IG_pts = detectSURFFeatures(IG);
[IG_features, IG_validPts] = extractFeatures(IG, IG_pts);
figure; imshow(RGB);
hold on; plot(IG_pts.selectStrongest(50));










%% Visualize branch length


BWM1 = imfill(BW, 'holes');
imshow(BWM1);

SE = strel('disk', 10); 
J = imclose(bw1,SE);
figure;imshow(J);

BW6 = bwskel(J);
figure;imshow(BW6);
It = bwmorph(BW6,'thin','inf');
imshow(It);
B =  bwmorph(It,'branchpoints');
[i,j] = find(bwmorph(It,'endpoints'));
D = bwdistgeodesic(It,find(B),'quasi');
imshow(BW6);
for n = 1:numel(i)
    text(j(n),i(n),[num2str(D(i(n),j(n)))],'color','g');
end

BWM2 = imclearborder(IG);
imshow(BWM2);
figure; imshow(IG);

imshow(bwskel(J));

imshow(bwmorph(BW,'fill',Inf));


%%
























%% Edge detection

% Close all windows; Clear the Workspace; Clear the Command Window
close all; clear; clc;
% Define directory
PhotoDir = 'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\IN';
% PhotoDir = pwd;
% Image path
mypic = fullfile(PhotoDir, '302_06.tif'); 
% Read image
RGB1 = imread(mypic);
% imshow(RGB1);
% Crop image
RGB2 = imcrop( RGB1, [0.5 10.5 5031 6933] );
% Threshold based on the red channel
BW1 = RGB2(:,:,1) > 40;
% imshow(mask1);
BW1 = bwareafilt(BW1, [10000, inf]);
% imshow(mask1);

% Redefine RGB
RGB3 = bsxfun(@times, RGB2, cast(BW1, 'like', RGB2));
% RGB to gray
Gray1 = rgb2gray(RGB3);
% imshow(IG);
% Convert gray image to double
Gray1 = im2double(Gray1);

% Label spikes with 8-connectivity
[Spk, num_spks] = bwlabel(BW1, 8);
% Get current file's name as string
% a = char(filenames(i));

% Tests image
BW = Spk == 1;
CS = bsxfun(@times, RGB3, cast(BW, 'like', RGB3));  % Colored spike
GS = rgb2gray(CS); % Gray Spike
% imshow(GS);


%% 
% Check this out: https://www.mathworks.com/help/images/examples.html?category=object-analysis&s_tid=CRUX_topnav
se1 = strel('disk', 10);

% Closing
bw1_pre = imclose(BW, se1);
% imshow(bw1_pre);

bw1 = edge(bw1_pre, 'Canny');
bw2 = edge(bw1_pre, 'sobel');
figure;
imshowpair(bw1,bw2,'montage')



%%
imshow(BW1);

% https://blogs.mathworks.com/steve/2010/07/30/visualizing-regionprops-ellipse-measurements/
s = regionprops(BW1, 'Orientation', 'MajorAxisLength', ...
    'MinorAxisLength', 'Eccentricity', 'Centroid');

imshow(BW1)
hold on

phi = linspace(0,2*pi,50);
cosphi = cos(phi);
sinphi = sin(phi);

for k = 1:length(s)
    xbar = s(k).Centroid(1);
    ybar = s(k).Centroid(2);

    a = s(k).MajorAxisLength/2;
    b = s(k).MinorAxisLength/2;

    theta = pi*s(k).Orientation/180;
    R = [ cos(theta)   sin(theta)
         -sin(theta)   cos(theta)];

    xy = [a*cosphi; b*sinphi];
    xy = R*xy;

    x = xy(1,:) + xbar;
    y = xy(2,:) + ybar;

    plot(x,y,'r','LineWidth',2);
end
hold off





LengthBr = bwmorph(BW,'skel',Inf);
endPts    = bwmorph(LengthBr, 'endpoints');
figure;imshow(LengthBr );
hold on; plot(endPts(:,2),endPts(:,1),'*');
hold off;
       
   

%% Length

BW = Spk == 14;
figure; imshow(BW);

bw1 = imdilate(BW, strel('disk', 30));
figure; imshow(bw1);
blurry = imgaussfilt(double(bw1),30);
figure; imshow(blurry);
bw2 = blurry > 0.5; % Rethreshold
figure; imshow(bw2);
bw3 = bwskel(bw1, 'MinBranchLength', 1);
figure; imshow(bw3);


SE = strel('disk', 30); 
J = imopen(bw1,SE);
figure;imshow(J);

BW6 = bwskel(J);
figure;imshow(BW6);



windowSize = 51;
kernel = ones(windowSize) / windowSize ^ 2;
blurryImage = conv2(single(bw1), kernel, 'same');
bw2 = blurryImage > 0.5; % Rethreshold
imshow(bw2);

bw1 = imopen(BW, strel('line', 300, 9));
% imshow(bw1);
bw2 = imdilate(bw1, strel('disk', 5));
% imshow(bw2);
bw3 = bwskel(bw2, 'MinBranchLength', 300);
% imshow(bw3);
% let's leave it like this

imshow(BW - imdilate(bw3, strel('disk', 5)));
overlay1 = imoverlay(BW, bw2, [.3 1 .3]);
imshow(overlay1);

% We could fit a polynomial thorugh branch and end points
%https://www.mathworks.com/matlabcentral/answers/552067-fit-a-polynimial-function-on-a-image-curve





bw1 = imdilate(BW, strel('disk', 20));
imshow(bw1);

bw2 = bwskel(bw1);

bw1 = imdilate(bw2, strel('square', 40));
imshow(bw1);
imshow(bwmorph(bw1,'thin',inf));


imshow(bwskel(imdilate(bw1,strel('square',30))));
imshow(bwmorph(imdilate(bw1,strel('line',50,0)),'thin',inf));

imshow(bwmorph(bw1, 'skel', 10));

bw2 = imopen(bw1, strel('line', 50, 90));
imshow(bw2);


BW1 = imdilate(BW, strel('disk', 10));
imshow(BW1);
BW2 = bwmorph(BW1,'thin',19);
imshow(BW2);
BW1 = imdilate(BW2, strel('disk', 10));
% imshow(BW1);
BW2 = bwmorph(BW1,'thin',19);
imshow(BW2);
BW1 = imdilate(BW2, strel('disk', 10));
% imshow(BW1);
BW2 = bwmorph(BW1,'thin',19);
imshow(BW2);
BW1 = imdilate(BW2, strel('disk', 10));
% imshow(BW1);
BW2 = bwmorph(BW1,'thin',19);
imshow(BW2);
BW1 = imdilate(BW2, strel('disk', 10));
% imshow(BW1);
BW2 = bwmorph(BW1,'thin',19);
imshow(BW2);
skel1 = bwskel(BW2, 'MinBranchLength', 10);
overlay1 = imoverlay(BW, skel1, [.3 1 .3]);
imshow(overlay1);


BWC = imclose(BW,strel('disk', 30));
imshow(BWC);

% Erosion
bw1 = imerode(BWC, strel('square', 20));
imshow(bw1);
% External boundary
bw1 = imdilate(bw1, strel('disk', 5)) & ~BW;
imshow(bw1);

BW2 = bwskel(BW, 'MinBranchLength', 10);
BW2 = imfill(BW2,'holes');
BW2 = bwskel(BW2, 'MinBranchLength', 100);
BW2 = imfill(BW2,'holes');
% BW2 = bwmorph(BWC,'thin', 50);
% imshow(BW2);

overlay1 = imoverlay(BW, BW2, [.3 1 .3]);
imshow(overlay1);





GF1 = imgaussfilt(double(BW),30);
GF1 = adapthisteq(GF1);
bw1 = imerode(GF1, strel('disk', 15));
imshow(bw1);


BW1 = im2bw(GF1(:), .);
imshow(uint8(GF1)>  uint8(0.4));

bw1 = imopen(BW, strel('line', 150, 0));

GF1 = imgaussfilt(double(bw1),30);
GF1 = adapthisteq(GF1);
imshow(GF1);


bw1 = imerode(BW, strel('rectangle', [60 20]));
bw1 = imopen(BW, strel('square', 20));
% imshow(bw1);
bw1 = bwareafilt(bw1, [1000, inf]);
% imshow(bw1);
bw1 = imfill(bw1, 'holes');
imshow(bw1);
bw1 = bwmorph(bw1, 'thin');
imshow(bw1);
bw2 = imerode(bw1, strel('line', 40, 135));
imshow(bw2);

GF1 = imgaussfilt(double(bw1),30);
GF1 = adapthisteq(GF1);
bw1 = imerode(GF1, strel('disk', 15));
imshow(bw1);


bw1 = imerode(bw1, strel('square', 20));
BWM1 = imfill(BW, 'holes');



imshow(bwdist(imgaussfilt(double(bw1),5),'cityblock'));
imshow(bwmorph(bw1, 'bridge', 100));
imshow(imclose(bw1, strel('disk', 10)));


bw1 = imopen(bw1, strel('square', 20));
imshow(bw1);

imshow(imopen(bw1, strel('rectangle', [40 20])));
imshow(bwmorph(bw1, 'thicken', 5));
imshow(imdilate(bw1, strel('line',5, 135)));
bw1 = imclose(bw1, strel('square', 20));
imshow(bw1);

imshow(BW - bw1);

% Dilation
bw1 = imdilate(BW, strel('disk', 10));
imshow(bw1);

% BWR = imrotate(BW,-90,'bilinear','crop');
BW2 = bwmorph(bw1,'thin', 50);
imshow(BW2);  

IM2=bwmorph(IM,'thin',50) 
       
       