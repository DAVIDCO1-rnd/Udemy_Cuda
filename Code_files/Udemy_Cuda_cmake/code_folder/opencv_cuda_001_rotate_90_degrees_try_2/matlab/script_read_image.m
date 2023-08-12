clear;
close all;
clc;

% image_path = '../images/00013.jpg';
% rgb_img = imread(image_path);
% grayscale_image = rgb2gray(rgb_img);
% grayscale_image(1,1) = 10;
% grayscale_image(1,2) = 20;
% grayscale_image(1,3) = 30;
% grayscale_image(2,1) = 12;
% grayscale_image(3,1) = 13;
% grayscale_image(4,1) = 14;

grayscale_image = [ 11, 21, 31, 41; 
                    52, 62, 72, 82;
                    93, 103, 113, 123];
grayscale_image_uint8 = uint8(grayscale_image);
                


grayscale_path = '../images/grayscale.png';

% imshow(grayscale_image_uint8);
imwrite(grayscale_image_uint8, grayscale_path, 'bitdepth', 8);
grayscale_image1 = imread(grayscale_path);

grayscale_image
grayscale_image1