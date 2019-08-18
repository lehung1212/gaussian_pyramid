function [pyr] = gausspyr(img, numlvl)
% gausspyr create the gaussian pyramid with numlvl levels from img
% preconditions: the values of img are doubles from 0 to 1. Numlvl is a
% postive integer. postconditions: pyr is a cell containing a series of
% downsamples of img. the first entry in pyr is the original image. Before
% each downsample, we smooth the image with a Gaussian filter with a
% standard deviation of 2.5.

% create gaussian kernel
gauss1D = gkern(2.5^2); % Gaussian kernel with sigma=2.5

% set up the pyramid
pyr{numlvl} = 0; % pre-allocate
pyr{1} = img; % first image is original

for i =  2:numlvl % for all the scale levels
    % blur img with gaussian convolution
    img = conv2(gauss1D, gauss1D, img, 'same');
    
    % downsample
    img = img(1:2:end, 1:2:end);
    
    %add to pyramid
    pyr{i} = img;
end

end
   


