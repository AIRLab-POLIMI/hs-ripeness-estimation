%% Procedure:
% 1) get calibration parameters from calibration images (through
% checkerboard) for each wavelength
% 2) get mean calibration parameters for each sensor (considering
% corresponding wavelengths' parameters)
% 3) correct the wavelenghts according to the sensor they are associated
% with


wavelength_range = [1:53]; 
% Sensor 2: from band 2 to band 21
% Sensor 1: from band 22 to band 53 (band 22 is 646.9000 nm)
FIRST_NIR_BAND = 22;
LAST_NIR_BAND = 53;
FIRST_VIS_BAND = 2;
LAST_VIS_BAND = 21;
selected_range = [FIRST_VIS_BAND:LAST_NIR_BAND]; % band 1 cannot be used


% Generate world coordinates of the corners of the squares
squareSize = 108; % millimeters
% Get coordinates of the (internal) corners of the squares of size squareSize of a
% chekerboard of height 7 and width 9
worldPoints = generateCheckerboardPoints([7,9], squareSize);
imageSize = [1024,1024];
% 1xbands struct containing the geometric calibration parameters for each wavelength, that is:
% Focal length (1x2)
% Principal point (1x2)
% Radial distortion coefficients (1x2)
% Tangetial distortion coefficients (1x2)
% Mean projection error (standard error of estimate parameters
geom_calib_params = struct('FocalLength', cell(1, max(wavelength_range)), ...
    'PrincipalPoint', cell(1, max(wavelength_range)), 'RadialDistortion', ...
    cell(1, max(wavelength_range)), 'MeanReprojectionError', ...
    cell(1, max(wavelength_range)), 'TangentialDistortion', cell(1, max(wavelength_range)));
% For every wavelength
for i = wavelength_range
    if ismember(i, selected_range)
        % Create a set of calibration images
        images = imageDatastore(['Senop geometric calibration/band_', num2str(i)]);
        imageFileNames = images.Files;
        
        % Detect calibration pattern (return points and dimensions of
        % detected checkerboard; set 'PartialDetections' to false to 
        % discard partially detected checkerboards)
        [imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames, 'PartialDetections', false);
        
        % Calibrate the camera calibration parameters and the standard 
        % estimation errors for the single camera calibrations using the 
        % generated and detected checkerboard points 
        [params, ~, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
            'ImageSize', imageSize, 'NumRadialDistortionCoefficients', 2, 'EstimateTangentialDistortion', true);
        
        % Assign the detected parameters
        geom_calib_params(i).FocalLength = params.FocalLength;
        geom_calib_params(i).PrincipalPoint = params.PrincipalPoint;
        geom_calib_params(i).RadialDistortion = params.RadialDistortion;
        geom_calib_params(i).MeanReprojectionError = params.MeanReprojectionError;
        geom_calib_params(i).TangentialDistortion = params.TangentialDistortion;
    end
end

%save('Senop geometric calibration/geom_calib_params.mat', 'geom_calib_params')
load('Senop geometric calibration/geom_calib_params.mat','geom_calib_params') % !!!!!!!!! LOAD SAVED PARAMETERS !!!!!!!!!!

% Mean reprojection error plot
figure
hold on;
plot([FIRST_NIR_BAND:LAST_NIR_BAND],[geom_calib_params(FIRST_NIR_BAND:LAST_NIR_BAND).MeanReprojectionError], 'marker','o', 'MarkerFaceColor', 'r')
plot([FIRST_VIS_BAND:LAST_VIS_BAND],[geom_calib_params(FIRST_VIS_BAND:LAST_VIS_BAND).MeanReprojectionError], 'marker','o', 'MarkerFaceColor', 'g')
title('Mean reprojection error')
xlabel('Band #')
ylabel('Mean reprojection error (px)')
legend('Sensor 1', 'Sensor 2', 'Location', 'southeast')
grid on
hold off;

% Retreieve focal length for each wavelength in mm
focalLengthmm = zeros(max(wavelength_range),1);
for i = wavelength_range
    if ismember(i, selected_range)
        focalLengthmm(i) = mean(geom_calib_params(i).FocalLength)*0.0055; % the sensor pixel edge is 0.0055 mm
    end
end

% there is a clear distinction bewtween the focal of the two sensors
% Sensor 2: from band 2 to band 21
% Sensor 1: from band 22 to band 53 (band 22 is 646.9000 nm)
figure
hold on;
plot([FIRST_NIR_BAND:LAST_NIR_BAND],focalLengthmm(FIRST_NIR_BAND:LAST_NIR_BAND), 'linestyle','none','marker','o', 'MarkerFaceColor', 'r')
plot([FIRST_VIS_BAND:LAST_VIS_BAND],focalLengthmm(FIRST_VIS_BAND:LAST_VIS_BAND), 'linestyle','none','marker','o', 'MarkerFaceColor', 'g')
title('Focal length vs Bands')
xlabel('Bands')
ylabel('Focal Length (mm)')
grid on
legend('Sensor 1', 'Sensor 2', 'Location', 'southeast')
hold off;

% Retrieve principal point for each wavelength in mm
principalPointmm = zeros(max(wavelength_range),2);
for i = wavelength_range
    if ismember(i, selected_range)
        principalPointmm(i,:) = (geom_calib_params(i).PrincipalPoint)*0.0055; % the sensor pixel edge is 0.0055 mm
    end 
end

figure
hold on;
plot(principalPointmm(FIRST_NIR_BAND:LAST_NIR_BAND,1), principalPointmm(FIRST_NIR_BAND:LAST_NIR_BAND,2), 'linestyle','none','marker','o', 'MarkerFaceColor', 'r')
plot(principalPointmm(FIRST_VIS_BAND:LAST_VIS_BAND,1), principalPointmm(FIRST_VIS_BAND:LAST_VIS_BAND,2), 'linestyle','none','marker','o', 'MarkerFaceColor', 'g')
title('Principal Point coordinates')
xlabel('c_x (mm)')
ylabel('c_y (mm)')
grid on
legend('Sensor 1', 'Sensor 2', 'Location', 'southeast')
hold off;
% bands 20 and 53 not well separated

% Retrieve focal length for each wavelength
focalLength = zeros(max(wavelength_range),2);
for i = wavelength_range
    if ismember(i, selected_range)
        focalLength(i,:) = geom_calib_params(i).FocalLength;
    end
end

% Get mean focal length of each sensor
focalLength_sensor_1 = mean(focalLength(FIRST_NIR_BAND:LAST_NIR_BAND,:));
focalLength_sensor_2 = mean(focalLength(FIRST_VIS_BAND:LAST_VIS_BAND,:));

% Retrieve principal point for each wavelength
principalPoint = zeros(max(wavelength_range),2);
for i = wavelength_range
    if ismember(i, selected_range)
        principalPoint(i,:) = geom_calib_params(i).PrincipalPoint;
    end
end

% Get mean principal point of each sensor
principalPoint_sensor_1 = mean(principalPoint(FIRST_NIR_BAND:LAST_NIR_BAND,:));
principalPoint_sensor_2 = mean(principalPoint(FIRST_VIS_BAND:LAST_VIS_BAND,:));

% Get radial distortion for each wavelength
radialDistortion = zeros(max(wavelength_range),2);
for i = wavelength_range
    if ismember(i, selected_range)
        radialDistortion(i,:) = geom_calib_params(i).RadialDistortion;
    end
end

% Get mean radial distortion for each sensor
radialDistortion_sensor_1 = mean(radialDistortion(FIRST_NIR_BAND:LAST_NIR_BAND,:));
radialDistortion_sensor_2 = mean(radialDistortion(FIRST_VIS_BAND:LAST_VIS_BAND,:));

% Get tangential distortion for each wavelength
tangentialDistortion = zeros(max(wavelength_range),2);
for i = wavelength_range
    if ismember(i, selected_range)
        tangentialDistortion(i,:) = geom_calib_params(i).TangentialDistortion;
    end
end

% Get mean tangential distortion for each sensor
tangentialDistortion_sensor_1 = mean(tangentialDistortion(FIRST_NIR_BAND:LAST_NIR_BAND,:));
tangentialDistortion_sensor_2 = mean(tangentialDistortion(FIRST_VIS_BAND:LAST_VIS_BAND,:));

% choose a sample image to undistort
I = imread('Senop geometric calibration/band_2/experiment_230315_040435.tif');
imageSize = [size(I, 1), size(I, 2)];

% Returns a camera intrinsics object that contains the focal length, the 
% camera's principal point, radial and tangential distortion that have been
% calculated
intrinsics_sensor_1 = cameraIntrinsics(focalLength_sensor_1, principalPoint_sensor_1, imageSize, 'RadialDistortion', radialDistortion_sensor_1, 'TangentialDistortion', tangentialDistortion_sensor_1); 
intrinsics_sensor_2 = cameraIntrinsics(focalLength_sensor_2, principalPoint_sensor_2, imageSize, 'RadialDistortion', radialDistortion_sensor_2, 'TangentialDistortion', tangentialDistortion_sensor_2); 

%save('Senop geometric calibration/intrinsics_sensor_1.mat', 'intrinsics_sensor_1')
%save('Senop geometric calibration/intrinsics_sensor_2.mat', 'intrinsics_sensor_2')
load('Senop geometric calibration/intrinsics_sensor_1.mat', 'intrinsics_sensor_1') % !!!!!!!!! LOAD SAVED PARAMETERS !!!!!!!!!!
load('Senop geometric calibration/intrinsics_sensor_2.mat', 'intrinsics_sensor_2') % !!!!!!!!! LOAD SAVED PARAMETERS !!!!!!!!!!

% undistort the sample image
J = undistortImage(I,intrinsics_sensor_1);

figure; imshowpair(I,J,'montage');
title('Original Image (left) vs. Corrected Image (right)');

% choose an image to show the effectiveness of the calibration
experiment_number = '230315_041416';
band_num = 2;
% draw the red line to select a proper row
I = imread(['Senop geometric calibration/band_', num2str(band_num),'/experiment_', experiment_number,'.tif']);

% line plot on the image
figure,imshow(I)
% make sure the image doesn't disappear if we plot something else
hold on
% define points (in matrix coordinates)
row = 665;
p1 = [row,10];
p2 = [row,850];
% plot the points.
% Note that depending on the definition of the points,
% you may have to swap x and y
plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','r','LineWidth',1)
hold off;

% plot the spectrum of the selected row of pixels (distorted image) for the selected
% bands of Sensor 1 and Sensor 2
figure();
hold on
h = zeros(20,1);
for i = 12:31
    I = imread(['Senop geometric calibration/band_', num2str(i),'/experiment_', experiment_number,'.tif']);
    plot(I(row,10:850),'Color',rand(1,3),'LineWidth',1);
end
title(['Distorted image - row ', num2str(row)])
ylabel('Digital Number')
xlabel('Column #')
grid on
legend('Band 12','Band 13','Band 14','Band 15','Band 16','Band 17', ...
    'Band 18','Band 19','Band 20','Band 21','Band 22','Band 23', ...
    'Band 24','Band 25','Band 26','Band 27','Band 28','Band 29',...
    'Band 30','Band 31')
hold off;

% plot the spectrum of the selected row of pixels (undistorted image) for the selected
% bands of Sensor 1 and Sensor 2
figure();
hold on
for i = 12:31
    I = imread(['Senop geometric calibration/band_', num2str(i),'/experiment_', experiment_number,'.tif']);
    if i > 21
        J = undistortImage(I,intrinsics_sensor_1);
    else
        J = undistortImage(I,intrinsics_sensor_2);
    end
    %imshowpair(I,J,'montage');
    plot(J(row,10:850),'Color',rand(1,3),'LineWidth',1);
end
title(['Undistorted image - row ', num2str(row)])
ylabel('Digital Number')
xlabel('Column #')
grid on
legend('Band 12','Band 13','Band 14','Band 15','Band 16','Band 17', ...
    'Band 18','Band 19','Band 20','Band 21','Band 22','Band 23', ...
    'Band 24','Band 25','Band 26','Band 27','Band 28','Band 29',...
    'Band 30','Band 31')
hold off;

I2 = imread(['Senop geometric calibration/band_', num2str(21),'/experiment_', experiment_number,'.tif']);
I1 = imread(['Senop geometric calibration/band_', num2str(22),'/experiment_', experiment_number,'.tif']);
J2 = undistortImage(I2,intrinsics_sensor_2);
J1 = undistortImage(I1,intrinsics_sensor_1);
        
imshow(J2)
imshow(J1)