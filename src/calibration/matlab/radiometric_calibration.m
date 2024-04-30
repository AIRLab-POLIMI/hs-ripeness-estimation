%% Procedure:
% 1) considering a calibration image (e.g. calibration panel), compute mean
% digital number in a ROI for each wavelength
% 2) save the theoretical value of the reflectance of the calibration image
% (e.g.0.95)
% 3) compute linear regression coefficients of the mean digital numbers
% against the theoretical reflectance for each wavelength
% 4) correct image (<-> compute reflectance) by applying the computed 
% regression coefficients specific for each wavelength


clear all; 

sequence_path = 'Senop calibration/Vineyard Cattolica script_220901_204900/Vineyard Cattolica sequence/Vineyard Cattolica sequence_000005/';

% load a calibration hypercube
hcube = hypercube([sequence_path,'Vineyard Cattolica sequence_000005.dat'],[sequence_path,'Vineyard Cattolica sequence_000005.hdr']);

% false color image of the calibration panel
falsecolorImg = colorize(hcube);
figure
imagesc(falsecolorImg)
title('False color Image of Data Cube')

% size of the hypercube (sz3 = num_wavelengths)
num_wavelengths = length(hcube.Wavelength);
sz1 = size(hcube.DataCube,1);
sz2 = size(hcube.DataCube,2);
sz3 = size(hcube.DataCube,3);

% plot of the calibration panel spectrum before calibration (mean digital
% number for each band in selected the ROI)
before_calibration_DNs = zeros(2, num_wavelengths);
% for each wavelength
for i = 1:num_wavelengths    
    ROI_i = hcube.DataCube(312:712,312:712,i); % select a ROI of 400x400 px in the center of the image (1024x1024 px)
    before_calibration_DNs(1,i) = hcube.Wavelength(i);
    before_calibration_DNs(2,i) = mean(ROI_i, 'all'); % mean of all Digital Numbers in the ROI
end

figure(100)
plot(before_calibration_DNs(1,:), before_calibration_DNs(2,:), 'b -');
xlabel('Wavelength (nm)');
ylabel('Digital Number (-)');
title('Before calibration');

saveas(figure(100),[sequence_path,'Before_calibration_DNs.png'])
close

% find the band with highest pixel values
[argvalue, argmax] = max(before_calibration_DNs(2,:));
% extract the 6th (armax) band (532.7 nm)
band_6 = hcube.DataCube(:,:,6); % 1024-by-1024 array
% flatten the 2D array
flatten_6 = reshape(band_6',[1 size(band_6,1)*size(band_6,2)]);
figure, hold on
histogram(flatten_6,0:65536)
xlabel('Pixel value');
ylabel('Count');
title('Histogram of the calibration panel - band 532.7 nm');
hold off

% SphereOptics calibration panel with 95% reflectivity (theoretical value)
FieldSpectra = 0.95;

DataSpectra = zeros(num_wavelengths,2);
% Mean digital number in the ROI for each wavelength 
for i = 1:num_wavelengths
    DataSpectra(i,1) = hcube.Wavelength(i);
    
    ROI_i = hcube.DataCube(312:712,312:712,i); % for each band I select a ROI of 400x400 px in the center of the image (1024x1024 px)
    mean_ROI_i = mean(ROI_i, 'all'); % mean of all Digital Numbers in the ROI
    DataSpectra(i,2) = mean_ROI_i; 
end

% Table for the results
% Columns: wavelength, b0, b1 e R2
Results = zeros(num_wavelengths,4);
Results(:,1) = DataSpectra(:,1);

% for each wavelength
for k = 1:num_wavelengths 
    x = DataSpectra(k,2); % mean DN of the ROI for band k
    y = FieldSpectra;     % a priori known target 
    % plot of the point we use for the regression
    % figure(k)
    % plot(x,y,'r o')
    % xlabel('Digital Number (-)')
    % ylabel('Riflettanza 95% da pannello SphereOptics (-)');
    % txt = [' Calibrazione Radiometrica, ' num2str(DataSpectra(k,1)) ' nm '];
    % title(txt);
    
    % Regression line
    p = polyfit(x,y,1);
    x1 = [0:1000:30000];        % x grid for plot
    y1 = p(1,1)*x1 + p(1,2);    % corresponding regressed y
    % save the regression coefficients
    Results(k,2) = p(1,2);  
    Results(k,3) = p(1,1);
    
%     % R2 evaluation
%     ymean = mean(FieldSpectra); % 0.95
%     y_hat = p(1,1)*DataSpectra(k,2) + p(1,2);
%     s = 0;
%     t = 0;
%     for j = 1:4
%         n = (FieldSpectra(k,j+1)-ymean)^2;
%         s = s + n;
%         m = (y_hat(1,j)-ymean)^2;
%         t = t + m;
%     end
%     MSt = (1/4)*s;
%     MSm = (1/4)*t;
%     R2 = MSm/MSt;
%     Results(k,4) = R2;
    
    % Regression line plot
%     hold on
%     plot(x1,y1, 'b --')
%     lgd = legend('Punti da pannello', 'Linea di tendenza'); %, 'c');
%     c = lgd.Location;
%     lgd.Location = 'southeast';
%     txt1 = ['y = ' num2str(p(1,1)) 'x ' num2str(p(1,2))];
%     text(16, 0.25, txt1);
% %     txt2 = ['R^2 = ' num2str(R2)];
% %     text(16, 0.1, txt2);
%     hold off 
%     saveas(figure(k),[sequence_path, 'Banda',num2str(k),'.png'])
%     close
end


% calibrate image with computed coefficients
calibrated_image = zeros(sz1, sz2, sz3);
for i = 1:sz1
    for j = 1:sz2
        for k = 1:sz3
            calibrated_image(i,j,k) = double(hcube.DataCube(i,j,k))*Results(k,3);
        end
    end
end

% mean values in ROI after calibration for each wavelength
after_calibration_DNs = zeros(2, num_wavelengths);
for i = 1:num_wavelengths    
    ROI_i = calibrated_image(312:712,312:712,i); % for each band I select a ROI of 400x400 px in the center of the image (1024x1024 px)
    after_calibration_DNs(1,i) = hcube.Wavelength(i);
    after_calibration_DNs(2,i) = mean(ROI_i, 'all'); % mean of all Digital Numbers in the ROI
end

% plot of the calibrated mean reflectance in the ROI for each wavelength
% (0.95 everywhere)
figure(101)
plot(after_calibration_DNs(1,:), after_calibration_DNs(2,:), 'b -')
axis([min(after_calibration_DNs(1,:)) max(after_calibration_DNs(1,:)) 0 1])
xlabel('Wavelength (nm)');
ylabel('Digital Number (-)');
title('After calibration');

saveas(figure(101),[sequence_path,'After_calibration_DNs.png'])
close

save([sequence_path,'Results.mat'],'Results');