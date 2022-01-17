clc
clear all


easy = 1 % easy = 1 means that the test runs are not time-consuming metrics, easy = 0 means that the test is time-consuming metrics
row_name1 = 'row1';
row_data1 = 'row2';

names = {'Ours'};
method_name = cellstr(names(1));
row = 'A';
row_name = strrep(row_name1, 'row', row);
row_data = strrep(row_data1, 'row', row);
fileFolder=fullfile('D:\Github\SeAFusion\test_imgs\ir'); % Folder where infrared images or visible images are located
dirOutput=dir(fullfile(fileFolder,'*.png')); % the suffix name of the source and fused images
fileNames = {dirOutput.name};
[m, num] = size(fileNames);
ir_dir = 'D:\Github\SeAFusion\test_imgs\ir'; % Folder where infrared images are located
vi_dir = 'D:\Github\SeAFusion\test_imgs\vi'; % Folder where visible images are located
Fused_dir = 'D:\Github\SeAFusion\Fusion_results'; % Folder where fused images are located
EN_set = [];    SF_set = [];SD_set = [];PSNR_set = [];
MSE_set = [];MI_set = [];VIF_set = []; AG_set = [];
CC_set = [];SCD_set = []; Qabf_set = [];
SSIM_set = []; MS_SSIM_set = [];
Nabf_set = [];FMI_pixel_set = [];
FMI_dct_set = []; FMI_w_set = [];
for j = 1:num
    fileName_source_ir = fullfile(ir_dir, fileNames{j});
    fileName_source_vi = fullfile(vi_dir, fileNames{j}); 
    fileName_Fusion = fullfile(Fused_dir, fileNames{j});
    ir_image = imread(fileName_source_ir);
    vi_image = imread(fileName_source_vi);
    fused_image   = imread(fileName_Fusion);
    if size(ir_image, 3)>2
        ir_image = rgb2gray(ir_image);
    end

    if size(vi_image, 3)>2
        vi_image = rgb2gray(vi_image);
    end

    if size(fused_image, 3)>2
        fused_image = rgb2gray(fused_image);
    end
    [m, n] = size(fused_image);
%     fused_image = fused_image(7:m-6, 7:n-6);
    ir_size = size(ir_image);
    vi_size = size(vi_image);
    fusion_size = size(fused_image);
    if length(ir_size) < 3 & length(vi_size) < 3
        [EN, SF,SD,PSNR,MSE, MI, VIF, AG, CC, SCD, Qabf, Nabf, SSIM, MS_SSIM, FMI_pixel, FMI_dct, FMI_w] = analysis_Reference(fused_image,ir_image,vi_image, easy);
        EN_set = [EN_set, EN];SF_set = [SF_set,SF];SD_set = [SD_set, SD];PSNR_set = [PSNR_set, PSNR];
        MSE_set = [MSE_set, MSE];MI_set = [MI_set, MI]; VIF_set = [VIF_set, VIF];
        AG_set = [AG_set, AG]; CC_set = [CC_set, CC];SCD_set = [SCD_set, SCD];
        Qabf_set = [Qabf_set, Qabf]; Nabf_set = [Nabf_set, Nabf];
        SSIM_set = [SSIM_set, SSIM]; MS_SSIM_set = [MS_SSIM_set, MS_SSIM];
        FMI_pixel_set = [FMI_pixel_set, FMI_pixel]; FMI_dct_set = [FMI_dct_set,FMI_dct];
        FMI_w_set = [FMI_w_set, FMI_w];
    else
        disp('unsucessful!')
        disp( fileName_Fusion)
    end
    fprintf('Fusion Method:%s, Image Name: %s\n', cell2mat(method_name), fileNames{j})
end
file_name = './metric.xls' % File name for writing metrics to excel
if easy ==1
    xlswrite(file_name, method_name,'EN',row_name)
    xlswrite(file_name, method_name,'SF',row_name)
    xlswrite(file_name, method_name,'SD',row_name) 
    xlswrite(file_name, method_name,'PSNR',row_name)
    xlswrite(file_name, method_name,'MSE',row_name)
    xlswrite(file_name, method_name,'MI',row_name)
    xlswrite(file_name, method_name,'VIF',row_name)
    xlswrite(file_name, method_name,'AG',row_name)
    xlswrite(file_name, method_name,'CC',row_name)
    xlswrite(file_name, method_name,'SCD',row_name)
    xlswrite(file_name, method_name,'Qabf',row_name)
    xlswrite(file_name,SF_set','SF',row_data)

    xlswrite(file_name,SD_set','SD',row_data) 
    xlswrite(file_name,PSNR_set','PSNR',row_data)
    xlswrite(file_name,MSE_set','MSE',row_data)
    xlswrite(file_name,MI_set','MI',row_data)
    xlswrite(file_name,VIF_set','VIF',row_data)
    xlswrite(file_name,AG_set','AG',row_data)
    xlswrite(file_name,CC_set','CC',row_data)
    xlswrite(file_name,EN_set','EN',row_data)
    xlswrite(file_name,SCD_set','SCD',row_data)
    xlswrite(file_name,Qabf_set','Qabf',row_data)
else        
    xlswrite(file_name, method_name,'Nabf',row_name)
    xlswrite(file_name, method_name,'SSIM',row_name)
    xlswrite(file_name, method_name,'MS_SSIM',row_name)
    xlswrite(file_name, method_name,'FMI_pixel',row_name)
    xlswrite(file_name, method_name,'FMI_dct',row_name)
    xlswrite(file_name, method_name,'FMI_w',row_name)

    xlswrite(file_name,Nabf_set','Nabf',row_data)
    xlswrite(file_name,SSIM_set','SSIM',row_data)
    xlswrite(file_name,MS_SSIM_set','MS_SSIM',row_data)
    xlswrite(file_name,FMI_pixel_set','FMI_pixel',row_data)
    xlswrite(file_name,FMI_dct_set','FMI_dct',row_data)
    xlswrite(file_name,FMI_w_set','FMI_w',row_data)
end
