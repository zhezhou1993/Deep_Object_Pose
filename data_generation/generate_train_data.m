source_folder = '/media/logan/data/data/LF_dataset_material_recong/image/';
patch_folder = '/media/logan/data/data/LF_dataset_material_recong/patch_image/';
train_patch_folder = strcat(patch_folder,'train/'); 
test_patch_folder = strcat(patch_folder,'test/');

folders = dir(source_folder);
folders(1:2) = [];
sub_size = [376, 541];
image_size = 224;
avaiable_size = sub_size - image_size;
every_image_patch_num = 1;
train_percent = 0.8;
csv_train_info = {};
csv_test_info = {};
sample_num = 10;
% random_row = randi([0 avaiable_size(1)], 1, 1200 * every_image_patch_num);
% random_col = randi([0 avaiable_size(2)], 1, 1200 * every_image_patch_num);
for i = 1 : size(folders,1)
    cu_folder = strcat(source_folder,folders(i).name);
    cu_train_folder = strcat(train_patch_folder, folders(i).name);
    cu_test_folder = strcat(test_patch_folder, folders(i).name);
    delete(cu_train_folder);
    delete(cu_test_folder);
    mkdir(cu_train_folder);
    mkdir(cu_test_folder);
    
    [image_path, ~,~,~] = LFFindFilesRecursive(cu_folder,{'*.png'});
%     train_test_id = randperm(size(image_path,1));
    
    train_test_id = randperm(sample_num);
    for j = 1 : sample_num
        if(j < round(train_percent * sample_num))
            temp_image = imread(strcat(cu_folder,'/',image_path{train_test_id(j)}));
            for k = 1 : every_image_patch_num
                random_row = randi([0 avaiable_size(1)]);
                random_col = randi([0 avaiable_size(2)]);
                cropped_patch = temp_image(random_row*7 + 1 : (random_row+image_size)* 7, ...
                                           random_col*7 + 1: (random_col+image_size)* 7, :);
%                 img_patch_name = strcat(cu_train_folder,'/',string(image_path{train_test_id(j)}(1:end-4)),...
%                                         '_',sprintf("%04d",k),'.png');     
                img_patch_name = sprintf('%s/%s_%04d.png',cu_train_folder,image_path{train_test_id(j)}(1:end-4),k);
                imwrite(cropped_patch, img_patch_name);
                csv_train_info(end+1,:) = {img_patch_name,i};
            end
        else
            temp_image = imread(strcat(cu_folder,'/',image_path{train_test_id(j)}));
            for k = 1 : every_image_patch_num
                random_row = randi([0 avaiable_size(1)]);
                random_col = randi([0 avaiable_size(2)]);
                cropped_patch = temp_image(random_row*7 + 1 : (random_row+image_size)* 7, ...
                                           random_col*7 + 1: (random_col+image_size)* 7, :);
                img_patch_name = sprintf('%s/%s_%04d.png',cu_test_folder,image_path{train_test_id(j)}(1:end-4),k);                     
                imwrite(cropped_patch, img_patch_name);
                csv_test_info(end+1,:) = {img_patch_name,i};
            end
            
        end
    
    end
end
% Convert cell to a table and use first row as variable names
train_T = cell2table(csv_train_info);
test_T = cell2table(csv_test_info); 
% Write the table to a CSV file
writetable(train_T,sprintf("%s%s",patch_folder,'train.csv'));
writetable(test_T,sprintf("%s%s",patch_folder,'test.csv'));