source_folder = '/media/logan/data/data/LF_dataset_material_recong/image/';
patch_folder = '/media/logan/data/data/LF_dataset_material_recong/patch_image/';
train_patch_folder = strcat(patch_folder,'train/'); 
test_patch_folder = strcat(patch_folder,'test/');

folders = dir(source_folder);
folders(1:2) = [];
sub_size = [376, 541];
image_size = 224;
avaiable_size = sub_size - image_size;
every_image_patch_num = 2;
train_percent = 0.8;
csv_train_info = {};
csv_test_info = {};
sample_num = 10;
angular_size = 7;
% random_row = randi([0 avaiable_size(1)], 1, 1200 * every_image_patch_num);
% random_col = randi([0 avaiable_size(2)], 1, 1200 * every_image_patch_num);
for i = 1 : size(folders,1)
    cu_folder = strcat(source_folder,folders(i).name);
    cu_train_folder = strcat(train_patch_folder, folders(i).name);
    cu_test_folder = strcat(test_patch_folder, folders(i).name);
    if exist(cu_train_folder) == 7
        status = rmdir(cu_train_folder,'s');
        if (~status)
            rmdir(cu_train_folder);
        end
    end
    if exist(cu_train_folder) == 7
        status = rmdir(cu_test_folder,'s');
        if (~status)
            rmdir(cu_train_folder);
        end
    end
    mkdir(cu_train_folder);
    mkdir(cu_test_folder);
    
    [image_path, ~,~,~] = LFFindFilesRecursive(cu_folder,{'*.png'});
%     train_test_id = randperm(size(image_path,1));
    
    train_test_id = randperm(sample_num);
    for j = 1 : sample_num
        if(j <= round(train_percent * sample_num))
            json_msg.image_path = strcat(cu_folder,'/',image_path{train_test_id(j)});
            json_msg.image_size = image_size;
            json_msg.angular_res = angular_size;
            random_rowmin = randi([0 avaiable_size(1)], [every_image_patch_num,1]);
            random_colmin = randi([0 avaiable_size(2)], [every_image_patch_num,1]);
            json_msg.crop_data = [random_rowmin,random_colmin];
            json = jsonencode(json_msg);
            json_name = sprintf('%s/%s.json',cu_train_folder,image_path{train_test_id(j)}(1:end-4));
            fid = fopen(json_name, 'w');
            if fid == -1, error('Cannot create JSON file'); end
            fwrite(fid, json, 'char');
            fclose(fid);
%             temp_image = imread(strcat(cu_folder,'/',image_path{train_test_id(j)}));
%             for k = 1 : every_image_patch_num
%                 random_row = randi([0 avaiable_size(1)]);
%                 random_col = randi([0 avaiable_size(2)]);
%                 cropped_patch = temp_image(random_row*7 + 1 : (random_row+image_size)* 7, ...
%                                            random_col*7 + 1: (random_col+image_size)* 7, :);
% %                 img_patch_name = strcat(cu_train_folder,'/',string(image_path{train_test_id(j)}(1:end-4)),...
% %                                         '_',sprintf("%04d",k),'.png');     
%                 img_patch_name = sprintf('%s/%s_%04d.png',cu_train_folder,image_path{train_test_id(j)}(1:end-4),k);
%                 imwrite(cropped_patch, img_patch_name);
            for k = 1 : every_image_patch_num
                csv_train_info(end+1,:) = {json_name,k,i};
            end
%             end
        else
            json_msg.image_path = strcat(cu_folder,'/',image_path{train_test_id(j)});
            json_msg.image_size = image_size;
            random_rowmin = randi([0 avaiable_size(1)], [1 every_image_patch_num]);
            random_colmin = randi([0 avaiable_size(2)], [1 every_image_patch_num]);
            json_msg.crop_data = [random_rowmin;random_colmin];
            json = jsonencode(json_msg);
            json_name = sprintf('%s/%s.json',cu_test_folder,image_path{train_test_id(j)}(1:end-4));
            fid = fopen(json_name, 'w');
            if fid == -1, error('Cannot create JSON file'); end
            fwrite(fid, json, 'char');
            fclose(fid);
%             temp_image = imread(strcat(cu_folder,'/',image_path{train_test_id(j)}));
%             for k = 1 : every_image_patch_num
%                 random_row = randi([0 avaiable_size(1)]);
%                 random_col = randi([0 avaiable_size(2)]);
%                 cropped_patch = temp_image(random_row*7 + 1 : (random_row+image_size)* 7, ...
%                                            random_col*7 + 1: (random_col+image_size)* 7, :);
%                 img_patch_name = sprintf('%s/%s_%04d.png',cu_test_folder,image_path{train_test_id(j)}(1:end-4),k);                     
%                 imwrite(cropped_patch, img_patch_name);
%                 csv_test_info(end+1,:) = {img_patch_name,i};
%             end
            for k = 1 : every_image_patch_num
                csv_test_info(end+1,:) = {json_name,k,i};
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