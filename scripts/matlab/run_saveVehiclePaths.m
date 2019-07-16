function run_saveVehiclePaths (base_dir)
% Modified from KITTI RAW DATA DEVELOPMENT KIT
%
% Saves OXTS poses of a list of sequences
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)

% clear and close everything
clear all; close all; dbstop error; clc;
disp('======= KITTI DevKit Demo =======');

% fileID = fopen('obj_raw_drive_paths.txt', 'r');
fileID = fopen('raw_drive_dirs.txt', 'r');

all_drive_ids = {};
while feof(fileID) == 0
    line = fgetl(fileID);
    all_drive_ids = [all_drive_ids, line];
end
all_drive_ids

for i = 1:length(all_drive_ids)

    base_dir = all_drive_ids{i};

    base_dir

    % load oxts data
    oxts = loadOxtsliteData(base_dir);

    % transform to poses
    poses = convertOxtsToPose(oxts);

    % save to dir
    save(strcat(base_dir, '/poses.mat'), 'poses')

end
