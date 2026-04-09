% path = 'data\widespread\hrtf\P00001.mat'; % Widespread 数据集
path = 'data\sonicom\hrtf\P0001.mat';   % Sonicom 数据集
data = load(path);  % 加载文件
theta = data.theta;          % 提取 theta 变量（2562x2 double）

% 步骤2：定义目标值（第一列是方位角phi，第二列是仰角theta）
target_col1 = 0;    % 第一列目标值
target_col2 = 0;   % 第二列目标值

% target_col1 = 0;    % 第一列目标值
% target_col2 = 90;   % 第二列目标值

% target_col1 = 90;    % 第一列目标值
% target_col2 = 0;   % 第二列目标值

% target_col1 = 20;    % 第一列目标值
% target_col2 = 45;   % 第二列目标值

% 步骤3：查找满足条件的行索引（精确匹配）
% 方法1：直接逻辑索引（适用于整数或精确浮点数）
%row_indices = find(theta(:,1) == target_col1 & theta(:,2) == target_col2);

% 方法2：容差匹配（推荐用于浮点数，如theta包含26.5652等值）
 tolerance = 3;  % 设置容差
 row_indices = find(abs(theta(:,1) - target_col1) < tolerance & ...
                   abs(theta(:,2) - target_col2) < tolerance);

% 步骤4：显示结果
if isempty(row_indices)
    disp('未找到匹配的行。请检查目标值或数据。');
else
    disp(['匹配的行索引: ', num2str(row_indices')]);
    disp('对应的数据行:');
    disp(theta(row_indices, :));  % 显示匹配的具体数据
end