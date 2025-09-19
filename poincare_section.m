function poincare_section(data, varargin)
% POINCARE_SECTION 庞加莱截面分析
% 通过庞加莱截面可视化系统的动力学结构
%
% 输入参数:
%   data - 时间序列数据 (N x dim)
%   可选参数:
%     'SectionPlane' - 截面定义 [dim, value] (默认: 自动选择)
%     'Tolerance'    - 截面容差 (默认: 自动选择)
%     'Method'       - 插值方法 'linear'或'cubic' (默认: 'linear')

    % 解析输入参数
    p = inputParser;
    addRequired(p, 'data', @(x) isnumeric(x) && ismatrix(x));
    addParameter(p, 'SectionPlane', [], @(x) isnumeric(x) && length(x) == 2);
    addParameter(p, 'Tolerance', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'Method', 'linear', @(x) ischar(x));
    
    parse(p, data, varargin{:});
    
    [N, dim] = size(data);
    
    if dim < 2
        error('庞加莱截面需要至少2维数据');
    end
    
    fprintf('=== 庞加莱截面分析 ===\n');
    
    % 自动选择截面
    section_plane = p.Results.SectionPlane;
    if isempty(section_plane)
        % 选择变化最大的维度作为截面法向
        data_ranges = max(data) - min(data);
        [~, max_dim] = max(data_ranges);
        section_value = mean(data(:, max_dim));
        section_plane = [max_dim, section_value];
        fprintf('自动选择截面: 第%d维 = %.4f\n', max_dim, section_value);
    end
    
    section_dim = section_plane(1);
    section_value = section_plane(2);
    
    % 容差设置
    tolerance = p.Results.Tolerance;
    if isempty(tolerance)
        data_std = std(data(:, section_dim));
        tolerance = data_std * 0.01;  % 1%标准差
    end
    
    method = p.Results.Method;
    
    % 寻找截面交点
    intersections = find_intersections(data, section_dim, section_value, tolerance, method);
    
    if isempty(intersections)
        warning('未找到庞加莱截面交点');
        return;
    end
    
    fprintf('找到 %d 个截面交点\n', size(intersections, 1));
    
    % 可视化庞加莱截面
    visualize_poincare_section(data, intersections, section_plane);
    
    % 分析庞加莱截面特征
    analyze_poincare_features(intersections);
end

function intersections = find_intersections(data, section_dim, section_value, tolerance, method)
% 寻找与庞加莱截面的交点
    [N, dim] = size(data);
    intersections = [];
    
    % 寻找穿越截面的点
    section_data = data(:, section_dim);
    
    for i = 1:N-1
        y1 = section_data(i) - section_value;
        y2 = section_data(i+1) - section_value;
        
        % 检查是否穿越截面（符号改变）
        if y1 * y2 < 0
            % 计算精确交点
            if strcmp(method, 'linear')
                % 线性插值
                t = -y1 / (y2 - y1);  % 插值参数
                intersection = data(i, :) + t * (data(i+1, :) - data(i, :));
            else
                % 简单取中点
                intersection = 0.5 * (data(i, :) + data(i+1, :));
            end
            
            % 验证交点确实在截面上
            if abs(intersection(section_dim) - section_value) <= tolerance
                intersections = [intersections; intersection];
            end
        end
    end
    
    % 移除重复点
    if size(intersections, 1) > 1
        distances = pdist(intersections);
        if any(distances < tolerance)
            % 使用聚类移除重复点
            intersections = remove_duplicates(intersections, tolerance);
        end
    end
end

function unique_points = remove_duplicates(points, tolerance)
% 移除重复的交点
    [N, ~] = size(points);
    unique_indices = true(N, 1);
    
    for i = 1:N-1
        if unique_indices(i)
            distances = sqrt(sum((points(i+1:end, :) - points(i, :)).^2, 2));
            duplicate_indices = find(distances < tolerance) + i;
            unique_indices(duplicate_indices) = false;
        end
    end
    
    unique_points = points(unique_indices, :);
end

function visualize_poincare_section(data, intersections, section_plane)
% 可视化庞加莱截面
    [~, dim] = size(data);
    section_dim = section_plane(1);
    
    figure('Name', '庞加莱截面分析', 'Position', [100, 100, 1200, 800]);
    
    if dim == 2
        % 2D情况：显示相轨道和截面交点
        subplot(2, 2, 1);
        plot(data(:, 1), data(:, 2), 'b-', 'LineWidth', 0.5);
        hold on;
        
        if section_dim == 1
            % 垂直截面线
            ylim_range = ylim;
            plot([section_plane(2), section_plane(2)], ylim_range, 'r--', 'LineWidth', 2);
            plot(intersections(:, 1), intersections(:, 2), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'red');
        else
            % 水平截面线
            xlim_range = xlim;
            plot(xlim_range, [section_plane(2), section_plane(2)], 'r--', 'LineWidth', 2);
            plot(intersections(:, 1), intersections(:, 2), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'red');
        end
        
        xlabel('x');
        ylabel('y');
        title('相轨道与庞加莱截面');
        grid on;
        legend('轨道', '截面', '交点', 'Location', 'best');
        
    elseif dim == 3
        % 3D情况：显示相轨道和截面
        subplot(2, 2, 1);
        plot3(data(:, 1), data(:, 2), data(:, 3), 'b-', 'LineWidth', 0.5);
        hold on;
        plot3(intersections(:, 1), intersections(:, 2), intersections(:, 3), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'red');
        
        xlabel('x');
        ylabel('y');
        zlabel('z');
        title('3D相轨道与庞加莱截面');
        grid on;
        view(3);
        legend('轨道', '截面交点', 'Location', 'best');
        
        % 庞加莱截面投影（去掉截面维度）
        subplot(2, 2, 2);
        other_dims = setdiff(1:dim, section_dim);
        if length(other_dims) >= 2
            plot(intersections(:, other_dims(1)), intersections(:, other_dims(2)), 'ro', 'MarkerSize', 4);
            xlabel(sprintf('x_%d', other_dims(1)));
            ylabel(sprintf('x_%d', other_dims(2)));
            title('庞加莱截面（2D投影）');
            grid on;
        end
    end
    
    % 显示截面交点的时间演化
    if size(intersections, 1) > 1
        subplot(2, 2, 3);
        distances = sqrt(sum(diff(intersections).^2, 2));
        plot(1:length(distances), distances, 'g.-');
        xlabel('交点序号');
        ylabel('相邻交点距离');
        title('相邻截面交点间距离');
        grid on;
        
        % 返回映射（如果可能）
        subplot(2, 2, 4);
        if size(intersections, 1) > 2
            other_dims = setdiff(1:dim, section_dim);
            if ~isempty(other_dims)
                coord = intersections(:, other_dims(1));
                plot(coord(1:end-1), coord(2:end), 'mo', 'MarkerSize', 4);
                xlabel('x_n');
                ylabel('x_{n+1}');
                title('返回映射');
                grid on;
            end
        end
    end
end

function analyze_poincare_features(intersections)
% 分析庞加莱截面的特征
    [N, dim] = size(intersections);
    
    fprintf('\n=== 庞加莱截面特征分析 ===\n');
    fprintf('截面交点数量: %d\n', N);
    
    if N < 5
        fprintf('交点太少，无法进行详细分析\n');
        return;
    end
    
    % 1. 计算相邻交点间距离分布
    distances = sqrt(sum(diff(intersections).^2, 2));
    mean_dist = mean(distances);
    std_dist = std(distances);
    
    fprintf('相邻交点平均距离: %.6f\n', mean_dist);
    fprintf('距离标准差: %.6f\n', std_dist);
    fprintf('距离变异系数: %.3f\n', std_dist / mean_dist);
    
    % 2. 检测周期轨道
    periods = detect_periodic_orbits(intersections);
    if ~isempty(periods)
        fprintf('检测到周期轨道，周期长度: %s\n', mat2str(periods));
    else
        fprintf('未检测到明显的周期轨道\n');
    end
    
    % 3. 分析点的分布特征
    if dim >= 2
        % 计算点云的主成分
        centered_points = intersections - mean(intersections);
        [~, ~, eigenvals] = svd(centered_points);
        eigenvals = diag(eigenvals).^2 / (N-1);  % 方差
        
        fprintf('主成分方差比例: ');
        for i = 1:min(3, length(eigenvals))
            fprintf('%.3f ', eigenvals(i) / sum(eigenvals));
        end
        fprintf('\n');
        
        % 维数估计
        effective_dim = sum(eigenvals > 0.01 * max(eigenvals));
        fprintf('有效维数估计: %d\n', effective_dim);
    end
    
    % 4. 熵估计（简化）
    if N > 20
        box_entropy = estimate_box_counting_entropy(intersections);
        fprintf('盒计数熵估计: %.4f\n', box_entropy);
    end
end

function periods = detect_periodic_orbits(intersections)
% 检测周期轨道
    [N, ~] = size(intersections);
    periods = [];
    
    % 检测周期长度从2到N/4
    max_period = min(20, floor(N/4));
    tolerance = 0.1 * std(intersections(:));  % 容差
    
    for p = 2:max_period
        if N < 3*p  % 需要至少3个周期来验证
            continue;
        end
        
        % 检查是否存在周期p的轨道
        is_periodic = true;
        for i = 1:N-2*p
            dist1 = sqrt(sum((intersections(i, :) - intersections(i+p, :)).^2));
            dist2 = sqrt(sum((intersections(i+p, :) - intersections(i+2*p, :)).^2));
            
            if dist1 > tolerance || dist2 > tolerance
                is_periodic = false;
                break;
            end
        end
        
        if is_periodic
            periods = [periods, p];
        end
    end
end

function entropy = estimate_box_counting_entropy(points)
% 简化的盒计数熵估计
    [N, dim] = size(points);
    
    % 归一化点到[0,1]^dim
    min_vals = min(points);
    max_vals = max(points);
    ranges = max_vals - min_vals;
    
    % 避免除零
    ranges(ranges == 0) = 1;
    
    normalized_points = (points - min_vals) ./ ranges;
    
    % 不同的盒子大小
    box_sizes = [0.1, 0.05, 0.02, 0.01];
    log_counts = [];
    log_sizes = [];
    
    for box_size = box_sizes
        % 计算每个维度的盒子数
        boxes_per_dim = ceil(1 / box_size);
        
        % 将点分配到盒子
        box_indices = floor(normalized_points * boxes_per_dim) + 1;
        box_indices = min(box_indices, boxes_per_dim);  % 边界处理
        
        % 计算唯一盒子数
        if dim == 1
            unique_boxes = unique(box_indices);
        else
            % 多维情况：将多维索引转换为唯一标识
            multipliers = boxes_per_dim .^ (0:dim-1);
            box_ids = sum((box_indices - 1) .* multipliers, 2);
            unique_boxes = unique(box_ids);
        end
        
        box_count = length(unique_boxes);
        
        if box_count > 1
            log_counts = [log_counts, log(box_count)];
            log_sizes = [log_sizes, log(box_size)];
        end
    end
    
    % 线性回归估计维数
    if length(log_counts) >= 2
        p = polyfit(log_sizes, log_counts, 1);
        entropy = -p(1);  % 斜率的负值
    else
        entropy = NaN;
    end
end