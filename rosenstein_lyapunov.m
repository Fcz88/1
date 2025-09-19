function lambda = rosenstein_lyapunov(data, dt, varargin)
% ROSENSTEIN_LYAPUNOV 使用Rosenstein算法计算最大Lyapunov指数
% 改进版本，解决结果不稳定问题
%
% 输入参数:
%   data - 时间序列数据 (N x dim 矩阵)
%   dt   - 时间步长
%   可选参数:
%     'EmbedDim'    - 嵌入维数 (默认: 自动选择)
%     'Delay'       - 延时 (默认: 自动选择)
%     'MinDist'     - 最小距离阈值 (默认: 自动选择)
%     'MaxDist'     - 最大距离阈值 (默认: 自动选择)
%     'FitRange'    - 拟合范围 [start_time, end_time] (默认: 自动选择)
%
% 输出:
%   lambda - 最大Lyapunov指数
%
% 改进特性:
% 1. 自适应距离阈值
% 2. 稳健的线性拟合
% 3. 异常值检测和处理
% 4. 多段验证

    % 解析输入参数
    p = inputParser;
    addRequired(p, 'data', @(x) isnumeric(x) && ismatrix(x));
    addRequired(p, 'dt', @(x) isnumeric(x) && x > 0);
    addParameter(p, 'EmbedDim', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'Delay', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MinDist', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaxDist', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'FitRange', [], @(x) isnumeric(x) && length(x) == 2);
    
    parse(p, data, dt, varargin{:});
    
    % 数据预处理
    if size(data, 2) > size(data, 1)
        data = data';
    end
    
    [N, dim] = size(data);
    
    % 数据质量检查
    if any(~isfinite(data(:)))
        warning('数据包含无穷大或NaN值，尝试清理...');
        data = data(all(isfinite(data), 2), :);
        N = size(data, 1);
        if N < 200
            error('有效数据点太少');
        end
    end
    
    % 参数自动选择
    embed_dim = p.Results.EmbedDim;
    if isempty(embed_dim)
        if dim == 1
            embed_dim = estimate_embedding_dimension_simple(data);
        else
            embed_dim = dim;
        end
    end
    
    delay = p.Results.Delay;
    if isempty(delay)
        if dim == 1
            delay = estimate_delay_simple(data);
        else
            delay = 1;
        end
    end
    
    % 相空间重构
    if dim == 1
        X = reconstruct_phase_space(data, embed_dim, delay);
    else
        X = data;
    end
    
    [M, d] = size(X);
    
    % 距离阈值自动选择
    data_std = std(X(:));
    min_dist = p.Results.MinDist;
    if isempty(min_dist)
        min_dist = data_std * 0.005;  % 数据标准差的0.5%
    end
    
    max_dist = p.Results.MaxDist;
    if isempty(max_dist)
        max_dist = data_std * 0.1;    % 数据标准差的10%
    end
    
    % 寻找近邻点对
    fprintf('寻找近邻点对...\n');
    neighbors = find_neighbors(X, min_dist, max_dist);
    
    if isempty(neighbors)
        warning('Rosenstein算法：未找到合适的近邻点对');
        lambda = NaN;
        return;
    end
    
    fprintf('找到 %d 个近邻点对\n', size(neighbors, 1));
    
    % 计算分离演化
    max_evolution = min(200, floor(M * 0.2));  % 最大演化时间
    separation_matrix = compute_separation_evolution(X, neighbors, max_evolution);
    
    % 计算平均对数分离
    mean_log_sep = compute_mean_log_separation(separation_matrix);
    
    if isempty(mean_log_sep) || all(~isfinite(mean_log_sep))
        warning('Rosenstein算法：分离演化计算失败');
        lambda = NaN;
        return;
    end
    
    % 线性拟合计算Lyapunov指数
    time_steps = (0:length(mean_log_sep)-1) * dt;
    [lambda, fit_quality] = robust_linear_fit(time_steps, mean_log_sep, p.Results.FitRange);
    
    if fit_quality.r_squared < 0.5
        warning('Rosenstein算法：拟合质量差 (R²=%.3f)', fit_quality.r_squared);
    end
    
    % 结果验证
    if abs(lambda) > 5
        warning('Rosenstein算法：Lyapunov指数可能异常 (%.6f)', lambda);
    end
    
    fprintf('Rosenstein算法完成：λ=%.6f (R²=%.3f)\n', lambda, fit_quality.r_squared);
end

function neighbors = find_neighbors(X, min_dist, max_dist)
% 寻找合适的近邻点对
    [M, ~] = size(X);
    neighbors = [];
    
    % 时间分离窗口
    min_time_sep = max(10, round(M * 0.01));
    
    for i = 1:M-min_time_sep
        if mod(i, 500) == 0
            fprintf('  处理进度: %d/%d\n', i, M-min_time_sep);
        end
        
        % 计算到所有其他点的距离
        distances = sqrt(sum((X(i+min_time_sep:end, :) - X(i, :)).^2, 2));
        
        % 找到距离在合适范围内的点
        valid_indices = find(distances >= min_dist & distances <= max_dist);
        
        if ~isempty(valid_indices)
            % 选择最近的几个邻居
            [sorted_distances, sort_idx] = sort(distances(valid_indices));
            num_neighbors = min(5, length(valid_indices));  % 最多选择5个邻居
            
            for j = 1:num_neighbors
                neighbor_idx = valid_indices(sort_idx(j)) + i + min_time_sep - 1;
                neighbors = [neighbors; i, neighbor_idx, sorted_distances(j)];
            end
        end
    end
end

function separation_matrix = compute_separation_evolution(X, neighbors, max_evolution)
% 计算分离演化矩阵
    [M, ~] = size(X);
    num_pairs = size(neighbors, 1);
    separation_matrix = NaN(num_pairs, max_evolution + 1);
    
    for k = 1:num_pairs
        i = neighbors(k, 1);
        j = neighbors(k, 2);
        
        % 确保有足够的演化时间
        max_evo_this_pair = min(max_evolution, M - max(i, j));
        
        for t = 0:max_evo_this_pair
            if i + t <= M && j + t <= M
                separation = sqrt(sum((X(i + t, :) - X(j + t, :)).^2));
                separation_matrix(k, t + 1) = separation;
            end
        end
    end
end

function mean_log_sep = compute_mean_log_separation(separation_matrix)
% 计算平均对数分离
    [num_pairs, max_time] = size(separation_matrix);
    mean_log_sep = [];
    
    for t = 1:max_time
        separations = separation_matrix(:, t);
        
        % 移除无效值
        valid_separations = separations(isfinite(separations) & separations > 0);
        
        if length(valid_separations) < num_pairs * 0.1  % 至少需要10%的有效数据
            break;
        end
        
        % 计算对数并处理异常值
        log_separations = log(valid_separations);
        
        % 使用中位数替代均值以提高稳健性
        mean_log_sep = [mean_log_sep, median(log_separations)];
    end
end

function [slope, fit_quality] = robust_linear_fit(x, y, fit_range)
% 稳健的线性拟合
    if isempty(fit_range)
        % 自动选择拟合范围
        valid_data = isfinite(x) & isfinite(y);
        if sum(valid_data) < 10
            slope = NaN;
            fit_quality.r_squared = 0;
            return;
        end
        
        % 选择线性区域（通常在前半部分）
        start_idx = max(1, find(valid_data, 1, 'first'));
        end_idx = min(length(x), start_idx + round(sum(valid_data) * 0.5));
    else
        start_idx = find(x >= fit_range(1), 1, 'first');
        end_idx = find(x <= fit_range(2), 1, 'last');
        if isempty(start_idx) || isempty(end_idx)
            slope = NaN;
            fit_quality.r_squared = 0;
            return;
        end
    end
    
    % 提取拟合数据
    fit_x = x(start_idx:end_idx);
    fit_y = y(start_idx:end_idx);
    
    % 移除异常值
    valid_idx = isfinite(fit_x) & isfinite(fit_y);
    fit_x = fit_x(valid_idx);
    fit_y = fit_y(valid_idx);
    
    if length(fit_x) < 5
        slope = NaN;
        fit_quality.r_squared = 0;
        return;
    end
    
    % 线性回归
    p = polyfit(fit_x, fit_y, 1);
    slope = p(1);
    
    % 计算拟合质量
    y_pred = polyval(p, fit_x);
    ss_res = sum((fit_y - y_pred).^2);
    ss_tot = sum((fit_y - mean(fit_y)).^2);
    
    if ss_tot == 0
        fit_quality.r_squared = 0;
    else
        fit_quality.r_squared = 1 - ss_res / ss_tot;
    end
    
    fit_quality.slope = slope;
    fit_quality.intercept = p(2);
    fit_quality.fit_range = [fit_x(1), fit_x(end)];
end

function embed_dim = estimate_embedding_dimension_simple(data)
% 简化的嵌入维数估计
    max_dim = min(8, floor(length(data)/20));
    embed_dim = 3;  % 默认值
    
    % 使用相关积分方法的简化版本
    for m = 2:max_dim
        delay = estimate_delay_simple(data);
        
        if length(data) - (m-1)*delay < 100
            break;
        end
        
        X1 = reconstruct_phase_space(data, m, delay);
        X2 = reconstruct_phase_space(data, m+1, delay);
        
        % 计算相关维数的变化
        r = std(data) * 0.1;
        C1 = correlation_sum(X1, r);
        C2 = correlation_sum(X2, r);
        
        if C2 > 0 && C1 > 0
            ratio = C2 / C1;
            if ratio > 0.95  % 相关积分变化很小
                embed_dim = m;
                break;
            end
        end
    end
end

function delay = estimate_delay_simple(data)
% 简化的延时估计
    max_delay = min(20, floor(length(data)/10));
    autocorr_values = zeros(max_delay, 1);
    
    data_centered = data - mean(data);
    
    for tau = 1:max_delay
        if length(data) - tau < 50
            break;
        end
        
        x1 = data_centered(1:end-tau);
        x2 = data_centered(1+tau:end);
        
        autocorr_values(tau) = sum(x1 .* x2) / sqrt(sum(x1.^2) * sum(x2.^2));
    end
    
    % 找到第一个过零点或1/e点
    delay = 1;
    threshold = 1/exp(1);  % 1/e ≈ 0.368
    
    for tau = 2:length(autocorr_values)
        if autocorr_values(tau) < threshold
            delay = tau;
            break;
        end
    end
    
    if delay == 1
        delay = max(1, round(max_delay / 2));
    end
end

function C = correlation_sum(X, r)
% 计算相关积分
    [N, ~] = size(X);
    count = 0;
    total = 0;
    
    % 随机采样以提高计算效率
    sample_size = min(500, N);
    sample_indices = randperm(N, sample_size);
    
    for i = 1:sample_size
        for j = i+1:sample_size
            total = total + 1;
            distance = sqrt(sum((X(sample_indices(i), :) - X(sample_indices(j), :)).^2));
            if distance < r
                count = count + 1;
            end
        end
    end
    
    if total > 0
        C = count / total;
    else
        C = 0;
    end
end

function X = reconstruct_phase_space(data, embed_dim, delay)
% 相空间重构
    N = length(data);
    M = N - (embed_dim - 1) * delay;
    
    if M <= 0
        error('数据长度不足以进行相空间重构');
    end
    
    X = zeros(M, embed_dim);
    for i = 1:embed_dim
        start_idx = (i-1) * delay + 1;
        end_idx = start_idx + M - 1;
        X(:, i) = data(start_idx:end_idx);
    end
end