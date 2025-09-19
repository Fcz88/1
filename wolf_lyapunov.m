function lambda = wolf_lyapunov(data, dt, varargin)
% WOLF_LYAPUNOV 使用Wolf算法计算最大Lyapunov指数
% 改进版本，解决数值稳定性问题，避免异常大值
%
% 输入参数:
%   data - 时间序列数据 (N x dim 矩阵)
%   dt   - 时间步长
%   可选参数:
%     'EmbedDim'     - 嵌入维数 (默认: 自动选择)
%     'Delay'        - 延时 (默认: 自动选择)
%     'EvolveTime'   - 演化时间 (默认: 自动选择)
%     'MinSepDist'   - 最小分离距离 (默认: 自动选择)
%     'MaxIter'      - 最大迭代次数 (默认: 1000)
%
% 输出:
%   lambda - 最大Lyapunov指数
%
% 改进特性:
% 1. 数值稳定性检查
% 2. 异常值过滤
% 3. 自适应参数选择
% 4. 收敛性验证

    % 解析输入参数
    p = inputParser;
    addRequired(p, 'data', @(x) isnumeric(x) && ismatrix(x));
    addRequired(p, 'dt', @(x) isnumeric(x) && x > 0);
    addParameter(p, 'EmbedDim', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'Delay', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'EvolveTime', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MinSepDist', [], @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaxIter', 1000, @(x) isnumeric(x) && x > 0);
    
    parse(p, data, dt, varargin{:});
    
    % 数据预处理
    if size(data, 2) > size(data, 1)
        data = data';  % 确保 N x dim 格式
    end
    
    [N, dim] = size(data);
    
    % 数据质量检查
    if any(~isfinite(data(:)))
        warning('数据包含无穷大或NaN值，尝试清理...');
        data = data(all(isfinite(data), 2), :);
        N = size(data, 1);
        if N < 100
            error('有效数据点太少');
        end
    end
    
    % 参数自动选择
    embed_dim = p.Results.EmbedDim;
    if isempty(embed_dim)
        if dim == 1
            embed_dim = estimate_embedding_dimension(data);
        else
            embed_dim = dim;
        end
    end
    
    delay = p.Results.Delay;
    if isempty(delay)
        if dim == 1
            delay = estimate_delay(data);
        else
            delay = 1;
        end
    end
    
    evolve_time = p.Results.EvolveTime;
    if isempty(evolve_time)
        evolve_time = max(10, round(0.1 * N));  % 自适应演化时间
    end
    
    min_sep_dist = p.Results.MinSepDist;
    if isempty(min_sep_dist)
        data_std = std(data(:));
        min_sep_dist = data_std * 0.01;  % 数据标准差的1%
    end
    
    max_iter = p.Results.MaxIter;
    
    % 相空间重构
    if dim == 1
        X = reconstruct_phase_space(data, embed_dim, delay);
    else
        X = data;
        embed_dim = dim;
    end
    
    [M, ~] = size(X);
    
    % Wolf算法主体
    lambda_sum = 0;
    valid_iterations = 0;
    lambda_history = [];
    
    % 选择起始点
    start_idx = max(1, round(M * 0.1));  % 跳过初始瞬态
    end_idx = min(M - evolve_time, round(M * 0.9));  % 确保有足够演化时间
    
    for iter = 1:max_iter
        if start_idx > end_idx
            break;
        end
        
        % 随机选择参考点
        ref_idx = start_idx + randi(end_idx - start_idx);
        ref_point = X(ref_idx, :);
        
        % 寻找最近邻点，排除时间上相近的点
        distances = sqrt(sum((X(1:end-evolve_time, :) - ref_point).^2, 2));
        
        % 排除时间窗口内的点
        time_window = max(10, round(0.01 * N));  % 时间窗口
        exclude_idx = max(1, ref_idx - time_window):min(M, ref_idx + time_window);
        distances(exclude_idx) = Inf;
        
        % 找到最小距离大于阈值的最近邻
        valid_distances = find(distances > min_sep_dist & distances < Inf);
        if isempty(valid_distances)
            continue;
        end
        
        [min_dist, min_idx_in_valid] = min(distances(valid_distances));
        min_idx = valid_distances(min_idx_in_valid);
        
        % 检查是否有足够的演化时间
        if min_idx + evolve_time > M || ref_idx + evolve_time > M
            continue;
        end
        
        % 计算演化后的距离
        ref_evolved = X(ref_idx + evolve_time, :);
        neighbor_evolved = X(min_idx + evolve_time, :);
        evolved_dist = sqrt(sum((ref_evolved - neighbor_evolved).^2));
        
        % 数值稳定性检查
        if min_dist <= 0 || evolved_dist <= 0 || ~isfinite(min_dist) || ~isfinite(evolved_dist)
            continue;
        end
        
        % 计算局部Lyapunov指数
        local_lambda = log(evolved_dist / min_dist) / (evolve_time * dt);
        
        % 异常值过滤
        max_reasonable_lambda = 10;  % 合理的最大Lyapunov指数
        min_reasonable_lambda = -10; % 合理的最小Lyapunov指数
        
        if local_lambda > max_reasonable_lambda || local_lambda < min_reasonable_lambda
            continue;
        end
        
        % 累积有效的局部指数
        lambda_sum = lambda_sum + local_lambda;
        valid_iterations = valid_iterations + 1;
        lambda_history = [lambda_history, local_lambda];
        
        % 收敛性检查
        if valid_iterations >= 50 && mod(valid_iterations, 10) == 0
            recent_mean = mean(lambda_history(end-19:end));
            if valid_iterations >= 100
                previous_mean = mean(lambda_history(end-39:end-20));
                if abs(recent_mean - previous_mean) < 0.001
                    break;  % 收敛
                end
            end
        end
    end
    
    % 计算最终结果
    if valid_iterations == 0
        warning('Wolf算法：没有找到有效的近邻点对');
        lambda = NaN;
        return;
    end
    
    lambda = lambda_sum / valid_iterations;
    
    % 最终稳定性检查
    if valid_iterations >= 20
        lambda_std = std(lambda_history);
        if lambda_std > abs(lambda)  % 标准差太大
            warning('Wolf算法：结果不稳定，标准差=%.6f', lambda_std);
        end
    end
    
    % 结果合理性检查
    if abs(lambda) > 5
        warning('Wolf算法：Lyapunov指数可能异常 (%.6f)', lambda);
    end
    
    fprintf('Wolf算法完成：有效迭代 %d/%d, λ=%.6f\n', valid_iterations, max_iter, lambda);
end

function embed_dim = estimate_embedding_dimension(data)
% 估计嵌入维数（False Nearest Neighbors方法的简化版本）
    max_dim = min(10, floor(length(data)/10));
    embed_dim = 3;  % 默认值
    
    for m = 2:max_dim
        % 简化的FNN计算
        delay = estimate_delay(data);
        X = reconstruct_phase_space(data, m, delay);
        X_next = reconstruct_phase_space(data, m+1, delay);
        
        if size(X, 1) < 100
            break;
        end
        
        false_neighbors = 0;
        total_neighbors = min(100, size(X, 1));
        
        for i = 1:total_neighbors
            distances_m = sqrt(sum((X - X(i,:)).^2, 2));
            [~, nearest_idx] = min(distances_m(distances_m > 0));
            
            if nearest_idx <= size(X_next, 1)
                dist_m = distances_m(nearest_idx);
                dist_m1 = sqrt(sum((X_next(i,:) - X_next(nearest_idx,:)).^2));
                
                if dist_m > 0 && dist_m1 / dist_m > 2
                    false_neighbors = false_neighbors + 1;
                end
            end
        end
        
        fnn_ratio = false_neighbors / total_neighbors;
        if fnn_ratio < 0.1  % 阈值
            embed_dim = m;
            break;
        end
    end
end

function delay = estimate_delay(data)
% 估计延时（互信息方法的简化版本）
    max_delay = min(50, floor(length(data)/4));
    delays = 1:max_delay;
    mi = zeros(size(delays));
    
    for i = 1:length(delays)
        tau = delays(i);
        if length(data) - tau < 50
            break;
        end
        
        x1 = data(1:end-tau);
        x2 = data(1+tau:end);
        
        % 简化的互信息计算
        mi(i) = abs(corr(x1, x2));
    end
    
    % 找到第一个局部最小值
    delay = 1;
    for i = 2:length(mi)-1
        if mi(i) < mi(i-1) && mi(i) < mi(i+1)
            delay = delays(i);
            break;
        end
    end
    
    if delay == 1
        delay = max(1, round(max_delay / 3));  % 默认值
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