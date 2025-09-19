function lambda = discrete_lyapunov(x, params)
% DISCRETE_LYAPUNOV 计算离散系统的Lyapunov指数
% 适用于映射等离散动力学系统
%
% 输入参数:
%   x      - 时间序列数据 (N x dim)
%   params - 系统参数（用于Jacobian计算，可选）
%
% 输出:
%   lambda - 最大Lyapunov指数

    [N, dim] = size(x);
    
    if dim == 2 && ~isempty(params) && length(params) == 2
        % Henon映射的解析Jacobian方法
        lambda = henon_analytical_lyapunov(x, params);
    else
        % 通用数值方法
        lambda = numerical_discrete_lyapunov(x);
    end
end

function lambda = henon_analytical_lyapunov(x, params)
% Henon映射的解析Lyapunov指数计算
    a = params(1);
    b = params(2);
    
    N = size(x, 1);
    log_sum = 0;
    valid_count = 0;
    
    % 计算每一点的Jacobian矩阵的行列式
    for i = 1:N-1
        xi = x(i, 1);
        
        % Henon映射的Jacobian矩阵
        % J = [-2*a*x, 1; b, 0]
        J = [-2*a*xi, 1; b, 0];
        
        % 计算特征值
        eigenvals = eig(J);
        
        % 取最大特征值的模
        max_eigenval = max(abs(eigenvals));
        
        if max_eigenval > 0 && isfinite(max_eigenval)
            log_sum = log_sum + log(max_eigenval);
            valid_count = valid_count + 1;
        end
    end
    
    if valid_count > 0
        lambda = log_sum / valid_count;
    else
        lambda = NaN;
    end
    
    fprintf('Henon解析方法: λ=%.6f\n', lambda);
end

function lambda = numerical_discrete_lyapunov(x)
% 通用数值方法计算离散系统Lyapunov指数
    [N, dim] = size(x);
    
    % 最小分离距离
    data_std = std(x(:));
    min_dist = data_std * 0.01;
    max_dist = data_std * 0.1;
    
    % 寻找近邻点对
    neighbors = [];
    min_time_sep = max(10, round(N * 0.05));  % 时间分离
    
    for i = 1:N-min_time_sep-1
        if mod(i, 1000) == 0
            fprintf('  处理进度: %d/%d\n', i, N-min_time_sep-1);
        end
        
        % 计算距离
        ref_point = x(i, :);
        distances = sqrt(sum((x(i+min_time_sep:end-1, :) - ref_point).^2, 2));
        
        % 找到合适距离范围内的点
        valid_idx = find(distances >= min_dist & distances <= max_dist);
        
        if ~isempty(valid_idx)
            % 选择最近的几个邻居
            [sorted_dist, sort_idx] = sort(distances(valid_idx));
            num_neighbors = min(3, length(valid_idx));
            
            for j = 1:num_neighbors
                neighbor_idx = valid_idx(sort_idx(j)) + i + min_time_sep - 1;
                if neighbor_idx < N
                    neighbors = [neighbors; i, neighbor_idx, sorted_dist(j)];
                end
            end
        end
    end
    
    if isempty(neighbors)
        warning('未找到合适的近邻点对');
        lambda = NaN;
        return;
    end
    
    fprintf('找到 %d 个近邻点对\n', size(neighbors, 1));
    
    % 计算Lyapunov指数
    log_expansions = [];
    
    for k = 1:size(neighbors, 1)
        i = neighbors(k, 1);
        j = neighbors(k, 2);
        
        if i < N && j < N
            % 当前距离
            current_dist = sqrt(sum((x(i, :) - x(j, :)).^2));
            
            % 下一步距离
            next_dist = sqrt(sum((x(i+1, :) - x(j+1, :)).^2));
            
            if current_dist > 0 && next_dist > 0 && isfinite(current_dist) && isfinite(next_dist)
                expansion = next_dist / current_dist;
                
                % 异常值过滤
                if expansion > 0.1 && expansion < 10
                    log_expansions = [log_expansions, log(expansion)];
                end
            end
        end
    end
    
    if isempty(log_expansions)
        warning('没有有效的扩张率计算');
        lambda = NaN;
    else
        lambda = mean(log_expansions);
        
        % 稳定性检查
        if length(log_expansions) > 10
            expansion_std = std(log_expansions);
            if expansion_std > abs(lambda)
                warning('离散Lyapunov指数不稳定，标准差=%.6f', expansion_std);
            end
        end
    end
    
    fprintf('数值方法: λ=%.6f (样本数: %d)\n', lambda, length(log_expansions));
end