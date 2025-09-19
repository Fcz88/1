function x = henon_map(x0, params, N)
% HENON_MAP Henon映射迭代计算
% 二维离散动力学系统
%
% 输入参数:
%   x0     - 初始条件 [x0, y0]
%   params - 系统参数 [a, b]
%   N      - 迭代次数
%
% 输出:
%   x - 状态矩阵 [x, y] (N x 2)
%
% Henon映射:
%   x(n+1) = 1 - a * x(n)^2 + y(n)
%   y(n+1) = b * x(n)

    % 参数检查
    if length(x0) ~= 2
        error('初始条件必须是2维向量');
    end
    
    if length(params) ~= 2
        error('参数必须包含 [a, b]');
    end
    
    if N <= 0
        error('迭代次数必须为正数');
    end
    
    % 提取参数
    a = params(1);
    b = params(2);
    
    % 初始化
    x = zeros(N, 2);
    x(1, :) = x0;
    
    % 迭代计算
    for i = 1:N-1
        xi = x(i, 1);
        yi = x(i, 2);
        
        % Henon映射
        x(i+1, 1) = 1 - a * xi^2 + yi;
        x(i+1, 2) = b * xi;
        
        % 数值稳定性检查
        if any(abs(x(i+1, :)) > 1e6)
            warning('迭代可能发散，在第 %d 步停止', i);
            x = x(1:i+1, :);
            break;
        end
    end
    
    fprintf('Henon映射迭代完成: %d个点\n', size(x, 1));
end