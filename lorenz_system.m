function [t, x] = lorenz_system(x0, params, dt, N)
% LORENZ_SYSTEM Lorenz混沌系统数值求解
% 使用四阶Runge-Kutta方法求解Lorenz微分方程组
%
% 输入参数:
%   x0     - 初始条件 [x0, y0, z0]
%   params - 系统参数 [sigma, beta, rho]
%   dt     - 时间步长
%   N      - 积分步数
%
% 输出:
%   t - 时间向量
%   x - 状态向量 [x, y, z]
%
% Lorenz方程:
%   dx/dt = sigma * (y - x)
%   dy/dt = x * (rho - z) - y  
%   dz/dt = x * y - beta * z

    % 参数检查
    if length(x0) ~= 3
        error('初始条件必须是3维向量');
    end
    
    if length(params) ~= 3
        error('参数必须包含 [sigma, beta, rho]');
    end
    
    if dt <= 0 || N <= 0
        error('时间步长和积分步数必须为正数');
    end
    
    % 提取参数
    sigma = params(1);
    beta = params(2);
    rho = params(3);
    
    % 初始化
    t = zeros(N, 1);
    x = zeros(N, 3);
    
    % 设置初始条件
    x(1, :) = x0;
    t(1) = 0;
    
    % 四阶Runge-Kutta积分
    for i = 1:N-1
        % 当前状态
        xi = x(i, 1);
        yi = x(i, 2);
        zi = x(i, 3);
        
        % RK4第一步
        k1x = sigma * (yi - xi);
        k1y = xi * (rho - zi) - yi;
        k1z = xi * yi - beta * zi;
        
        % RK4第二步
        x_temp = xi + 0.5 * dt * k1x;
        y_temp = yi + 0.5 * dt * k1y;
        z_temp = zi + 0.5 * dt * k1z;
        
        k2x = sigma * (y_temp - x_temp);
        k2y = x_temp * (rho - z_temp) - y_temp;
        k2z = x_temp * y_temp - beta * z_temp;
        
        % RK4第三步
        x_temp = xi + 0.5 * dt * k2x;
        y_temp = yi + 0.5 * dt * k2y;
        z_temp = zi + 0.5 * dt * k2z;
        
        k3x = sigma * (y_temp - x_temp);
        k3y = x_temp * (rho - z_temp) - y_temp;
        k3z = x_temp * y_temp - beta * z_temp;
        
        % RK4第四步
        x_temp = xi + dt * k3x;
        y_temp = yi + dt * k3y;
        z_temp = zi + dt * k3z;
        
        k4x = sigma * (y_temp - x_temp);
        k4y = x_temp * (rho - z_temp) - y_temp;
        k4z = x_temp * y_temp - beta * z_temp;
        
        % 更新状态
        x(i+1, 1) = xi + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x);
        x(i+1, 2) = yi + (dt/6) * (k1y + 2*k2y + 2*k3y + k4y);
        x(i+1, 3) = zi + (dt/6) * (k1z + 2*k2z + 2*k3z + k4z);
        
        % 更新时间
        t(i+1) = i * dt;
        
        % 数值稳定性检查
        if any(abs(x(i+1, :)) > 1e6)
            warning('数值解可能发散，在第 %d 步停止', i);
            t = t(1:i+1);
            x = x(1:i+1, :);
            break;
        end
    end
    
    fprintf('Lorenz系统积分完成: %d个时间步\n', length(t));
end