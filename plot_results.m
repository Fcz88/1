function plot_results(t, x, lambda, decision)
% PLOT_RESULTS 绘制Lyapunov指数分析结果
% 综合可视化时间序列、相空间、功率谱等
%
% 输入参数:
%   t        - 时间向量
%   x        - 状态矩阵
%   lambda   - Lyapunov指数
%   decision - 系统状态决策

    [N, dim] = size(x);
    
    % 创建主图窗
    figure('Name', 'Lyapunov指数分析结果', 'Position', [50, 50, 1400, 900]);
    
    % 设置颜色方案
    if lambda > 0.001
        color_scheme = 'red';    % 混沌
        state_color = [0.8, 0.2, 0.2];
    elseif lambda < -0.001
        color_scheme = 'blue';   % 稳定
        state_color = [0.2, 0.2, 0.8];
    else
        color_scheme = 'orange'; % 边界/准周期
        state_color = [0.8, 0.6, 0.2];
    end
    
    % 1. 时间序列图
    subplot(3, 4, [1, 2]);
    if dim == 1
        plot(t, x, 'Color', state_color, 'LineWidth', 1);
        ylabel('x(t)');
    else
        plot(t, x(:, 1), 'Color', state_color, 'LineWidth', 1);
        hold on;
        if dim >= 2
            plot(t, x(:, 2), 'Color', state_color * 0.7, 'LineWidth', 1);
        end
        if dim >= 3
            plot(t, x(:, 3), 'Color', state_color * 0.4, 'LineWidth', 1);
        end
        ylabel('状态变量');
        if dim == 2
            legend('x(t)', 'y(t)', 'Location', 'best');
        elseif dim >= 3
            legend('x(t)', 'y(t)', 'z(t)', 'Location', 'best');
        end
    end
    xlabel('时间 t');
    title('时间序列');
    grid on;
    
    % 2. 相空间图
    if dim >= 2
        subplot(3, 4, [3, 4]);
        if dim == 2
            plot(x(:, 1), x(:, 2), 'Color', state_color, 'LineWidth', 0.8);
            xlabel('x');
            ylabel('y');
            title('2D相空间');
        else
            plot3(x(:, 1), x(:, 2), x(:, 3), 'Color', state_color, 'LineWidth', 0.8);
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title('3D相空间');
            view(3);
        end
        grid on;
        axis equal;
    else
        % 1维数据：绘制延时嵌入相空间
        subplot(3, 4, [3, 4]);
        delay = estimate_delay(x);
        if N > delay
            x_delayed = [x(1:end-delay), x(1+delay:end)];
            plot(x_delayed(:, 1), x_delayed(:, 2), 'Color', state_color, 'LineWidth', 0.8);
            xlabel('x(t)');
            ylabel(sprintf('x(t+%d)', delay));
            title('延时嵌入相空间');
            grid on;
        end
    end
    
    % 3. 功率谱密度
    subplot(3, 4, 5);
    signal = x(:, 1);  % 使用第一个分量
    [psd, freq] = periodogram(signal, [], [], 1/mean(diff(t)));
    semilogy(freq, psd, 'Color', state_color, 'LineWidth', 1.5);
    xlabel('频率');
    ylabel('功率谱密度');
    title('功率谱');
    grid on;
    
    % 4. 自相关函数
    subplot(3, 4, 6);
    max_lag = min(200, floor(N/4));
    autocorr = compute_autocorrelation(signal, max_lag);
    lags = 0:max_lag-1;
    plot(lags, autocorr, 'Color', state_color, 'LineWidth', 1.5);
    xlabel('滞后');
    ylabel('自相关');
    title('自相关函数');
    grid on;
    
    % 5. 相空间重构（如果是1维数据）
    if dim == 1
        subplot(3, 4, 7);
        embed_dim = 3;
        delay = estimate_delay(signal);
        if N > (embed_dim-1)*delay
            X_embed = reconstruct_phase_space(signal, embed_dim, delay);
            plot3(X_embed(:, 1), X_embed(:, 2), X_embed(:, 3), 'Color', state_color, 'LineWidth', 0.5);
            xlabel('x(t)');
            ylabel(sprintf('x(t+%d)', delay));
            zlabel(sprintf('x(t+%d)', 2*delay));
            title('相空间重构');
            grid on;
            view(3);
        end
    else
        % 多维数据：绘制第一个和第三个分量（如果存在）
        subplot(3, 4, 7);
        if dim >= 3
            plot(x(:, 1), x(:, 3), 'Color', state_color, 'LineWidth', 0.8);
            xlabel('x');
            ylabel('z');
            title('x-z投影');
        else
            plot(x(:, 1), x(:, 2), 'Color', state_color, 'LineWidth', 0.8);
            xlabel('x');
            ylabel('y');
            title('相空间投影');
        end
        grid on;
    end
    
    % 6. 返回映射（Poincaré映射）
    subplot(3, 4, 8);
    if dim >= 2
        % 简单的返回映射：x(n+1) vs x(n)
        plot(x(1:end-1, 1), x(2:end, 1), '.', 'Color', state_color, 'MarkerSize', 2);
        xlabel('x(n)');
        ylabel('x(n+1)');
        title('返回映射');
    else
        % 1维情况：使用延时坐标
        delay = estimate_delay(signal);
        if N > delay
            plot(signal(1:end-delay), signal(1+delay:end), '.', 'Color', state_color, 'MarkerSize', 2);
            xlabel('x(n)');
            ylabel(sprintf('x(n+%d)', delay));
            title('延时返回映射');
        end
    end
    grid on;
    
    % 7. 分岔图风格的展示
    subplot(3, 4, 9);
    if dim >= 2
        % 显示局部最大值
        [peaks, locs] = findpeaks(x(:, 1), 'MinPeakDistance', 10);
        if ~isempty(peaks)
            scatter(locs * mean(diff(t)), peaks, 20, state_color, 'filled');
            xlabel('时间');
            ylabel('局部最大值');
            title('峰值分布');
        else
            plot(t, x(:, 1), 'Color', state_color);
            xlabel('时间');
            ylabel('x(t)');
            title('时间序列');
        end
    else
        [peaks, locs] = findpeaks(signal, 'MinPeakDistance', 10);
        if ~isempty(peaks)
            scatter(locs * mean(diff(t)), peaks, 20, state_color, 'filled');
            xlabel('时间');
            ylabel('局部最大值');
            title('峰值分布');
        else
            plot(t, signal, 'Color', state_color);
            xlabel('时间');
            ylabel('x(t)');
            title('时间序列');
        end
    end
    grid on;
    
    % 8. 系统信息面板
    subplot(3, 4, [10, 11, 12]);
    axis off;
    
    % 显示分析结果
    info_text = {
        '=== Lyapunov指数分析结果 ==='
        ''
        sprintf('最大Lyapunov指数: %.6f', lambda)
        sprintf('系统状态: %s', decision)
        ''
        '=== 系统特征 ==='
        sprintf('数据维度: %d', dim)
        sprintf('数据长度: %d点', N)
        sprintf('时间跨度: %.2f', t(end) - t(1))
        sprintf('采样间隔: %.4f', mean(diff(t)))
        ''
        '=== 动力学分类 ==='
    };
    
    if lambda > 0.001
        info_text{end+1} = '• 混沌态 (λ > 0)';
        info_text{end+1} = '• 轨道对初值敏感';
        info_text{end+1} = '• 长期不可预测';
        info_text{end+1} = '• 存在奇异吸引子';
    elseif lambda < -0.001
        info_text{end+1} = '• 稳定态 (λ < 0)';
        info_text{end+1} = '• 扰动衰减';
        info_text{end+1} = '• 趋向固定点';
        info_text{end+1} = '• 长期可预测';
    else
        info_text{end+1} = '• 边界态 (λ ≈ 0)';
        info_text{end+1} = '• 可能是准周期';
        info_text{end+1} = '• 或周期轨道';
        info_text{end+1} = '• 需进一步分析';
    end
    
    % 显示文本
    text(0.05, 0.95, info_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 10, 'FontName', 'Consolas', 'Color', state_color);
    
    % 添加标题
    sgtitle(sprintf('Lyapunov指数分析: λ = %.6f, 状态: %s', lambda, decision), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', state_color);
    
    % 调整子图间距
    tight_layout();
end

function autocorr = compute_autocorrelation(signal, max_lag)
% 计算自相关函数
    N = length(signal);
    signal = signal - mean(signal);  % 去均值
    autocorr = zeros(max_lag, 1);
    
    for lag = 1:max_lag
        if N - lag > 0
            autocorr(lag) = sum(signal(1:end-lag) .* signal(1+lag:end)) / ...
                           sqrt(sum(signal(1:end-lag).^2) * sum(signal(1+lag:end).^2));
        end
    end
end

function delay = estimate_delay(data)
% 简单的延时估计
    N = length(data);
    max_delay = min(50, floor(N/10));
    
    autocorr = compute_autocorrelation(data, max_delay);
    
    % 找到第一个过零点或1/e点
    delay = 1;
    threshold = 1/exp(1);
    
    for i = 2:length(autocorr)
        if autocorr(i) < threshold
            delay = i;
            break;
        end
    end
    
    if delay == 1
        delay = max(1, round(max_delay / 3));
    end
end

function X = reconstruct_phase_space(data, embed_dim, delay)
% 相空间重构
    N = length(data);
    M = N - (embed_dim - 1) * delay;
    
    if M <= 0
        error('数据长度不足');
    end
    
    X = zeros(M, embed_dim);
    for i = 1:embed_dim
        start_idx = (i-1) * delay + 1;
        end_idx = start_idx + M - 1;
        X(:, i) = data(start_idx:end_idx);
    end
end

function tight_layout()
% 简单的紧凑布局调整
    % 这个函数在较新的MATLAB版本中可能不需要
    % 这里提供一个简单的实现
    try
        % 尝试使用内置函数
        if exist('sgtitle', 'file')
            % 新版本MATLAB
        end
    catch
        % 旧版本MATLAB的处理
    end
end