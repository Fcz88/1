function [final_lambda, confidence, decision] = chaos_expert_system(lambda_wolf, lambda_rosenstein, data, varargin)
% CHAOS_EXPERT_SYSTEM 专家决策系统，综合多种指标判断系统状态
% 避免将数值噪声误判为混沌，提供可靠的混沌识别
%
% 输入参数:
%   lambda_wolf       - Wolf算法结果
%   lambda_rosenstein - Rosenstein算法结果  
%   data             - 原始时间序列数据
%   可选参数:
%     'Threshold'     - 混沌判断阈值 (默认: 0.001)
%     'NoiseLevel'    - 噪声水平阈值 (默认: 自动估计)
%
% 输出:
%   final_lambda - 最终Lyapunov指数估计
%   confidence   - 置信度 (0-1)
%   decision     - 系统状态决策字符串
%
% 专家规则:
% 1. 算法一致性检验
% 2. 噪声水平评估
% 3. 周期性检测
% 4. 功率谱分析
% 5. 相关维数估计

    % 解析输入参数
    p = inputParser;
    addRequired(p, 'lambda_wolf', @isnumeric);
    addRequired(p, 'lambda_rosenstein', @isnumeric);
    addRequired(p, 'data', @(x) isnumeric(x) && ismatrix(x));
    addParameter(p, 'Threshold', 0.001, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'NoiseLevel', [], @(x) isnumeric(x) && x > 0);
    
    parse(p, lambda_wolf, lambda_rosenstein, data, varargin{:});
    
    threshold = p.Results.Threshold;
    noise_level = p.Results.NoiseLevel;
    
    % 数据预处理
    if size(data, 2) > size(data, 1)
        data = data';
    end
    
    [N, dim] = size(data);
    
    % 初始化专家系统变量
    evidence = struct();
    confidence_scores = [];
    
    fprintf('\n=== 专家系统分析 ===\n');
    
    % 规则1: 算法一致性检验
    [consistency_score, consistent_lambda] = check_algorithm_consistency(lambda_wolf, lambda_rosenstein);
    evidence.consistency = consistency_score;
    confidence_scores = [confidence_scores, consistency_score];
    
    fprintf('算法一致性: %.2f, 一致lambda: %.6f\n', consistency_score, consistent_lambda);
    
    % 规则2: 噪声水平评估
    if isempty(noise_level)
        noise_level = estimate_noise_level(data);
    end
    [noise_score, is_noisy] = assess_noise_level(data, noise_level);
    evidence.noise = noise_score;
    confidence_scores = [confidence_scores, noise_score];
    
    fprintf('噪声评估: %.2f (噪声水平: %.6f)\n', noise_score, noise_level);
    
    % 规则3: 周期性检测
    [periodicity_score, dominant_period] = detect_periodicity(data);
    evidence.periodicity = periodicity_score;
    confidence_scores = [confidence_scores, periodicity_score];
    
    fprintf('周期性检测: %.2f', periodicity_score);
    if ~isnan(dominant_period)
        fprintf(' (主周期: %.1f)', dominant_period);
    end
    fprintf('\n');
    
    % 规则4: 功率谱分析
    [spectrum_score, spectral_features] = analyze_power_spectrum(data);
    evidence.spectrum = spectrum_score;
    confidence_scores = [confidence_scores, spectrum_score];
    
    fprintf('功率谱分析: %.2f (谱特征: 宽带=%.2f)\n', spectrum_score, spectral_features.broadband_ratio);
    
    % 规则5: 相关维数估计
    if dim == 1 && N > 1000
        [correlation_score, corr_dim] = estimate_correlation_dimension(data);
        evidence.correlation_dim = correlation_score;
        confidence_scores = [confidence_scores, correlation_score];
        fprintf('相关维数: %.2f (D2≈%.2f)\n', correlation_score, corr_dim);
    else
        correlation_score = 0.5;  % 中性分数
        evidence.correlation_dim = correlation_score;
        confidence_scores = [confidence_scores, correlation_score];
        fprintf('相关维数: 跳过 (数据不足或多维)\n');
    end
    
    % 专家决策融合
    [final_lambda, confidence, decision] = expert_decision_fusion(evidence, consistent_lambda, threshold);
    
    % 输出详细分析
    fprintf('\n=== 专家系统决策 ===\n');
    fprintf('算法一致性: %.2f\n', evidence.consistency);
    fprintf('噪声评估: %.2f\n', evidence.noise);
    fprintf('周期性: %.2f\n', evidence.periodicity);
    fprintf('功率谱: %.2f\n', evidence.spectrum);
    fprintf('相关维数: %.2f\n', evidence.correlation_dim);
    fprintf('综合置信度: %.2f\n', confidence);
    fprintf('最终决策: %s\n', decision);
end

function [consistency_score, consistent_lambda] = check_algorithm_consistency(lambda_wolf, lambda_rosenstein)
% 检查两个算法结果的一致性
    valid_wolf = ~isnan(lambda_wolf) && isfinite(lambda_wolf);
    valid_rosenstein = ~isnan(lambda_rosenstein) && isfinite(lambda_rosenstein);
    
    if ~valid_wolf && ~valid_rosenstein
        consistency_score = 0;
        consistent_lambda = 0;
        return;
    elseif ~valid_wolf
        consistency_score = 0.3;
        consistent_lambda = lambda_rosenstein;
        return;
    elseif ~valid_rosenstein
        consistency_score = 0.3;
        consistent_lambda = lambda_wolf;
        return;
    end
    
    % 两个算法都有效
    diff = abs(lambda_wolf - lambda_rosenstein);
    avg_lambda = (lambda_wolf + lambda_rosenstein) / 2;
    
    % 相对误差
    if abs(avg_lambda) > 1e-6
        rel_error = diff / abs(avg_lambda);
    else
        rel_error = diff;
    end
    
    % 一致性评分
    if rel_error < 0.1
        consistency_score = 1.0;  % 非常一致
    elseif rel_error < 0.3
        consistency_score = 0.8;  % 比较一致
    elseif rel_error < 0.5
        consistency_score = 0.6;  % 中等一致
    elseif rel_error < 1.0
        consistency_score = 0.4;  % 一致性差
    else
        consistency_score = 0.2;  % 非常不一致
    end
    
    consistent_lambda = avg_lambda;
    
    % 如果两个结果符号不同，降低一致性
    if sign(lambda_wolf) ~= sign(lambda_rosenstein)
        consistency_score = consistency_score * 0.5;
    end
end

function [noise_score, is_noisy] = assess_noise_level(data, noise_level)
% 评估数据的噪声水平
    if size(data, 2) == 1
        signal = data;
    else
        signal = data(:, 1);  % 使用第一个分量
    end
    
    % 估计信号功率
    signal_power = var(signal);
    
    % 估计噪声功率（使用高频成分）
    if length(signal) > 100
        % 使用差分来估计噪声
        diff_signal = diff(signal);
        noise_power = var(diff_signal) / 2;  % 差分会放大噪声
    else
        noise_power = noise_level^2;
    end
    
    % 信噪比
    if noise_power > 0
        snr = signal_power / noise_power;
    else
        snr = Inf;
    end
    
    % 噪声评分（SNR越高，噪声评分越高）
    if snr > 100
        noise_score = 1.0;    % 低噪声
        is_noisy = false;
    elseif snr > 20
        noise_score = 0.8;    % 中等噪声
        is_noisy = false;
    elseif snr > 5
        noise_score = 0.6;    % 较高噪声
        is_noisy = true;
    elseif snr > 2
        noise_score = 0.4;    % 高噪声
        is_noisy = true;
    else
        noise_score = 0.2;    % 极高噪声
        is_noisy = true;
    end
end

function [periodicity_score, dominant_period] = detect_periodicity(data)
% 检测时间序列的周期性
    if size(data, 2) == 1
        signal = data;
    else
        signal = data(:, 1);
    end
    
    N = length(signal);
    dominant_period = NaN;
    
    % 自相关函数分析
    max_lag = min(N/4, 500);
    autocorr = zeros(max_lag, 1);
    
    signal_centered = signal - mean(signal);
    signal_var = var(signal_centered);
    
    for lag = 1:max_lag
        if N - lag > 10
            autocorr(lag) = sum(signal_centered(1:end-lag) .* signal_centered(1+lag:end)) / ((N-lag) * signal_var);
        end
    end
    
    % 寻找周期性峰值
    try
        % 使用findpeaks函数
        [peaks, locs] = findpeaks(autocorr, 'MinPeakHeight', 0.1, 'MinPeakDistance', 5);
        
        if ~isempty(peaks)
            [~, max_idx] = max(peaks);
            dominant_period = locs(max_idx);
            max_peak = peaks(max_idx);
            
            % 周期性评分
            if max_peak > 0.7
                periodicity_score = 0.2;  % 强周期性（非混沌）
            elseif max_peak > 0.5
                periodicity_score = 0.4;  % 中等周期性
            elseif max_peak > 0.3
                periodicity_score = 0.6;  % 弱周期性
            else
                periodicity_score = 0.8;  % 很弱周期性（可能混沌）
            end
        else
            periodicity_score = 1.0;  % 无明显周期性
        end
    catch
        % findpeaks函数出错时的备用方法
        max_autocorr = max(autocorr(5:end));  % 排除小滞后的自相关
        if max_autocorr > 0.5
            periodicity_score = 0.3;
        else
            periodicity_score = 0.8;
        end
    end
end

function [spectrum_score, spectral_features] = analyze_power_spectrum(data)
% 功率谱分析
    if size(data, 2) == 1
        signal = data;
    else
        signal = data(:, 1);
    end
    
    N = length(signal);
    
    % 计算功率谱密度
    [psd, freq] = periodogram(signal, [], [], 1);
    
    % 归一化功率谱
    psd_norm = psd / sum(psd);
    
    % 计算谱特征
    spectral_features = struct();
    
    % 1. 宽带比例（高频能量比例）
    high_freq_idx = freq > 0.1;  % 高频部分
    spectral_features.broadband_ratio = sum(psd_norm(high_freq_idx));
    
    % 2. 谱峰个数
    try
        [peaks, ~] = findpeaks(psd_norm, 'MinPeakHeight', max(psd_norm) * 0.1);
        spectral_features.num_peaks = length(peaks);
    catch
        spectral_features.num_peaks = 0;
    end
    
    % 3. 谱熵
    spectral_features.spectral_entropy = -sum(psd_norm .* log(psd_norm + eps));
    
    % 谱评分（宽带、高熵表示可能的混沌）
    if spectral_features.broadband_ratio > 0.7
        spectrum_score = 0.9;  % 宽带谱，可能混沌
    elseif spectral_features.broadband_ratio > 0.5
        spectrum_score = 0.7;
    elseif spectral_features.broadband_ratio > 0.3
        spectrum_score = 0.5;
    else
        spectrum_score = 0.3;  % 窄带谱，可能周期
    end
    
    % 考虑谱峰个数
    if spectral_features.num_peaks == 1
        spectrum_score = spectrum_score * 0.5;  % 单峰可能是周期
    elseif spectral_features.num_peaks <= 3
        spectrum_score = spectrum_score * 0.7;  % 少量峰可能是准周期
    end
end

function [correlation_score, corr_dim] = estimate_correlation_dimension(data)
% 估计相关维数
    if size(data, 2) > 1
        signal = data(:, 1);
    else
        signal = data;
    end
    
    % 相空间重构参数
    embed_dim = 5;
    delay = estimate_delay_simple(signal);
    
    % 相空间重构
    X = reconstruct_phase_space(signal, embed_dim, delay);
    [N, ~] = size(X);
    
    if N < 500
        correlation_score = 0.5;
        corr_dim = NaN;
        return;
    end
    
    % 计算不同距离下的相关积分
    data_std = std(X(:));
    r_values = data_std * logspace(-2, -0.5, 10);  % 距离范围
    C_values = zeros(size(r_values));
    
    % 随机采样以提高效率
    sample_size = min(1000, N);
    sample_idx = randperm(N, sample_size);
    X_sample = X(sample_idx, :);
    
    for i = 1:length(r_values)
        r = r_values(i);
        count = 0;
        total = 0;
        
        for j = 1:sample_size
            for k = j+1:sample_size
                total = total + 1;
                distance = sqrt(sum((X_sample(j, :) - X_sample(k, :)).^2));
                if distance < r
                    count = count + 1;
                end
            end
        end
        
        C_values(i) = count / total;
    end
    
    % 计算相关维数（线性拟合log(C) vs log(r)）
    valid_idx = C_values > 0;
    if sum(valid_idx) >= 5
        log_r = log(r_values(valid_idx));
        log_C = log(C_values(valid_idx));
        
        p = polyfit(log_r, log_C, 1);
        corr_dim = p(1);  % 斜率即为相关维数
        
        % 相关维数评分
        if corr_dim > 2.5
            correlation_score = 0.9;  % 高维，可能混沌
        elseif corr_dim > 2.0
            correlation_score = 0.7;
        elseif corr_dim > 1.5
            correlation_score = 0.5;
        else
            correlation_score = 0.3;  % 低维，可能周期
        end
    else
        correlation_score = 0.5;
        corr_dim = NaN;
    end
end

function [final_lambda, confidence, decision] = expert_decision_fusion(evidence, consistent_lambda, threshold)
% 专家决策融合
    % 权重设计
    weights = struct();
    weights.consistency = 0.3;    % 算法一致性权重
    weights.noise = 0.2;          % 噪声评估权重  
    weights.periodicity = 0.2;    % 周期性权重
    weights.spectrum = 0.2;       % 功率谱权重
    weights.correlation_dim = 0.1; % 相关维数权重
    
    % 计算加权置信度
    confidence = weights.consistency * evidence.consistency + ...
                weights.noise * evidence.noise + ...
                weights.periodicity * evidence.periodicity + ...
                weights.spectrum * evidence.spectrum + ...
                weights.correlation_dim * evidence.correlation_dim;
    
    % 决策逻辑
    if evidence.consistency < 0.3
        decision = '算法不一致，结果不可靠';
        final_lambda = consistent_lambda;
        confidence = confidence * 0.5;
    elseif evidence.noise < 0.4
        decision = '噪声过大，可能误判';
        final_lambda = 0;  % 认为是噪声
        confidence = confidence * 0.6;
    elseif evidence.periodicity < 0.4 && abs(consistent_lambda) < threshold
        decision = '稳定周期态';
        final_lambda = 0;
        confidence = confidence * 0.9;
    elseif consistent_lambda > threshold && confidence > 0.6
        decision = '混沌态';
        final_lambda = consistent_lambda;
    elseif consistent_lambda < -threshold && confidence > 0.6
        decision = '稳定不动点';
        final_lambda = consistent_lambda;
    elseif abs(consistent_lambda) <= threshold
        decision = '边界态或准周期态';
        final_lambda = consistent_lambda;
    else
        decision = '不确定状态';
        final_lambda = consistent_lambda;
        confidence = confidence * 0.7;
    end
    
    % 最终置信度调整
    confidence = max(0.1, min(1.0, confidence));
end

function delay = estimate_delay_simple(data)
% 简化的延时估计
    max_delay = min(20, floor(length(data)/10));
    delay = 1;
    
    if max_delay > 1
        autocorr_values = zeros(max_delay, 1);
        data_centered = data - mean(data);
        
        for tau = 1:max_delay
            if length(data) - tau > 10
                x1 = data_centered(1:end-tau);
                x2 = data_centered(1+tau:end);
                autocorr_values(tau) = abs(sum(x1 .* x2)) / sqrt(sum(x1.^2) * sum(x2.^2));
            end
        end
        
        % 找到第一个1/e点
        threshold = 1/exp(1);
        for tau = 2:length(autocorr_values)
            if autocorr_values(tau) < threshold
                delay = tau;
                break;
            end
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
        error('数据长度不足以进行相空间重构');
    end
    
    X = zeros(M, embed_dim);
    for i = 1:embed_dim
        start_idx = (i-1) * delay + 1;
        end_idx = start_idx + M - 1;
        X(:, i) = data(start_idx:end_idx);
    end
end