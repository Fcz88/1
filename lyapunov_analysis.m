function lyapunov_analysis()
% LYAPUNOV_ANALYSIS 主函数：Lyapunov指数计算和混沌分析
% 解决数值稳定性问题，提供多算法交叉验证
% 
% 功能：
% 1. 多种Lyapunov指数计算算法
% 2. 数值稳定性处理
% 3. 专家决策系统
% 4. 可视化分析
%
% 作者：Lyapunov Analysis System
% 日期：2024

    fprintf('=== Lyapunov指数分析系统 ===\n');
    fprintf('1. Lorenz系统分析\n');
    fprintf('2. Henon映射分析\n');
    fprintf('3. 自定义系统分析\n');
    fprintf('4. 批量参数扫描\n');
    fprintf('请选择分析类型 (1-4): ');
    
    choice = input('');
    if isempty(choice)
        choice = 1;
    end
    
    switch choice
        case 1
            analyze_lorenz_system();
        case 2
            analyze_henon_map();
        case 3
            analyze_custom_system();
        case 4
            parameter_sweep_analysis();
        otherwise
            fprintf('无效选择，默认分析Lorenz系统\n');
            analyze_lorenz_system();
    end
    
    fprintf('\n分析完成！\n');
end

function analyze_lorenz_system()
% 分析Lorenz系统的Lyapunov指数
    fprintf('\n--- Lorenz系统分析 ---\n');
    
    % 默认参数
    sigma = 10;
    beta = 8/3;
    rho = 28;
    
    % 初始条件
    x0 = [1, 1, 1];
    
    % 时间参数
    dt = 0.01;
    N = 10000;  % 数据点数
    
    fprintf('参数: sigma=%.2f, beta=%.4f, rho=%.2f\n', sigma, beta, rho);
    fprintf('初始条件: [%.2f, %.2f, %.2f]\n', x0);
    fprintf('时间步长: %.4f, 数据点数: %d\n', dt, N);
    
    % 生成时间序列
    [t, x] = lorenz_system(x0, [sigma, beta, rho], dt, N);
    
    % 多算法计算Lyapunov指数
    fprintf('\n计算Lyapunov指数...\n');
    
    % Wolf算法
    try
        lambda_wolf = wolf_lyapunov(x, dt);
        fprintf('Wolf算法结果: %.6f\n', lambda_wolf);
    catch ME
        fprintf('Wolf算法出错: %s\n', ME.message);
        lambda_wolf = NaN;
    end
    
    % Rosenstein算法
    try
        lambda_rosenstein = rosenstein_lyapunov(x, dt);
        fprintf('Rosenstein算法结果: %.6f\n', lambda_rosenstein);
    catch ME
        fprintf('Rosenstein算法出错: %s\n', ME.message);
        lambda_rosenstein = NaN;
    end
    
    % 专家系统决策
    [final_lambda, confidence, decision] = chaos_expert_system(lambda_wolf, lambda_rosenstein, x);
    
    fprintf('\n--- 专家系统分析结果 ---\n');
    fprintf('最终Lyapunov指数: %.6f\n', final_lambda);
    fprintf('置信度: %.2f%%\n', confidence * 100);
    fprintf('系统状态: %s\n', decision);
    
    % 可视化
    plot_results(t, x, final_lambda, decision);
    
    % Poincaré截面分析
    poincare_section(x);
end

function analyze_henon_map()
% 分析Henon映射的Lyapunov指数
    fprintf('\n--- Henon映射分析 ---\n');
    
    % 默认参数
    a = 1.4;
    b = 0.3;
    
    % 初始条件
    x0 = [0.1, 0.1];
    
    % 迭代次数
    N = 10000;
    
    fprintf('参数: a=%.2f, b=%.2f\n', a, b);
    fprintf('初始条件: [%.2f, %.2f]\n', x0);
    fprintf('迭代次数: %d\n', N);
    
    % 生成时间序列
    x = henon_map(x0, [a, b], N);
    
    % 计算Lyapunov指数（离散系统）
    fprintf('\n计算Lyapunov指数...\n');
    
    % 使用修改的算法处理离散系统
    try
        lambda_discrete = discrete_lyapunov(x, [a, b]);
        fprintf('离散系统Lyapunov指数: %.6f\n', lambda_discrete);
    catch ME
        fprintf('离散系统算法出错: %s\n', ME.message);
        lambda_discrete = NaN;
    end
    
    % 可视化
    figure('Name', 'Henon映射分析');
    subplot(2,2,1);
    plot(x(:,1), x(:,2), '.', 'MarkerSize', 1);
    title('Henon吸引子');
    xlabel('x');
    ylabel('y');
    grid on;
    
    subplot(2,2,2);
    plot(x(:,1));
    title('x时间序列');
    xlabel('迭代次数');
    ylabel('x');
    grid on;
    
    subplot(2,2,3);
    plot(x(:,2));
    title('y时间序列');
    xlabel('迭代次数');
    ylabel('y');
    grid on;
    
    subplot(2,2,4);
    text(0.1, 0.7, sprintf('Lyapunov指数: %.6f', lambda_discrete), 'FontSize', 12);
    if lambda_discrete > 0
        text(0.1, 0.5, '系统状态: 混沌', 'FontSize', 12, 'Color', 'red');
    elseif lambda_discrete < 0
        text(0.1, 0.5, '系统状态: 稳定', 'FontSize', 12, 'Color', 'blue');
    else
        text(0.1, 0.5, '系统状态: 边界', 'FontSize', 12, 'Color', 'orange');
    end
    axis off;
end

function analyze_custom_system()
% 分析用户自定义系统
    fprintf('\n--- 自定义系统分析 ---\n');
    fprintf('请输入时间序列数据文件名（.mat或.txt格式）: ');
    filename = input('', 's');
    
    if isempty(filename)
        fprintf('使用默认测试数据...\n');
        % 生成测试数据
        t = 0:0.01:100;
        x = [sin(t') + 0.1*randn(length(t),1), cos(t') + 0.1*randn(length(t),1)];
        dt = 0.01;
    else
        try
            if endsWith(filename, '.mat')
                data = load(filename);
                fields = fieldnames(data);
                x = data.(fields{1});
            else
                x = load(filename);
            end
            dt = 0.01;  % 默认时间步长
        catch
            fprintf('文件读取失败，使用默认测试数据\n');
            t = 0:0.01:100;
            x = [sin(t') + 0.1*randn(length(t),1), cos(t') + 0.1*randn(length(t),1)];
            dt = 0.01;
        end
    end
    
    fprintf('数据大小: %d x %d\n', size(x,1), size(x,2));
    
    % 多算法分析
    try
        lambda_wolf = wolf_lyapunov(x, dt);
        lambda_rosenstein = rosenstein_lyapunov(x, dt);
        
        [final_lambda, confidence, decision] = chaos_expert_system(lambda_wolf, lambda_rosenstein, x);
        
        fprintf('\nWolf算法: %.6f\n', lambda_wolf);
        fprintf('Rosenstein算法: %.6f\n', lambda_rosenstein);
        fprintf('专家系统结果: %.6f (置信度: %.2f%%)\n', final_lambda, confidence*100);
        fprintf('系统状态: %s\n', decision);
        
    catch ME
        fprintf('分析出错: %s\n', ME.message);
    end
end

function parameter_sweep_analysis()
% 参数扫描分析
    fprintf('\n--- 参数扫描分析 ---\n');
    fprintf('分析Lorenz系统的rho参数变化对Lyapunov指数的影响\n');
    
    rho_range = 10:0.5:30;
    lambdas = zeros(size(rho_range));
    
    sigma = 10;
    beta = 8/3;
    x0 = [1, 1, 1];
    dt = 0.01;
    N = 5000;
    
    fprintf('参数范围: rho = %.1f 到 %.1f\n', min(rho_range), max(rho_range));
    
    for i = 1:length(rho_range)
        rho = rho_range(i);
        
        try
            [~, x] = lorenz_system(x0, [sigma, beta, rho], dt, N);
            lambda_wolf = wolf_lyapunov(x, dt);
            lambda_rosenstein = rosenstein_lyapunov(x, dt);
            
            % 取两个算法的平均值
            if ~isnan(lambda_wolf) && ~isnan(lambda_rosenstein)
                lambdas(i) = (lambda_wolf + lambda_rosenstein) / 2;
            elseif ~isnan(lambda_wolf)
                lambdas(i) = lambda_wolf;
            elseif ~isnan(lambda_rosenstein)
                lambdas(i) = lambda_rosenstein;
            else
                lambdas(i) = NaN;
            end
            
        catch
            lambdas(i) = NaN;
        end
        
        if mod(i, 5) == 0
            fprintf('完成 %d/%d\n', i, length(rho_range));
        end
    end
    
    % 绘制分岔图
    figure('Name', '参数扫描结果');
    subplot(2,1,1);
    plot(rho_range, lambdas, 'b.-', 'LineWidth', 1.5);
    xlabel('rho');
    ylabel('最大Lyapunov指数');
    title('Lorenz系统参数扫描');
    grid on;
    hold on;
    plot(rho_range, zeros(size(rho_range)), 'r--', 'LineWidth', 1);
    legend('Lyapunov指数', '零线', 'Location', 'best');
    
    subplot(2,1,2);
    chaos_regions = lambdas > 0;
    bar(rho_range, chaos_regions, 'r');
    xlabel('rho');
    ylabel('混沌状态 (1=混沌, 0=非混沌)');
    title('混沌区域识别');
    grid on;
    
    fprintf('\n参数扫描完成！\n');
    fprintf('混沌参数区域: rho ∈ [%.1f, %.1f]\n', min(rho_range(chaos_regions)), max(rho_range(chaos_regions)));
end