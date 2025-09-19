function demo_lyapunov()
% DEMO_LYAPUNOV Lyapunov指数分析系统演示程序
% 展示完整的MATLAB Lyapunov指数计算功能
%
% 功能演示:
% 1. Lorenz系统混沌分析
% 2. Henon映射分析  
% 3. 数值稳定性验证
% 4. 专家系统决策
% 5. 可视化展示

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('  MATLAB Lyapunov指数分析系统演示\n');
    fprintf('========================================\n');
    fprintf('作者: Lyapunov Analysis Team\n');
    fprintf('版本: 1.0\n');
    fprintf('日期: 2024\n\n');
    
    % 演示菜单
    while true
        fprintf('请选择演示内容:\n');
        fprintf('1. Lorenz系统混沌分析\n');
        fprintf('2. Henon映射分析\n');
        fprintf('3. 参数对比研究\n');
        fprintf('4. 数值稳定性测试\n');
        fprintf('5. 专家系统演示\n');
        fprintf('6. 批量测试验证\n');
        fprintf('0. 退出\n');
        fprintf('请输入选择 (0-6): ');
        
        choice = input('');
        if isempty(choice)
            choice = 1;
        end
        
        switch choice
            case 0
                fprintf('演示结束，谢谢使用！\n');
                break;
            case 1
                demo_lorenz_chaos();
            case 2
                demo_henon_map();
            case 3
                demo_parameter_study();
            case 4
                demo_numerical_stability();
            case 5
                demo_expert_system();
            case 6
                demo_batch_validation();
            otherwise
                fprintf('无效选择，请重新输入。\n');
        end
        
        if choice ~= 0
            fprintf('\n按任意键继续...\n');
            pause;
            fprintf('\n');
        end
    end
end

function demo_lorenz_chaos()
% 演示Lorenz系统的混沌行为
    fprintf('\n=== Lorenz系统混沌分析演示 ===\n');
    
    % 经典混沌参数
    sigma = 10;
    beta = 8/3;
    rho = 28;
    
    x0 = [1, 1, 1];
    dt = 0.01;
    N = 8000;
    
    fprintf('系统参数: σ=%.1f, β=%.4f, ρ=%.1f\n', sigma, beta, rho);
    fprintf('初始条件: [%.1f, %.1f, %.1f]\n', x0);
    fprintf('积分参数: dt=%.3f, N=%d\n', dt, N);
    
    % 生成时间序列
    fprintf('\n正在生成Lorenz轨道...\n');
    [t, x] = lorenz_system(x0, [sigma, beta, rho], dt, N);
    
    % 计算Lyapunov指数
    fprintf('正在计算Lyapunov指数...\n');
    
    tic;
    lambda_wolf = wolf_lyapunov(x, dt);
    time_wolf = toc;
    
    tic;
    lambda_rosenstein = rosenstein_lyapunov(x, dt);
    time_rosenstein = toc;
    
    % 专家系统分析
    [final_lambda, confidence, decision] = chaos_expert_system(lambda_wolf, lambda_rosenstein, x);
    
    % 显示结果
    fprintf('\n=== 分析结果 ===\n');
    fprintf('Wolf算法:       λ = %.6f (耗时: %.2fs)\n', lambda_wolf, time_wolf);
    fprintf('Rosenstein算法: λ = %.6f (耗时: %.2fs)\n', lambda_rosenstein, time_rosenstein);
    fprintf('专家系统:       λ = %.6f (置信度: %.1f%%)\n', final_lambda, confidence*100);
    fprintf('系统状态:       %s\n', decision);
    
    % 理论值比较
    theoretical_lambda = 0.906;  % Lorenz系统的理论值
    error_wolf = abs(lambda_wolf - theoretical_lambda);
    error_rosenstein = abs(lambda_rosenstein - theoretical_lambda);
    error_expert = abs(final_lambda - theoretical_lambda);
    
    fprintf('\n=== 与理论值比较 (理论值 ≈ %.3f) ===\n', theoretical_lambda);
    fprintf('Wolf误差:       %.6f (相对误差: %.2f%%)\n', error_wolf, error_wolf/theoretical_lambda*100);
    fprintf('Rosenstein误差: %.6f (相对误差: %.2f%%)\n', error_rosenstein, error_rosenstein/theoretical_lambda*100);
    fprintf('专家系统误差:   %.6f (相对误差: %.2f%%)\n', error_expert, error_expert/theoretical_lambda*100);
    
    % 可视化
    plot_results(t, x, final_lambda, decision);
    
    % Poincaré截面分析
    fprintf('\n正在进行Poincaré截面分析...\n');
    poincare_section(x, 'SectionPlane', [3, 27]);  % z = 27截面
end

function demo_henon_map()
% 演示Henon映射分析
    fprintf('\n=== Henon映射分析演示 ===\n');
    
    % 经典混沌参数
    a = 1.4;
    b = 0.3;
    x0 = [0.1, 0.1];
    N = 10000;
    
    fprintf('系统参数: a=%.2f, b=%.2f\n', a, b);
    fprintf('初始条件: [%.2f, %.2f]\n', x0);
    fprintf('迭代次数: %d\n', N);
    
    % 生成Henon轨道
    fprintf('\n正在生成Henon轨道...\n');
    x = henon_map(x0, [a, b], N);
    
    % 计算Lyapunov指数
    fprintf('正在计算离散系统Lyapunov指数...\n');
    lambda_discrete = discrete_lyapunov(x, [a, b]);
    
    % 显示结果
    fprintf('\n=== 分析结果 ===\n');
    fprintf('离散Lyapunov指数: λ = %.6f\n', lambda_discrete);
    
    % 理论值比较
    theoretical_lambda = 0.419;  % Henon映射的理论值
    error = abs(lambda_discrete - theoretical_lambda);
    
    fprintf('理论值:          λ ≈ %.3f\n', theoretical_lambda);
    fprintf('计算误差:        %.6f (相对误差: %.2f%%)\n', error, error/theoretical_lambda*100);
    
    if lambda_discrete > 0
        fprintf('系统状态:        混沌\n');
    else
        fprintf('系统状态:        非混沌\n');
    end
    
    % 可视化Henon吸引子
    figure('Name', 'Henon映射演示');
    
    subplot(2,3,1);
    plot(x(:,1), x(:,2), '.', 'MarkerSize', 1, 'Color', [0.8, 0.2, 0.2]);
    title('Henon吸引子');
    xlabel('x');
    ylabel('y');
    grid on;
    axis equal;
    
    subplot(2,3,2);
    plot(x(1:1000,1), 'b-', 'LineWidth', 1);
    title('x时间序列 (前1000点)');
    xlabel('n');
    ylabel('x_n');
    grid on;
    
    subplot(2,3,3);
    plot(x(1:end-1,1), x(2:end,1), '.', 'MarkerSize', 2, 'Color', [0.2, 0.8, 0.2]);
    title('返回映射 x_{n+1} vs x_n');
    xlabel('x_n');
    ylabel('x_{n+1}');
    grid on;
    
    subplot(2,3,4);
    [psd, freq] = periodogram(x(:,1));
    semilogy(freq, psd, 'r-', 'LineWidth', 1.5);
    title('功率谱密度');
    xlabel('归一化频率');
    ylabel('PSD');
    grid on;
    
    subplot(2,3,5);
    hist(x(:,1), 50);
    title('x分量直方图');
    xlabel('x');
    ylabel('频数');
    grid on;
    
    subplot(2,3,6);
    text(0.1, 0.8, sprintf('Lyapunov指数: %.6f', lambda_discrete), 'FontSize', 12);
    text(0.1, 0.6, sprintf('理论值: %.3f', theoretical_lambda), 'FontSize', 12);
    text(0.1, 0.4, sprintf('误差: %.6f', error), 'FontSize', 12);
    if lambda_discrete > 0
        text(0.1, 0.2, '系统状态: 混沌', 'FontSize', 12, 'Color', 'red');
    else
        text(0.1, 0.2, '系统状态: 非混沌', 'FontSize', 12, 'Color', 'blue');
    end
    axis off;
end

function demo_parameter_study()
% 演示参数对Lyapunov指数的影响
    fprintf('\n=== 参数对比研究演示 ===\n');
    
    % Lorenz系统参数扫描
    fprintf('研究Lorenz系统rho参数的影响...\n');
    
    rho_values = [15, 20, 24.74, 28, 35];  % 包括分岔点
    results = [];
    
    sigma = 10;
    beta = 8/3;
    x0 = [1, 1, 1];
    dt = 0.01;
    N = 5000;
    
    fprintf('\n参数    | Wolf算法  | Rosenstein | 专家系统  | 状态\n');
    fprintf('--------|-----------|------------|-----------|----------\n');
    
    for i = 1:length(rho_values)
        rho = rho_values(i);
        
        % 生成数据
        [~, x] = lorenz_system(x0, [sigma, beta, rho], dt, N);
        
        % 计算Lyapunov指数
        try
            lambda_wolf = wolf_lyapunov(x, dt);
        catch
            lambda_wolf = NaN;
        end
        
        try
            lambda_rosenstein = rosenstein_lyapunov(x, dt);
        catch
            lambda_rosenstein = NaN;
        end
        
        [final_lambda, confidence, decision] = chaos_expert_system(lambda_wolf, lambda_rosenstein, x);
        
        results = [results; rho, lambda_wolf, lambda_rosenstein, final_lambda, confidence];
        
        fprintf('rho=%.2f | %9.6f | %10.6f | %9.6f | %s\n', ...
                rho, lambda_wolf, lambda_rosenstein, final_lambda, decision);
    end
    
    % 绘制参数研究结果
    figure('Name', '参数对比研究');
    
    subplot(2,1,1);
    plot(results(:,1), results(:,2), 'bo-', 'DisplayName', 'Wolf算法', 'LineWidth', 1.5);
    hold on;
    plot(results(:,1), results(:,3), 'rs-', 'DisplayName', 'Rosenstein算法', 'LineWidth', 1.5);
    plot(results(:,1), results(:,4), 'g^-', 'DisplayName', '专家系统', 'LineWidth', 1.5);
    plot(results(:,1), zeros(size(results(:,1))), 'k--', 'DisplayName', '零线');
    xlabel('rho参数');
    ylabel('Lyapunov指数');
    title('Lorenz系统参数研究');
    legend('Location', 'best');
    grid on;
    
    subplot(2,1,2);
    bar(results(:,1), results(:,5));
    xlabel('rho参数');
    ylabel('专家系统置信度');
    title('专家系统置信度');
    grid on;
end

function demo_numerical_stability()
% 演示数值稳定性测试
    fprintf('\n=== 数值稳定性测试演示 ===\n');
    
    % 测试不同的初始条件
    fprintf('测试初始条件敏感性...\n');
    
    sigma = 10; beta = 8/3; rho = 28;
    dt = 0.01; N = 5000;
    
    % 多个相近的初始条件
    x0_base = [1, 1, 1];
    perturbations = [0, 1e-6, 1e-5, 1e-4, 1e-3];
    
    lambdas_wolf = [];
    lambdas_rosenstein = [];
    
    fprintf('\n扰动量    | Wolf算法  | Rosenstein | 差异\n');
    fprintf('----------|-----------|------------|----------\n');
    
    for i = 1:length(perturbations)
        pert = perturbations(i);
        x0 = x0_base + [pert, 0, 0];
        
        [~, x] = lorenz_system(x0, [sigma, beta, rho], dt, N);
        
        lambda_w = wolf_lyapunov(x, dt);
        lambda_r = rosenstein_lyapunov(x, dt);
        
        lambdas_wolf = [lambdas_wolf, lambda_w];
        lambdas_rosenstein = [lambdas_rosenstein, lambda_r];
        
        if i == 1
            fprintf('%9.0e | %9.6f | %10.6f | --------\n', pert, lambda_w, lambda_r);
        else
            diff_w = abs(lambda_w - lambdas_wolf(1));
            diff_r = abs(lambda_r - lambdas_rosenstein(1));
            fprintf('%9.0e | %9.6f | %10.6f | %.2e\n', pert, lambda_w, lambda_r, max(diff_w, diff_r));
        end
    end
    
    % 测试不同时间步长
    fprintf('\n测试时间步长影响...\n');
    dt_values = [0.001, 0.005, 0.01, 0.02, 0.05];
    x0 = [1, 1, 1];
    
    fprintf('\n时间步长  | Wolf算法  | Rosenstein | 计算时间\n');
    fprintf('----------|-----------|------------|----------\n');
    
    for dt_test = dt_values
        N_test = round(50 / dt_test);  % 保持总时间50秒
        
        [~, x] = lorenz_system(x0, [sigma, beta, rho], dt_test, N_test);
        
        tic;
        lambda_w = wolf_lyapunov(x, dt_test);
        time_w = toc;
        
        tic;
        lambda_r = rosenstein_lyapunov(x, dt_test);
        time_r = toc;
        
        fprintf('%9.3f | %9.6f | %10.6f | %.2f+%.2fs\n', ...
                dt_test, lambda_w, lambda_r, time_w, time_r);
    end
end

function demo_expert_system()
% 演示专家系统决策过程
    fprintf('\n=== 专家系统决策演示 ===\n');
    
    % 测试不同系统状态
    test_cases = {
        struct('name', 'Lorenz混沌', 'params', [10, 8/3, 28], 'x0', [1,1,1], 'type', 'lorenz')
        struct('name', 'Lorenz稳定', 'params', [10, 8/3, 5], 'x0', [1,1,1], 'type', 'lorenz')
        struct('name', '噪声信号', 'params', [], 'x0', [], 'type', 'noise')
        struct('name', '周期信号', 'params', [], 'x0', [], 'type', 'periodic')
    };
    
    dt = 0.01;
    N = 5000;
    
    for i = 1:length(test_cases)
        test_case = test_cases{i};
        fprintf('\n--- %s ---\n', test_case.name);
        
        % 生成测试数据
        switch test_case.type
            case 'lorenz'
                [~, x] = lorenz_system(test_case.x0, test_case.params, dt, N);
            case 'noise'
                t = (0:N-1) * dt;
                x = [randn(N,1), randn(N,1), randn(N,1)] * 0.5;
            case 'periodic'
                t = (0:N-1) * dt;
                x = [sin(2*pi*t'), cos(2*pi*t'), sin(4*pi*t')];
        end
        
        % 计算Lyapunov指数
        try
            lambda_wolf = wolf_lyapunov(x, dt);
        catch ME
            lambda_wolf = NaN;
            fprintf('Wolf算法出错: %s\n', ME.message);
        end
        
        try
            lambda_rosenstein = rosenstein_lyapunov(x, dt);
        catch ME
            lambda_rosenstein = NaN;
            fprintf('Rosenstein算法出错: %s\n', ME.message);
        end
        
        % 专家系统分析
        [final_lambda, confidence, decision] = chaos_expert_system(lambda_wolf, lambda_rosenstein, x);
        
        fprintf('结果: λ=%.6f, 置信度=%.2f%%, 决策=%s\n', ...
                final_lambda, confidence*100, decision);
    end
end

function demo_batch_validation()
% 批量验证测试
    fprintf('\n=== 批量验证测试演示 ===\n');
    
    % 创建已知结果的测试案例
    test_systems = {
        struct('name', 'Lorenz (ρ=28)', 'func', @() generate_lorenz([10, 8/3, 28]), 'expected', 0.906, 'tol', 0.2)
        struct('name', 'Lorenz (ρ=24)', 'func', @() generate_lorenz([10, 8/3, 24]), 'expected', 0.7, 'tol', 0.3)
        struct('name', 'Henon (a=1.4)', 'func', @() generate_henon([1.4, 0.3]), 'expected', 0.419, 'tol', 0.1)
        struct('name', '周期信号', 'func', @() generate_periodic(), 'expected', 0, 'tol', 0.05)
    };
    
    fprintf('\n系统           | 期望值   | Wolf     | Rosenstein | 专家系统 | 通过\n');
    fprintf('---------------|----------|----------|------------|----------|------\n');
    
    total_tests = length(test_systems);
    passed_tests = 0;
    
    for i = 1:total_tests
        test = test_systems{i};
        
        % 生成测试数据
        [x, dt] = test.func();
        
        % 计算Lyapunov指数
        try
            lambda_wolf = wolf_lyapunov(x, dt);
            if isnan(lambda_wolf)
                lambda_wolf = Inf;
            end
        catch
            lambda_wolf = Inf;
        end
        
        try
            lambda_rosenstein = rosenstein_lyapunov(x, dt);
            if isnan(lambda_rosenstein)
                lambda_rosenstein = Inf;
            end
        catch
            lambda_rosenstein = Inf;
        end
        
        [final_lambda, ~, ~] = chaos_expert_system(lambda_wolf, lambda_rosenstein, x);
        
        % 验证结果
        error_wolf = abs(lambda_wolf - test.expected);
        error_rosenstein = abs(lambda_rosenstein - test.expected);
        error_expert = abs(final_lambda - test.expected);
        
        pass_wolf = error_wolf <= test.tol;
        pass_rosenstein = error_rosenstein <= test.tol;
        pass_expert = error_expert <= test.tol;
        
        overall_pass = pass_expert;  % 以专家系统结果为准
        if overall_pass
            passed_tests = passed_tests + 1;
        end
        
        fprintf('%-14s | %8.3f | %8.3f | %10.3f | %8.3f | %s\n', ...
                test.name, test.expected, lambda_wolf, lambda_rosenstein, final_lambda, ...
                char('通过' * overall_pass + '失败' * ~overall_pass));
    end
    
    fprintf('\n=== 测试总结 ===\n');
    fprintf('总测试数: %d\n', total_tests);
    fprintf('通过数量: %d\n', passed_tests);
    fprintf('通过率: %.1f%%\n', passed_tests/total_tests*100);
    
    if passed_tests == total_tests
        fprintf('✓ 所有测试通过！系统工作正常。\n');
    else
        fprintf('⚠ 有测试失败，需要检查算法实现。\n');
    end
end

% 辅助函数
function [x, dt] = generate_lorenz(params)
    dt = 0.01;
    N = 5000;
    x0 = [1, 1, 1];
    [~, x] = lorenz_system(x0, params, dt, N);
end

function [x, dt] = generate_henon(params)
    dt = 1;  % 离散系统
    N = 5000;
    x0 = [0.1, 0.1];
    x = henon_map(x0, params, N);
end

function [x, dt] = generate_periodic()
    dt = 0.01;
    t = 0:dt:50;
    x = [sin(2*pi*t'), cos(2*pi*t'), sin(4*pi*t')];
end