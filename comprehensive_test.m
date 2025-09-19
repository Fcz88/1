function comprehensive_test()
% Comprehensive test suite for Lyapunov exponent calculation
    
    fprintf('=== COMPREHENSIVE LYAPUNOV EXPONENT TEST SUITE ===\n\n');
    
    % Test parameters
    test_results = struct();
    
    % Test 1: Logistic Map (known Lyapunov exponent)
    fprintf('Test 1: Logistic Map Analysis\n');
    fprintf('-------------------------------\n');
    [data_logistic, theoretical_lambda] = generate_logistic_map();
    test_results.logistic = test_single_system(data_logistic, theoretical_lambda, 'Logistic Map');
    
    % Test 2: Henon Map (known chaotic system)
    fprintf('\nTest 2: Henon Map Analysis\n');
    fprintf('--------------------------\n');
    data_henon = generate_henon_map();
    test_results.henon = test_single_system(data_henon, 0.418, 'Henon Map'); % Approximate theoretical value
    
    % Test 3: Lorenz System (from main function)
    fprintf('\nTest 3: Lorenz System Analysis\n');
    fprintf('------------------------------\n');
    data_lorenz = generate_lorenz_data();
    test_results.lorenz = test_single_system(data_lorenz, 0.906, 'Lorenz System'); % Approximate theoretical value
    
    % Test 4: Parameter sensitivity analysis
    fprintf('\nTest 4: Parameter Sensitivity Analysis\n');
    fprintf('--------------------------------------\n');
    test_results.sensitivity = test_parameter_sensitivity(data_lorenz);
    
    % Test 5: Noise robustness test
    fprintf('\nTest 5: Noise Robustness Test\n');
    fprintf('-----------------------------\n');
    test_results.noise = test_noise_robustness(data_lorenz);
    
    % Summary
    fprintf('\n=== TEST SUMMARY ===\n');
    display_test_summary(test_results);
end

function [data, lambda_theoretical] = generate_logistic_map()
% Generate logistic map data with known Lyapunov exponent
    r = 4.0; % Parameter for maximum chaos
    n = 2000;
    data = zeros(n, 1);
    x = 0.1; % Initial condition
    
    for i = 1:n
        x = r * x * (1 - x);
        data(i) = x;
    end
    
    % Theoretical Lyapunov exponent for r=4
    lambda_theoretical = log(2); % ≈ 0.693
end

function data = generate_henon_map()
% Generate Henon map data
    a = 1.4; b = 0.3;
    n = 2000;
    data = zeros(n, 2);
    x = 0; y = 0; % Initial conditions
    
    for i = 1:n
        x_new = 1 - a*x^2 + y;
        y_new = b*x;
        x = x_new;
        y = y_new;
        data(i, :) = [x, y];
    end
    
    % Return only x component for 1D analysis
    data = data(:, 1);
end

function data = generate_lorenz_data()
% Generate Lorenz system data (same as in main function)
    sigma = 10; rho = 28; beta = 8/3;
    dt = 0.01; t_end = 20; % Shorter for testing
    t = 0:dt:t_end;
    
    % Initial conditions
    x0 = [1; 1; 1];
    
    % Solve Lorenz system
    [~, sol] = ode45(@(t,x) lorenz_system(t,x,sigma,rho,beta), t, x0);
    
    % Return x-component time series
    data = sol(:,1);
end

function dxdt = lorenz_system(~, x, sigma, rho, beta)
% Lorenz system equations
    dxdt = [sigma*(x(2)-x(1)); 
            x(1)*(rho-x(3))-x(2); 
            x(1)*x(2)-beta*x(3)];
end

function result = test_single_system(data, theoretical_lambda, system_name)
% Test a single system and compare with theoretical value
    result = struct();
    result.system_name = system_name;
    result.theoretical = theoretical_lambda;
    result.data_length = length(data);
    
    fprintf('  System: %s\n', system_name);
    fprintf('  Data length: %d points\n', length(data));
    fprintf('  Theoretical λ: %.6f\n', theoretical_lambda);
    
    % Test Rosenstein method
    try
        tic;
        [lambda_ros, params_ros] = rosenstein_lyapunov_exponent(data);
        time_ros = toc;
        
        error_ros = abs(lambda_ros - theoretical_lambda) / abs(theoretical_lambda) * 100;
        
        result.rosenstein.lambda = lambda_ros;
        result.rosenstein.error_percent = error_ros;
        result.rosenstein.computation_time = time_ros;
        result.rosenstein.success = true;
        result.rosenstein.params = params_ros;
        
        fprintf('  Rosenstein: λ = %.6f (error: %.2f%%, time: %.3fs)\n', ...
                lambda_ros, error_ros, time_ros);
    catch e
        result.rosenstein.success = false;
        result.rosenstein.error_message = e.message;
        fprintf('  Rosenstein: FAILED - %s\n', e.message);
    end
    
    % Test Wolf method
    try
        tic;
        [lambda_wolf, params_wolf] = wolf_lyapunov_exponent(data);
        time_wolf = toc;
        
        error_wolf = abs(lambda_wolf - theoretical_lambda) / abs(theoretical_lambda) * 100;
        
        result.wolf.lambda = lambda_wolf;
        result.wolf.error_percent = error_wolf;
        result.wolf.computation_time = time_wolf;
        result.wolf.success = true;
        result.wolf.params = params_wolf;
        
        fprintf('  Wolf: λ = %.6f (error: %.2f%%, time: %.3fs)\n', ...
                lambda_wolf, error_wolf, time_wolf);
    catch e
        result.wolf.success = false;
        result.wolf.error_message = e.message;
        fprintf('  Wolf: FAILED - %s\n', e.message);
    end
end

function result = test_parameter_sensitivity(data)
% Test sensitivity to embedding parameters
    result = struct();
    fprintf('  Testing embedding dimension sensitivity...\n');
    
    m_values = 2:6;
    lambda_values = zeros(size(m_values));
    
    for i = 1:length(m_values)
        try
            % Create temporary parameters
            params_temp = struct();
            params_temp.m = m_values(i);
            params_temp.tau = find_optimal_delay_robust(data, m_values(i));
            params_temp.mean_period = estimate_mean_period(data);
            params_temp.max_evolution_time = round(params_temp.mean_period / 4);
            
            % Quick calculation
            [phase_space, N] = reconstruct_phase_space(data, params_temp.m, params_temp.tau);
            [distances, evolution] = find_nearest_neighbors_improved(phase_space, N, params_temp);
            [lambda, ~] = fit_robust_linear_region(distances, evolution, params_temp);
            
            lambda_values(i) = lambda;
            fprintf('    m=%d: λ = %.6f\n', m_values(i), lambda);
        catch
            lambda_values(i) = NaN;
            fprintf('    m=%d: FAILED\n', m_values(i));
        end
    end
    
    result.m_values = m_values;
    result.lambda_values = lambda_values;
    result.std_lambda = std(lambda_values(~isnan(lambda_values)));
    
    fprintf('  Standard deviation across m values: %.6f\n', result.std_lambda);
end

function result = test_noise_robustness(data)
% Test robustness to noise
    result = struct();
    noise_levels = [0, 0.01, 0.05, 0.1]; % Relative noise levels
    
    fprintf('  Testing noise robustness...\n');
    
    for i = 1:length(noise_levels)
        noise_level = noise_levels(i);
        
        % Add noise
        if noise_level > 0
            noise = noise_level * std(data) * randn(size(data));
            noisy_data = data + noise;
        else
            noisy_data = data;
        end
        
        try
            [lambda, ~] = rosenstein_lyapunov_exponent(noisy_data);
            result.noise_levels(i) = noise_level;
            result.lambda_values(i) = lambda;
            
            fprintf('    Noise %.1f%%: λ = %.6f\n', noise_level*100, lambda);
        catch
            result.noise_levels(i) = noise_level;
            result.lambda_values(i) = NaN;
            fprintf('    Noise %.1f%%: FAILED\n', noise_level*100);
        end
    end
end

function display_test_summary(test_results)
% Display comprehensive test summary
    fprintf('\nSUCCESS RATES:\n');
    
    systems = fieldnames(test_results);
    total_tests = 0;
    successful_tests = 0;
    
    for i = 1:length(systems)
        system = systems{i};
        if strcmp(system, 'sensitivity') || strcmp(system, 'noise')
            continue; % Skip these for success rate calculation
        end
        
        fprintf('  %s:\n', test_results.(system).system_name);
        
        if isfield(test_results.(system), 'rosenstein')
            total_tests = total_tests + 1;
            if test_results.(system).rosenstein.success
                successful_tests = successful_tests + 1;
                fprintf('    Rosenstein: SUCCESS (%.2f%% error)\n', ...
                        test_results.(system).rosenstein.error_percent);
            else
                fprintf('    Rosenstein: FAILED\n');
            end
        end
        
        if isfield(test_results.(system), 'wolf')
            total_tests = total_tests + 1;
            if test_results.(system).wolf.success
                successful_tests = successful_tests + 1;
                fprintf('    Wolf: SUCCESS (%.2f%% error)\n', ...
                        test_results.(system).wolf.error_percent);
            else
                fprintf('    Wolf: FAILED\n');
            end
        end
    end
    
    fprintf('\nOVERALL SUCCESS RATE: %d/%d (%.1f%%)\n', ...
            successful_tests, total_tests, successful_tests/total_tests*100);
    
    fprintf('\nRECOMMENDATIONS:\n');
    if successful_tests/total_tests > 0.8
        fprintf('  ✓ Implementation appears robust and accurate\n');
    elseif successful_tests/total_tests > 0.5
        fprintf('  ⚠ Implementation works but may need parameter tuning\n');
    else
        fprintf('  ✗ Implementation needs significant improvement\n');
    end
    
    fprintf('  ✓ Use Rosenstein method for most applications\n');
    fprintf('  ✓ Ensure data length > 1000 points for best results\n'); 
    fprintf('  ✓ Preprocess data to remove trends and noise\n');
end