function lyapunov_example()
% Simple example demonstrating the improved Lyapunov exponent calculation
% This function shows how to use the main implementation with different datasets

    fprintf('=== LYAPUNOV EXPONENT CALCULATION EXAMPLE ===\n\n');
    
    % Example 1: Lorenz System
    fprintf('Example 1: Lorenz System (Chaotic)\n');
    fprintf('Expected λ ≈ 0.9\n');
    fprintf('----------------------------------\n');
    data_lorenz = generate_example_lorenz();
    run_example_analysis(data_lorenz, 'Lorenz System');
    
    % Example 2: Logistic Map
    fprintf('\nExample 2: Logistic Map (r=4.0)\n');
    fprintf('Expected λ ≈ 0.693\n');
    fprintf('---------------------------\n');
    data_logistic = generate_example_logistic();
    run_example_analysis(data_logistic, 'Logistic Map');
    
    % Example 3: Periodic System (for comparison)
    fprintf('\nExample 3: Sine Wave (Periodic)\n');
    fprintf('Expected λ ≈ 0 (negative or zero)\n');
    fprintf('---------------------------\n');
    data_sine = generate_example_sine();
    run_example_analysis(data_sine, 'Sine Wave');
    
    fprintf('\n=== EXAMPLES COMPLETE ===\n');
    fprintf('For full analysis, run: Copy_3_of_hundun1()\n');
end

function data = generate_example_lorenz()
% Generate Lorenz system data
    sigma = 10; rho = 28; beta = 8/3;
    dt = 0.01; t_end = 50; % Longer series for better accuracy
    t = 0:dt:t_end;
    
    x0 = [1; 1; 1];
    [~, sol] = ode45(@(t,x) lorenz_eqs(t,x,sigma,rho,beta), t, x0);
    data = sol(:,1); % x-component
end

function dxdt = lorenz_eqs(~, x, sigma, rho, beta)
    dxdt = [sigma*(x(2)-x(1)); 
            x(1)*(rho-x(3))-x(2); 
            x(1)*x(2)-beta*x(3)];
end

function data = generate_example_logistic()
% Generate logistic map data
    r = 4.0;
    n = 5000;
    data = zeros(n, 1);
    x = 0.1;
    
    for i = 1:n
        x = r * x * (1 - x);
        data(i) = x;
    end
end

function data = generate_example_sine()
% Generate sine wave (periodic system)
    t = linspace(0, 20*pi, 2000);
    data = sin(t)' + 0.1*sin(3*t)'; % Add some complexity
end

function run_example_analysis(data, system_name)
% Run simplified analysis for examples
    fprintf('Analyzing %s (%d points)...\n', system_name, length(data));
    
    % Simple parameter estimation
    m = estimate_embedding_dim_simple(data);
    tau = estimate_delay_simple(data);
    
    fprintf('  Estimated parameters: m=%d, τ=%d\n', m, tau);
    
    % Calculate Lyapunov exponent using simplified Rosenstein method
    try
        lambda = calculate_lyapunov_simple(data, m, tau);
        fprintf('  Lyapunov exponent: λ = %.6f\n', lambda);
        
        % Interpret result
        if lambda > 0.01
            fprintf('  → System appears CHAOTIC\n');
        elseif lambda > -0.01
            fprintf('  → System appears at EDGE OF CHAOS\n');
        else
            fprintf('  → System appears STABLE/PERIODIC\n');
        end
        
    catch e
        fprintf('  ERROR: %s\n', e.message);
    end
end

function m = estimate_embedding_dim_simple(data)
% Simple embedding dimension estimation
    % Use rule of thumb: m should be > 2*D where D is fractal dimension
    % For most chaotic systems, m=3-5 works well
    n = length(data);
    if n > 1000
        m = 3;
    else
        m = 2;
    end
end

function tau = estimate_delay_simple(data)
% Simple delay estimation using autocorrelation first zero
    max_tau = min(100, floor(length(data)/10));
    autocorr_vals = simple_autocorr_example(data, max_tau);
    
    % Find first zero crossing
    tau = 1;
    for i = 2:length(autocorr_vals)
        if autocorr_vals(i) <= 0
            tau = i;
            break;
        end
    end
    
    % Ensure reasonable bounds
    tau = max(1, min(tau, max_tau));
end

function autocorr_vals = simple_autocorr_example(data, max_lag)
% Simple autocorrelation
    data = data(:) - mean(data);
    N = length(data);
    autocorr_vals = zeros(max_lag + 1, 1);
    
    for lag = 0:max_lag
        if lag == 0
            autocorr_vals(lag + 1) = 1;
        else
            if N - lag > 0
                c = mean(data(1:N-lag) .* data(lag+1:N));
                autocorr_vals(lag + 1) = c / var(data);
            else
                autocorr_vals(lag + 1) = 0;
            end
        end
    end
end

function lambda = calculate_lyapunov_simple(data, m, tau)
% Simplified Lyapunov calculation
    
    % Phase space reconstruction
    N_data = length(data);
    N = N_data - (m-1)*tau;
    
    if N < 50
        error('Insufficient data for analysis');
    end
    
    phase_space = zeros(N, m);
    for i = 1:m
        phase_space(:, i) = data((i-1)*tau + 1 : (i-1)*tau + N);
    end
    
    % Find nearest neighbors and track divergence
    max_evolution = min(50, floor(N/10));
    distances = [];
    evolution_times = [];
    
    for i = 1:N-max_evolution
        % Find nearest neighbor (with temporal separation)
        min_separation = max(10, tau);
        candidates = (i+min_separation):N-max_evolution;
        
        if isempty(candidates)
            continue;
        end
        
        dists = sqrt(sum((phase_space(candidates,:) - ...
                        repmat(phase_space(i,:), length(candidates), 1)).^2, 2));
        
        [min_dist, min_idx] = min(dists);
        nearest_neighbor = candidates(min_idx);
        
        % Track evolution
        for dt = 1:max_evolution
            if i+dt <= N && nearest_neighbor+dt <= N
                evolved_dist = sqrt(sum((phase_space(i+dt,:) - ...
                                       phase_space(nearest_neighbor+dt,:)).^2));
                
                if evolved_dist > 0 && min_dist > 0
                    distances(end+1) = log(evolved_dist / min_dist);
                    evolution_times(end+1) = dt;
                end
            end
        end
    end
    
    if length(distances) < 10
        error('Insufficient neighbor pairs found');
    end
    
    % Simple linear fit
    unique_times = unique(evolution_times);
    mean_distances = zeros(size(unique_times));
    
    for i = 1:length(unique_times)
        t = unique_times(i);
        idx = evolution_times == t;
        mean_distances(i) = mean(distances(idx));
    end
    
    % Linear regression (least squares)
    X = [ones(length(unique_times), 1), unique_times(:)];
    beta = X \ mean_distances(:);
    lambda = beta(2); % Slope is Lyapunov exponent
    
    % Validation
    if lambda < -1 || lambda > 5
        warning('Questionable result: λ = %.6f', lambda);
    end
end