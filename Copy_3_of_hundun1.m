function Copy_3_of_hundun1()
% Improved Lyapunov Exponent Calculation with Multiple Methods
% This script provides various algorithms for computing Lyapunov exponents
% with enhanced accuracy and numerical stability

    clc; clear; close all;
    
    % Example time series data generation (Lorenz system)
    fprintf('Generating example data from Lorenz system...\n');
    data = generate_lorenz_data();
    
    % Display available methods
    fprintf('\n=== Available Lyapunov Exponent Calculation Methods ===\n');
    fprintf('1. Improved Rosenstein Method\n');
    fprintf('2. Wolf Method\n');
    fprintf('3. Multi-algorithm Comparison\n');
    
    % Calculate using multiple methods
    results = calculate_lyapunov_multi_method(data);
    
    % Display results
    display_results(results);
    
    % Plot comparison
    plot_comparison(results);
end

function data = generate_lorenz_data()
% Generate sample data from Lorenz system for testing
    sigma = 10; rho = 28; beta = 8/3;
    dt = 0.01; t_end = 100;
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

function results = calculate_lyapunov_multi_method(data)
% Calculate Lyapunov exponents using multiple methods
    
    fprintf('\n=== Starting Multi-Method Lyapunov Calculation ===\n');
    
    % Prepare results structure
    results = struct();
    
    % Method 1: Improved Rosenstein Method
    fprintf('Calculating with Improved Rosenstein Method...\n');
    try
        [lambda_ros, params_ros] = rosenstein_lyapunov_exponent(data);
        results.rosenstein.lambda = lambda_ros;
        results.rosenstein.params = params_ros;
        results.rosenstein.success = true;
        fprintf('Rosenstein Method: λ = %.6f\n', lambda_ros);
    catch ME
        results.rosenstein.success = false;
        results.rosenstein.error = ME.message;
        fprintf('Rosenstein Method failed: %s\n', ME.message);
    end
    
    % Method 2: Wolf Method
    fprintf('Calculating with Wolf Method...\n');
    try
        [lambda_wolf, params_wolf] = wolf_lyapunov_exponent(data);
        results.wolf.lambda = lambda_wolf;
        results.wolf.params = params_wolf;
        results.wolf.success = true;
        fprintf('Wolf Method: λ = %.6f\n', lambda_wolf);
    catch ME
        results.wolf.success = false;
        results.wolf.error = ME.message;
        fprintf('Wolf Method failed: %s\n', ME.message);
    end
    
    % Method 3: Average and confidence interval
    successful_methods = {};
    lambda_values = [];
    
    if results.rosenstein.success
        successful_methods{end+1} = 'Rosenstein';
        lambda_values(end+1) = results.rosenstein.lambda;
    end
    
    if results.wolf.success
        successful_methods{end+1} = 'Wolf';
        lambda_values(end+1) = results.wolf.lambda;
    end
    
    if ~isempty(lambda_values)
        results.combined.mean = mean(lambda_values);
        results.combined.std = std(lambda_values);
        results.combined.methods = successful_methods;
        results.combined.values = lambda_values;
        fprintf('Combined Result: λ = %.6f ± %.6f\n', ...
                results.combined.mean, results.combined.std);
    end
end

function [lambda, params] = rosenstein_lyapunov_exponent(data)
% Improved Rosenstein method for Lyapunov exponent calculation
% Enhanced with better parameter selection and numerical stability

    % Adaptive parameter selection
    params = struct();
    params.m = find_optimal_embedding_dimension(data);
    params.tau = find_optimal_delay_robust(data, params.m);
    params.mean_period = estimate_mean_period(data);
    params.max_evolution_time = round(params.mean_period / 4);
    
    fprintf('  Embedding dimension: %d\n', params.m);
    fprintf('  Delay time: %d\n', params.tau);
    fprintf('  Mean period: %.2f\n', params.mean_period);
    
    % Phase space reconstruction
    [phase_space, N] = reconstruct_phase_space(data, params.m, params.tau);
    
    % Find nearest neighbors with improved algorithm
    fprintf('  Finding nearest neighbors...\n');
    [distances, evolution] = find_nearest_neighbors_improved(phase_space, N, params);
    
    % Robust linear fitting
    fprintf('  Performing robust linear fitting...\n');
    [lambda, fit_info] = fit_robust_linear_region(distances, evolution, params);
    
    % Store additional information
    params.fit_info = fit_info;
    params.phase_space_size = N;
    
    % Validate result
    if lambda < 0 || lambda > 10 || isnan(lambda) || isinf(lambda)
        warning('Rosenstein method: Questionable result λ = %.6f', lambda);
    end
end

function [lambda, params] = wolf_lyapunov_exponent(data)
% Wolf et al. method for Lyapunov exponent calculation
% Implements the algorithm with periodic reorthogonalization

    params = struct();
    params.m = find_optimal_embedding_dimension(data);
    params.tau = find_optimal_delay_robust(data, params.m);
    params.evolve_time = 10; % Evolution time between reorthogonalizations
    params.min_separation = std(data) * 0.01; % Minimum separation threshold
    params.max_separation = std(data) * 0.1;  % Maximum separation threshold
    
    fprintf('  Wolf method parameters:\n');
    fprintf('    Embedding dimension: %d\n', params.m);
    fprintf('    Delay time: %d\n', params.tau);
    fprintf('    Evolution time: %d\n', params.evolve_time);
    
    % Phase space reconstruction
    [phase_space, N] = reconstruct_phase_space(data, params.m, params.tau);
    
    % Initialize
    lambda_sum = 0;
    num_replacements = 0;
    current_point = 1;
    
    fprintf('  Starting Wolf algorithm iterations...\n');
    
    while current_point + params.evolve_time < N
        % Find nearest neighbor
        remaining_points = N - current_point;
        if remaining_points <= 1
            break;
        end
        
        distances = sqrt(sum((phase_space(current_point+1:min(N, current_point+remaining_points),:) - ...
                            repmat(phase_space(current_point,:), ...
                            min(remaining_points, N-current_point), 1)).^2, 2));
        
        [min_dist, min_idx] = min(distances);
        min_idx = min_idx + current_point;
        
        % Check if separation is adequate and indices are valid
        if min_dist < params.min_separation || min_idx + params.evolve_time > N
            current_point = current_point + 1;
            continue;
        end
        
        % Evolve both points
        if current_point + params.evolve_time <= N && min_idx + params.evolve_time <= N
            evolved_dist = sqrt(sum((phase_space(current_point + params.evolve_time,:) - ...
                                   phase_space(min_idx + params.evolve_time,:)).^2));
            
            % Calculate local Lyapunov exponent
            if evolved_dist > 0 && min_dist > 0
                local_lambda = log(evolved_dist / min_dist) / params.evolve_time;
                lambda_sum = lambda_sum + local_lambda;
                num_replacements = num_replacements + 1;
            end
        end
        
        % Move to next point
        current_point = current_point + params.evolve_time;
        
        % Check for replacement if separation too large
        if evolved_dist > params.max_separation
            % Find new nearest neighbor for the evolved point
            current_point = current_point + params.evolve_time;
        end
    end
    
    if num_replacements > 0
        lambda = lambda_sum / num_replacements;
        params.num_replacements = num_replacements;
        fprintf('  Completed %d replacement steps\n', num_replacements);
    else
        lambda = NaN;
        warning('Wolf method: No valid replacement steps found');
    end
end

function m_opt = find_optimal_embedding_dimension(data)
% Find optimal embedding dimension using false nearest neighbors
    max_m = min(10, floor(length(data)/50)); % Reasonable upper bound
    fnn_threshold = 0.1; % False nearest neighbor threshold
    
    for m = 1:max_m
        tau = find_optimal_delay_robust(data, m);
        fnn_ratio = calculate_false_nearest_neighbors(data, m, tau);
        
        if fnn_ratio < fnn_threshold
            m_opt = m;
            return;
        end
    end
    
    m_opt = max_m; % Default if no optimal found
end

function fnn_ratio = calculate_false_nearest_neighbors(data, m, tau)
% Calculate false nearest neighbors ratio
    if m == 1
        fnn_ratio = 1; % Always need at least m=2
        return;
    end
    
    % Reconstruct in m and m+1 dimensions
    [phase_m, N_m] = reconstruct_phase_space(data, m, tau);
    [phase_m1, N_m1] = reconstruct_phase_space(data, m+1, tau);
    
    N = min(N_m, N_m1);
    false_neighbors = 0;
    total_neighbors = 0;
    
    Rtol = 15; % Tolerance for false neighbor detection
    
    for i = 1:min(N, 500) % Limit for computational efficiency
        % Find nearest neighbor in m dimensions
        distances_m = sqrt(sum((phase_m(i+1:N,:) - ...
                              repmat(phase_m(i,:), N-i, 1)).^2, 2));
        [~, nn_idx] = min(distances_m);
        nn_idx = nn_idx + i;
        
        if nn_idx <= N
            % Check if still nearest neighbor in m+1 dimensions
            dist_m = sqrt(sum((phase_m(i,:) - phase_m(nn_idx,:)).^2));
            dist_m1 = sqrt(sum((phase_m1(i,:) - phase_m1(nn_idx,:)).^2));
            
            if dist_m > 0
                ratio = abs(dist_m1 - dist_m) / dist_m;
                if ratio > Rtol
                    false_neighbors = false_neighbors + 1;
                end
                total_neighbors = total_neighbors + 1;
            end
        end
    end
    
    if total_neighbors > 0
        fnn_ratio = false_neighbors / total_neighbors;
    else
        fnn_ratio = 1;
    end
end

function tau_opt = find_optimal_delay_robust(data, m)
% Robust method to find optimal delay time using mutual information
% and autocorrelation analysis
    
    N = length(data);
    max_tau = min(50, floor(N/10)); % Reasonable upper bound
    
    % Method 1: First minimum of mutual information
    mi_values = zeros(1, max_tau);
    for tau = 1:max_tau
        mi_values(tau) = mutual_information(data, tau);
    end
    
    % Find first local minimum
    tau_mi = find_first_minimum(mi_values);
    
    % Method 2: First zero of autocorrelation
    autocorr_values = simple_autocorr(data, max_tau);
    tau_ac = find_first_zero_crossing(autocorr_values);
    
    % Method 3: 1/e point of autocorrelation
    tau_1e = find_1e_point(autocorr_values);
    
    % Choose the most reasonable value
    tau_candidates = [tau_mi, tau_ac, tau_1e];
    tau_candidates = tau_candidates(tau_candidates > 0 & tau_candidates <= max_tau);
    
    if isempty(tau_candidates)
        tau_opt = max(1, round(max_tau/4)); % Default fallback
    else
        tau_opt = round(median(tau_candidates));
    end
    
    % Ensure reasonable bounds
    tau_opt = max(1, min(tau_opt, max_tau));
end

function mi = mutual_information(data, tau)
% Calculate mutual information between data(t) and data(t+tau)
    N = length(data);
    if tau >= N
        mi = 0;
        return;
    end
    
    x = data(1:N-tau);
    y = data(tau+1:N);
    
    % Discretize data for histogram-based MI calculation
    nbins = max(10, min(50, floor(sqrt(length(x)))));
    
    % Create 2D histogram
    try
        [counts, ~, ~] = histcounts2(x, y, nbins);
    catch
        % Fallback for systems without histcounts2
        counts = simple_hist2d(x, y, nbins);
    end
    counts = counts + eps; % Avoid log(0)
    
    % Normalize to get probabilities
    p_xy = counts / sum(counts(:));
    p_x = sum(p_xy, 2);
    p_y = sum(p_xy, 1);
    
    % Calculate mutual information
    mi = 0;
    for i = 1:size(p_xy, 1)
        for j = 1:size(p_xy, 2)
            if p_xy(i,j) > 0
                mi = mi + p_xy(i,j) * log2(p_xy(i,j) / (p_x(i) * p_y(j)));
            end
        end
    end
end

function idx = find_first_minimum(values)
% Find first local minimum in a sequence
    idx = 1;
    for i = 2:length(values)-1
        if values(i) < values(i-1) && values(i) < values(i+1)
            idx = i;
            return;
        end
    end
end

function idx = find_first_zero_crossing(values)
% Find first zero crossing
    idx = 1;
    for i = 2:length(values)
        if values(i-1) > 0 && values(i) <= 0
            idx = i;
            return;
        end
    end
end

function idx = find_1e_point(values)
% Find where autocorrelation drops to 1/e of initial value
    if isempty(values) || values(1) <= 0
        idx = 1;
        return;
    end
    
    target = values(1) / exp(1);
    idx = 1;
    
    for i = 2:length(values)
        if values(i) <= target
            idx = i;
            return;
        end
    end
end

function mean_period = estimate_mean_period(data)
% Estimate mean period of the time series using autocorrelation
    N = length(data);
    max_lag = min(500, floor(N/4));
    
    % Calculate autocorrelation
    autocorr_vals = simple_autocorr(data, max_lag);
    
    % Find peaks (potential periods)
    [peaks, locs] = simple_findpeaks(autocorr_vals(2:end)); % Skip lag 0
    locs = locs + 1; % Adjust for skipped first element
    
    if isempty(locs)
        mean_period = max_lag / 4; % Default estimate
    else
        % Use the location of the highest peak as period estimate
        [~, max_peak_idx] = max(peaks);
        mean_period = locs(max_peak_idx);
    end
end

function [phase_space, N] = reconstruct_phase_space(data, m, tau)
% Reconstruct phase space using time delay embedding
    data = data(:); % Ensure column vector
    N_data = length(data);
    N = N_data - (m-1)*tau;
    
    if N <= 0
        error('Insufficient data length for reconstruction with m=%d, tau=%d', m, tau);
    end
    
    phase_space = zeros(N, m);
    for i = 1:m
        phase_space(:, i) = data((i-1)*tau + 1 : (i-1)*tau + N);
    end
end

function [distances, evolution] = find_nearest_neighbors_improved(phase_space, N, params)
% Improved nearest neighbor finding with temporal constraints
    
    min_temporal_separation = max(params.mean_period, 2*params.tau);
    max_evolution_time = params.max_evolution_time;
    
    distances = [];
    evolution = [];
    
    for i = 1:N-max_evolution_time
        % Find potential neighbors (with temporal constraint)
        potential_neighbors = [(i+min_temporal_separation):N-max_evolution_time];
        
        if isempty(potential_neighbors)
            continue;
        end
        
        % Calculate distances to potential neighbors
        neighbor_distances = sqrt(sum((phase_space(potential_neighbors,:) - ...
                                     repmat(phase_space(i,:), ...
                                     length(potential_neighbors), 1)).^2, 2));
        
        % Find nearest neighbor
        [min_dist, min_idx] = min(neighbor_distances);
        nearest_neighbor = potential_neighbors(min_idx);
        
        % Track evolution
        for dt = 1:max_evolution_time
            if i+dt <= N && nearest_neighbor+dt <= N
                evolved_dist = sqrt(sum((phase_space(i+dt,:) - ...
                                       phase_space(nearest_neighbor+dt,:)).^2));
                
                if evolved_dist > 0 && min_dist > 0
                    distances(end+1) = log(evolved_dist / min_dist);
                    evolution(end+1) = dt;
                end
            end
        end
    end
    
    if isempty(distances)
        error('No valid nearest neighbors found');
    end
end

function [lambda, fit_info] = fit_robust_linear_region(distances, evolution, params)
% Robust linear fitting with automatic region selection
    
    % Remove outliers
    [distances_clean, evolution_clean] = remove_outliers(distances, evolution);
    
    % Group by evolution time and calculate means
    unique_times = unique(evolution_clean);
    mean_distances = zeros(size(unique_times));
    std_distances = zeros(size(unique_times));
    
    for i = 1:length(unique_times)
        t = unique_times(i);
        idx = evolution_clean == t;
        mean_distances(i) = mean(distances_clean(idx));
        std_distances(i) = std(distances_clean(idx));
    end
    
    % Find optimal linear region
    [fit_region, best_r2] = find_optimal_linear_region(unique_times, mean_distances);
    
    % Perform weighted linear fit
    fit_times = unique_times(fit_region);
    fit_distances = mean_distances(fit_region);
    fit_weights = 1 ./ (std_distances(fit_region) + eps);
    
    % Weighted least squares
    X = [ones(length(fit_times), 1), fit_times(:)];
    W = diag(fit_weights);
    beta = (X' * W * X) \ (X' * W * fit_distances(:));
    
    lambda = beta(2); % Slope is the Lyapunov exponent
    
    % Store fit information
    fit_info = struct();
    fit_info.fit_region = fit_region;
    fit_info.r_squared = best_r2;
    fit_info.intercept = beta(1);
    fit_info.slope = beta(2);
    fit_info.num_points = length(fit_times);
    fit_info.fit_times = fit_times;
    fit_info.fit_distances = fit_distances;
    fit_info.residuals = fit_distances(:) - X * beta;
end

function [distances_clean, evolution_clean] = remove_outliers(distances, evolution)
% Remove outliers using interquartile range method
    Q1 = quantile(distances, 0.25);
    Q3 = quantile(distances, 0.75);
    IQR = Q3 - Q1;
    
    lower_bound = Q1 - 1.5 * IQR;
    upper_bound = Q3 + 1.5 * IQR;
    
    valid_idx = distances >= lower_bound & distances <= upper_bound;
    distances_clean = distances(valid_idx);
    evolution_clean = evolution(valid_idx);
end

function [best_region, best_r2] = find_optimal_linear_region(times, distances)
% Find the region with the best linear fit
    n = length(times);
    min_points = max(5, floor(n/4)); % Minimum points for fitting
    
    best_r2 = -inf;
    best_region = 1:min_points;
    
    % Try different starting points and lengths
    for start_idx = 1:n-min_points+1
        for end_idx = start_idx+min_points-1:n
            region = start_idx:end_idx;
            
            % Calculate R-squared for this region
            X = [ones(length(region), 1), times(region)'];
            y = distances(region)';
            
            if rank(X) == 2 % Ensure full rank
                beta = X \ y;
                y_pred = X * beta;
                r2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);
                
                if r2 > best_r2
                    best_r2 = r2;
                    best_region = region;
                end
            end
        end
    end
end

function display_results(results)
% Display comprehensive results
    fprintf('\n=== LYAPUNOV EXPONENT CALCULATION RESULTS ===\n');
    
    if results.rosenstein.success
        fprintf('\nRosenstein Method:\n');
        fprintf('  Lyapunov Exponent: %.6f\n', results.rosenstein.lambda);
        fprintf('  Embedding Dimension: %d\n', results.rosenstein.params.m);
        fprintf('  Delay Time: %d\n', results.rosenstein.params.tau);
        fprintf('  Fit R²: %.4f\n', results.rosenstein.params.fit_info.r_squared);
        fprintf('  Fit Points: %d\n', results.rosenstein.params.fit_info.num_points);
    else
        fprintf('\nRosenstein Method: FAILED\n');
        fprintf('  Error: %s\n', results.rosenstein.error);
    end
    
    if results.wolf.success
        fprintf('\nWolf Method:\n');
        fprintf('  Lyapunov Exponent: %.6f\n', results.wolf.lambda);
        fprintf('  Embedding Dimension: %d\n', results.wolf.params.m);
        fprintf('  Delay Time: %d\n', results.wolf.params.tau);
        fprintf('  Replacement Steps: %d\n', results.wolf.params.num_replacements);
    else
        fprintf('\nWolf Method: FAILED\n');
        fprintf('  Error: %s\n', results.wolf.error);
    end
    
    if isfield(results, 'combined') && ~isempty(results.combined.values)
        fprintf('\nCombined Analysis:\n');
        fprintf('  Mean Lyapunov Exponent: %.6f ± %.6f\n', ...
                results.combined.mean, results.combined.std);
        fprintf('  Successful Methods: %s\n', strjoin_compat(results.combined.methods, ', '));
        fprintf('  Individual Values: %s\n', ...
                sprintf('%.6f ', results.combined.values));
    end
    
    fprintf('\n=== Analysis Complete ===\n');
end

function plot_comparison(results)
% Create comparison plots
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Rosenstein method details
    if results.rosenstein.success
        subplot(2, 2, 1);
        fit_info = results.rosenstein.params.fit_info;
        plot(fit_info.fit_times, fit_info.fit_distances, 'bo-', 'MarkerSize', 6);
        hold on;
        
        % Plot fit line
        fit_line = fit_info.intercept + fit_info.slope * fit_info.fit_times;
        plot(fit_info.fit_times, fit_line, 'r-', 'LineWidth', 2);
        
        xlabel('Time Steps');
        ylabel('ln(divergence)');
        title(sprintf('Rosenstein Method (λ = %.6f, R² = %.4f)', ...
                     results.rosenstein.lambda, fit_info.r_squared));
        legend('Data', 'Linear Fit', 'Location', 'best');
        grid on;
    end
    
    % Plot 2: Method comparison
    subplot(2, 2, 2);
    if isfield(results, 'combined')
        methods = results.combined.methods;
        values = results.combined.values;
        
        bar(values);
        set(gca, 'XTickLabel', methods);
        ylabel('Lyapunov Exponent');
        title('Method Comparison');
        
        % Add error bars if multiple methods
        if length(values) > 1
            hold on;
            errorbar(1:length(values), values, zeros(size(values)), ...
                    repmat(results.combined.std, size(values)), 'k.', 'LineWidth', 1.5);
        end
        grid on;
    end
    
    % Plot 3: Parameter analysis (embedding dimension sensitivity)
    subplot(2, 2, 3);
    if results.rosenstein.success
        % Show embedding dimension analysis
        m_values = 1:min(8, results.rosenstein.params.m + 3);
        lambda_m = zeros(size(m_values));
        
        for i = 1:length(m_values)
            try
                % Quick calculation with different m
                temp_data = generate_lorenz_data();
                temp_params = results.rosenstein.params;
                temp_params.m = m_values(i);
                [lambda_temp, ~] = rosenstein_lyapunov_exponent(temp_data);
                lambda_m(i) = lambda_temp;
            catch
                lambda_m(i) = NaN;
            end
        end
        
        plot(m_values, lambda_m, 'go-', 'MarkerSize', 8, 'LineWidth', 2);
        xlabel('Embedding Dimension');
        ylabel('Lyapunov Exponent');
        title('Sensitivity to Embedding Dimension');
        grid on;
        
        % Highlight chosen value
        chosen_idx = find(m_values == results.rosenstein.params.m);
        if ~isempty(chosen_idx)
            hold on;
            plot(m_values(chosen_idx), lambda_m(chosen_idx), 'ro', ...
                 'MarkerSize', 12, 'LineWidth', 3);
        end
    end
    
    % Plot 4: Data characteristics
    subplot(2, 2, 4);
    temp_data = generate_lorenz_data();
    plot(temp_data(1:1000));
    xlabel('Time');
    ylabel('Amplitude');
    title('Sample Time Series (Lorenz x-component)');
    grid on;
    
    sgtitle('Lyapunov Exponent Analysis Results', 'FontSize', 16, 'FontWeight', 'bold');
end

function autocorr_vals = simple_autocorr(data, max_lag)
% Simple autocorrelation function
    data = data(:) - mean(data); % Center the data
    N = length(data);
    autocorr_vals = zeros(max_lag + 1, 1);
    
    for lag = 0:max_lag
        if lag == 0
            autocorr_vals(lag + 1) = var(data);
        else
            if N - lag > 0
                autocorr_vals(lag + 1) = mean(data(1:N-lag) .* data(lag+1:N));
            else
                autocorr_vals(lag + 1) = 0;
            end
        end
    end
    
    % Normalize by lag-0 value
    if autocorr_vals(1) ~= 0
        autocorr_vals = autocorr_vals / autocorr_vals(1);
    end
end

function [peaks, locs] = simple_findpeaks(data)
% Simple peak finding function
    peaks = [];
    locs = [];
    
    for i = 2:length(data)-1
        if data(i) > data(i-1) && data(i) > data(i+1)
            peaks(end+1) = data(i);
            locs(end+1) = i;
        end
    end
end

function counts = simple_hist2d(x, y, nbins)
% Simple 2D histogram for mutual information
    x_min = min(x); x_max = max(x);
    y_min = min(y); y_max = max(y);
    
    x_edges = linspace(x_min, x_max, nbins + 1);
    y_edges = linspace(y_min, y_max, nbins + 1);
    
    counts = zeros(nbins, nbins);
    
    for i = 1:length(x)
        x_bin = max(1, min(nbins, floor((x(i) - x_min) / (x_max - x_min) * nbins) + 1));
        y_bin = max(1, min(nbins, floor((y(i) - y_min) / (y_max - y_min) * nbins) + 1));
        
        if x_bin > 0 && x_bin <= nbins && y_bin > 0 && y_bin <= nbins
            counts(x_bin, y_bin) = counts(x_bin, y_bin) + 1;
        end
    end
end

function result = strjoin_compat(cell_array, delimiter)
% Compatible string join function
    if isempty(cell_array)
        result = '';
        return;
    end
    
    result = cell_array{1};
    for i = 2:length(cell_array)
        result = [result, delimiter, cell_array{i}];
    end
end