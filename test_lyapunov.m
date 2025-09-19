function test_lyapunov()
% Simple test function to validate the Lyapunov code structure
    
    fprintf('Testing Lyapunov Exponent Calculation Code...\n');
    
    % Test 1: Generate test data
    fprintf('1. Generating test data...\n');
    data = generate_simple_data();
    fprintf('   Data length: %d\n', length(data));
    
    % Test 2: Test parameter estimation functions
    fprintf('2. Testing parameter estimation...\n');
    m = 3;  % Fixed embedding dimension for testing
    tau = 5; % Fixed delay time for testing
    fprintf('   Using m=%d, tau=%d\n', m, tau);
    
    % Test 3: Test phase space reconstruction
    fprintf('3. Testing phase space reconstruction...\n');
    try
        [phase_space, N] = reconstruct_phase_space(data, m, tau);
        fprintf('   Phase space dimensions: %dx%d\n', size(phase_space,1), size(phase_space,2));
        fprintf('   SUCCESS: Phase space reconstruction\n');
    catch e
        fprintf('   ERROR in phase space reconstruction: %s\n', e.message);
    end
    
    % Test 4: Test mutual information calculation (simplified)
    fprintf('4. Testing mutual information...\n');
    try
        mi = simple_mutual_information(data, tau);
        fprintf('   Mutual information: %.4f\n', mi);
        fprintf('   SUCCESS: Mutual information\n');
    catch e
        fprintf('   ERROR in mutual information: %s\n', e.message);
    end
    
    fprintf('\nTest completed.\n');
end

function data = generate_simple_data()
% Generate simple chaotic data for testing
    n = 1000;
    data = zeros(n, 1);
    x = 0.1; % Initial condition
    r = 3.8; % Logistic map parameter
    
    for i = 1:n
        x = r * x * (1 - x);
        data(i) = x;
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

function mi = simple_mutual_information(data, tau)
% Simplified mutual information calculation
    N = length(data);
    if tau >= N
        mi = 0;
        return;
    end
    
    x = data(1:N-tau);
    y = data(tau+1:N);
    
    % Simple correlation-based approximation
    mi = abs(corr(x, y));
end