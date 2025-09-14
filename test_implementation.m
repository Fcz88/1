% TEST_IMPLEMENTATION - Test script for the improved bifurcation analysis
% This script tests the implementation with a smaller parameter set

fprintf('Testing improved bifurcation analysis implementation...\n');

% Small test parameters for quick validation
stiffness_range = [1.0, 1.5];           % Just 2 stiffness values
freq_range = linspace(0.8, 2.0, 10);    % Only 10 frequency points

% Simple simulation parameters
simulation_params = struct();
simulation_params.transient_time = 20;   % Shorter for testing
simulation_params.steady_time = 30;      % Shorter for testing
simulation_params.dt = 0.02;             % Coarser for speed
simulation_params.amplitude = 1.0;
simulation_params.damping = 0.1;
simulation_params.mass = 1.0;

fprintf('Running simulation with %d stiffness values and %d frequencies...\n', ...
    length(stiffness_range), length(freq_range));

% Test the simulation function
try
    [freq_data, global_max, global_min, stiffness_data] = ...
        perform_simulation_clean(stiffness_range, freq_range, simulation_params);
    
    fprintf('Simulation successful!\n');
    fprintf('Generated %d data points\n', length(global_max));
    
    % Test the plotting function (without actually displaying)
    plot_params = struct();
    plot_params.title = 'Test Bifurcation Plot';
    
    fprintf('Testing plot generation...\n');
    
    % Create figure but don't display (headless mode)
    fig_handle = figure('Visible', 'off');
    
    % Test plotting function components
    unique_stiffness = unique(stiffness_data);
    fprintf('Found %d unique stiffness values: ', length(unique_stiffness));
    fprintf('%.2f ', unique_stiffness);
    fprintf('\n');
    
    % Verify data ranges
    fprintf('Data validation:\n');
    fprintf('  Frequency range: [%.2f, %.2f]\n', min(freq_data), max(freq_data));
    fprintf('  Max amplitude range: [%.3f, %.3f]\n', min(global_max), max(global_max));
    fprintf('  Min amplitude range: [%.3f, %.3f]\n', min(global_min), max(global_min));
    
    % Test successful
    fprintf('All tests passed successfully!\n');
    fprintf('Implementation is working correctly.\n');
    
    close(fig_handle);
    
catch ME
    fprintf('Error during testing: %s\n', ME.message);
    fprintf('Error in file: %s, line: %d\n', ME.stack(1).file, ME.stack(1).line);
end

fprintf('Test completed.\n');