% MAIN_BIFURCATION_ANALYSIS - Main script demonstrating improved bifurcation analysis
%
% This script demonstrates the key improvements made to the bifurcation
% diagram generation:
% 1. Clean envelope approach using global extrema only
% 2. Improved visualization with line plots and markers
% 3. Better visual distinction between different stiffness cases
%
% The improvements address the original problem of cluttered bifurcation
% diagrams by eliminating the plotting of all local extrema and instead
% focusing on the global envelope of the system's behavior.

clear all;
close all;
clc;

fprintf('===============================================\n');
fprintf('   IMPROVED BIFURCATION DIAGRAM ANALYSIS\n');
fprintf('===============================================\n');
fprintf('Demonstrating key improvements:\n');
fprintf('1. Global extrema only (clean envelope)\n');
fprintf('2. Line plots with markers (-o, --s)\n');
fprintf('3. Enhanced visual distinction\n');
fprintf('===============================================\n\n');

% Define analysis parameters
fprintf('Setting up analysis parameters...\n');

% Stiffness range - different cases to compare
stiffness_range = [0.8, 1.2, 1.6, 2.0];
fprintf('Stiffness values: [%.1f, %.1f, %.1f, %.1f]\n', stiffness_range);

% Frequency range - sweep across potential bifurcation region
freq_range = linspace(0.6, 2.4, 60);
fprintf('Frequency range: %.1f to %.1f Hz (%d points)\n', ...
    min(freq_range), max(freq_range), length(freq_range));

% Simulation parameters
simulation_params = struct();
simulation_params.transient_time = 80;    % Allow system to reach steady state
simulation_params.steady_time = 120;      % Analyze steady state behavior
simulation_params.dt = 0.005;             % Fine time resolution
simulation_params.amplitude = 1.2;        % Forcing amplitude
simulation_params.damping = 0.08;         % Light damping for rich dynamics
simulation_params.mass = 1.0;             % Unit mass

fprintf('Simulation parameters:\n');
fprintf('  Transient time: %.0f units\n', simulation_params.transient_time);
fprintf('  Steady state time: %.0f units\n', simulation_params.steady_time);
fprintf('  Time step: %.3f\n', simulation_params.dt);
fprintf('  Forcing amplitude: %.1f\n', simulation_params.amplitude);
fprintf('  Damping: %.3f\n', simulation_params.damping);

fprintf('\n--- PERFORMING CLEAN SIMULATION ---\n');

% Execute the improved simulation approach
tic;
[freq_data, global_max, global_min, stiffness_data] = ...
    perform_simulation_clean(stiffness_range, freq_range, simulation_params);
simulation_time = toc;

fprintf('Simulation completed in %.2f seconds\n', simulation_time);
fprintf('Generated %d data points total\n', length(global_max));

% Analyze the results
unique_stiffness = unique(stiffness_data);
for i = 1:length(unique_stiffness)
    count = sum(stiffness_data == unique_stiffness(i));
    fprintf('  Stiffness k=%.1f: %d points\n', unique_stiffness(i), count);
end

fprintf('\n--- CREATING IMPROVED VISUALIZATION ---\n');

% Set up enhanced plotting parameters
plot_params = struct();
plot_params.title = 'Clean Bifurcation Diagram: Global Extrema Envelope';
plot_params.xlabel = 'Frequency (Hz)';
plot_params.ylabel = 'Displacement Amplitude';
plot_params.MarkerSize = 4;
plot_params.LineWidth = 1.3;

% Custom color scheme for better distinction
plot_params.Colors = [
    0.1, 0.3, 0.8;  % Deep blue
    0.8, 0.1, 0.1;  % Deep red
    0.0, 0.7, 0.2;  % Forest green
    0.9, 0.4, 0.0;  % Orange
];

% Generate the improved bifurcation plot
tic;
fig_handle = plot_stiffness_bifurcation(freq_data, global_max, global_min, ...
                                       stiffness_data, plot_params);
plot_time = toc;

fprintf('Plot generated in %.3f seconds\n', plot_time);

% Enhance the figure with additional information
figure(fig_handle);

% Add analysis summary as text
summary_text = {
    'ANALYSIS SUMMARY:', ...
    sprintf('• %d stiffness cases analyzed', length(unique_stiffness)), ...
    sprintf('• %d frequency points per case', length(freq_range)), ...
    sprintf('• Global extrema approach (clean envelope)'), ...
    sprintf('• Line plots with markers for clarity'), ...
    sprintf('• Simulation time: %.2f s', simulation_time)
};

annotation('textbox', [0.65, 0.15, 0.33, 0.25], ...
    'String', summary_text, ...
    'FontSize', 9, ...
    'BackgroundColor', [0.95, 0.95, 0.95], ...
    'FitBoxToText', 'on', ...
    'EdgeColor', 'black', ...
    'LineWidth', 1);

% Save the figure
output_filename = 'improved_bifurcation_diagram.png';
fprintf('\n--- SAVING RESULTS ---\n');
saveas(fig_handle, output_filename);
fprintf('Figure saved as: %s\n', output_filename);

% Display analysis statistics
fprintf('\n--- ANALYSIS STATISTICS ---\n');
fprintf('Data range analysis:\n');
fprintf('  Frequency range: [%.2f, %.2f] Hz\n', min(freq_data), max(freq_data));
fprintf('  Maximum amplitude range: [%.3f, %.3f]\n', min(global_max), max(global_max));
fprintf('  Minimum amplitude range: [%.3f, %.3f]\n', min(global_min), max(global_min));

% Calculate amplitude envelope width for each stiffness
fprintf('\nAmplitude envelope analysis:\n');
for i = 1:length(unique_stiffness)
    idx = stiffness_data == unique_stiffness(i);
    max_vals = global_max(idx);
    min_vals = global_min(idx);
    envelope_width = max(max_vals) - min(min_vals);
    fprintf('  k=%.1f: Envelope width = %.3f\n', unique_stiffness(i), envelope_width);
end

fprintf('\n===============================================\n');
fprintf('         BIFURCATION ANALYSIS COMPLETE\n');
fprintf('===============================================\n');
fprintf('Key improvements successfully implemented:\n');
fprintf('✓ Clean envelope approach (global extrema only)\n');
fprintf('✓ Enhanced line plots with markers\n');
fprintf('✓ Improved visual distinction between cases\n');
fprintf('✓ Reduced visual clutter and overlapping\n');
fprintf('===============================================\n');

% Display final message
fprintf('\nResults saved and ready for analysis.\n');
fprintf('The improved bifurcation diagram shows clear envelopes\n');
fprintf('without the clutter of local extrema.\n');