% DEMO_IMPROVEMENTS - Demonstrate the bifurcation diagram improvements
%
% This script creates a comprehensive demonstration showing the key
% improvements made to address the cluttered bifurcation diagram problem.

clear all;
close all;
clc;

fprintf('=======================================================\n');
fprintf('  BIFURCATION DIAGRAM IMPROVEMENTS DEMONSTRATION\n');
fprintf('=======================================================\n');
fprintf('This demo shows the key improvements:\n');
fprintf('1. Clean envelope using global extrema only\n');
fprintf('2. Enhanced visualization with line plots + markers\n');
fprintf('3. Better distinction between stiffness cases\n');
fprintf('=======================================================\n\n');

% Demo parameters - balanced for good visualization
stiffness_range = [0.9, 1.3, 1.7];       % 3 distinct stiffness cases
freq_range = linspace(0.7, 2.3, 30);     % Good resolution for smooth lines

% Simulation parameters optimized for demonstration
simulation_params = struct();
simulation_params.transient_time = 40;    % Adequate settling time
simulation_params.steady_time = 60;       % Good steady-state data
simulation_params.dt = 0.01;              % Fine resolution
simulation_params.amplitude = 1.1;        % Clear nonlinear behavior
simulation_params.damping = 0.09;         % Light damping for rich dynamics
simulation_params.mass = 1.0;

fprintf('Demo Parameters:\n');
fprintf('  Stiffness values: [%.1f, %.1f, %.1f]\n', stiffness_range);
fprintf('  Frequency range: %.1f - %.1f Hz (%d points)\n', ...
    min(freq_range), max(freq_range), length(freq_range));
fprintf('  Expected data points: %d\n', length(stiffness_range) * length(freq_range));

fprintf('\n--- RUNNING CLEAN SIMULATION ---\n');

% Execute the improved bifurcation analysis
tic;
[freq_data, global_max, global_min, stiffness_data] = ...
    perform_simulation_clean(stiffness_range, freq_range, simulation_params);
sim_time = toc;

fprintf('Simulation completed in %.2f seconds\n', sim_time);
fprintf('Generated %d clean data points\n', length(global_max));

% Verify data integrity
unique_k = unique(stiffness_data);
fprintf('\nData verification:\n');
for i = 1:length(unique_k)
    count = sum(stiffness_data == unique_k(i));
    fprintf('  k=%.1f: %d points\n', unique_k(i), count);
end

fprintf('\n--- CREATING DEMONSTRATION PLOT ---\n');

% Enhanced plot parameters for demonstration
plot_params = struct();
plot_params.title = 'Clean Bifurcation Diagram: Global Extrema Envelope Method';
plot_params.xlabel = 'Driving Frequency (Hz)';
plot_params.ylabel = 'Response Amplitude';
plot_params.MarkerSize = 5;
plot_params.LineWidth = 1.5;

% Distinctive color scheme
plot_params.Colors = [
    0.0, 0.3, 0.9;  % Bright blue
    0.9, 0.1, 0.1;  % Bright red  
    0.0, 0.7, 0.1;  % Bright green
];

% Generate the improved plot
fig_handle = plot_stiffness_bifurcation(freq_data, global_max, global_min, ...
                                       stiffness_data, plot_params);

% Enhance the demonstration figure
figure(fig_handle);

% Add key improvements annotation
improvement_text = {
    'KEY IMPROVEMENTS IMPLEMENTED:', ...
    '', ...
    '✓ CLEAN ENVELOPE APPROACH:', ...
    '  • Global extrema only (no local peaks)', ...
    '  • Eliminates visual clutter', ...
    '  • Clear system behavior envelope', ...
    '', ...
    '✓ ENHANCED VISUALIZATION:', ...
    '  • Line plots with markers (-o, --s)', ...
    '  • Increased marker size and line width', ...
    '  • Distinct colors for each stiffness', ...
    '', ...
    '✓ IMPROVED CLARITY:', ...
    '  • No overlapping data clouds', ...
    '  • Clear distinction between cases', ...
    '  • Professional appearance'
};

annotation('textbox', [0.02, 0.55, 0.35, 0.42], ...
    'String', improvement_text, ...
    'FontSize', 9, ...
    'FontWeight', 'normal', ...
    'BackgroundColor', [0.98, 0.98, 0.98], ...
    'FitBoxToText', 'on', ...
    'EdgeColor', [0.3, 0.3, 0.3], ...
    'LineWidth', 1.2);

% Add analysis summary
summary_stats = {
    'ANALYSIS SUMMARY:', ...
    '', ...
    sprintf('Simulation Time: %.2f s', sim_time), ...
    sprintf('Data Points: %d', length(global_max)), ...
    sprintf('Stiffness Cases: %d', length(unique_k)), ...
    sprintf('Frequency Points: %d', length(freq_range)), ...
    '', ...
    'Data Range:', ...
    sprintf('  Freq: [%.2f, %.2f] Hz', min(freq_data), max(freq_data)), ...
    sprintf('  Ampl: [%.3f, %.3f]', min([global_max; global_min]), max([global_max; global_min]))
};

annotation('textbox', [0.65, 0.02, 0.33, 0.30], ...
    'String', summary_stats, ...
    'FontSize', 9, ...
    'FontWeight', 'normal', ...
    'BackgroundColor', [0.95, 0.98, 0.95], ...
    'FitBoxToText', 'on', ...
    'EdgeColor', [0.2, 0.5, 0.2], ...
    'LineWidth', 1);

% Save the demonstration figure
demo_filename = 'bifurcation_improvements_demo.png';
fprintf('Saving demonstration figure...\n');

% Try different methods to save the figure without display issues
try
    print(fig_handle, demo_filename, '-dpng', '-r150');
    fprintf('Demo figure saved as: %s\n', demo_filename);
catch
    try
        saveas(fig_handle, demo_filename);
        fprintf('Demo figure saved as: %s (fallback method)\n', demo_filename);
    catch
        fprintf('Note: Figure created but could not save in headless environment\n');
    end
end

fprintf('\n--- ANALYSIS RESULTS ---\n');

% Calculate and display envelope characteristics
fprintf('Envelope Analysis:\n');
for i = 1:length(unique_k)
    idx = stiffness_data == unique_k(i);
    max_env = global_max(idx);
    min_env = global_min(idx);
    
    envelope_width = max(max_env) - min(min_env);
    max_amplitude = max(max_env);
    min_amplitude = min(min_env);
    
    fprintf('  Stiffness k=%.1f:\n', unique_k(i));
    fprintf('    Envelope width: %.4f\n', envelope_width);
    fprintf('    Max amplitude: %.4f\n', max_amplitude);
    fprintf('    Min amplitude: %.4f\n', min_amplitude);
end

% Show the benefit of the clean approach
fprintf('\nBENEFITS OF IMPROVEMENTS:\n');
fprintf('• Eliminated clutter from local extrema\n');
fprintf('• Clear envelope shows system boundaries\n');
fprintf('• Line plots reveal frequency response trends\n');
fprintf('• Enhanced visual distinction between cases\n');
fprintf('• Professional, publication-ready appearance\n');

fprintf('\n=======================================================\n');
fprintf('           DEMONSTRATION COMPLETED\n');
fprintf('=======================================================\n');
fprintf('The improved bifurcation analysis successfully addresses\n');
fprintf('the original problem of cluttered, overlapping data by:\n');
fprintf('1. Using global extrema only (clean envelope)\n');
fprintf('2. Enhanced line plots with markers\n');
fprintf('3. Improved visual styling and distinction\n');
fprintf('=======================================================\n');

% Display completion message
fprintf('\nDemonstration complete! The improved method provides:\n');
fprintf('✓ Clear, uncluttered bifurcation diagrams\n');
fprintf('✓ Better insight into system behavior\n');
fprintf('✓ Professional visualization quality\n');