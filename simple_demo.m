% SIMPLE_DEMO - Simplified demonstration of bifurcation improvements
% This script provides a clean demonstration compatible with Octave

fprintf('====================================================\n');
fprintf('  CLEAN BIFURCATION DIAGRAM DEMONSTRATION\n');
fprintf('====================================================\n');

% Simple but effective demo parameters
stiffness_range = [1.0, 1.5, 2.0];
freq_range = linspace(0.8, 2.2, 25);

% Efficient simulation parameters
simulation_params = struct();
simulation_params.transient_time = 30;
simulation_params.steady_time = 50;
simulation_params.dt = 0.02;
simulation_params.amplitude = 1.0;
simulation_params.damping = 0.1;
simulation_params.mass = 1.0;

fprintf('Running clean bifurcation analysis...\n');
fprintf('  Stiffness cases: %d\n', length(stiffness_range));
fprintf('  Frequency points: %d\n', length(freq_range));

% Run the improved simulation
tic;
[freq_data, global_max, global_min, stiffness_data] = ...
    perform_simulation_clean(stiffness_range, freq_range, simulation_params);
runtime = toc;

fprintf('Analysis completed in %.1f seconds\n', runtime);
fprintf('Generated %d clean data points\n', length(global_max));

% Create clean visualization
fprintf('Creating improved visualization...\n');

figure('Position', [100, 100, 800, 500]);
hold on;

% Define colors for each stiffness
colors = [0.0, 0.4, 0.8; 0.8, 0.2, 0.2; 0.0, 0.6, 0.3];
marker_size = 4;
line_width = 1.2;

% Plot each stiffness case
unique_stiffness = unique(stiffness_data);

for i = 1:length(unique_stiffness)
    current_k = unique_stiffness(i);
    idx = stiffness_data == current_k;
    
    freq_k = freq_data(idx);
    max_k = global_max(idx);
    min_k = global_min(idx);
    
    % Sort by frequency
    [freq_sorted, sort_idx] = sort(freq_k);
    max_sorted = max_k(sort_idx);
    min_sorted = min_k(sort_idx);
    
    % Plot maximum envelope
    plot(freq_sorted, max_sorted, '-o', ...
        'Color', colors(i,:), ...
        'MarkerSize', marker_size, ...
        'LineWidth', line_width, ...
        'MarkerFaceColor', colors(i,:));
    
    % Plot minimum envelope
    plot(freq_sorted, min_sorted, '--s', ...
        'Color', colors(i,:), ...
        'MarkerSize', marker_size * 0.8, ...
        'LineWidth', line_width * 0.8, ...
        'MarkerFaceColor', 'none');
end

% Enhance plot
grid on;
xlabel('Frequency (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
title('Clean Bifurcation Diagram: Global Extrema Envelope', ...
      'FontSize', 14, 'FontWeight', 'bold');

% Add legend manually
legend_text = {};
for i = 1:length(unique_stiffness)
    legend_text{end+1} = sprintf('k=%.1f (max)', unique_stiffness(i));
    legend_text{end+1} = sprintf('k=%.1f (min)', unique_stiffness(i));
end

try
    legend(legend_text, 'Location', 'northeast', 'FontSize', 9);
catch
    % Skip legend if it causes issues
    fprintf('Note: Legend skipped due to compatibility\n');
end

hold off;

% Display results summary
fprintf('\n=== ANALYSIS SUMMARY ===\n');
fprintf('Key Improvements Implemented:\n');
fprintf('✓ Global extrema only (clean envelope)\n');
fprintf('✓ Line plots with markers for clarity\n');
fprintf('✓ Enhanced visual distinction\n');
fprintf('✓ Eliminated visual clutter\n');

fprintf('\nData Summary:\n');
for i = 1:length(unique_stiffness)
    idx = stiffness_data == unique_stiffness(i);
    envelope_width = max(global_max(idx)) - min(global_min(idx));
    fprintf('  k=%.1f: envelope width = %.4f\n', unique_stiffness(i), envelope_width);
end

fprintf('\nBenefits of Clean Approach:\n');
fprintf('• No overlapping data clouds\n');
fprintf('• Clear system behavior envelope\n');
fprintf('• Better insight into bifurcations\n');
fprintf('• Professional visualization\n');

fprintf('\n====================================================\n');
fprintf('    DEMONSTRATION SUCCESSFULLY COMPLETED\n');
fprintf('====================================================\n');

% Try to save the figure
try
    print('clean_bifurcation_demo.png', '-dpng', '-r150');
    fprintf('Figure saved as: clean_bifurcation_demo.png\n');
catch
    fprintf('Note: Figure display successful but save skipped\n');
end

fprintf('The improved method provides clear, uncluttered\n');
fprintf('bifurcation diagrams suitable for analysis.\n');