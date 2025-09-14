function fig_handle = plot_stiffness_bifurcation(freq_range, global_max, global_min, stiffness_values, plot_params)
    % PLOT_STIFFNESS_BIFURCATION - Create clean bifurcation diagram with improved visualization
    %
    % This function creates a bifurcation diagram using line plots with markers
    % instead of scattered points for better visual clarity and distinction
    % between different stiffness cases.
    %
    % Inputs:
    %   freq_range      - Frequency values for each data point
    %   global_max      - Global maximum values
    %   global_min      - Global minimum values  
    %   stiffness_values - Stiffness values for each data point
    %   plot_params     - Optional structure with plotting parameters
    %
    % Outputs:
    %   fig_handle - Handle to the created figure

    % Default plot parameters if not provided
    if nargin < 5
        plot_params = struct();
    end
    
    % Set default plotting parameters for improved visualization
    if ~isfield(plot_params, 'MarkerSize')
        plot_params.MarkerSize = 4; % Larger markers for better visibility
    end
    if ~isfield(plot_params, 'LineWidth')
        plot_params.LineWidth = 1.2; % Thicker lines for better distinction
    end
    if ~isfield(plot_params, 'Colors')
        % Define distinct colors for different stiffness cases
        plot_params.Colors = [
            0.0, 0.4, 0.8;  % Blue
            0.8, 0.2, 0.2;  % Red  
            0.0, 0.6, 0.3;  % Green
            0.9, 0.5, 0.0;  % Orange
            0.6, 0.0, 0.8;  % Purple
            0.8, 0.8, 0.0;  % Yellow
            0.0, 0.8, 0.8;  % Cyan
            0.8, 0.0, 0.4;  % Magenta
        ];
    end
    if ~isfield(plot_params, 'title')
        plot_params.title = 'Clean Bifurcation Diagram - Global Extrema Envelope';
    end
    if ~isfield(plot_params, 'xlabel')
        plot_params.xlabel = 'Frequency';
    end
    if ~isfield(plot_params, 'ylabel')
        plot_params.ylabel = 'Amplitude';
    end
    
    % Create new figure with improved settings
    fig_handle = figure('Position', [100, 100, 900, 600]);
    hold on;
    
    % Get unique stiffness values and sort them
    unique_stiffness = unique(stiffness_values);
    num_stiffness = length(unique_stiffness);
    
    fprintf('Plotting bifurcation diagram for %d stiffness values...\n', num_stiffness);
    
    % Plot each stiffness case separately with lines and markers
    legend_entries = {};
    
    for i = 1:num_stiffness
        current_stiffness = unique_stiffness(i);
        
        % Find indices for current stiffness value
        stiffness_idx = (stiffness_values == current_stiffness);
        
        if sum(stiffness_idx) > 0
            % Extract data for current stiffness
            freq_current = freq_range(stiffness_idx);
            max_current = global_max(stiffness_idx);
            min_current = global_min(stiffness_idx);
            
            % Sort by frequency for proper line plotting
            [freq_sorted, sort_idx] = sort(freq_current);
            max_sorted = max_current(sort_idx);
            min_sorted = min_current(sort_idx);
            
            % Select color for current stiffness (cycle through available colors)
            color_idx = mod(i-1, size(plot_params.Colors, 1)) + 1;
            current_color = plot_params.Colors(color_idx, :);
            
            % Plot maximum envelope with line and markers
            plot(freq_sorted, max_sorted, '-o', ...
                'Color', current_color, ...
                'MarkerSize', plot_params.MarkerSize, ...
                'LineWidth', plot_params.LineWidth, ...
                'MarkerFaceColor', current_color, ...
                'MarkerEdgeColor', current_color * 0.7);
            
            % Plot minimum envelope with line and markers (slightly different style)
            plot(freq_sorted, min_sorted, '--s', ...
                'Color', current_color, ...
                'MarkerSize', plot_params.MarkerSize * 0.8, ...
                'LineWidth', plot_params.LineWidth * 0.9, ...
                'MarkerFaceColor', 'none', ...
                'MarkerEdgeColor', current_color);
            
            % Add legend entries
            legend_entries{end+1} = sprintf('k=%.1f (max)', current_stiffness);
            legend_entries{end+1} = sprintf('k=%.1f (min)', current_stiffness);
        end
    end
    
    % Enhance plot appearance
    grid on;
    grid minor;
    
    % Set labels and title
    xlabel(plot_params.xlabel, 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(plot_params.ylabel, 'FontSize', 12, 'FontWeight', 'bold');
    title(plot_params.title, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Add legend with improved positioning (Octave compatible)
    if ~isempty(legend_entries)
        legend_handle = legend(legend_entries, 'Location', 'northeast', 'FontSize', 10);
        set(legend_handle, 'Box', 'on');
    end
    
    % Improve axis appearance (Octave compatible)
    ax = gca;
    set(ax, 'FontSize', 10, 'LineWidth', 1.0, 'Box', 'on');
    
    % Set axis limits with some padding
    if ~isempty(freq_range)
        freq_margin = (max(freq_range) - min(freq_range)) * 0.05;
        xlim([min(freq_range) - freq_margin, max(freq_range) + freq_margin]);
    end
    
    if ~isempty([global_max; global_min])
        all_amplitudes = [global_max; global_min];
        amp_margin = (max(all_amplitudes) - min(all_amplitudes)) * 0.1;
        ylim([min(all_amplitudes) - amp_margin, max(all_amplitudes) + amp_margin]);
    end
    
    hold off;
    
    % Add text annotation with key improvements
    annotation('textbox', [0.02, 0.02, 0.4, 0.1], ...
        'String', {'Improvements:', '• Global extrema only (clean envelope)', '• Line plots with markers (-o, --s)', '• Enhanced visual distinction'}, ...
        'FontSize', 9, ...
        'BackgroundColor', 'white', ...
        'FitBoxToText', 'on', ...
        'EdgeColor', 'black');
    
    fprintf('Bifurcation plot completed with %d data points.\n', length(global_max));
end

function demo_bifurcation_analysis()
    % DEMO_BIFURCATION_ANALYSIS - Demonstration of the improved bifurcation analysis
    %
    % This function demonstrates the complete workflow using the improved
    % bifurcation diagram generation with clean envelope approach.
    
    fprintf('=== Bifurcation Analysis Demonstration ===\n');
    
    % Define parameter ranges for the analysis
    stiffness_range = [0.5, 1.0, 1.5, 2.0]; % Different stiffness values
    freq_range = linspace(0.5, 2.5, 50);     % Frequency sweep
    
    % Set simulation parameters
    simulation_params = struct();
    simulation_params.transient_time = 50;   % Time to reach steady state
    simulation_params.steady_time = 100;     % Time for steady state analysis
    simulation_params.dt = 0.01;             % Time step
    simulation_params.amplitude = 1.0;       % Forcing amplitude
    simulation_params.damping = 0.1;         % Damping coefficient
    simulation_params.mass = 1.0;            % Mass
    
    % Perform the clean simulation (global extrema only)
    fprintf('Running simulation with clean envelope approach...\n');
    [freq_data, global_max, global_min, stiffness_data] = ...
        perform_simulation_clean(stiffness_range, freq_range, simulation_params);
    
    % Create the improved bifurcation plot
    fprintf('Creating improved bifurcation plot...\n');
    plot_params = struct();
    plot_params.title = 'Improved Bifurcation Diagram - Clean Envelope Method';
    plot_params.MarkerSize = 5;
    plot_params.LineWidth = 1.5;
    
    fig_handle = plot_stiffness_bifurcation(freq_data, global_max, global_min, stiffness_data, plot_params);
    
    % Save the figure
    saveas(fig_handle, 'clean_bifurcation_diagram.png');
    fprintf('Figure saved as clean_bifurcation_diagram.png\n');
    
    fprintf('=== Analysis Complete ===\n');
    fprintf('Key improvements implemented:\n');
    fprintf('1. Global extrema only (no local peaks) -> Clean envelope\n');
    fprintf('2. Line plots with markers -> Better visual distinction\n');
    fprintf('3. Enhanced styling -> Improved clarity\n');
end