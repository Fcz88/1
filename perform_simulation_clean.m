function [freq_range, global_max, global_min, stiffness_values] = perform_simulation_clean(stiffness_range, freq_range, simulation_params)
    % PERFORM_SIMULATION_CLEAN - Generate clean bifurcation diagram data
    % 
    % This function performs simulations across frequency and stiffness ranges
    % and extracts only the global maximum and minimum of the steady-state
    % time series for each frequency, creating a clean envelope for the
    % bifurcation diagram instead of plotting all local extrema.
    %
    % Inputs:
    %   stiffness_range - Array of stiffness values to simulate
    %   freq_range      - Array of frequency values to simulate  
    %   simulation_params - Structure containing simulation parameters
    %
    % Outputs:
    %   freq_range      - Frequency values used in simulation
    %   global_max      - Global maximum values for each frequency/stiffness
    %   global_min      - Global minimum values for each frequency/stiffness
    %   stiffness_values - Stiffness values corresponding to each data point

    % Default parameters if not provided
    if nargin < 3
        simulation_params = struct();
    end
    
    % Set default simulation parameters
    if ~isfield(simulation_params, 'transient_time')
        simulation_params.transient_time = 100; % Time to skip for steady state
    end
    if ~isfield(simulation_params, 'steady_time')
        simulation_params.steady_time = 200; % Time for steady state analysis
    end
    if ~isfield(simulation_params, 'dt')
        simulation_params.dt = 0.01; % Time step
    end
    if ~isfield(simulation_params, 'amplitude')
        simulation_params.amplitude = 1.0; % Forcing amplitude
    end
    if ~isfield(simulation_params, 'damping')
        simulation_params.damping = 0.1; % Damping coefficient
    end
    if ~isfield(simulation_params, 'mass')
        simulation_params.mass = 1.0; % Mass
    end
    
    % Initialize output arrays
    global_max = [];
    global_min = [];
    freq_output = [];
    stiffness_output = [];
    
    fprintf('Starting bifurcation simulation...\n');
    
    % Loop through each stiffness value
    for i = 1:length(stiffness_range)
        current_stiffness = stiffness_range(i);
        fprintf('Processing stiffness %.3f (%d/%d)\n', current_stiffness, i, length(stiffness_range));
        
        % Loop through each frequency value
        for j = 1:length(freq_range)
            current_freq = freq_range(j);
            
            % Simulate the system for current stiffness and frequency
            [t, W_series_B] = simulate_duffing_oscillator(current_stiffness, current_freq, simulation_params);
            
            % Find steady-state portion (skip transient)
            transient_samples = round(simulation_params.transient_time / simulation_params.dt);
            if length(W_series_B) > transient_samples
                W_steady = W_series_B(transient_samples+1:end);
                
                % Find global maximum and minimum of steady-state time series
                % This is the key change: instead of using findpeaks to get all
                % local extrema, we only get the global extrema for a clean envelope
                current_max = max(W_steady);
                current_min = min(W_steady);
                
                % Store results
                global_max = [global_max; current_max];
                global_min = [global_min; current_min];
                freq_output = [freq_output; current_freq];
                stiffness_output = [stiffness_output; current_stiffness];
            end
        end
    end
    
    % Return the frequency range and stiffness values used
    freq_range = freq_output;
    stiffness_values = stiffness_output;
    
    fprintf('Simulation completed. Generated %d data points.\n', length(global_max));
end

function [t, x] = simulate_duffing_oscillator(stiffness, freq, params)
    % SIMULATE_DUFFING_OSCILLATOR - Simulate a Duffing oscillator system
    %
    % This function simulates a nonlinear Duffing oscillator with the form:
    % m*x'' + c*x' + k*x + alpha*x^3 = F*cos(omega*t)
    %
    % Inputs:
    %   stiffness - Linear stiffness coefficient (k)
    %   freq      - Forcing frequency (omega)  
    %   params    - Structure with simulation parameters
    %
    % Outputs:
    %   t - Time vector
    %   x - Displacement response time series (W_series_B)
    
    % Extract parameters
    dt = params.dt;
    total_time = params.transient_time + params.steady_time;
    amplitude = params.amplitude;
    damping = params.damping;
    mass = params.mass;
    
    % Nonlinear stiffness coefficient (typical Duffing parameter)
    alpha = stiffness * 0.1; % Scale nonlinear term relative to linear stiffness
    
    % Time vector
    t = 0:dt:total_time;
    n_steps = length(t);
    
    % Initialize state variables [position, velocity]
    x = zeros(n_steps, 1);
    v = zeros(n_steps, 1);
    
    % Initial conditions (can be modified as needed)
    x(1) = 0.1; % Small initial displacement
    v(1) = 0.0; % Zero initial velocity
    
    % Numerical integration using 4th-order Runge-Kutta
    omega = 2*pi*freq; % Convert frequency to angular frequency
    
    for i = 1:(n_steps-1)
        % Current state
        xi = x(i);
        vi = v(i);
        ti = t(i);
        
        % Forcing term
        F = amplitude * cos(omega * ti);
        
        % Duffing oscillator equation: x'' = (F - c*x' - k*x - alpha*x^3) / m
        k1_v = dt * vi;
        k1_a = dt * (F - damping*vi - stiffness*xi - alpha*xi^3) / mass;
        
        k2_v = dt * (vi + k1_a/2);
        k2_a = dt * (F - damping*(vi + k1_a/2) - stiffness*(xi + k1_v/2) - alpha*(xi + k1_v/2)^3) / mass;
        
        k3_v = dt * (vi + k2_a/2);
        k3_a = dt * (F - damping*(vi + k2_a/2) - stiffness*(xi + k2_v/2) - alpha*(xi + k2_v/2)^3) / mass;
        
        k4_v = dt * (vi + k3_a);
        k4_a = dt * (F - damping*(vi + k3_a) - stiffness*(xi + k3_v) - alpha*(xi + k3_v)^3) / mass;
        
        % Update state
        x(i+1) = xi + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6;
        v(i+1) = vi + (k1_a + 2*k2_a + 2*k3_a + k4_a)/6;
    end
end