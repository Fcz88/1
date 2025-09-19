# Improved Lyapunov Exponent Calculation

This repository contains an improved MATLAB implementation for calculating Lyapunov exponents from time series data, addressing accuracy and numerical stability issues in chaotic dynamical systems analysis.

## Features

### Multiple Calculation Methods
1. **Improved Rosenstein Method**: Enhanced version with adaptive parameter selection and robust linear fitting
2. **Wolf Method**: Implementation of the Wolf et al. algorithm with periodic reorthogonalization
3. **Multi-algorithm Comparison**: Provides results from multiple methods for validation

### Key Improvements

#### 1. Enhanced Numerical Stability
- Robust outlier removal using interquartile range method
- Improved handling of small divergence distances
- Weighted least squares fitting for better accuracy

#### 2. Adaptive Parameter Selection
- Automatic embedding dimension selection using false nearest neighbors
- Optimal delay time estimation using mutual information and autocorrelation
- Adaptive linear fitting region selection

#### 3. Improved Nearest Neighbor Search
- Temporal constraints to avoid spurious neighbors
- Better handling of trajectory evolution characteristics
- Enhanced distance calculation with numerical safeguards

#### 4. Robust Linear Fitting
- Automatic optimal region selection based on R² values
- Weighted fitting considering error variance
- Multiple fitting strategies with fallback mechanisms

## Usage

### Basic Usage
```matlab
% Run the main function with example data
Copy_3_of_hundun1();
```

### Custom Data Analysis
```matlab
% Load your time series data
data = load('your_timeseries.txt');

% Calculate using multiple methods
results = calculate_lyapunov_multi_method(data);

% Display results
display_results(results);
```

### Individual Method Usage
```matlab
% Rosenstein method only
[lambda_ros, params_ros] = rosenstein_lyapunov_exponent(data);

% Wolf method only
[lambda_wolf, params_wolf] = wolf_lyapunov_exponent(data);
```

## Algorithm Details

### Rosenstein Method Improvements
- **Adaptive embedding dimension**: Uses false nearest neighbors algorithm
- **Robust delay time**: Combines mutual information and autocorrelation analysis
- **Enhanced neighbor search**: Implements temporal separation constraints
- **Optimal fitting region**: Automatically selects best linear region

### Wolf Method Implementation
- **Periodic reorthogonalization**: Prevents numerical drift
- **Adaptive separation thresholds**: Maintains optimal trajectory separation
- **Robust evolution tracking**: Enhanced handling of trajectory evolution

### Parameter Estimation
- **Embedding dimension (m)**: Automatically determined using false nearest neighbors
- **Delay time (τ)**: Optimized using mutual information first minimum
- **Evolution time**: Adaptive based on system characteristics
- **Fitting region**: Automatically selected for best linear fit

## Expected Results

For the Lorenz system (example data):
- Theoretical λ₁ ≈ 0.906
- Our implementation typically produces values in the range 0.8-1.1
- Results depend on data length, sampling rate, and noise level

## Functions Overview

### Main Functions
- `Copy_3_of_hundun1()`: Main entry point with example
- `calculate_lyapunov_multi_method()`: Multi-algorithm calculation
- `rosenstein_lyapunov_exponent()`: Improved Rosenstein method
- `wolf_lyapunov_exponent()`: Wolf method implementation

### Parameter Estimation
- `find_optimal_embedding_dimension()`: False nearest neighbors algorithm
- `find_optimal_delay_robust()`: Multi-method delay time optimization
- `estimate_mean_period()`: System period estimation

### Utility Functions
- `reconstruct_phase_space()`: Time-delay embedding
- `find_nearest_neighbors_improved()`: Enhanced neighbor search
- `fit_robust_linear_region()`: Robust linear fitting
- `mutual_information()`: Mutual information calculation

## Compatibility

- **MATLAB**: Tested on R2018b and later
- **Octave**: Compatible with Octave 4.0+ (with minor feature limitations)
- **Dependencies**: No external toolboxes required (all functions implemented from scratch)

## Testing

Run the test function to validate the installation:
```matlab
test_lyapunov();
```

## Troubleshooting

### Common Issues
1. **Insufficient data length**: Ensure time series has at least 1000 points
2. **Poor embedding parameters**: Check if automatic parameter selection is reasonable
3. **Numerical instability**: Try different data preprocessing (normalization, filtering)

### Parameter Tuning
If automatic parameter selection doesn't work well for your data:
- Manually set embedding dimension (typically 3-10)
- Adjust delay time based on system characteristics
- Modify fitting region selection criteria

## References

1. Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. Physica D, 65(1-2), 117-134.

2. Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985). Determining Lyapunov exponents from a time series. Physica D, 16(3), 285-317.

3. Kantz, H., Schreiber, T. (2004). Nonlinear time series analysis. Cambridge university press.