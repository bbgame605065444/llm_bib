"""
Cumulative Differences Calibration Analysis
Implements the recommended approach from Kolmogorov-Wiener methods
Avoids the failure modes of classical binning methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, anderson
import warnings
warnings.filterwarnings('ignore')

class CumulativeDifferencesCalibration:
    """
    Cumulative Differences Method for Calibration Analysis
    
    This method avoids the problems of classical binning approaches and provides
    more robust measures of calibration quality.
    """
    
    def __init__(self):
        self.results = {}
        
    def compute_cumulative_differences(self, predicted_probs, true_outcomes):
        """
        Compute cumulative differences between predicted and observed probabilities
        
        Args:
            predicted_probs (array): Predicted probabilities (scores)
            true_outcomes (array): True binary outcomes (responses)
            
        Returns:
            dict: Cumulative difference statistics
        """
        # Sort by predicted probabilities
        sorted_indices = np.argsort(predicted_probs)
        sorted_probs = predicted_probs[sorted_indices]
        sorted_outcomes = true_outcomes[sorted_indices]
        
        n = len(predicted_probs)
        
        # Compute cumulative predicted probabilities
        cumulative_predicted = np.cumsum(sorted_probs)
        
        # Compute cumulative observed outcomes
        cumulative_observed = np.cumsum(sorted_outcomes)
        
        # Compute cumulative differences
        cumulative_diff = cumulative_observed - cumulative_predicted
        
        # Standardized cumulative differences (accounts for expected variance)
        # Under null hypothesis of perfect calibration, variance is sum of p(1-p)
        expected_variance = np.cumsum(sorted_probs * (1 - sorted_probs))
        standardized_diff = np.zeros_like(cumulative_diff)
        
        # Avoid division by zero
        non_zero_var = expected_variance > 1e-10
        standardized_diff[non_zero_var] = cumulative_diff[non_zero_var] / np.sqrt(expected_variance[non_zero_var])
        
        return {
            'sorted_probs': sorted_probs,
            'sorted_outcomes': sorted_outcomes,
            'cumulative_predicted': cumulative_predicted,
            'cumulative_observed': cumulative_observed,
            'cumulative_diff': cumulative_diff,
            'standardized_diff': standardized_diff,
            'expected_variance': expected_variance,
            'sorted_indices': sorted_indices
        }
    
    def kolmogorov_smirnov_test(self, predicted_probs, true_outcomes):
        """
        Kolmogorov-Smirnov test for calibration
        
        Tests whether the distribution of outcomes matches predicted probabilities
        """
        # Create empirical CDF of outcomes
        sorted_indices = np.argsort(predicted_probs)
        sorted_probs = predicted_probs[sorted_indices]
        sorted_outcomes = true_outcomes[sorted_indices]
        
        n = len(predicted_probs)
        
        # Empirical CDF of predictions
        empirical_pred_cdf = np.arange(1, n + 1) / n
        
        # Empirical CDF of outcomes weighted by predictions
        # This is more complex - we need to compare the distribution of residuals
        residuals = sorted_outcomes - sorted_probs
        
        # KS test comparing residuals to expected distribution under perfect calibration
        # Under perfect calibration, residuals should have specific distribution
        ks_statistic, p_value = kstest(residuals, 'norm', args=(0, np.std(residuals)))
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'residuals': residuals
        }
    
    def anderson_darling_test(self, predicted_probs, true_outcomes):
        """
        Anderson-Darling test for calibration
        More sensitive to tail deviations than KS test
        """
        sorted_indices = np.argsort(predicted_probs)
        sorted_probs = predicted_probs[sorted_indices]
        sorted_outcomes = true_outcomes[sorted_indices]
        
        # Compute standardized residuals
        residuals = sorted_outcomes - sorted_probs
        
        # Anderson-Darling test
        ad_result = anderson(residuals, dist='norm')
        
        return {
            'ad_statistic': ad_result.statistic,

            'critical_values': ad_result.critical_values,
            'significance_level': ad_result.significance_level,  # 修改这一行
            'residuals': residuals
        }
    
    def compute_calibration_metrics(self, predicted_probs, true_outcomes):
        """
        Compute various calibration metrics using cumulative differences
        """
        cum_diff = self.compute_cumulative_differences(predicted_probs, true_outcomes)
        
        # Maximum absolute cumulative difference
        max_abs_diff = np.max(np.abs(cum_diff['cumulative_diff']))
        
        # Root mean square cumulative difference
        rms_diff = np.sqrt(np.mean(cum_diff['cumulative_diff'] ** 2))
        
        # Maximum standardized difference
        max_std_diff = np.max(np.abs(cum_diff['standardized_diff']))
        
        # Area under the cumulative difference curve (absolute)
        auc_diff = np.trapz(np.abs(cum_diff['cumulative_diff']))
        
        # Expected Calibration Error using cumulative method
        # This is more robust than binning-based ECE
        n = len(predicted_probs)
        positions = np.arange(1, n + 1) / n  # Positions along sorted array
        
        # Weight by density of predictions
        weights = np.diff(positions, prepend=0)
        # 修复形状不匹配的问题：确保weights和cumulative_diff长度一致
        if len(weights) > len(cum_diff['cumulative_diff']):
            weights = weights[:-1]
        ece_cumulative = np.sum(weights * np.abs(cum_diff['cumulative_diff'])) if n > 1 else 0
        
        return {
            'max_absolute_difference': max_abs_diff,
            'rms_difference': rms_diff,
            'max_standardized_difference': max_std_diff,
            'area_under_difference_curve': auc_diff,
            'expected_calibration_error_cumulative': ece_cumulative,
            'cumulative_data': cum_diff
        }
    
    def analyze_subgroup_calibration(self, predicted_probs, true_outcomes, metadata, 
                                   subgroup_column):
        """
        Analyze calibration for different subgroups (for fairness analysis)
        """
        subgroup_results = {}
        
        # Extract subgroup information
        subgroups = [meta[subgroup_column] for meta in metadata]
        unique_subgroups = list(set(subgroups))
        
        for subgroup in unique_subgroups:
            # Get indices for this subgroup
            subgroup_mask = np.array([sg == subgroup for sg in subgroups])
            
            if np.sum(subgroup_mask) < 10:  # Skip small subgroups
                continue
                
            # Get data for this subgroup
            subgroup_probs = predicted_probs[subgroup_mask]
            subgroup_outcomes = true_outcomes[subgroup_mask]
            
            # Compute calibration metrics
            metrics = self.compute_calibration_metrics(subgroup_probs, subgroup_outcomes)
            
            # Statistical tests
            ks_result = self.kolmogorov_smirnov_test(subgroup_probs, subgroup_outcomes)
            
            subgroup_results[subgroup] = {
                'n_samples': np.sum(subgroup_mask),
                'mean_predicted': np.mean(subgroup_probs),
                'mean_observed': np.mean(subgroup_outcomes),
                'calibration_metrics': metrics,
                'ks_test': ks_result
            }
        
        return subgroup_results
    
    def plot_cumulative_differences(self, predicted_probs, true_outcomes, title="Calibration Analysis"):
        """
        Plot cumulative differences for visual calibration assessment
        """
        cum_diff = self.compute_cumulative_differences(predicted_probs, true_outcomes)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Cumulative differences
        axes[0, 0].plot(cum_diff['sorted_probs'], cum_diff['cumulative_diff'], 
                       linewidth=2, alpha=0.8, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Predicted Probability (sorted)')
        axes[0, 0].set_ylabel('Cumulative Difference')
        axes[0, 0].set_title('Cumulative Difference Curve')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Standardized cumulative differences
        axes[0, 1].plot(cum_diff['sorted_probs'], cum_diff['standardized_diff'], 
                       linewidth=2, alpha=0.8, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=2, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        axes[0, 1].axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
        axes[0, 1].set_xlabel('Predicted Probability (sorted)')
        axes[0, 1].set_ylabel('Standardized Difference')
        axes[0, 1].set_title('Standardized Cumulative Difference')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Predicted vs Observed (cumulative)
        n = len(predicted_probs)
        positions = np.linspace(0, 1, n)
        axes[1, 0].plot(cum_diff['cumulative_predicted'], cum_diff['cumulative_observed'], 
                       'o-', alpha=0.6, markersize=2)
        axes[1, 0].plot([0, cum_diff['cumulative_predicted'][-1]], 
                       [0, cum_diff['cumulative_predicted'][-1]], 
                       'r--', alpha=0.7, label='Perfect Calibration')
        axes[1, 0].set_xlabel('Cumulative Predicted')
        axes[1, 0].set_ylabel('Cumulative Observed')
        axes[1, 0].set_title('Cumulative Predicted vs Observed')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals distribution
        residuals = cum_diff['sorted_outcomes'] - cum_diff['sorted_probs']
        axes[1, 1].hist(residuals, bins=50, density=True, alpha=0.7, color='purple')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals (Observed - Predicted)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{title} - Cumulative Differences Method', fontsize=16)
        plt.tight_layout()
        plt.savefig('cumulative_calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_subgroup_comparison(self, subgroup_results, metric='max_absolute_difference'):
        """
        Plot calibration metrics comparison across subgroups
        """
        subgroups = list(subgroup_results.keys())
        values = [subgroup_results[sg]['calibration_metrics'][metric] for sg in subgroups]
        sample_sizes = [subgroup_results[sg]['n_samples'] for sg in subgroups]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Calibration metric by subgroup
        bars1 = ax1.bar(subgroups, values, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Subgroup')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'Calibration Quality by Subgroup\n({metric.replace("_", " ").title()})')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 2: Sample sizes
        bars2 = ax2.bar(subgroups, sample_sizes, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('Subgroup')
        ax2.set_ylabel('Sample Size')
        ax2.set_title('Sample Size by Subgroup')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, sample_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_sizes)*0.01,
                    f'{value}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('subgroup_calibration_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_calibration_report(self, predicted_probs, true_outcomes, metadata=None):
        """
        Generate comprehensive calibration analysis report
        """
        print("=" * 60)
        print("CUMULATIVE DIFFERENCES CALIBRATION ANALYSIS REPORT")
        print("=" * 60)
        
        # Overall calibration metrics
        overall_metrics = self.compute_calibration_metrics(predicted_probs, true_outcomes)
        
        print(f"\nOVERALL CALIBRATION METRICS:")
        print(f"{'='*40}")
        print(f"Sample Size: {len(predicted_probs)}")
        print(f"Mean Predicted Probability: {np.mean(predicted_probs):.4f}")
        print(f"Mean Observed Rate: {np.mean(true_outcomes):.4f}")
        print(f"Overall Bias: {np.mean(true_outcomes) - np.mean(predicted_probs):.4f}")
        print()
        
        print(f"CUMULATIVE DIFFERENCE METRICS:")
        print(f"Max Absolute Difference: {overall_metrics['max_absolute_difference']:.4f}")
        print(f"RMS Difference: {overall_metrics['rms_difference']:.4f}")
        print(f"Max Standardized Difference: {overall_metrics['max_standardized_difference']:.4f}")
        print(f"Expected Calibration Error (Cumulative): {overall_metrics['expected_calibration_error_cumulative']:.4f}")
        print(f"Area Under Difference Curve: {overall_metrics['area_under_difference_curve']:.4f}")
        
        # Statistical tests
        ks_result = self.kolmogorov_smirnov_test(predicted_probs, true_outcomes)
        ad_result = self.anderson_darling_test(predicted_probs, true_outcomes)
        
        print(f"\nSTATISTICAL TESTS:")
        print(f"{'='*40}")
        print(f"Kolmogorov-Smirnov Test:")
        print(f"  Statistic: {ks_result['ks_statistic']:.4f}")
        print(f"  P-value: {ks_result['p_value']:.4f}")
        print(f"  Interpretation: {'Well calibrated' if ks_result['p_value'] > 0.05 else 'Poorly calibrated'}")
        print()
        
        print(f"Anderson-Darling Test:")
        print(f"  Statistic: {ad_result['ad_statistic']:.4f}")
        print(f"  Critical Values (15%, 10%, 5%, 2.5%, 1%): {ad_result['critical_values']}")
        
        # Subgroup analysis if metadata provided
        subgroup_results = {}
        if metadata is not None:
            print(f"\nSUBGROUP ANALYSIS:")
            print(f"{'='*40}")
            
            # Analyze by region
            if any('region' in meta for meta in metadata):
                print(f"\nCALIBRATION BY REGION:")
                region_results = self.analyze_subgroup_calibration(
                    predicted_probs, true_outcomes, metadata, 'region'
                )
                subgroup_results['region'] = region_results
                
                for region, results in region_results.items():
                    print(f"\n{region}:")
                    print(f"  Sample Size: {results['n_samples']}")
                    print(f"  Mean Predicted: {results['mean_predicted']:.4f}")
                    print(f"  Mean Observed: {results['mean_observed']:.4f}")
                    print(f"  Bias: {results['mean_observed'] - results['mean_predicted']:.4f}")
                    print(f"  Max Abs Difference: {results['calibration_metrics']['max_absolute_difference']:.4f}")
                    print(f"  KS p-value: {results['ks_test']['p_value']:.4f}")
            
            # Analyze by company size
            if any('company_size' in meta for meta in metadata):
                print(f"\nCALIBRATION BY COMPANY SIZE:")
                size_results = self.analyze_subgroup_calibration(
                    predicted_probs, true_outcomes, metadata, 'company_size'
                )
                subgroup_results['company_size'] = size_results
                
                for size, results in size_results.items():
                    print(f"\n{size} Companies:")
                    print(f"  Sample Size: {results['n_samples']}")
                    print(f"  Mean Predicted: {results['mean_predicted']:.4f}")
                    print(f"  Mean Observed: {results['mean_observed']:.4f}")
                    print(f"  Bias: {results['mean_observed'] - results['mean_predicted']:.4f}")
                    print(f"  Max Abs Difference: {results['calibration_metrics']['max_absolute_difference']:.4f}")
                    print(f"  KS p-value: {results['ks_test']['p_value']:.4f}")
        
        print(f"\n{'='*60}")
        print("INTERPRETATION GUIDE:")
        print("• Max Absolute Difference: Lower is better (< 0.1 good, < 0.05 excellent)")
        print("• RMS Difference: Lower is better (< 0.05 good, < 0.02 excellent)")
        print("• KS p-value: > 0.05 suggests good calibration")
        print("• Bias: Close to 0 is better (difference between observed and predicted rates)")
        print("=" * 60)
        
        # Generate plots
        self.plot_cumulative_differences(predicted_probs, true_outcomes)
        
        if subgroup_results:
            for subgroup_type, results in subgroup_results.items():
                if len(results) > 1:  # Only plot if multiple subgroups
                    self.plot_subgroup_comparison(results)
        
        return {
            'overall_metrics': overall_metrics,
            'statistical_tests': {'ks': ks_result, 'anderson_darling': ad_result},
            'subgroup_results': subgroup_results
        }
    
    def analyze_calibration(self, predicted_probs, true_outcomes, metadata=None):
        """
        Main entry point for calibration analysis
        
        Args:
            predicted_probs (array): Model predicted probabilities
            true_outcomes (array): True binary outcomes
            metadata (list): List of dictionaries with metadata for each sample
            
        Returns:
            dict: Complete calibration analysis results
        """
        print("Running Cumulative Differences Calibration Analysis...")
        print("This method avoids the failure modes of classical binning approaches.")
        
        # Validate inputs
        predicted_probs = np.array(predicted_probs)
        true_outcomes = np.array(true_outcomes).astype(int)
        
        if len(predicted_probs) != len(true_outcomes):
            raise ValueError("Predicted probabilities and true outcomes must have same length")
        
        if metadata is not None and len(metadata) != len(predicted_probs):
            raise ValueError("Metadata must have same length as predictions")
        
        # Check for valid probability range
        if np.any((predicted_probs < 0) | (predicted_probs > 1)):
            raise ValueError("Predicted probabilities must be between 0 and 1")
        
        # Check for valid outcomes
        if not np.all(np.isin(true_outcomes, [0, 1])):
            raise ValueError("True outcomes must be binary (0 or 1)")
        
        # Generate comprehensive report
        results = self.generate_calibration_report(predicted_probs, true_outcomes, metadata)
        
        # Store results for later use
        self.results = results
        
        return results
    
    def compare_models(self, model_results_dict):
        """
        Compare calibration across multiple models
        
        Args:
            model_results_dict (dict): Dictionary with model names as keys and 
                                     (predicted_probs, true_outcomes) tuples as values
        """
        print("\nMODEL CALIBRATION COMPARISON:")
        print("=" * 50)
        
        comparison_results = {}
        
        for model_name, (pred_probs, true_outcomes) in model_results_dict.items():
            metrics = self.compute_calibration_metrics(pred_probs, true_outcomes)
            ks_result = self.kolmogorov_smirnov_test(pred_probs, true_outcomes)
            
            comparison_results[model_name] = {
                'max_abs_diff': metrics['max_absolute_difference'],
                'rms_diff': metrics['rms_difference'],
                'ece_cumulative': metrics['expected_calibration_error_cumulative'],
                'ks_pvalue': ks_result['p_value'],
                'mean_predicted': np.mean(pred_probs),
                'mean_observed': np.mean(true_outcomes),
                'bias': np.mean(true_outcomes) - np.mean(pred_probs)
            }
        
        # Display comparison table
        df_comparison = pd.DataFrame(comparison_results).T
        print(df_comparison.round(4))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(comparison_results.keys())
        
        # Max Absolute Difference
        values = [comparison_results[m]['max_abs_diff'] for m in models]
        axes[0, 0].bar(models, values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Max Absolute Difference')
        axes[0, 0].set_ylabel('Difference')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMS Difference
        values = [comparison_results[m]['rms_diff'] for m in models]
        axes[0, 1].bar(models, values, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('RMS Difference')
        axes[0, 1].set_ylabel('Difference')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Expected Calibration Error
        values = [comparison_results[m]['ece_cumulative'] for m in models]
        axes[1, 0].bar(models, values, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Expected Calibration Error (Cumulative)')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Bias
        values = [comparison_results[m]['bias'] for m in models]
        axes[1, 1].bar(models, values, alpha=0.7, color='gold')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Calibration Bias')
        axes[1, 1].set_ylabel('Bias (Observed - Predicted)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_calibration_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_results

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create test predictions and outcomes
    true_probs = np.random.beta(2, 2, n_samples)  # True underlying probabilities
    predicted_probs = true_probs + np.random.normal(0, 0.1, n_samples)  # Add some miscalibration
    predicted_probs = np.clip(predicted_probs, 0.01, 0.99)  # Keep in valid range
    
    # Generate outcomes based on true probabilities
    true_outcomes = np.random.binomial(1, true_probs, n_samples)
    
    # Create metadata
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    company_sizes = np.random.choice(['Small', 'Medium', 'Large'], n_samples)
    
    metadata = [{'region': regions[i], 'company_size': company_sizes[i]} 
                for i in range(n_samples)]
    
    # Run analysis
    analyzer = CumulativeDifferencesCalibration()
    results = analyzer.analyze_calibration(predicted_probs, true_outcomes, metadata)
    
    print("Test completed successfully!")