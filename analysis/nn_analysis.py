"""
Main Neural Network Training and Analysis System for Time Series
Implements multivariate time-series prediction with bias/fairness analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom analysis modules
from cumulative_differences_calibration import CumulativeDifferencesCalibration
# Additional imports will be added as we create more modules

class TimeSeriesDataGenerator:
    """Generate synthetic multivariate time-series data for analysis"""
    
    def __init__(self, n_samples=10000, n_features=8, sequence_length=50):
        self.n_samples = n_samples
        self.n_features = n_features
        self.sequence_length = sequence_length
        
    def generate_data(self):
        """Generate multivariate time-series with protected attributes"""
        np.random.seed(42)
        
        # Time index
        dates = pd.date_range(start='2020-01-01', periods=self.n_samples, freq='H')
        
        # Generate base time series features
        t = np.arange(self.n_samples)
        
        # Economic indicators (with trends and seasonality)
        gdp_growth = 0.02 + 0.01 * np.sin(2 * np.pi * t / 8760) + np.random.normal(0, 0.005, self.n_samples)
        unemployment = 5.0 - 2.0 * np.sin(2 * np.pi * t / 8760) + np.random.normal(0, 0.3, self.n_samples)
        inflation = 2.0 + 1.5 * np.cos(2 * np.pi * t / 4380) + np.random.normal(0, 0.2, self.n_samples)
        
        # Market indicators
        stock_volatility = 0.15 + 0.05 * np.sin(2 * np.pi * t / 2190) + np.random.normal(0, 0.02, self.n_samples)
        interest_rate = 3.0 + 1.0 * np.cos(2 * np.pi * t / 8760) + np.random.normal(0, 0.1, self.n_samples)
        
        # Technology adoption metrics
        tech_adoption = np.cumsum(np.random.normal(0.001, 0.0005, self.n_samples))
        
        # Protected attributes (for fairness analysis)
        # Geographic region (affects economic conditions)
        regions = np.random.choice(['North', 'South', 'East', 'West'], self.n_samples, 
                                 p=[0.3, 0.25, 0.25, 0.2])
        region_effect = {'North': 1.1, 'South': 0.9, 'East': 1.05, 'West': 0.95}
        region_multiplier = np.array([region_effect[r] for r in regions])
        
        # Company size category
        company_sizes = np.random.choice(['Small', 'Medium', 'Large'], self.n_samples,
                                       p=[0.5, 0.3, 0.2])
        size_effect = {'Small': 0.8, 'Medium': 1.0, 'Large': 1.3}
        size_multiplier = np.array([size_effect[s] for s in company_sizes])
        
        # Create feature matrix
        features = np.column_stack([
            gdp_growth * region_multiplier,
            unemployment,
            inflation,
            stock_volatility,
            interest_rate,
            tech_adoption,
            np.random.normal(0, 1, self.n_samples),  # Random feature
            np.random.exponential(1, self.n_samples)  # Another random feature
        ])
        
        # Target variable: Business success probability (0-1)
        # Influenced by all factors with some non-linear relationships
        linear_component = (
            0.3 * gdp_growth * region_multiplier +
            -0.2 * (unemployment / 10) +
            -0.1 * (inflation / 5) +
            -0.15 * stock_volatility +
            -0.1 * (interest_rate / 5) +
            0.4 * tech_adoption +
            0.1 * np.random.normal(0, 1, self.n_samples)
        )
        
        # Add non-linear interactions
        interaction_term = 0.1 * gdp_growth * tech_adoption * size_multiplier
        
        # Convert to probabilities using sigmoid
        raw_target = linear_component + interaction_term
        target_probs = 1 / (1 + np.exp(-raw_target))
        
        # Generate binary outcomes based on probabilities
        binary_targets = np.random.binomial(1, target_probs)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'gdp_growth': features[:, 0],
            'unemployment': features[:, 1],
            'inflation': features[:, 2],
            'stock_volatility': features[:, 3],
            'interest_rate': features[:, 4],
            'tech_adoption': features[:, 5],
            'random_feature_1': features[:, 6],
            'random_feature_2': features[:, 7],
            'region': regions,
            'company_size': company_sizes,
            'target_probability': target_probs,
            'binary_target': binary_targets
        })
        
        return df
    
    def create_sequences(self, df, target_col='target_probability'):
        """Create sequences for time series prediction"""
        feature_cols = ['gdp_growth', 'unemployment', 'inflation', 'stock_volatility',
                       'interest_rate', 'tech_adoption', 'random_feature_1', 'random_feature_2']
        
        X, y, metadata = [], [], []
        
        for i in range(self.sequence_length, len(df)):
            # Features: previous sequence_length time steps
            X.append(df[feature_cols].iloc[i-self.sequence_length:i].values)
            # Target: current time step
            y.append(df[target_col].iloc[i])
            # Metadata: protected attributes for current time step
            metadata.append({
                'region': df['region'].iloc[i],
                'company_size': df['company_size'].iloc[i],
                'timestamp': df['timestamp'].iloc[i],
                'binary_target': df['binary_target'].iloc[i]
            })
        
        return np.array(X), np.array(y), metadata

class TimeSeriesLSTM(nn.Module):
    """LSTM Neural Network for Time Series Prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(lstm_out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

class NeuralNetworkTrainer:
    """Main training and analysis system"""
    
    def __init__(self, sequence_length=50, hidden_size=64):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize data generator
        self.data_generator = TimeSeriesDataGenerator(
            n_samples=10000, 
            sequence_length=sequence_length
        )
        
        # Initialize analysis modules
        self.calibration_analyzer = CumulativeDifferencesCalibration()
        
    def prepare_data(self):
        """Generate and prepare training data"""
        print("Generating synthetic time-series data...")
        
        # Generate data
        self.df = self.data_generator.generate_data()
        
        # Create sequences
        X, y, self.metadata = self.data_generator.create_sequences(self.df)
        
        # Split data
        X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
            X, y, self.metadata, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
            X_temp, y_temp, meta_temp, test_size=0.25, random_state=42
        )
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train_scaled).to(self.device)
        self.X_val = torch.FloatTensor(X_val_scaled).to(self.device)
        self.X_test = torch.FloatTensor(X_test_scaled).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Store metadata
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test
        
        print(f"Data prepared - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
    def build_model(self):
        """Build and initialize the neural network"""
        input_size = self.X_train.shape[2]  # Number of features
        
        self.model = TimeSeriesLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        print(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train_model(self, epochs=100, batch_size=64):
        """Train the neural network"""
        print("Starting training...")
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X).squeeze()
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_model(self):
        """Evaluate model and prepare for bias analysis"""
        print("Evaluating model...")
        
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            train_pred = self.model(self.X_train).squeeze().cpu().numpy()
            val_pred = self.model(self.X_val).squeeze().cpu().numpy()
            test_pred = self.model(self.X_test).squeeze().cpu().numpy()
            
            # Get true values
            train_true = self.y_train.cpu().numpy()
            val_true = self.y_val.cpu().numpy()
            test_true = self.y_test.cpu().numpy()
        
        # Store predictions for analysis
        self.predictions = {
            'train': {'pred': train_pred, 'true': train_true, 'meta': self.meta_train},
            'val': {'pred': val_pred, 'true': val_true, 'meta': self.meta_val},
            'test': {'pred': test_pred, 'true': test_true, 'meta': self.meta_test}
        }
        
        # Calculate basic metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        
        for split in ['train', 'val', 'test']:
            pred_binary = (self.predictions[split]['pred'] > 0.5).astype(int)
            # 使用二进制目标值而不是连续概率值
            true_binary = np.array([meta['binary_target'] for meta in self.predictions[split]['meta']])
            
            acc = accuracy_score(true_binary, pred_binary)
            auc = roc_auc_score(true_binary, self.predictions[split]['pred'])
            
            print(f"{split.upper()} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        return self.predictions
    
    def run_calibration_analysis(self):
        """Run calibration analysis using cumulative differences method"""
        print("\n=== CALIBRATION ANALYSIS ===")
        
        # Use test set for calibration analysis
        test_data = self.predictions['test']
        
        # Run cumulative differences calibration analysis
        calibration_results = self.calibration_analyzer.analyze_calibration(
            predicted_probs=test_data['pred'],
            true_outcomes=test_data['true'],
            metadata=test_data['meta']
        )
        
        return calibration_results
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete neural network training and bias analysis...")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Build model
        self.build_model()
        
        # Step 3: Train model
        self.train_model(epochs=100, batch_size=64)
        
        # Step 4: Evaluate model
        predictions = self.evaluate_model()
        
        # Step 5: Run calibration analysis
        calibration_results = self.run_calibration_analysis()
        
        print("\nAnalysis complete!")
        return predictions, calibration_results

if __name__ == "__main__":
    # Initialize and run the complete system
    trainer = NeuralNetworkTrainer(sequence_length=50, hidden_size=64)
    predictions, calibration_results = trainer.run_complete_analysis()
    
    print("\nSystem ready for additional bias and fairness analyses...")