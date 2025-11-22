"""Time Series & Forecasting"""

from .core import RangeTensor
from .layers import RangeModule, RangeLSTM, RangeLinear

class RangeLSTMForecaster(RangeModule):
    """LSTM forecasting with confidence intervals"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = RangeLSTM(input_dim, hidden_dim)
        self.fc = RangeLinear(hidden_dim, output_dim)
    
    def forward(self, x, uncertainty=0.1):
        x_range = RangeTensor.from_epsilon_ball(x, uncertainty)
        outputs, (h_n, c_n) = self.lstm(x_range)
        forecast = self.fc(h_n)
        return forecast

class RangeGRUForecaster(RangeModule):
    """GRU forecasting"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        from .layers import RangeGRU
        self.gru = RangeGRU(input_dim, hidden_dim)
        self.fc = RangeLinear(hidden_dim, output_dim)
    
    def forward(self, x, uncertainty=0.1):
        x_range = RangeTensor.from_epsilon_ball(x, uncertainty)
        outputs, h_n = self.gru(x_range)
        return self.fc(h_n)

def anomaly_detection(model, timeseries, threshold_width):
    """Detect anomalies by uncertainty spike"""
    predictions = model(timeseries)
    widths = predictions.width()
    
    # Anomalies = predictions with high uncertainty
    anomalies = widths > threshold_width
    return anomalies

def predict_with_intervals(model, data, confidence=0.95):
    """Get prediction intervals"""
    pred_range = model(data)
    min_val, max_val = pred_range.decay()
    
    # Assuming Gaussian, 95% CI ≈ 2 std
    center = pred_range.avg()
    width = pred_range.width()
    
    return {
        'prediction': center,
        'lower': center - width / 2,
        'upper': center + width / 2
    }