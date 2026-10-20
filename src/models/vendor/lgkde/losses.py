
import torch
import torch.nn as nn

class AnomalyDetectionLoss(nn.Module):
    """
    Anomaly Detection Loss based on KDE probabilities.
    For each graph G_i (normal), ensure P_i >= P_i' (negative sample).
    Objective: Maximize (P_i - P_i') / (P_i + epsilon)
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, P_normal, P_negative):
        """
        P_normal: [batch_size]
        P_negative: [batch_size]
        """
        loss = -torch.mean((P_normal - P_negative) / (P_normal + self.epsilon))
        return loss


class LogRatioLossAD(nn.Module):
    """
    Log-ratio based loss for anomaly detection.
    
    L = -mean( log( (P_normal + eps) / (P_negative + eps) ) )
      = -mean( [log(P_normal + eps) - log(P_negative + eps)] )

    This encourages P_normal >> P_negative, but in a more stable log scale.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, P_normal: torch.Tensor, P_negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P_normal: [batch_size] normal scores
            P_negative: [batch_size] negative scores
        Returns:
            loss: a scalar tensor
        """
        log_pos = torch.log(P_normal + self.epsilon)
        log_neg = torch.log(P_negative + self.epsilon)
        return -torch.mean(log_pos - log_neg)
    
    
    
if __name__ == "__main__":
    # Test AnomalyDetectionLoss
    loss_fn = AnomalyDetectionLoss()
    P_normal = torch.tensor([0.9, 0.8, 0.7])
    P_negative = torch.tensor([0.1, 0.2, 0.3])
    loss = loss_fn(P_normal, P_negative)
    print(loss)
    
    # Test LogRatioLossAD
    loss_fn = LogRatioLossAD()
    loss = loss_fn(P_normal, P_negative)
    print(loss)
    
    # Test with negative values
    P_normal = torch.tensor([-0.9, -0.8, -0.7])
    P_negative = torch.tensor([-0.1, -0.2, -0.3])
    loss = loss_fn(P_normal, P_negative)
    print("Not suitable for negative values")
    print(loss)

    # Test with zero values
    P_normal = torch.tensor([0.0, 0.0, 0.0])
    P_negative = torch.tensor([0.0, 0.0, 0.0])
    loss = loss_fn(P_normal, P_negative)
    print("Loss = 0 when P_normal = P_negative")
    print(loss)