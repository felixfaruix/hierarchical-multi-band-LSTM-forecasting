#!/usr/bin/env python3
"""
Example usage of the improved HierForecastNet model.

This script demonstrates how the improved model.py is now more user-friendly
and follows best practices.
"""

import sys
sys.path.append('src')

import torch
from model import HierForecastNet

def main():
    print("=== HierForecastNet Usage Example ===\n")
    
    # Model is now much easier to understand and use
    print("1. Creating model with descriptive parameter names:")
    model = HierForecastNet(
        input_features=10,      # Clear parameter name instead of 'in_f'
        hidden_dim=128,         # Clear parameter name instead of 'hid'  
        dropout_rate=0.1        # Clear parameter name instead of 'p'
    )
    print("   ✓ Model created successfully\n")
    
    print("2. Model provides helpful information:")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Model structure is well documented in docstrings")
    print(f"   - Input validation prevents common errors\n")
    
    print("3. Testing with sample data:")
    batch_size = 4
    
    # Create sample inputs with clear documentation
    x14 = torch.randn(batch_size, 14, 10)  # 14 days of 10 features each
    week_fifo = torch.randn(batch_size, 7, 128)  # 7 weekly tokens history
    month_fifo = torch.randn(batch_size, 12, 128)  # 12 monthly tokens history
    
    print(f"   - Input features shape: {x14.shape}")
    print(f"   - Weekly history shape: {week_fifo.shape}")
    print(f"   - Monthly history shape: {month_fifo.shape}")
    
    # Forward pass with clear output documentation
    outputs = model(x14, week_fifo, month_fifo)
    day_pred, week_pred, month_pred_seq, wk0_tok, week_tok = outputs
    
    print(f"\n4. Model outputs are clearly documented:")
    print(f"   - Daily prediction: {day_pred.shape} (next day value)")
    print(f"   - Weekly prediction: {week_pred.shape} (next 7 days)")
    print(f"   - Monthly prediction: {month_pred_seq.shape} (next 30 days)")
    print(f"   - Week token: {wk0_tok.shape} (for FIFO update)")
    print(f"   - Processed week token: {week_tok.shape} (for FIFO update)")
    
    print("\n5. Input validation helps catch errors:")
    try:
        # This will raise a helpful error message
        bad_model = HierForecastNet(input_features=-5)
    except ValueError as e:
        print(f"   ✓ Caught invalid input: {e}")
    
    print("\n6. Backward compatibility maintained:")
    # Old-style usage still works
    old_style_model = HierForecastNet(10, 64, 0.2)
    print("   ✓ Old-style positional arguments still work")
    
    print("\n=== Summary ===")
    print("The improved model.py now provides:")
    print("• Clear, comprehensive documentation")
    print("• Descriptive parameter and variable names")
    print("• Input validation with helpful error messages")
    print("• Usage examples in all class docstrings")
    print("• Type hints for better IDE support")
    print("• Named constants instead of magic numbers")
    print("• Complete backward compatibility")
    print("\nThe code is now much more user-friendly and follows Python best practices!")

if __name__ == "__main__":
    main()