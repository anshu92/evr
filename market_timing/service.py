import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_timing.data_engine import DataEngine
from market_timing.model import MarketTimingClassifier
from market_timing.scanner import MarketScanner, PortfolioManager

class MarketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, epochs=5):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
    else:
        print("No trained model found, starting from scratch.")

def generate_report_from_scanner(portfolio_mgr, recommendations):
    """Generates the report files."""
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append(f"MARKET TIMING AI PORTFOLIO REPORT ({datetime.now().strftime('%Y-%m-%d')})")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    report_lines.append(portfolio_mgr.generate_summary())
    
    report_lines.append("\n--- TOP RECOMMENDATIONS ---")
    for rec in recommendations[:10]:
        report_lines.append(f"🟢 {rec['ticker']} (Conf: {rec['prob_bullish']*100:.1f}%) - ${rec['price']:.2f}")
        
    with open("market_report.txt", "w") as f:
        f.write("\n".join(report_lines))
        
    # Email Body
    email_body = "<h2>AI Market Portfolio Update</h2>"
    email_body += "<pre>" + portfolio_mgr.generate_summary() + "</pre>"
    email_body += "<h3>Top Buy Signals</h3>"
    for rec in recommendations[:5]:
        email_body += f"<div style='margin-bottom:10px; padding:5px; border-left: 3px solid green;'>"
        email_body += f"<b>{rec['ticker']}</b>: {rec['prob_bullish']*100:.1f}% Confidence<br>"
        email_body += f"<small>{rec['summary']}</small></div>"
        
    with open("email_body.html", "w") as f:
        f.write(email_body)
    
    print("Report generated: market_report.txt and email_body.html")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "update_and_predict"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_path", default="trained_parameters/market_timing_model.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Engine and Model
    engine = DataEngine()
    model = MarketTimingClassifier().to(device)
    
    # Always load existing model if available
    load_model(model, args.model_path, device)
    
    if args.mode == "train":
        print("Fetching training data...")
        texts, labels = engine.get_training_data()
        
        if not texts:
            print("No data found. Exiting.")
            return

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        train_dataset = MarketDataset(train_texts, train_labels, model.tokenizer)
        val_dataset = MarketDataset(val_texts, val_labels, model.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # Train
        train_model(model, train_loader, val_loader, device, epochs=args.epochs)
        save_model(model, args.model_path)
        
    if args.mode == "update_and_predict":
        print("\nRunning Market Scan & Portfolio Management...")
        scanner = MarketScanner(model_path=args.model_path)
        portfolio_mgr = PortfolioManager()
        
        # 1. Scan Market
        recommendations = scanner.scan_market(limit=None) # No limit, scans all available tickers
        
        # 2. Update Existing Positions (Needs current prices)
        # Quick fetch of current prices for portfolio tickers
        current_prices = {}
        for pos in portfolio_mgr.state["positions"]:
            ticker = pos["ticker"]
            df = scanner.engine.fetch_data(ticker)
            if not df.empty:
                current_prices[ticker] = float(df.iloc[-1]["Close"])
        
        portfolio_mgr.update_positions(current_prices)
        
        # 3. Allocate New Positions
        portfolio_mgr.allocate_new_positions(recommendations)
        
        # 4. Generate Reports
        generate_report_from_scanner(portfolio_mgr, recommendations)

if __name__ == "__main__":
    main()
















