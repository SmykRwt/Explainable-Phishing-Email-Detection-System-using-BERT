import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import mlflow
from typing import Dict, Any
import torch

from backend.app.ml.dataset import prepare_dataset
from backend.app.ml.classical.pipeline import ClassicalModelWrapper
from backend.app.ml.distilbert.pipeline import DistilBERTPipeline
from backend.app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishing_benchmark")

# Configure MLflow
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment("Phishing_Detection_Classifier_Benchmark")

def run_benchmarks(data_path: str = "spam.csv", epochs: int = 2):
    """Runs train and evaluate cycles for all classical models and DistilBERT."""
    logger.info("Initializing dataset preparation...")
    # Prepare datasets
    train_path, val_path, test_path = prepare_dataset(data_path, "datasets")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # ── Classical Models ──
    models_to_test = ["logistic_regression", "svm", "random_forest", "xgboost"]
    classical_results = {}
    
    X_train, y_train = train_df["text"].tolist(), train_df["label_num"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label_num"].tolist()
    X_val, y_val = val_df["text"].tolist(), val_df["label_num"].tolist()

    # Train splits for classical (combine train and val for broader dataset if appropriate, but keeping test isolated)
    for model_name in models_to_test:
        logger.info(f"Training classical model: {model_name}...")
        with mlflow.start_run(run_name=f"Classical_{model_name}"):
            mlflow.log_param("model_type", model_name)
            
            t0 = time.time()
            wrapper = ClassicalModelWrapper(model_name)
            wrapper.train(X_train, y_train)
            train_time = time.time() - t0
            
            # Predict latency
            t_inf_0 = time.time()
            _ = wrapper.predict(X_test[:100])
            inf_time = (time.time() - t_inf_0) / 100.0  # per sample
            
            # Evaluate on Test
            metrics = wrapper.evaluate(X_test, y_test)
            metrics["train_time_sec"] = train_time
            metrics["inference_time_ms"] = inf_time * 1000.0
            
            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
                
            classical_results[model_name] = {
                "wrapper": wrapper,
                "metrics": metrics
            }
            logger.info(f"Model {model_name} test metrics: {metrics}")

    # ── DistilBERT Model ──
    logger.info("Fine-tuning DistilBERT model...")
    bert_metrics = {}
    bert_wrapper = DistilBERTPipeline(save_dir="./bert_model")
    
    with mlflow.start_run(run_name="Transformer_DistilBERT"):
        mlflow.log_param("model_type", "distilbert")
        mlflow.log_param("epochs", epochs)
        
        t0 = time.time()
        # Train DistilBERT
        bert_wrapper.train(
            train_texts=X_train,
            train_labels=y_train,
            val_texts=X_val,
            val_labels=y_val,
            epochs=epochs,
            batch_size=16,
            lr=2e-5
        )
        train_time = time.time() - t0
        
        # Eval DistilBERT on Test set
        test_enc = bert_wrapper.tokenizer(X_test, truncation=True, padding=True, max_length=128)
        test_ds = torch.utils.data.TensorDataset(
            torch.tensor(test_enc["input_ids"]),
            torch.tensor(test_enc["attention_mask"])
        )
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)
        
        # Measure inference latency
        bert_wrapper.model.eval()
        t_inf_0 = time.time()
        with torch.no_grad():
            for batch in test_loader:
                ids, mask = batch[0].to(bert_wrapper.device), batch[1].to(bert_wrapper.device)
                _ = bert_wrapper.model(input_ids=ids, attention_mask=mask)
        inf_time = (time.time() - t_inf_0) / len(X_test)
        
        # Test evaluation metrics
        probs = bert_wrapper.predict_proba(X_test)
        y_pred = np.argmax(probs, axis=1)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, probs[:, 1])

        bert_metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "train_time_sec": train_time,
            "inference_time_ms": inf_time * 1000.0
        }
        
        for k, v in bert_metrics.items():
            mlflow.log_metric(k, v)
        
        logger.info(f"DistilBERT test metrics: {bert_metrics}")

    # ── Compile Summary Report ──
    summary_data = []
    for model_name, res in classical_results.items():
        m = res["metrics"]
        summary_data.append({
            "Model": model_name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "Train Time (s)": m["train_time_sec"],
            "Inference Time (ms)": m["inference_time_ms"]
        })
        
    summary_data.append({
        "Model": "DistilBERT",
        "Accuracy": bert_metrics["accuracy"],
        "Precision": bert_metrics["precision"],
        "Recall": bert_metrics["recall"],
        "F1": bert_metrics["f1"],
        "ROC-AUC": bert_metrics["roc_auc"],
        "Train Time (s)": bert_metrics["train_time_sec"],
        "Inference Time (ms)": bert_metrics["inference_time_ms"]
    })
    
    summary_df = pd.DataFrame(summary_data)
    os.makedirs("experiments", exist_ok=True)
    summary_df.to_csv("experiments/benchmark_report.csv", index=False)
    
    # Print Table
    logger.info("\n" + "="*80 + "\nMODEL COMPARISON SUMMARY\n" + "="*80)
    logger.info("\n" + summary_df.to_string(index=False))
    
    # ── Select and Save Best Classical Model ──
    best_classical_name = ""
    best_classical_f1 = -1.0
    for name, res in classical_results.items():
        if res["metrics"]["f1"] > best_classical_f1:
            best_classical_f1 = res["metrics"]["f1"]
            best_classical_name = name
            
    best_wrapper = classical_results[best_classical_name]["wrapper"]
    best_wrapper.save("best_model.pkl")
    logger.info(f"Saved best classical model ({best_classical_name}) to best_model.pkl")

    # ── Generate & Save Evaluation Charts ──
    generate_charts(summary_df)
    logger.info("Benchmark charts exported to experiments/")

def generate_charts(df: pd.DataFrame):
    """Generates comparison bar plots and saves to experiments/."""
    os.makedirs("experiments", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Melt dataframe for easy comparison with seaborn
    melted = df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    melted.columns = ["Model", "Metric", "Score"]
    
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.title("Phishing Detection Platform - Model Performance Comparison")
    plt.ylim(0.7, 1.02)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments/model_comparison_metrics.png", dpi=150)
    plt.close()

    # Latency comparison chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="Inference Time (ms)", palette="viridis")
    plt.yscale("log")
    plt.title("Inference Latency Per Sample (Log Scale)")
    plt.ylabel("Inference Time (ms)")
    plt.tight_layout()
    plt.savefig("experiments/model_latency_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_benchmarks(epochs=1)
