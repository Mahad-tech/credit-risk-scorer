import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def explain(model_path: str = "models/credit_model.joblib",
            data_path: str = "data/processed/X_test.csv",
            output_dir: str = "plots"):
    """Generate SHAP summary and waterfall plots."""

    model   = joblib.load(model_path)
    X_test  = pd.read_csv(data_path)

    os.makedirs(output_dir, exist_ok=True)

    # SHAP explainer
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot 1 — Summary plot (all features, all test rows)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {output_dir}/shap_summary.png")

    # Plot 2 — Waterfall plot (single prediction — first test row)
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns.tolist()
    )
    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Waterfall plot saved to {output_dir}/shap_waterfall.png")

if __name__ == "__main__":
    explain()