def recommend_actions(best_row: dict) -> list:
    """Turn metrics into stakeholder actions."""
    recs = []
    auc = best_row.get("roc_auc", float("nan"))
    f1  = best_row.get("f1", float("nan"))
    model = best_row.get("model", "Model")

    # Threshold-based prescriptions (transparent and safe to claim)
    if auc < 0.6:
        recs.append("Collect richer features (e.g., claim history depth, provider attributes, fraud flags).")
        recs.append("Pilot cost-sensitive learning or calibrated thresholds to prioritize precision on 'Approved'.")
        recs.append("Add geographic market context (premiums/metal mix) and socio-demographics if available.")
    else:
        recs.append(f"Deploy {model} with calibrated threshold; monitor drift weekly.")
        recs.append("Enable human-in-the-loop review for borderline scores (0.45–0.55).")
        recs.append("Track approval turnaround time and adverse outcomes post-decision.")

    recs.append("Create a data dictionary; enforce schema checks to reduce silent data quality issues.")
    recs.append("Log model inputs/outputs for auditability and future error analysis.")
    return recs
