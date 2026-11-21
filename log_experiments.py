"""Log executed assignment runs and outputs to local MLflow tracking."""
from __future__ import annotations

import json
from pathlib import Path

from experiment_tracking import configure_tracking, log_run

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_DIR / "artifacts"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing(*paths: Path) -> list[Path]:
    return [path for path in paths if path.is_file()]


def log_completed_runs() -> dict:
    association = _json(ARTIFACTS / "association" / "association_metadata.json")
    feature_mining = association["feature_only_mining"]
    class_mining = association["class_association_rule_mining"]
    model = _json(ARTIFACTS / "models" / "classification_clustering_summary.json")
    spark = _json(ARTIFACTS / "spark" / "pyspark_benchmark_summary.json")
    prep = _json(ARTIFACTS / "processed" / "preprocessing_config.json")

    run_ids = {}
    association_artifacts = _existing(
        ARTIFACTS / "association" / "association_results.md",
        ARTIFACTS / "association" / "apriori_class_rules.csv",
        ARTIFACTS / "association" / "charm_closed_itemsets.csv",
        ARTIFACTS / "association" / "toy_sequential_patterns.csv",
        ARTIFACTS / "association" / "association_metadata.json",
    )
    run_ids["association_mining"] = log_run(
        "Apriori and CHARM",
        params={"feature_apriori": feature_mining["apriori_parameters"],
                "feature_charm": feature_mining["charm_parameters"],
                "class_apriori": class_mining["apriori_parameters"],
                "class_charm": class_mining["charm_parameters"],
                "selected_binary_feature_count": association["selected_binary_feature_count"],
                "transactions": association["rows_as_transactions"]},
        metrics={
            "feature_apriori_itemsets": feature_mining["apriori_frequent_itemset_count"],
            "feature_apriori_rules": feature_mining["apriori_rule_count_at_min_confidence"],
            "feature_charm_closed_itemsets": feature_mining["charm_closed_itemset_count"],
            "feature_charm_verification_passed": int(feature_mining["charm_exhaustive_verification"]["matches_exhaustive"]),
            "class_apriori_itemsets": class_mining["apriori_itemset_count"],
            "class_apriori_rules": class_mining["apriori_class_rule_count_at_min_confidence"],
            "class_charm_closed_itemsets": class_mining["charm_class_augmented_closed_itemset_count"],
            "class_charm_rules": class_mining["charm_class_rule_count_at_min_confidence"],
            "feature_apriori_seconds": feature_mining["apriori_runtime_seconds_excluding_transaction_build"],
            "feature_charm_seconds": feature_mining["charm_runtime_seconds_excluding_transaction_build"],
            "class_apriori_seconds": class_mining["apriori_runtime_seconds_excluding_transaction_build"],
            "class_charm_seconds": class_mining["charm_runtime_seconds_excluding_transaction_build"],
            "class_charm_verification_passed": int(class_mining["charm_exhaustive_verification"]["matches_exhaustive"]),
        },
        tags={"dataset": "UCI Internet Advertisements", "snapshot": "corrected target-separated outputs", "sequence_dataset_mining": "infeasible; no event order"},
        artifacts=association_artifacts,
    )

    tree = model["decision_tree"]
    tree_artifacts = _existing(
        ARTIFACTS / "models" / "classification_clustering_summary.json",
        ARTIFACTS / "models" / "decision_tree_if_then_rules.csv",
        ARTIFACTS / "models" / "decision_tree_rules.txt",
        ARTIFACTS / "models" / "decision_tree_confusion_matrix.png",
        ARTIFACTS / "models" / "decision_tree_roc.png",
        ARTIFACTS / "models" / "decision_tree_structure.png",
        ARTIFACTS / "models" / "decision_tree_test_predictions.csv",
        ARTIFACTS / "models" / "tree_split_metadata.json",
    )
    run_ids["decision_tree"] = log_run(
        "Grouped holdout Decision Tree",
        params={**tree["tree_parameters"],
                "train_rows": tree["n_train"], "test_rows": tree["n_test"],
                "group_signature_overlap": tree["split"]["signature_overlap"],
                "preprocessing": tree["preprocessing"]},
        metrics={
            "accuracy": tree["accuracy"], "balanced_accuracy": tree["balanced_accuracy"],
            "ad_precision": tree["precision_ad"], "ad_recall": tree["recall_ad"],
            "ad_f1": tree["f1_ad"], "roc_auc": tree["roc_auc"],
            "preprocessing_seconds": tree["preprocessing_fit_transform_seconds"],
            "fit_seconds": tree["tree_fit_seconds"], "predict_seconds": tree["prediction_seconds"],
            "if_then_rules": tree["n_if_then_rules"],
        },
        tags={"evaluation": "grouped 80/20 holdout", "preprocessing_fit": "training partition only", "snapshot": "corrected rule probability label"},
        artifacts=tree_artifacts,
    )

    dbscan = model["dbscan"]
    dbscan_artifacts = _existing(
        ARTIFACTS / "models" / "dbscan_metrics.json",
        ARTIFACTS / "models" / "dbscan_parameter_sensitivity.csv",
        ARTIFACTS / "models" / "dbscan_membership_and_noise.csv",
        ARTIFACTS / "models" / "dbscan_noise_points.csv",
        ARTIFACTS / "models" / "dbscan_cluster_class_profile.csv",
        ARTIFACTS / "models" / "dbscan_k_distance.png",
        ARTIFACTS / "models" / "dbscan_geometry_clusters.png",
    )
    run_ids["dbscan"] = log_run(
        "DBSCAN geometry clustering",
        params={"features": dbscan["features_used"], "distance": dbscan["distance"],
                "eps": dbscan["eps_selected"], "min_samples": dbscan["min_samples_selected"],
                "epsilon_method": dbscan["epsilon_method"]},
        metrics={
            "clusters_excluding_noise": dbscan["clusters_excluding_noise"],
            "noise_count": dbscan["noise_count"], "noise_fraction": dbscan["noise_fraction"],
            "silhouette_non_noise_sampled": dbscan["silhouette_non_noise_sampled"],
            "adjusted_rand_vs_class_posthoc": dbscan["adjusted_rand_vs_class_all_rows_posthoc"],
            "target_used_to_fit": int(dbscan["target_used_to_fit"]),
        },
        tags={"target_role": "post-hoc evaluation only", "geometry_preprocessing": dbscan["distance"], "snapshot": "corrected final outputs"},
        artifacts=dbscan_artifacts,
    )

    local_tree = spark.get("local_tree", {})
    local_apriori = spark.get("local_apriori", {})
    spark_artifacts = _existing(
        ARTIFACTS / "spark" / "pyspark_benchmark_summary.json",
        ARTIFACTS / "spark" / "local_apriori_itemsets.csv",
        ARTIFACTS / "spark" / "local_apriori_class_rules.csv",
        ARTIFACTS / "spark" / "spark_fpgrowth_itemsets.csv",
        ARTIFACTS / "spark" / "spark_fpgrowth_rules.csv",
    )
    benchmark_completed = spark.get("status") == "completed"
    spark_metrics = {
        "local_tree_fit_seconds_median": local_tree.get("fit_seconds_median"),
        "local_tree_predict_seconds_median": local_tree.get("predict_seconds_median"),
        "local_apriori_seconds": local_apriori.get("elapsed_seconds"),
        "spark_initialization_completed": int(bool(spark.get("spark_initialization_completed"))),
        "spark_tasks_completed": int(bool(spark.get("spark_tasks_completed", benchmark_completed))),
    }
    spark_tags = {
        "spark_status": spark.get("status", "unknown"),
        "snapshot": "completed Spark benchmark with task-level local comparisons" if benchmark_completed else "Spark benchmark attempt",
    }
    if benchmark_completed:
        spark_tree = spark["spark_tree"]
        spark_fp = spark["spark_fpgrowth"]
        tree_e2e = spark["tree_timing_comparison"]["spark_end_to_end_excluding_startup_seconds"]
        arm_e2e = spark["arm_timing_comparison"]["spark_fp_end_to_end_excluding_startup_seconds"]
        local_tree_e2e = local_tree["fit_seconds_median"] + local_tree["predict_seconds_median"]
        spark_metrics.update({
            "spark_startup_seconds": spark["spark_startup_seconds"],
            "spark_tree_fit_seconds_median": spark_tree["fit_seconds_median"],
            "spark_tree_predict_collect_seconds_median": spark_tree["predict_and_collect_seconds_median"],
            "spark_tree_conversion_materialization_seconds": spark_tree["spark_dataframe_conversion_and_materialization_seconds"],
            "spark_tree_end_to_end_excluding_startup_seconds": tree_e2e,
            "spark_tree_fit_over_local_ratio": spark["tree_timing_comparison"]["spark_fit_over_local_ratio"],
            "spark_tree_end_to_end_over_local_ratio": tree_e2e / local_tree_e2e,
            "spark_tree_accuracy": spark_tree["metrics"]["accuracy"],
            "spark_fp_fit_seconds_median": spark_fp["fit_seconds_median"],
            "spark_fp_collect_seconds_median": spark_fp["collect_itemsets_and_rules_seconds_median"],
            "spark_fp_conversion_materialization_seconds": spark_fp["spark_dataframe_conversion_and_materialization_seconds"],
            "spark_fp_end_to_end_excluding_startup_seconds": arm_e2e,
            "spark_fp_end_to_end_over_local_ratio": arm_e2e / local_apriori["elapsed_seconds"],
            "spark_fp_frequent_itemsets": spark_fp["frequent_itemset_count"],
            "spark_fp_singleton_consequent_rules": spark_fp["association_rule_count"],
        })
        spark_tags["interpretation"] = "Spark completed; slower than matched local task baselines on this small local dataset. Apriori and FP-Growth are different algorithms; local all-consequent rule count differs from Spark singleton-consequent scope."
    else:
        startup_attempt = spark.get("spark_startup_attempt_seconds")
        if startup_attempt is not None:
            spark_metrics["spark_startup_attempt_seconds"] = startup_attempt
        spark_tags["interpretation"] = "Spark workload did not complete; no Spark task timing or speed comparison is recorded."
        if spark.get("spark_error_type"):
            spark_tags["spark_error_type"] = spark["spark_error_type"]
    run_ids["spark_benchmark"] = log_run(
        "Matched PySpark benchmarks" if benchmark_completed else "PySpark benchmark attempt",
        params={"spark_master": spark.get("spark_master"),
                "pyspark_version": spark.get("pyspark_version"),
                "visible_cpu_count": spark.get("hardware", {}).get("available_cpu_count"),
                "rows": spark.get("dataset", {}).get("rows"),
                "tree_parameters": spark.get("parameters", {}).get("tree_spark"),
                "association_parameters": spark.get("parameters", {}).get("arm")},
        metrics=spark_metrics,
        tags=spark_tags,
        artifacts=spark_artifacts,
    )

    run_ids["experiment_id"] = configure_tracking()
    return {
        "tracking_uri": f"sqlite:///{(PROJECT_DIR / 'artifacts' / 'mlflow.db').resolve().as_posix()}",
        "artifact_directory": str(PROJECT_DIR / "artifacts" / "mlflow"),
        "run_ids": run_ids,
        "prepared_rows": prep["normalized_copy"]["rows"],
    }


if __name__ == "__main__":
    print(json.dumps(log_completed_runs(), indent=2))
