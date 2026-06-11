"""Model lifecycle commands: train, evaluate, predict, explain.

These are plain functions; registration on the top-level app happens in
:mod:`sentinel.cli` so the full command surface stays in one place.
"""

from __future__ import annotations

import typer
from rich.console import Console

from sentinel.config import load_config

console = Console()


def train(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    track: bool = typer.Option(
        False, "--track/--no-track", help="Log params/metrics/artifacts to MLflow."
    ),
    experiment: str = typer.Option(
        "sentinel", "--experiment", help="MLflow experiment name (only used with --track)."
    ),
    mlflow_uri: str | None = typer.Option(
        None, "--mlflow-uri", help="MLflow tracking URI (default: mlflow's built-in, ./mlruns)."
    ),
) -> None:
    """Train a baseline model on SYMBOL's features."""
    from sentinel.models.registry import save_model, train_model
    from sentinel.storage import get_store
    from sentinel.tracking import get_tracker

    store = get_store()
    features = store.read_features(symbol)
    if features.empty:
        console.print(
            f"[red]No features for {symbol}.[/red] Run `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    tracker = get_tracker(track=track, experiment=experiment, tracking_uri=mlflow_uri)
    with tracker.start_run(run_name=f"train__{symbol.upper()}__{model}"):
        tracker.set_tag("command", "train")
        tracker.set_tag("symbol", symbol.upper())
        tracker.set_tag("model", model)
        result = train_model(features, model_name=model, cfg=cfg)
        path = save_model(symbol, model, result)
        tracker.log_params(
            {
                "symbol": symbol.upper(),
                "model": model,
                "random_state": cfg.modeling.random_state,
                "test_fraction": cfg.modeling.test_fraction,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "n_features": len(result.feature_names),
            }
        )
        tracker.log_metrics(
            {
                "holdout_accuracy": result.holdout_accuracy,
                "holdout_f1": result.holdout_f1,
                "holdout_roc_auc": result.holdout_roc_auc,
                "baseline_accuracy": result.baseline_accuracy,
                "class_balance": float(result.metadata.get("class_balance", float("nan"))),
            }
        )
        tracker.log_artifact(path)

    console.print(
        f"[green]✓[/green] Trained [bold]{model}[/bold] on [cyan]{symbol}[/cyan]. "
        f"Holdout accuracy = [bold]{result.holdout_accuracy:.3f}[/bold] "
        f"(baseline = {result.baseline_accuracy:.3f}). Saved to [dim]{path}[/dim]."
    )


def evaluate(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
) -> None:
    """Walk-forward evaluation with full metric report."""
    from sentinel.evaluation.walk_forward import walk_forward_evaluate
    from sentinel.reporting.console import render_evaluation
    from sentinel.storage import get_store

    store = get_store()
    features = store.read_features(symbol)
    if features.empty:
        console.print(
            f"[red]No features for {symbol}.[/red] Run `sentinel features build {symbol}` first."
        )
        raise typer.Exit(code=1)

    report = walk_forward_evaluate(features, model_name=model, cfg=load_config())
    render_evaluation(symbol, model, report)


def predict(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
) -> None:
    """Generate the latest prediction using the saved model."""
    from sentinel.models.registry import load_model, predict_latest
    from sentinel.reporting.console import render_prediction
    from sentinel.storage import get_store

    store = get_store()
    features = store.read_features(symbol)
    if features.empty:
        console.print(f"[red]No features for {symbol}.[/red]")
        raise typer.Exit(code=1)

    artifact = load_model(symbol, model)
    if artifact is None:
        console.print(
            f"[red]No saved {model} model for {symbol}.[/red] "
            f"Run `sentinel train {symbol} --model {model}` first."
        )
        raise typer.Exit(code=1)

    pred = predict_latest(artifact, features)
    render_prediction(symbol, model, pred)


def explain(
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    model: str = typer.Option("logistic", help="Model name: logistic | random_forest | xgboost | lightgbm."),
    method: str = typer.Option(
        "permutation", "--method", help="Importance method: permutation | shap."
    ),
    top: int = typer.Option(20, "--top", help="Rows to render in the table."),
    n_repeats: int = typer.Option(
        10, "--n-repeats", help="Shuffles per feature (permutation method only)."
    ),
    scoring: str = typer.Option(
        "accuracy", "--scoring", help="Score to permute against: accuracy | roc_auc."
    ),
    max_samples: int = typer.Option(
        500, "--max-samples", help="Subsample size for SHAP (ignored for permutation)."
    ),
) -> None:
    """Explain which features the saved model actually leans on.

    Uses the saved ``(symbol, model)`` pipeline. Run ``sentinel train`` first.
    """
    from sentinel.evaluation.importance import permutation_importance, shap_importance
    from sentinel.models.registry import load_model
    from sentinel.reporting.console import render_importance
    from sentinel.storage import get_store

    method = method.lower()
    if method not in {"permutation", "shap"}:
        console.print(f"[red]Unknown method {method!r}.[/red] Use 'permutation' or 'shap'.")
        raise typer.Exit(code=1)

    store = get_store()
    features = store.read_features(symbol)
    if features.empty:
        console.print(f"[red]No features for {symbol}.[/red]")
        raise typer.Exit(code=1)

    artifact = load_model(symbol, model)
    if artifact is None:
        console.print(
            f"[red]No saved {model} model for {symbol}.[/red] "
            f"Run `sentinel train {symbol} --model {model}` first."
        )
        raise typer.Exit(code=1)

    feat_cols = artifact.feature_names
    X = features[feat_cols].astype(float).to_numpy()
    y = features["target_direction"].astype(int).to_numpy()

    if method == "permutation":
        result = permutation_importance(
            artifact.pipeline,
            X,
            y,
            feat_cols,
            n_repeats=n_repeats,
            random_state=load_config().modeling.random_state,
            scoring=scoring,
        )
    else:
        try:
            result = shap_importance(
                artifact.pipeline,
                X,
                feat_cols,
                max_samples=max_samples,
                random_state=load_config().modeling.random_state,
            )
        except ImportError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1) from e

    render_importance(symbol, model, result, top=top)
