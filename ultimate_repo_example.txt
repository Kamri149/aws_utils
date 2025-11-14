Ultimate repo example:

trading-platform/
  pyproject.toml / setup.cfg / requirements.txt

  trading/                     # Python package
    __init__.py

    config/                    # YAML / pydantic configs
      markets/
        binance_spot_eth.yaml
        binance_spot_btc.yaml
        equities_usa.yaml
      bots/
        eth_meanrev_v1.yaml
        btc_trend_v2.yaml
        equities_value_long.yaml

    data/
      loaders/
        binance_trades.py
        binance_klines.py
        equities_prices.py
      schemas.py               # common bar/trade schemas
      lake_layout.py           # S3 paths, Glue table names

    features/
      common_features.py       # generic: returns, vol, skew, etc.
      crypto_features.py       # funding, perp-specific stuff
      equities_features.py     # fundamentals, sectors etc.
      windows.py               # your windowing + metrics / clustering prep
      metrics_registry.py      # the thing we’re building

    backtest/
      engine.py
      portfolio.py
      slippage_fees.py

    execution/
      binance_execution.py
      equities_broker_api.py
      throttling.py
      order_models.py

    risk/
      limits.py
      exposure.py
      stop_out.py

    strategies/
      crypto/
        eth_meanrev/
          strategy.py
          signal.py
          config.py
        btc_trend/
          strategy.py
      equities/
        long_short_value/
          strategy.py
        momentum_intraday/
          strategy.py

    monitoring/
      metrics_emit.py           # push to CloudWatch / Prometheus
      health_checks.py

    infra/
      aws/
        emr/
          cluster_templates/
        lambda/
        step_functions/
        cloudformation/terraform_scripts/

  jobs/                         # entrypoints for EMR / batch / step functions
    build_ohlcv_from_trades.py
    compute_window_metrics_spark.py
    train_clustering_model.py
    run_live_bot.py
    run_paper_trading.py

  notebooks/
    research_crypto/
    research_equities/

  scripts/
    package_for_emr.sh
    deploy_cloudformation.sh
