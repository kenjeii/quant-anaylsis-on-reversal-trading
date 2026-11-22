"""Portfolio accounting, sizing and risk limits."""
from __future__ import annotations
import math


class Portfolio:
    def __init__(self, config: dict):
        self.config = config
        self.initial_equity = float(config["project"]["initial_equity"])
        self.cash = self.initial_equity
        self.realized_pnl = 0.0
        self.total_costs = 0.0
        self.total_turnover = 0.0
        self.positions: list[dict] = []
        self.recent_signals: list[dict] = []
        self.equity_curve: list[dict] = []

    def mtm_equity(self, prices: dict) -> float:
        equity = self.cash
        for position in self.positions:
            price = prices.get(position["symbol"])
            if price is None:
                continue
            direction = 1 if position["direction"] == "long" else -1
            equity += position["remaining_shares"] * (price - position["entry_price"]) * direction
        return equity

    def open_risk(self) -> float:
        return sum(p["remaining_shares"] * abs(p["entry_price"] - p["stop_price"])
                   for p in self.positions)

    def can_open(self, row, shares: int, entry_price: float, stop_price: float,
                 confirmation_index: int, prices: dict | None = None) -> tuple[bool, str]:
        risk_cfg = self.config["risk"]
        limits = {**risk_cfg, **self.config.get("portfolio", {})}
        if shares <= 0:
            return False, "zero_shares"
        if len(self.positions) >= int(limits["maximum_simultaneous_positions"]):
            return False, "max_positions"
        equity = self.mtm_equity(prices or {})
        if entry_price * shares > equity * risk_cfg["maximum_position_value"]:
            return False, "max_position_value"
        trade_risk = shares * abs(entry_price - stop_price)
        if self.open_risk() + trade_risk > equity * risk_cfg["maximum_total_open_risk"]:
            return False, "max_total_open_risk"
        window = int(risk_cfg["correlated_window_bars"])
        same_direction = [s for s in self.recent_signals
                          if s["direction"] == row["direction"] and
                          confirmation_index - s["index"] <= window]
        for signal in same_direction:
            other_risk = signal["risk"]
            if (trade_risk + other_risk) > equity * risk_cfg["correlated_group_max_risk"]:
                return False, "correlated_same_direction_risk"
        trade_dir = row["direction"]
        regime_cfg = self.config.get("regime", {})
        soft_sizing_only = bool(regime_cfg.get("soft_sizing_only", False))
        use_regime = bool(regime_cfg.get("use_regime_multiplier", True))
        if use_regime:
            if not soft_sizing_only and (
                (trade_dir == "long" and bool(row["long_prohibited"])) or
                (trade_dir == "short" and bool(row["short_prohibited"]))
            ):
                return False, "regime_direction_prohibition"
            if float(row["regime_multiplier"]) <= 0:
                return False, "crisis_no_trade"
        return True, "ok"

    def add_position(self, position: dict) -> None:
        self.positions.append(position)
        entry_notional = position["entry_price"] * position["shares"]
        self.total_turnover += entry_notional
        self.recent_signals.append({"index": position["confirmation_index"],
                                    "direction": position["direction"],
                                    "risk": position["initial_risk"],
                                    "symbol": position["symbol"]})
        # Keep only genuinely recent decisions.
        self.recent_signals = [s for s in self.recent_signals
                               if position["confirmation_index"] - s["index"] <= 20]

    def register_exit(self, position: dict, quantity: int, exit_price: float,
                      fee: float, gross_pnl: float, reason: str, timestamp,
                      raw_gross: float | None = None, exit_slippage: float | None = None) -> dict:
        position["remaining_shares"] -= quantity
        self.cash += gross_pnl - fee
        self.realized_pnl += gross_pnl
        self.total_costs += fee
        self.total_turnover += exit_price * quantity
        record = {"trade_id": position.get("trade_id"), "symbol": position["symbol"],
                  "exit_time": timestamp, "exit_price": exit_price, "exit_reason": reason,
                  "quantity": quantity, "gross_pnl": gross_pnl, "fee": fee,
                  "raw_gross": gross_pnl if raw_gross is None else raw_gross,
                  "exit_slippage": 0.0 if exit_slippage is None else exit_slippage}
        position["exit_records"].append(record)
        if position["remaining_shares"] <= 0 and position in self.positions:
            self.positions.remove(position)
        return record

    def record_equity(self, timestamp, prices: dict) -> None:
        self.equity_curve.append({"timestamp": timestamp, "equity": self.mtm_equity(prices),
                                  "open_positions": len(self.positions)})

    def final_trade_rows(self, trade_meta: dict) -> list[dict]:
        rows = []
        for position in trade_meta.values():
            records = position["exit_records"]
            if not records:
                continue
            gross = sum(r["gross_pnl"] for r in records)
            pre_slippage = sum(r.get("raw_gross", r["gross_pnl"]) for r in records)
            slippage_cost = (float(position.get("entry_slippage_cost", 0.0)) +
                             sum(float(r.get("exit_slippage", 0.0)) for r in records))
            fees = sum(r["fee"] for r in records) + position["entry_fee"]
            first = records[0]
            last = records[-1]
            rows.append({**position["meta"], "shares": position["shares"],
                         "exit_time": last["exit_time"], "exit_price": last["exit_price"],
                         "exit_reason": "|".join(r["exit_reason"] for r in records),
                         "pre_slippage_pnl": round(pre_slippage, 4),
                         "slippage_cost": round(slippage_cost, 4),
                         "gross_pnl": round(gross, 4), "cost": round(fees, 4),
                         "net_pnl": round(gross - fees, 4),
                         "r_multiple": round((gross - fees) / position["initial_risk"], 4)
                         if position["initial_risk"] else None,
                         "holding_bars": position["holding_bars"],
                         "mfe_r": round(position["mfe_r"], 4), "mae_r": round(position["mae_r"], 4)})
        return rows
