import numpy as np
from typing import Dict, Any, List

class BollingerGridStrategy:
    """
    Bollinger Band Grid Trading Strategy.

    This strategy operates based on a grid built around a daily moving average (MA).
    It triggers trades only when the price crosses a grid line, not on repeated
    touches. It includes a risk management feature to halt trading if the price
    breaches a 3-sigma band around the MA.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the strategy with given configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing all strategy parameters.
        """
        # === Indicator & Grid Config ===
        self.ma_period = config.get('ma_period', 20)
        self.sigma_k = config.get('sigma_k', 1)
        self.grid1_num = config.get('grid1_num', 10)
        self.grid2_num = config.get('grid2_num', 6)
        self.grid3_num = config.get('grid3_num', 2)
        self.regrid_interval_min = config.get('regrid_interval_min', 5)

        # === Fee Config ===
        self.fee_rate = config.get('fee_rate', 0.0003)
        self.fee_min = config.get('fee_min', 2.0)
        self.tax_rate = config.get('tax_rate', 0.001)
        self.lot_size = config.get('lot_size', 1)
        self.tick_size = config.get('tick_size', 0.01)

        # === Position Config ===
        self.min_position_pct = config.get('min_position_pct', 0.10)
        self.max_position_pct = config.get('max_position_pct', 0.90)

        # === Runtime State Variables ===
        self.position: int = 0
        self.grid_edges: List[float] = []
        self.active_orders: List[Any] = []
        self.trading_halted: bool = False
        self.last_grid_index: int | None = None
        self.last_regrid_ts: int = -1  # Timestamp of the last grid rebuild

        # Total number of grid cells
        self.total_grids = self.grid1_num + 2 * (self.grid2_num + self.grid3_num)

    def _rebuild_grid(self, ma: float, sigma: float):
        """
        Rebuilds the grid price levels based on the latest MA and Sigma.

        The grid is constructed in several segments and then combined:
        - 3σ to 2σ (lower)
        - 2σ to 1σ (lower)
        - 1σ to +1σ (center)
        - +1σ to +2σ (upper)
        - +2σ to +3σ (upper)
        """
        s = self.sigma_k * sigma

        # Create each grid segment using numpy.linspace
        # The number of points is num_grids + 1
        g3_lower = np.linspace(ma - 3 * s, ma - 2 * s, self.grid3_num + 1)
        g2_lower = np.linspace(ma - 2 * s, ma - 1 * s, self.grid2_num + 1)
        g1_center = np.linspace(ma - 1 * s, ma + 1 * s, self.grid1_num + 1)
        g2_upper = np.linspace(ma + 1 * s, ma + 2 * s, self.grid2_num + 1)
        g3_upper = np.linspace(ma + 2 * s, ma + 3 * s, self.grid3_num + 1)

        # Combine segments, removing duplicate points between them
        grid_edges_np = np.concatenate([
            g3_lower[:-1],
            g2_lower[:-1],
            g1_center[:-1],
            g2_upper[:-1],
            g3_upper
        ])

        self.grid_edges = [float(edge) for edge in grid_edges_np]

    def _get_grid_index(self, price: float) -> int | None:
        """
        Finds the integer index of the grid cell for a given price.

        The index represents the cell number from the bottom. For example,
        a price below the lowest grid edge is in cell 0. A price between
        the lowest and the second-lowest edge is in cell 1, and so on.

        Args:
            price (float): The price to locate within the grid.

        Returns:
            int | None: The integer index of the grid cell, or None if the grid
                        is not yet built.
        """
        if not self.grid_edges:
            return None

        # np.searchsorted finds the index where the price would be inserted
        # to maintain order. This is exactly what we need for the grid index.
        # 'right' means if price is equal to an edge, it's considered in the
        # cell to the right (higher index).
        index = np.searchsorted(self.grid_edges, price, side='right')
        return int(index)

    def on_bar(self, bar_1m: Dict[str, float], ma: float, sigma: float):
        """
        The main entry point for the strategy on each 1-minute bar.
        It executes the following logic in order:
        1. Risk Management: Halts or resumes trading based on 3-sigma breaches.
        2. Grid Refresh: Rebuilds the grid periodically.
        3. Trading Logic: Places orders when the price crosses grid lines.
        """
        price = bar_1m.get('close')
        ts = bar_1m.get('ts')
        if price is None or ts is None:
            return  # Not enough data to proceed

        # 1. Risk Management Check
        s3_upper_bound = ma + 3 * self.sigma_k * sigma
        s3_lower_bound = ma - 3 * self.sigma_k * sigma

        if price > s3_upper_bound or price < s3_lower_bound:
            if not self.trading_halted:
                self.trading_halted = True
                # In a real system, this would trigger portfolio operations.
                # SIMULATE: flat_all() - Liquidate all positions
                # SIMULATE: cancel_all_orders()
                self.position = 0
                print(f"Timestamp {ts}: Price {price:.2f} breached 3-sigma band. Halting trading.")
        else:
            if self.trading_halted:
                self.trading_halted = False
                print(f"Timestamp {ts}: Price {price:.2f} returned within 3-sigma band. Resuming trading.")

        if self.trading_halted:
            return  # Stop all trading activity if halted

        # 2. Grid Refresh Check
        # Note: Assumes timestamp 'ts' is in seconds.
        if self.last_regrid_ts < 0 or (ts - self.last_regrid_ts) >= self.regrid_interval_min * 60:
            self._rebuild_grid(ma, sigma)
            # Reset index after grid changes to prevent false crosses.
            self.last_grid_index = None
            self.last_regrid_ts = ts
            # print(f"Timestamp {ts}: Grid rebuilt around MA {ma:.2f}.")

        if not self.grid_edges:
            return  # Cannot trade without a grid

        # 3. Grid Crossing and Trading Logic
        current_grid_index = self._get_grid_index(price)
        if current_grid_index is None:
            return

        # For the first bar after a grid rebuild, we only set the index.
        if self.last_grid_index is None:
            self.last_grid_index = current_grid_index
            return

        if current_grid_index != self.last_grid_index:
            # A grid line has been crossed.
            grids_crossed = current_grid_index - self.last_grid_index
            order_size = grids_crossed * self.lot_size

            # In a real system, this would create and send an order.
            # SIMULATE: place_order(size=order_size)
            self.position += order_size
            print(f"Timestamp {ts}: Price {price:.2f} crossed grid. Order size: {order_size}. New position: {self.position}")

        # Update state for the next bar
        self.last_grid_index = current_grid_index