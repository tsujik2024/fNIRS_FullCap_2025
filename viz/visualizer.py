# visualizer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import matplotlib as mpl
import re
from fnirs_FullCap_2025.read.channel_utils import regions as REGION_MAP

mpl.rcParams['figure.max_open_warning'] = 0


class FNIRSVisualizer:
    """fNIRS visualizer producing separate PDFs for each plot type:
    - Combined raw channels (all channels in one PDF)
    - Raw regional plots
    - Raw overall plot
    - Processed overall plot
    - Processed regional plots
    """

    def __init__(self, fs: float = 50.0):
        self.fs = fs
        plt.ioff()
        self.styles = {
            'raw': {'HbO': '#E41A1C', 'HHb': '#377EB8'},
            'processed': {'HbO': '#FF6B6B', 'HHb': '#4ECDC4', 'HHb_linestyle': '--'}
        }

    def _get_time(self, length: int) -> np.ndarray:
        """Return a time axis in seconds for data of given length."""
        return np.arange(length, dtype='float64') / self.fs

    def plot_raw_all_channels(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot every HbO/HHb channel in its own subplot.
        The left-hand title now shows the CH-number (or whatever prefix is
        before the first space in the column name).
        """
        time = self._get_time(len(data))

        # Only keep the HbO / O2Hb columns (HHb titles are inferred)
        hbo_cols = [c for c in data.columns if re.search(r'\b(HbO|O2Hb)\b', c)]
        n = len(hbo_cols)

        fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
        axes = np.atleast_1d(axes)  # always iterable

        for ax, hbo_col in zip(axes, hbo_cols):
            ch = hbo_col.split()[0]  # “CH0”, “Rx1-Tx3”, etc.
            hhb_col = hbo_col.replace('HbO', 'HHb').replace('O2Hb', 'HHb')

            ax.plot(time, pd.to_numeric(data[hbo_col], errors='coerce'),
                    color=self.styles['raw']['HbO'], linewidth=1)
            if hhb_col in data.columns:
                ax.plot(time, pd.to_numeric(data[hhb_col], errors='coerce'),
                        color=self.styles['raw']['HHb'], linewidth=1)

            # ← NEW: channel label
            ax.set_title(ch, loc='left', fontsize=9, fontweight='bold')
            ax.set_ylabel('μM')
            if y_limits:
                ax.set_ylim(y_limits)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    from fnirs_FullCap_2025.read.channel_utils import regions as REGION_MAP
    import re
    import numpy as np
    import os

    def plot_raw_regions(
            self,
            regional_data: Dict[str, pd.DataFrame],
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot raw regional averages.  The title now includes the constituent
        channel numbers, e.g.  “PFC_L  (CH4, CH6, CH7)”.
        """
        # Keep only regions that actually have *_oxy / *_deoxy in the frame
        valid_regions = [
            r for r in regional_data
            if f'{r}_oxy' in regional_data[r].columns
               and f'{r}_deoxy' in regional_data[r].columns
        ]
        if not valid_regions:
            return

        fig, axes = plt.subplots(len(valid_regions), 1,
                                 figsize=(12, 3 * len(valid_regions)),
                                 sharex=True)
        axes = np.atleast_1d(axes)

        for ax, region in zip(axes, valid_regions):
            df = regional_data[region]
            t = self._get_time(len(df))

            ax.plot(t, df[f'{region}_oxy'],
                    color=self.styles['raw']['HbO'], label=f'{region} HbO')
            ax.plot(t, df[f'{region}_deoxy'],
                    color=self.styles['raw']['HHb'], label=f'{region} HHb')

            # ← NEW: list the CH-numbers that defined this region
            ch_list = REGION_MAP.get(region, [])
            # keep only the unique “CH##” prefixes and sort them
            ch_numbers = sorted({re.match(r'CH(\d+)', c).group(1)
                                 for c in ch_list if 'HbO' in c})
            ax.set_title(f"{region}  (CH{', CH'.join(ch_numbers)})", loc='left')
            ax.set_ylabel('μM')
            ax.legend(loc='upper right')
            if y_limits:
                ax.set_ylim(y_limits)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_raw_overall(
        self,
        data: pd.DataFrame,
        output_path: str,
        y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Plot grand mean of raw data."""
        t = self._get_time(len(data))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, data['grand_oxy'], label='Raw Mean HbO',
                color=self.styles['raw']['HbO'])
        ax.plot(t, data['grand_deoxy'], label='Raw Mean HHb',
                color=self.styles['raw']['HHb'])
        ax.set_title('Raw Overall Mean')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('μM')
        ax.legend(loc='upper right')
        if y_limits:
            ax.set_ylim(y_limits)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    # ──────────────────────────────────────────────────────────────
    #  PROCESSED ● OVERALL  (grand mean)
    # ──────────────────────────────────────────────────────────────
    def plot_processed_overall(
        self,
        data: pd.DataFrame,
        output_path: str,
        y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot grand-average (HbO / HHb) of processed data.

        Expects columns:
            grand_oxy   – mean of all *_oxy columns
            grand_deoxy – mean of all *_deoxy columns
        """
        t = self._get_time(len(data))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, data['grand_oxy'],
                label='Processed Mean HbO',
                color=self.styles['processed']['HbO'])
        ax.plot(t, data['grand_deoxy'],
                label='Processed Mean HHb',
                color=self.styles['processed']['HHb'],
                linestyle=self.styles['processed']['HHb_linestyle'])

        ax.set_title('Processed Overall Mean')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('μM')
        ax.legend(loc='upper right')
        if y_limits:
            ax.set_ylim(y_limits)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    # ──────────────────────────────────────────────────────────────
    #  PROCESSED ● BY-REGION
    # ──────────────────────────────────────────────────────────────
    def plot_processed_regions(
        self,
        data: pd.DataFrame,
        output_path: str,
        y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot processed HbO / HHb time-courses for each region.

        Looks for columns named:
            <region>_oxy   and   <region>_deoxy
        where <region> can contain underscores (e.g. PFC_L, M1_R, …).
        """
        # get full prefix before the trailing “_oxy”
        regions = sorted({c[:-4] for c in data.columns if c.endswith('_oxy')})
        if not regions:        # nothing to plot
            return

        t = self._get_time(len(data))
        fig, axs = plt.subplots(len(regions), 1,
                                figsize=(12, 3 * len(regions)),
                                sharex=True)
        axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]

        for ax, region in zip(axs, regions):
            oxy   = f'{region}_oxy'
            deoxy = f'{region}_deoxy'
            if oxy not in data.columns or deoxy not in data.columns:
                continue    # skip incomplete pair

            ax.plot(t, data[oxy],
                    label=f'{region} HbO',
                    color=self.styles['processed']['HbO'])
            ax.plot(t, data[deoxy],
                    label=f'{region} HHb',
                    color=self.styles['processed']['HHb'],
                    linestyle=self.styles['processed']['HHb_linestyle'])

            ax.set_title(region, loc='left')
            ax.set_ylabel('μM')
            ax.legend(loc='upper right')
            if y_limits:
                ax.set_ylim(y_limits)

        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
