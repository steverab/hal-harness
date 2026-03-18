"""Epoch AI-style charts for social media sharing."""

import json
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from reliability_eval.constants import PROVIDER_COLORS, PROVIDER_MARKERS
from reliability_eval.loaders.agent_names import (
    get_model_metadata,
    sort_agents_by_provider_and_date,
    strip_agent_prefix,
)
from reliability_eval.metrics.consistency import compute_weighted_r_con
from reliability_eval.plots.helpers import (
    _CI_Z,
    _clip_yerr,
    _get_weighted_r_con_yerr,
    generate_shaded_colors,
)

# ── Static assets ─────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent.parent / "website" / "static"
_HAL_LOGO_PNG = _STATIC_DIR / "logo.png"
_PRINCETON_LOGO_PNG = _STATIC_DIR / "princeton-light.png"
_HAL_LOGO_PDF = _STATIC_DIR / "logo.pdf"
_PRINCETON_LOGO_PDF = _STATIC_DIR / "princeton-light.pdf"

# ── Style constants ───────────────────────────────────────────────────
_FONT_FAMILY = ["DM Sans", "Helvetica", "DejaVu Sans"]
_COLOR_TEXT = "#1a1a1a"
_COLOR_SUBTLE = "#6b7280"
_BG_COLOR = "#fafaf8"
_BAR_HEIGHT = 0.6


def _place_logo(fig, png_path: Path, rect: list, anchor: str = "W"):
    """Reserve space for a logo on *fig* at *rect* [l, b, w, h].

    Creates an invisible axes so that ``bbox_inches='tight'`` preserves the
    area.  The actual vector logo is stamped later by
    :func:`_stamp_vector_logos`.

    *anchor* controls alignment inside the rect when aspect-ratio
    doesn't match: 'W' = left-aligned, 'E' = right-aligned, 'C' = centred.
    """
    logo_ax = fig.add_axes(rect)
    logo_ax.set_anchor(anchor)
    logo_ax.axis("off")
    # Background-colored patch so bbox_inches="tight" accounts for this area
    # (fully invisible patches are excluded from the tight bounding box).
    logo_ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=logo_ax.transAxes,
                                     facecolor=_BG_COLOR, edgecolor="none"))
    return logo_ax


def _stamp_vector_logos(pdf_path: Path, fig, left_margin: float,
                        right_edge: float, logo_y: float, logo_h: float,
                        padding: float):
    """Overlay vector logo PDFs onto *pdf_path* after matplotlib saves it.

    Positions are derived from the figure-fraction coordinates used for the
    placeholder axes (matching the rects passed to ``_place_logo``), then
    converted to PDF points accounting for ``bbox_inches='tight'``.

    The URL text (rendered by matplotlib) is vertically centred in the
    footer; we centre each logo on the same y to keep alignment tight.
    """
    from pypdf import PdfReader, PdfWriter, Transformation

    if not _HAL_LOGO_PDF.exists() or not _PRINCETON_LOGO_PDF.exists():
        return

    # -- coordinate mapping: figure-fraction → final-PDF points ----------
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)          # already in inches
    fig_w = fig.get_figwidth()                   # inches
    fig_h = fig.get_figheight()                  # inches
    tx0 = tight.x0                               # tight-bbox origin (inches)
    ty0 = tight.y0

    def _to_pts(fx, fy):
        return (fx * fig_w - tx0 + padding) * 72, \
               (fy * fig_h - ty0 + padding) * 72

    main = PdfReader(str(pdf_path))
    page = main.pages[0]

    # Vertical centre of the footer (where URL text sits) in PDF pts.
    _, footer_centre_y = _to_pts(0, logo_y + logo_h / 2)

    logo_rect_w = 0.25  # figure-fraction width reserved for each logo

    for logo_pdf, lm, anchor in [
        (_HAL_LOGO_PDF, left_margin, "W"),
        (_PRINCETON_LOGO_PDF, right_edge - logo_rect_w, "E"),
    ]:
        logo = PdfReader(str(logo_pdf))
        lpage = logo.pages[0]
        lw_pts = float(lpage.mediabox.width)
        lh_pts = float(lpage.mediabox.height)

        # Scale to match the placeholder height.
        target_h = logo_h * fig_h * 72
        scale = target_h / lh_pts

        # x from figure-fraction; y centred on footer_centre_y.
        bx, _ = _to_pts(lm, 0)
        by = footer_centre_y - (lh_pts * scale) / 2

        if anchor == "E":
            rect_w_pts = logo_rect_w * fig_w * 72
            bx = bx + rect_w_pts - lw_pts * scale

        page.merge_transformed_page(
            lpage,
            Transformation().scale(scale, scale).translate(bx, by),
        )

    writer = PdfWriter()
    writer.add_page(page)
    writer.write(str(pdf_path))


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived reliability columns if missing."""
    df_sorted = sort_agents_by_provider_and_date(df)

    if "reliability_consistency" not in df_sorted.columns:
        df_sorted["reliability_consistency"] = compute_weighted_r_con(
            df_sorted["consistency_outcome"],
            df_sorted["consistency_trajectory_distribution"],
            df_sorted["consistency_trajectory_sequence"],
            df_sorted["consistency_resource"],
        )
    if "reliability_predictability" not in df_sorted.columns:
        df_sorted["reliability_predictability"] = df_sorted[
            "predictability_brier_score"
        ]
    if "reliability_robustness" not in df_sorted.columns:
        df_sorted["reliability_robustness"] = df_sorted[
            [
                "robustness_fault_injection",
                "robustness_structural",
                "robustness_prompt_variation",
            ]
        ].mean(axis=1, skipna=True)
    if "reliability_overall" not in df_sorted.columns:
        df_sorted["reliability_overall"] = df_sorted[
            [
                "reliability_consistency",
                "reliability_predictability",
                "reliability_robustness",
            ]
        ].mean(axis=1, skipna=True)
    if "provider" not in df_sorted.columns:
        df_sorted["provider"] = df_sorted["agent"].map(
            lambda x: get_model_metadata(x).get("provider", "Unknown")
        )
    return df_sorted


def _draw_panel(ax, labels, values, colors, title, show_xticks=False, xerr=None):
    """Draw one horizontal-bar panel on *ax* in Epoch AI style."""
    n = len(labels)
    y_pos = np.arange(n)

    for j in range(n):
        ax.barh(
            y_pos[j],
            values[j],
            height=_BAR_HEIGHT,
            color=colors[j],
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )

    # Draw error bars on top
    if xerr is not None:
        ax.errorbar(
            values,
            y_pos,
            xerr=xerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, color=_COLOR_TEXT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=4)

    if show_xticks:
        ax.set_xticklabels(
            ["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=_COLOR_SUBTLE
        )
    else:
        ax.set_xticklabels([])

    # Light vertical grid only
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#a0a0a0", linewidth=1.0)
    ax.yaxis.grid(False)
    # Prominent zero line
    ax.axvline(0, color="black", linewidth=1.5, zorder=1.5)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor(_BG_COLOR)


def plot_social_overall_reliability(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Generate an Epoch AI-style horizontal bar chart for social media.

    One panel per benchmark, showing Overall Reliability per model,
    with provider-shaded colors and logos.

    Output: social_overall_reliability.pdf

    Args:
        benchmark_data: List of (benchmark_name, dataframe) tuples.
        output_dir: Directory to save the output image.
        padding: Uniform padding around the plot as a figure fraction
            (0.0 = no padding, 0.06 = 6% on each side).
    """
    if not benchmark_data:
        print("  No benchmark data for social media plot")
        return

    # Temporarily override font for this plot only
    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    # Prepare data for each benchmark panel
    panels = []
    for bm_name, bm_df in benchmark_data:
        df_sorted = _prepare_dataframe(bm_df)
        # Restrict to OpenAI models only
        df_sorted = df_sorted[df_sorted["provider"] == "OpenAI"].reset_index(drop=True)
        valid = df_sorted["reliability_overall"].notna()
        df_valid = df_sorted[valid].copy()
        if len(df_valid) == 0:
            continue

        # Sort by reliability (highest at top of chart)
        df_valid = df_valid.sort_values(
            "reliability_overall", ascending=True
        ).reset_index(drop=True)

        labels = [strip_agent_prefix(a) for a in df_valid["agent"]]
        values = df_valid["reliability_overall"].values
        presorted = sort_agents_by_provider_and_date(df_valid)
        colors = generate_shaded_colors(presorted)
        color_map = dict(zip(presorted["agent"], colors))
        colors_sorted = [color_map.get(a, "#999999") for a in df_valid["agent"]]

        # Propagate SE for reliability_overall = mean(R_con, R_pred, R_rob)
        # SE(mean) = sqrt(se_con^2 + se_pred^2 + se_rob^2) / 3
        se_con = (
            _get_weighted_r_con_yerr(df_valid) / _CI_Z
        )  # undo CI scaling to get raw SE
        se_pred_col = "predictability_brier_score_se"
        se_pred = (
            df_valid[se_pred_col].values
            if se_pred_col in df_valid.columns
            else np.zeros(len(df_valid))
        )
        se_pred = np.where(np.isnan(se_pred), 0, se_pred)
        rob_se_cols = [
            "robustness_fault_injection_se",
            "robustness_structural_se",
            "robustness_prompt_variation_se",
        ]
        rob_existing = [c for c in rob_se_cols if c in df_valid.columns]
        if rob_existing:
            rob_sq = np.zeros(len(df_valid))
            for c in rob_existing:
                se = np.where(np.isnan(df_valid[c].values), 0, df_valid[c].values)
                rob_sq += se**2
            se_rob = np.sqrt(rob_sq) / len(rob_existing)
        else:
            se_rob = np.zeros(len(df_valid))
        xerr = _CI_Z * np.sqrt(se_con**2 + se_pred**2 + se_rob**2) / 3
        xerr = _clip_yerr(xerr, values)

        display_name = _display.get(bm_name, bm_name)
        panels.append((display_name, labels, values, colors_sorted, xerr))

    if not panels:
        print("  No valid panels for social media plot")
        return

    n_panels = len(panels)
    panel_sizes = [len(p[1]) for p in panels]

    # Size: ~0.38in per bar row, plus space for title header and footer logos
    header_height = 1.1  # title + subtitle
    footer_height = 0.6  # logos
    panel_gap = 0.6  # gap between panels
    bar_row_height = 0.38
    body_height = sum(s * bar_row_height for s in panel_sizes) + panel_gap * (
        n_panels - 1
    )
    fig_height = header_height + body_height + footer_height + 0.3
    fig_width = 7

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)

    # Use gridspec: header row (fixed), one row per panel, footer row (fixed)
    height_ratios = (
        [header_height] + [s * bar_row_height for s in panel_sizes] + [footer_height]
    )
    gs = gridspec.GridSpec(
        n_panels + 2,
        1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.28,
        right=0.97,
        top=0.97,
        bottom=0.02,
    )

    # Header: title + subtitle. We render a dummy panel first to find where
    # the y-tick labels start, then align header/footer to that x position.
    # For now, place header text using axes transAxes of the first bar panel
    # (drawn later), so fall back to fig.text at a matching left margin.
    _LEFT_MARGIN = 0.05  # figure-fraction; tuned to align with y-tick labels
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_bbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        header_bbox.y1 - 0.01,
        "More capable models are not more reliable",
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        header_bbox.y0 + header_bbox.height * 0.35,
        "Overall Reliability by model, sorted by score.",
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    # Bar panels
    for i, (title, labels, values, colors, xerr) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i])
        is_last = i == n_panels - 1
        _draw_panel(ax, labels, values, colors, title, show_xticks=is_last, xerr=xerr)
        # Rotated benchmark label on the far left
        ax_bbox = ax.get_position()
        fig.text(
            0.01,
            ax_bbox.y0 + ax_bbox.height / 2,
            title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # Footer: logos
    footer_ax = fig.add_subplot(gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)

    # Position logos within the footer row using inset axes in figure coords
    # Get footer bbox in figure coordinates
    fig.canvas.draw()
    footer_bbox = footer_ax.get_position()

    logo_h = footer_bbox.height * 0.8
    logo_y = footer_bbox.y0 + footer_bbox.height * 0.1
    _RIGHT_EDGE = 0.97  # match gridspec right edge
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )

    # Centered URL in figure coords
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        footer_bbox.y0 + footer_bbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "overall_reliability.pdf"
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    # Restore previous font settings
    for k, v in prev_rc.items():
        plt.rcParams[k] = v


def plot_social_openai_overall(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Epoch AI-style horizontal bar chart of overall reliability for all OpenAI models.

    Output: social_openai_overall.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for GPT 5.2 vs 5.4 social plot")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }
    _OPENAI_GREEN = "#10A37F"

    panels = []
    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        df_prep = df_prep[df_prep["provider"] == "OpenAI"].reset_index(drop=True)
        valid = df_prep["reliability_overall"].notna()
        df_valid = df_prep[valid].copy()
        if len(df_valid) == 0:
            continue

        df_valid = df_valid.sort_values(
            "reliability_overall", ascending=True
        ).reset_index(drop=True)

        labels = [strip_agent_prefix(a) for a in df_valid["agent"]]
        values = df_valid["reliability_overall"].values
        colors = [_OPENAI_GREEN if "5_4" in a else _BG_COLOR for a in df_valid["agent"]]

        xerrs = []
        for _, row in df_valid.iterrows():
            se_con = _se_for_metric(row, df_valid, "reliability_consistency")
            se_pred = _se_for_metric(row, df_valid, "reliability_predictability")
            se_rob = _se_for_metric(row, df_valid, "reliability_robustness")
            xerrs.append(_CI_Z * np.sqrt(se_con**2 + se_pred**2 + se_rob**2) / 3)

        values = np.array(values)
        xerrs = np.array(xerrs)
        xerrs = _clip_yerr(xerrs, values)

        display_name = _display.get(bm_name, bm_name)
        panels.append((display_name, labels, values, colors, xerrs))

    if not panels:
        print("  No valid panels for OpenAI overall reliability social plot")
        for k, v in prev_rc.items():
            plt.rcParams[k] = v
        return

    n_panels = len(panels)
    panel_sizes = [len(p[1]) for p in panels]

    header_height = 1.1
    footer_height = 0.6
    bar_row_height = 0.45
    body_height = sum(s * bar_row_height for s in panel_sizes) + 0.6 * (n_panels - 1)
    fig_height = header_height + body_height + footer_height + 0.3
    fig_width = 7

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)

    height_ratios = (
        [header_height] + [s * bar_row_height for s in panel_sizes] + [footer_height]
    )
    gs = gridspec.GridSpec(
        n_panels + 2,
        1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.28,
        right=0.97,
        top=0.97,
        bottom=0.02,
    )

    _LEFT_MARGIN = 0.05
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_bbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        header_bbox.y1 - 0.01,
        "Overall Reliability across OpenAI Models",
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        header_bbox.y0 + header_bbox.height * 0.35,
        "All OpenAI models compared across benchmarks. GPT 5.4 highlighted.",
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    for i, (title, labels, values, colors, xerr) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i])
        is_last = i == n_panels - 1
        _draw_panel(ax, labels, values, colors, title, show_xticks=is_last, xerr=xerr)
        # Rotated benchmark label on the far left
        ax_bbox = ax.get_position()
        fig.text(
            0.01,
            ax_bbox.y0 + ax_bbox.height / 2,
            title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)

    fig.canvas.draw()
    footer_bbox = footer_ax.get_position()

    logo_h = footer_bbox.height * 0.8
    logo_y = footer_bbox.y0 + footer_bbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )

    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        footer_bbox.y0 + footer_bbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "openai_overall.pdf"
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── Shared model specs & colors for 5.2-vs-5.4 plots ─────────────────
_BENCHMARK_MODELS_52_54 = {
    "gaia": [
        ("gpt_5_2", "GPT 5.2"),
        ("gpt_5_4", "GPT 5.4"),
        ("gpt_5_2_medium", "GPT 5.2 (medium)"),
        ("gpt_5_4_medium", "GPT 5.4 (medium)"),
    ],
    "taubench_airline": [
        ("gpt_5_2", "GPT 5.2"),
        ("gpt_5_4", "GPT 5.4"),
        ("gpt_5_2_xhigh", "GPT 5.2 (xhigh)"),
        ("gpt_5_4_xhigh", "GPT 5.4 (xhigh)"),
    ],
}


_MODEL_COLORS_52_54 = {
    "gpt_5_2": _BG_COLOR,  # background color (5.2 = "empty" bar)
    "gpt_5_2_medium": _BG_COLOR,
    "gpt_5_2_xhigh": _BG_COLOR,
    "gpt_5_4": "#10A37F",  # solid OpenAI green (5.4)
    "gpt_5_4_medium": "#10A37F",
    "gpt_5_4_xhigh": "#10A37F",
}

_REASONING_SUFFIXES_52_54 = {
    "gpt_5_2_medium",
    "gpt_5_2_xhigh",
    "gpt_5_4_medium",
    "gpt_5_4_xhigh",
}

_BM_DISPLAY = {
    "taubench_airline": r"$\tau$-bench",
    "gaia": "GAIA",
}


def _get_model_row(df_prep, suffix):
    """Return the first row whose agent name ends with *suffix*, or None."""
    matches = df_prep[df_prep["agent"].str.endswith(suffix)]
    return matches.iloc[0] if not matches.empty else None


def _se_for_metric(row, df_prep, metric):
    """Return the raw SE (not CI-scaled) for a single model row and metric."""
    if metric == "accuracy":
        se_col = "accuracy_se"
        return (
            0
            if se_col not in df_prep.columns
            else (0 if np.isnan(row.get(se_col, np.nan)) else row[se_col])
        )
    if metric == "reliability_consistency":
        single = df_prep[df_prep["agent"] == row["agent"]]
        return float(_get_weighted_r_con_yerr(single) / _CI_Z)
    if metric == "reliability_predictability":
        se_col = "predictability_brier_score_se"
        return (
            0
            if se_col not in df_prep.columns
            else (0 if np.isnan(row.get(se_col, np.nan)) else row[se_col])
        )
    if metric == "reliability_robustness":
        rob_se_cols = [
            "robustness_fault_injection_se",
            "robustness_structural_se",
            "robustness_prompt_variation_se",
        ]
        existing = [c for c in rob_se_cols if c in df_prep.columns]
        if not existing:
            return 0
        sq = sum((0 if np.isnan(row.get(c, np.nan)) else row[c]) ** 2 for c in existing)
        return np.sqrt(sq) / len(existing)
    return 0


def plot_social_openai_detailed(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Horizontal bar grid showing reliability breakdown for all OpenAI models.

    Layout: rows = benchmarks, columns = metrics.  Each cell shows
    horizontal bars for all OpenAI models with a shared y-axis
    (model labels only on the leftmost column).  Column headers give
    the metric name.

    Output: social_openai_detailed.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for detailed OpenAI social plot")
        return

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }
    _OPENAI_GREEN = "#10A37F"

    metrics = [
        ("accuracy", "Accuracy"),
        ("reliability_consistency", "Consistency"),
        ("reliability_predictability", "Predictability"),
        ("reliability_robustness", "Robustness"),
    ]

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    # ── Collect per-benchmark data ────────────────────────────────────
    # panels[i] = (bm_display, entries)
    # entries[j] = (agent_name, label, [val_per_metric], [yerr_per_metric])
    panels = []
    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        df_prep = df_prep[df_prep["provider"] == "OpenAI"].reset_index(drop=True)
        if len(df_prep) == 0:
            continue

        entries = []
        for _, row in df_prep.iterrows():
            agent = row["agent"]
            display_label = strip_agent_prefix(agent)
            vals, yerrs = [], []
            for col, _ in metrics:
                v = row.get(col, np.nan)
                vals.append(v if not np.isnan(v) else 0)
                yerrs.append(_CI_Z * _se_for_metric(row, df_prep, col))
            entries.append((agent, display_label, vals, yerrs))

        if not entries:
            continue
        panels.append((_display.get(bm_name, bm_name), entries))

    if not panels:
        print("  No valid panels for detailed OpenAI social plot")
        for k, v in prev_rc.items():
            plt.rcParams[k] = v
        return

    # ── Figure layout ─────────────────────────────────────────────────
    n_rows = len(panels)
    n_cols = len(metrics)
    n_models = max(len(entries) for _, entries in panels)

    bar_h = _BAR_HEIGHT
    header_h = 1.2
    footer_h = 0.6
    row_h = 0.45 * n_models + 0.5  # per-benchmark row height
    fig_h = header_h + n_rows * row_h + footer_h + 0.4
    col_w = 2.8
    label_w = 3.0  # extra width for y-tick labels in first column
    fig_w = label_w + (n_cols - 1) * col_w + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)

    # Outer grid: header | body rows | footer
    outer_height_ratios = [header_h] + [row_h] * n_rows + [footer_h]
    outer_gs = gridspec.GridSpec(
        n_rows + 2,
        1,
        figure=fig,
        height_ratios=outer_height_ratios,
        hspace=0.55,
        left=0.15,
        right=0.98,
        top=0.97,
        bottom=0.02,
    )

    _LEFT_MARGIN = 0.02

    # ── Header ────────────────────────────────────────────────────────
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        hbox.y1 - 0.01,
        "Reliability Breakdown across OpenAI Models",
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        hbox.y0 + hbox.height * 0.30,
        "Accuracy and each reliability pillar. GPT 5.4 highlighted.",
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    # ── Body: one sub-grid per benchmark row ──────────────────────────
    for r_idx, (bm_title, entries) in enumerate(panels):
        # Inner grid: 1 row x n_cols, with wider first column for labels
        inner_gs = outer_gs[1 + r_idx].subgridspec(
            1,
            n_cols,
            width_ratios=[label_w] + [col_w] * (n_cols - 1),
            wspace=0.30,
        )

        labels = [lbl for _, lbl, _, _ in entries]
        agents = [a for a, _, _, _ in entries]
        bar_colors = [_OPENAI_GREEN if "5_4" in a else _BG_COLOR for a in agents]
        y_pos = np.arange(len(labels))

        for c_idx, (metric_col, metric_title) in enumerate(metrics):
            ax = fig.add_subplot(inner_gs[0, c_idx])

            vals = np.array([e[2][c_idx] for e in entries])
            yerrs = np.array([e[3][c_idx] for e in entries])
            yerrs = _clip_yerr(yerrs, vals)

            for j in range(len(labels)):
                ax.barh(
                    y_pos[j],
                    vals[j],
                    height=bar_h,
                    color=bar_colors[j],
                    edgecolor="black",
                    linewidth=1,
                    zorder=2,
                )

            # Draw error bars on top
            ax.errorbar(
                vals,
                y_pos,
                xerr=yerrs,
                fmt="none",
                ecolor="#333333",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=3,
            )

            # Shared y-axis: labels only on the first column
            ax.set_yticks(y_pos)
            if c_idx == 0:
                ax.set_yticklabels(labels, fontsize=10, color=_COLOR_TEXT)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=6)

            # X-axis: 0–1 range, ticks only on bottom row
            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
            ax.tick_params(axis="x", length=0, pad=4)
            if r_idx == n_rows - 1:
                ax.set_xticklabels(
                    ["0%", "25%", "50%", "75%", "100%"],
                    fontsize=9,
                    color=_COLOR_SUBTLE,
                )
            else:
                ax.set_xticklabels([])

            # Grid & spines
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, color="#a0a0a0", linewidth=1.0)
            ax.yaxis.grid(False)
            ax.axvline(0, color="black", linewidth=1.5, zorder=1.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor(_BG_COLOR)

            # Column title (metric name) above the top row only
            if r_idx == 0:
                ax.set_title(
                    metric_title,
                    fontsize=12,
                    fontweight="bold",
                    color=_COLOR_TEXT,
                    pad=8,
                )

        # Rotated benchmark label on the far left
        first_ax = fig.axes[-n_cols]  # first axes we just added
        first_bbox = first_ax.get_position()
        fig.text(
            0.01,
            first_bbox.y0 + first_bbox.height / 2,
            bm_title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # ── Footer ────────────────────────────────────────────────────────
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)

    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.99
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "openai_detailed.pdf"
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── JSON column parsers for curve data from CSV ──────────────────────


def _parse_calibration_bins(row):
    """Parse calibration bins from a DataFrame row's JSON column."""
    col = "_calibration_bins_json"
    if col not in row.index:
        return []
    val = row[col]
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_aurc_data(row):
    """Parse AURC (accuracy-coverage) data from DataFrame row JSON columns."""
    cols = ("_aurc_coverages_json", "_aurc_risks_json", "_aurc_optimal_risks_json")
    for col in cols:
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val) or val == "" or val == "[]":
            return None
    try:
        return {
            "coverages": np.array(json.loads(row["_aurc_coverages_json"])),
            "risks": np.array(json.loads(row["_aurc_risks_json"])),
            "optimal_risks": np.array(json.loads(row["_aurc_optimal_risks_json"])),
        }
    except (json.JSONDecodeError, TypeError):
        return None


# ── Line colors for curve plots ──────────────────────────────────────

_LINE_COLOR_52 = "#888888"  # gray for 5.2
_LINE_COLOR_54 = "#10A37F"  # OpenAI green for 5.4


def _get_line_color_52_54(suffix: str) -> str:
    """Return line color based on model suffix."""
    return _LINE_COLOR_54 if "5_4" in suffix else _LINE_COLOR_52


# ── Panel drawing helpers for curve plots ─────────────────────────────


def _style_curve_axes(ax, xlabel=None, ylabel=None, show_xlabel=True):
    """Apply shared styling to a calibration/coverage axes."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=9,
        color=_COLOR_SUBTLE,
    )
    ax.set_yticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=9,
        color=_COLOR_SUBTLE,
    )
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_aspect("equal")
    ax.set_facecolor(_BG_COLOR)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#a0a0a0", linewidth=0.5)
    ax.yaxis.grid(True, color="#a0a0a0", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.5, zorder=1.5)
    ax.axhline(0, color="black", linewidth=1.5, zorder=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel and show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=_COLOR_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=_COLOR_TEXT)


def _draw_calibration_panel(ax, model_data):
    """Draw calibration diagram (confidence vs accuracy) on *ax*.

    *model_data*: list of ``(label, bins, color)`` where *bins* is a list
    of dicts with keys ``avg_confidence``, ``avg_accuracy``, ``count``.
    """
    # Perfect calibration reference line
    ax.plot(
        [0, 1],
        [0, 1],
        color="#cccccc",
        linewidth=1.5,
        linestyle="--",
        zorder=1,
        label="Perfect",
    )

    for label, bins, color in model_data:
        valid = [b for b in bins if b.get("count", 0) > 0]
        if not valid:
            continue
        confs = [b["avg_confidence"] for b in valid]
        accs = [b["avg_accuracy"] for b in valid]
        counts = [b["count"] for b in valid]
        max_count = max(counts) if counts else 1
        sizes = [c / max_count * 200 + 30 for c in counts]

        ax.plot(confs, accs, color=color, linewidth=2, alpha=0.7, zorder=2)
        ax.scatter(
            confs,
            accs,
            s=sizes,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            label=label,
        )

    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)


def _draw_accuracy_coverage_panel(ax, model_data):
    """Draw accuracy-coverage curves on *ax*.

    *model_data*: list of ``(label, aurc_data, color)`` where *aurc_data*
    has keys ``coverages``, ``risks``, ``optimal_risks``.
    """
    for label, data, color in model_data:
        coverages = np.array(data["coverages"])
        accuracies = 1 - np.array(data["risks"])
        optimal = 1 - np.array(data["optimal_risks"])

        ax.plot(coverages, accuracies, color=color, linewidth=2, label=label, zorder=3)
        ax.plot(
            coverages,
            optimal,
            color=color,
            linewidth=1,
            linestyle="--",
            alpha=0.4,
            zorder=2,
        )

    ax.legend(fontsize=9, loc="lower left", framealpha=0.8)


# ── Shared layout for 5.2-vs-5.4 curve plots ─────────────────────────


def _collect_curve_panels(benchmark_data, extractor_fn):
    """Collect per-benchmark panel data for curve plots.

    *extractor_fn(row)* returns curve data for a single model row, or
    ``None`` when data is unavailable.

    Returns a list of ``(bm_display, base_data, reasoning_data)`` tuples
    where each ``*_data`` is ``[(label, curve_data, color), ...]``.
    """
    panels = []
    for bm_name, bm_df in benchmark_data:
        models = _BENCHMARK_MODELS_52_54.get(bm_name)
        if not models:
            continue
        df_prep = _prepare_dataframe(bm_df)

        base_data, reasoning_data = [], []
        for suffix, label in models:
            row = _get_model_row(df_prep, suffix)
            if row is None:
                continue
            curve = extractor_fn(row)
            if curve is None:
                continue
            entry = (label, curve, _get_line_color_52_54(suffix))
            if suffix in _REASONING_SUFFIXES_52_54:
                reasoning_data.append(entry)
            else:
                base_data.append(entry)

        if base_data or reasoning_data:
            panels.append(
                (_BM_DISPLAY.get(bm_name, bm_name), base_data, reasoning_data)
            )
    return panels


def _plot_social_52_54_curves(
    panels,
    output_dir,
    draw_fn,
    title,
    subtitle,
    filename,
    xlabel,
    ylabel,
    *,
    padding=0.5,
):
    """Shared figure layout for 5.2-vs-5.4 curve comparison plots.

    *draw_fn(ax, model_data)* fills a single axes with the appropriate
    curve type (calibration scatter or accuracy-coverage lines).
    """
    if not panels:
        print(f"  No valid panels for {filename}")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    n_rows = len(panels)
    panel_w = 3.8
    header_h = 1.2
    footer_h = 0.6
    row_h = panel_w + 0.5
    fig_w = 2 * panel_w + 2.0
    fig_h = header_h + n_rows * row_h + footer_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    outer_gs = gridspec.GridSpec(
        n_rows + 2,
        1,
        figure=fig,
        height_ratios=[header_h] + [row_h] * n_rows + [footer_h],
        hspace=0.4,
        left=0.12,
        right=0.95,
        top=0.97,
        bottom=0.02,
    )

    _LEFT_MARGIN = 0.03

    # Header
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        hbox.y1 - 0.01,
        title,
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        hbox.y0 + hbox.height * 0.30,
        subtitle,
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    # Body panels
    for r_idx, (bm_title, base_data, reasoning_data) in enumerate(panels):
        inner_gs = outer_gs[1 + r_idx].subgridspec(1, 2, wspace=0.30)
        is_last = r_idx == n_rows - 1

        # Base panel (left)
        ax_base = fig.add_subplot(inner_gs[0, 0])
        draw_fn(ax_base, base_data)
        _style_curve_axes(ax_base, xlabel=xlabel, ylabel=ylabel, show_xlabel=is_last)
        if r_idx == 0:
            ax_base.set_title(
                "Base", fontsize=12, fontweight="bold", color=_COLOR_TEXT, pad=8
            )

        # Reasoning panel (right)
        ax_reas = fig.add_subplot(inner_gs[0, 1])
        draw_fn(ax_reas, reasoning_data)
        _style_curve_axes(ax_reas, xlabel=xlabel, ylabel=None, show_xlabel=is_last)
        if r_idx == 0:
            ax_reas.set_title(
                "Reasoning", fontsize=12, fontweight="bold", color=_COLOR_TEXT, pad=8
            )

        # Rotated benchmark label on the far left
        base_bbox = ax_base.get_position()
        fig.text(
            0.01,
            base_bbox.y0 + base_bbox.height / 2,
            bm_title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / filename
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── Public entry points for calibration / discrimination ──────────────


def plot_social_gpt52_vs_gpt54_calibration(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Calibration diagrams comparing GPT 5.2 vs 5.4.

    For each benchmark, two side-by-side panels: base models (left) and
    reasoning models (right).  Each panel overlays calibration scatter
    (confidence vs accuracy) for both model generations.

    Output: social_gpt52_vs_gpt54_calibration.pdf
    """
    panels = _collect_curve_panels(
        benchmark_data,
        lambda row: _parse_calibration_bins(row) or None,
    )
    _plot_social_52_54_curves(
        panels,
        output_dir,
        _draw_calibration_panel,
        title="GPT 5.4 Improves Calibration over GPT 5.2",
        subtitle="Calibration measures the alignment of a model's expressed confidence and ground truth accuracy. GPT 5.4 (both the base and reasoning variants) shows improved calibration. For GAIA and 5.4 (medium), we observe a switch from an overconfident model to an underconfident model vs 5.2.",
        filename="gpt52_vs_gpt54_calibration.pdf",
        xlabel="Confidence",
        ylabel="Accuracy",
        padding=padding,
    )


def plot_social_gpt52_vs_gpt54_discrimination(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Accuracy-coverage curves comparing GPT 5.2 vs 5.4.

    For each benchmark, two side-by-side panels: base models (left) and
    reasoning models (right).  Each panel overlays accuracy-vs-coverage
    curves (solid) and ideal curves (dashed) for both model generations.

    Output: social_gpt52_vs_gpt54_discrimination.pdf
    """
    panels = _collect_curve_panels(benchmark_data, _parse_aurc_data)
    _plot_social_52_54_curves(
        panels,
        output_dir,
        _draw_accuracy_coverage_panel,
        title="GPT 5.2 vs GPT 5.4: Discrimination",
        subtitle="Accuracy vs coverage, base and reasoning variants.",
        filename="gpt52_vs_gpt54_discrimination.pdf",
        xlabel="Coverage",
        ylabel="Accuracy",
        padding=padding,
    )


def _parse_confidence_densities(row):
    """Parse correct/incorrect confidence arrays from DataFrame row."""
    for col in ("_correct_confidences_json", "_incorrect_confidences_json"):
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val) or val == "" or val == "[]":
            return None
    try:
        correct = json.loads(row["_correct_confidences_json"])
        incorrect = json.loads(row["_incorrect_confidences_json"])
        if not correct and not incorrect:
            return None
        return {
            "correct": np.array(correct),
            "incorrect": np.array(incorrect),
        }
    except (json.JSONDecodeError, TypeError):
        return None


def _draw_density_panel(ax, label, data, color):
    """Draw correct vs incorrect confidence density curves on *ax*.

    *data* has keys ``correct`` and ``incorrect`` (arrays of confidence
    scores).  A Gaussian KDE is fitted to each and the overlap region
    is shaded.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import roc_auc_score

    xs = np.linspace(0, 1, 300)
    correct = data["correct"]
    incorrect = data["incorrect"]

    # We need at least 2 points for a KDE
    has_correct = len(correct) >= 2
    has_incorrect = len(incorrect) >= 2

    # Compute AUROC if both groups are present
    auroc = None
    if has_correct and has_incorrect:
        scores = np.concatenate([correct, incorrect])
        labels = np.concatenate([np.ones(len(correct)), np.zeros(len(incorrect))])
        auroc = roc_auc_score(labels, scores)

    _CORRECT_COLOR = "#3b82f6"  # blue
    _INCORRECT_COLOR = "#f59e0b"  # orange

    if has_correct:
        kde_c = gaussian_kde(correct, bw_method=0.15)
        ys_c = kde_c(xs)
        ax.fill_between(xs, ys_c, alpha=0.25, color=_CORRECT_COLOR, zorder=2)
        ax.plot(xs, ys_c, color=_CORRECT_COLOR, linewidth=2, label="Correct", zorder=3)

    if has_incorrect:
        kde_i = gaussian_kde(incorrect, bw_method=0.15)
        ys_i = kde_i(xs)
        ax.fill_between(xs, ys_i, alpha=0.25, color=_INCORRECT_COLOR, zorder=2)
        ax.plot(
            xs, ys_i, color=_INCORRECT_COLOR, linewidth=2, label="Incorrect", zorder=3
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(
        ["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=_COLOR_SUBTLE
    )
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_facecolor(_BG_COLOR)
    for name, spine in ax.spines.items():
        if name in ("left", "bottom"):
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color(_COLOR_TEXT)
        else:
            spine.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    if auroc is not None:
        ax.text(
            0.03,
            0.67,
            f"AUROC = {auroc:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            color=_COLOR_TEXT,
        )


def plot_social_gpt52_vs_gpt54_discrimination_2(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Correct vs incorrect confidence density plots for GPT 5.2 vs 5.4.

    Four side-by-side panels per benchmark: GPT 5.2 base, GPT 5.2
    reasoning, GPT 5.4 base, GPT 5.4 reasoning.

    Output: social_gpt52_vs_gpt54_discrimination_2.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for discrimination density plot")
        return

    # Collect per-benchmark, per-model density data
    panels = []  # [(bm_display, [(label, data, color, col_idx), ...])]
    for bm_name, bm_df in benchmark_data:
        models = _BENCHMARK_MODELS_52_54.get(bm_name)
        if not models:
            continue
        df_prep = _prepare_dataframe(bm_df)

        entries = []  # (label, data, color, col_idx)
        for suffix, label in models:
            row = _get_model_row(df_prep, suffix)
            if row is None:
                continue
            densities = _parse_confidence_densities(row)
            if densities is None:
                continue
            color = _get_line_color_52_54(suffix)
            is_reasoning = suffix in _REASONING_SUFFIXES_52_54
            is_54 = "5_4" in suffix
            col_idx = int(is_reasoning) * 2 + int(is_54)
            entries.append((label, densities, color, col_idx))

        if entries:
            panels.append((_BM_DISPLAY.get(bm_name, bm_name), entries))

    if not panels:
        print("  No valid panels for discrimination density plot")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    n_rows = len(panels)
    panel_w = 2.6
    header_h = 1.2
    footer_h = 0.6
    row_h = 2.8
    fig_w = 4 * panel_w + 2.0
    fig_h = header_h + n_rows * row_h + footer_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    outer_gs = gridspec.GridSpec(
        n_rows + 2,
        1,
        figure=fig,
        height_ratios=[header_h] + [row_h] * n_rows + [footer_h],
        hspace=0.8,
        left=0.06,
        right=0.97,
        top=0.97,
        bottom=0.02,
    )

    _LEFT_MARGIN = 0.03

    # Header
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        hbox.y1 - 0.01,
        "GPT 5.2 vs GPT 5.4: Score Distributions",
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        hbox.y0 + hbox.height * 0.30,
        "Confidence densities for correct vs incorrect predictions. ",
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    # Body panels
    for r_idx, (bm_title, entries) in enumerate(panels):
        inner_gs = outer_gs[1 + r_idx].subgridspec(1, 4, wspace=0.20)

        # Derive column titles from actual model labels
        col_titles = [None] * 4
        for entry_label, _data, _color, col_idx in entries:
            col_titles[col_idx] = entry_label

        axes = [fig.add_subplot(inner_gs[0, c]) for c in range(4)]
        for entry_label, data, color, col_idx in entries:
            _draw_density_panel(axes[col_idx], entry_label, data, color)

        for c, ax in enumerate(axes):
            ax.set_xlabel("Confidence", fontsize=10, color=_COLOR_TEXT)
            if col_titles[c]:
                ax.set_title(
                    col_titles[c],
                    fontsize=11,
                    fontweight="bold",
                    color=_COLOR_TEXT,
                    pad=8,
                )

        # Rotated benchmark label on the far left
        base_bbox = axes[0].get_position()
        fig.text(
            0.01,
            base_bbox.y0 + base_bbox.height / 2,
            bm_title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "gpt52_vs_gpt54_discrimination_2.pdf"
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


def plot_social_discrimination_all_models(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
    grid_cols: int = 4,
):
    """Correct vs incorrect confidence density plots for ALL models.

    One PDF per benchmark.  Models arranged in a grid (rows x *grid_cols*),
    each cell is a KDE density panel.

    Output: social/discrimination_all_{bm_name}.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for all-model discrimination density plot")
        return

    prev_rc = _social_font_setup()

    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)

        # Collect panels for models that have density data
        panels = []  # [(label, densities, color)]
        for _, row in df_prep.iterrows():
            densities = _parse_confidence_densities(row)
            if densities is None:
                continue
            label = strip_agent_prefix(row["agent"])
            provider = row.get("provider", "Unknown")
            color = PROVIDER_COLORS.get(provider, "#999999")
            panels.append((label, densities, color))

        if not panels:
            print(f"  No density data for {bm_name}")
            continue

        n_panels = len(panels)
        n_cols = min(grid_cols, n_panels)
        n_rows = math.ceil(n_panels / n_cols)

        _LM = 0.03
        header_h, footer_h, row_h = 0.9, 0.5, 2.6
        panel_w = 2.6
        fig_w = n_cols * panel_w + 1.0
        fig_h = header_h + n_rows * row_h + footer_h

        bm_display = _BM_DISPLAY.get(bm_name, bm_name)
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
        outer_gs = gridspec.GridSpec(
            3, 1, figure=fig,
            height_ratios=[header_h, n_rows * row_h, footer_h],
            hspace=0.25, left=0.06, right=0.97, top=0.98, bottom=0.02,
        )
        _add_social_header(
            fig, outer_gs[0],
            f"Score distributions — {bm_display}",
            "Confidence densities for correct vs incorrect predictions "
            "across all models.",
            left_margin=_LM,
        )

        body_gs = outer_gs[1].subgridspec(n_rows, n_cols, wspace=0.20, hspace=0.55)
        for idx, (label, data, color) in enumerate(panels):
            r, c = divmod(idx, n_cols)
            ax = fig.add_subplot(body_gs[r, c])
            _draw_density_panel(ax, label, data, color)
            ax.set_title(
                label, fontsize=10, fontweight="bold",
                color=_COLOR_TEXT, pad=8,
            )
            ax.set_xlabel("Confidence", fontsize=9, color=_COLOR_TEXT)

        # Hide leftover empty cells
        for idx in range(n_panels, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            ax = fig.add_subplot(body_gs[r, c])
            ax.axis("off")

        _add_social_footer_and_save(
            fig, outer_gs[-1], output_dir,
            f"discrimination_all_{bm_name}.pdf",
            left_margin=_LM, padding=padding, prev_rc=None,
        )

    # Restore fonts once after all benchmarks
    if prev_rc:
        for k, v in prev_rc.items():
            plt.rcParams[k] = v


# ── Consistency tile heatmap ──────────────────────────────────────────

# Red (inconsistent) → Orange → Green (consistent), matching the website.
_TILE_CMAP = LinearSegmentedColormap.from_list(
    "consistency_tiles",
    [
        (239 / 255, 68 / 255, 68 / 255),  # red   (sr=0.5)
        (245 / 255, 158 / 255, 11 / 255),  # orange (sr≈0.75)
        (16 / 255, 185 / 255, 129 / 255),  # green  (sr=0 or 1)
    ],
)
_TILE_CMAP.set_bad(color="#e5e5e5")


def _parse_task_outcomes(row):
    """Parse per-task success rates from DataFrame row JSON column."""
    col = "_consistency_task_outcomes_json"
    if col not in row.index:
        return {}
    val = row[col]
    if pd.isna(val) or val == "" or val == "{}":
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_task_levels(row):
    """Parse task difficulty levels from DataFrame row JSON column."""
    col = "_task_levels_json"
    if col not in row.index:
        return {}
    val = row[col]
    if pd.isna(val) or val == "" or val == "{}":
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}


def plot_social_openai_consistency_tiles(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Tile heatmap showing per-task outcome consistency for all OpenAI models.

    For each benchmark: rows = models, columns = tasks (sorted by
    level then task ID).  Each tile is colored green (consistent: always
    pass or always fail) to red (inconsistent: mixed across repetitions).

    Output: social_openai_consistency_tiles.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for consistency tile plot")
        return

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    # ── Collect panel data ────────────────────────────────────────────
    panels = []
    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        df_prep = df_prep[df_prep["provider"] == "OpenAI"].reset_index(drop=True)
        if len(df_prep) == 0:
            continue

        model_entries = []  # [(label, outcomes_dict, agent_name)]
        all_task_ids: set[str] = set()
        task_levels: dict[str, str] = {}

        for _, row in df_prep.iterrows():
            agent = row["agent"]
            label = strip_agent_prefix(agent)
            outcomes = _parse_task_outcomes(row)
            if not outcomes:
                continue
            model_entries.append((label, outcomes, agent))
            all_task_ids.update(outcomes.keys())
            levels = _parse_task_levels(row)
            if levels:
                task_levels.update(levels)

        if not model_entries or not all_task_ids:
            continue

        # Sort tasks: by level (if available), then numerically by ID
        def _sort_key(tid):
            level = task_levels.get(tid, "9")
            try:
                lnum = int(level)
            except ValueError:
                lnum = 9
            try:
                tnum = int(tid)
            except ValueError:
                tnum = 0
            return (lnum, tnum, tid)

        sorted_tasks = sorted(all_task_ids, key=_sort_key)

        # Build matrix: rows = models, cols = tasks
        n_m = len(model_entries)
        n_t = len(sorted_tasks)
        matrix = np.full((n_m, n_t), np.nan)
        labels, agents_list = [], []
        for i, (label, outcomes, agent) in enumerate(model_entries):
            labels.append(label)
            agents_list.append(agent)
            for j, tid in enumerate(sorted_tasks):
                if tid in outcomes:
                    matrix[i, j] = outcomes[tid]

        # Level group boundaries
        level_breaks = []
        if task_levels:
            prev_level = None
            for j, tid in enumerate(sorted_tasks):
                level = task_levels.get(tid, "9")
                if prev_level is not None and level != prev_level:
                    level_breaks.append(j)
                prev_level = level

        panels.append(
            (
                _display.get(bm_name, bm_name),
                labels,
                agents_list,
                matrix,
                sorted_tasks,
                task_levels,
                level_breaks,
            )
        )

    if not panels:
        print("  No valid panels for consistency tile plot")
        return

    # ── Figure layout ─────────────────────────────────────────────────
    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    n_rows = len(panels)
    header_h = 1.2
    footer_h = 0.6
    legend_h = 0.35
    model_row_h = 0.45
    row_padding = 0.8
    row_heights = [len(p[1]) * model_row_h + row_padding for p in panels]
    fig_h = header_h + sum(row_heights) + legend_h + footer_h + 0.4
    fig_w = 10.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    outer_gs = gridspec.GridSpec(
        n_rows + 3,
        1,
        figure=fig,
        height_ratios=[header_h] + row_heights + [legend_h, footer_h],
        hspace=0.4,
        left=0.18,
        right=0.97,
        top=0.97,
        bottom=0.02,
    )

    _LEFT_MARGIN = 0.02

    # ── Header ────────────────────────────────────────────────────────
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        hbox.y1 - 0.01,
        "Outcome Consistency across OpenAI Models",
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        hbox.y0 + hbox.height * 0.30,
        "Per-task consistency across repeated runs. "
        "Green = always same outcome, Red = mixed results.",
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    # ── Body panels ──────────────────────────────────────────────────
    for r_idx, (
        bm_title,
        labels,
        _agents,
        matrix,
        sorted_tasks,
        task_levels,
        level_breaks,
    ) in enumerate(panels):
        ax = fig.add_subplot(outer_gs[1 + r_idx])
        n_m, n_t = matrix.shape

        # Convert success_rate → normalized consistency [0, 1]
        # consistency = max(sr, 1-sr) ∈ [0.5, 1.0] → (c-0.5)*2 ∈ [0, 1]
        norm_matrix = np.where(
            np.isnan(matrix),
            np.nan,
            (np.maximum(matrix, 1 - matrix) - 0.5) * 2,
        )
        norm_masked = np.ma.masked_invalid(norm_matrix)

        ax.imshow(
            norm_masked,
            cmap=_TILE_CMAP,
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )

        # White grid lines between cells
        for i in range(n_m + 1):
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
        for j in range(n_t + 1):
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

        # Level separators (thicker dark lines)
        for brk in level_breaks:
            ax.axvline(brk - 0.5, color=_COLOR_TEXT, linewidth=2)

        # Level group labels above grid
        if task_levels and level_breaks:
            boundaries = [0] + level_breaks + [n_t]
            for k in range(len(boundaries) - 1):
                start = boundaries[k]
                end = boundaries[k + 1]
                center = (start + end - 1) / 2
                level = task_levels.get(sorted_tasks[start], "")
                if level and level != "9":
                    ax.text(
                        center,
                        -0.9,
                        f"Level {level}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=_COLOR_SUBTLE,
                        fontweight="bold",
                    )

        # Y-axis labels
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(labels, fontsize=11, color=_COLOR_TEXT)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(_BG_COLOR)

        # Rotated benchmark label on the far left
        ax_bbox = ax.get_position()
        fig.text(
            0.01,
            ax_bbox.y0 + ax_bbox.height / 2,
            bm_title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # ── Color legend ──────────────────────────────────────────────────
    legend_ax = fig.add_subplot(outer_gs[-2])
    legend_ax.axis("off")
    legend_ax.set_facecolor(_BG_COLOR)

    lbox = legend_ax.get_position()
    gradient_w = 0.15
    gradient_h = lbox.height * 0.4
    gradient_x = (1 - gradient_w) / 2
    gradient_y = lbox.y0 + lbox.height * 0.3

    grad_ax = fig.add_axes([gradient_x, gradient_y, gradient_w, gradient_h])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    grad_ax.imshow(gradient, cmap=_TILE_CMAP, aspect="auto")
    grad_ax.set_xticks([])
    grad_ax.set_yticks([])
    for spine in grad_ax.spines.values():
        spine.set_visible(False)
    fig.text(
        gradient_x - 0.01,
        gradient_y + gradient_h / 2,
        "Inconsistent",
        fontsize=9,
        color=_COLOR_SUBTLE,
        va="center",
        ha="right",
    )
    fig.text(
        gradient_x + gradient_w + 0.01,
        gradient_y + gradient_h / 2,
        "Consistent",
        fontsize=9,
        color=_COLOR_SUBTLE,
        va="center",
        ha="left",
    )

    # ── Footer ────────────────────────────────────────────────────────
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "openai_consistency_tiles.pdf"
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── All-OpenAI single-metric bar charts ───────────────────────────────


def _plot_social_openai_metric(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    metric_col: str,
    se_col: str,
    title: str,
    subtitle: str,
    filename: str,
    *,
    padding: float = 0.5,
):
    """Shared layout for all-OpenAI horizontal bar charts of a single metric."""
    if not benchmark_data:
        print(f"  No benchmark data for {filename}")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    panels = []
    for bm_name, bm_df in benchmark_data:
        df_sorted = _prepare_dataframe(bm_df)
        df_sorted = df_sorted[df_sorted["provider"] == "OpenAI"].reset_index(drop=True)
        valid = df_sorted[metric_col].notna()
        df_valid = df_sorted[valid].copy()
        if len(df_valid) == 0:
            continue

        df_valid = df_valid.sort_values(metric_col, ascending=True).reset_index(
            drop=True
        )

        labels = [strip_agent_prefix(a) for a in df_valid["agent"]]
        values = df_valid[metric_col].values
        # Highlight GPT 5.4 models in OpenAI green, others in background
        _OPENAI_GREEN = "#10A37F"
        colors_sorted = [
            _OPENAI_GREEN if "5_4" in a else _BG_COLOR for a in df_valid["agent"]
        ]

        if se_col and se_col in df_valid.columns:
            se = np.where(np.isnan(df_valid[se_col].values), 0, df_valid[se_col].values)
            xerr = _CI_Z * se
            xerr = _clip_yerr(xerr, values)
        else:
            xerr = None

        display_name = _display.get(bm_name, bm_name)
        panels.append((display_name, labels, values, colors_sorted, xerr))

    if not panels:
        print(f"  No valid panels for {filename}")
        for k, v in prev_rc.items():
            plt.rcParams[k] = v
        return

    n_panels = len(panels)
    panel_sizes = [len(p[1]) for p in panels]

    header_height = 1.1
    footer_height = 0.6
    bar_row_height = 0.38
    body_height = sum(s * bar_row_height for s in panel_sizes) + 0.6 * (n_panels - 1)
    fig_height = header_height + body_height + footer_height + 0.3
    fig_width = 7

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)
    height_ratios = (
        [header_height] + [s * bar_row_height for s in panel_sizes] + [footer_height]
    )
    gs = gridspec.GridSpec(
        n_panels + 2,
        1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.28,
        right=0.97,
        top=0.97,
        bottom=0.02,
    )

    _LEFT_MARGIN = 0.05
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_bbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN,
        header_bbox.y1 - 0.01,
        title,
        fontsize=19,
        fontweight="bold",
        color=_COLOR_TEXT,
        va="top",
        ha="left",
    )
    fig.text(
        _LEFT_MARGIN,
        header_bbox.y0 + header_bbox.height * 0.35,
        subtitle,
        fontsize=12,
        color=_COLOR_SUBTLE,
        va="top",
        ha="left",
        wrap=True,
    )

    for i, (bm_title, labels, values, colors, xerr) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i])
        is_last = i == n_panels - 1
        _draw_panel(
            ax, labels, values, colors, bm_title, show_xticks=is_last, xerr=xerr
        )
        ax_bbox = ax.get_position()
        fig.text(
            0.01,
            ax_bbox.y0 + ax_bbox.height / 2,
            bm_title,
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
            va="center",
            ha="center",
            rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    footer_bbox = footer_ax.get_position()
    logo_h = footer_bbox.height * 0.8
    logo_y = footer_bbox.y0 + footer_bbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E"
    )
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2,
        footer_bbox.y0 + footer_bbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10,
        color=_COLOR_SUBTLE,
        ha="center",
        va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / filename
    fig.savefig(
        output_path,
        dpi=300,
        format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight",
        pad_inches=padding,
    )
    _stamp_vector_logos(output_path, fig, _LEFT_MARGIN, _RIGHT_EDGE, logo_y, logo_h, padding)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


def plot_social_outcome_consistency(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """All-OpenAI horizontal bar chart for outcome consistency.

    Output: social_outcome_consistency.pdf
    """
    _plot_social_openai_metric(
        benchmark_data,
        output_dir,
        metric_col="consistency_outcome",
        se_col="consistency_outcome_se",
        title="Outcome Consistency across OpenAI Models",
        subtitle="Fraction of tasks with consistent outcomes across repeated runs.",
        filename="openai_outcome_consistency.pdf",
        padding=padding,
    )


def plot_social_calibration(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """All-OpenAI horizontal bar chart for calibration.

    Output: social_calibration.pdf
    """
    _plot_social_openai_metric(
        benchmark_data,
        output_dir,
        metric_col="predictability_calibration",
        se_col="predictability_calibration_se",
        title="Calibration across OpenAI Models",
        subtitle="How well confidence scores match actual success rates.",
        filename="openai_calibration.pdf",
        padding=padding,
    )


def plot_social_discrimination(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """All-OpenAI horizontal bar chart for discrimination (AUROC).

    Output: social_discrimination.pdf
    """
    _plot_social_openai_metric(
        benchmark_data,
        output_dir,
        metric_col="predictability_roc_auc",
        se_col="predictability_roc_auc_se",
        title="Discrimination across OpenAI Models",
        subtitle="How well confidence separates correct from incorrect (AUROC).",
        filename="openai_discrimination.pdf",
        padding=padding,
    )


# ── Cross-provider scatter plot infrastructure ───────────────────────

_HIGHLIGHT_SUFFIXES = [
    ("gpt_5_4_medium", "GPT 5.4 (medium)"),
    ("gpt_4_turbo", "GPT-4 Turbo"),
    ("claude_haiku_3_5", "Claude 3.5 Haiku"),
    ("claude_opus_4_5", "Claude 4.5 Opus"),
]


def _social_font_setup():
    """Set social plot fonts; return dict to restore later."""
    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY
    return prev_rc


def _add_social_header(fig, gs_slot, title, subtitle, left_margin=0.05):
    """Draw header with title and subtitle into a gridspec slot."""
    ax = fig.add_subplot(gs_slot)
    ax.axis("off")
    hbox = ax.get_position()
    fig.text(
        left_margin, hbox.y1 - 0.01, title,
        fontsize=19, fontweight="bold", color=_COLOR_TEXT, va="top", ha="left",
    )
    fig.text(
        left_margin, hbox.y0 + hbox.height * 0.40, subtitle,
        fontsize=12, color=_COLOR_SUBTLE, va="top", ha="left", wrap=True,
    )
    # Invisible anchor at the bottom of the header box so that
    # bbox_inches="tight" does not collapse the header–body gap.
    fig.text(left_margin, hbox.y0, " ", fontsize=1, alpha=0)


def _add_social_footer_and_save(
    fig, gs_slot, output_dir, filename, *,
    left_margin=0.05, right_edge=0.97, padding=0.5, prev_rc=None,
):
    """Add logo footer, save to social/ subdir, close fig, restore fonts."""
    footer_ax = fig.add_subplot(gs_slot)
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _place_logo(fig, _HAL_LOGO_PNG, [left_margin, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(
        fig, _PRINCETON_LOGO_PNG,
        [right_edge - 0.25, logo_y, 0.25, logo_h], anchor="E",
    )
    fig.text(
        (left_margin + right_edge) / 2, fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )
    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    path = social_dir / filename
    fig.savefig(
        path, dpi=300, format="pdf", facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    _stamp_vector_logos(path, fig, left_margin, right_edge, logo_y, logo_h,
                        padding)
    print(f"  Saved: {path}")
    plt.close(fig)
    if prev_rc:
        for k, v in prev_rc.items():
            plt.rcParams[k] = v


def _style_social_scatter(
    ax, xlabel="", ylabel="",
    emphasized_y_spine=False, emphasized_spines=False,
):
    """Apply social styling to scatter axes.

    *emphasized_y_spine*: keep only the left spine (horizontal bar charts).
    *emphasized_spines*: keep both left and bottom spines (scatter / density).
    """
    ax.set_facecolor(_BG_COLOR)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#a0a0a0", linewidth=0.5)
    ax.yaxis.grid(True, color="#a0a0a0", linewidth=0.5)
    _emph = set()
    if emphasized_spines:
        _emph = {"left", "bottom"}
    elif emphasized_y_spine:
        _emph = {"left"}
    for name, spine in ax.spines.items():
        if name in _emph:
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color(_COLOR_TEXT)
        else:
            spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=_COLOR_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=_COLOR_TEXT)
    ax.tick_params(
        axis="both", length=0, pad=4, labelsize=9, labelcolor=_COLOR_SUBTLE,
    )


def _aggregate_across_benchmarks(benchmark_data):
    """Average model metrics across benchmarks by display name."""
    records = []
    for _bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        for _, row in df_prep.iterrows():
            records.append({
                "display_name": strip_agent_prefix(row["agent"]),
                "agent": row["agent"],
                "accuracy": row.get("accuracy", np.nan),
                "reliability_overall": row.get("reliability_overall", np.nan),
                "reliability_consistency": row.get(
                    "reliability_consistency", np.nan
                ),
                "reliability_predictability": row.get(
                    "reliability_predictability", np.nan
                ),
                "release_timestamp": row.get("release_timestamp"),
                "provider": row.get("provider", "Unknown"),
            })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.groupby("display_name").agg({
        "accuracy": "mean",
        "reliability_overall": "mean",
        "reliability_consistency": "mean",
        "reliability_predictability": "mean",
        "release_timestamp": "first",
        "provider": "first",
        "agent": "first",
    }).reset_index()


def _place_annotations(ax, annotations, overrides=None):
    """Place bold, provider-colored annotations with overlap avoidance.

    *annotations*: list of ``(x, y, text, color)``.
    *overrides*: optional dict mapping display-name substring to a fixed
    y-offset in points (positive = above, negative = below).
    """
    if not annotations:
        return
    overrides = overrides or {}
    _bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7,
                 edgecolor="none")
    # Candidate y-offsets (x-offset always 0 to centre on dot's x)
    _Y_OFFSETS = [10, -12, 18, -20, 26, -28]
    placed: list[tuple] = []  # (x, y, oy)
    for x, y, text, color in annotations:
        # Check for a manual override first
        manual = None
        for key, offset in overrides.items():
            if key in text:
                manual = offset
                break
        if manual is not None:
            chosen_oy = manual
        else:
            chosen_oy = _Y_OFFSETS[0]
            for oy in _Y_OFFSETS:
                ok = True
                for px, py, poy in placed:
                    if abs(x - px) < 0.1 and abs(y - py) < 0.06:
                        if abs(oy - poy) < 16:
                            ok = False
                            break
                if ok:
                    chosen_oy = oy
                    break
        txt = ax.annotate(
            text, (x, y),
            textcoords="offset points", xytext=(0, chosen_oy),
            fontsize=8, color=color, ha="center", va="center",
            zorder=5, bbox=_bbox,
        )
        # DM Sans / Helvetica have no bold .ttf on this system;
        # DejaVu Sans ships DejaVuSans-Bold.ttf with matplotlib.
        txt.set_fontfamily("DejaVu Sans")
        txt.set_fontweight("bold")
        placed.append((x, y, chosen_oy))


def _draw_provider_scatter(
    ax, df, x_col, y_col, *,
    highlight=True, add_trend=True, show_legend=True,
    highlight_suffixes=None, annotation_overrides=None,
):
    """Draw provider-colored scatter, optionally highlighting specific models."""
    from matplotlib.lines import Line2D

    suffixes = highlight_suffixes if highlight_suffixes is not None else _HIGHLIGHT_SUFFIXES
    annotations: list[tuple] = []

    for provider in ["OpenAI", "Google", "Anthropic"]:
        mask = df["provider"] == provider
        if mask.sum() == 0:
            continue
        sub = df[mask]
        color = PROVIDER_COLORS.get(provider, "#999999")
        marker = PROVIDER_MARKERS.get(provider, "o")

        if highlight:
            hl_mask = sub["agent"].apply(
                lambda a: any(a.endswith(s) for s, _ in suffixes)
            )
            bg = sub[~hl_mask]
            fg = sub[hl_mask]
            if len(bg) > 0:
                ax.scatter(
                    bg[x_col], bg[y_col], c=color, marker=marker,
                    s=40, alpha=0.45, edgecolors="black", linewidth=0.4, zorder=2,
                )
            if len(fg) > 0:
                ax.scatter(
                    fg[x_col], fg[y_col], c=color, marker=marker,
                    s=100, alpha=0.9, edgecolors="black", linewidth=0.8, zorder=4,
                )
                for _, row in fg.iterrows():
                    annotations.append((
                        row[x_col], row[y_col],
                        strip_agent_prefix(row["agent"]),
                        color,
                    ))
        else:
            ax.scatter(
                sub[x_col], sub[y_col], c=color, marker=marker,
                s=60, alpha=0.85, edgecolors="black", linewidth=0.6, zorder=3,
            )

    _place_annotations(ax, annotations, overrides=annotation_overrides)

    if add_trend:
        from scipy import stats as sp_stats

        valid = df[x_col].notna() & df[y_col].notna()
        if valid.sum() >= 3:
            x_vals = df.loc[valid, x_col].values.astype(float)
            y_vals = df.loc[valid, y_col].values.astype(float)
            slope, intercept, r_value, _, _ = sp_stats.linregress(x_vals, y_vals)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(
                x_range, slope * x_range + intercept, "--",
                color="#333333", linewidth=1.5, alpha=0.7, zorder=1,
            )
            ax.annotate(
                f"slope = {slope:+.2f}  r = {r_value:+.2f}",
                xy=(0.03, 0.05),
                xycoords="axes fraction", fontsize=9, fontweight="bold",
                ha="left", va="bottom", color=_COLOR_TEXT,
            )

    if show_legend:
        handles = []
        for prov in ["OpenAI", "Google", "Anthropic"]:
            if (df["provider"] == prov).sum() > 0:
                handles.append(Line2D(
                    [0], [0], marker=PROVIDER_MARKERS.get(prov, "o"),
                    color="w", markerfacecolor=PROVIDER_COLORS.get(prov, "#999"),
                    markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                    label=prov,
                ))
        if handles:
            ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.8)


# ── 1. Release Date vs Accuracy & Reliability vs Accuracy ────────────


def plot_social_date_and_reliability_vs_accuracy(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Side-by-side scatter: release date vs accuracy and reliability vs accuracy.

    Aggregated across benchmarks. All models colored by provider.

    Output: social/date_reliability_vs_accuracy.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for date/reliability vs accuracy plot")
        return

    import matplotlib.dates as mdates
    from scipy import stats as sp_stats
    from matplotlib.lines import Line2D

    agg = _aggregate_across_benchmarks(benchmark_data)
    if agg.empty:
        return

    prev_rc = _social_font_setup()
    _LM = 0.03
    header_h, spacer_h, footer_h, body_h = 0.8, 0.6, 0.6, 4.0
    fig = plt.figure(
        figsize=(9, header_h + body_h + spacer_h + footer_h + 0.2),
        facecolor=_BG_COLOR,
    )
    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[header_h, body_h, spacer_h, footer_h],
        hspace=0.05, left=0.10, right=0.97, top=0.97, bottom=0.02,
    )
    _add_social_header(
        fig, gs[0],
        "Reliability gains lag capability improvements",
        "Release date vs accuracy (left) and accuracy vs overall "
        "reliability (right), averaged across benchmarks.",
        left_margin=_LM,
    )

    inner = gs[1].subgridspec(1, 2, wspace=0.25)
    # Invisible spacer row to absorb rotated date tick labels
    spacer_ax = fig.add_subplot(gs[2])
    spacer_ax.axis("off")
    spacer_ax.set_facecolor(_BG_COLOR)

    # Left: release date vs accuracy
    ax_l = fig.add_subplot(inner[0, 0])
    for prov in ["OpenAI", "Google", "Anthropic"]:
        m = agg["provider"] == prov
        if m.sum() == 0:
            continue
        ax_l.scatter(
            agg.loc[m, "release_timestamp"], agg.loc[m, "accuracy"],
            c=PROVIDER_COLORS.get(prov, "#999"),
            marker=PROVIDER_MARKERS.get(prov, "o"),
            s=60, alpha=0.85, edgecolors="black", linewidth=0.6,
            zorder=3, label=prov,
        )
    valid = agg["release_timestamp"].notna() & agg["accuracy"].notna()
    if valid.sum() >= 3:
        xd = agg.loc[valid, "release_timestamp"]
        xn = (xd - xd.min()).dt.days.values
        yv = agg.loc[valid, "accuracy"].values
        # Regress in years so the slope is human-readable
        xn_yr = xn / 365.25
        sl, ic, rv, _, _ = sp_stats.linregress(xn_yr, yv)
        xr = np.array([xn.min(), xn.max()])
        xr_yr = xr / 365.25
        ax_l.plot(
            [xd.min() + pd.Timedelta(days=d) for d in xr],
            sl * xr_yr + ic, "--", color="#333", linewidth=1.5, alpha=0.7, zorder=1,
        )
        ax_l.annotate(
            f"slope = {sl:+.2f}/yr  r = {rv:+.2f}",
            xy=(0.03, 0.05), xycoords="axes fraction",
            fontsize=9, fontweight="bold", ha="left", va="bottom",
            color=_COLOR_TEXT,
        )
    ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_l.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax_l.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_l.set_ylim(0, 1.05)
    _style_social_scatter(ax_l, xlabel="Release Date", ylabel="Accuracy",
                          emphasized_spines=True)
    ax_l.legend(fontsize=9, loc="upper left", framealpha=0.8)

    # Right: accuracy vs reliability (no legend — left panel already has one)
    ax_r = fig.add_subplot(inner[0, 1])
    _draw_provider_scatter(
        ax_r, agg, "accuracy", "reliability_overall",
        highlight=False, show_legend=False,
    )
    ax_r.set_xlim(0, 1.05)
    ax_r.set_ylim(0, 1.05)
    _style_social_scatter(ax_r, xlabel="Accuracy", ylabel="Overall Reliability",
                          emphasized_spines=True)

    _add_social_footer_and_save(
        fig, gs[-1], output_dir, "date_reliability_vs_accuracy.pdf",
        left_margin=_LM, padding=padding, prev_rc=prev_rc,
    )


# ── 2 & 3. Metric vs Accuracy (aggregated + per-benchmark) ──────────


def _plot_social_metric_vs_accuracy(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    y_col: str,
    y_label: str,
    title: str,
    subtitle: str,
    filename: str,
    per_benchmark: bool = False,
    padding: float = 0.5,
    annotation_overrides=None,
):
    """Shared scatter of a reliability metric vs accuracy."""
    if not benchmark_data:
        print(f"  No benchmark data for {filename}")
        return

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    prev_rc = _social_font_setup()
    _LM = 0.03

    if per_benchmark:
        filtered = [
            (bm, df) for bm, df in benchmark_data
            if bm != "taubench_airline_original"
        ]
        if not filtered:
            for k, v in prev_rc.items():
                plt.rcParams[k] = v
            return

        n_panels = len(filtered)
        header_h, spacer_h, footer_h, panel_h = 0.8, 0.5, 0.6, 3.5
        body_h = n_panels * panel_h + (n_panels - 1) * 0.4  # panels + gaps
        fig_h = header_h + body_h + spacer_h + footer_h + 0.2
        fig = plt.figure(figsize=(7, fig_h), facecolor=_BG_COLOR)
        gs = gridspec.GridSpec(
            4, 1, figure=fig,
            height_ratios=[header_h, body_h, spacer_h, footer_h],
            hspace=0.05, left=0.12, right=0.95, top=0.97, bottom=0.02,
        )
        _add_social_header(fig, gs[0], title, subtitle, left_margin=_LM)
        inner = gs[1].subgridspec(n_panels, 1, hspace=0.20)

        _TAUBENCH_HIGHLIGHTS = _HIGHLIGHT_SUFFIXES + [
            ("gpt_5_4_xhigh", "GPT 5.4 (xhigh)"),
        ]
        for i, (bm_name, bm_df) in enumerate(filtered):
            ax = fig.add_subplot(inner[i])
            df_prep = _prepare_dataframe(bm_df)
            hl = _TAUBENCH_HIGHLIGHTS if "taubench" in bm_name else None
            _draw_provider_scatter(
                ax, df_prep, "accuracy", y_col, highlight_suffixes=hl,
                annotation_overrides=annotation_overrides,
            )
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            show_x = i == n_panels - 1
            _style_social_scatter(
                ax,
                xlabel="Accuracy" if show_x else "",
                ylabel=y_label,
                emphasized_spines=True,
            )
            if not show_x:
                ax.set_xticklabels([])
            ax_bbox = ax.get_position()
            fig.text(
                0.01, ax_bbox.y0 + ax_bbox.height / 2,
                _display.get(bm_name, bm_name),
                fontsize=14, fontweight="bold", color=_COLOR_TEXT,
                va="center", ha="center", rotation=90,
            )

        # Invisible spacer to separate body from footer
        spacer_ax = fig.add_subplot(gs[-2])
        spacer_ax.axis("off")
        spacer_ax.set_facecolor(_BG_COLOR)

        _add_social_footer_and_save(
            fig, gs[-1], output_dir, filename,
            left_margin=_LM, padding=padding, prev_rc=prev_rc,
        )
    else:
        agg = _aggregate_across_benchmarks(benchmark_data)
        if agg.empty:
            for k, v in prev_rc.items():
                plt.rcParams[k] = v
            return

        header_h, spacer_h, footer_h, body_h = 0.8, 0.5, 0.6, 5.0
        fig = plt.figure(
            figsize=(7, header_h + body_h + spacer_h + footer_h + 0.2),
            facecolor=_BG_COLOR,
        )
        gs = gridspec.GridSpec(
            4, 1, figure=fig,
            height_ratios=[header_h, body_h, spacer_h, footer_h],
            hspace=0.05, left=0.12, right=0.95, top=0.97, bottom=0.02,
        )
        _add_social_header(fig, gs[0], title, subtitle, left_margin=_LM)

        ax = fig.add_subplot(gs[1])
        _draw_provider_scatter(ax, agg, "accuracy", y_col,
                               annotation_overrides=annotation_overrides)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        _style_social_scatter(ax, xlabel="Accuracy", ylabel=y_label,
                              emphasized_spines=True)

        # Invisible spacer to separate body from footer
        spacer_ax = fig.add_subplot(gs[2])
        spacer_ax.axis("off")
        spacer_ax.set_facecolor(_BG_COLOR)

        _add_social_footer_and_save(
            fig, gs[-1], output_dir, filename,
            left_margin=_LM, padding=padding, prev_rc=prev_rc,
        )


def plot_social_consistency_vs_accuracy(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Consistency vs accuracy scatter, aggregated across benchmarks.

    Highlights GPT 5.4 (medium), GPT-4 Turbo, Claude 3.5 Haiku, Claude 4.5 Opus.

    Output: social/consistency_vs_accuracy.pdf
    """
    _plot_social_metric_vs_accuracy(
        benchmark_data, output_dir,
        y_col="reliability_consistency",
        y_label="Consistency",
        title="Consistency does not track accuracy",
        subtitle="Consistency vs accuracy, averaged across benchmarks. "
                 "Highlighted: GPT 5.4 (medium), GPT-4 Turbo, "
                 "Claude 3.5 Haiku, Claude 4.5 Opus.",
        filename="consistency_vs_accuracy.pdf",
        padding=padding,
        annotation_overrides={"5.4 (medium)": -14},
    )


def plot_social_consistency_vs_accuracy_by_benchmark(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Consistency vs accuracy scatter, per benchmark.

    Output: social/consistency_vs_accuracy_by_benchmark.pdf
    """
    _plot_social_metric_vs_accuracy(
        benchmark_data, output_dir,
        y_col="reliability_consistency",
        y_label="Consistency",
        title="Consistency does not track accuracy",
        subtitle="Consistency vs accuracy, shown independently per benchmark.",
        filename="consistency_vs_accuracy_by_benchmark.pdf",
        per_benchmark=True,
        padding=padding,
        annotation_overrides={"4.5 Opus": 22, "5.4 (medium)": -22},
    )


def plot_social_predictability_vs_accuracy(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Predictability vs accuracy scatter, aggregated across benchmarks.

    Output: social/predictability_vs_accuracy.pdf
    """
    _plot_social_metric_vs_accuracy(
        benchmark_data, output_dir,
        y_col="reliability_predictability",
        y_label="Predictability",
        title="Predictability has improved with accuracy",
        subtitle="Predictability vs accuracy, averaged across benchmarks.",
        filename="predictability_vs_accuracy.pdf",
        padding=padding,
        annotation_overrides={"4.5 Opus": 14, "5.4 (medium)": -14},
    )


def plot_social_predictability_vs_accuracy_by_benchmark(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Predictability vs accuracy scatter, per benchmark.

    Output: social/predictability_vs_accuracy_by_benchmark.pdf
    """
    _plot_social_metric_vs_accuracy(
        benchmark_data, output_dir,
        y_col="reliability_predictability",
        y_label="Predictability",
        title="Predictability has improved with accuracy",
        subtitle="Predictability vs accuracy, shown independently per benchmark.",
        filename="predictability_vs_accuracy_by_benchmark.pdf",
        per_benchmark=True,
        padding=padding,
        annotation_overrides={"4.5 Opus": 14, "5.4 (medium)": -14},
    )


# ── 4. GAIA 4-panel calibration ──────────────────────────────────────


def plot_social_gaia_calibration_4panel(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Four-panel calibration diagram for GAIA: one per highlighted model.

    Models: GPT 5.4 (medium), GPT-4 Turbo, Claude 3.5 Haiku, Claude 4.5 Opus.

    Output: social/gaia_calibration_4panel.pdf
    """
    gaia_df = None
    for bm_name, bm_df in benchmark_data:
        if bm_name == "gaia":
            gaia_df = bm_df
            break
    if gaia_df is None:
        print("  No GAIA data for 4-panel calibration plot")
        return

    df_prep = _prepare_dataframe(gaia_df)

    # Order: GPT-4 Turbo, GPT 5.4 (medium), Claude 3.5 Haiku, Claude 4.5 Opus
    _CALIB_ORDER = [
        ("gpt_4_turbo", "GPT-4 Turbo"),
        ("gpt_5_4_medium", "GPT 5.4 (medium)"),
        ("claude_haiku_3_5", "Claude 3.5 Haiku"),
        ("claude_opus_4_5", "Claude 4.5 Opus"),
    ]
    # Legend positions per panel index — nudged to avoid axis overlap
    # Left column: upper left but shifted right; right column: shifted up
    _LEGEND_BBOX = {
        0: {"loc": "upper left", "bbox_to_anchor": (0.05, 0.98)},
        1: {"loc": "lower right", "bbox_to_anchor": (0.98, 0.06)},
        2: {"loc": "upper left", "bbox_to_anchor": (0.05, 0.98)},
        3: {"loc": "lower right", "bbox_to_anchor": (0.98, 0.06)},
    }

    model_panels = []
    for suffix, label in _CALIB_ORDER:
        row = _get_model_row(df_prep, suffix)
        if row is None:
            continue
        bins = _parse_calibration_bins(row)
        if not bins:
            continue
        provider = row.get("provider", "Unknown")
        color = PROVIDER_COLORS.get(provider, "#999999")
        model_panels.append((label, bins, color))

    if not model_panels:
        print("  No calibration data for highlighted models on GAIA")
        return

    prev_rc = _social_font_setup()
    _LM = 0.03
    header_h, footer_h = 0.8, 0.6
    panel_w = 3.8
    panel_h = panel_w + 0.3
    n_rows = (len(model_panels) + 1) // 2
    body_h = n_rows * panel_h
    fig_w = 2 * panel_w + 2.0
    fig_h = header_h + body_h + footer_h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    gs = gridspec.GridSpec(
        n_rows + 2, 1, figure=fig,
        height_ratios=[header_h] + [panel_h] * n_rows + [footer_h],
        hspace=0.30, left=0.10, right=0.95, top=0.97, bottom=0.02,
    )
    _add_social_header(
        fig, gs[0],
        "Calibration across providers on GAIA",
        "Confidence vs accuracy calibration curves. "
        "Points sized by bin count. Dashed line = perfect calibration.",
        left_margin=_LM,
    )

    inner_gs = None
    for i, (label, bins, color) in enumerate(model_panels):
        row_idx = i // 2
        col_idx = i % 2
        if col_idx == 0:
            inner_gs = gs[1 + row_idx].subgridspec(1, 2, wspace=0.30)
        ax = fig.add_subplot(inner_gs[0, col_idx])
        _draw_calibration_panel(ax, [(label, bins, color)])
        _style_curve_axes(
            ax, xlabel="Confidence",
            ylabel="Accuracy" if col_idx == 0 else None,
        )
        # Widen limits so circles aren't clipped
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            label, fontsize=12, fontweight="bold", color=_COLOR_TEXT, pad=8,
        )
        # Override legend position per panel
        handles, labels = ax.get_legend_handles_labels()
        lkw = _LEGEND_BBOX.get(i, {"loc": "lower right"})
        ax.legend(handles, labels, fontsize=9, framealpha=0.8, **lkw)

    _add_social_footer_and_save(
        fig, gs[-1], output_dir, "gaia_calibration_4panel.pdf",
        left_margin=_LM, padding=padding, prev_rc=prev_rc,
    )


# ── 5. GAIA levels: accuracy & mean actions by difficulty ─────────────


def plot_social_gaia_levels(
    df: pd.DataFrame,
    all_metrics,
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Side-by-side accuracy and mean actions on GAIA by difficulty level.

    Horizontal bar chart with shared model y-axis, grouped by GAIA difficulty.
    Requires *all_metrics* (list of ReliabilityMetrics) for per-level data.

    Output: social/gaia_levels.pdf
    """
    if all_metrics is None:
        print("  No all_metrics for GAIA levels social plot")
        return

    has_level = any(
        "level_metrics" in m.extra and m.extra["level_metrics"]
        for m in all_metrics
    )
    if not has_level:
        print("  No GAIA level data for social levels plot")
        return

    df_sorted = sort_agents_by_provider_and_date(df)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}
    agents_display = [strip_agent_prefix(a) for a in df_sorted["agent"]]
    agents_full = df_sorted["agent"].tolist()
    n_agents = len(agents_display)
    y_pos = np.arange(n_agents)
    levels = ["1", "2", "3"]
    level_colors = {"1": "#4CAF50", "2": "#FF9800", "3": "#F44336"}
    level_labels = {"1": "L1 (Easy)", "2": "L2 (Med)", "3": "L3 (Hard)"}
    bar_w = 0.25

    prev_rc = _social_font_setup()
    _LM = 0.03
    header_h, footer_h = 0.8, 0.6
    body_h = max(4.5, n_agents * 0.4)
    fig = plt.figure(
        figsize=(10, header_h + body_h + footer_h + 0.2), facecolor=_BG_COLOR,
    )
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[header_h, body_h, footer_h],
        hspace=0.20, left=0.08, right=0.97, top=0.97, bottom=0.02,
    )
    _add_social_header(
        fig, gs[0],
        "Performance across GAIA difficulty levels",
        "Accuracy and mean actions per task, broken out by "
        "difficulty level for all models.",
        left_margin=_LM,
    )

    inner = gs[1].subgridspec(1, 2, wspace=0.08)

    # Left: accuracy by level
    ax_l = fig.add_subplot(inner[0, 0])
    for i, level in enumerate(levels):
        vals, ses = [], []
        for agent in agents_full:
            m = agent_to_metrics.get(agent)
            lm = m.extra.get("level_metrics", {}) if m else {}
            val = lm.get("accuracy_by_level", {}).get(level, np.nan)
            vals.append(val)
            se = lm.get("accuracy_by_level_se", {}).get(level, 0.0)
            ses.append(se if se and not np.isnan(se) else 0.0)
        offset = (i - 1) * bar_w
        xerr = np.array(ses)
        ax_l.barh(
            y_pos + offset, vals, bar_w,
            label=level_labels[level], color=level_colors[level],
            alpha=0.8, edgecolor="black", linewidth=0.5,
            xerr=xerr if np.any(xerr > 0) else None, capsize=2,
            error_kw={"linewidth": 0.8, "color": "black"},
        )
    ax_l.set_xlabel("Accuracy", fontsize=11, color=_COLOR_TEXT)
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(agents_display, fontsize=9)
    ax_l.set_xlim(0, 1.05)
    ax_l.invert_yaxis()
    _style_social_scatter(ax_l, emphasized_y_spine=True)
    ax_l.axvline(0, color="black", linewidth=1.5, zorder=1.5)

    # Right: mean actions by level (shared y-axis)
    ax_r = fig.add_subplot(inner[0, 1], sharey=ax_l)
    max_traj = 0
    for m in all_metrics:
        td = m.extra.get("level_metrics", {}).get("trajectory_complexity", {})
        for v in td.values():
            if v and not np.isnan(v) and v > max_traj:
                max_traj = v
    for i, level in enumerate(levels):
        vals, ses = [], []
        for agent in agents_full:
            m = agent_to_metrics.get(agent)
            lm = m.extra.get("level_metrics", {}) if m else {}
            val = lm.get("trajectory_complexity", {}).get(level, np.nan)
            vals.append(val)
            se = lm.get("trajectory_complexity_se", {}).get(level, 0.0)
            ses.append(se if se and not np.isnan(se) else 0.0)
        offset = (i - 1) * bar_w
        xerr = np.array(ses)
        ax_r.barh(
            y_pos + offset, vals, bar_w,
            label=level_labels[level], color=level_colors[level],
            alpha=0.8, edgecolor="black", linewidth=0.5,
            xerr=xerr if np.any(xerr > 0) else None, capsize=2,
            error_kw={"linewidth": 0.8, "color": "black"},
        )
    ax_r.set_xlabel("Mean Actions", fontsize=11, color=_COLOR_TEXT)
    plt.setp(ax_r.get_yticklabels(), visible=False)
    ax_r.set_xlim(0, max(max_traj * 1.1, 10))
    _style_social_scatter(ax_r, emphasized_y_spine=True)
    ax_r.axvline(0, color="black", linewidth=1.5, zorder=1.5)
    ax_r.legend(fontsize=8, loc="upper right", framealpha=0.8)

    _add_social_footer_and_save(
        fig, gs[-1], output_dir, "gaia_levels.pdf",
        left_margin=_LM, padding=padding, prev_rc=prev_rc,
    )
