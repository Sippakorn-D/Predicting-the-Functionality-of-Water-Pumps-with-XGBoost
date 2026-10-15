import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

def generate_dashboard_pdf(df: pd.DataFrame) -> bytes:
    """
    Generate an executive 1-page PDF summary dashboard report
    formatted for Landscape A4 (11.69 x 8.27 in).
    """
    total_n = len(df)
    n_func = int((df["predicted_status"] == "functional").sum())
    n_rep = int((df["predicted_status"] == "functional needs repair").sum())
    n_non = int((df["predicted_status"] == "non functional").sum())
    pct_func = (n_func / total_n * 100) if total_n else 0
    pct_rep = (n_rep / total_n * 100) if total_n else 0
    pct_non = (n_non / total_n * 100) if total_n else 0
    pct_crit = ((n_rep + n_non) / total_n * 100) if total_n else 0

    # Colors
    c_func = "#10b981"   # Emerald
    c_rep = "#f59e0b"    # Amber
    c_non = "#ef4444"    # Crimson
    c_dark = "#0f172a"
    c_muted = "#64748b"
    c_bg_card = "#f8fafc"
    c_border = "#e2e8f0"

    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # Grid: Header/KPIs (row 0), 3 Charts (row 1), Top 10 Table (row 2)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 2.3, 2.2], hspace=0.38, wspace=0.32)

    # -------------------------------------------------------------
    # Row 0: Title Banner & KPI Cards
    # -------------------------------------------------------------
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)

    # Title text
    ax_top.text(
        0.0, 0.88,
        "TANZANIA WATERPOINT FUNCTIONALITY — EXECUTIVE TRIAGE DASHBOARD",
        fontsize=13, fontweight="bold", color=c_dark
    )
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax_top.text(
        0.0, 0.68,
        f"Automated Field Maintenance Report | Evaluated Volume: {total_n} Waterpoints | Generated: {now_str}",
        fontsize=8.5, color=c_muted
    )

    # 5 KPI Cards in banner
    kpi_defs = [
        ("Total Evaluated", f"{total_n:,}", "100%", "#3b82f6"),
        ("Functional", f"{n_func:,}", f"{pct_func:.1f}%", c_func),
        ("Needs Repair", f"{n_rep:,}", f"{pct_rep:.1f}%", c_rep),
        ("Non-Functional", f"{n_non:,}", f"{pct_non:.1f}%", c_non),
        ("Critical Attention", f"{n_rep + n_non:,}", f"{pct_crit:.1f}%", "#dc2626"),
    ]
    card_w = 0.185
    gap = 0.015
    for i, (title, val, sub, col) in enumerate(kpi_defs):
        x = i * (card_w + gap)
        # Background box
        rect = plt.Rectangle(
            (x, 0.0), card_w, 0.54,
            transform=ax_top.transAxes,
            facecolor=c_bg_card, edgecolor=c_border,
            linewidth=1.0
        )
        ax_top.add_patch(rect)
        # Color bar on top of card
        top_bar = plt.Rectangle(
            (x, 0.50), card_w, 0.04,
            transform=ax_top.transAxes,
            facecolor=col, edgecolor="none"
        )
        ax_top.add_patch(top_bar)
        # Text inside card
        ax_top.text(x + 0.01, 0.35, title, transform=ax_top.transAxes, fontsize=7.5, color=c_muted, fontweight="semibold")
        ax_top.text(x + 0.01, 0.12, val, transform=ax_top.transAxes, fontsize=12, color=c_dark, fontweight="bold")
        ax_top.text(x + card_w - 0.01, 0.12, sub, transform=ax_top.transAxes, fontsize=8.5, color=col, fontweight="bold", ha="right")

    # -------------------------------------------------------------
    # Row 1, Col 0: Donut Chart (Status Distribution)
    # -------------------------------------------------------------
    ax_pie = fig.add_subplot(gs[1, 0])
    status_counts = [n_func, n_rep, n_non]
    status_labels = ["Functional", "Needs Repair", "Non-Functional"]
    status_colors = [c_func, c_rep, c_non]

    # Filter zero counts for cleaner pie
    active_slices = [(c, l, col) for c, l, col in zip(status_counts, status_labels, status_colors) if c > 0]
    if active_slices:
        counts_p, labels_p, colors_p = zip(*active_slices)
        wedges, texts, autotexts = ax_pie.pie(
            counts_p,
            labels=labels_p,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors_p,
            textprops={"fontsize": 7.5, "color": c_dark},
            wedgeprops={"width": 0.42, "edgecolor": "#ffffff", "linewidth": 2},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(7.5)
            at.set_fontweight("bold")
            at.set_color("#ffffff")
    ax_pie.set_title("1. Waterpoint Status Breakdown", fontsize=9.5, fontweight="bold", color=c_dark, pad=8)

    # -------------------------------------------------------------
    # Row 1, Col 1: Triage Action Breakdown (Horizontal Bar)
    # -------------------------------------------------------------
    ax_triage = fig.add_subplot(gs[1, 1])
    triage_counts = df["triage_action"].value_counts()
    categories = [
        "URGENT: Full Rehabilitation / Replacement",
        "HIGH PRIORITY: Preventive Repair",
        "Routine Monitoring",
    ]
    t_vals = [triage_counts.get(cat, 0) for cat in categories]
    t_short = ["Urgent Rehab", "Preventive Repair", "Routine Monitor"]
    t_colors = [c_non, c_rep, c_func]

    y_pos = np.arange(len(categories))
    bars = ax_triage.barh(y_pos, t_vals, color=t_colors, height=0.55, edgecolor="none")
    ax_triage.set_yticks(y_pos)
    ax_triage.set_yticklabels(t_short, fontsize=8, color=c_dark)
    ax_triage.invert_yaxis()
    ax_triage.set_xlabel("Number of Waterpoints", fontsize=7.5, color=c_muted)
    ax_triage.set_title("2. Triage Priority Action", fontsize=9.5, fontweight="bold", color=c_dark, pad=8)
    ax_triage.spines["top"].set_visible(False)
    ax_triage.spines["right"].set_visible(False)
    ax_triage.spines["left"].set_color(c_border)
    ax_triage.spines["bottom"].set_color(c_border)
    ax_triage.tick_params(axis="both", labelsize=7.5, colors=c_muted)
    # Add count labels on bars
    for bar in bars:
        w = bar.get_width()
        ax_triage.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=7.5, fontweight="bold", color=c_dark)

    # -------------------------------------------------------------
    # Row 1, Col 2: Top Regions with Critical Pumps
    # -------------------------------------------------------------
    ax_reg = fig.add_subplot(gs[1, 2])
    crit_df = df[df["predicted_status"] != "functional"]
    if not crit_df.empty and "region" in crit_df.columns:
        reg_counts = crit_df["region"].value_counts().head(5)
        bars_r = ax_reg.bar(reg_counts.index, reg_counts.values, color="#6366f1", width=0.55, edgecolor="none")
        ax_reg.set_title("3. Top Critical Regions", fontsize=9.5, fontweight="bold", color=c_dark, pad=8)
        ax_reg.set_ylabel("Critical Pumps", fontsize=7.5, color=c_muted)
        ax_reg.tick_params(axis="x", rotation=25, labelsize=7.5, colors=c_dark)
        ax_reg.tick_params(axis="y", labelsize=7.5, colors=c_muted)
        ax_reg.spines["top"].set_visible(False)
        ax_reg.spines["right"].set_visible(False)
        ax_reg.spines["left"].set_color(c_border)
        ax_reg.spines["bottom"].set_color(c_border)
        for bar in bars_r:
            h = bar.get_height()
            ax_reg.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{int(h)}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=c_dark)
    else:
        ax_reg.text(0.5, 0.5, "No critical waterpoints", ha="center", va="center", color=c_muted, fontsize=8)
        ax_reg.set_title("3. Top Critical Regions", fontsize=9.5, fontweight="bold", color=c_dark)
        ax_reg.axis("off")

    # -------------------------------------------------------------
    # Row 2: Top 10 Critical Waterpoints Table
    # -------------------------------------------------------------
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    ax_table.set_title("4. Top 10 High-Risk Waterpoints Requiring Immediate Field Dispatch", fontsize=9.5, fontweight="bold", color=c_dark, loc="left", pad=10)

    # Sort top 10 by attention score
    top10 = df.sort_values("needs_attention_score", ascending=False).head(10).copy()
    cols_to_show = ["id", "region", "waterpoint_type_group", "quantity_group", "predicted_status", "needs_attention_score", "triage_action"]
    available_cols = [c for c in cols_to_show if c in top10.columns]
    
    # Rename for presentation
    col_names = {
        "id": "Waterpoint ID",
        "region": "Region",
        "waterpoint_type_group": "Type",
        "quantity_group": "Quantity",
        "predicted_status": "Predicted Status",
        "needs_attention_score": "Risk Score",
        "triage_action": "Recommended Field Action"
    }
    
    table_data = []
    headers = [col_names.get(c, c) for c in available_cols]
    for _, row in top10[available_cols].iterrows():
        r_vals = []
        for c in available_cols:
            val = row[c]
            if c == "needs_attention_score":
                r_vals.append(f"{val:.1f}%")
            elif c == "triage_action":
                # Shorten for table
                if "Rehabilitation" in str(val):
                    r_vals.append("URGENT: Rehab")
                elif "Preventive" in str(val):
                    r_vals.append("HIGH: Repair")
                else:
                    r_vals.append("Monitoring")
            else:
                r_vals.append(str(val)[:24])
        table_data.append(r_vals)

    if table_data:
        t_obj = ax_table.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="left"
        )
        t_obj.auto_set_font_size(False)
        t_obj.set_fontsize(7.5)
        t_obj.scale(1.0, 1.25)
        # Format table styling
        for (r, c), cell in t_obj.get_celld().items():
            cell.set_edgecolor(c_border)
            if r == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="#ffffff", fontweight="bold")
            else:
                cell.set_facecolor("#ffffff" if r % 2 != 0 else "#f8fafc")
                # Highlight status column
                status_idx = available_cols.index("predicted_status") if "predicted_status" in available_cols else -1
                if c == status_idx:
                    stat_val = table_data[r-1][status_idx]
                    if stat_val == "non functional":
                        cell.set_text_props(color="#dc2626", fontweight="bold")
                    elif stat_val == "functional needs repair":
                        cell.set_text_props(color="#d97706", fontweight="bold")
                    else:
                        cell.set_text_props(color="#16a34a", fontweight="bold")

    # Footer note
    fig.text(0.08, 0.015, "Confidential | Ministry of Water (Tanzania) & Taarifa Platform | Generated via Machine Learning Operational Pipeline", fontsize=7, color=c_muted)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

if __name__ == "__main__":
    # Test execution
    n = 50
    test_df = pd.DataFrame({
        "id": np.arange(1001, 1001+n),
        "region": np.random.choice(["Iringa", "Kigoma", "Lindi", "Kilimanjaro", "Mtwara", "Tabora"], size=n),
        "waterpoint_type_group": np.random.choice(["communal standpipe", "hand pump", "other"], size=n),
        "quantity_group": np.random.choice(["enough", "insufficient", "dry", "seasonal"], size=n),
        "predicted_status": np.random.choice(["functional", "functional needs repair", "non functional"], size=n, p=[0.6, 0.1, 0.3]),
        "needs_attention_score": np.random.uniform(10, 95, size=n).round(1),
        "triage_action": np.random.choice(["Routine Monitoring", "HIGH PRIORITY: Preventive Repair", "URGENT: Full Rehabilitation / Replacement"], size=n, p=[0.6, 0.1, 0.3])
    })
    pdf_bytes = generate_dashboard_pdf(test_df)
    with open("scratch/test_executive_dashboard.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF Generated successfully! Size:", len(pdf_bytes), "bytes")
