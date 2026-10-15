import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

def generate_single_report_pdf(
    pred_label: str,
    probs: dict,
    needs_attention_prob: float,
    input_df: pd.DataFrame,
    diag_notes: list,
    engineered_row: pd.Series,
    use_group_median: bool,
) -> bytes:
    """
    Generate an official 1-page Portrait A4 (8.27 x 11.69 in)
    Waterpoint Field Inspection & Triage PDF Report.
    """
    row = input_df.iloc[0]
    
    # Palette
    palette = {
        "functional": "#10b981",              # Emerald green
        "functional needs repair": "#f59e0b", # Amber orange
        "non functional": "#ef4444",          # Crimson red
    }
    status_color = palette.get(pred_label, "#2563eb")
    c_dark = "#0f172a"
    c_muted = "#64748b"
    c_border = "#cbd5e1"
    c_bg_card = "#f8fafc"

    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # Layout using gridspec:
    # 0: Header Banner (0.9)
    # 1: ML Prediction Outcome Card (1.6)
    # 2: Field Specifications Table (2.8)
    # 3: Risk Diagnostics & Engineered Features (2.6)
    # 4: Field Crew Dispatch Sign-off (1.5)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.9, 1.7, 3.0, 2.7, 1.4], hspace=0.32)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ref_id = f"TZ-WP-{abs(hash(str(row.to_dict()))) % 1000000:06d}"

    # -------------------------------------------------------------
    # 0. Header Banner
    # -------------------------------------------------------------
    ax_hdr = fig.add_subplot(gs[0])
    ax_hdr.axis("off")
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)

    # Header title & branding
    hdr_bg = plt.Rectangle((0, 0.1), 1, 0.88, transform=ax_hdr.transAxes, facecolor="#1e293b", edgecolor="none")
    ax_hdr.add_patch(hdr_bg)
    ax_hdr.text(0.04, 0.65, "TANZANIA MINISTRY OF WATER & TAARIFA PLATFORM", transform=ax_hdr.transAxes, fontsize=11, fontweight="bold", color="#ffffff")
    ax_hdr.text(0.04, 0.35, "OFFICIAL WATERPOINT INSPECTION & OPERATIONAL TRIAGE REPORT", transform=ax_hdr.transAxes, fontsize=8.5, color="#94a3b8")
    ax_hdr.text(0.96, 0.65, f"REF: {ref_id}", transform=ax_hdr.transAxes, fontsize=8.5, fontweight="bold", color="#38bdf8", ha="right")
    ax_hdr.text(0.96, 0.35, f"Date: {now_str}", transform=ax_hdr.transAxes, fontsize=7.5, color="#94a3b8", ha="right")

    # -------------------------------------------------------------
    # 1. Prediction Outcome & Operational Triage Card
    # -------------------------------------------------------------
    ax_pred = fig.add_subplot(gs[1])
    ax_pred.axis("off")
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)

    # Bordered card
    card_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_pred.transAxes, facecolor=c_bg_card, edgecolor=c_border, linewidth=1.2)
    ax_pred.add_patch(card_rect)
    accent_bar = plt.Rectangle((0, 0.94), 1, 0.06, transform=ax_pred.transAxes, facecolor=status_color, edgecolor="none")
    ax_pred.add_patch(accent_bar)

    # Status & Confidence
    ax_pred.text(0.04, 0.76, "PREDICTED 3-CLASS FUNCTIONALITY STATUS", transform=ax_pred.transAxes, fontsize=8, fontweight="bold", color=c_muted)
    ax_pred.text(0.04, 0.45, pred_label.upper(), transform=ax_pred.transAxes, fontsize=17, fontweight="heavy", color=status_color)
    conf_pct = probs.get(pred_label, 0.0) * 100
    ax_pred.text(0.04, 0.20, f"Model Confidence: {conf_pct:.1f}%", transform=ax_pred.transAxes, fontsize=9, fontweight="bold", color=c_dark)

    # Probability mini-bars on right side of card
    ax_pred.text(0.55, 0.76, "PROBABILITY DISTRIBUTION", transform=ax_pred.transAxes, fontsize=8, fontweight="bold", color=c_muted)
    y_prob = 0.55
    for c_name, c_col in [("functional", palette["functional"]), ("functional needs repair", palette["functional needs repair"]), ("non functional", palette["non functional"])]:
        p_val = probs.get(c_name, 0.0)
        ax_pred.text(0.55, y_prob, f"{c_name.title()[:18]}:", transform=ax_pred.transAxes, fontsize=7.5, color=c_dark)
        ax_pred.text(0.78, y_prob, f"{p_val*100:.1f}%", transform=ax_pred.transAxes, fontsize=7.5, fontweight="bold", color=c_dark)
        # Bar track
        track = plt.Rectangle((0.84, y_prob+0.01), 0.13, 0.05, transform=ax_pred.transAxes, facecolor="#e2e8f0", edgecolor="none")
        ax_pred.add_patch(track)
        fill = plt.Rectangle((0.84, y_prob+0.01), 0.13 * min(1.0, p_val), 0.05, transform=ax_pred.transAxes, facecolor=c_col, edgecolor="none")
        ax_pred.add_patch(fill)
        y_prob -= 0.18

    # Triage action banner
    triage_txt = "ROUTINE MONITORING — No immediate dispatch required" if (pred_label == "functional" and needs_attention_prob < 0.45) else "URGENT DISPATCH — Maintenance crew required for preventive repair or overhaul"
    triage_bg = plt.Rectangle((0.04, 0.04), 0.92, 0.12, transform=ax_pred.transAxes, facecolor="#e2e8f0" if "ROUTINE" in triage_txt else "#fee2e2", edgecolor="none")
    ax_pred.add_patch(triage_bg)
    ax_pred.text(0.06, 0.08, f"2x2 OPERATIONAL TRIAGE: {triage_txt} (Risk: {needs_attention_prob*100:.1f}%)", transform=ax_pred.transAxes, fontsize=7.5, fontweight="bold", color="#15803d" if "ROUTINE" in triage_txt else "#b91c1c")

    # -------------------------------------------------------------
    # 2. Waterpoint Field Parameters Table (18 Features)
    # -------------------------------------------------------------
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")
    ax_tbl.set_title("WATERPOINT FIELD SPECIFICATIONS & ATTRIBUTES (18 PARAMETERS)", fontsize=9, fontweight="bold", color=c_dark, loc="left", pad=6)

    # 2 columns of 9 parameters each
    spec_data = [
        ("Water Quantity", str(row.get("quantity_group", "N/A")), "Extraction Hardware", str(row.get("extraction_type_class", "N/A"))),
        ("Water Quality", str(row.get("quality_group", "N/A")), "Waterpoint Type", str(row.get("waterpoint_type_group", "N/A"))),
        ("Source Class", str(row.get("source_class", "N/A")), "Infrastructure Age", f"{float(row.get('age', 0)):.0f} years"),
        ("Total Static Head (TSH)", f"{float(row.get('amount_tsh', 0)):,.0f}", "Payment Revenue Model", str(row.get("payment_type", "N/A"))),
        ("Administrative Region", str(row.get("region", "N/A")), "Management Entity", str(row.get("management_group", "N/A"))),
        ("Water Basin", str(row.get("basin", "N/A")), "Scheme Management", str(row.get("scheme_management", "N/A"))),
        ("GPS Elevation (Height)", f"{int(row.get('gps_height', 0)):,} m", "Funder Organization", str(row.get("funder", "N/A"))[:20]),
        ("Population Served", f"{int(row.get('population', 0)):,} people", "Installer Entity", str(row.get("installer", "N/A"))[:20]),
        ("Public Meeting Recorded", str(row.get("public_meeting", "N/A")), "Water Permit Issued", str(row.get("permit", "N/A"))),
    ]

    table_cells = []
    for r in spec_data:
        table_cells.append([r[0], r[1], r[2], r[3]])

    t_spec = ax_tbl.table(
        cellText=table_cells,
        colLabels=["Water & Geography", "Value", "Hardware & Governance", "Value"],
        loc="center",
        cellLoc="left",
    )
    t_spec.auto_set_font_size(False)
    t_spec.set_fontsize(7.5)
    t_spec.scale(1.0, 1.35)
    for (r, c), cell in t_spec.get_celld().items():
        cell.set_edgecolor(c_border)
        if r == 0:
            cell.set_facecolor("#334155")
            cell.set_text_props(color="#ffffff", fontweight="bold")
        else:
            if c in [0, 2]:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(fontweight="semibold", color="#334155")
            else:
                cell.set_facecolor("#ffffff")
                cell.set_text_props(color=c_dark)

    # -------------------------------------------------------------
    # 3. Pipeline Diagnostics & FeatureEngineer Output
    # -------------------------------------------------------------
    ax_diag = fig.add_subplot(gs[3])
    ax_diag.axis("off")
    ax_diag.set_xlim(0, 1)
    ax_diag.set_ylim(0, 1)

    ax_diag.text(0, 0.95, "PIPELINE RISK DRIVERS & DIAGNOSTICS", fontsize=9, fontweight="bold", color=c_dark)
    
    y_diag = 0.82
    for note in diag_notes[:3]:
        # Strip markdown asterisks and emojis for clean PDF text
        clean_note = note.replace("**", "").replace("🚨 ", "[CRITICAL] ").replace("⚠️ ", "[WARNING] ").replace("✅ ", "[OK] ")
        ax_diag.text(0.02, y_diag, f"• {clean_note}", transform=ax_diag.transAxes, fontsize=7.5, color=c_dark)
        y_diag -= 0.11

    # Engineered Features Mini Table
    ax_diag.text(0, 0.44, "REAL-TIME FEATUREENGINEER OUTPUT VECTOR", fontsize=8.5, fontweight="bold", color=c_dark)
    fe_headers = ["age (imputed)", "log_pop", "log_tsh", "age_missing", "gps_zero", "pop_zero", "tsh_pos", "same_funder"]
    fe_vals = [
        str(round(float(engineered_row.get("age", 0)), 1)),
        str(round(float(engineered_row.get("log_pop", 0)), 2)),
        str(round(float(engineered_row.get("log_tsh", 0)), 2)),
        str(int(use_group_median)),
        str(int(engineered_row.get("gps_zero", 0))),
        str(int(engineered_row.get("pop_zero", 0))),
        str(int(engineered_row.get("tsh_pos", 0))),
        str(int(engineered_row.get("same_funder_installer", 0))),
    ]
    t_fe = ax_diag.table(
        cellText=[fe_vals],
        colLabels=fe_headers,
        bbox=[0, 0.05, 1, 0.30],
        cellLoc="center"
    )
    t_fe.auto_set_font_size(False)
    t_fe.set_fontsize(7.5)
    for (r, c), cell in t_fe.get_celld().items():
        cell.set_edgecolor(c_border)
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold", color="#1e293b", fontsize=7.0)
        else:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color=c_dark, fontsize=7.5)

    # -------------------------------------------------------------
    # 4. Field Crew Dispatch Sign-off
    # -------------------------------------------------------------
    ax_sign = fig.add_subplot(gs[4])
    ax_sign.axis("off")
    ax_sign.set_xlim(0, 1)
    ax_sign.set_ylim(0, 1)

    sign_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_sign.transAxes, facecolor="#f8fafc", edgecolor=c_border, linewidth=1.0)
    ax_sign.add_patch(sign_rect)
    ax_sign.text(0.03, 0.78, "FIELD INSPECTOR DISPATCH AUTHORIZATION & SIGN-OFF", fontsize=8.5, fontweight="bold", color=c_dark)
    
    # Signature blanks
    ax_sign.text(0.03, 0.40, "Inspector Name: _______________________", fontsize=7.5, color=c_muted)
    ax_sign.text(0.40, 0.40, "Signature: _______________________", fontsize=7.5, color=c_muted)
    ax_sign.text(0.72, 0.40, "Dispatch Date: _____________", fontsize=7.5, color=c_muted)
    ax_sign.text(0.03, 0.14, "Action Status: [  ] Routine Inspection Completed    [  ] Maintenance Ticket Issued    [  ] Equipment Replaced", fontsize=7, color=c_muted)

    # Footer note
    fig.text(0.06, 0.015, "Confidential | Ministry of Water (Tanzania) | Intelligent Lavoisier Machine Learning Decision Support System", fontsize=6.8, color=c_muted)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

if __name__ == "__main__":
    sample_df = pd.DataFrame([{
        "amount_tsh": 500.0,
        "funder": "Danida",
        "gps_height": 1450,
        "installer": "DANIDA",
        "basin": "Lake Nyasa",
        "region": "Iringa",
        "population": 250,
        "public_meeting": "True",
        "scheme_management": "VWC",
        "permit": "True",
        "extraction_type_class": "gravity",
        "management_group": "user-group",
        "payment_type": "annually",
        "quality_group": "good",
        "quantity_group": "enough",
        "source_class": "groundwater",
        "waterpoint_type_group": "communal standpipe",
        "age": 5.0,
    }])
    sample_eng = pd.Series({
        "age": 5.0, "log_pop": 5.525, "log_tsh": 6.217,
        "gps_zero": 0, "pop_zero": 0, "tsh_pos": 1, "same_funder_installer": 1
    })
    pdf_out = generate_single_report_pdf(
        pred_label="functional",
        probs={"functional": 0.826, "functional needs repair": 0.052, "non functional": 0.122},
        needs_attention_prob=0.174,
        input_df=sample_df,
        diag_notes=["✅ Sufficient Water Supply", "✅ Modern Installation", "✅ Active Revenue Model"],
        engineered_row=sample_eng,
        use_group_median=False,
    )
    with open("scratch/test_single_inspection_report.pdf", "wb") as f:
        f.write(pdf_out)
    print("Single PDF report created! Size:", len(pdf_out), "bytes")
