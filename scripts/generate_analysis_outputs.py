from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
NOTEBOOK = ROOT / "AB_Lending_Message_Part_A_B.ipynb"

VARIANTS = ["original", "A", "B"]
LABELS = ["Original", "Rewrite A", "Rewrite B"]
PRIMARY = "doc_submitted_72h"
GUARDRAILS = ["unsub_7d", "complaint_7d", "support_contact_7d"]
PAIRWISE_COMPARISONS = [("original", "A"), ("original", "B"), ("B", "A")]
ANALYSIS_DF = None


def load_analysis_data():
    return pd.read_csv(DATA / "applicants_experiment.csv")


def hypothesis_test(
    control_variant,
    treatment_variant,
    metric,
    alpha=0.05,
    control_df=None,
    treatment_df=None,
):
    if control_df is None:
        control_df = ANALYSIS_DF[ANALYSIS_DF["variant"] == control_variant]
    if treatment_df is None:
        treatment_df = ANALYSIS_DF[ANALYSIS_DF["variant"] == treatment_variant]

    control = control_df[metric]
    treatment = treatment_df[metric]

    n_control = control.count()
    n_treat = treatment.count()
    x_control = control.sum()
    x_treat = treatment.sum()

    control_rate = x_control / n_control
    treatment_rate = x_treat / n_treat
    abs_lift = treatment_rate - control_rate
    rel_lift = abs_lift / control_rate if control_rate != 0 else np.nan

    z_stat, p_value = proportions_ztest(
        count=[x_treat, x_control],
        nobs=[n_treat, n_control],
        alternative="two-sided",
    )
    ci_low, ci_high = confint_proportions_2indep(
        count1=x_treat,
        nobs1=n_treat,
        count2=x_control,
        nobs2=n_control,
        method="wald",
        compare="diff",
        alpha=alpha,
    )

    return pd.DataFrame(
        [
            {
                "comparison": f"{treatment_variant} vs {control_variant}",
                "control_variant": control_variant,
                "treatment_variant": treatment_variant,
                "metric": metric,
                "n_control": n_control,
                "n_treat": n_treat,
                "x_control": x_control,
                "x_treat": x_treat,
                "control_rate": control_rate,
                "treatment_rate": treatment_rate,
                "abs_lift": abs_lift,
                "rel_lift": rel_lift,
                "z_stat": z_stat,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        ]
    )


def primary_summary(df):
    summary = (
        df.groupby("variant")[PRIMARY]
        .agg(successes="sum", n="count", rate="mean")
        .reindex(VARIANTS)
    )
    control_rate = summary.loc["original", "rate"]
    summary["abs_lift_vs_original"] = summary["rate"] - control_rate
    summary["rel_lift_vs_original"] = summary["abs_lift_vs_original"] / control_rate
    summary["ci_low"] = [
        max(0, rate - 1.96 * np.sqrt(rate * (1 - rate) / n))
        for rate, n in zip(summary["rate"], summary["n"])
    ]
    summary["ci_high"] = [
        min(1, rate + 1.96 * np.sqrt(rate * (1 - rate) / n))
        for rate, n in zip(summary["rate"], summary["n"])
    ]
    return summary


def pairwise_tests(df, metric):
    global ANALYSIS_DF
    ANALYSIS_DF = df
    tests = pd.concat(
        [
            hypothesis_test(
                control,
                treatment,
                metric,
                control_df=df[df["variant"] == control],
                treatment_df=df[df["variant"] == treatment],
            )
            for control, treatment in PAIRWISE_COMPARISONS
        ],
        ignore_index=True,
    )
    tests["p_bonferroni"] = (tests["p_value"] * len(PAIRWISE_COMPARISONS)).clip(upper=1)
    tests["significant_bonferroni"] = tests["p_bonferroni"] < 0.05
    return tests[
        [
            "comparison",
            "control_variant",
            "treatment_variant",
            "metric",
            "n_control",
            "n_treat",
            "x_control",
            "x_treat",
            "control_rate",
            "treatment_rate",
            "abs_lift",
            "rel_lift",
            "z_stat",
            "p_value",
            "p_bonferroni",
            "significant_bonferroni",
            "ci_low",
            "ci_high",
        ]
    ]


def power_analysis(primary_tests, alpha=0.05):
    result = primary_tests[primary_tests["comparison"] == "A vs original"].iloc[0]
    alpha_adjusted = alpha / len(PAIRWISE_COMPARISONS)
    effect_size = abs(
        proportion_effectsize(result["treatment_rate"], result["control_rate"])
    )
    ratio = result["n_treat"] / result["n_control"]
    power_model = NormalIndPower()
    required_n_control = power_model.solve_power(
        effect_size=effect_size,
        power=0.80,
        alpha=alpha_adjusted,
        ratio=ratio,
        alternative="two-sided",
    )
    achieved_power_unadjusted = power_model.power(
        effect_size=effect_size,
        nobs1=result["n_control"],
        alpha=alpha,
        ratio=ratio,
        alternative="two-sided",
    )
    achieved_power_adjusted = power_model.power(
        effect_size=effect_size,
        nobs1=result["n_control"],
        alpha=alpha_adjusted,
        ratio=ratio,
        alternative="two-sided",
    )
    return pd.DataFrame(
        [
            {
                "comparison": "A vs original",
                "control_rate": result["control_rate"],
                "treatment_rate": result["treatment_rate"],
                "abs_lift": result["abs_lift"],
                "effect_size_cohens_h": effect_size,
                "alpha_adjusted": alpha_adjusted,
                "n_control_observed": result["n_control"],
                "n_treat_observed": result["n_treat"],
                "achieved_power_alpha_0_05": achieved_power_unadjusted,
                "achieved_power_bonferroni_alpha": achieved_power_adjusted,
                "required_n_control_for_80pct_power": required_n_control,
                "required_n_treat_for_80pct_power": required_n_control * ratio,
            }
        ]
    )


def guardrail_summary(df):
    rates = df.groupby("variant")[GUARDRAILS].mean().reindex(VARIANTS)
    counts = df.groupby("variant")[GUARDRAILS].sum().reindex(VARIANTS)
    tests = pd.concat([pairwise_tests(df, metric) for metric in GUARDRAILS], ignore_index=True)
    return rates, counts, tests


def sample_ratio_check(df):
    observed = df["variant"].value_counts().reindex(VARIANTS)
    expected = np.repeat(len(df) / len(VARIANTS), len(VARIANTS))
    chi2_stat, p_value = chisquare(observed.values, expected)
    return pd.DataFrame(
        {
            "variant": VARIANTS,
            "observed_n": observed.values,
            "expected_n": expected,
            "observed_share": observed.values / len(df),
            "expected_share": expected / len(df),
            "difference_n": observed.values - expected,
            "chi2_stat": chi2_stat,
            "p_value": p_value,
        }
    )


def randomization_balance(df):
    rows = []
    for variable in ["channel", "risk_band", "missing_doc_type"]:
        table = pd.crosstab(df["variant"], df[variable]).reindex(VARIANTS)
        chi2_stat, p_value, dof, _ = chi2_contingency(table)
        shares = table.div(table.sum(axis=1), axis=0)
        max_share_diff = (shares.max(axis=0) - shares.min(axis=0)).max()
        rows.append(
            {
                "variable": variable,
                "chi2_stat": chi2_stat,
                "dof": dof,
                "p_value": p_value,
                "max_cell_share_minus_min_cell_share": max_share_diff,
            }
        )
    return pd.DataFrame(rows)


def delivered_sensitivity(df):
    delivered = df[df["delivered"] == 1].copy()
    summary = primary_summary(delivered)
    tests = pairwise_tests(delivered, PRIMARY)
    return delivered, summary, tests


def segment_summaries(df):
    outputs = {}
    for col in ["channel", "risk_band", "missing_doc_type"]:
        segment = (
            df.groupby([col, "variant"])[PRIMARY]
            .agg(successes="sum", n="count", rate="mean")
            .reset_index()
        )
        control = (
            segment[segment["variant"] == "original"][[col, "rate"]]
            .rename(columns={"rate": "original_rate"})
        )
        segment = segment.merge(control, on=col)
        segment["abs_lift_vs_original"] = segment["rate"] - segment["original_rate"]
        outputs[col] = segment
    return outputs


def priority_segment_tests(df):
    tests = pd.concat(
        [
            hypothesis_test(
                "original",
                "A",
                PRIMARY,
                control_df=df[
                    (df["risk_band"] == "high") & (df["variant"] == "original")
                ],
                treatment_df=df[(df["risk_band"] == "high") & (df["variant"] == "A")],
            ).assign(segment="High risk"),
            hypothesis_test(
                "original",
                "A",
                PRIMARY,
                control_df=df[
                    (df["missing_doc_type"] == "both")
                    & (df["variant"] == "original")
                ],
                treatment_df=df[
                    (df["missing_doc_type"] == "both") & (df["variant"] == "A")
                ],
            ).assign(segment="Missing both docs"),
            hypothesis_test(
                "original",
                "A",
                PRIMARY,
                control_df=df[(df["channel"] == "sms") & (df["variant"] == "original")],
                treatment_df=df[(df["channel"] == "sms") & (df["variant"] == "A")],
            ).assign(segment="SMS channel"),
        ],
        ignore_index=True,
    )
    return tests[
        [
            "segment",
            "comparison",
            "n_control",
            "n_treat",
            "control_rate",
            "treatment_rate",
            "abs_lift",
            "rel_lift",
            "p_value",
            "ci_low",
            "ci_high",
        ]
    ]


def save_hypothesis_figure(test_results, output_path, title, significance_level=0.05):
    plot_data = test_results.iloc[::-1].copy()
    if "segment" in plot_data.columns:
        labels = plot_data["segment"]
    elif "metric" in plot_data.columns and plot_data["metric"].nunique() > 1:
        labels = plot_data["metric"].astype(str) + ": " + plot_data["comparison"].astype(str)
    else:
        labels = plot_data["comparison"]

    y = np.arange(len(plot_data))
    effects = plot_data["abs_lift"].values * 100
    ci_low = plot_data["ci_low"].values * 100
    ci_high = plot_data["ci_high"].values * 100
    xerr = np.vstack([effects - ci_low, ci_high - effects])
    colors_sig = np.where(plot_data["p_value"] < significance_level, "#2563eb", "#6b7280")

    fig_height = max(3.2, 0.55 * len(plot_data) + 1.8)
    fig, (ax_ci, ax_p) = plt.subplots(
        1,
        2,
        figsize=(12, fig_height),
        gridspec_kw={"width_ratios": [1.45, 1]},
        sharey=True,
    )
    ax_ci.errorbar(
        effects,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#374151",
        elinewidth=1.5,
        capsize=4,
    )
    ax_ci.scatter(effects, y, color=colors_sig, s=70, zorder=2)
    ax_ci.axvline(0, color="#111827", linestyle="--", linewidth=1)
    ax_ci.set_yticks(y, labels)
    ax_ci.set_xlabel("Absolute lift, percentage points")
    ax_ci.set_title("Lift with confidence interval")
    ax_ci.grid(axis="x", alpha=0.25)

    p_values = plot_data["p_value"].clip(lower=1e-12).values
    min_x = max(min(p_values.min(), significance_level) / 5, 1e-6)
    max_x = min(max(p_values.max(), significance_level) * 2, 1)
    ax_p.hlines(y, min_x, p_values, color=colors_sig, linewidth=2, alpha=0.65)
    ax_p.scatter(p_values, y, color=colors_sig, s=70, zorder=2)
    ax_p.axvline(significance_level, color="#dc2626", linestyle="--", linewidth=1.2)
    ax_p.set_xscale("log")
    ax_p.set_xlim(min_x, max_x)
    ax_p.set_xlabel("p-value, log scale")
    ax_p.set_title(f"p-value vs alpha={significance_level:.4g}")
    ax_p.grid(axis="x", alpha=0.25)
    ax_p.tick_params(axis="y", labelleft=False)
    for row_y, p_value in zip(y, p_values):
        ax_p.annotate(
            f"{p_value:.3g}",
            xy=(p_value, row_y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_figures(summary, primary_tests, guardrail_rates, segments, segment_tests):
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = ["#6b7280", "#2563eb", "#f97316"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(VARIANTS))
    rates = summary["rate"].values * 100
    yerr = np.vstack(
        (
            (summary["rate"] - summary["ci_low"]).values * 100,
            (summary["ci_high"] - summary["rate"]).values * 100,
        )
    )
    bars = ax.bar(x, rates, color=colors, width=0.62)
    ax.errorbar(x, rates, yerr=yerr, fmt="none", ecolor="#111827", capsize=5)
    ax.axhline(summary.loc["original", "rate"] * 100, color="#6b7280", linestyle="--")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Document submitted within 72h (%)")
    ax.set_title("Primary outcome by variant")
    ax.set_ylim(0, max((summary["ci_high"] * 100).max() + 3, 24))
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "primary_outcome_by_variant.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.24
    x = np.arange(len(GUARDRAILS))
    metric_labels = ["Unsubscribe", "Complaint", "Support contact"]
    for i, variant in enumerate(VARIANTS):
        values = guardrail_rates.loc[variant].values * 100
        ax.bar(x + (i - 1) * width, values, width=width, label=LABELS[i], color=colors[i])
        for j, value in enumerate(values):
            ax.text(
                j + (i - 1) * width,
                value + 0.08,
                f"{value:.2f}%",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("7-day rate (%)")
    ax.set_title("Guardrail rates by variant")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(guardrail_rates.max().max() * 100 + 1.2, 4))
    fig.tight_layout()
    fig.savefig(FIGURES / "guardrails_by_variant.png", dpi=180)
    plt.close(fig)

    segment_frames = []
    for col, segment in segments.items():
        temp = segment[segment["variant"].isin(["A", "B"])].copy()
        temp["segment"] = col + ": " + temp[col].astype(str)
        segment_frames.append(temp[["segment", "variant", "abs_lift_vs_original"]])
    plot_data = pd.concat(segment_frames, ignore_index=True)
    pivot = (
        plot_data.pivot(index="segment", columns="variant", values="abs_lift_vs_original")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5.8))
    y = np.arange(len(pivot.index))
    ax.axvline(0, color="#111827", linewidth=1)
    ax.barh(y - 0.18, pivot["A"] * 100, height=0.32, label="Rewrite A", color=colors[1])
    ax.barh(y + 0.18, pivot["B"] * 100, height=0.32, label="Rewrite B", color=colors[2])
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel("Absolute lift vs Original (percentage points)")
    ax.set_title("Primary outcome lift by segment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "segment_lift_vs_original.png", dpi=180)
    plt.close(fig)

    save_hypothesis_figure(
        primary_tests,
        FIGURES / "primary_hypothesis_tests.png",
        "Primary outcome pairwise tests",
        significance_level=0.05 / len(PAIRWISE_COMPARISONS),
    )
    save_hypothesis_figure(
        segment_tests,
        FIGURES / "segment_hypothesis_tests.png",
        "Priority segment hypothesis tests",
        significance_level=0.05,
    )


def drawing_conclusions_markdown():
    return (
        "## Drawing conclusions\n\n"
        "Variant A is the strongest candidate because it improves the primary outcome "
        "meaningfully and remains acceptable on the reviewed guardrails."
    )


def pct(value):
    return f"{value:.2%}"


def pp(value):
    return f"{value * 100:+.2f} pp"


def write_report(
    summary,
    primary_tests,
    power,
    guardrail_rates,
    delivered_summary,
    delivered_tests,
):
    primary_rows = "\n".join(
        [
            "<tr>"
            f"<td>{LABELS[i]}</td>"
            f"<td>{int(row['n']):,}</td>"
            f"<td>{int(row['successes']):,}</td>"
            f"<td>{pct(row['rate'])}</td>"
            f"<td>{pp(row['abs_lift_vs_original'])}</td>"
            f"<td>{pct(row['rel_lift_vs_original'])}</td>"
            "</tr>"
            for i, (_, row) in enumerate(summary.iterrows())
        ]
    )
    test_rows = "\n".join(
        [
            "<tr>"
            f"<td>{row['comparison']}</td>"
            f"<td>{pp(row['abs_lift'])}</td>"
            f"<td>{pp(row['ci_low'])} to {pp(row['ci_high'])}</td>"
            f"<td>{row['p_value']:.4f}</td>"
            f"<td>{row['p_bonferroni']:.4f}</td>"
            "</tr>"
            for _, row in primary_tests.iterrows()
        ]
    )
    guardrail_rows = "\n".join(
        [
            "<tr>"
            f"<td>{LABELS[i]}</td>"
            f"<td>{pct(row['unsub_7d'])}</td>"
            f"<td>{pct(row['complaint_7d'])}</td>"
            f"<td>{pct(row['support_contact_7d'])}</td>"
            "</tr>"
            for i, (_, row) in enumerate(guardrail_rates.iterrows())
        ]
    )
    delivered_rows = "\n".join(
        [
            "<tr>"
            f"<td>{LABELS[i]}</td>"
            f"<td>{int(row['n']):,}</td>"
            f"<td>{int(row['successes']):,}</td>"
            f"<td>{pct(row['rate'])}</td>"
            f"<td>{pp(row['abs_lift_vs_original'])}</td>"
            f"<td>{pct(row['rel_lift_vs_original'])}</td>"
            "</tr>"
            for i, (_, row) in enumerate(delivered_summary.iterrows())
        ]
    )
    delivered_test_rows = "\n".join(
        [
            "<tr>"
            f"<td>{row['comparison']}</td>"
            f"<td>{pp(row['abs_lift'])}</td>"
            f"<td>{pp(row['ci_low'])} to {pp(row['ci_high'])}</td>"
            f"<td>{row['p_value']:.4f}</td>"
            f"<td>{row['p_bonferroni']:.4f}</td>"
            "</tr>"
            for _, row in delivered_tests.iterrows()
        ]
    )
    power_row = power.iloc[0]
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lending Message A/B Test Output Report</title>
  <style>
    body {{ margin: 0; color: #111827; font-family: Arial, Helvetica, sans-serif; line-height: 1.45; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; }}
    h2 {{ border-top: 1px solid #d1d5db; font-size: 22px; margin-top: 34px; padding-top: 24px; }}
    p {{ margin: 0 0 14px; }}
    .summary {{ background: #f9fafb; border: 1px solid #d1d5db; padding: 16px; margin: 18px 0 24px; }}
    .summary strong {{ color: #2563eb; }}
    table {{ border-collapse: collapse; margin: 14px 0 24px; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f9fafb; }}
    figure {{ margin: 18px 0 28px; }}
    figure img {{ border: 1px solid #d1d5db; display: block; height: auto; max-width: 100%; }}
    figcaption {{ color: #4b5563; font-size: 14px; margin-top: 8px; }}
    code {{ background: #f9fafb; padding: 1px 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Lending Message A/B Test Output Report</h1>
  <p>Part A experiment analysis and Part B recommendation for the lending outreach message test.</p>
  <section class="summary">
    <p><strong>Recommendation:</strong> roll out Rewrite A with guardrail monitoring.</p>
    <p>Rewrite A improves the primary outcome from 17.18% to 19.36%, a +2.17 percentage-point absolute lift and +12.65% relative lift versus Original. Rewrite B is weaker on the primary outcome and worse on guardrails.</p>
  </section>

  <h2>Primary Outcome</h2>
  <p>The primary outcome is <code>doc_submitted_72h</code>, because the business goal is to get applicants to submit missing income documents, not only to click or respond.</p>
  <figure><img src="figures/primary_outcome_by_variant.png" alt="Primary outcome by variant"><figcaption>Document submission rate within 72 hours by message variant.</figcaption></figure>
  <table><thead><tr><th>Variant</th><th>Randomized n</th><th>Submitted docs</th><th>Rate</th><th>Abs. lift vs Original</th><th>Rel. lift vs Original</th></tr></thead><tbody>{primary_rows}</tbody></table>

  <h2>Hypothesis Tests</h2>
  <p>Primary pairwise tests use a Bonferroni adjustment across the three planned primary-outcome comparisons. Guardrail and segment tests are treated as diagnostic and directional.</p>
  <figure><img src="figures/primary_hypothesis_tests.png" alt="Primary hypothesis tests"><figcaption>Primary outcome pairwise tests: confidence intervals and p-values.</figcaption></figure>
  <table><thead><tr><th>Comparison</th><th>Abs. lift</th><th>95% CI</th><th>p-value</th><th>Bonferroni p-value</th></tr></thead><tbody>{test_rows}</tbody></table>

  <h2>Power Analysis</h2>
  <p>For the observed A vs Original effect, Cohen's h is {power_row['effect_size_cohens_h']:.4f}. Achieved power is {power_row['achieved_power_alpha_0_05']:.1%} at alpha 0.05 and {power_row['achieved_power_bonferroni_alpha']:.1%} at the Bonferroni-adjusted alpha of {power_row['alpha_adjusted']:.4f}.</p>

  <h2>Delivered-Only Sensitivity</h2>
  <p>The primary analysis uses all randomized applicants. This sensitivity check repeats the primary outcome analysis among delivered messages only.</p>
  <table><thead><tr><th>Variant</th><th>Delivered n</th><th>Submitted docs</th><th>Rate</th><th>Abs. lift vs Original</th><th>Rel. lift vs Original</th></tr></thead><tbody>{delivered_rows}</tbody></table>
  <table><thead><tr><th>Comparison</th><th>Abs. lift</th><th>95% CI</th><th>p-value</th><th>Bonferroni p-value</th></tr></thead><tbody>{delivered_test_rows}</tbody></table>

  <h2>Guardrails</h2>
  <p>Rewrite A does not show a material guardrail issue. Rewrite B has directionally worse unsubscribe and support-contact rates.</p>
  <figure><img src="figures/guardrails_by_variant.png" alt="Guardrail rates by variant"><figcaption>Unsubscribe, complaint, and support-contact rates by variant.</figcaption></figure>
  <table><thead><tr><th>Variant</th><th>Unsubscribe rate</th><th>Complaint rate</th><th>Support contact rate</th></tr></thead><tbody>{guardrail_rows}</tbody></table>

  <h2>Segment Review</h2>
  <p>Rewrite A is directionally positive across the reviewed segments. The weaker A-lift watch areas are high-risk borrowers, applicants missing both document types, and SMS users.</p>
  <figure><img src="figures/segment_lift_vs_original.png" alt="Segment lift vs original"><figcaption>Primary outcome lift by segment for Rewrite A and Rewrite B versus Original.</figcaption></figure>
  <figure><img src="figures/segment_hypothesis_tests.png" alt="Priority segment hypothesis tests"><figcaption>Priority segment A vs Original hypothesis tests. Confidence intervals are wide and cross zero, so these are directional checks.</figcaption></figure>

  <h2>Decision</h2>
  <p>Recommend Rewrite A as the winner. Roll it out with monitoring for unsubscribes, complaints, and support contacts. Treat segment-level results as directional because each subgroup has less sample than the full experiment.</p>
</main>
</body>
</html>
"""
    (ROOT / "output_report.html").write_text(report, encoding="utf-8")


def write_notebook():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Lending Message A/B Test: Part A and Part B\n\n"
            "This notebook analyzes the lending message experiment from "
            "`data/applicants_experiment.csv` and keeps only Part A and Part B "
            "for a GitHub-ready submission."
        ),
        nbf.v4.new_markdown_cell(
            "## Part A - Experiment Analysis\n\n"
            "Primary outcome: `doc_submitted_72h`, because the business goal is to get applicants "
            "to submit missing income documents. Clicks and responses are secondary diagnostics."
        ),
        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy.stats import chi2_contingency, chisquare\n"
            "from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep\n\n"
            "app = pd.read_csv('data/applicants_experiment.csv')\n"
            "variants = ['original', 'A', 'B']\n"
            "primary = 'doc_submitted_72h'\n\n"
            "print(app.info())\n"
            "display(app.head())"
        ),
        nbf.v4.new_code_cell(
            "duplicate_applicants = app['applicant_id'].duplicated().sum()\n"
            "print(f'Duplicate applicant IDs: {duplicate_applicants}')\n"
            "display(app.groupby('variant').size().reindex(variants).rename('n'))"
        ),
        nbf.v4.new_markdown_cell(
            "### Sample Ratio Mismatch Check\n\n"
            "This checks whether the observed assignment counts are consistent with an "
            "equal split across variants."
        ),
        nbf.v4.new_code_cell(
            "observed = app['variant'].value_counts().reindex(variants)\n"
            "expected = np.repeat(len(app) / len(variants), len(variants))\n"
            "srm_chi2, srm_p_value = chisquare(observed.values, expected)\n"
            "srm_check = pd.DataFrame({\n"
            "    'variant': variants,\n"
            "    'observed_n': observed.values,\n"
            "    'expected_n': expected,\n"
            "    'observed_share': observed.values / len(app),\n"
            "    'expected_share': expected / len(app),\n"
            "    'difference_n': observed.values - expected,\n"
            "    'chi2_stat': srm_chi2,\n"
            "    'p_value': srm_p_value,\n"
            "})\n"
            "display(srm_check.style.format({\n"
            "    'expected_n': '{:.0f}',\n"
            "    'observed_share': '{:.2%}',\n"
            "    'expected_share': '{:.2%}',\n"
            "    'difference_n': '{:+.0f}',\n"
            "    'chi2_stat': '{:.4f}',\n"
            "    'p_value': '{:.4f}',\n"
            "}))"
        ),
        nbf.v4.new_markdown_cell("### Group Balance Checks"),
        nbf.v4.new_code_cell(
            "display(pd.crosstab(app['variant'], app['channel'], normalize='index').reindex(variants))\n"
            "display(pd.crosstab(app['variant'], app['risk_band'], normalize='index').reindex(variants))\n"
            "display(pd.crosstab(app['variant'], app['missing_doc_type'], normalize='index').reindex(variants))"
        ),
        nbf.v4.new_code_cell(
            "balance_rows = []\n"
            "for variable in ['channel', 'risk_band', 'missing_doc_type']:\n"
            "    table = pd.crosstab(app['variant'], app[variable]).reindex(variants)\n"
            "    chi2_stat, p_value, dof, _ = chi2_contingency(table)\n"
            "    shares = table.div(table.sum(axis=1), axis=0)\n"
            "    balance_rows.append({\n"
            "        'variable': variable,\n"
            "        'chi2_stat': chi2_stat,\n"
            "        'dof': dof,\n"
            "        'p_value': p_value,\n"
            "        'max_cell_share_spread': (shares.max(axis=0) - shares.min(axis=0)).max(),\n"
            "    })\n"
            "balance_summary = pd.DataFrame(balance_rows)\n"
            "display(balance_summary.style.format({\n"
            "    'chi2_stat': '{:.4f}',\n"
            "    'p_value': '{:.4f}',\n"
            "    'max_cell_share_spread': '{:.2%}',\n"
            "}))"
        ),
        nbf.v4.new_markdown_cell(
            "### Power Analysis\n\n"
            "This checks whether the observed A vs Original effect was large enough to detect "
            "with the available sample. It reports power at alpha 0.05 and at the Bonferroni-adjusted "
            "alpha of 0.0167 used for the three primary pairwise comparisons."
        ),
        nbf.v4.new_code_cell(
            "from statsmodels.stats.power import NormalIndPower\n"
            "from statsmodels.stats.proportion import proportion_effectsize\n\n"
            "control = app[app['variant'] == 'original'][primary]\n"
            "treatment = app[app['variant'] == 'A'][primary]\n\n"
            "n_control = control.count()\n"
            "n_treat = treatment.count()\n"
            "control_rate = control.mean()\n"
            "treatment_rate = treatment.mean()\n"
            "abs_lift = treatment_rate - control_rate\n"
            "alpha_adjusted = 0.05 / 3\n"
            "effect_size = abs(proportion_effectsize(treatment_rate, control_rate))\n"
            "ratio = n_treat / n_control\n"
            "power_model = NormalIndPower()\n\n"
            "required_n_control = power_model.solve_power(\n"
            "    effect_size=effect_size,\n"
            "    power=0.80,\n"
            "    alpha=alpha_adjusted,\n"
            "    ratio=ratio,\n"
            "    alternative='two-sided'\n"
            ")\n"
            "achieved_power_alpha_0_05 = power_model.power(\n"
            "    effect_size=effect_size,\n"
            "    nobs1=n_control,\n"
            "    alpha=0.05,\n"
            "    ratio=ratio,\n"
            "    alternative='two-sided'\n"
            ")\n"
            "achieved_power_bonferroni_alpha = power_model.power(\n"
            "    effect_size=effect_size,\n"
            "    nobs1=n_control,\n"
            "    alpha=alpha_adjusted,\n"
            "    ratio=ratio,\n"
            "    alternative='two-sided'\n"
            ")\n\n"
            "power_summary = pd.DataFrame([{\n"
            "    'comparison': 'A vs original',\n"
            "    'control_rate': control_rate,\n"
            "    'treatment_rate': treatment_rate,\n"
            "    'abs_lift': abs_lift,\n"
            "    'effect_size_cohens_h': effect_size,\n"
            "    'alpha_adjusted': alpha_adjusted,\n"
            "    'n_control_observed': n_control,\n"
            "    'n_treat_observed': n_treat,\n"
            "    'achieved_power_alpha_0_05': achieved_power_alpha_0_05,\n"
            "    'achieved_power_bonferroni_alpha': achieved_power_bonferroni_alpha,\n"
            "    'required_n_control_for_80pct_power': required_n_control,\n"
            "    'required_n_treat_for_80pct_power': required_n_control * ratio,\n"
            "}])\n"
            "display(power_summary.style.format('{:.4f}', subset=[\n"
            "    'control_rate', 'treatment_rate', 'abs_lift', 'effect_size_cohens_h',\n"
            "    'alpha_adjusted', 'achieved_power_alpha_0_05',\n"
            "    'achieved_power_bonferroni_alpha', 'required_n_control_for_80pct_power',\n"
            "    'required_n_treat_for_80pct_power'\n"
            "]))"
        ),
        nbf.v4.new_markdown_cell(
            "### Reusable Statistical Test\n\n"
            "All pairwise statistical results below are calculated with `hypothesis_test`. "
            "It accepts explicit control and treatment dataframes, runs a two-sided "
            "two-proportion z-test, and returns the result as a one-row dataframe "
            "with conversion rates, lift, p-value, and a 95% confidence interval "
            "for the treatment-control difference."
        ),
        nbf.v4.new_code_cell(
            "def hypothesis_test(\n"
            "    control_variant,\n"
            "    treatment_variant,\n"
            "    metric,\n"
            "    alpha=0.05,\n"
            "    control_df=None,\n"
            "    treatment_df=None,\n"
            "):\n"
            "    if control_df is None:\n"
            "        control_df = app[app['variant'] == control_variant]\n"
            "    if treatment_df is None:\n"
            "        treatment_df = app[app['variant'] == treatment_variant]\n\n"
            "    control = control_df[metric]\n"
            "    treatment = treatment_df[metric]\n\n"
            "    n_control = control.count()\n"
            "    n_treat = treatment.count()\n"
            "    x_control = control.sum()\n"
            "    x_treat = treatment.sum()\n\n"
            "    control_rate = x_control / n_control\n"
            "    treatment_rate = x_treat / n_treat\n"
            "    abs_lift = treatment_rate - control_rate\n"
            "    rel_lift = abs_lift / control_rate if control_rate != 0 else np.nan\n\n"
            "    z_stat, p_value = proportions_ztest(\n"
            "        count=[x_treat, x_control],\n"
            "        nobs=[n_treat, n_control],\n"
            "        alternative='two-sided'\n"
            "    )\n"
            "    ci_low, ci_high = confint_proportions_2indep(\n"
            "        count1=x_treat,\n"
            "        nobs1=n_treat,\n"
            "        count2=x_control,\n"
            "        nobs2=n_control,\n"
            "        method='wald',\n"
            "        compare='diff',\n"
            "        alpha=alpha\n"
            "    )\n\n"
            "    return pd.DataFrame([\n"
            "        {\n"
            "            'comparison': f'{treatment_variant} vs {control_variant}',\n"
            "            'control_variant': control_variant,\n"
            "            'treatment_variant': treatment_variant,\n"
            "            'metric': metric,\n"
            "            'n_control': n_control,\n"
            "            'n_treat': n_treat,\n"
            "            'x_control': x_control,\n"
            "            'x_treat': x_treat,\n"
            "            'control_rate': control_rate,\n"
            "            'treatment_rate': treatment_rate,\n"
            "            'abs_lift': abs_lift,\n"
            "            'rel_lift': rel_lift,\n"
            "            'z_stat': z_stat,\n"
            "            'p_value': p_value,\n"
            "            'ci_low': ci_low,\n"
            "            'ci_high': ci_high,\n"
            "        }\n"
            "    ])"
        ),
        nbf.v4.new_markdown_cell(
            "### Reusable Hypothesis Test Visualization\n\n"
            "Use `plot_hypothesis_results` to visualize the output dataframe from "
            "`hypothesis_test`. The `significance_level` input controls the p-value "
            "reference line and the significant/non-significant color coding."
        ),
        nbf.v4.new_code_cell(
            "def plot_hypothesis_results(\n"
            "    test_results,\n"
            "    significance_level=0.05,\n"
            "    title='Hypothesis test results',\n"
            "):\n"
            "    results = test_results.copy()\n"
            "    if isinstance(results, pd.Series):\n"
            "        results = results.to_frame().T\n"
            "    elif isinstance(results, dict):\n"
            "        results = pd.DataFrame([results])\n\n"
            "    required_cols = {'comparison', 'abs_lift', 'ci_low', 'ci_high', 'p_value'}\n"
            "    missing_cols = required_cols - set(results.columns)\n"
            "    if missing_cols:\n"
            "        raise ValueError(f'Missing required columns: {sorted(missing_cols)}')\n\n"
            "    results = results.reset_index(drop=True)\n"
            "    if 'segment' in results.columns:\n"
            "        labels = results['segment'].astype(str)\n"
            "    elif 'metric' in results.columns and results['metric'].nunique() > 1:\n"
            "        labels = results['metric'].astype(str) + ': ' + results['comparison'].astype(str)\n"
            "    else:\n"
            "        labels = results['comparison'].astype(str)\n\n"
            "    results = results.assign(\n"
            "        label=labels,\n"
            "        significant=results['p_value'] < significance_level,\n"
            "        abs_lift_pp=results['abs_lift'] * 100,\n"
            "        ci_low_pp=results['ci_low'] * 100,\n"
            "        ci_high_pp=results['ci_high'] * 100,\n"
            "    ).iloc[::-1]\n\n"
            "    y = np.arange(len(results))\n"
            "    colors = np.where(results['significant'], '#2563eb', '#6b7280')\n"
            "    fig_height = max(3.2, 0.55 * len(results) + 1.8)\n"
            "    fig, (ax_ci, ax_p) = plt.subplots(\n"
            "        1,\n"
            "        2,\n"
            "        figsize=(12, fig_height),\n"
            "        gridspec_kw={'width_ratios': [1.45, 1]},\n"
            "        sharey=True,\n"
            "    )\n\n"
            "    effects = results['abs_lift_pp'].to_numpy(dtype=float)\n"
            "    ci_low = results['ci_low_pp'].to_numpy(dtype=float)\n"
            "    ci_high = results['ci_high_pp'].to_numpy(dtype=float)\n"
            "    xerr = np.vstack([effects - ci_low, ci_high - effects])\n\n"
            "    ax_ci.errorbar(\n"
            "        effects,\n"
            "        y,\n"
            "        xerr=xerr,\n"
            "        fmt='none',\n"
            "        ecolor='#374151',\n"
            "        elinewidth=1.5,\n"
            "        capsize=4,\n"
            "        zorder=1,\n"
            "    )\n"
            "    ax_ci.scatter(effects, y, color=colors, s=70, zorder=2)\n"
            "    ax_ci.axvline(0, color='#111827', linestyle='--', linewidth=1)\n"
            "    ax_ci.set_yticks(y)\n"
            "    ax_ci.set_yticklabels(results['label'])\n"
            "    ax_ci.set_xlabel('Absolute lift, percentage points')\n"
            "    ax_ci.set_title('Lift with confidence interval')\n"
            "    ax_ci.grid(axis='x', alpha=0.25)\n\n"
            "    p_values = results['p_value'].clip(lower=1e-12).to_numpy(dtype=float)\n"
            "    min_x = max(min(p_values.min(), significance_level) / 5, 1e-6)\n"
            "    max_x = min(max(p_values.max(), significance_level) * 2, 1)\n"
            "    ax_p.hlines(y, min_x, p_values, color=colors, linewidth=2, alpha=0.65)\n"
            "    ax_p.scatter(p_values, y, color=colors, s=70, zorder=2)\n"
            "    ax_p.axvline(significance_level, color='#dc2626', linestyle='--', linewidth=1.2)\n"
            "    ax_p.set_xscale('log')\n"
            "    ax_p.set_xlim(min_x, max_x)\n"
            "    ax_p.set_xlabel('p-value, log scale')\n"
            "    ax_p.set_title(f'p-value vs alpha={significance_level:.4g}')\n"
            "    ax_p.grid(axis='x', alpha=0.25)\n"
            "    ax_p.tick_params(axis='y', labelleft=False)\n\n"
            "    for row_y, p_value in zip(y, p_values):\n"
            "        ax_p.annotate(\n"
            "            f'{p_value:.3g}',\n"
            "            xy=(p_value, row_y),\n"
            "            xytext=(6, 0),\n"
            "            textcoords='offset points',\n"
            "            va='center',\n"
            "            fontsize=9,\n"
            "        )\n\n"
            "    fig.suptitle(title, y=1.02)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    return fig, (ax_ci, ax_p)"
        ),
        nbf.v4.new_markdown_cell("### Primary Outcome"),
        nbf.v4.new_code_cell(
            "primary_summary = app.groupby('variant')[primary].agg(successes='sum', n='count', rate='mean').reindex(variants)\n"
            "control_rate = primary_summary.loc['original', 'rate']\n"
            "primary_summary['abs_lift_vs_original'] = primary_summary['rate'] - control_rate\n"
            "primary_summary['rel_lift_vs_original'] = primary_summary['abs_lift_vs_original'] / control_rate\n"
            "display(primary_summary.style.format('{:.4f}', subset=['rate', 'abs_lift_vs_original', 'rel_lift_vs_original']))"
        ),
        nbf.v4.new_markdown_cell(
            "### Multiple-Testing Policy\n\n"
            "The three planned primary-outcome pairwise tests use Bonferroni correction. "
            "Guardrail and segment tests are diagnostic checks and should be interpreted "
            "directionally rather than as winner-selection tests."
        ),
        nbf.v4.new_code_cell(
            "primary_tests = pd.concat([\n"
            "    hypothesis_test('original', 'A', primary, control_df=app[app['variant'] == 'original'], treatment_df=app[app['variant'] == 'A']),\n"
            "    hypothesis_test('original', 'B', primary, control_df=app[app['variant'] == 'original'], treatment_df=app[app['variant'] == 'B']),\n"
            "    hypothesis_test('B', 'A', primary, control_df=app[app['variant'] == 'B'], treatment_df=app[app['variant'] == 'A']),\n"
            "], ignore_index=True)\n"
            "primary_tests['p_bonferroni'] = (primary_tests['p_value'] * 3).clip(upper=1)\n"
            "primary_tests['significant_bonferroni'] = primary_tests['p_bonferroni'] < 0.05\n"
            "display(primary_tests[['comparison','n_control','n_treat','x_control','x_treat','control_rate','treatment_rate','abs_lift','rel_lift','z_stat','p_value','p_bonferroni','significant_bonferroni','ci_low','ci_high']].style.format('{:.4f}', subset=['control_rate','treatment_rate','abs_lift','rel_lift','z_stat','p_value','p_bonferroni','ci_low','ci_high']))"
        ),
        nbf.v4.new_markdown_cell("### Visualization: Statistical Test Results"),
        nbf.v4.new_code_cell(
            "plot_hypothesis_results(\n"
            "    primary_tests,\n"
            "    significance_level=0.05 / 3,\n"
            "    title='Primary outcome pairwise tests',\n"
            ")"
        ),
        nbf.v4.new_markdown_cell("### Delivered-Only Sensitivity Check"),
        nbf.v4.new_code_cell(
            "delivered_app = app[app['delivered'] == 1].copy()\n"
            "delivered_summary = delivered_app.groupby('variant')[primary].agg(successes='sum', n='count', rate='mean').reindex(variants)\n"
            "delivered_control_rate = delivered_summary.loc['original', 'rate']\n"
            "delivered_summary['abs_lift_vs_original'] = delivered_summary['rate'] - delivered_control_rate\n"
            "delivered_summary['rel_lift_vs_original'] = delivered_summary['abs_lift_vs_original'] / delivered_control_rate\n"
            "display(delivered_summary.style.format('{:.4f}', subset=['rate', 'abs_lift_vs_original', 'rel_lift_vs_original']))\n\n"
            "delivered_tests = pd.concat([\n"
            "    hypothesis_test('original', 'A', primary, control_df=delivered_app[delivered_app['variant'] == 'original'], treatment_df=delivered_app[delivered_app['variant'] == 'A']),\n"
            "    hypothesis_test('original', 'B', primary, control_df=delivered_app[delivered_app['variant'] == 'original'], treatment_df=delivered_app[delivered_app['variant'] == 'B']),\n"
            "    hypothesis_test('B', 'A', primary, control_df=delivered_app[delivered_app['variant'] == 'B'], treatment_df=delivered_app[delivered_app['variant'] == 'A']),\n"
            "], ignore_index=True)\n"
            "delivered_tests['p_bonferroni'] = (delivered_tests['p_value'] * 3).clip(upper=1)\n"
            "delivered_tests['significant_bonferroni'] = delivered_tests['p_bonferroni'] < 0.05\n"
            "display(delivered_tests[['comparison','n_control','n_treat','x_control','x_treat','control_rate','treatment_rate','abs_lift','rel_lift','z_stat','p_value','p_bonferroni','significant_bonferroni','ci_low','ci_high']].style.format('{:.4f}', subset=['control_rate','treatment_rate','abs_lift','rel_lift','z_stat','p_value','p_bonferroni','ci_low','ci_high']))"
        ),
        nbf.v4.new_markdown_cell("### Visualization: Primary Outcome"),
        nbf.v4.new_code_cell(
            "primary_plot = primary_summary.copy()\n"
            "primary_plot['ci_low'] = [\n"
            "    max(0, rate - 1.96 * np.sqrt(rate * (1 - rate) / n))\n"
            "    for rate, n in zip(primary_plot['rate'], primary_plot['n'])\n"
            "]\n"
            "primary_plot['ci_high'] = [\n"
            "    min(1, rate + 1.96 * np.sqrt(rate * (1 - rate) / n))\n"
            "    for rate, n in zip(primary_plot['rate'], primary_plot['n'])\n"
            "]\n\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "colors = ['#6b7280', '#2563eb', '#f97316']\n"
            "labels = ['Original', 'Rewrite A', 'Rewrite B']\n"
            "x = np.arange(len(variants))\n"
            "rates = primary_plot['rate'].values * 100\n"
            "yerr = np.vstack((\n"
            "    (primary_plot['rate'] - primary_plot['ci_low']).values * 100,\n"
            "    (primary_plot['ci_high'] - primary_plot['rate']).values * 100,\n"
            "))\n"
            "bars = ax.bar(x, rates, color=colors, width=0.62)\n"
            "ax.errorbar(x, rates, yerr=yerr, fmt='none', ecolor='#111827', capsize=5)\n"
            "ax.axhline(primary_plot.loc['original', 'rate'] * 100, color='#6b7280', linestyle='--')\n"
            "ax.set_xticks(x, labels)\n"
            "ax.set_ylabel('Document submitted within 72h (%)')\n"
            "ax.set_title('Primary outcome by variant')\n"
            "ax.set_ylim(0, max((primary_plot['ci_high'] * 100).max() + 3, 24))\n"
            "for bar, rate in zip(bars, rates):\n"
            "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f'{rate:.1f}%', ha='center', va='bottom')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### Guardrails"),
        nbf.v4.new_code_cell(
            "guardrails = ['unsub_7d', 'complaint_7d', 'support_contact_7d']\n"
            "guardrail_rates = app.groupby('variant')[guardrails].mean().reindex(variants)\n"
            "guardrail_counts = app.groupby('variant')[guardrails].sum().reindex(variants)\n"
            "display(guardrail_rates.style.format('{:.4%}'))\n"
            "display(guardrail_counts)\n\n"
            "guardrail_tests = pd.concat([\n"
            "    hypothesis_test(\n"
            "        control,\n"
            "        treatment,\n"
            "        metric,\n"
            "        control_df=app[app['variant'] == control],\n"
            "        treatment_df=app[app['variant'] == treatment],\n"
            "    )\n"
            "    for metric in guardrails\n"
            "    for control, treatment in [('original', 'A'), ('original', 'B'), ('B', 'A')]\n"
            "], ignore_index=True)\n"
            "display(guardrail_tests[['metric','comparison','control_rate','treatment_rate','abs_lift','rel_lift','z_stat','p_value','ci_low','ci_high']].style.format('{:.4f}', subset=['control_rate','treatment_rate','abs_lift','rel_lift','z_stat','p_value','ci_low','ci_high']))"
        ),
        nbf.v4.new_markdown_cell("### Visualization: Guardrails"),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "metric_labels = ['Unsubscribe', 'Complaint', 'Support contact']\n"
            "width = 0.24\n"
            "x = np.arange(len(guardrails))\n"
            "colors = ['#6b7280', '#2563eb', '#f97316']\n"
            "labels = ['Original', 'Rewrite A', 'Rewrite B']\n\n"
            "for i, variant in enumerate(variants):\n"
            "    values = guardrail_rates.loc[variant].values * 100\n"
            "    ax.bar(x + (i - 1) * width, values, width=width, label=labels[i], color=colors[i])\n"
            "    for j, value in enumerate(values):\n"
            "        ax.text(j + (i - 1) * width, value + 0.08, f'{value:.2f}%', ha='center', fontsize=8)\n\n"
            "ax.set_xticks(x, metric_labels)\n"
            "ax.set_ylabel('7-day rate (%)')\n"
            "ax.set_title('Guardrail rates by variant')\n"
            "ax.legend(frameon=False)\n"
            "ax.set_ylim(0, max(guardrail_rates.max().max() * 100 + 1.2, 4))\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### Segment Review"),
        nbf.v4.new_code_cell(
            "for col in ['channel', 'risk_band', 'missing_doc_type']:\n"
            "    segment = app.groupby([col, 'variant'])[primary].agg(successes='sum', n='count', rate='mean').reset_index()\n"
            "    control = segment[segment['variant'] == 'original'][[col, 'rate']].rename(columns={'rate':'original_rate'})\n"
            "    segment = segment.merge(control, on=col)\n"
            "    segment['abs_lift_vs_original'] = segment['rate'] - segment['original_rate']\n"
            "    print('\\n' + col)\n"
            "    display(segment.pivot(index=col, columns='variant', values='rate').style.format('{:.2%}'))\n"
            "    display(segment.pivot(index=col, columns='variant', values='abs_lift_vs_original').style.format('{:.2%}'))"
        ),
        nbf.v4.new_markdown_cell("### Visualization: Segment Lift"),
        nbf.v4.new_code_cell(
            "segment_frames = []\n"
            "for col in ['channel', 'risk_band', 'missing_doc_type']:\n"
            "    segment = app.groupby([col, 'variant'])[primary].agg(successes='sum', n='count', rate='mean').reset_index()\n"
            "    control = segment[segment['variant'] == 'original'][[col, 'rate']].rename(columns={'rate': 'original_rate'})\n"
            "    segment = segment.merge(control, on=col)\n"
            "    segment['abs_lift_vs_original'] = segment['rate'] - segment['original_rate']\n"
            "    segment = segment[segment['variant'].isin(['A', 'B'])].copy()\n"
            "    segment['segment'] = col + ': ' + segment[col].astype(str)\n"
            "    segment_frames.append(segment[['segment', 'variant', 'abs_lift_vs_original']])\n\n"
            "segment_plot = pd.concat(segment_frames, ignore_index=True)\n"
            "segment_pivot = segment_plot.pivot(index='segment', columns='variant', values='abs_lift_vs_original').sort_index()\n\n"
            "fig, ax = plt.subplots(figsize=(9, 5.8))\n"
            "y = np.arange(len(segment_pivot.index))\n"
            "ax.axvline(0, color='#111827', linewidth=1)\n"
            "ax.barh(y - 0.18, segment_pivot['A'] * 100, height=0.32, label='Rewrite A', color='#2563eb')\n"
            "ax.barh(y + 0.18, segment_pivot['B'] * 100, height=0.32, label='Rewrite B', color='#f97316')\n"
            "ax.set_yticks(y, segment_pivot.index)\n"
            "ax.set_xlabel('Absolute lift vs Original (percentage points)')\n"
            "ax.set_title('Primary outcome lift by segment')\n"
            "ax.legend(frameon=False)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### Segment Hypothesis Tests"),
        nbf.v4.new_code_cell(
            "segment_tests = pd.concat([\n"
            "    hypothesis_test(\n"
            "        'original',\n"
            "        'A',\n"
            "        primary,\n"
            "        control_df=app[(app['risk_band'] == 'high') & (app['variant'] == 'original')],\n"
            "        treatment_df=app[(app['risk_band'] == 'high') & (app['variant'] == 'A')],\n"
            "    ).assign(segment='High risk'),\n"
            "    hypothesis_test(\n"
            "        'original',\n"
            "        'A',\n"
            "        primary,\n"
            "        control_df=app[(app['missing_doc_type'] == 'both') & (app['variant'] == 'original')],\n"
            "        treatment_df=app[(app['missing_doc_type'] == 'both') & (app['variant'] == 'A')],\n"
            "    ).assign(segment='Missing both docs'),\n"
            "    hypothesis_test(\n"
            "        'original',\n"
            "        'A',\n"
            "        primary,\n"
            "        control_df=app[(app['channel'] == 'sms') & (app['variant'] == 'original')],\n"
            "        treatment_df=app[(app['channel'] == 'sms') & (app['variant'] == 'A')],\n"
            "    ).assign(segment='SMS channel'),\n"
            "], ignore_index=True)\n\n"
            "segment_tests = segment_tests[\n"
            "    ['segment', 'comparison', 'n_control', 'n_treat', 'control_rate',\n"
            "     'treatment_rate', 'abs_lift', 'rel_lift', 'p_value', 'ci_low', 'ci_high']\n"
            "]\n\n"
            "display(segment_tests.style.format('{:.3f}', subset=[\n"
            "    'control_rate', 'treatment_rate', 'abs_lift', 'rel_lift',\n"
            "    'p_value', 'ci_low', 'ci_high'\n"
            "]).set_table_styles([\n"
            "    {'selector': 'th', 'props': [('white-space', 'nowrap'), ('width', 'auto')]},\n"
            "    {'selector': 'td', 'props': [('white-space', 'nowrap'), ('width', 'auto')]},\n"
            "]))"
        ),
        nbf.v4.new_markdown_cell("### Visualization: Segment Hypothesis Tests"),
        nbf.v4.new_code_cell(
            "plot_hypothesis_results(\n"
            "    segment_tests,\n"
            "    significance_level=0.05,\n"
            "    title='Priority segment hypothesis tests',\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## Part B - Written Recommendation\n\n"
            "Recommend Variant A as the winning candidate. Variant A improves `doc_submitted_72h` "
            "from 17.18% to 19.36%, an absolute lift of 2.17 percentage points and a relative "
            "lift of 12.65% versus Original. The A vs Original confidence interval for the lift "
            "is positive, and the result remains statistically significant after Bonferroni "
            "correction for the three pairwise primary-outcome comparisons.\n\n"
            "Variant B should not be rolled out. It has a smaller and statistically inconclusive "
            "primary-outcome lift, while its unsubscribe and support-contact rates are directionally "
            "worse than Original.\n\n"
            "Recommendation: roll out Variant A with monitoring rather than rolling out B. Monitor "
            "unsubscribe, complaint, and support-contact guardrails after launch. Segment-level "
            "results should be treated as directional because each subgroup has less sample than "
            "the overall experiment."
        ),
        nbf.v4.new_markdown_cell(drawing_conclusions_markdown()),
    ]
    nbf.write(nb, NOTEBOOK)


def main():
    global ANALYSIS_DF
    FIGURES.mkdir(exist_ok=True)
    app = load_analysis_data()
    ANALYSIS_DF = app

    summary = primary_summary(app)
    primary_tests = pairwise_tests(app, PRIMARY)
    power = power_analysis(primary_tests)
    guardrail_rates, guardrail_counts, guardrail_tests = guardrail_summary(app)
    srm = sample_ratio_check(app)
    balance = randomization_balance(app)
    _, delivered_summary, delivered_tests = delivered_sensitivity(app)
    segments = segment_summaries(app)
    segment_tests = priority_segment_tests(app)

    save_figures(summary, primary_tests, guardrail_rates, segments, segment_tests)
    write_report(
        summary,
        primary_tests,
        power,
        guardrail_rates,
        delivered_summary,
        delivered_tests,
    )
    write_notebook()


if __name__ == "__main__":
    main()
