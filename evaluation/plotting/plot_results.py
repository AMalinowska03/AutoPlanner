import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.DbModels import ExperimentMetric, User
from data.database import SessionLocal


def load_data() -> pd.DataFrame:
    """Load all experiment metrics joined with user attributes."""
    with SessionLocal() as session:
        query = (
            session.query(
                ExperimentMetric.id,
                ExperimentMetric.experiment_type,
                ExperimentMetric.algorithm,
                ExperimentMetric.user_id,
                ExperimentMetric.phase_order,
                ExperimentMetric.total_replans,
                ExperimentMetric.days_used,
                ExperimentMetric.avg_generating_time,
                ExperimentMetric.monthly_completion_score,
                ExperimentMetric.daily_completion_score,
                ExperimentMetric.time_estimation_error,
                ExperimentMetric.delay_score,
                ExperimentMetric.energy_score,
                ExperimentMetric.switch_efficiency,
                ExperimentMetric.instability,
                ExperimentMetric.total_overtime_hours,
                ExperimentMetric.break_ratio,
                ExperimentMetric.break_count,
                ExperimentMetric.long_stretch_penalty,
                ExperimentMetric.urgent_delayed_count,
                User.chronotype,
                User.procrastination_probability,

            )
            .join(User, ExperimentMetric.user_id == User.id)
        )
        df = pd.read_sql(query.statement, session.bind)
    return df


def get_sample_users_per_chronotype(df: pd.DataFrame) -> dict:
    """Select one deterministic sample user ID per chronotype."""
    sample_users = {}
    for chronotype in ["morning_lark", "intermediate", "night_owl"]:
        subset = df[df["chronotype"] == chronotype]
        if not subset.empty:
            sample_users[chronotype] = int(subset["user_id"].iloc[0])
    return sample_users


def generate_phase_plots(df: pd.DataFrame, sample_users: dict, phase_name: str, base_output_dir: str = "../results"):
    """Generate all charts filtered strictly for the given experimental phase."""
    output_dir = os.path.join(base_output_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)

    df_phase = df[df['experiment_type'] == phase_name].copy()
    if df_phase.empty:
        print(f"[Warning] No data found for phase: {phase_name}")
        return

    sns.set_theme(style="whitegrid", font_scale=1.05)
    palette = {"baseline": "#7f7f7f", "ppo": "#0e6cc9", "nsga": "#0ec940"}

    # -------------------------------------------------------------
    # 1. Macro Analysis: Disruptions or Pareto Trade-off
    # -------------------------------------------------------------
    if phase_name == "disruptions":
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.lineplot(
            ax=axes[0], data=df_phase, x="phase_order", y="instability",
            hue="algorithm", palette=palette, marker="o"
        )
        axes[0].set_title("Niestabilność Planu pod Wpływem Zakłóceń")
        axes[0].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[0].set_ylabel("Wskaźnik niestabilności (niższy = stabilniejszy)")

        sns.lineplot(
            ax=axes[1], data=df_phase, x="phase_order", y="total_replans",
            hue="algorithm", palette=palette, marker="s"
        )
        axes[1].set_title("Średnia Liczba Przeplanowań na Miesiąc")
        axes[1].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[1].set_ylabel("Liczba przeplanowań")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_1_disruptions_impact.png"), dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(9, 5))
        sns.scatterplot(
            data=df_phase, x="delay_score", y="energy_score",
            hue="algorithm", palette=palette, alpha=0.5, s=35
        )
        plt.title("Kompromis Pareto: Dopasowanie Energii vs Opóźnienia Zadań")
        plt.xlabel("Wskaźnik opóźnień w godzinach (niższy = lepszy)")
        plt.ylabel("Wskaźnik dopasowania energii (wyższy = lepszy)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_1_pareto_tradeoff.png"), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # 2. Context Switch Efficiency
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df_phase, x="algorithm", y="switch_efficiency",
        palette=palette, showmeans=True
    )
    plt.title(f"Wskaźnik Efektywności Przełączania Zadań (Faza: {phase_name.upper()})")
    plt.xlabel("Algorytm")
    plt.ylabel("Wskaźnik efektywności (1.0 = brak strat)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_2_context_switch_boxplot.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Procrastination Impact (Psychological Feature Analysis)
    # -------------------------------------------------------------
    bins = [0.0, 0.33, 0.66, 1.0]
    tier_labels = ["Niska (<0.33)", "Średnia (0.33-0.66)", "Wysoka (>0.66)"]
    df_phase["procrastination_tier"] = pd.cut(df_phase["procrastination_probability"], bins=bins, labels=tier_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(
        ax=axes[0], data=df_phase, x="procrastination_tier", y="daily_completion_score",
        hue="algorithm", palette=palette, errorbar="sd"
    )
    axes[0].set_title(f"Wpływ Prokrastynacji na Dotrzymywanie Planu Dnia (Faza: {phase_name.upper()})")
    axes[0].set_xlabel("Poziom prokrastynacji pracownika")
    axes[0].set_ylabel("Wskaźnik ukończenia dnia (wyższy = lepszy)")
    axes[0].set_ylim(0, 1.05)

    sns.barplot(
        ax=axes[1], data=df_phase, x="procrastination_tier", y="delay_score",
        hue="algorithm", palette=palette, errorbar="sd"
    )
    axes[1].set_title(f"Wzrost Opóźnień (Delay Score) u Prokrastynatorów (Faza: {phase_name.upper()})")
    axes[1].set_xlabel("Poziom prokrastynacji pracownika")
    axes[1].set_ylabel("Suma opóźnień w godzinach (niższy = lepszy)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_3_procrastination_impact.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Operational Metrics: Days Used & Total Replans Monthly Trends
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.lineplot(
        ax=axes[0], data=df_phase, x="phase_order", y="days_used",
        hue="algorithm", palette=palette, marker="o", errorbar=None
    )
    axes[0].set_title(f"Średnia Liczba Wykorzystanych Dni Roboczych (Faza: {phase_name.upper()})")
    axes[0].set_xlabel("Miesiąc eksperymentu (phase_order)")
    axes[0].set_ylabel("Liczba dni (maks. 20)")
    axes[0].set_ylim(0, 21)

    sns.lineplot(
        ax=axes[1], data=df_phase, x="phase_order", y="total_replans",
        hue="algorithm", palette=palette, marker="s", errorbar=None
    )
    axes[1].set_title(f"Średnia Liczba Przeplanowań na Miesiąc (Faza: {phase_name.upper()})")
    axes[1].set_xlabel("Miesiąc eksperymentu (phase_order)")
    axes[1].set_ylabel("Liczba przeplanowań")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_4_operational_metrics.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Population Daily Completion by Chronotype
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df_phase, x="chronotype", y="daily_completion_score",
        hue="algorithm", palette=palette, errorbar="sd"
    )
    plt.title(f"Dzienny Wskaźnik Ukończenia Zadań wg Chronotypu (Faza: {phase_name.upper()})")
    plt.xlabel("Chronotyp pracownika")
    plt.ylabel("Wskaźnik ukończenia dnia (1.0 = 100%)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_5_completion_by_chronotype.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Breaks and overtime (Work-Life Balance)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.boxplot(
        ax=axes[0], data=df_phase, x="algorithm", y="break_ratio",
        palette=palette, showmeans=True
    )
    axes[0].axhspan(10, 15, color='green', alpha=0.15, label='Pożądany przedział (10-15%)')
    axes[0].set_title(f"Udział Czasu Przerw w Czasie Pracy (Faza: {phase_name.upper()})")
    axes[0].set_xlabel("Algorytm")
    axes[0].set_ylabel("Udział przerw (%)")
    axes[0].legend(loc="upper right")

    sns.boxplot(
        ax=axes[1], data=df_phase, x="algorithm", y="total_overtime_hours",
        palette=palette, showmeans=True
    )
    axes[1].set_title(f"Miesięczny Czas Nadgodzin w Godzinach (Faza: {phase_name.upper()})")
    axes[1].set_xlabel("Algorytm")
    axes[1].set_ylabel("Nadgodziny łączne (h)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_10_ergonomics_and_overtime.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Urgent Tasks Protection
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_phase, x="algorithm", y="urgent_delayed_count",
        palette=palette, errorbar="sd"
    )
    plt.title(f"Średnia Liczba Opóźnionych Zadań Pilnych 'Urgent' (Faza: {phase_name.upper()})")
    plt.xlabel("Algorytm")
    plt.ylabel("Liczba spóźnionych zadań pilnych (niższy = lepszy)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_11_urgent_protection.png"), dpi=300)
    plt.close()

    # =============================================================
    # 6. Case Study: 3 Sample Users Month-by-Month Trends
    # =============================================================

    # Monthly Completion Rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df_phase[df_phase["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx], data=df_u, x="phase_order", y="monthly_completion_score",
            hue="algorithm", palette=palette, marker="o"
        )
        axes[idx].set_title(f"Pracownik #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Miesięczny wskaźnik ukończenia" if idx == 0 else "")
        axes[idx].set_ylim(0, 1.05)
    plt.suptitle(f"Miesięczny Wskaźnik Ukończenia Zadań w Czasie (Faza: {phase_name.upper()})", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_6_case_study_completion_monthly.png"), dpi=300)
    plt.close()

    # Daily Completion Rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df_phase[df_phase["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx], data=df_u, x="phase_order", y="daily_completion_score",
            hue="algorithm", palette=palette, marker="o"
        )
        axes[idx].set_title(f"Pracownik #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Dzienny wskaźnik ukończenia" if idx == 0 else "")
        axes[idx].set_ylim(0, 1.05)
    plt.suptitle(f"Dzienny Wskaźnik Dotrzymywania Planu w Czasie (Faza: {phase_name.upper()})", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_7_case_study_completion_daily.png"), dpi=300)
    plt.close()

    # Time Estimation Error
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df_phase[df_phase["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx], data=df_u, x="phase_order", y="time_estimation_error",
            hue="algorithm", palette=palette, marker="^"
        )
        axes[idx].set_title(f"Pracownik #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Względny błąd estymacji czasu" if idx == 0 else "")
    plt.suptitle(f"Błąd Estymacji Czasu Zadań (Plan vs Rzeczywistość) w Czasie (Faza: {phase_name.upper()})", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_8_case_study_estimation_error.png"), dpi=300)
    plt.close()

    # Delay Score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df_phase[df_phase["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx], data=df_u, x="phase_order", y="delay_score",
            hue="algorithm", palette=palette, marker="s"
        )
        axes[idx].set_title(f"Pracownik #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Suma opóźnień w godzinach" if idx == 0 else "")
    plt.suptitle(f"Średnie Opóźnienia Zadań w Czasie (Faza: {phase_name.upper()})", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_9_case_study_delay.png"), dpi=300)
    plt.close()

    print(f"Pomyślnie wygenerowano wykresy dla fazy '{phase_name}' w katalogu: {output_dir}")


def extract_derived_metrics_summary(
        df: pd.DataFrame,
        phase_name: str = "online",
        base_output_dir: str = "../results/"
):
    """
    Computes derived ergonomic and energy-tradeoff metrics
    strictly from existing columns in ExperimentMetric and User tables.
    """
    output_dir = os.path.join(base_output_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Filter strictly for the requested phase
    df_phase = df[df['experiment_type'] == phase_name].copy()
    if df_phase.empty:
        print(f"[Warning] No data found for phase: {phase_name}")
        return

    # 2. Calculate derived metrics on existing DataFrame columns
    # Trade-off: how much energy alignment was gained per hour of delay
    df_phase["energy_delay_ratio"] = df_phase["energy_score"] / (df_phase["delay_score"] + 1.0)

    # Work density: fraction of the 20-day horizon actually utilized
    df_phase["work_density"] = df_phase["days_used"] / 20.0

    # Daily pacing fidelity: consistency of daily vs monthly delivery
    df_phase["pacing_balance"] = df_phase["daily_completion_score"] / df_phase["monthly_completion_score"].clip(
        lower=1e-4)

    # In disruptions phase: instability cost per single replan event
    if phase_name == "disruptions":
        df_phase["instability_per_replan"] = df_phase["instability"] / (df_phase["total_replans"] + 1.0)

    derived_cols = ["energy_delay_ratio", "work_density", "pacing_balance"]
    if phase_name == "disruptions":
        derived_cols.append("instability_per_replan")

    # 3. Aggregate by algorithm (mean +- std)
    agg_summary = df_phase.groupby("algorithm")[derived_cols].agg(['mean', 'std'])

    cols_rename = {
        "energy_delay_ratio": "Efektywność kompromisu energii",
        "work_density": "Gęstość pracy (utylizacja miesiąca)",
        "pacing_balance": "Równomierność tempa pracy",
        "instability_per_replan": "Szok planu na przeplanowanie",
    }

    formatted_summary = pd.DataFrame(index=agg_summary.index)
    for col in derived_cols:
        clean_name = cols_rename[col]
        formatted_summary[clean_name] = agg_summary.apply(
            lambda row: f"{row[(col, 'mean')]:.3f} $\\pm$ {row[(col, 'std')]:.3f}", axis=1
        )

    # 4. Save to LaTeX
    tex_path = os.path.join(output_dir, f"tabela_wskazniki_pochodne_{phase_name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% =========================================================================\n")
        f.write(f"% WSKAŹNIKI POCHODNE ERGONOMII I ENERGII DLA FAZY: {phase_name.upper()}\n")
        f.write("% =========================================================================\n\n")
        f.write(formatted_summary.to_latex(
            caption=f"Wskaźniki pochodne ergonomii, tempa pracy i kompromisów energetycznych (Faza: {phase_name})",
            label=f"tab:derived_metrics_{phase_name}",
            position="htbp"
        ))

    print(f"Zapisano tabelę wskaźników pochodnych dla fazy '{phase_name}' w: {tex_path}")

def export_all_tables_to_latex(df: pd.DataFrame, phase_name: str, base_output_dir: str = "../results/"):
    """Export formatted summary LaTeX tables for a specific phase."""
    output_dir = os.path.join(base_output_dir, phase_name)
    os.makedirs(output_dir, exist_ok=True)

    df_phase = df[df['experiment_type'] == phase_name].copy()
    if df_phase.empty:
        print(f"[Warning] No data found for phase: {phase_name}")
        return

    cols_rename = {
        "energy_score": "Dystrybucja energii",
        "monthly_completion_score": "Ukończenie miesięczne",
        "daily_completion_score": "Ukończenie dzienne",
        "time_estimation_error": "Błąd estymacji czasu",
        "delay_score": "Opóźnienia (godz.)",
        "switch_efficiency": "Efektywność przełączeń",
        "instability": "Niestabilność planu",
        "avg_generating_time": "Czas generowania (s)",
        "total_replans": "Liczba przeplanowań",
        "days_used": "Wykorzystane dni",
        "total_overtime_hours": "Nadgodziny (h)",
        "break_ratio": "Udział przerw (%)",
        "break_count": "Liczba przerw (szt.)",
        "long_stretch_penalty": "Praca ciągła >3.5h (kara)",
        "urgent_delayed_count": "Opóźnione pilne zadania (szt.)"
    }
    tracked_metrics = list(cols_rename.keys())

    # Table 1: Chronotype x Algorithm (mean +- std)
    agg_chronotype = df_phase.groupby(["chronotype", "algorithm"])[tracked_metrics].agg(['mean', 'std'])
    chronotype_formatted = pd.DataFrame(index=agg_chronotype.index)
    for col in tracked_metrics:
        clean_name = cols_rename[col]
        chronotype_formatted[clean_name] = agg_chronotype.apply(
            lambda row: f"{row[(col, 'mean')]:.3f} $\\pm$ {row[(col, 'std')]:.3f}", axis=1
        )

    # Table 2: Global averages per algorithm (mean +- std)
    agg_global = df_phase.groupby("algorithm")[tracked_metrics].agg(['mean', 'std'])
    global_formatted = pd.DataFrame(index=agg_global.index)
    for col in tracked_metrics:
        clean_name = cols_rename[col]
        global_formatted[clean_name] = agg_global.apply(
            lambda row: f"{row[(col, 'mean')]:.3f} $\\pm$ {row[(col, 'std')]:.3f}", axis=1
        )

    # Table 3: Relative gain/loss compared to Baseline
    mean_global = df_phase.groupby("algorithm")[tracked_metrics].mean().rename(columns=cols_rename)
    baseline_row = mean_global.loc["baseline"]
    relative_gain = ((mean_global - baseline_row) / baseline_row) * 100.0
    relative_gain = relative_gain.drop(index="baseline")
    if "Czas generowania (s)" in relative_gain.columns:
        relative_gain = relative_gain.drop(columns=["Czas generowania (s)"])

    output_path = os.path.join(output_dir, f"tabele_{phase_name}.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% =========================================================================\n")
        f.write(f"% WYNIKI DLA FAZY: {phase_name.upper()}\n")
        f.write("% =========================================================================\n\n")

        f.write("% --- TABELA 1: Chronotyp x Algorytm (mean ± std) ---\n")
        f.write(chronotype_formatted.to_latex(
            caption=f"Wyniki efektywności w rozbiciu na chronotypy (Faza: {phase_name})",
            label=f"tab:{phase_name}_by_chronotype",
            position="htbp"
        ))

        f.write("\n\n% --- TABELA 2: Średnie globalne (mean ± std) ---\n")
        f.write(global_formatted.to_latex(
            caption=f"Globalne zestawienie parametrów algorytmów (Faza: {phase_name})",
            label=f"tab:{phase_name}_global_metrics",
            position="htbp"
        ))

        f.write("\n\n% --- TABELA 3: Zysk/strata względna do Baseline (%) ---\n")
        f.write(relative_gain.to_latex(
            float_format="%+.2f\\%%",
            caption=f"Względna zmiana wskaźników jakościowych względem Baseline (Faza: {phase_name})",
            label=f"tab:{phase_name}_relative_gain",
            position="htbp"
        ))

    print(f"Pomyślnie wygenerowano tabele LaTeX dla fazy '{phase_name}' w: {output_path}")


if __name__ == "__main__":
    data = load_data()
    sample_users = get_sample_users_per_chronotype(data)
    print(f"Chosen user sample: {sample_users}")

    # Process phase: 'online' (Experiment 1)
    # generate_phase_plots(data, sample_users, phase_name="online", base_output_dir="../results")
    # export_all_tables_to_latex(data, phase_name="online", base_output_dir="../results")
    # extract_derived_metrics_summary(data, phase_name="online", base_output_dir="../results")

    # Process phase: 'disruptions' (Experiment 2)
    generate_phase_plots(data, sample_users, phase_name="disruptions", base_output_dir="../results")
    export_all_tables_to_latex(data, phase_name="disruptions", base_output_dir="../results")
    extract_derived_metrics_summary(data, phase_name="disruptions", base_output_dir="../results")
