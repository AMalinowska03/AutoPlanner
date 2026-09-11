import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.DbModels import ExperimentMetric, User
from data.database import SessionLocal


def load_data() -> pd.DataFrame:
    with SessionLocal() as session:
        query = (
            session.query(
                ExperimentMetric.id,
                ExperimentMetric.experiment_type,
                ExperimentMetric.algorithm,
                ExperimentMetric.user_id,
                ExperimentMetric.phase_order,
                ExperimentMetric.group_id,
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
                User.chronotype,
                User.procrastination_probability,
            )
            .join(User, ExperimentMetric.user_id == User.id)
        )
        df = pd.read_sql(query.statement, session.bind)
    return df


def get_sample_users_per_chronotype(df: pd.DataFrame) -> dict:
    """User sample: {'morning_lark': id, 'intermediate': id, 'night_owl': id}"""
    sample_users = {}
    for chronotype in ["morning_lark", "intermediate", "night_owl"]:
        subset = df[df["chronotype"] == chronotype]
        if not subset.empty:
            sample_users[chronotype] = int(subset["user_id"].iloc[0])
    return sample_users


def generate_all_plots(df: pd.DataFrame, sample_users: dict, output_dir: str = "../results"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    palette = {"baseline": "#7f7f7f", "ppo": "#1f77b4", "nsga": "#2ca02c"}

    # -------------------------------------------------------------
    # Instability score (Experiment 2)
    # -------------------------------------------------------------
    df_disr = df[df['experiment_type'] == 'disruptions']
    if not df_disr.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        sns.lineplot(
            ax=axes[0],
            data=df_disr,
            x="phase_order",
            y="instability",
            hue="algorithm",
            palette=palette,
            marker="o"
        )
        axes[0].set_title("Niestabilność Planu Pod Wpływem Zakłóceń")
        axes[0].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[0].set_ylabel("Instability Score (niższy = stabilniejszy)")

        sns.lineplot(
            ax=axes[1],
            data=df_disr,
            x="phase_order",
            y="total_replans",
            hue="algorithm",
            palette=palette,
            marker="s"
        )
        axes[1].set_title("Średnia Liczba Przeplanowań na Miesiąc")
        axes[1].set_xlabel("Faza zakłóceń (phase_order)")
        axes[1].set_ylabel("Liczba replanów")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plot_1_disruptions_impact.png"), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # Context Switch Efficiency
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df[df['experiment_type'] == 'online'],
        x="algorithm",
        y="switch_efficiency",
        palette=palette,
        showmeans=True
    )
    plt.title("Wskaźnik Efektywności Przełączania Zadań (Context Switching)")
    plt.ylabel("Wskaźnij efektywności (1.0 = brak strat)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_2_context_switch_boxplot.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Pareto Trade-off (Energy vs Delays)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=df[df['experiment_type'] == 'online'],
        x="delay_score",
        y="energy_score",
        hue="algorithm",
        palette=palette,
        alpha=0.5,
        s=35
    )
    plt.title("Kompromis Pareto: Dopasowanie Energii vs Opóźnienia Zadań")
    plt.xlabel("Wskaźnik opóźnień w godzinach (niższy = lepszy)")
    plt.ylabel("Wskaźnik energii (wyższy = lepszy)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_3_pareto_tradeoff.png"), dpi=300)
    plt.close()

    # =============================================================
    # Month by month analysis for 3 chosen users
    # =============================================================

    # Monthly Completion Score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df[df["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx],
            data=df_u,
            x="phase_order",
            y="monthly_completion_score",
            hue="algorithm",
            palette=palette,
            marker="o"
        )
        axes[idx].set_title(f"User #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Daily Completion Score" if idx == 0 else "")
        axes[idx].set_ylim(0, 1.05)

    plt.suptitle("Miesięczny wWkaźnik Ukończenia Zadań w Czasie", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_4_case_study_completion_monthly.png"), dpi=300)
    plt.close()

    # Daily Completion Score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df[df["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx],
            data=df_u,
            x="phase_order",
            y="daily_completion_score",
            hue="algorithm",
            palette=palette,
            marker="o"
        )
        axes[idx].set_title(f"User #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Daily Completion Score" if idx == 0 else "")
        axes[idx].set_ylim(0, 1.05)

    plt.suptitle("Dzienny Wskaźnik Ukończenia Zadań Dnia w Czasie", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_4_case_study_completion_daily.png"), dpi=300)
    plt.close()

    # Time Estimation Error
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df[df["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx],
            data=df_u,
            x="phase_order",
            y="time_estimation_error",
            hue="algorithm",
            palette=palette,
            marker="^"
        )
        axes[idx].set_title(f"User #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Błąd estymacji czasu trwania" if idx == 0 else "")

    plt.suptitle("Błąd Estymacji Czasu Zadań (Plan vs Rzeczywistość) w Czasie", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_5_case_study_estimation_error.png"), dpi=300)
    plt.close()

    # Delay Score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (chronotype, u_id) in enumerate(sample_users.items()):
        df_u = df[df["user_id"] == u_id]
        sns.lineplot(
            ax=axes[idx],
            data=df_u,
            x="phase_order",
            y="delay_score",
            hue="algorithm",
            palette=palette,
            marker="s"
        )
        axes[idx].set_title(f"User #{u_id} ({chronotype})")
        axes[idx].set_xlabel("Miesiąc eksperymentu (phase_order)")
        axes[idx].set_ylabel("Współczynnik opóźnienia (godz.)" if idx == 0 else "")

    plt.suptitle("Średnie Opóźnienia Zadań w Czasie dla Reprezentantów Chronotypów", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_6_case_study_delay.png"), dpi=300)
    plt.close()


def export_all_tables_to_latex(df: pd.DataFrame, phase_name:str, output_dir: str = "../results/"):
    os.makedirs(output_dir, exist_ok=True)
    df_phase = df[df['experiment_type'] == phase_name].copy()
    if df_phase.empty:
        print(f"[Warning] Brak danych dla fazy: {phase_name}")
        return

    cols_rename = {
        "energy_score": "Współczynnik dystrybucji energii",
        "monthly_completion_score": "Miesięczne ukończenie zadań",
        "daily_completion_score": "Dzienne ukończenie zadań",
        "time_estimation_error": "Błąd szacowania czasu",
        "delay_score": "Opóźnienie (godz.)",
        "switch_efficiency": "Wydajność przełączeń",
        "instability": "Niestabilność planu",
        "avg_generating_time": "Czas generowania (s)",
        "total_replans": "Liczba przeplanowań",
        "days_used": "Wykorzystane dni miesiąca",
    }
    tracked_metrics = list(cols_rename.keys())

    # -------------------------------------------------------------
    # TABLE 1: Summary per chronotype & algorithm (mean +- std)
    # -------------------------------------------------------------
    agg_chronotype = df_phase.groupby(["chronotype", "algorithm"])[tracked_metrics].agg(['mean', 'std'])

    chronotype_formatted = pd.DataFrame(index=agg_chronotype.index)
    for col in tracked_metrics:
        clean_name = cols_rename[col]
        chronotype_formatted[clean_name] = agg_chronotype.apply(
            lambda row: f"{row[(col, 'mean')]:.3f} $\\pm$ {row[(col, 'std')]:.3f}", axis=1
        )

    # -------------------------------------------------------------
    # TABLE 2: Global avg per algorithm (mean +- std)
    # -------------------------------------------------------------
    agg_global = df_phase.groupby("algorithm")[tracked_metrics].agg(['mean', 'std'])

    global_formatted = pd.DataFrame(index=agg_global.index)
    for col in tracked_metrics:
        clean_name = cols_rename[col]
        global_formatted[clean_name] = agg_global.apply(
            lambda row: f"{row[(col, 'mean')]:.3f} $\\pm$ {row[(col, 'std')]:.3f}", axis=1
        )

    # -------------------------------------------------------------
    # TABLE 3: Percentage gain/loss compare to Baseline
    # -------------------------------------------------------------
    mean_global = df_phase.groupby("algorithm")[tracked_metrics].mean().rename(columns=cols_rename)
    baseline_row = mean_global.loc["baseline"]

    # percent difference
    relative_gain = ((mean_global - baseline_row) / baseline_row) * 100.0
    relative_gain = relative_gain.drop(index="baseline")

    if "Czas generowania (s)" in relative_gain.columns:  # remove time for averages, pointless
        relative_gain = relative_gain.drop(columns=["Czas generowania (s)"])

    output_path = os.path.join(output_dir, f"tabele_faza_{phase_name}.tex")
    # save to tex
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"% =========================================================================\n")
        f.write(f"% WYNIKI DLA FAZY: {phase_name.upper()}\n")
        f.write(f"% =========================================================================\n\n")

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

        f.write("\n\n% --- TABELA 3: Zysk/strata względna do Baseline (%) bez czasu obliczeń ---\n")
        f.write(relative_gain.to_latex(
            float_format="%+.2f\\%%",
            caption=f"Względna zmiana wskaźników jakościowych względem Baseline (Faza: {phase_name})",
            label=f"tab:{phase_name}_relative_gain",
            position="htbp"
        ))

    print(f"Tables LaTeX generated and saved to: {output_path}")


if __name__ == "__main__":
    data = load_data()
    sample_users = get_sample_users_per_chronotype(data)
    print(f"Sample users: {sample_users}")

    generate_all_plots(data, sample_users, output_dir="../results")
    export_all_tables_to_latex(data, output_path="../results/tabele_wynikow.tex")