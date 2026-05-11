import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ============================================================
# Styling
# ============================================================
plt.style.use("default")
sns.set_theme(style='white',palette="Set2")

# 1) Page Config must be the FIRST streamlit command
st.set_page_config(layout="centered", page_title="COVID-19 Response Dashboard")
st.markdown("""
<style>
/* Generic card container */
.card {
    background-color: white;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

/* KPI card */
.kpi-card {
    background-color: white;
    border: 1px solid #DADADA;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}

/* Chart card */
.chart-card {
    background-color: white;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 10px;
}

/* Subtle shadow (optional) */
.card-shadow {
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

### title setup
st.markdown(
    "<h2 style='text-align: center;'>COVID-19 Survey Data on Symptoms, Demographics, and Mental Health</h2>",
    unsafe_allow_html=True
)

def chart_card(render_fn):
    with st.container():
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        render_fn()
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Data load
# ============================================================
@st.cache_data
def load_data():
    df_ = pd.read_excel("cleaned_COVID_Survey.xlsx")
    df_.columns = df_.columns.str.strip().str.replace(" ", "_")
    return df_

df = load_data()

# ============================================================
# Helpers
# ============================================================
def to_int_col(frame: pd.DataFrame, col: str):
    if col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)

def to_num_col(frame: pd.DataFrame, col: str):
    if col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

def st_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

# Normalize column names ONCE
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Core columns
symptom_cols = ["Symptom_Fever", "Symptom_Cough", "Symptom_Breath_Shortness"]
for c in symptom_cols:
    to_int_col(df, c)

for c in ["Known_Exposure", "Recent_Traveler", "Has_Preexisting_Condition", "COVID_Tested", "Is_Probable_Case"]:
    to_int_col(df, c)
# ============================================================
# KPI CALCULATIONS
# ============================================================

# ----------  Overall Symptom Prevalence (%) ----------
symptom_cols = ["Symptom_Fever", "Symptom_Cough", "Symptom_Breath_Shortness"]
if all(c in df.columns for c in symptom_cols):
    for c in symptom_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    overall_symptom_prev = (df[symptom_cols].sum(axis=1) > 0).mean() * 100
else:
    overall_symptom_prev = 0.0


# ---------- Probable Case Rate (%) ----------
if "Is_Probable_Case" in df.columns:
    df["Is_Probable_Case"] = pd.to_numeric(df["Is_Probable_Case"], errors="coerce").fillna(0)
    probable_case_rate = df["Is_Probable_Case"].mean() * 100
else:
    probable_case_rate = 0.0


# More meaningful than .count(): average number of symptoms reported per respondent
avg_symptoms_preveliance = df[symptom_cols].sum(axis=1).mean() if all(c in df.columns for c in symptom_cols) else 0.0

probable_cases = int(df["Is_Probable_Case"].sum()) if "Is_Probable_Case" in df.columns else 0

# ============================================================
# Tabs
# ============================================================
tab_clinical, tab_shielding, tab_policy, tab_wellbeing, tab_corr = st.tabs([
    "📊  KPIs &🫁 Clinical Triage ",
    "🧬 High-Risk Shielding",
    "🧾 Policy & Quarantine Optimization",
    "🌿 Resource allocation,Wellbeing & Recovery ",
    "🔗 Correlation",
])

# ============================================================
# TAB: CLINICAL TRIAGE
# ============================================================
with tab_clinical:
    
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="kpi-card card-shadow">
        <div style="font-size:14px; color:#555;">Overall Symptom Prevalence</div>
        <div style="font-size:32px; font-weight:700; margin-top:4px;">
            {overall_symptom_prev:.1f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card card-shadow">
        <div style="font-size:14px; color:#555;">Avg Symptoms per Respondent</div>
        <div style="font-size:32px; font-weight:700; margin-top:4px;">
            {avg_symptoms_preveliance:.2f}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card card-shadow">
        <div style="font-size:14px; color:#555;">Probable Case Rate</div>
        <div style="font-size:32px; font-weight:700; margin-top:4px;">
            {probable_case_rate:.1f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # Symptom Cluster → Probable Case (Green-Lane)
    # -------------------------

    st.subheader("Clinical Decision Support: Green-Lane Testing Candidates")

    needed_cols = {
        "Symptom_Fever",
        "Symptom_Cough",
        "Symptom_Breath_Shortness",
        "Is_Probable_Case"
    }

    if not needed_cols.issubset(df.columns):
        st.warning(f"Missing required columns: {sorted(list(needed_cols - set(df.columns)))}")
    else:
        df_temp = df.copy()

        # --------------------------------------------------
        #  Ensure numeric symptom + probable columns
        # --------------------------------------------------
        symptom_cols = [
            "Symptom_Fever",
            "Symptom_Cough",
            "Symptom_Breath_Shortness"
        ]

        for col in symptom_cols + ["Is_Probable_Case"]:
            df_temp[col] = (
                pd.to_numeric(df_temp[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        # --------------------------------------------------
        #  Create Symptom Cluster
        # --------------------------------------------------
        def get_cluster(row):
            s = []
            if row["Symptom_Fever"] == 1:
                s.append("Fever")
            if row["Symptom_Cough"] == 1:
                s.append("Cough")
            if row["Symptom_Breath_Shortness"] == 1:
                s.append("Shortness of Breath")
            return " + ".join(s) if s else "No Major Symptoms"

        df_temp["Symptom_Cluster"] = df_temp.apply(get_cluster, axis=1)

        # --------------------------------------------------
        # 3) Analyze probability per cluster
        # --------------------------------------------------
        analysis = (
            df_temp.groupby("Symptom_Cluster")["Is_Probable_Case"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        if analysis.empty:
            st.info("No valid symptom cluster data available.")
        else:
            # --------------------------------------------------
            # 4) Visualization
            # --------------------------------------------------
            fig, ax = plt.subplots(figsize=(6.8, 4.5))

            bar_colors = [
                "#C62828" if v >= 90 else "#455A64"
                for v in analysis.values
            ]

            sns.barplot(
                x=analysis.values,
                y=analysis.index,
                ax=ax,
                palette=bar_colors,
                edgecolor="black",
                linewidth=1
            )

            # 90% Threshold Line
            ax.axvline(
                90,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Green-Lane Threshold (90%)"
            )

            # Data labels
            for i, v in enumerate(analysis.values):
                ax.text(
                    v + 1,
                    i,
                    f"{v:.1f}%",
                    va="center",
                    fontweight="bold",
                    fontsize=9
                )

            
            ax.set_xlabel("Probability of Being a Probable Case (%)")
            ax.set_ylabel("Symptom Cluster")
            ax.set_xlim(0, 100)

            ax.legend(loc="lower right")

            #plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # -------------------------
    # Probable COVID Classification by indicators (pie charts)
    # -------------------------
    st.subheader("Probable COVID-19 Classification by Exposure & Health Indicators")

    def plot_pie(ax, series, title):
        series = series.reindex([0, 1]).fillna(0)
        labels = ["No", "Yes"]
        explode = (0, 0.08)
        colors = ["#1E8449", "#922B21"]

        ax.pie(
            series.values,
            labels=labels,
            autopct="%.1f%%",
            startangle=90,
            explode=explode,
            shadow=True,
            colors=colors,
            textprops={"fontsize": 10}
        )
        ax.set_title(title, fontsize=12, fontweight="bold")

    needed = {"Is_Probable_Case", "Known_Exposure", "Recent_Traveler", "Has_Preexisting_Condition", "COVID_Tested"}

    if needed.issubset(set(df.columns)):
        for c in ["Known_Exposure", "Recent_Traveler", "Has_Preexisting_Condition", "COVID_Tested", "Is_Probable_Case"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        known_exposure = df.groupby("Known_Exposure")["Is_Probable_Case"].mean() * 100
        recent_travel  = df.groupby("Recent_Traveler")["Is_Probable_Case"].mean() * 100
        pre_existing   = df.groupby("Has_Preexisting_Condition")["Is_Probable_Case"].mean() * 100
        covid_tested   = df.groupby("COVID_Tested")["Is_Probable_Case"].mean() * 100

        fig, axes = plt.subplots(2, 2, figsize=(5,4))
        axes = axes.flatten()

        plot_pie(axes[0], known_exposure, "Known Exposure")
        plot_pie(axes[1], recent_travel,  "Recent Travel")
        plot_pie(axes[2], pre_existing,   "Pre-existing Condition")
        plot_pie(axes[3], covid_tested,   "COVID Testing Status")
        st_plot(fig)
    else:
        st.warning(
            "Missing columns for this view. Required: "
            "Known_Exposure, Recent_Traveler, Has_Preexisting_Condition, COVID_Tested, Is_Probable_Case"
        )

    st.divider()

    # -------------------------
    # Tobacco Users pie
    # -------------------------
    st.subheader("Probable COVID-19 Classification Among Tobacco Users")

    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_")

    required_cols = {"Tobacco_Usage", "Is_Probable_Case"}

    if required_cols.issubset(df_temp.columns):

        df_temp["Tobacco_Usage_Label"] = df_temp["Tobacco_Usage"].astype(str).str.lower().str.strip()

        df_temp["Tobacco_Usage_Flag"] = df_temp["Tobacco_Usage_Label"].map({
            "current smoker": 1,
            "former smoker": 1,
            "yes": 1,
            "1": 1,
            "never smoker": 0,
            "no": 0,
            "0": 0
        })

        df_temp["Is_Probable_Case"] = pd.to_numeric(df_temp["Is_Probable_Case"], errors="coerce")

        df_temp = df_temp.dropna(subset=["Tobacco_Usage_Flag", "Is_Probable_Case"])

        tobacco_users = df_temp[df_temp["Tobacco_Usage_Flag"] == 1]

        if len(tobacco_users) > 0:
            probable_dist = (
                tobacco_users["Is_Probable_Case"]
                .value_counts(normalize=True)
                .reindex([0, 1])
                .fillna(0)
                * 100
            )

            labels = ["Not Probable Case", "Probable Case"]
            colors = ["#1E8449", "#922B21"]

            fig, ax = plt.subplots(figsize=(4,3))
            ax.pie(
                probable_dist.values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                explode=(0, 0.08),
                shadow=True,
                textprops={"fontsize": 11}
            )
            st_plot(fig)
            st.caption(f"Sample size (tobacco users): {len(tobacco_users)}")
        else:
            st.info("No valid tobacco-user records available for analysis.")
    else:
        st.warning("Missing required columns: Tobacco_Usage, Is_Probable_Case")

# ============================================================
# TAB: High Risk Group Shielding
# ============================================================
with tab_shielding:
   
    st.subheader("Urgency Analysis: COVID-19 Probability by Chronic Condition")

    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_")

    required_cols = {"Is_Probable_Case", "Conditions"}

    if required_cols.issubset(set(df_temp.columns)):

        df_temp["Is_Probable_Case"] = pd.to_numeric(df_temp["Is_Probable_Case"], errors="coerce").fillna(0).astype(int)

        exploded_df = (
            df_temp.assign(
                Condition=df_temp["Conditions"].fillna("NR").astype(str).str.split(",")
            ).explode("Condition")
        )
        exploded_df["Condition"] = exploded_df["Condition"].astype(str).str.strip()

        exclude = ["NR", "noneOfTheAbove", "other", ""]
        specific_df = exploded_df[~exploded_df["Condition"].isin(exclude)]

        if len(specific_df) > 0:
            risk_stats = (
                specific_df.groupby("Condition")["Is_Probable_Case"]
                .mean().mul(100).sort_values(ascending=False)
            )

            #st.markdown("### Urgency Score Table")
            #formatted_table = risk_stats.apply(lambda x: f"{x:.2f}%").to_frame(name="Urgency_Score_(Risk%)")
            #st.dataframe(formatted_table, use_container_width=True)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=risk_stats.values,
                y=risk_stats.index,
                palette="Reds_r",
                edgecolor="black",
                linewidth=1,
                ax=ax
            )

            for i, v in enumerate(risk_stats.values):
                ax.text(v + 1, i, f"{v:.2f}%", va="center", fontweight="bold")

            #ax.set_title("Urgency Analysis: COVID-19 Probability by Chronic Condition", fontsize=14, fontweight="bold")
            ax.set_xlabel("Risk Probability (%)")
            ax.set_ylabel("")
            st_plot(fig)


        else:
            st.warning("No valid chronic condition records available after filtering.")
    else:
        st.warning(f"Missing required columns for this analysis: {sorted(list(required_cols - set(df_temp.columns)))}")

    st.divider()

    # ============================================================
    # Age Groups Risk Index
    # ============================================================
    st.subheader("Age Groups Requiring Intensified Risk Communication (Risk Index)")

    required_cols_age = {"Age_Range", "Is_Probable_Case"}
    if required_cols_age.issubset(set(df.columns)):

        df_temp = df.copy()
        df_temp["Is_Probable_Case"] = pd.to_numeric(df_temp["Is_Probable_Case"], errors="coerce")
        df_temp = df_temp.dropna(subset=["Age_Range", "Is_Probable_Case"])

        age_summary = (
            df_temp.groupby("Age_Range")
            .agg(
                Participant_Count=("Age_Range", "count"),
                Probable_Rate=("Is_Probable_Case", "mean")
            ).reset_index()
        )

        age_summary["Probable_Rate"] *= 100
        age_summary["Population_Share"] = (age_summary["Participant_Count"] / age_summary["Participant_Count"].sum()) * 100
        age_summary["Risk_Index"] = age_summary["Probable_Rate"] / age_summary["Population_Share"].replace(0, np.nan)
        age_summary = age_summary.dropna(subset=["Risk_Index"]).sort_values("Risk_Index")

        fig, ax = plt.subplots(figsize=(5, 4))
        y_pos = np.arange(len(age_summary))

        ax.hlines(y=y_pos, xmin=0, xmax=age_summary["Risk_Index"], color="#1E8449", linewidth=6)
        ax.plot(age_summary["Risk_Index"], y_pos, "o", color="#F1C40F",
                markersize=15, markeredgecolor="black", markeredgewidth=1.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(age_summary["Age_Range"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Relative Risk Index (Higher = Higher Priority)", fontsize=12, fontweight="bold")
        ax.grid(False)

        legend_elements = [
            Line2D([0], [0], color="#1E8449", lw=6, label="Relative Risk Intensity"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#F1C40F",
                   markeredgecolor="black",
                   markersize=14,
                   label="Age Group Priority Point")
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=12)
        st_plot(fig)
        st.dataframe(
            age_summary[["Age_Range", "Participant_Count", "Probable_Rate", "Population_Share", "Risk_Index"]]
            .sort_values("Risk_Index", ascending=False),
            use_container_width=True
        )
    else:
        st.warning(f"Missing columns for age risk view: {sorted(list(required_cols_age - set(df.columns)))}")

    st.divider()

    # ============================================================
    # Vulnerability Under 60
    # ============================================================
    st.subheader("Vulnerability Among Respondents Under Age 60")

    required_cols_u60 = {"Age_Range", "Has_Preexisting_Condition"}
    if required_cols_u60.issubset(set(df.columns)):

        df_under_60 = df[df["Age_Range"] == "<60"].copy()

        if len(df_under_60) > 0:
            df_under_60["Has_Preexisting_Condition"] = pd.to_numeric(
                df_under_60["Has_Preexisting_Condition"], errors="coerce"
            ).fillna(0)

            vulnerable_count = int(df_under_60["Has_Preexisting_Condition"].sum())
            vulnerable_rate = df_under_60["Has_Preexisting_Condition"].mean() * 100

            c1, c2 = st.columns(2)
            c1.metric("Vulnerable Individuals (<60)", vulnerable_count)
            c2.metric("Vulnerability Rate (<60)", f"{vulnerable_rate:.2f}%")

            st.info("Prescriptive action: prioritize protective measures for vulnerable under-60 individuals.")
        else:
            st.warning("No respondents found under age 60 (Age_Range == '<60').")
    else:
        st.warning(f"Missing columns for under-60 vulnerability view: {sorted(list(required_cols_u60 - set(df.columns)))}")

# ============================================================
# TAB: POLICY Optimization
# ============================================================
with tab_policy:
    st.subheader("Mobility & Travel-Based COVID-19 Risk Analysis")

    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_")

    required_cols = {"Travel_Within_Canada", "Is_Probable_Case"}

    if required_cols.issubset(set(df_temp.columns)):

        df_temp["Is_Probable_Case"] = pd.to_numeric(df_temp["Is_Probable_Case"], errors="coerce").fillna(0).astype(int)

        travel_groups = ["Frontline Worker", "Non-Essential Travel", "Remote Work", "Not a Commuter", "Stopped Commuting"]

        df_travel = df_temp[df_temp["Travel_Within_Canada"].isin(travel_groups)].copy()

        if len(df_travel) > 0:
            probable_by_travel = (
                df_travel.groupby("Travel_Within_Canada")["Is_Probable_Case"]
                .mean().mul(100).reset_index(name="Probable_Case_Rate_%")
            )

            group_counts = (
                df_travel["Travel_Within_Canada"]
                .value_counts().rename_axis("Travel_Within_Canada")
                .reset_index(name="Respondents_n")
            )

            probable_by_travel = probable_by_travel.merge(group_counts, on="Travel_Within_Canada", how="left")
            probable_by_travel["Probable_Case_Rate_%"] = probable_by_travel["Probable_Case_Rate_%"].round(2)

            st.markdown("### Probable COVID-19 Case Rate by Travel Pattern")
            st.dataframe(probable_by_travel, use_container_width=True)

            high_mobility = ["Frontline Worker", "Non-Essential Travel"]
            low_mobility = ["Remote Work", "Not a Commuter", "Stopped Commuting"]

            df_travel["Mobility_Group"] = df_travel["Travel_Within_Canada"].apply(
                lambda x: "High Mobility" if x in high_mobility else "Low Mobility"
            )

            probable_counts = df_travel[df_travel["Is_Probable_Case"] == 1]["Mobility_Group"].value_counts()
            ordered_groups = ["High Mobility", "Low Mobility"]
            probable_counts = probable_counts.reindex(ordered_groups).fillna(0)

            st.subheader("Share of Probable COVID-19 Cases by Mobility Group")

            fig, ax = plt.subplots(figsize=(5, 4))
            wedges, texts, autotexts = ax.pie(
                probable_counts.values,
                autopct="%.2f%%",
                startangle=140,
                explode=(0.08, 0),
                colors=["firebrick", "lightgreen"],
                pctdistance=0.65,
                textprops={"fontsize": 11}
            )

            ax.legend(
                wedges,
                [
                    "High Mobility (Frontline Worker, Non-Essential Travel)",
                    "Low Mobility (Remote Work, Not a Commuter, Stopped Commuting)"
                ],
                title="Mobility Groups",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5)
            )

            ax.set_title("Share of Probable COVID-19 Cases by Mobility Group", fontsize=13, pad=10)
            ax.axis("equal")
            st_plot(fig)
        else:
            st.warning("No valid records found for selected travel categories.")
    else:
        st.warning(f"Missing required columns for mobility analysis: {sorted(list(required_cols - set(df_temp.columns)))}")

    st.divider()

    # Tobacco × Quarantine heatmap
    st.subheader("Tobacco Usage × Quarantine Profile")
    if {"Tobacco_Usage", "Quarantine", "Is_Probable_Case"}.issubset(set(df.columns)):
        temp = df.copy()
        temp = temp[
            (temp["Tobacco_Usage"].astype(str).str.lower() != "nr") &
            (temp["Quarantine"].astype(str).str.lower() != "nr")
        ].copy()

        summary = (
            temp.groupby(["Tobacco_Usage", "Quarantine"])
            .agg(Probable_Rate=("Is_Probable_Case", "mean"), Count=("Is_Probable_Case", "size"))
            .reset_index()
        )
        summary["Probable_Rate_%"] = summary["Probable_Rate"] * 100
        st.dataframe(summary.sort_values("Probable_Rate_%", ascending=False), use_container_width=True)

        pivot = summary.pivot(index="Tobacco_Usage", columns="Quarantine", values="Probable_Rate_%")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title("Probable Case Rate (%) — Tobacco × Quarantine")
        st_plot(fig)
    else:
        st.caption("Tobacco_Usage/Quarantine columns not found; heatmap will appear when available.")

# ============================================================
# TAB: Resource Allocation ,WELLBEING & RECOVERY
# ============================================================

with tab_wellbeing:
    
    st.subheader("FSA Support Prioritization (Medical vs Financial)")

    needed_fsa = {
        "Symptom_Breath_Shortness", "Symptom_Fever", "Symptom_Cough",
        "Known_Exposure", "Has_Preexisting_Condition",
        "Age_Range", "Is_Probable_Case", "Postal_District"
    }

    if not needed_fsa.issubset(df.columns):
        st.warning(f"Missing required columns for FSA prioritization: {sorted(list(needed_fsa - set(df.columns)))}")
    else:
        # Ensure numeric flags
        for c in ["Symptom_Breath_Shortness","Symptom_Fever","Symptom_Cough",
                  "Known_Exposure","Has_Preexisting_Condition","Is_Probable_Case"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        df["Age_Range"] = df["Age_Range"].astype(str)
        df["Postal_District"] = df["Postal_District"].astype(str)

        # 1) Triage score logic
        def calculate_triage_score(row):
            score = 0
            if row["Symptom_Breath_Shortness"] == 1: score += 3
            if row["Symptom_Fever"] == 1: score += 2
            if row["Symptom_Cough"] == 1: score += 1
            if row["Known_Exposure"] == 1: score += 2
            if row["Has_Preexisting_Condition"] == 1: score += 2
            age_str = str(row["Age_Range"])
            if (">60" in age_str) or (">65" in age_str): score += 3
            elif "45-64" in age_str: score += 1
            return score

        temp = df.copy()
        temp["Final_Triage_Score"] = temp.apply(calculate_triage_score, axis=1)

        # 2) Mortality vulnerability flag
        temp["Mortality_Vulnerability"] = (
            (temp["Is_Probable_Case"] == 1) &
            (temp["Age_Range"].str.contains(">60|>65", na=False)) &
            (temp["Has_Preexisting_Condition"] == 1)
        ).astype(int)

        # Drop NR / empty postal districts
        temp = temp[~temp["Postal_District"].str.lower().isin(["nr", "nan", "none", "unknown", ""])]

        if temp.empty:
            st.info("No valid Postal_District records available for FSA analysis.")
        else:
            # 3) Aggregate by Postal District
            dashboard_data = temp.groupby("Postal_District").agg(
                Transmission_Density=("Is_Probable_Case", "mean"),
                Mortality_Risk_Count=("Mortality_Vulnerability", "sum"),
                Total_Reports=("Postal_District", "count"),
                Avg_Triage_Score=("Final_Triage_Score", "mean")
            ).reset_index()

            # 4) Risk indices
            dashboard_data["Medical_Risk_Index"] = dashboard_data["Mortality_Risk_Count"] / dashboard_data["Total_Reports"].replace(0, np.nan)
            dashboard_data["Social_Risk_Index"] = dashboard_data["Transmission_Density"]

            # 5) Recommend support based on median Medical_Risk_Index
            median_med = dashboard_data["Medical_Risk_Index"].median(skipna=True)

            dashboard_data["Recommended_Support"] = np.where(
                dashboard_data["Medical_Risk_Index"] > median_med,
                "Medical Support",
                "Financial Support"
            )

            # 6) Filter stability
            viz_data = dashboard_data[dashboard_data["Total_Reports"] >= 5].copy()

            if viz_data.empty:
                st.info("Not enough stable FSA records (need Total_Reports >= 5).")
            else:
                # Top 10 charts
                top_medical = viz_data[viz_data["Recommended_Support"] == "Medical Support"] \
                    .nlargest(10, "Medical_Risk_Index")

                top_financial = viz_data[viz_data["Recommended_Support"] == "Financial Support"] \
                    .nlargest(10, "Social_Risk_Index")

                fig0, axes = plt.subplots(2, 1, figsize=(7.5, 6.8))

                # Medical
                sns.barplot(
                    data=top_medical,
                    x="Postal_District",
                    y="Medical_Risk_Index",
                    ax=axes[0],
                    palette="Purples_r",
                    edgecolor="black",
                    linewidth=1
                )
                axes[0].set_title("Top 10 FSAs Requiring MEDICAL Support (Clinical Risk Index)",
                                  fontsize=11, fontweight="bold")
                axes[0].set_ylabel("Medical Risk Index")
                axes[0].set_xlabel("")
                axes[0].tick_params(axis="x", rotation=0)

                # Financial
                sns.barplot(
                    data=top_financial,
                    x="Postal_District",
                    y="Social_Risk_Index",
                    ax=axes[1],
                    palette="YlOrBr",
                    edgecolor="black",
                    linewidth=1
                )
                axes[1].set_title("Top 10 FSAs Requiring FINANCIAL Support (Transmission Density)",
                                  fontsize=11, fontweight="bold")
                axes[1].set_ylabel("Transmission Density (Social Risk)")
                axes[1].set_xlabel("Postal District (FSA)")
                axes[1].tick_params(axis="x", rotation=0)

                plt.tight_layout()
                st.pyplot(fig0)
                plt.close(fig0)
    st.divider()
    # ============================================================
    # Ethnicity Priority (SECOND)
    # ============================================================
    st.subheader("Priority Ethnic Groups (Probable Risk × Support Needs)")

    needed_eth = {"Ethnicity", "Is_Probable_Case", "Needs"}
    if not needed_eth.issubset(df.columns):
        st.warning(f"Missing required columns for ethnicity priority: {sorted(list(needed_eth - set(df.columns)))}")
    else:
        df_eth = df.copy()

        # ---- Clean Ethnicity ----
        df_eth["Ethnicity"] = (
            df_eth["Ethnicity"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df_eth = df_eth[~df_eth["Ethnicity"].isin(["na", "nr", "nan", "none", "unknown", ""])].copy()
        df_eth["Ethnicity"] = df_eth["Ethnicity"].str.title()

        # ---- Clean probable ----
        df_eth["Is_Probable_Case"] = pd.to_numeric(df_eth["Is_Probable_Case"], errors="coerce")
        df_eth = df_eth[df_eth["Is_Probable_Case"].isin([0, 1])].copy()

        # ---- Support needs flag ----
        needs_text = df_eth["Needs"].astype(str).str.strip().str.lower()
        df_eth["Has_Support_Needs"] = np.where(
            needs_text.isin(["nr", "none", "noneoftheabove", "", "nan", "unknown"]),
            0,
            1
        ).astype(int)

        # ---- Ethnicity summary ----
        eth_summary = (
            df_eth.groupby("Ethnicity")
            .agg(
                Respondents_n=("Ethnicity", "size"),
                Probable_Rate=("Is_Probable_Case", lambda s: s.mean() * 100),
                SupportNeeds_Rate=("Has_Support_Needs", lambda s: s.mean() * 100)
            )
            .reset_index()
        )

        MIN_N = 30
        eth_summary = eth_summary[eth_summary["Respondents_n"] >= MIN_N].copy()

        if eth_summary.empty:
            st.info(f"No ethnicity groups meet the minimum sample size (n ≥ {MIN_N}).")
        else:
            eth_summary["Priority_Score"] = (
                (eth_summary["Probable_Rate"] / 100) *
                (eth_summary["SupportNeeds_Rate"] / 100) * 100
            )

            eth_summary["Probable_Rate"] = eth_summary["Probable_Rate"].round(2)
            eth_summary["SupportNeeds_Rate"] = eth_summary["SupportNeeds_Rate"].round(2)
            eth_summary["Priority_Score"] = eth_summary["Priority_Score"].round(2)

            eth_summary = eth_summary.sort_values("Priority_Score", ascending=False)

            # ---- Top 10 table (instead of print) ----
            st.markdown("**Top 10 Priority Ethnic Groups** (n ≥ 30)")
            st.dataframe(
                eth_summary.head(10)[["Ethnicity", "Respondents_n", "Probable_Rate", "SupportNeeds_Rate", "Priority_Score"]],
                use_container_width=True
            )

            # ---- Heatmap ----
            heatmap_data = eth_summary[["Priority_Score"]].to_numpy()

            figE, axE = plt.subplots(figsize=(5.5, 6))
            im = axE.imshow(heatmap_data, aspect="auto", cmap="Reds")

            cbar = figE.colorbar(im, ax=axE)
            cbar.set_label("Priority Score (Probable Risk × Support Needs)")

            axE.set_yticks(range(len(eth_summary)))
            axE.set_yticklabels(eth_summary["Ethnicity"].astype(str))

            axE.set_xticks([0])
            axE.set_xticklabels(["Priority"])

            axE.set_title(
                "Ethnic Groups Prioritized for Public Health Resources\n"
                "(Combined Probable COVID Risk and Support Needs)",
                fontsize=11,
                fontweight="bold"
            )

            plt.tight_layout()
            st.pyplot(figE)
            plt.close(figE)

    st.divider()


    # ============================================================
    #  Mental Health Impact by Month (SECOND)
    # ============================================================
    st.subheader("Psychological Shift: Mental Health Impact by Month")

    needed_cols = {"Mental_Health_Status", "Month"}
    if not needed_cols.issubset(df.columns):
        st.warning(f"Missing required columns: {sorted(list(needed_cols - set(df.columns)))}")
    else:
        df["MH_Status"] = (
            df["Mental_Health_Status"].astype(str).str.lower().str.strip()
        )

        df_filtered = df[
            ~df["MH_Status"].isin(["nr", "nan", "none", "unknown", ""])
        ].copy()

        month_order = ["March", "April", "May", "June", "July"]
        df_filtered["Month"] = pd.Categorical(df_filtered["Month"], categories=month_order, ordered=True)
        df_filtered = df_filtered[df_filtered["Month"].notna()]

        mh_dist = pd.crosstab(df_filtered["Month"], df_filtered["MH_Status"], normalize="index") * 100

        if mh_dist.empty:
            st.info("No valid mental health records available after filtering.")
        else:
            colors = {"negative": "#C62828", "no impact": "#455A64", "positive": "#2E7D32"}

            fig1, ax1 = plt.subplots(figsize=(6.5, 4))
            mh_dist.plot(
                kind="bar",
                stacked=True,
                ax=ax1,
                color=[colors.get(x, "#ddd") for x in mh_dist.columns],
                edgecolor="black",
                linewidth=1
            )

            ax1.set_ylabel("Percentage of Population (%)")
            ax1.set_xlabel("Quarantine Period (2020)")
            ax1.tick_params(axis="x", rotation=0)

            ax1.legend(
                title="Impact Status",
                labels=[c.capitalize() for c in mh_dist.columns],
                bbox_to_anchor=(1.02, 1),
                loc="upper left"
            )

            for n, month in enumerate(mh_dist.index.values):
                cumulative = 0
                for status in mh_dist.columns:
                    proportion = float(mh_dist.loc[month, status])
                    if proportion > 5:
                        ax1.text(
                            x=n,
                            y=cumulative + proportion / 2,
                            s=f"{proportion:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="white"
                        )
                    cumulative += proportion

           # plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

    st.divider()

    # ============================================================
    # Quarantine Compliance by Age (THIRD)
    # ============================================================
    st.subheader("Quarantine Compliance Among Probable COVID-19 Cases")

    needed2 = {"Is_Probable_Case", "Quarantine", "Age_Range"}
    if not needed2.issubset(df.columns):
        st.warning(f"Missing columns: {sorted(list(needed2 - set(df.columns)))}")
    else:
        df_temp = df.copy()
        df_temp["Is_Probable_Case"] = pd.to_numeric(df_temp["Is_Probable_Case"], errors="coerce")
        df_temp["Quarantine"] = pd.to_numeric(df_temp["Quarantine"], errors="coerce")

        df_probable = df_temp[df_temp["Is_Probable_Case"] == 1].dropna(subset=["Quarantine", "Age_Range"])

        if df_probable.empty:
            st.info("No probable cases with valid quarantine + age data available.")
        else:
            qmax = df_probable["Quarantine"].max(skipna=True)
            scale = 100.0 if (qmax is not None and qmax <= 1.0) else 1.0

            summary = (
                df_probable.groupby("Age_Range")["Quarantine"]
                .mean().mul(scale).reset_index()
            )

            summary.rename(columns={"Quarantine": "Compliance_Rate"}, inplace=True)
            summary["Non_Compliance_Rate"] = 100 - summary["Compliance_Rate"]

            x = np.arange(len(summary))

            fig2, ax2 = plt.subplots(figsize=(6.5, 4))

            ax2.bar(
                x,
                summary["Compliance_Rate"],
                color="#F1C40F",
                width=0.6,
                label="Quarantine Compliance (%)",
                edgecolor="black",
                linewidth=1
            )

            ax2.plot(
                x,
                summary["Non_Compliance_Rate"],
                color="#800000",
                marker="o",
                linewidth=3,
                label="Non-Compliance (%)"
            )

            for i in range(len(x)):
                ax2.text(
                    x[i],
                    float(summary["Compliance_Rate"].iloc[i]) + 1,
                    f"{summary['Compliance_Rate'].iloc[i]:.1f}%",
                    ha="center",
                    fontweight="bold",
                    fontsize=9
                )

            ax2.set_xticks(x)
            ax2.set_xticklabels(summary["Age_Range"], rotation=0)
            ax2.set_ylabel("Percentage of Probable COVID-19 Cases")
            ax2.set_xlabel("Age Group")
         #  ax2.set_title(
         #       "Quarantine Compliance Among Probable COVID-19 Cases\nIdentifying Age Groups for Targeted Follow-Up",
         #       fontweight="bold"
         #   )
            ax2.set_ylim(0, 100)

            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=2,
                frameon=False
            )

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)


# ============================================================
# TAB: CORRELATION
# ============================================================
with tab_corr:
    corr_cols = [c for c in [
        "Is_Probable_Case",
        "Symptom_Fever",
        "Symptom_Cough",
        "Symptom_Breath_Shortness",
        "Known_Exposure",
        "Has_Preexisting_Condition",
        "Recent_Traveler",
        "COVID_Tested"
    ] if c in df.columns]

    if len(corr_cols) >= 2:
        temp = df[corr_cols].copy()
        for c in corr_cols:
            temp[c] = pd.to_numeric(temp[c], errors="coerce").fillna(0)

        corr = temp.corr()

        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        st_plot(fig)
    else:
        st.info("Not enough columns available to compute correlation heatmap.")