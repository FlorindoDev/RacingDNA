"""
Dataset Normalization Script - Per-Corner Normalization

Normalizza i dati per ogni (GrandPrix, CornerID) in modo che il modello
impari gli stili di guida relativi invece delle differenze assolute tra curve.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "data/dataset/dataset_curves.csv"
OUTPUT_FILE = "data/dataset/normalized_dataset_per_corner.npz"
PADDING_VALUE = -1000.0

# Colonne da rimuovere dalla normalizzazione (posizione, tempo, distanza)
COLUMNS_TO_REMOVE_PATTERNS = [
    r"^x_\d+$",      # Posizione X
    r"^y_\d+$",      # Posizione Y  
    r"^z_\d+$",      # Posizione Z
    r"^time_\d+$",   # Tempo
    r"^distance_\d+$"  # Distanza
]

# Colonne metadata da preservare per il raggruppamento
METADATA_COLUMNS = ["GrandPrix", "Session", "Driver", "Lap", "CornerID"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def identify_column_groups(columns: list) -> Tuple[Dict[str, list], list]:
    """
    Identifica gruppi di colonne (es: speed_0, speed_1, ...) e colonne singole.
    
    Returns:
        Tuple of (grouped_cols dict, single_cols list)
    """
    grouped_cols = {}
    single_cols = []
    
    for col in columns:
        match = re.match(r"^(.*)_\d+$", col)
        if match:
            prefix = match.group(1)
            if prefix not in grouped_cols:
                grouped_cols[prefix] = []
            grouped_cols[prefix].append(col)
        else:
            single_cols.append(col)
    
    return grouped_cols, single_cols


def should_remove_column(col: str) -> bool:
    """Check if column should be removed based on patterns."""
    for pattern in COLUMNS_TO_REMOVE_PATTERNS:
        if re.match(pattern, col):
            return True
    return False


def compute_group_stats(
    df: pd.DataFrame, 
    telemetry_cols: list
) -> pd.DataFrame:
    """
    Calcola mean e std per ogni (GrandPrix, CornerID).
    
    Returns:
        DataFrame con multi-index (GrandPrix, CornerID) e colonne mean_*, std_*
    """
    # Sostituisci padding con NaN per il calcolo delle statistiche
    df_for_stats = df.copy()
    df_for_stats[telemetry_cols] = df_for_stats[telemetry_cols].replace(PADDING_VALUE, np.nan)
    
    # Raggruppa per (GrandPrix, CornerID)
    grouped = df_for_stats.groupby(["GrandPrix", "CornerID"])
    
    # Calcola statistiche per ogni gruppo
    stats_list = []
    
    for (gp, corner_id), group in grouped:
        group_stats = {"GrandPrix": gp, "CornerID": corner_id}
        
        for col in telemetry_cols:
            col_data = group[col].values.flatten()
            col_data = col_data[~np.isnan(col_data)]  # Rimuovi NaN
            
            if len(col_data) > 0:
                group_stats[f"mean_{col}"] = np.mean(col_data)
                group_stats[f"std_{col}"] = np.std(col_data)
                # Safety: std = 0 -> 1
                if group_stats[f"std_{col}"] == 0:
                    group_stats[f"std_{col}"] = 1.0
            else:
                group_stats[f"mean_{col}"] = 0.0
                group_stats[f"std_{col}"] = 1.0
        
        stats_list.append(group_stats)
    
    return pd.DataFrame(stats_list)


def normalize_per_corner(
    df: pd.DataFrame, 
    stats_df: pd.DataFrame, 
    telemetry_cols: list
) -> pd.DataFrame:
    """
    Normalizza ogni riga usando le statistiche del suo (GrandPrix, CornerID).
    """
    df_normalized = df.copy()
    
    # Crea lookup per le statistiche
    stats_lookup = stats_df.set_index(["GrandPrix", "CornerID"])
    
    # Normalizza gruppo per gruppo
    for (gp, corner_id), indices in df.groupby(["GrandPrix", "CornerID"]).groups.items():
        if (gp, corner_id) in stats_lookup.index:
            row_stats = stats_lookup.loc[(gp, corner_id)]
            
            for col in telemetry_cols:
                mean_val = row_stats[f"mean_{col}"]
                std_val = row_stats[f"std_{col}"]
                
                # Normalizza solo valori non-padding
                mask = df_normalized.loc[indices, col] != PADDING_VALUE
                df_normalized.loc[indices[mask], col] = (
                    (df_normalized.loc[indices[mask], col] - mean_val) / std_val
                )
    
    return df_normalized


# ============================================================================
# MAIN PROCESSING
# ============================================================================
def main():
    print("=" * 60)
    print("Per-Corner Normalization")
    print("=" * 60)
    
    # 1. Carica dataset
    print("\n[1/6] Caricamento dataset...")
    df = pd.read_csv(INPUT_FILE, sep=",", encoding="utf-8", decimal=".")
    print(f"      Shape originale: {df.shape}")
    print(f"      Colonne metadata: {[c for c in METADATA_COLUMNS if c in df.columns]}")
    
    # 2. Rimuovi colonne non necessarie
    print("\n[2/6] Rimozione colonne (x, y, z, time, distance)...")
    cols_to_remove = [c for c in df.columns if should_remove_column(c)]
    # Rimuovi anche Stint che non serve
    if "Stint" in df.columns:
        cols_to_remove.append("Stint")
    
    df = df.drop(columns=cols_to_remove, errors='ignore')
    print(f"      Rimosse {len(cols_to_remove)} colonne")
    print(f"      Shape dopo rimozione: {df.shape}")
    
    # 3. One-Hot Encoding per Compound
    print("\n[3/6] One-Hot Encoding per Compound...")
    if "Compound" in df.columns:
        target_categories = ['HARD', 'INTERMEDIATE', 'MEDIUM', 'SOFT']
        compound_dummies = pd.get_dummies(df['Compound'], prefix='Compound', dtype=float)
        
        # Aggiungi colonne mancanti
        for cat in target_categories:
            col = f"Compound_{cat}"
            if col not in compound_dummies.columns:
                compound_dummies[col] = 0.0
        
        compound_dummies = compound_dummies[[f"Compound_{cat}" for cat in target_categories]]
        df = pd.concat([df, compound_dummies], axis=1)
        df = df.drop('Compound', axis=1)
        print(f"      Aggiunte colonne: {list(compound_dummies.columns)}")
    
    # 4. Identifica colonne telemetria
    print("\n[4/6] Identificazione colonne telemetria...")
    non_telemetry = METADATA_COLUMNS + ["Session"]
    compound_cols = [c for c in df.columns if c.startswith("Compound_")]
    telemetry_cols = [
        c for c in df.columns 
        if c not in non_telemetry 
        and c not in compound_cols
        and c not in ["TireLife"]  # TireLife è scalare, normalizzeremo separatamente
    ]
    print(f"      Trovate {len(telemetry_cols)} colonne telemetria")
    
    # 5. Calcola statistiche per-corner
    print("\n[5/6] Calcolo statistiche per (GrandPrix, CornerID)...")
    stats_df = compute_group_stats(df, telemetry_cols)
    n_groups = len(stats_df)
    print(f"      Gruppi unici: {n_groups}")
    
    # 6. Normalizza
    print("\n[6/6] Normalizzazione per-corner...")
    df_normalized = normalize_per_corner(df, stats_df, telemetry_cols)
    
    # Normalizza TireLife globalmente
    if "TireLife" in df_normalized.columns:
        life_mean = df["TireLife"].mean()
        life_std = df["TireLife"].std()
        if life_std == 0:
            life_std = 1.0
        df_normalized["TireLife"] = (df["TireLife"] - life_mean) / life_std
    
    # 7. Crea mask
    print("\n[7/7] Creazione mask e salvataggio...")
    
    # Rimuovi colonne metadata per il modello
    cols_for_model = [c for c in df_normalized.columns if c not in METADATA_COLUMNS and c != "Session"]
    df_final = df_normalized[cols_for_model]
    
    # Sostituisci padding con 0 dopo normalizzazione
    mask = (df_final != PADDING_VALUE).astype(float)
    df_final = df_final.replace(PADDING_VALUE, 0.0)
    
    # CRITICAL: Replace any remaining NaN/Inf with 0.0
    nan_count = df_final.isna().sum().sum()
    inf_count = np.isinf(df_final.values).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"      Warning: Found {nan_count} NaN and {inf_count} Inf values, replacing with 0.0")
        df_final = df_final.fillna(0.0)
        df_final = df_final.replace([np.inf, -np.inf], 0.0)
    
    # Prepara mean/std per denormalizzazione
    # Usiamo la media delle statistiche (approssimazione)
    grouped_cols, single_cols = identify_column_groups(telemetry_cols)
    mean_dict = {}
    std_dict = {}
    
    for col in df_final.columns:
        if col.startswith("Compound_"):
            mean_dict[col] = 0.0
            std_dict[col] = 1.0
        elif col == "TireLife":
            mean_dict[col] = life_mean if "TireLife" in df.columns else 0.0
            std_dict[col] = life_std if "TireLife" in df.columns else 1.0
        else:
            # Media delle statistiche per-corner (approssimazione)
            mean_dict[col] = stats_df[f"mean_{col}"].mean() if f"mean_{col}" in stats_df.columns else 0.0
            std_dict[col] = stats_df[f"std_{col}"].mean() if f"std_{col}" in stats_df.columns else 1.0
    
    mean = pd.Series(mean_dict)[df_final.columns]
    std = pd.Series(std_dict)[df_final.columns]
    
    # Salva
    np.savez(
        OUTPUT_FILE,
        data=df_final.values.astype(np.float32),
        mask=mask.values.astype(np.float32),
        mean=mean.values.astype(np.float32),
        std=std.values.astype(np.float32),
        columns=df_final.columns.values,
        # Salva anche le statistiche per-corner per uso futuro
        corner_stats=stats_df.to_dict('records')
    )
    
    print(f"\n{'=' * 60}")
    print(f"Salvato: {OUTPUT_FILE}")
    print(f"Shape: {df_final.shape}")
    print(f"Colonne: {len(df_final.columns)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
