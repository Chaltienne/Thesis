import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

def robust_quantile_mapping(
    obs: pd.DataFrame,
    model: pd.DataFrame,
    time_col: str = "time",
    precip_col: str = "precip",
    threshold: float = 0.1,
    n_quantiles: int = 1000,
    season_group: bool = True
) -> (pd.DataFrame, dict):
    """
    Retourne:
    - DataFrame corrigé
    - Dictionnaire correcteur avec modèles EQM, seuil, et paramètres
    """
    
    df_model = model.copy()
    df_obs = obs.copy()
    corrector = {
        "threshold": threshold,
        "season_group": season_group,
        "models": {}
    }

    # 1. Définition des saisons
    if season_group:
        season_mapping = {
            12: "DJF", 1: "DJF", 2: "DJF",
            3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA",
            9: "SON", 10: "SON", 11: "SON"
        }
        df_model["season"] = df_model[time_col].dt.month.map(season_mapping)
        df_obs["season"] = df_obs[time_col].dt.month.map(season_mapping)

    # 2. Fonction de correction par groupe
    def process_group(group_model, group_name="annual"):
        if season_group:
            group_obs = df_obs[df_obs["season"] == group_name]
        else:
            group_obs = df_obs

        # Jours humides observés
        obs_wet = group_obs[group_obs[precip_col] >= threshold][precip_col].values.reshape(-1, 1)
        
        # Initialisation à zéro
        corrected = group_model[precip_col].values.copy()
        corrected[:] = 0.0

        if len(obs_wet) > 0:
            # Entraînement EQM si données valides
            model_wet_mask = group_model[precip_col] >= threshold
            model_wet = group_model[model_wet_mask][precip_col].values.reshape(-1, 1)
            
            if len(model_wet) > 0:
                eqm = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")
                eqm.fit(obs_wet)
                corrected_wet = eqm.inverse_transform(eqm.transform(model_wet))
                corrected[model_wet_mask] = corrected_wet.flatten()
                corrector["models"][group_name] = eqm

        return corrected

    # 3. Application par saison ou globalement
    if season_group:
        grouped = df_model.groupby("season", group_keys=False)
        df_model["precip_corrected"] = grouped.apply(
            lambda g: process_group(g, g.name)
        ).values
    else:
        df_model["precip_corrected"] = process_group(df_model)

    # 4. Formatage final
    df_model["precip_corrected"] = np.clip(df_model["precip_corrected"], 0, None)
    
    return (
        df_model[[time_col, "precip_corrected"]].rename(columns={"precip_corrected": "precip"}),
        corrector
    )

# -----------------------------------------------------------
# Exemple d'utilisation avec récupération du correcteur
# -----------------------------------------------------------
if __name__ == "__main__":
    # Génération de données
    dates = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    np.random.seed(42)
    obs_data = np.random.gamma(shape=0.5, scale=0.5, size=len(dates))
    model_data = obs_data * 1.8 + 0.2  # Biais artificiel
    
    df_obs = pd.DataFrame({"time": dates, "precip": obs_data})
    df_model = pd.DataFrame({"time": dates, "precip": model_data})

    # Application avec récupération du correcteur
    corrected_df, eqm_corrector = robust_quantile_mapping(
        obs=df_obs,
        model=df_model,
        threshold=0.1,
        season_group=True
    )

    print("Corrector Contents:", eqm_corrector.keys())
    print("Saved Models:", eqm_corrector["models"].keys())