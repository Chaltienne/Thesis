import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from typing import Optional

@dataclass
class DQMCorrector:
    """Conteneur pour les paramètres de correction DQM"""
    o_mean: float
    p_mean: float
    x_quantiles: np.ndarray
    y_quantiles: np.ndarray
    trend_model: Optional[LinearRegression]
    precip: bool
    pr_threshold: float
    detrend: bool
    n_quantiles: int
    trend_coef: Optional[float] = None
    trend_intercept: Optional[float] = None

def dqm(
    o: np.ndarray,
    p: np.ndarray, 
    s: np.ndarray,
    precip: bool,
    pr_threshold: float,
    n_quantiles: Optional[int] = None,
    detrend: bool = True
) -> tuple[np.ndarray, DQMCorrector]:
    """
    Version améliorée qui retourne à la fois les données corrigées et le correcteur
    
    Returns:
        tuple: (données corrigées, objet DQMCorrector)
    """
    
    # Copie pour éviter les effets de bord
    o = o.copy()
    p = p.copy()
    s = s.copy()

    # Initialisation du conteneur de correction
    corrector = DQMCorrector(
        o_mean=np.nan,
        p_mean=np.nan,
        x_quantiles=np.array([]),
        y_quantiles=np.array([]),
        trend_model=None,
        precip=precip,
        pr_threshold=pr_threshold,
        detrend=detrend,
        n_quantiles=n_quantiles if n_quantiles else len(p)
    )

    # Gestion des valeurs manquantes
    if np.all(np.isnan(o)) or np.all(np.isnan(p)) or np.all(np.isnan(s)):
        return np.full_like(s, np.nan), corrector

    # Pré-traitement des précipitations
    if precip:
        eps = np.finfo(float).eps
        for arr in [o, p, s]:
            mask = (arr < pr_threshold) & (~np.isnan(arr))
            arr[mask] = np.random.uniform(eps, pr_threshold, size=np.sum(mask))

    # Calcul des moyennes
    corrector.o_mean = float(np.nanmean(o))
    corrector.p_mean = float(np.nanmean(p))

    # Correction initiale
    if precip:
        s = s * (corrector.o_mean / corrector.p_mean)
    else:
        s = s - corrector.p_mean + corrector.o_mean

    # Détrending
    if detrend:
        non_na_idx = ~np.isnan(s)
        X = np.arange(np.sum(non_na_idx)).reshape(-1, 1)
        corrector.trend_model = LinearRegression()
        corrector.trend_model.fit(X, s[non_na_idx])
        corrector.trend_coef = corrector.trend_model.coef_[0]
        corrector.trend_intercept = corrector.trend_model.intercept_
        s_trend = corrector.trend_model.predict(X)
    else:
        s_trend = np.full_like(s, corrector.o_mean)

    # Calcul des quantiles
    n_quantiles = n_quantiles or len(p)
    tau = np.linspace(1/n_quantiles, 1 - 1/n_quantiles, n_quantiles - 1)
    
    if precip:
        corrector.x_quantiles = np.quantile(p/corrector.p_mean, tau, method='linear')
        corrector.y_quantiles = np.quantile(o/corrector.o_mean, tau, method='linear')
    else:
        corrector.x_quantiles = np.quantile(p - corrector.p_mean, tau, method='linear')
        corrector.y_quantiles = np.quantile(o - corrector.o_mean, tau, method='linear')

    # Application de la correction
    yout = _apply_dqm_correction(s, s_trend, corrector)
    
    return yout, corrector

def apply_dqm(s_new: np.ndarray, corrector: DQMCorrector) -> np.ndarray:
    """
    Applique une correction pré-entraînée à de nouvelles données
    
    Args:
        s_new: Nouvelles données simulées à corriger
        corrector: Objet DQMCorrector entraîné
        
    Returns:
        np.ndarray: Données corrigées
    """
    s = s_new.copy()
    
    # Application de la correction de base
    if corrector.precip:
        s = s * (corrector.o_mean / corrector.p_mean)
    else:
        s = s - corrector.p_mean + corrector.o_mean

    # Application du détrenting
    if corrector.detrend and corrector.trend_model:
        non_na_idx = ~np.isnan(s)
        X = np.arange(np.sum(non_na_idx)).reshape(-1, 1)
        s_trend = corrector.trend_model.predict(X)
    else:
        s_trend = np.full_like(s, corrector.o_mean)

    return _apply_dqm_correction(s, s_trend, corrector)

def _apply_dqm_correction(s: np.ndarray, s_trend: np.ndarray, corrector: DQMCorrector) -> np.ndarray:
    """Fonction interne de correction quantile"""
    yout = np.full_like(s, np.nan)
    
    from scipy import __version__ as scipy_version
    from packaging import version
    
    if version.parse(scipy_version) >= version.parse('1.10'):
        fill_val = 'extrapolate'
    else:
        fill_val = (corrector.y_quantiles[0], corrector.y_quantiles[-1])
    
    if corrector.precip:
        # Calcul des ratios
        ratio = s / s_trend
        f = interp1d(corrector.x_quantiles, corrector.y_quantiles, 
                    bounds_error=False, fill_value=fill_val)
        yout = f(ratio) * s_trend
        
        # Gestion des extrapolations
        mask_low = ratio < corrector.x_quantiles[0]
        mask_high = ratio > corrector.x_quantiles[-1]
        
        if np.any(mask_low):
            yout[mask_low] = (corrector.y_quantiles[0] * 
                             (ratio[mask_low]/corrector.x_quantiles[0]) * 
                             s_trend[mask_low])
            
        if np.any(mask_high):
            yout[mask_high] = (corrector.y_quantiles[-1] * 
                              (ratio[mask_high]/corrector.x_quantiles[-1]) * 
                              s_trend[mask_high])
        
        # Post-traitement
        yout[yout < np.sqrt(np.finfo(float).eps)] = 0
        
    else:
        # Version pour variables non précipitation
        anomaly = s - s_trend
        f = interp1d(corrector.x_quantiles, corrector.y_quantiles,
                    bounds_error=False, fill_value=fill_val)
        yout = f(anomaly) + s_trend
        
    return yout


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def validate_dqm_correction(obs: np.ndarray, 
                           raw: np.ndarray, 
                           corrected: np.ndarray,
                           directory: str = "",
                           #corrector: DQMCorrector,
                           variable_name: str = "Precipitations"
                          ):
    """Génère un rapport de validation complet avec figures et métriques"""
    
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"DQM correction validation - {variable_name}", y=0.95)
    
    # 1. Séries temporelles
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)

    if isinstance(obs, type(None)):
      time = np.arange(len(raw))
      _print = False
    else:
      time = np.arange(len(obs))
      _print = True

    if _print:
      ax1.plot(time, obs, 'g-', alpha=0.7, label='Observed')
    ax1.plot(time, raw, 'r-', alpha=0.5, label='Raw')
    ax1.plot(time, corrected, 'b-', alpha=0.7, label='Corrected')
    ax1.set_title("Time Series")
    ax1.legend()
    
    ax1.grid(True)
    
    # 2. Diagramme Quantile-Quantile
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    if isinstance(obs, type(None)):
      qq_plot(raw, corrected, ax2, variable_name)
    else:
      qq_plot(obs, corrected, ax2, variable_name)
    
    # 3. Distribution cumulée
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    plot_cdf(obs, raw, corrected, ax3, variable_name)
    
    # 4. Densité de probabilité
    ax4 = plt.subplot2grid((3, 3), (1, 2))
    plot_pdf(obs, raw, corrected, ax4, variable_name)
    
    # 5. Résidus
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    if isinstance(obs, type(None)):
      plot_residuals(raw, corrected, ax5)
    else:
      plot_residuals(obs, corrected, ax5)
    
    plt.tight_layout()
    plt.savefig(f"{directory}dqm_{variable_name.lower()}.png")
    print(f"{directory}dqm_{variable_name.lower()}.png")
    plt.show()
    plt.close()
    
    # Calcul des métriques statistiques
    if isinstance(obs, type(None)):
      return -1

    metrics = calculate_metrics(obs, raw, corrected)
    
    return metrics
    
def qq_plot(obs, corrected, ax, title):
    """Diagramme Quantile-Quantile avec ligne 1:1"""
    
    q_obs = np.quantile(obs[~np.isnan(obs)], np.linspace(0, 1, 100))
    q_corr = np.quantile(corrected[~np.isnan(corrected)], np.linspace(0, 1, 100))
    
    ax.plot(q_obs, q_corr, 'bo', alpha=0.5)
    ax.plot([q_obs.min(), q_obs.max()], [q_obs.min(), q_obs.max()], 'r--')
    ax.set_title("Q-Q Plot Corrected vs Observed")
    ax.set_xlabel("Observed Quantiles")
    ax.set_ylabel("Corrected Quantiles")
    ax.grid(True)
    
def plot_cdf(obs, raw, corrected, ax, title):
    """Fonction de distribution cumulative"""

    if isinstance(obs, type(None)):
      for data, label, color in zip([raw, corrected], 
                                ['Raw', 'Corrected'], 
                                ['r', 'b']):
        sorted_data = np.sort(data[~np.isnan(data)])
        cdf = np.arange(1, len(sorted_data)+1)/len(sorted_data)
        ax.plot(sorted_data, cdf, color, alpha=0.7, label=label)

    else:
      for data, label, color in zip([obs, raw, corrected], 
                                  ['Observed', 'Raw', 'Corrected'], 
                                  ['g', 'r', 'b']):
          sorted_data = np.sort(data[~np.isnan(data)])
          cdf = np.arange(1, len(sorted_data)+1)/len(sorted_data)
          ax.plot(sorted_data, cdf, color, alpha=0.7, label=label)
    
    ax.set_title("Cumulative Distribution Function (CDF)")
    ax.legend()
    ax.grid(True)

def plot_pdf(obs, raw, corrected, ax, title):
    """Densité de probabilité (histogramme normalisé)"""
      
    if isinstance(obs, type(None)):
      bins = np.linspace(min(raw.min(), corrected.min()),
                      max(raw.max(), corrected.max()), 50)

    else:
      bins = np.linspace(min(obs.min(), raw.min(), corrected.min()),
                        max(obs.max(), raw.max(), corrected.max()), 50)
      
      ax.hist(obs, bins=bins, density=True, alpha=0.5, color='g', label='Observed')

    ax.hist(raw, bins=bins, density=True, alpha=0.5, color='r', label='Raw')
    ax.hist(corrected, bins=bins, density=True, alpha=0.5, color='b', label='Corrected')
    ax.set_title("Probability Density")
    ax.legend()
    ax.grid(True)

def plot_residuals(obs, corrected, ax):
    """Résidus et autocorrélation"""
    residuals = corrected - obs
    valid = ~np.isnan(residuals)
    
    ax.plot(np.arange(len(residuals))[valid], residuals[valid], 'k.', alpha=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title("Residual (Corrected - Observations)")
    ax.set_ylabel("Error")
    ax.grid(True)

def calculate_metrics(obs, raw, corrected):
    """Calcule les métriques de performance"""
    mask = ~np.isnan(obs) & ~np.isnan(corrected)
    
    return { 'RMSE Brut': np.sqrt(np.mean((raw[mask] - obs[mask])**2)),
        'RMSE Corrige': np.sqrt(np.mean((corrected[mask] - obs[mask])**2)),
        'Biais Brut': np.mean(raw[mask] - obs[mask]),
        'Biais Corrige': np.mean(corrected[mask] - obs[mask]),
        'Corrélation Brut': np.corrcoef(obs[mask], raw[mask])[0,1],
        'Corrélation Corrige': np.corrcoef(obs[mask], corrected[mask])[0,1],
        'KS-test p-value': stats.ks_2samp(obs[mask], corrected[mask])[1]
    }
    
def calculate_extended_metrics(obs, raw, corrected):
    """Calcule des métriques avancées pour les études climatiques"""
    mask = ~np.isnan(obs) & ~np.isnan(corrected)
    
    return {
        # Métriques standard
        'RMSE Brut': np.sqrt(np.mean((raw[mask] - obs[mask])**2)),
        'RMSE Corrige': np.sqrt(np.mean((corrected[mask] - obs[mask])**2)),
        'Biais Brut': np.mean(raw[mask] - obs[mask]),
        'Biais Corrige': np.mean(corrected[mask] - obs[mask]),
        
        # Métriques climatiques spécifiques
        'R95p Brut': np.percentile(raw[mask], 95) - np.percentile(obs[mask], 95),
        'R95p Corrige': np.percentile(corrected[mask], 95) - np.percentile(obs[mask], 95),
        'Dry Days Brut': np.sum(raw[mask] < 0.1) - np.sum(obs[mask] < 0.1),
        'Dry Days Corrige': np.sum(corrected[mask] < 0.1) - np.sum(obs[mask] < 0.1),
        'KS-test p-value': stats.ks_2samp(obs[mask], corrected[mask])[1]
    }

"""
def plot_climate_validation(obs, raw, corrected, title):
    #Génère les figures de validation spécifiques au climat
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Séries temporelles mensuelles
    monthly_obs = obs.resample('M', on='time').mean()
    monthly_corr = corrected.resample('M', on='time').mean()
    axs[0].plot(monthly_obs, 'g-', label='Observations')
    axs[0].plot(monthly_corr, 'b-', label='Corrigé')
    axs[0].set_title(f"{title} - Saisonnalité Mensuelle")
    
    # Distribution des extrêmes
    axs[1].hist([obs['precip'], corrected['precip']], bins=50, density=True, 
               label=['Observations', 'Corrigé'], alpha=0.6)
    axs[1].set_title("Distribution des Précipitations Extrêmes")
    
    # Évolution du biais décennal
    bias = corrected.groupby(pd.Grouper(key='time', freq='Y')).mean() - \
           obs.groupby(pd.Grouper(key='time', freq='Y')).mean()
    axs[2].bar(bias.index.year, bias.values)
    axs[2].set_title("Biais Annual Résiduel")
    
    plt.tight_layout()
    plt.savefig(f"validation_dqm_climatique.png")
    return fig
  """

def plot_climate_validation(obs, raw, corrected, title, directory, aggr="M"):
    """
    Génère les figures de validation spécifiques au climat.
    Version corrigée avec gestion d'erreur améliorée.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # --- Préparation des données ---
    for df in [obs, raw, corrected]:
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)
        df.reset_index(drop=True, inplace=True)

    # --- Plot 1: Séries temporelles mensuelles ---
    for name, df, color in zip(['Observed', 'Raw Data', 'Corrected data'], 
                              [obs, raw, corrected], ['g', 'r', 'b']):
        monthly = df.resample('ME', on='time')['precip'].mean()
        axs[0].plot(monthly.index, monthly.values, color+'-', label=name)
    
    axs[0].set_title(f"{title} - Monthly Mean")
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot 2: Distribution des précipitations ---
    axs[1].hist(
        [obs['precip'], raw['precip'], corrected['precip']],
        bins=50, density=True,
        label=['Observed', 'Raw', 'Corrected'],
        alpha=0.7, histtype='stepfilled'
    )
    axs[1].set_title("Precipitation Distribution")
    axs[1].legend()
    axs[1].grid(True)

    # --- Plot 3: Biais annuel résiduel ---
    try:
        # Calcul du biais annuel
        bias = (
            corrected.groupby(pd.Grouper(key='time', freq='YE'))['precip'].mean() 
            - obs.groupby(pd.Grouper(key='time', freq='YE'))['precip'].mean()
        )

        if not bias.empty:
            # Conversion et nettoyage des données
            years = bias.index.year.to_numpy(dtype='int64')
            bias_values = np.nan_to_num(bias.values.astype('float64'), nan=0.0)
            
            # Vérification cohérence des dimensions
            if len(years) == len(bias_values):
                axs[2].bar(
                    years, 
                    bias_values, 
                    color='purple',
                    width=0.8,
                    edgecolor='k', 
                    linewidth=0.5  # Scalaire explicite
                )
              
                  
                axs[2].set_title(f"{title} - Residual Annual Bias")
                axs[2].axhline(0, color='r', ls='--', lw=0.8)
                axs[2].grid(axis='y')
            else:
                print("Erreur: Mismatch années/valeurs")
        else:
            print("Avertissement: Aucun biais calculable")

    except Exception as e:
        print(f"Erreur lors du tracé du biais: {str(e)}")

    plt.tight_layout()
    plt.savefig(directory + "validation_climatique.png")
    #print(f"{directory}dqm_{title}.png")
    plt.show()
    
    return fig


def generate_contextual_report(metrics_train, metrics_test, region):
    """Génère un rapport adapté au contexte local"""
    report = {
        'region': region,
        'performance_gain': {
            'rmse': 1 - metrics_test['RMSE Corrigé']/metrics_test['RMSE Brut'],
            'bias': 1 - abs(metrics_test['Biais Corrigé'])/abs(metrics_test['Biais Brut'])
        },
        'extreme_events': {
            'r95p_improvement': metrics_test['R95p Brut'] - metrics_test['R95p Corrigé'],
            'dry_days_improvement': metrics_test['Dry Days Brut'] - metrics_test['Dry Days Corrigé']
        },
        'operational_advice': {
            'recalibration_frequency': '2 ans' if metrics_test['KS-test p-value'] < 0.05 else '5 ans',
            'extremes_handling': 'Acceptable' if abs(metrics_test['R95p Corrigé']) < 0.2 else 'Revoir'
        }
    }
    return report


        
# Exemple d'utilisation avec données synthétiques
if __name__ == "__main__":
    # Configuration commune
    np.random.seed(42)
    n = 1000
    split_idx = 800  # 80% entraînement, 20% test

    # Cas 1: Précipitations (distribution gamma)
    print("\n" + "="*50)
    print("Validation pour les précipitations")
    print("="*50)
    
    # Génération des données
    obs_full = np.random.gamma(2, scale=2, size=n)
    raw_full = obs_full * 1.5 + np.random.normal(0, 0.5, n)
    raw_full[raw_full < 0] = 0  # Force les précipitations positives
    
    # Séparation entraînement/test
    obs_train, obs_test = obs_full[:split_idx], obs_full[split_idx:]
    p_train, s_test = raw_full[:split_idx], raw_full[split_idx:]
    
    # Entraînement du correcteur sur la période historique
    corrected_train, corrector = dqm(
        o=obs_train,
        p=p_train,
        s=p_train,  # s=p_train pour la calibration
        precip=True,
        pr_threshold=0.1
    )
    
    # Application aux projections futures
    corrected_test = apply_dqm(s_test, corrector)
    
    # Validation sur la période de test
    metrics_test = validate_dqm_correction(
        obs=obs_test, 
        raw=s_test, 
        corrected=corrected_test,
        #corrector=corrector,
        variable_name="Précipitations (mm/jour)"
    )
    
    print("\nPerformance sur l'ensemble de test:")
    for k, v in metrics_test.items():
        print(f"{k:20}: {v:.4f}")

    # Cas 2: Température (distribution normale)
    print("\n" + "="*50)
    print("Validation pour la température")
    print("="*50)
    
    # Génération des données
    obs_temp_full = np.random.normal(15, 3, n)
    raw_temp_full = obs_temp_full * 1.2 + 2 + np.random.normal(0, 1, n)
    
    # Séparation entraînement/test
    obs_temp_train, obs_temp_test = obs_temp_full[:split_idx], obs_temp_full[split_idx:]
    p_temp_train, s_temp_test = raw_temp_full[:split_idx], raw_temp_full[split_idx:]
    
    # Entraînement et application
    corrected_temp_train, corrector_temp = dqm(
        o=obs_temp_train,
        p=p_temp_train,
        s=p_temp_train,
        precip=False,
        pr_threshold=0
    )
    corrected_temp_test = apply_dqm(s_temp_test, corrector_temp)
    
    # Validation
    metrics_temp_test = validate_dqm_correction(
        obs=obs_temp_test,
        raw=s_temp_test,
        corrected=corrected_temp_test,
        #corrector=corrector_temp,
        variable_name="Température (°C)"
    )
    
    print("\nPerformance sur l'ensemble de test:")
    for k, v in metrics_temp_test.items():
      print(f"{k:20}: {v:.4f}")
