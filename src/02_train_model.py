#!/usr/bin/env python3
"""
Pipeline de Treinamento LSTM para Previsão de Vazões Mínimas (Secas)
Bacia do Rio Aquidauana - Mestrado
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, random, traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import gc

# ======================== CONFIGURAÇÕES ========================
@dataclass
class Config:
    drive_data_path: str = '/content/drive/MyDrive/MESTRADO GERAL/0526-mestrado-Regressao-dados-estacoes-e-geral'
    root_output: str = '/content/drive/MyDrive/MESTRADO GERAL/0526-v100-mestrado-regressão'
    dataset_file: str = 'final_dataset_ANA_regressao.csv'

    # Grid intermediário
    grid_search: Dict = field(default_factory=lambda: {
        "n_passos": [7, 30, 60, 90, 180, 365],   # 6 janelas (curta à anual)
        "units": [32, 64, 128],                        # 2 tamanhos
        "layers": [1, 2],                         # 1 ou 2 camadas
        "dropout": [0.2, 0.3, 0.4],                    # regularização
        "learning_rate": [0.001],                 # fixo (estável)
        "activation": ['tanh'],                   # fixo (padrão LSTM)
        "batch_size": [32, 64],                       # fixo
        "epochs": 250,
        "split": [0.70, 0.15, 0.15]
    })

    seed: int = 42
    early_stop_patience: int = 12
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 0.0001

    # Índice da coluna de vazão alvo (Vazao_66945000) no array após remover 'Datas'
    target_col_idx: int = 8

    palette: Dict = field(default_factory=lambda: {
        'primary': '#003366', 'secondary': '#3399ff',
        'observed': '#003366', 'simulated': '#ff6b35',
        'residuals': '#2ecc71', 'grid': '#e0e0e0', 'line_1to1': '#e74c3c'
    })
    max_time_hours: int = 50
    checkpoint_frequency: int = 10

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class Timer:
    def __init__(self): self.start = datetime.now()
    def elapsed(self): return (datetime.now() - self.start).total_seconds() / 3600
    def should_continue(self, max_h, progress):
        if progress == 0: return True
        return (max_h - self.elapsed() / progress) > 0.1

# ======================== GERENCIAMENTO DE DADOS (CORRIGIDO) ========================
class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.n_features = None
        self.train_end = self.val_end = None
        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None
        self.y_train_real = self.y_val_real = self.y_test_real = None
        self.dates = None
        self.train_dates = self.val_dates = self.test_dates = None
        self._train_data = self._val_data = self._test_data = None
        self.target_idx = config.target_col_idx

    def load_and_preprocess(self):
        file_path = os.path.join(self.config.drive_data_path, self.config.dataset_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

        df = pd.read_csv(file_path, sep=',', decimal=',')
        df['Datas'] = pd.to_datetime(df['Datas'])
        self.dates = df['Datas'].values

        # Features cíclicas (não escaladas)
        month_sin = np.sin(2 * np.pi * df['Datas'].dt.month / 12)
        month_cos = np.cos(2 * np.pi * df['Datas'].dt.month / 12)

        # Dados hidrológicos (serão escalados)
        hydro_data = df.drop('Datas', axis=1).values
        self.n_features = hydro_data.shape[1] + 2  # 9 hidro + 2 cíclicas

        n = len(hydro_data)

        # Split 70/15/15
        self.train_end = int(n * self.config.grid_search["split"][0])
        self.val_end = self.train_end + int(n * self.config.grid_search["split"][1])

        # Separar e guardar cada partição individualmente
        train_hydro = hydro_data[:self.train_end]
        val_hydro = hydro_data[self.train_end:self.val_end]
        test_hydro = hydro_data[self.val_end:]

        # Scaler ajustado apenas no treino
        self.scaler.fit(train_hydro)

        train_scaled = self.scaler.transform(train_hydro)
        val_scaled = self.scaler.transform(val_hydro)
        test_scaled = self.scaler.transform(test_hydro)

        # Adicionar features cíclicas (não escaladas) a cada partição
        train_cyclic = np.column_stack([month_sin[:self.train_end], month_cos[:self.train_end]])
        val_cyclic = np.column_stack([month_sin[self.train_end:self.val_end], month_cos[self.train_end:self.val_end]])
        test_cyclic = np.column_stack([month_sin[self.val_end:], month_cos[self.val_end:]])

        # Guardar arrays completos de cada partição (já com cíclicas)
        self._train_data = np.hstack([train_scaled, train_cyclic])
        self._val_data = np.hstack([val_scaled, val_cyclic])
        self._test_data = np.hstack([test_scaled, test_cyclic])

        self._target_col_in_hydro = self.target_idx
        return self

    def prepare_sequences(self, n_steps):
        """
        Gera sequências independentemente para cada partição.
        """
        target_col = self._target_col_in_hydro

        # Gerar sequências separadamente
        X_train, y_train_scaled = self._gen_sequences(self._train_data, n_steps, target_col)
        X_val, y_val_scaled = self._gen_sequences(self._val_data, n_steps, target_col)
        X_test, y_test_scaled = self._gen_sequences(self._test_data, n_steps, target_col)

        # Atribuir diretamente
        self.X_train, self.y_train_scaled = X_train, y_train_scaled
        self.X_val, self.y_val_scaled = X_val, y_val_scaled
        self.X_test, self.y_test_scaled = X_test, y_test_scaled

        # Targets reais para métricas
        self.y_train_real = self.inverse_transform_flow(y_train_scaled)
        self.y_val_real = self.inverse_transform_flow(y_val_scaled)
        self.y_test_real = self.inverse_transform_flow(y_test_scaled)

        # Datas correspondentes (após o deslocamento da janela)
        if self.dates is not None:
            self.train_dates = self.dates[n_steps : self.train_end]
            self.val_dates = self.dates[self.train_end + n_steps : self.val_end]
            self.test_dates = self.dates[self.val_end + n_steps :]

        return self

    def _gen_sequences(self, data, n_steps, target_col):
        n_samples = len(data) - n_steps
        if n_samples <= 0:
            return np.array([]), np.array([])
        X = np.zeros((n_samples, n_steps, data.shape[1]))
        y = np.zeros(n_samples)
        for i in range(n_steps):
            X[:, i, :] = data[i:n_samples+i, :]
        y[:] = data[n_steps:, target_col]
        return X, y

    def inverse_transform_flow(self, y_scaled):
        if len(y_scaled) == 0:
            return np.array([])
        dummy = np.zeros((len(y_scaled), self.scaler.n_features_in_))
        dummy[:, self._target_col_in_hydro] = y_scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, self._target_col_in_hydro]

    @staticmethod
    def log_flow(y_real, eps=1e-6):
        return np.log(np.maximum(y_real, eps))

    @staticmethod
    def exp_flow(y_log):
        return np.exp(y_log)

# ======================== MÉTRICAS ========================
class MetricsCalculator:
    @staticmethod
    def calculate_all(obs, sim):
        if len(obs) == 0 or np.std(obs) == 0:
            return {k: np.nan for k in ['RMSE','NSE','NSElog','PBIAS',
                    'Q90_Obs','Q95_Obs','Q90_Sim','Q95_Sim','ErrQ90','ErrQ95','NSE_Seca','pct_negativos']}
        rmse = np.sqrt(mean_squared_error(obs, sim))
        nse = 1 - np.sum((obs-sim)**2) / np.sum((obs-np.mean(obs))**2)
        pbias = 100 * np.sum(obs-sim) / np.sum(obs)
        obs_log = np.log(np.maximum(obs, 1e-6))
        sim_log = np.log(np.maximum(sim, 1e-6))
        nse_log = (1 - np.sum((obs_log-sim_log)**2) / np.sum((obs_log-np.mean(obs_log))**2)) if np.std(obs_log)>0 else np.nan
        q90_obs, q95_obs = np.percentile(obs, 10), np.percentile(obs, 5)
        q90_sim, q95_sim = np.percentile(sim, 10), np.percentile(sim, 5)
        err_q90 = 100*(q90_sim - q90_obs)/q90_obs if q90_obs else np.nan
        err_q95 = 100*(q95_sim - q95_obs)/q95_obs if q95_obs else np.nan
        idx_seca = obs <= q90_obs
        nse_seca = (1 - np.sum((obs[idx_seca]-sim[idx_seca])**2) / np.sum((obs[idx_seca]-np.mean(obs[idx_seca]))**2)) if np.sum(idx_seca)>1 and np.std(obs[idx_seca])>0 else np.nan
        pct_negativos = 100 * np.sum(sim < 0) / len(sim)
        return {'RMSE':rmse, 'NSE':nse, 'NSElog':nse_log, 'PBIAS':pbias,
                'Q90_Obs':q90_obs, 'Q95_Obs':q95_obs, 'Q90_Sim':q90_sim, 'Q95_Sim':q95_sim,
                'ErrQ90':err_q90, 'ErrQ95':err_q95, 'NSE_Seca':nse_seca, 'pct_negativos':pct_negativos}

    @staticmethod
    def calculate_by_season(obs, sim, dates):
        """
        Calcula métricas separadamente para cada período hidrológico.
        dates: array de datetime64 ou similar.
        """
        dt = pd.to_datetime(dates)
        seasons = {
            'Cheia': [12, 1, 2, 3],
            'Vazante': [4, 5, 6],
            'Seca': [7, 8, 9, 10, 11]
        }
        results = {}
        for season_name, months in seasons.items():
            mask = np.isin(dt.month, months)
            if mask.sum() > 1 and np.std(obs[mask]) > 0:
                obs_s = obs[mask]
                sim_s = sim[mask]
                nse_s = 1 - np.sum((obs_s - sim_s)**2) / np.sum((obs_s - np.mean(obs_s))**2)
                obs_log = np.log(np.maximum(obs_s, 1e-6))
                sim_log = np.log(np.maximum(sim_s, 1e-6))
                nselog_s = (1 - np.sum((obs_log - sim_log)**2) /
                           np.sum((obs_log - np.mean(obs_log))**2)) if np.std(obs_log) > 0 else np.nan
                results[season_name] = {'NSE': round(nse_s, 4), 'NSElog': round(nselog_s, 4), 'n_dias': mask.sum()}
            else:
                results[season_name] = {'NSE': np.nan, 'NSElog': np.nan, 'n_dias': mask.sum()}
        return results

# ======================== VISUALIZAÇÃO  ========================
class VisualizationManager:
    def __init__(self, config):
        self.palette = config.palette
        plt.rcParams.update({'font.size':12, 'axes.titlesize':14, 'axes.labelsize':12,
                             'figure.dpi':150, 'savefig.dpi':300, 'savefig.bbox':'tight'})
    def _safe_save(self, fig, path):
        try: fig.savefig(path)
        except Exception as e: print(f"   ⚠️ Erro gráfico: {e}")
        finally: plt.close(fig)

    def plot_loss(self, history, path, exp_name):
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(history.history['loss'], label='Treino', color=self.palette['observed'])
        ax.plot(history.history['val_loss'], label='Validação', color=self.palette['simulated'])
        ax.set_title(f'Loss - {exp_name}'); ax.legend(); ax.grid(alpha=0.3)
        self._safe_save(fig, path)

    def plot_hydrogram(self, obs, sim, nse, path, title="Hidrograma"):
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(obs, label='Observado', color=self.palette['observed'], alpha=0.8)
        ax.plot(sim, label='Simulado', color=self.palette['simulated'], ls='--', alpha=0.8)
        ax.set_title(f"{title} - NSE: {nse:.3f}"); ax.legend(); ax.grid(alpha=0.3)
        self._safe_save(fig, path)

    def plot_scatter(self, obs, sim, path):
        fig, ax = plt.subplots(figsize=(7,7))
        ax.scatter(obs, sim, alpha=0.5, color=self.palette['secondary'])
        lim = [min(obs.min(), sim.min()), max(obs.max(), sim.max())]
        ax.plot(lim, lim, '--', color=self.palette['line_1to1'], label='1:1')
        ax.legend(); ax.grid(alpha=0.3)
        self._safe_save(fig, path)

    def plot_flow_duration_curve(self, obs, sim, path):
        fig, ax = plt.subplots(figsize=(10,6))
        for data, label, color in [(obs,'Observado',self.palette['observed']),(sim,'Simulado',self.palette['simulated'])]:
            sorted_data = np.sort(data)[::-1]
            prob = np.arange(1, len(sorted_data)+1)/len(sorted_data)*100
            ax.plot(prob, sorted_data, label=label, color=color)
        q90_obs, q95_obs = np.percentile(obs,10), np.percentile(obs,5)
        ax.axvline(90, color=self.palette['line_1to1'], ls='--')
        ax.text(90.5, q90_obs, f'Q90: {q90_obs:.2f}', color=self.palette['line_1to1'], fontweight='bold')
        ax.axvline(95, color='orange', ls='--')
        ax.text(95.5, q95_obs, f'Q95: {q95_obs:.2f}', color='orange', fontweight='bold')
        ax.set_yscale('log'); ax.set_xlabel('Permanência (%)'); ax.set_ylabel('Vazão (m³/s)')
        ax.legend(); ax.grid(True, which='both', alpha=0.3); ax.set_xlim(0,105)
        self._safe_save(fig, path)

    def plot_residuals(self, obs, sim, path):
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(obs-sim, color=self.palette['residuals'], alpha=0.7)
        ax.axhline(0, color='black', ls='--')
        ax.set_title('Resíduos (Obs - Sim)'); ax.grid(alpha=0.3)
        self._safe_save(fig, path)

# ======================== CONSTRUÇÃO DO MODELO  ========================
class LSTMModelBuilder:
    @staticmethod
    def build(n_steps, n_features, params):
        """
        Modelo LSTM simples com MSE.
        A seleção do melhor modelo será feita via NSElog na validação,
        garantindo foco em vazões baixas sem distorcer a função de perda.
        """
        model = Sequential()
        model.add(Input(shape=(n_steps, n_features)))
        for i in range(params['layers']):
            return_seq = i < params['layers'] - 1
            model.add(LSTM(params['units'],
                           activation=params['activation'],
                           return_sequences=return_seq,
                           kernel_regularizer=l2(0.001),
                           recurrent_regularizer=l2(0.001),
                           dropout=params['dropout']))
            model.add(Dropout(params['dropout']))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=params['learning_rate'], clipnorm=1.0),
                      loss='mse')
        return model

# ======================== EXECUÇÃO DO EXPERIMENTO ========================
class ExperimentRunner:
    def __init__(self, config, data_manager):
        self.config = config
        self.data = data_manager
        self.viz = VisualizationManager(config)
        self.metrics = MetricsCalculator()

    def _parse_existing_results(self, exp_path):
        results_file = os.path.join(exp_path, "resultados_completos.txt")
        linha = {"Experimento": os.path.basename(exp_path)}
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
            current_period = None
            for line in lines:
                if line.startswith('--- PERÍODO:'):
                    current_period = line.split(':')[1].strip()
                elif ':' in line and current_period:
                    key, val = line.split(':', 1)
                    key = key.strip(); val = val.strip()
                    try:
                        linha[f"{current_period}_{key}"] = float(val) if '.' in val or val.lstrip('-').isdigit() else val
                    except: pass
            return linha
        except: return None

    def run_single(self, steps, params, bs, exp_folder, exp_path):
        model_path = os.path.join(exp_path, "modelo_lstm.keras")
        results_path = os.path.join(exp_path, "resultados_completos.txt")
        if os.path.exists(model_path) and os.path.exists(results_path):
            print(f"   ✅ Já completo. Recuperando métricas...")
            return self._parse_existing_results(exp_path)

        self.data.prepare_sequences(steps)
        X_train, y_train_scaled = self.data.X_train, self.data.y_train_scaled
        X_val, y_val_scaled = self.data.X_val, self.data.y_val_scaled
        X_test, y_test_scaled = self.data.X_test, self.data.y_test_scaled

        if len(X_train) == 0:
            raise ValueError(f"Sequências vazias para steps={steps}")

        # Verificações de sanidade
        assert np.isfinite(X_train).all(), "NaN/Inf no X_train"
        assert np.isfinite(y_train_scaled).all(), "NaN/Inf no y_train_scaled"
        assert np.isfinite(X_val).all(), "NaN/Inf no X_val"
        assert np.isfinite(y_val_scaled).all(), "NaN/Inf no y_val_scaled"

        # Treinamento em log
        y_train_log = self.data.log_flow(self.data.y_train_real)
        y_val_log = self.data.log_flow(self.data.y_val_real)

        assert np.isfinite(y_train_log).all(), "NaN/Inf no y_train_log"
        assert np.isfinite(y_val_log).all(), "NaN/Inf no y_val_log"

        model = LSTMModelBuilder.build(steps, self.data.n_features, params)
        early_stop = EarlyStopping(monitor='val_loss', patience=self.config.early_stop_patience,
                                   restore_best_weights=True, verbose=0)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=self.config.reduce_lr_factor,
                                                         patience=self.config.reduce_lr_patience,
                                                         min_lr=self.config.min_learning_rate, verbose=0)

        history = model.fit(
            X_train, y_train_log,
            validation_data=(X_val, y_val_log),
            epochs=self.config.grid_search["epochs"],
            batch_size=bs,
            callbacks=[early_stop, reduce_lr],
            verbose=0, shuffle=False
        )

        model.save(model_path)
        self.viz.plot_loss(history, os.path.join(exp_path, "curva_loss.png"), exp_folder)

        return self._evaluate(model, history, exp_folder, exp_path, steps)

    def _evaluate(self, model, history, exp_folder, exp_path, steps):
        sets = {
            "Treino": (self.data.X_train, self.data.y_train_real),
            "Validação": (self.data.X_val, self.data.y_val_real),
            "Teste": (self.data.X_test, self.data.y_test_real)
        }
        linha = {"Experimento": exp_folder, "Steps": steps}
        results_file = os.path.join(exp_path, "resultados_completos.txt")

        with open(results_file, "w") as f:
            f.write(f"Experimento: {exp_folder}\n{'='*60}\n")
            f.write(f"Épocas: {len(history.history['loss'])}\n\n")

            for nome, (X, y_real) in sets.items():
                if len(X) == 0:
                    continue
                pred_log = model.predict(X, verbose=0)
                sim = self.data.exp_flow(pred_log.flatten())
                obs = y_real
                m = self.metrics.calculate_all(obs, sim)

                f.write(f"--- PERÍODO: {nome} ---\n")
                f.write(f"RMSE: {m['RMSE']:.4f}\nNSE: {m['NSE']:.4f}\nNSElog: {m['NSElog']:.4f}\nPBIAS: {m['PBIAS']:.2f}%\n")
                f.write(f"Q90_Obs: {m['Q90_Obs']:.2f} | Q90_Sim: {m['Q90_Sim']:.2f} | ErrQ90: {m['ErrQ90']:.2f}%\n")
                f.write(f"Q95_Obs: {m['Q95_Obs']:.2f} | Q95_Sim: {m['Q95_Sim']:.2f} | ErrQ95: {m['ErrQ95']:.2f}%\n")
                f.write(f"NSE_Seca: {m['NSE_Seca']:.4f}\n")

                for k, v in m.items():
                    linha[f"{nome}_{k}"] = v

                # ----- MÉTRICAS SAZONAIS  -----
                if nome == "Teste":
                    datas_teste = self.data.test_dates
                    if datas_teste is not None and len(datas_teste) == len(obs):
                        seasonal = self.metrics.calculate_by_season(obs, sim, datas_teste)
                        f.write(f"\n--- MÉTRICAS POR PERÍODO HIDROLÓGICO (Teste) ---\n")
                        for season, met in seasonal.items():
                            f.write(f"{season}: NSE={met['NSE']:.4f}, NSElog={met['NSElog']:.4f}, n_dias={met['n_dias']}\n")
                            linha[f"Teste_{season}_NSE"] = met['NSE']
                            linha[f"Teste_{season}_NSElog"] = met['NSElog']
                # ---------------------------------------------

                if nome in ["Treino", "Validação", "Teste"]:
                    self.viz.plot_hydrogram(obs, sim, m['NSE'],
                                            os.path.join(exp_path, f"hidrograma_{nome.lower()}.png"), f"Hidrograma - {nome}")
                    self.viz.plot_scatter(obs, sim, os.path.join(exp_path, f"dispersao_{nome.lower()}.png"))
                    self.viz.plot_flow_duration_curve(obs, sim, os.path.join(exp_path, f"curva_permanencia_{nome.lower()}.png"))
                    self.viz.plot_residuals(obs, sim, os.path.join(exp_path, f"residuos_{nome.lower()}.png"))

        linha["Epocas_Treinadas"] = len(history.history['loss'])
        return linha

# ======================== ORQUESTRADOR DO GRID SEARCH ========================
class GridSearchOrchestrator:
    def __init__(self, config):
        self.config = config
        self.data_mgr = None
        self.runner = None
        self.timer = Timer()
        self.results = []
        self.completed = set()

    def setup(self):
        print(f"\n{'='*60}\n🚀 PIPELINE LSTM v6.1 – GRID INTERMEDIÁRIO + MÉTRICAS SAZONAIS\n{'='*60}")
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive'):
                drive.mount('/content/drive')
        except ImportError: pass
        os.makedirs(self.config.root_output, exist_ok=True)
        set_seeds(self.config.seed)
        self.data_mgr = DataManager(self.config).load_and_preprocess()
        self.runner = ExperimentRunner(self.config, self.data_mgr)
        self._load_checkpoint()
        print(f"✅ {len(self.completed)} experimentos já concluídos.\n")
        return self

    def _load_checkpoint(self):
        csv_path = os.path.join(self.config.root_output, "resultado_geral_mestrado.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'Experimento' in df.columns and len(df) > 0:
                    self.results = df.to_dict('records')
                    self.completed = set(df['Experimento'].tolist())
                    return
            except: pass
        print("ℹ️ Nenhum checkpoint válido.")

    def _save_results(self):
        if not self.results: return
        df = pd.DataFrame(self.results)
        path = os.path.join(self.config.root_output, "resultado_geral_mestrado.csv")
        df.to_csv(path.replace('.csv','.tmp'), index=False)
        if os.path.exists(path): os.remove(path)
        os.rename(path.replace('.csv','.tmp'), path)

    def _scan_folders(self):
        for folder in os.listdir(self.config.root_output):
            if not folder.startswith("EXP_"): continue
            p = os.path.join(self.config.root_output, folder)
            if os.path.exists(os.path.join(p,"modelo_lstm.keras")) and os.path.exists(os.path.join(p,"resultados_completos.txt")):
                self.completed.add(folder)

    def _generate_queue(self):
        q = []
        for s in self.config.grid_search["n_passos"]:
            for u in self.config.grid_search["units"]:
                for lc in self.config.grid_search["layers"]:
                    for dr in self.config.grid_search["dropout"]:
                        for lr in self.config.grid_search["learning_rate"]:
                            for act in self.config.grid_search["activation"]:
                                for bs in self.config.grid_search["batch_size"]:
                                    params = {'units':u, 'layers':lc, 'dropout':dr,
                                              'learning_rate':lr, 'activation':act}
                                    folder = f"EXP_st{s}_u{u}_L{lc}_dr{dr}_lr{lr}_{act}_bs{bs}"
                                    if folder not in self.completed:
                                        q.append((s, params, bs, folder, os.path.join(self.config.root_output, folder)))
        return q

    def run(self):
        self._scan_folders()
        queue = self._generate_queue()
        total = len(queue) + len(self.completed)
        print(f"🔧 Grid: {total} combinações. Pendentes: {len(queue)}.\n")
        if not queue:
            print("✅ Todos concluídos!")
            return self.results

        for idx, (steps, params, bs, folder, path) in enumerate(queue):
            progress = (len(self.completed)+idx)/total
            if not self.timer.should_continue(self.config.max_time_hours, progress):
                print("⏰ Tempo limite. Salvando...")
                break

            os.makedirs(path, exist_ok=True)
            print(f"[{self.timer.elapsed():.1f}h] {idx+1}/{len(queue)}: {folder}")
            try:
                res = self.runner.run_single(steps, params, bs, folder, path)
                if res is not None:
                    res.update({"Steps":steps, "LR":params['learning_rate'], "Layers":params['layers'],
                                "Act":params['activation'], "Units":params['units'], "Dropout":params['dropout'],
                                "Batch":bs})
                    self.results.append(res)
                self.completed.add(folder)
                if (idx+1) % self.config.checkpoint_frequency == 0:
                    self._save_results()
                    print(f"   💾 Checkpoint salvo.")
            except Exception as e:
                print(f"   ❌ Erro: {e}")
                traceback.print_exc()
            finally:
                K.clear_session(); gc.collect()

        self._save_results()
        print(f"\n✅ Finalizado. Total: {len(self.results)} no CSV.")
        if self.results:
            df = pd.DataFrame(self.results)
            if 'Validação_NSElog' in df.columns:
                best = df.loc[df['Validação_NSElog'].idxmax()]
                print(f"\n🏆 Melhor modelo (Validação NSElog): {best['Experimento']}")
                print(f"   NSElog Val: {best['Validação_NSElog']:.4f} | NSElog Teste: {best.get('Teste_NSElog', np.nan):.4f}")
                print(f"   NSE Seca Teste: {best.get('Teste_NSE_Seca', np.nan):.4f}")
        return self.results

def main():
    config = Config()
    print(f"📅 Início: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    orch = GridSearchOrchestrator(config)
    try:
        orch.setup()
        return orch.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido. Salvando checkpoint...")
        orch._save_results()
        return orch.results
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        orch._save_results()
        raise

if __name__ == "__main__":
    main()