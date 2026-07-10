# -*- coding: utf-8 -*-
"""
Script 1 – ETL para a bacia do Rio Aquidauana
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from google.colab import drive

# ========== CONFIGURAÇÕES ==========
CONFIG = {
    "drive_path": "/content/drive/MyDrive/0526-mestrado",
    "start_date": "1994-02-01",
    "end_date": "2024-01-31",
    "min_common_pairs": 30,
    "output_file": "final_dataset_ANA_regressao.csv",
    "log_file": "etl_log.txt",
    "diagnostic_file": "preenchimento_diagnostico.csv",
    "stations": [
        {"file": "1954002_Chuvas.csv",   "name": "Precipitacao_1954002", "type": "Chuva"},
        {"file": "2054019_Chuvas.csv",   "name": "Precipitacao_2054019", "type": "Chuva"},
        {"file": "66926000_Vazoes.csv",  "name": "Vazao_66926000",       "type": "Vazao"},
        {"file": "2054005_Chuvas.csv",   "name": "Precipitacao_2054005", "type": "Chuva"},
        {"file": "2054009_Chuvas.csv",   "name": "Precipitacao_2054009", "type": "Chuva"},
        {"file": "2055003_Chuvas.csv",   "name": "Precipitacao_2055003", "type": "Chuva"},
        {"file": "66941000_Vazoes.csv",  "name": "Vazao_66941000",       "type": "Vazao"},
        {"file": "2055002_Chuvas.csv",   "name": "Precipitacao_2055002", "type": "Chuva"},
        {"file": "66945000_Vazoes.csv",  "name": "Vazao_66945000",       "type": "Vazao"}
    ],
    "desired_order": [
        "Precipitacao_1954002", "Precipitacao_2054019", "Vazao_66926000",
        "Precipitacao_2054005", "Precipitacao_2054009", "Precipitacao_2055003",
        "Vazao_66941000", "Precipitacao_2055002", "Vazao_66945000"
    ]
}

# ========== LOGGING ==========
def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Limpa handlers anteriores
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    # Handler para console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)
    # Handler para arquivo
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

# ========== CLASSE PRINCIPAL ==========
class ETLProcessor:
    def __init__(self, config):
        self.config = config
        self.master_range = pd.date_range(start=config["start_date"],
                                          end=config["end_date"], freq='D')
        self.final_df = None
        self.fill_report = []  # para diagnóstico

    def mount_drive(self):
        drive.mount('/content/drive', force_remount=True)
        return True

    def process_station(self, info):
        file_path = os.path.join(self.config["drive_path"], info["file"])
        if not os.path.exists(file_path):
            logger.warning(f"Arquivo não encontrado: {info['file']}")
            return None

        skip = 14 if info["type"] == "Chuva" else 15
        df = pd.read_csv(file_path, sep=';', skiprows=skip, encoding='latin1', low_memory=False)
        prefix = info["type"]
        val_cols = [f"{prefix}{str(i).zfill(2)}" for i in range(1, 32)]

        df_long = df.melt(id_vars=['Data'], value_vars=val_cols,
                          var_name='Day_Str', value_name='val')
        df_long['val'] = pd.to_numeric(df_long['val'].astype(str).str.replace(',', '.'),
                                       errors='coerce')
        df_long['date'] = pd.to_datetime(df_long['Data'], format='%d/%m/%Y', errors='coerce')
        df_long['Day'] = df_long['Day_Str'].str.extract(r'(\d+)').astype(int)
        df_long['date'] = df_long['date'] + pd.to_timedelta(df_long['Day'] - 1, unit='D')

        df_clean = df_long.dropna(subset=['date']).drop_duplicates('date')
        merged = pd.DataFrame({'date': self.master_range}).merge(
            df_clean[['date', 'val']], on='date', how='left'
        )
        return merged.set_index('date')['val']

    def fill_with_regression(self):
      """
      Preenchimento exclusivamente por regressão linear.
      Para cada estação alvo com falhas:
        - Ajusta regressão com cada possível preditora do mesmo tipo.
        - Para cada dia faltante, usa a primeira preditora disponível naquele dia
          (em ordem decrescente de correlação).
        - Se nenhuma preditora tiver dado, preenche com a média da estação alvo
          (equivalente a uma regressão constante).
      """
      df = self.final_df.copy()
      for target in df.columns:
          missing_mask = df[target].isna()
          missing_count = missing_mask.sum()
          if missing_count == 0:
              self.fill_report.append([target, 0, 0, 0, "completo", ""])
              continue

          # Determinar tipo
          if target.startswith('Precipitacao'):
              tipo = 'Precipitacao'
          elif target.startswith('Vazao'):
              tipo = 'Vazao'
          else:
              tipo = None

          pred_cols = [c for c in df.columns if c.startswith(tipo) and c != target] if tipo else df.columns.tolist()
          if not pred_cols:
              # Sem preditoras disponíveis: usar a média da própria estação
              mean_val = df[target].mean()
              df[target] = df[target].fillna(mean_val)
              self.fill_report.append([target, missing_count, 0, missing_count, "media_propria", ""])
              logger.info(f"{target}: preenchida com média própria (sem preditoras)")
              continue

          # 1. Calcular correlação e coeficientes de regressão para cada preditora
          regr_coeffs = {}  # chave: preditora, valor: (a, b, corr)
          for pred in pred_cols:
              common = df[target].notna() & df[pred].notna()
              if common.sum() < self.config["min_common_pairs"]:
                  continue
              x = df.loc[common, pred].values
              y = df.loc[common, target].values
              a, b = np.polyfit(x, y, 1)
              corr = np.corrcoef(x, y)[0, 1]
              regr_coeffs[pred] = (a, b, corr)

          # Ordenar preditoras por correlação decrescente
          ordered_preds = sorted(regr_coeffs.keys(), key=lambda p: regr_coeffs[p][2], reverse=True)

          filled = df[target].copy()
          filled_by_regr = 0
          filled_by_mean = 0

          for idx in df.index[missing_mask]:
              value_filled = False
              # Tenta preencher com regressão de cada preditora disponível no dia
              for pred in ordered_preds:
                  pred_val = df.loc[idx, pred]
                  if pd.notna(pred_val):
                      a, b, _ = regr_coeffs[pred]
                      estimated = a * pred_val + b
                      filled.loc[idx] = max(0, estimated)
                      filled_by_regr += 1
                      value_filled = True
                      break
              # Se nenhuma preditora tiver dado naquele dia, usar a média da estação alvo
              if not value_filled:
                  mean_target = df[target].mean()
                  filled.loc[idx] = max(0, mean_target)
                  filled_by_mean += 1

          df[target] = filled
          metodo = "regressao" if filled_by_regr > 0 else "media_propria"
          self.fill_report.append([target, missing_count, filled_by_regr, filled_by_mean,
                                  metodo, ", ".join(ordered_preds[:3])])  # lista até 3 melhores preditoras
          logger.info(f"{target}: {missing_count} falhas → {filled_by_regr} preenchidas por regressão, "
                      f"{filled_by_mean} pela média (sem preditor disponível).")

      self.final_df = df

    def run(self):
        logger.info("=== INÍCIO DO PROCESSAMENTO ETL ===")
        if not self.mount_drive():
            return

        # 1. Montagem inicial (com NaN)
        self.final_df = pd.DataFrame(index=self.master_range)
        for info in self.config["stations"]:
            series = self.process_station(info)
            if series is not None:
                self.final_df[info["name"]] = series

        # 2. Preenchimento
        self.fill_with_regression()

        # 3. Ordenação
        self.final_df.index.name = "Datas"
        output_df = self.final_df.reset_index()
        final_cols = ["Datas"] + [c for c in self.config["desired_order"]
                                  if c in output_df.columns]
        output_df = output_df[final_cols]

        # 4. Verificação final
        if output_df.isna().any().any():
            logger.error("Ainda existem NaN no dataset final!")
        else:
            logger.info("Nenhum NaN remanescente.")

        # 5. Exportação
        output_path = os.path.join(self.config["drive_path"], self.config["output_file"])
        output_df.to_csv(output_path, index=False)
        logger.info(f"Dataset final salvo em: {output_path}")

        # 6. Relatório de preenchimento
        diag_path = os.path.join(self.config["drive_path"], self.config["diagnostic_file"])
        pd.DataFrame(self.fill_report,
                     columns=["Estacao", "Falhas_antes", "Preenchidas_regressao",
                              "Preenchidas_interpol", "Metodo", "Preditora"]).to_csv(diag_path, index=False)
        logger.info(f"Diagnóstico salvo em: {diag_path}")
        logger.info("=== ETL CONCLUÍDO ===")

if __name__ == "__main__":
    log_path = os.path.join(CONFIG["drive_path"], CONFIG["log_file"])
    os.makedirs(CONFIG["drive_path"], exist_ok=True)
    logger = setup_logging(log_path)
    processor = ETLProcessor(CONFIG)
    processor.run()