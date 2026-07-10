# Aquidauana Streamflow Forecasting

[![Dissertação ProfÁgua](https://img.shields.io/badge/Dissertação-ProfÁgua%2FUEMS-blue.svg)](https://github.com/ana-lapas/aquidauana-streamflow-forecasting)
[![MVP Link](https://img.shields.io/badge/MVP-EWS_Rio_Aquidauana-green.svg)](https://github.com/ana-lapas/EWS-Rio-Aquidauana)

Repositório técnico referente à dissertação: **"Previsão de vazões mínimas usando redes neurais recorrentes: estudo de caso para a bacia hidrográfica do rio Aquidauana/MS"**.

---

## 📋 Sobre o Projeto
Este sistema foi desenvolvido para aprimorar a previsão de vazões mínimas em bacias tropicais sazonais, oferecendo suporte técnico à decisão para outorgas de recursos hídricos e mitigação de secas, superando as limitações dos modelos hidrológicos tradicionais.

## 🛠️ Arquitetura do Produto
O pipeline está estruturado em três módulos:
*   **ETL (`src/01_etl_aquidauana.py`):** Limpeza, padronização e imputação de falhas (regressão linear) das séries ANA (1994-2024).
*   **Modelagem (`src/02_train_lstm_model.py`):** Treinamento de redes LSTM via *Grid Search* (216 configs), otimizadas para baixas vazões através de métricas logarítmicas ($NSE_{log}$) e correção de viés de Duan.
*   **Interface (MVP):** Sistema de visualização de previsões em tempo real.

## 🚀 Guia de Configuração e Uso

### 1. Pré-requisitos
Certifique-se de ter o ambiente configurado com as dependências necessárias:

```bash
pip install -r requirements.txt
```

### 2. Fluxo de Execução
Para replicar os resultados da pesquisa, siga a ordem estrita dos scripts:Passo A: Processamento de Dados (ETL)

```Bash
python 01_etl_aquidauana.py
```

Este script gera o final_dataset_ANA_regressao.csv, essencial para o treinamento.

### 3. Modelagem
Para realizar o treinamento e validação das 216 combinações de hiperparâmetros:

```Bash
python 02_train_lstm_model.py
```
O resultado deste script gera o modelo final (.keras) que sustenta o MVP.

### 🛡️ Reprodutibilidade e Rigor Científico

O pipeline foi projetado para assegurar rigor científico em todas as etapas:

Prevenção de Vazamento (Data Leakage): A imputação por regressão linear e a normalização dos dados foram calibradas estritamente na partição de treino.

Correção de Viés: Aplicação do estimador de espalhamento de Duan (1983) para eliminar subestimativas sistemáticas da retransformação logarítmica.

Validação: Divisão cronológica estrita (70% Treino, 15% Validação, 15% Teste).

### 👤 Autor
Ana Paula Lapas Leão

Mestrado Profissional (ProfÁgua/UEMS) – 2026

Orientação: Prof. Dr. Ariel Ortiz Gomes