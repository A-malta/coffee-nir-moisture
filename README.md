# Coffee NIR Moisture

Pipeline em Python para **pré-processamento de espectros NIR** de amostras de café, com plano de expansão para **treinamento de modelos de regressão** para previsão de teor de umidade.

---

## Objetivos do projeto

1. Organizar em um único fluxo o tratamento de espectros NIR de café.
2. Aplicar pré-processamentos espectrais padronizados e reproduzíveis.
3. (Futuro) Treinar e comparar diferentes modelos de regressão para prever umidade:
   - Partial Least Squares (PLS)
   - Random Forest Regressor
   - Support Vector Regression (SVR)
   - Perceptron Multicamadas (MLP)

---

## Estrutura do repositório

```text
coffee-nir-moisture/
├── data/
│   └── raw/              # Dados brutos (espectros, referências de umidade, etc.)
├── preprocessing/        # Funções de pré-processamento espectral
├── pipeline.py           # Pipeline ATUAL: pré-processamento dos dados
├── requirements.txt      # Dependências Python
└── .gitignore


---

---

## Pré-processamento de Espectros

Abaixo estão os gráficos gerados para cada técnica de pré-processamento aplicada aos espectros NIR:

```text
Diretório: coffee-nir-moisture/output/preprocessed/plots
```

**Área Normalizada**  
![](output/preprocessed/plots/area_normalization.png)

**Correção de Linha de Base**  
![](output/preprocessed/plots/baseline_correction.png)

**Espectros Brutos**  
![](output/preprocessed/plots/bruto.png)

**Centralização pela Média**  
![](output/preprocessed/plots/mean_centering.png)

**Multiplicative Scatter Correction (MSC)**  
![](output/preprocessed/plots/msc.png)

**Derivada 1ª ordem (Savitzky-Golay)**  
![](output/preprocessed/plots/savitzkygolay_1a_derivada.png)

**Derivada 2ª ordem (Savitzky-Golay)**  
![](output/preprocessed/plots/savitzkygolay_2a_derivada.png)

**Standard Normal Variate (SNV)**  
![](output/preprocessed/plots/snv.png)
