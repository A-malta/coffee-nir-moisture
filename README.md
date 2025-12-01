# Coffee NIR Moisture

Pipeline em Python para **pré-processamento de espectros NIR** de amostras de café e **treinamento de modelos de regressão** para previsão de teor de umidade.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

Para executar o pipeline completo (pré-processamento e modelagem):

```bash
python main.py
```

## Estrutura do Projeto

- `data/`: Dados brutos.
- `preprocessing/`: Scripts de tratamento de dados.
- `modeling/`: Scripts de treinamento e avaliação de modelos.
- `output/`: Resultados gerados (datasets processados, gráficos, modelos salvos).
