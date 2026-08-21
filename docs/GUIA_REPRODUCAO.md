# Guia de Reproducao

## Ambiente

- Python: usar a versao registrada em `requirements.lock.txt` quando disponivel.
- Instalar dependencias: `python -m pip install -r requirements.txt`.
- O dataset UCI id=350 e carregado automaticamente pelo pipeline; nao e necessario upload manual nem credencial.
- Seed principal do candidato final: `42`.

## Estrutura

- `src/credit_default/`: ingestao, features, preprocessing, modelagem e avaliacao.
- `notebooks/`: interface Colab fina.
- `scripts/`: smoke test, verificacao de artefatos e scripts experimentais.
- `artifacts/final/`: evidencia persistida do desenvolvimento e do holdout final.
- `docs/`: relatorio, resumo e este guia.

## Verificacoes permitidas

```powershell
python -m pytest
python scripts/smoke_test.py
python scripts/verify_artifacts.py
```

O TEST ja foi consumido como holdout final. Nao execute novamente `scripts/run_final_test.py`; o runner possui guarda contra reexecucao e exige recuperacao explicita apenas para o incidente de persistencia ja registrado.

## Artefato final

Consulte `artifacts/final/development_winner.json` para modelo, features, hiperparametros, seed e threshold congelados. Consulte `artifacts/final/final_test_metrics.json` e `final_test_gate_status.json` para a avaliacao final. As figuras estao em `artifacts/final/` e em `delivery/figures/`.

## Auditoria

`artifacts/final/leakage_audit.json` registra fit no TRAIN, OOF interno, ausencia de uso do TEST para selecao e a contagem real de acessos fisicos ao holdout. Nenhuma nova rodada de treino foi executada para preparar esta entrega.

