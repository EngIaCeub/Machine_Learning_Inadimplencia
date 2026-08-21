# Delivery Manifest

Status: `FINAL_ACADEMIC_DELIVERY_READY`

## Conteudo

- `RELATORIO_FINAL.md`: narrativa tecnica completa.
- `RESUMO_EXECUTIVO.md`: sintese para submissao.
- `GUIA_REPRODUCAO.md`: ambiente, comandos, artefatos e protecao do TEST.
- `figures/`: copias das figuras finais persistidas, sem recalculo.
- `../artifacts/final/`: evidencia fonte, metricas, predictions, auditoria e comparacoes.

## Modelo e metricas

- Modelo: `CatBoostClassifier`, A3 = `ROUND1 + BILL + PAYMENT`.
- Threshold: `0.247743`.
- Validation: AUC `0.7811`, Recall `0.6050`, Macro F1 `0.7025`.
- Test: AUC `0.7865`, Recall `0.6104`, Macro F1 `0.7019`.
- Gates A1: `3/3 PASS`.

## Auditoria e testes

- Target positivo: `default=1`.
- Test access count: `2`, com segunda execucao deterministica de recuperacao de persistencia.
- Post-test tuning: `NO`.
- Nenhum novo experimento foi executado na preparacao desta entrega.
- Data de preparacao: `2026-08-21`.
- Colab publication link: `https://colab.research.google.com/github/EngIaCeub/Machine_Learning_Inadimplencia/blob/main/notebooks/FINAL_COLAB_REPRODUCIBLE.ipynb`.

## Identificacao

Os hashes e metadados dos artefatos principais podem ser verificados no proprio diretorio `artifacts/final/`. O pacote nao inclui secrets, tokens, credenciais ou caminhos pessoais absolutos.

SHA-256 dos arquivos principais na preparacao:

- `artifacts/final/development_winner.json`: `1B0165503957249C8C73B786382BA47AA63FB8A8806087499E67D1EB89F0A9FE`
- `artifacts/final/final_test_metrics.json`: `771C5F32879CD0FDD0B703969EDF7615E0C7932451450BC17CFDCA8F926053D3`
- `artifacts/final/final_test_gate_status.json`: `517A301DF375995256B575E25DAB69462F89896DDC3D6D4E07DAA19434FEE297`
- `artifacts/final/leakage_audit.json`: `45F3DFB4F3CB2C4BF4FE5F66A05BFC3963D4AE6539C020398D8DD34A49422C43`
