# Smart Irrigation — звіт про реалізовану ML-частину

**Дата:** 31.10.2026  
**Статус:** завершений дослідницький прототип, не готовий до автономного поливу.

## 1. Що реально реалізовано

- **Forecasting (попередня версія):** LightGBM регресія для 24, 72, 168 годин; порівняння з persistence, time-aware train/test, RMSE/MAE. На цьому наборі persistence поки кращий.
- **Physics (FAO-56-style):** коефіцієнт культур Kc із припущеного календаря, ETc=Kc×ET₀, оцінка TAW/RAW і depletion для супутникової SMAP-вологості кореневого шару. Діапазон FC/PWP — припущення з наданого попереднього ноутбука, а не виміряне значення для полів.
- **Proxy classification:** окремий LightGBMClassifier на 3 категорії `no_action`, `monitor`, `review_irrigation`, які створені фізичним правилом. Немає жодних вигаданих записів про реальні дії.
- **Гібридний модуль:** фізичне правило є основним; ML надає допоміжне рішення. При суперечності і невизначеності показуємо ручну перевірку. Немає керування насосом.
- **Оцінювання:** 3 walk-forward time windows із 1-добовим gap, поріг ризикового класу налаштований лише на 2024 р., test — 2025 р.; окремі оцінки по полях і помилки.

## 2. Обсяг даних

Вхід: `irrigation_ml_dataset.csv`, близько 27 346 записів, 131 колонка, 2 поля, 2024–2025 роки. Сформовано **1 160 добових знімків**: 497 `no_action_out_of_season`, 22 `insufficient_data`, 339 `no_action`, 241 `monitor`, 61 `review_irrigation`.

Для ML-класифікації доступно **641** знімок трьох класів. Серед них **61** умовно позитивний proxy-сигнал. **Усі позитиви належать `field_3335`**. Виведено `daily_decision_table_PROXY.csv` та `decision_labels_daily_PROXY_NOT_VALIDATED.csv`, щоб уникнути їх плутанини зі справжніми мітками `decision_labels_daily.csv`.

## 3. Метрики (лише порівняння з rule teacher)

| Фолд | Тестовий період | N | Review у proxy | Recall | Precision | F₂ | FN | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2025-06-01 — 2025-07-15 | 88 | 11 | 0.000 | nan | 0.000 | 11 | 0.549 |
| 2 | 2025-07-15 — 2025-08-15 | 62 | 16 | 0.938 | 0.750 | 0.893 | 1 | 0.812 |
| 3 | 2025-08-15 — 2025-10-06 | 94 | 10 | 0.900 | 0.391 | 0.714 | 1 | 0.548 |


Сукупно на незалежних часових тестових вікнах **244** приклади, позитивних у фізичного правила **37**. Серед них ML відтворила **24**, пропустила **13**. **Macro F1 = 0.687**, **Review Recall = 0.649**, **Precision = 0.558**, **F₂ = 0.628**, **PR-AUC = 0.587**. Це **не** оцінка точності реального зрошення.

Перший тестовий фолд не розпізнав жодного із 11 proxy-сигналів `review_irrigation`. Тому навіть відтворення приблизного physics teacher є нестабільним; у використанні залишаємо первинний прозорий rule engine.

## 4. Конкретна ML-архітектура

```
Вхідний historical CSV (2 поля)
  -> causally formed 06:00 daily snapshots
  -> FAO-56-style physical rule (unverified assumptions)
      -> no_action / monitor / review_irrigation
      -> missing/out of season handled separately
  -> past-only feature engineering
      -> LightGBM 3-class proxy classifier
      -> train-only threshold selection (2024)
      -> 2025 three walk-forward holdouts
  -> hybrid historical advisory (manual-review gating)
```

Ознаки: SMAP root-zone θ і вік спостереження, ET₀ за минулу добу, припущені Kc/глибина кореня, температура, опади за попередні 24 години, лаги SMAP за реальні календарні дні, sensor moisture, battery, quality flags, IF anomaly indicator **тільки коли він реально є**. Майбутні опади, дані `t+H`, мітки та прямі формульні виходи `RAW/depletion` до фіч не входять. Великі пропуски не заповнюються інтерполяцією.

## 5. Обмеження та ризики

1. **Proxy labels:** `review_irrigation` — лише наслідок розрахунку за припущеними ґрунтовими константами; це не факт правильної агрономічної рекомендації.
2. **Невизначеність ґрунту:** **всі 61 позитивні випадки** чутливі до зміни FC/PWP в дозволеному широкому діапазоні.
3. **Один позитивний майданчик:** для `field_3334` не було жодного позитиву; узагальнення на інші поля не тестувалося.
4. **Немає еталонів:** відсутні достовірні журнали фактичних поливів, калібрування вологості, підтверджені дати посіву і незалежні агрономічні labels.
5. **Реальна робота:** архівні 2024–2025 дані, без гарантованої issue-time погодної прогностики. Не можна приймати їх за live 2026 рекомендації.
6. **Не плутати метрики:** F1/Recall — лише узгодженість із teacher. `hybrid_advisory_historical.csv` містить можливі **in-sample** передбачення фінальної моделі, тому **не** застосовується для підрахунку незалежних метрик; для цього призначено окремий `decision_policy_walkforward_predictions.csv`.

## 6. Рекомендовані файли для передачі

- `ML_Decision_Policy_FAO56.ipynb` — **виконаний** notebook з таблицями, графіками і висновками.
- `train_decision_policy.py` — навчання, 3 walk-forward splits, class weighting, F₂-threshold, моделі, QA та метрики.
- `irrigation_recommendations.py` — фізичний FAO-56-style rule engine.
- `predict_hybrid_policy.py` — поєднання physical rule і ML як перевірки узгодженості.
- `export_proxy_datasets.py` — вивантаження і чітке маркування proxy-фіч/міток.
- `models/irrigation_policy_proxy_MULTICLASS_UNVERIFIED.joblib` — модель і provenance.
- `outputs/decision_policy_*` — metrics, confusion matrix, predictions, class balance.
- `outputs/hybrid_advisory_historical.csv` — всі історичні advisory-статуси (не тестова оцінка).
- `Smart_Irrigation_ML.ipynb`, `ml_pipeline.py` — попередня частина прогнозу вологості.
- `tests/` — 13 автоматичних перевірок виконалися успішно.

## 7. Команди запуску

```bash
cd smart_irrigation_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ml_pipeline.py --data input/irrigation_ml_dataset.csv
python irrigation_recommendations.py --data input/irrigation_ml_dataset.csv
python train_decision_policy.py --data input/irrigation_ml_dataset.csv --teacher outputs/decision_prototype_daily.csv
python export_proxy_datasets.py --data input/irrigation_ml_dataset.csv --teacher outputs/decision_prototype_daily.csv
python predict_hybrid_policy.py --data input/irrigation_ml_dataset.csv
python -m unittest discover -s tests -v
jupyter notebook ML_Decision_Policy_FAO56.ipynb
```

**Висновок:** користувацька частина ML-дослідження реалізована на наявних даних без вигаданих справжніх міток. Для наукового прототипу це повний training/evaluation pipeline. Для заяви про точні рекомендації реального поливу потрібна незалежна валідація.
