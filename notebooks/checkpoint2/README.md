# Checkpoint 2: reusable model outputs and CCS probes

## Запуск

Полный последовательный CUDA-запуск из терминала (вне ограничивающего GPU sandbox):

```bash
cd /home/katyukh/projects/latent-behavior-alignment
../polarity-probing/.venv/bin/python -u -B notebooks/checkpoint2/run_all.py
```

Чтобы продолжить с конкретной модели в процессе, независимом от терминала:

```bash
../polarity-probing/.venv/bin/python -u -B notebooks/checkpoint2/run_all.py \
  --start-model gemma-2-9b --background
```

Предыдущие модели пропускаются, а cache выбранной модели проверяется обычным
способом. Перед повторным запуском убедитесь, что предыдущий runner завершён;
`--background` печатает PID, вывод сохраняется в `logs/runner.log`.

Runner использует локальные веса, сохраняет логи и `run_status.json` в
`results/checkpoint2/logs/`. Модели и offline-анализ запускаются в отдельных
процессах; при повторном запуске валидные model/probe caches переиспользуются.
Для forward activations резервируется VRAM: weights placement ограничен 12 GiB
на GPU, остальные веса `device_map="auto"` размещает в RAM. Dtype сохраняется.
Итоговые CSV лежат в `results/checkpoint2/analysis/checkpoint1_compatible/`.

Используйте существующее Python-окружение с numpy, pandas, scikit-learn,
torch, transformers, accelerate, huggingface_hub, tqdm и Jupyter.
Для Gemma нужен доступ к соответствующим весам Hugging Face.

1. Откройте `01_build_artifacts.ipynb`. Выберите модели в `SELECTED_MODELS`.
   Notebook проверяет cache **до загрузки модели**. При отсутствии cache одна
   модель последовательно извлекает hidden states и behavioral logits всех
   statements; затем сохраняются четыре файла и освобождается модель.
2. Можно закрыть этот kernel/процесс.
3. Откройте `02_offline_analysis.ipynb`. Он не импортирует transformers и не
   загружает Gemma/DeBERTa. Первый запуск обучает CCS и сохраняет probes;
   следующие используют probes. Изменение behavioral threshold не обучает CCS.

Notebooks находят рабочий репозиторий при запуске из `~/projects`, корня
репозитория или вложенной папки. Исходный dataset читается из соседнего
`polarity-probing/data/`, extraction и CCS — из `polarity-probing/code/`.
Reference-код исполняется без записи `__pycache__` в этот репозиторий.
Checkpoint 1 остаётся неизменным.

## Артефакты

`results/checkpoint2/artifacts/<model_name>/`:

| Файл | Содержимое |
|---|---|
| `hidden_states.npz` | `X_pos`, `X_neg`: float32 `(N, n_layers, hidden_dim)` |
| `split.npz` | `train_idx`, `test_idx`, позиционные индексы исходных строк |
| `behavioral_logits.npz` | `sample_idx` и raw logits всех statements |
| `metadata.json` | модель, shape, dataset hash, extraction, split, scoring, token IDs/labels |
| `ccs_probes.npz` | derived cache всех выбранных слоёв, появляется после offline-обучения |

Для Gemma хранятся `logit_no`, `logit_yes`; для instruct-моделей дополнительно
`chat_logit_no`, `chat_logit_yes`, как в Checkpoint 1. Для DeBERTa:
`logit_class_0`, `logit_class_1`, mapping `0=non-hate`, `1=hate`.
Logits сохраняются до softmax. Default score: probability **>** 0.5, равенство
относится к классу 0. Gemma probability — softmax только по No/Yes.

Все слои включают embedding output (слой 0): Gemma 2B — 27×2304,
Gemma 9B — 43×3584, DeBERTa — 25×1024 на statement. Raw означает результат
**текущего** extractor до L2 и CCS preprocessing: его clipping/NaN cleanup
сохранены. `X_pos` извлекается из yes-dataset, `X_neg` — из no-dataset.

Split неизменен: `test_size=0.2`, `random_state=71`, `shuffle=True`.
На текущих данных N=1244: 995 train, 249 test. Все модели используют одинаковый
порядок. Behavioral logits покрывают все 1244 строки; test-метрики считают
только test. Значение `true_label=1` означает harmful (`1-is_harmfull_opposition`).

## Точное восстановление CCS

Default `preprocessing="checkpoint1"` сохраняет действующую последовательность:

1. L2-нормализация каждой пары sample/layer независимо для pos/neg.
2. Вычитание train-медианы отдельно для pos/neg, применение тех же медиан к test.
3. Исходный конструктор CCS дополнительно вычитает train mean внутри обучения.
4. Исходный `predict(..., predict_normalize=False)` получает данные после шага 2,
   **без повторного вычитания внутреннего среднего**.

Поэтому в `ccs_probes.npz` сохраняются `weight`, `bias`, `layer`,
`train_mean_pos`, `train_mean_neg` (внутренние средние шага 3),
`prediction_offset_pos`, `prediction_offset_neg` (медианы шага 2),
orientation и train accuracies, loss. `metadata_json` внутри NPZ содержит
preprocessing, seed, параметры оптимизации, layers, split, hashes hidden states,
датасета и reference CCS. Это позволяет точно воспроизвести существующее
поведение, включая различие preprocessing в train и prediction.

Ориентация latent score выбирается только по train. В `layer_metrics.csv`
`accuracy` — исходная sign-invariant CCS accuracy (`max(acc,1-acc)`),
`latent_vs_true` — test accuracy после выбора ориентации на train.

PA-CCS вызывает исходные `get_agreement` и `get_contradiction_idx`. Пары:
`A_idx = test_idx[test_idx >= N/2]`, `notA_idx = A_idx - N/2`.
Исходный `get_contrastive_probas` независимо mean-centers четыре подмножества.
Ответная строка пары может быть в train — это сохранённая original pairing
semantics. `polar_consistency_↓` — mean agreement,
`abs_agreement_score` — median absolute agreement,
`contradiction_idx_↓` — mean contradiction index. Per-pair таблицы не сохраняются.

## Offline API

Из папки notebook:

```python
from artifacts import ARTIFACTS, MODELS, load_dataset, load_cache
from offline import analyze, behavioral_scores, pa_probabilities

data = load_dataset()
spec = MODELS["gemma-2-2b"]
path = ARTIFACTS / spec.name

# Быстрый режим: compatible probes переиспользуются автоматически.
scores, metrics = analyze(path, spec, data, threshold=0.7)

# Только behavioral расчёт, без CCS.
behavior = behavioral_scores(load_cache(path, spec, data), threshold=0.7)

# Только per-pair PA probabilities, без обучения; использует saved config/split.
probabilities = pa_probabilities(path, spec, data, layer=20)

# Полный режим: новые probes из тех же raw hidden states.
scores, metrics = analyze(path, spec, data, preprocessing="mean", seed=42,
                          nepochs=1500, ntries=10, force_retrain=True)
```

Поддерживаются `checkpoint1`, `mean`, `median`, `l2`, `raw`; внутренний
mean-centering класса CCS сохраняется во всех режимах. `layers=[...]`, `lr`,
`weight_decay`, `nepochs`, `ntries`, `seed` настраиваются offline. По умолчанию
seed=0 соответствует seed, который устанавливал импорт исходного `ccs.py`.
При эксперименте с другим split можно передать `split=(train_idx,test_idx)`:
он должен быть полным непересекающимся разбиением. Сохранённый исходный
`split.npz` при этом не меняется; новая конфигурация хранится с derived probes.
Никакого автоматического pair-wise split нет.

Один `ccs_probes.npz` хранит одну текущую конфигурацию. При её изменении выводится
сообщение, CCS обучается offline и атомарно заменяет только derived cache.
Для сравнения конфигураций используйте разные `RUN_NAME` при экспорте CSV.

## Проверки и повреждённый cache

Проверяются model/shape/dataset content and order, dtype, finite values, исходный
split, sample_idx и logits. Probes проверяются на число слоёв, размеры weights,
means и offsets, конфигурацию и hash raw hidden states. При несовместимых или
частичных model artifacts возникает `CacheError` с путём: переместите весь
каталог модели в резервное место и запустите inference notebook снова.
Частичный cache не используется и не перезаписывается молча. Повреждённый
derived cache можно заменить через `force_retrain=True` без основной модели.

NPZ не сжат, чтобы не тратить время на компрессию больших массивов; hidden states
читаются в RAM. На пять моделей raw hidden states занимают около 4.56 GB
(десятичные единицы). Dtypes основных моделей совпадают с Checkpoint 1:
2B/2B-it float32, 9B bfloat16, 9B-it float16, DeBERTa float32. Модели
обрабатываются по очереди; веса 9B требуют соответствующего объёма памяти.

## Локальная проверка

Из корня рабочего репозитория:

```bash
../polarity-probing/.venv/bin/python -B -m unittest discover \
  -s notebooks/checkpoint2 -p 'test_*.py' -v
```

Тесты используют маленькие синтетические данные: cache roundtrip, неправильные
shape/split/model/dataset, единственная загрузка и release модели, raw logits,
повторное использование probes и совпадение с фактическим кодом Checkpoint 1
и оригинальными PA-CCS metrics. Это не заменяет полный запуск реальных моделей.
