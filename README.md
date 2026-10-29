# Latent–Behavior Alignment in Language Models

Сравнение поведенческих оценок вредоносности с сигналами из скрытых представлений
языковых моделей: CCS, PA-CCS и случаи расхождения между двумя оценками.
Исследуются датасеты `not` и `mixed`, пять моделей и два протокола разбиения:
row-wise и pair-wise. Датасеты анализируются отдельно.

## Результаты

Выполненные ноутбуки содержат сохранённые таблицы и графики. Для просмотра на
GitHub установка библиотек и загрузка данных не нужны.

- [Gemma 2 2B](notebooks/01_gemma_2_2b.ipynb)
- [Gemma 2 2B Instruct](notebooks/02_gemma_2_2b_it.ipynb)
- [Gemma 2 9B](notebooks/03_gemma_2_9b.ipynb)
- [Gemma 2 9B Instruct](notebooks/04_gemma_2_9b_it.ipynb)
- [DeBERTa hate classifier](notebooks/05_deberta_hate_tuned.ipynb)
- [Агрегация по глубине и размеру моделей](notebooks/06_aggregation.ipynb)

Сохранённые выводы получены до переноса в эту структуру; встречающиеся в них
исходные пути относятся к тому запуску. Код ячеек использует новую структуру.

## Структура

```text
README.md
requirements.txt
notebooks/                 # выполненные ноутбуки с результатами
modules/
    __init__.py
    experiments.py         # эксперименты, метрики, загрузка результатов
    plots.py               # визуализация
    artifacts.py           # спецификации моделей, данные, проверка кэша
    ccs.py                 # исходная реализация CCS
```

Модули используют локальные импорты и не требуют соседнего репозитория.

## Установка

Python 3.12. Из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

## Повторное выполнение

Выполненные ноутбуки загружают уже рассчитанные результаты и не обучают probes
заново. Исходные CSV, hidden states, logits и результаты **не включены** в эту
публикацию. Для выполнения всех ячеек подготовьте:

```text
data/
    raw/
        mixed_dataset.csv
        not_hate_dataset.csv
    yes_no/
        mixed_dataset_yes.csv
        mixed_dataset_no.csv
        not_dataset_yes.csv
        not_dataset_no.csv
    mixed/artifacts/<model>/
        hidden_states.npz
        split.npz
        behavioral_logits.npz
        metadata.json
    not/artifacts/<model>/
        ...
results/
    saved_full/
        mixed/<model>/rowwise/
        mixed/<model>/pairwise/
        not/<model>/rowwise/
        not/<model>/pairwise/
```

Каждая папка протокола должна содержать полные сохранённые результаты, включая
`manifest.json`, CSV и `probes/`. Сохраняйте их содержимое: при загрузке проверяются
контрольные суммы. Геометрические графики в модельных ноутбуках дополнительно
используют исходные CSV и hidden states; агрегация использует сохранённые результаты.

Для переноса из исходного рабочего проекта: содержимое
`results/model_experiments/saved_full/` положите в `results/saved_full/`,
`results/checkpoint2/{mixed,not}/artifacts/` — в `data/{mixed,not}/artifacts/`,
а `data/raw/` и `data/yes_no/` из `polarity-probing` — в одноимённые папки `data/`.
Абсолютные пути в старых метаданных могут отличаться; содержимое и порядок строк
датасета по-прежнему проверяются по fingerprint.

Новый запуск обучения доступен отдельно от выполненных ноутбуков:

```python
from modules.experiments import run_dataset

results = run_dataset("gemma-2-2b", "not", device="cpu")
```

По умолчанию новый запуск пишет в `results/full/`; ноутбуки читают
`results/saved_full/`. Для GPU задайте `device="cuda"` и установите подходящую
сборку PyTorch. Извлечение представлений языковой моделью в эту сборку не входит.
Папки `data/` и `results/` исключены из Git через `.gitignore`.

## Метод

Для каждого split/fold Yes- и No-представления независимо нормализуются по L2.
Два арифметических средних вычисляются только на train, затем вычитаются из
train и test. Дополнительного центрирования внутри CCS нет. Основной запуск:
1500 эпох, 10 инициализаций, AdamW, learning rate 0.015, weight decay 0.01.
Ориентация probe и поведенческий порог выбираются только на train.

Pair-wise CV удерживает оба участника семантической пары в одном fold.
В row-wise протоколе участники пары могут попасть в разные выборки; PA-метрики
сохраняют исходное правило выбора пар. Величина `counterpart_outside_split`
показывает число партнёров за пределами оцениваемой выборки.

Здесь alignment означает согласие оценки вредоносности с меткой датасета.
Behavioral scores относятся к заданию классификации, а latent scores — к
сигналу линейного probe.

## Источники

Датасеты и исходный CCS:
[polarity-probing](https://github.com/SadSabrina/polarity-probing).
Файл `modules/ccs.py` скопирован без изменений из `code/ccs.py` локальной рабочей
копии этого проекта. Остальные модули адаптированы из текущего исследовательского
проекта; определения метрик и алгоритм обучения сохранены.
