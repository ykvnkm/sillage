"""Train the shipping model and write the artefact plus its model card.

    uv run python scripts/train_model.py

Weights go to models/<name>/ (not committed). The human-readable model card goes to
docs/model-cards/<name>.md (committed), so the claims about the model are reviewable
even by someone who never downloads it.
"""

from __future__ import annotations

import argparse
import json

from sillage.models.artifact import OdorModel
from sillage.paths import project_root
from sillage.research.train import (
    DEFAULT_FEATURE_SET,
    DEFAULT_NAME,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    train_odor_model,
)

CARD_TEMPLATE = """# Model card: {name}

Автоматически сгенерировано `scripts/train_model.py`. Не редактировать руками —
правки перезапишутся при следующем обучении. Время обучения сюда намеренно не
пишется (оно есть в `card.json` рядом с весами): иначе переобучение той же модели с
тем же сидом меняло бы этот файл, и в истории git появлялись бы пустые правки.

## Что эта модель делает

По структуре молекулы предсказывает, какие из {n_labels} ольфакторных дескрипторов
ей свойственны. Выход — {n_labels} независимых оценок; по {n_calibrated} из них это
откалиброванные вероятности (раздел «Калибровка»), по остальным — только порядок.

## Область определения

Модель определена на **однокомпонентных нейтральных органических молекулах**
массой {min_weight:.0f}–{max_weight:.0f} Da, состоящих только из элементов
{elements}. Сама модель вне области не отказывается — отказывает сценарий
`PredictOdor` до её вызова, так что числа о молекуле вне области не возникает вообще.

Основание: на входах вне области модель выдаёт уверенные и неверные ответы. Карбонат
магния получает 1.000 за семь дескрипторов и 0.000 за свою единственную метку
`odorless`. Вне области определения молекула примерно в шесть раз чаще оказывается
катастрофической ошибкой. Подробности — журнал E-007 и E-008.

## Данные

| | |
| --- | --- |
| источник | `{data_source}` |
| контрольная сумма | `{data_sha256}` |
| разбиение | {split_strategy}, сид {seed} |
| обучено на | {n_train} молекулах |
| измерено на | {n_test} молекулах |

## Качество

Рядом с каждой метрикой — **пол**: что показывает предсказатель, который не видел ни
одной молекулы и знает только частоты дескрипторов. Без пола эти числа нечитаемы:
micro AUC начинается с 0.78 просто от ранжирования частых дескрипторов выше редких.

| Метрика | Модель | Пол | Во сколько раз выше |
| --- | --- | --- | --- |
| macro AP | **{macro_ap:.4f}** | {macro_ap_floor:.4f} | ×{macro_ap_lift:.1f} |
| macro AUC | {macro_auc:.4f} | {macro_auc_floor:.4f} | — |
| micro AP | {micro_ap:.4f} | {micro_ap_floor:.4f} | ×{micro_ap_lift:.1f} |
| micro AUC | {micro_auc:.4f} | {micro_auc_floor:.4f} | — |
{consensus_row}
Измерено по {labels_scored:.0f} из {labels_total:.0f} дескрипторов. Основная метрика
проекта — **macro AP**: только у неё пол честно низкий.

## Калибровка

Выход **откалиброван** по каждой метке: Platt scaling на предсказаниях для молекул,
которых модель при обучении не видела
([ADR-0011](../adr/0011-calibration-is-per-label-and-refuses-weak-labels.md)).
Откалибровано {n_calibrated} меток из {n_labels}. Для остальных ({n_uncalibrated})
слишком мало примеров: по ним выдаётся не вероятность, а только порядок, и продукт
показывает пометку «без проверки».

{calibration_check}

{ceiling}## Известные ограничения

1. **По {n_uncalibrated} дескрипторам вероятности нет** — только ранжирование.
2. **Стереоизомеры не различаются.** Зеркальные изомеры пахнут по-разному (карвон:
   тмин против мяты), но признаки модели их не различают. Учёт хиральности в
   отпечатке Моргана роста не дал (журнал E-019): различить их сможет только
   представление, которое видит граф молекулы целиком.
3. **Разметка субъективна.** Дескрипторы — мнение экспертов, не измерение, и на
   расплывчатых словах (`fresh`, `sweet`, `warm`, `sharp`) эксперты двух баз
   согласны друг с другом слабо. Там слабее и модель.
4. **«Во сколько раз лучше случайного» обманчиво для частых нот.** Этот множитель
   ограничен сверху единицей, делённой на частоту ноты: у `fruity` (40% молекул)
   потолок — 2.5, и модель с её 2.0 берёт 80% возможного, при AP 0.79. Сравнивать
   ноты по нему нельзя; по AP и по согласию экспертов — можно.
"""

CEILING_TEMPLATE = """## Потолок: насколько согласны сами эксперты

Метки обучения взяты из двух независимых экспертных баз, Goodscents и Leffingwell.
На {molecules} молекулах, описанных обеими и не виденных моделью ни при обучении,
ни при калибровке, по {descriptors} нотам с достаточным числом примеров
(каппа Коэна: 0 — случайное совпадение, 1 — полное):

| Кто с кем | Каппа |
| --- | --- |
| эксперты Goodscents ↔ эксперты Leffingwell | **{experts:.2f}** |
| модель ↔ Leffingwell | {model_vs_leffingwell:.2f} |
| модель ↔ Goodscents | {model_vs_goodscents:.2f} |

Модель согласна с одной базой примерно на {share:.0%} того, насколько с ней согласна
другая база экспертов. По отдельным нотам качество модели следует за согласием
экспертов (ранговая корреляция {spearman:.2f}): где эксперты расходятся, модели не на
что опереться. Подробности — журнал E-018, `results/annotator_ceiling_{model}.csv`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dataset",
        choices=("openpom", "v2"),
        default="openpom",
        help="v2: the two expert collections rebuilt with soft, masked labels (E-019)",
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="re-render docs/model-cards/<name>.md from the saved artifact, no training",
    )
    args = parser.parse_args()

    if args.card_only:
        root = project_root()
        name = (
            args.name
            if args.name != DEFAULT_NAME or args.dataset == "openpom"
            else "odor-baseline-v2"
        )
        model = OdorModel.load(root / "models" / name)
        card_path = root / "docs" / "model-cards" / f"{name}.md"
        card_path.write_text(_render_card(_Saved(model)), encoding="utf-8")
        print(f"wrote {card_path.relative_to(root)}")
        return 0

    if args.dataset == "v2":
        from sillage.research.train_model_v2 import DEFAULT_NAME as V2_NAME
        from sillage.research.train_model_v2 import train_odor_model_v2

        name = V2_NAME if args.name == DEFAULT_NAME else args.name
        outcome = train_odor_model_v2(name=name, feature_set=args.feature_set, seed=args.seed)
        args.name = name
    else:
        outcome = train_odor_model(
            name=args.name,
            feature_set=args.feature_set,
            split_strategy=args.split,
            seed=args.seed,
        )
    print(outcome.summary())

    root = project_root()
    directory = root / "models" / args.name
    outcome.model.save(directory)
    size = sum(f.stat().st_size for f in directory.iterdir()) / 1e6
    print(f"\nwrote {directory.relative_to(root)}  ({size:.1f} MB)")

    reloaded = OdorModel.load(directory)
    print(
        f"reload check: {len(reloaded.label_names)} labels, "
        f"{len(reloaded.reference)} reference molecules"
    )

    card_path = root / "docs" / "model-cards" / f"{args.name}.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(_render_card(outcome), encoding="utf-8")
    print(f"wrote {card_path.relative_to(root)}")
    return 0


class _Saved:
    """A loaded artifact in the shape _render_card expects from a training outcome."""

    def __init__(self, model: OdorModel) -> None:
        self.model = model


def _render_ceiling(model_name: str) -> str:
    """The ceiling section, only if it was measured for this very model."""
    path = project_root() / "results" / f"annotator_ceiling_{model_name}.json"
    if not path.exists():
        return ""
    summary = json.loads(path.read_text("utf-8"))
    if summary.get("model") != model_name:
        return ""
    return (
        CEILING_TEMPLATE.format(
            model=model_name,
            molecules=summary["molecules"],
            descriptors=summary["descriptors"],
            experts=summary["experts"],
            model_vs_leffingwell=summary["model_vs_leffingwell"],
            model_vs_goodscents=summary["model_vs_goodscents"],
            share=summary["model_vs_leffingwell"] / summary["experts"],
            spearman=summary["spearman_experts_vs_model_ap"],
        )
        + "\n"
    )


def _render_card(outcome) -> str:
    card = outcome.model.card
    metrics, floor = card.metrics, card.baseline_metrics
    if "calibration_likely_promised" in metrics:
        calibration_check = (
            "Проверено на тестовом сплите: в диапазоне 0.85–1.00 модель обещает "
            f"{metrics['calibration_likely_promised']:.3f} и оказывается права в "
            f"{metrics['calibration_likely_observed']:.3f} случаев "
            f"({metrics['calibration_likely_cells']:.0f} пар «молекула–нота»)."
        )
    else:  # odor-baseline-v1, measured by hand in journal E-011
        calibration_check = (
            "Проверено на тестовом сплите: в диапазоне 0.85–1.00 модель обещает 0.903 и\n"
            "оказывается права в 0.889 случаев. До калибровки обещала 0.942 при реальных 0.826."
        )
    consensus_row = ""
    if "consensus_macro_average_precision" in metrics:
        consensus_row = (
            f"| macro AP, консенсус экспертов ({metrics['consensus_labels_scored']:.0f} нот) | "
            f"**{metrics['consensus_macro_average_precision']:.4f}** | "
            f"{floor['consensus_macro_average_precision']:.4f} | — |\n"
        )
    return CARD_TEMPLATE.format(
        calibration_check=calibration_check,
        consensus_row=consensus_row,
        ceiling=_render_ceiling(card.name),
        name=card.name,
        n_labels=len(card.label_names),
        n_calibrated=outcome.model.calibrator.n_fitted if outcome.model.calibrator else 0,
        n_uncalibrated=(
            len(card.label_names) - outcome.model.calibrator.n_fitted
            if outcome.model.calibrator
            else len(card.label_names)
        ),
        min_weight=card.scope["min_weight"],
        max_weight=card.scope["max_weight"],
        elements=", ".join(card.scope["allowed_elements"] or ()),
        data_source=card.data_source,
        data_sha256=card.data_sha256,
        split_strategy=card.split_strategy,
        seed=card.seed,
        n_train=card.n_train,
        n_test=card.n_test,
        macro_ap=metrics["macro_average_precision"],
        macro_ap_floor=floor["macro_average_precision"],
        macro_ap_lift=metrics["macro_average_precision"] / floor["macro_average_precision"],
        macro_auc=metrics["macro_roc_auc"],
        macro_auc_floor=floor["macro_roc_auc"],
        micro_ap=metrics["micro_average_precision"],
        micro_ap_floor=floor["micro_average_precision"],
        micro_ap_lift=metrics["micro_average_precision"] / floor["micro_average_precision"],
        micro_auc=metrics["micro_roc_auc"],
        micro_auc_floor=floor["micro_roc_auc"],
        labels_scored=metrics["labels_scored"],
        labels_total=metrics["labels_total"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
