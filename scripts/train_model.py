"""Train the shipping model and write the artefact plus its model card.

    uv run python scripts/train_model.py

Weights go to models/<name>/ (not committed). The human-readable model card goes to
docs/model-cards/<name>.md (committed), so the claims about the model are reviewable
even by someone who never downloads it.
"""

from __future__ import annotations

import argparse

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

Автоматически сгенерировано `scripts/train_model.py` {created_at}.
Не редактировать руками — правки перезапишутся при следующем обучении.

## Что эта модель делает

По структуре молекулы предсказывает, какие из {n_labels} ольфакторных дескрипторов
ей свойственны. Выход — {n_labels} независимых оценок, не вероятности: калибровки
пока нет, и показывать эти числа пользователю напрямую нельзя.

## Область определения

Модель определена на **однокомпонентных нейтральных органических молекулах**
массой {min_weight:.0f}–{max_weight:.0f} Da. Вне этой области она не отказывается
отвечать сама — отказ обеспечивает вызывающий код.

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

Измерено по {labels_scored:.0f} из {labels_total:.0f} дескрипторов. Основная метрика
проекта — **macro AP**: только у неё пол честно низкий.

## Известные ограничения

1. **Выход не калиброван.** 0.9 не означает «в 90% случаев прав».
2. **Стереохимия отсутствует в данных.** Колонка источника называется
   `nonStereoSMILES`. Зеркальные изомеры пахнут по-разному (карвон: тмин против
   мяты), и модель их не различает в принципе.
3. **Разметка субъективна.** Дескрипторы — мнение экспертов, не измерение. Потолок
   согласия между независимыми источниками пока не измерен, поэтому неизвестно,
   какую долю достижимого модель берёт.
4. **Качество сильно зависит от дескриптора.** Узкие химические семейства
   (`clove`, `popcorn`) предсказываются в 100+ раз лучше случайного; зонтичные
   термины (`fresh`, `sweet`, `green`) — едва в 2 раза.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

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
        f"{len(reloaded.reference_smiles)} reference molecules"
    )

    card_path = root / "docs" / "model-cards" / f"{args.name}.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(_render_card(outcome), encoding="utf-8")
    print(f"wrote {card_path.relative_to(root)}")
    return 0


def _render_card(outcome) -> str:
    card = outcome.model.card
    metrics, floor = card.metrics, card.baseline_metrics
    return CARD_TEMPLATE.format(
        name=card.name,
        created_at=card.created_at,
        n_labels=len(card.label_names),
        min_weight=card.scope["min_weight"],
        max_weight=card.scope["max_weight"],
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
