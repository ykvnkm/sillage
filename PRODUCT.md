# Product

<!-- impeccable:product-schema 1 -->

Facts below are inferred from the repository (README, roadmap, model card, journal)
and the owner's brief for step 2.5; the owner works without design interviews by
agreement. Inferred items are marked *(inferred)*.

## Platform

web

## Users

- **Perfumers and perfumery students** checking what a raw material's main molecule
  is expected to smell of, and what known molecules it resembles. *(inferred from
  CONCEPT: "учебный инструмент для начинающих парфюмеров")*
- **Fragrance hobbyists** curious why vanillin smells of vanilla. Most will not know
  SMILES until name search lands in step 2.7. *(inferred)*
- **Technical reviewers** (ML engineers, hiring managers) judging whether the model
  is honest about what it knows. This is a portfolio project. *(confirmed by owner)*

## Product Purpose

Predict the odour of a molecule from its structure, and say how sure the prediction
is in words that mean something measured: "likely" is right about 89% of the time on
unseen molecules. Success: a person types a molecule and leaves knowing what it
probably smells of, how far to trust each word, and which known molecules it is
closest to.

## Positioning

Honest uncertainty as the product. Calibrated confidence words with measured hit
rates; descriptors too rare to calibrate are shown without numbers; molecules the
model was trained on are flagged so agreement is not mistaken for skill; inputs
outside the declared chemistry are refused with a reason instead of answered.

## Operating Context

- Input today is a SMILES string; name search comes in step 2.7.
- Answers arrive in about 10 ms from a CPU container; the UI is served by the same
  FastAPI app (no separate frontend build), later deployed to HF Spaces (step 2.6).
- Vocabulary: 138 odour descriptors from Goodscents/Leffingwell (fruity, woody,
  vanilla, musk...), confidence bands likely / probable / possible.

## Capabilities and Constraints

- Scope: single-component, neutral, organic molecules, 30-400 Da, elements C, N, O, S.
- 107 of 138 descriptors calibrated; 31 rank only and never carry a number.
- Stereochemistry is seen by the model's graph networks, but differences between
  isomers are unreliable: the experts it learnt from describe isomers
  inconsistently (journal E-020). The UI says so whenever the input has stereo.
- Nearest-neighbour similarity is orientation, not a confidence signal (journal E-008).
- Not yet available: molecule names, odour map, mixtures. Do not imply them.

## Evidence on Hand

- Real model output via `POST /predict`; model card at
  `docs/model-cards/odor-ensemble-v3.md` (macro AP 0.402 vs floor 0.038; 137 of 138
  descriptors calibrated).
- 4777 dataset molecules with expert labels (reference library).
- No testimonials, users, or usage numbers exist. Do not fabricate them.

## Product Principles

1. Never show a number that has not been measured to mean what it says.
2. A refusal is an answer: explain it in plain words.
3. Facts (expert labels) and predictions are always visually distinct.
4. The molecule itself is the hero, not the chrome around it.

## Accessibility & Inclusion

WCAG 2.2 AA contrast and full keyboard use *(inferred default)*. Confidence must
never be encoded by colour alone.
