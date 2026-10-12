# Design

The visual system of the sillage web interface (`src/sillage/interfaces/web/`).
Product truth lives in [PRODUCT.md](PRODUCT.md).

## World

A perfumery lab's raw-material library: amber glass bottles on a lab shelf, each
with a typed label. The world lends four things and only these: palette, type,
density, and one signature move. Layout, navigation and controls are a standard
working web tool.

**Signature move: the answer is a material label.** The molecule sits on a label
card under an amber band carrying its formula and weight; each predicted note is a
row with a confidence grade and a meter whose tick marks the note's background
rate, so "far past the tick" reads as "the model sees something".

## Palette

| Token | Value | Job |
| --- | --- | --- |
| `--amber-900` | `#2b1709` | shell bar, footer |
| `--amber-800` | `#4a2810` | label band, primary button |
| `--amber-700` | `#6e3a10` | "likely" grade and meter, links |
| `--amber-400` | `#d08a32` | "probable" meter |
| `--amber-300` | `#e6ae62` | "possible" meter |
| `--amber-200` / `--amber-100` | `#f2d3a3` / `#f8e9d2` | "probable" grade fill, caveats, neighbour bands |
| `--ink` / `--ink-2` / `--ink-3` | `#26190f` / `#5b4535` / `#6b5646` | text, secondary, tertiary (all >= 4.5:1 on ground) |
| `--paper` | `#fdfcf9` | label stock: every card |
| `--ground` | `#e5e6e2` | the shelf: page background, meter track |
| `--refuse` | `#7d2213` | refusal stamp and detail |

Confidence is always a word as well as a colour. Molecule drawings use the ink
colour only, so element colours never compete with confidence colours.

## Type

The interface is in Russian, so faces need full Cyrillic. Fira Sans Condensed
(600-700) for headings and every label-like element: grades, row labels, table
heads; uppercase small labels track at +0.08 to +0.12em. Fira Sans (400-600) for
body text. SMILES strings are the only monospace, because they are data, and are
marked `lang="en"`. Numerals tabular; decimals use the Russian comma. Molecular mass is shown in г/моль, not
daltons: the number is identical, but «Да» on a Russian page reads as "yes".

## Language

All visible text is Russian, including the 138 descriptor names
(`static/descriptors.js`; a test fails on any untranslated descriptor). The API
stays English: it is a machine contract. Confidence words, strongest first:
«Очень вероятно» (likely), «Вероятно» (probable), «Возможно» (possible),
«Без проверки» (unverified).

## Shape and depth

Cards: 12px radius, 1px `--rule` border, short shadow
(`0 1px 2px` + `0 3px 8px -4px`). Grades: 4px radius. Chips and tags: pill.
Elevation is declared once: border plus a short shadow, never a wide diffuse one.

## Motion

One authored moment: on each answer the meters reveal left to right with
`clip-path`, 700ms exponential ease-out, 40ms stagger. Nothing else animates
beyond hover and the button's busy spinner. `prefers-reduced-motion` disables it.

## Numbers

Percentages are rounded **down**. Band edges are whole percentages, so flooring
keeps number and word consistent: 0.846 is «Вероятно 84%», never «Вероятно 85%»
next to an «Очень вероятно from 85%» table.

## Search

One field takes a name or a SMILES string. Suggestions open under it as a WAI-ARIA
combobox (arrow keys, Enter, Escape; focus never leaves the input); each option shows
the name to display and, in tertiary ink, the synonym that matched. The label band
leads with the name the user typed when it is a trade name or synonym ("Iso-E super"),
with the systematic name under it in the band's secondary colour.

Known detector false positive: `cramped-padding` on `.query__field` -- the
`.combobox` wrapper touches the field's edge, but the text sits 16px inside it through
the input's own padding.

## Odour map

A canvas scatter of the 4777 reference molecules, positions from the model's
embedding (UMAP, offline), families from experts' labels. **One family is highlighted
at a time** (`#a6581a`, validated: lightness, chroma, >= 3:1 on the paper surface);
everything else is recessive grey `#c9c9c1`. Eight coloured families at once would
break the scatter-plot rule of at most three categorical colours distinguishable
under colour-vision deficiency. Family chips act as the legend and the selector;
the default is the family of the molecule's most probable note.

The current molecule is a shape, not a colour: an ink ring with a surface halo and a
direct label; its nearest known molecules get small ink rings. Hover shows name and
family; click predicts for that molecule. Dot radius scales with map width
(1.1-2.3 px); map height follows the layout's own aspect ratio. On phones the family
chips become one horizontally scrolling row.
