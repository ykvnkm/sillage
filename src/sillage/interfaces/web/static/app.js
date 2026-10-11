// sillage web interface. No framework, no build step: the page is served by the
// same FastAPI app that answers /predict, and every string from the API is placed
// with textContent, never as HTML.

const DEFAULT_SMILES = "COc1cc(C=O)ccc1O";

const REFUSALS = {
  invalid_input: {
    title: "Эту строку не получается прочитать как структуру.",
    body: "Проверьте SMILES на опечатки, незакрытые циклы и скобки. Надёжнее всего скопировать строку из поля «Isomeric SMILES» на PubChem.",
  },
  multiple_components: {
    title: "Это соль или смесь, а не одна молекула.",
    body: "Модель читает отдельные молекулы. Соли почти не испаряются и поэтому редко пахнут; попробуйте нейтральное исходное соединение.",
  },
  inorganic: {
    title: "В молекуле нет углерода: это неорганика.",
    body: "Модель обучена только на органических душистых веществах. Любой ответ здесь был бы догадкой, выданной за знание.",
  },
  charged: {
    title: "У молекулы есть заряд.",
    body: "Ионы не летучи и до носа не долетают. Попробуйте нейтральную форму молекулы.",
  },
  unseen_element: {
    title: "Модель никогда не видела этот элемент.",
    body: "Ни в одной обучающей молекуле его не было, поэтому модель ничего не знает о том, как он меняет запах. На таких молекулах она ошибалась гораздо чаще, поэтому отказывается.",
  },
  weight_out_of_range: {
    title: "Масса вне диапазона, для которого модель отвечает.",
    body: "Молекулы тяжелее примерно 400 г/моль слишком плохо испаряются, чтобы долететь до носа. Модель отвечает для масс от 30 до 400 г/моль.",
  },
};

const GRADE_LABEL = {
  likely: "Очень вероятно",
  probable: "Вероятно",
  possible: "Возможно",
  unverified: "Без проверки",
};

const form = document.getElementById("query-form");
const input = document.getElementById("smiles");
const submit = document.getElementById("submit");
const result = document.getElementById("result");

let controller = null;

// ---- tiny DOM helper ---------------------------------------------------------

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.setAttribute("style", value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function drawing(smiles, width, height, alt) {
  const src = `/depict?${new URLSearchParams({ smiles, width, height })}`;
  return h("img", { src, alt, width, height, loading: "lazy", decoding: "async" });
}

// C8H8O3 -> C<sub>8</sub>H<sub>8</sub>O<sub>3</sub>, built as nodes.
function formula(text) {
  const span = h("span", { class: "label__formula" });
  for (const part of text.match(/[A-Z][a-z]?|\d+|[^A-Za-z\d]+/g) || []) {
    span.append(/^\d+$/.test(part) ? h("sub", {}, part) : part);
  }
  return span;
}

// Rounded down, never to nearest: 0.846 is "probable", and showing it as 85% would
// put the number on the "likely" side of an edge the word sits below.
function percent(p) {
  return `${Math.floor(p * 100 + 1e-9)}%`;
}

const decimal = (value, digits) =>
  value.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });

// Grams per mole, not daltons: the number is the same, but "Да" on a Russian page
// reads as "yes", while г/моль is the unit everyone met in school chemistry.
function molarMass(weight) {
  return `${decimal(weight, 1)}\u00a0г/моль`;
}

// The API's `detail` is English and meant for logs; the page rebuilds it in Russian
// from the stable reason code, taking only the facts (elements, weight) from it.
function refusalDetail(refusal) {
  const detail = refusal.detail || "";
  switch (refusal.reason) {
    case "unseen_element": {
      const found = detail.match(/contains (.+), absent/);
      return found ? `Содержит ${found[1]}: таких атомов нет ни в одной обучающей молекуле.` : "Содержит элемент, которого нет в обучающих данных.";
    }
    case "weight_out_of_range": {
      const found = detail.match(/molecular weight (\d+)/);
      return found ? `Молекулярная масса ${found[1]}\u00a0г/моль, а модель отвечает для 30–400\u00a0г/моль.` : "Масса вне диапазона 30–400 г/моль.";
    }
    case "multiple_components":
      return "Несколько несвязанных частей: соль или смесь.";
    case "inorganic":
      return "В структуре нет ни одного атома углерода.";
    case "charged":
      return "У атомов в структуре есть формальный заряд.";
    default:
      if (/not a valid SMILES/.test(detail)) return "Химическая библиотека RDKit не смогла разобрать эту строку.";
      if (/at most|too long|characters long/.test(detail)) return "Строка длиннее 500 символов.";
      return "Пустая строка или запрос не прошёл проверку.";
  }
}

// ---- request -----------------------------------------------------------------

async function predict(smiles, { push = true } = {}) {
  smiles = smiles.trim();
  if (!smiles) {
    input.focus();
    return;
  }
  input.value = smiles;
  if (push) {
    const url = new URL(location.href);
    url.searchParams.set("smiles", smiles);
    history.pushState({ smiles }, "", url);
  }

  controller?.abort();
  controller = new AbortController();
  setBusy(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ smiles, neighbours: 5 }),
      signal: controller.signal,
    });
    const body = await response.json().catch(() => null);
    if (body && body.refused) render(renderRefusal(body));
    else if (response.ok && body) render(renderReport(body));
    else render(renderProblem(`Сервис ответил кодом ${response.status}. Попробуйте ещё раз через минуту.`));
  } catch (error) {
    if (error.name === "AbortError") return;
    render(renderProblem("Не удалось связаться с сервисом. Проверьте соединение и попробуйте ещё раз."));
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  submit.setAttribute("aria-busy", String(busy));
  result.setAttribute("aria-busy", String(busy));
}

function render(content) {
  result.replaceChildren(...[content].flat());
  result.hidden = false;
  // One authored moment: the meters grow from zero once the label is on screen.
  result.dataset.state = "entering";
  requestAnimationFrame(() => requestAnimationFrame(() => (result.dataset.state = "shown")));
}

// ---- report ------------------------------------------------------------------

function renderReport(report) {
  const m = report.molecule;
  const known = report.known;
  const knownSet = new Set(known ? known.descriptors : []);

  const label = h(
    "article",
    { class: "label", "aria-label": "Молекула" },
    h("div", { class: "label__band" }, formula(m.formula), h("span", { class: "label__weight" }, `масса ${molarMass(m.weight)}`)),
    h("figure", { class: "label__structure" }, drawing(m.smiles, 480, 360, `Структура ${m.formula}`)),
    h(
      "dl",
      { class: "label__rows" },
      h("div", {}, h("dt", {}, "SMILES"), h("dd", {}, h("code", { lang: "en" }, m.smiles))),
      m.input_smiles !== m.smiles &&
        h("div", {}, h("dt", {}, "Введено"), h("dd", {}, h("code", { lang: "en" }, m.input_smiles))),
      h("div", {}, h("dt", {}, "В данных"), h("dd", {}, knownLine(known))),
    ),
    report.caveats.includes("stereo_discarded") &&
      h(
        "p",
        { class: "caveat", role: "note" },
        h("strong", {}, "Стереохимия не учтена. "),
        "Модель не различает зеркальные молекулы, поэтому обе формы получают этот же ответ.",
      ),
  );

  const predicted = report.predicted;
  const readout = h(
    "section",
    { class: "readout", "aria-labelledby": "readout-title" },
    h("h2", { id: "readout-title" }, "Предсказанный запах"),
    h(
      "p",
      { class: "readout__sub" },
      predicted.length ? "Ноты, которым модель даёт хотя бы один шанс из трёх; самые вероятные сверху." : "",
    ),
    predicted.length
      ? h("ol", { class: "notes" }, predicted.map((d, i) => noteRow(d, i)))
      : h("p", { class: "empty-note" }, "Ни одна нота не дотягивает до «возможно». Уверенного мнения об этой молекуле у модели нет."),
    predicted.length > 0 &&
      h(
        "p",
        { class: "legend", "aria-hidden": "true" },
        h("span", {}, h("i", { class: "legend__bar" }), "шанс, что эксперт назовёт ноту"),
        h("span", {}, h("i", { class: "legend__tick" }), "как часто её называют у любой молекулы"),
      ),
    report.unverified.length > 0 &&
      h(
        "div",
        { class: "subsection" },
        h("h3", {}, "Склоняется, но без проверки"),
        h("p", {}, "Ноты, слишком редкие в данных, чтобы измерить на них модель. Она склоняется к ним, но не может сказать, как часто права, поэтому числа нет."),
        h("ul", { class: "tags" }, report.unverified.map((d) => h("li", { class: "tag tag--unverified" }, noteName(d.descriptor)))),
      ),
    known && recordBlock(known, predicted, knownSet),
  );

  const neighbours = report.neighbours.length
    ? h(
        "section",
        { class: "neighbours", "aria-labelledby": "neighbours-title" },
        h("h2", { id: "neighbours-title" }, "Ближайшие известные молекулы"),
        h("p", {}, "Молекулы из датасета с самой похожей структурой и то, что записали для них эксперты. Похожая структура не гарантирует похожий запах; нажмите на карточку, чтобы увидеть предсказание для неё."),
        h("ul", { class: "neighbours__list" }, report.neighbours.map(specimen)),
      )
    : null;

  return [label, readout, neighbours].filter(Boolean);
}

function knownLine(known) {
  if (!known) return "Нет: это настоящее предсказание.";
  return known.seen_in_training ? "Есть, и модель на ней обучалась" : "Есть, но в обучение не входила";
}

function noteRow(d, index) {
  const grade = d.confidence;
  return h(
    "li",
    { class: "note" },
    h("span", { class: "note__name" }, noteName(d.descriptor)),
    h("span", { class: `grade grade--${grade}` }, GRADE_LABEL[grade] || grade),
    h(
      "span",
      {
        class: "meter",
        role: "img",
        "aria-label": `шанс ${percent(d.probability)}; у всех молекул встречается в ${percent(d.base_rate)}`,
        style: `--i: ${index}`,
      },
      h("span", { class: `meter__fill grade-fill--${grade}`, style: `--p: ${d.probability}; --i: ${index}` }),
      h("span", { class: "meter__base", style: `--b: ${d.base_rate}` }),
    ),
    h("span", { class: "note__pct" }, percent(d.probability)),
  );
}

function recordBlock(known, predicted, knownSet) {
  const predictedSet = new Set(predicted.map((d) => d.descriptor));
  const hits = known.descriptors.filter((d) => predictedSet.has(d)).length;
  return h(
    "section",
    { class: "record", "aria-labelledby": "record-title" },
    h("h3", { id: "record-title" }, "Что записали эксперты", h("span", { class: "record__source" }, "запись из датасета, не предсказание")),
    known.descriptors.length
      ? h(
          "ul",
          { class: "tags" },
          known.descriptors.map((d) =>
            h("li", { class: `tag${predictedSet.has(d) ? " tag--match" : ""}`, title: predictedSet.has(d) ? "Тоже предсказано" : null }, noteName(d)),
          ),
        )
      : h("p", { class: "record__note" }, "Записана без описаний запаха."),
    h(
      "p",
      { class: "record__note" },
      `${hits} из ${known.descriptors.length} записанных нот тоже предсказаны (обведены). `,
      known.seen_in_training
        ? h("strong", {}, "Модель обучалась на этой молекуле, поэтому совпадение здесь — память, а не доказательство умения.")
        : "При обучении модель эту молекулу не видела, так что совпадение здесь — честная проверка.",
    ),
  );
}

function specimen(n) {
  const notes = n.descriptors.length ? n.descriptors.slice(0, 5).map(noteName).join(", ") : "описаний нет";
  return h(
    "li",
    {},
    h(
      "button",
      {
        type: "button",
        class: "specimen",
        "aria-label": `Предсказать для ${n.smiles}: сходство ${percent(n.similarity)}, эксперты: ${notes}`,
        onclick: () => {
          predict(n.smiles);
          document.getElementById("query-title").scrollIntoView({ behavior: "smooth" });
        },
      },
      h("span", { class: "specimen__band" }, h("strong", {}, percent(n.similarity)), "сходство"),
      h("span", { class: "specimen__structure" }, drawing(n.smiles, 300, 225, "")),
      h("code", { class: "specimen__smiles", lang: "en" }, n.smiles),
      h("span", { class: "specimen__notes" }, notes),
    ),
  );
}

// ---- refusal and problems ----------------------------------------------------

function renderRefusal(refusal) {
  const copy = REFUSALS[refusal.reason] || { title: "Модель отказывается отвечать.", body: "" };
  const m = refusal.molecule;
  return h(
    "div",
    { class: "refusal" },
    h(
      "section",
      { class: "refusal__card", "aria-labelledby": "refusal-title" },
      h("span", { class: "stamp" }, "Без ответа"),
      h("h2", { id: "refusal-title" }, copy.title),
      h("p", {}, copy.body),
      h("p", { class: "refusal__detail" }, refusalDetail(refusal)),
    ),
    m &&
      h(
        "article",
        { class: "label", "aria-label": "Молекула" },
        h("div", { class: "label__band" }, formula(m.formula), h("span", { class: "label__weight" }, `масса ${molarMass(m.weight)}`)),
        h("figure", { class: "label__structure" }, drawing(m.smiles, 480, 360, `Структура ${m.formula}`)),
      ),
  );
}

function renderProblem(message) {
  return h("p", { class: "problem", role: "alert" }, message);
}

// ---- model facts -------------------------------------------------------------

async function loadModelFacts() {
  try {
    const response = await fetch("/model");
    if (!response.ok) return;
    const model = await response.json();
    const ap = model.metrics.macro_average_precision;
    const floor = model.floor.macro_average_precision;
    const facts = {
      name: model.name,
      trained_on: `${model.trained_on.toLocaleString("ru-RU")} молекулах`,
      quality: `macro AP ${decimal(ap, 3)} против ${decimal(floor, 3)} у угадывания по частотам: в ${decimal(ap / floor, 1)} раза лучше`,
    };
    for (const [key, value] of Object.entries(facts)) {
      const node = document.querySelector(`[data-fact="${key}"]`);
      if (node) node.textContent = value;
    }
    document.getElementById("unverified-count").textContent = model.unverified.length;
  } catch {
    // The static text in the page is already correct for the shipped model.
  }
}

// ---- wiring ------------------------------------------------------------------

form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict(input.value);
});

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => predict(chip.dataset.smiles)),
);

window.addEventListener("popstate", () => {
  const smiles = new URL(location.href).searchParams.get("smiles");
  if (smiles) predict(smiles, { push: false });
});

loadModelFacts();
predict(new URL(location.href).searchParams.get("smiles") || DEFAULT_SMILES, { push: false });
