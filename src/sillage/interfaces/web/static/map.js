// The odour map: 4777 known molecules laid out by how the model thinks they smell.
// Positions come from the model (journal E-016), families from experts' labels.
//
// One family is highlighted at a time, on purpose. A scatter plot keeps at most
// three colours apart for colour-blind readers; eight coloured families at once
// would be confetti. Everything else stays a recessive grey, and the current
// molecule is marked by shape (a ring and a label), never by colour alone.

const OdorMapView = (() => {
  const HIGHLIGHT = "#a6581a"; // validated: lightness, chroma, >= 3:1 on the surface
  const CONTEXT = "#c9c9c1"; // recessive context, deliberately below 3:1
  const INK = "#26190f";
  const SURFACE = "#fdfcf9";
  const PAD = 18;

  let data = null;
  let pending = null;
  const view = { family: 0, report: null, canvas: null, tooltip: null, scale: null, hover: -1 };

  async function load() {
    if (data) return data;
    pending ??= fetch("/map")
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    data = await pending;
    return data;
  }

  async function show(container, report) {
    const map = await load();
    if (!container.isConnected) return; // a newer answer replaced this one meanwhile
    if (!map) {
      container.remove();
      return;
    }
    view.report = report;
    view.family = defaultFamily(map, report);
    build(container, map);
  }

  // The family of the molecule's most probable note; else of what experts recorded.
  function defaultFamily(map, report) {
    const notes = [...report.predicted.map((d) => d.descriptor), ...(report.known?.descriptors || [])];
    for (const note of notes) {
      const index = map.descriptor_family[note];
      if (index !== undefined && map.families[index].id !== "other") return index;
    }
    return 0;
  }

  function build(container, map) {
    const families = map.families.map((f, index) => ({ ...f, index })).filter((f) => f.id !== "other");
    const chips = families.map((f) =>
      h(
        "button",
        {
          type: "button",
          class: "chip map__family",
          "aria-pressed": String(f.index === view.family),
          onclick: (event) => {
            view.family = f.index;
            container.querySelectorAll(".map__family").forEach((c) => c.setAttribute("aria-pressed", "false"));
            event.currentTarget.setAttribute("aria-pressed", "true");
            draw();
          },
        },
        capitalize(f.name),
        h("span", { class: "map__count" }, f.count.toLocaleString("ru-RU")),
      ),
    );

    view.canvas = h("canvas", { class: "map__canvas", role: "img" });
    view.tooltip = h("div", { class: "map__tooltip", hidden: true });
    const frame = h("div", { class: "map__frame" }, view.canvas, view.tooltip);

    container.replaceChildren(
      h("h2", { id: "map-title" }, "Карта запахов"),
      h(
        "p",
        { class: "map__lede" },
        "Каждая точка — молекула из датасета. Рядом стоят те, что пахнут похоже по мнению модели. " +
          "Подсветка — семейство запахов по меткам экспертов: чем плотнее оно собрано, тем лучше модель его понимает.",
      ),
      h("div", { class: "map__families", role: "group", "aria-label": "Семейство для подсветки" }, chips),
      frame,
      h(
        "p",
        { class: "map__legend" },
        h("span", {}, h("i", { class: "map__key map__key--family" }), "выбранное семейство"),
        h("span", {}, h("i", { class: "map__key map__key--rest" }), "остальные молекулы"),
        h("span", {}, h("i", { class: "map__key map__key--query" }), "эта молекула"),
        h("span", {}, h("i", { class: "map__key map__key--near" }), "ближайшие известные"),
      ),
    );

    view.canvas.addEventListener("mousemove", onHover);
    view.canvas.addEventListener("mouseleave", hideTooltip);
    view.canvas.addEventListener("click", onClick);
    new ResizeObserver(() => requestAnimationFrame(draw)).observe(frame);
    draw();
  }

  // ---- drawing ------------------------------------------------------------------

  function layout(width) {
    const xs = data.points.x;
    const ys = data.points.y;
    const spanX = Math.max(...xs) || 1;
    const spanY = Math.max(...ys) || 1;
    // Height follows the map's own proportions, so a phone gets no empty bands.
    const height = Math.round(Math.min(Math.max((width - 2 * PAD) * (spanY / spanX) + 2 * PAD, 180), 620));
    const unit = Math.min((width - 2 * PAD) / spanX, (height - 2 * PAD) / spanY);
    const offsetX = (width - unit * spanX) / 2;
    const offsetY = (height - unit * spanY) / 2;
    return {
      height,
      // Dots scale with the map: 2.3 px on a desktop-wide map, ~1.2 px on a phone.
      dot: Math.max(1.1, Math.min(2.3, width / 480)),
      x: (v) => offsetX + v * unit,
      y: (v) => height - offsetY - v * unit, // UMAP's y grows upward
    };
  }

  function draw() {
    const canvas = view.canvas;
    if (!canvas || !canvas.isConnected || !data) return;
    const width = canvas.parentElement.clientWidth;
    const scale = layout(width);
    view.scale = scale;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(scale.height * ratio);
    canvas.style.height = `${scale.height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = SURFACE;
    ctx.fillRect(0, 0, width, scale.height);

    const { x, y, family } = data.points;
    const dot = (i, r) => {
      ctx.beginPath();
      ctx.arc(scale.x(x[i]), scale.y(y[i]), r, 0, 2 * Math.PI);
      ctx.fill();
    };

    ctx.fillStyle = CONTEXT;
    for (let i = 0; i < x.length; i++) if (family[i] !== view.family) dot(i, scale.dot * 0.75);
    ctx.fillStyle = HIGHLIGHT;
    for (let i = 0; i < x.length; i++) if (family[i] === view.family) dot(i, scale.dot);

    const report = view.report;
    const index = new Map(data.points.smiles.map((s, i) => [s, i]));
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = INK;
    for (const n of report.neighbours) {
      const i = index.get(n.smiles);
      if (i === undefined) continue;
      ctx.beginPath();
      ctx.arc(scale.x(x[i]), scale.y(y[i]), 5, 0, 2 * Math.PI);
      ctx.stroke();
    }

    if (report.map_position) drawQuery(ctx, scale, report);

    const familyName = data.families[view.family].name;
    canvas.setAttribute(
      "aria-label",
      `Карта запахов из ${x.length.toLocaleString("ru-RU")} молекул; подсвечено семейство «${familyName}». ` +
        "Эта молекула отмечена кольцом с подписью.",
    );
  }

  function drawQuery(ctx, scale, report) {
    const px = scale.x(report.map_position.x);
    const py = scale.y(report.map_position.y);
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, 2 * Math.PI);
    ctx.lineWidth = 5;
    ctx.strokeStyle = SURFACE; // a surface ring keeps the marker legible over dense dots
    ctx.stroke();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = INK;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, 2 * Math.PI);
    ctx.fillStyle = INK;
    ctx.fill();

    const label = capitalize(report.matched_name || report.names?.display || "эта молекула");
    ctx.font = '600 14px "Fira Sans Condensed", "Fira Sans", sans-serif';
    const textWidth = ctx.measureText(label).width;
    const right = px + 14 + textWidth < ctx.canvas.width / (window.devicePixelRatio || 1) - 4;
    const tx = right ? px + 14 : px - 14 - textWidth;
    ctx.lineJoin = "round";
    ctx.lineWidth = 4;
    ctx.strokeStyle = SURFACE;
    ctx.strokeText(label, tx, py + 5);
    ctx.fillStyle = INK;
    ctx.fillText(label, tx, py + 5);
  }

  // ---- interaction --------------------------------------------------------------

  function nearestPoint(event) {
    const rect = view.canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const { x, y } = data.points;
    let best = -1;
    let bestDistance = 10 * 10; // hit target larger than the dot
    for (let i = 0; i < x.length; i++) {
      const dx = view.scale.x(x[i]) - mx;
      const dy = view.scale.y(y[i]) - my;
      const d = dx * dx + dy * dy;
      if (d < bestDistance) {
        bestDistance = d;
        best = i;
      }
    }
    return { index: best, mx, my };
  }

  function onHover(event) {
    const { index, mx, my } = nearestPoint(event);
    if (index < 0) return hideTooltip();
    const { name, smiles, family } = data.points;
    view.tooltip.replaceChildren(
      h("strong", {}, capitalize(name[index] || "Без названия")),
      h("span", { class: "map__tooltip-family" }, capitalize(data.families[family[index]].name)),
      !name[index] && h("code", { lang: "en" }, smiles[index]),
    );
    view.tooltip.hidden = false;
    const frameWidth = view.canvas.clientWidth;
    view.tooltip.style.left = `${Math.min(mx + 14, frameWidth - 220)}px`;
    view.tooltip.style.top = `${my + 14}px`;
    view.canvas.style.cursor = "pointer";
  }

  function hideTooltip() {
    if (view.tooltip) view.tooltip.hidden = true;
    if (view.canvas) view.canvas.style.cursor = "";
  }

  function onClick(event) {
    const { index } = nearestPoint(event);
    if (index < 0) return;
    predict(data.points.name[index] || data.points.smiles[index]);
    document.getElementById("query-title").scrollIntoView({ behavior: "smooth" });
  }

  return { show };
})();
