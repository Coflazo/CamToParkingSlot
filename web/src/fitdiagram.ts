/**
 * The signature moment: your car, drawn to scale, inside that bay.
 *
 * Every other parking app tells you a space exists. This draws the space, draws your car
 * in it, and dimensions the gaps in centimetres, the way a workshop drawing would. It is
 * the product's whole claim made checkable in one glance: the bay is surveyed, the car is
 * registered, and the arithmetic between them is the answer.
 *
 * Two rules keep it honest.
 *
 * **One scale for everything.** Bay and car share a single metres-to-pixels factor, so the
 * picture cannot flatter a fit. A drawing that stretched the car to look comfortable would
 * be worse than no drawing.
 *
 * **The binding constraint is named.** When a bay is tight it is tight in one specific
 * dimension, and that dimension is highlighted and labelled. "It fits" is not useful when
 * the answer is 4 cm of door clearance on the passenger side.
 */

import type { Recommendation } from "./api";

export interface CarDimensions {
  lengthCm: number;
  bodyWidthCm: number;
  mirrorWidthCm: number;
  label: string;
}

const NS = "http://www.w3.org/2000/svg";

function el(name: string, attrs: Record<string, string | number>): SVGElement {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

/**
 * A dimension line with end ticks and a label, as on an engineering drawing.
 *
 * The ticks matter more than they look: without them the eye cannot tell where a
 * measurement starts and stops, and the number stops meaning anything precise.
 */
function dimension(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  text: string,
  className: string,
): SVGGElement {
  const group = el("g", { class: `fd-dim ${className}` }) as SVGGElement;
  const horizontal = Math.abs(y2 - y1) < 0.5;
  const tick = 4;

  group.appendChild(el("line", { x1, y1, x2, y2 }));
  if (horizontal) {
    group.appendChild(el("line", { x1, y1: y1 - tick, x2: x1, y2: y1 + tick }));
    group.appendChild(el("line", { x1: x2, y1: y2 - tick, x2, y2: y2 + tick }));
  } else {
    group.appendChild(el("line", { x1: x1 - tick, y1, x2: x1 + tick, y2: y1 }));
    group.appendChild(el("line", { x1: x2 - tick, y1: y2, x2: x2 + tick, y2 }));
  }

  const label = el("text", {
    x: (x1 + x2) / 2,
    y: horizontal ? y1 - 7 : (y1 + y2) / 2,
    "text-anchor": "middle",
    "dominant-baseline": horizontal ? "auto" : "middle",
  });
  label.textContent = text;
  group.appendChild(label);
  return group;
}

/**
 * Draw the bay and the car.
 *
 * Returns null when there is nothing honest to draw: a car park rather than a marked bay,
 * or no vehicle selected. An empty frame says "we do not know" far better than an
 * illustration of a generic car in a generic rectangle.
 */
export function render(result: Recommendation, car: CarDimensions | null): SVGSVGElement | null {
  const bayL = result.bay_length_cm;
  const bayW = result.bay_width_cm;
  if (!result.is_exact_space || bayL <= 0 || bayW <= 0 || !car) return null;

  const parallel = (result.orientation || "").toLowerCase().includes("parallel");
  // A parallel bay is drawn along its length, a perpendicular one across it, because that
  // is how each looks from the road the driver is on.
  const alongCm = parallel ? bayL : bayW;
  const acrossCm = parallel ? bayW : bayL;
  const carAlongCm = parallel ? car.lengthCm : car.bodyWidthCm;
  const carAcrossCm = parallel ? car.bodyWidthCm : car.lengthCm;
  const mirrorAlongCm = parallel ? car.lengthCm : car.mirrorWidthCm;
  const mirrorAcrossCm = parallel ? car.mirrorWidthCm : car.lengthCm;

  // Asymmetric padding: the right edge carries the side dimension and its label, the
  // top carries the two end dimensions, and the bottom carries the captions.
  const padL = 16;
  const padR = 62;
  const padT = 34;
  const padB = 34;
  const width = 560;
  const scale = (width - padL - padR) / alongCm;
  const height = acrossCm * scale + padT + padB;

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "fit-diagram",
    role: "img",
    "aria-label":
      `${car.label} drawn to scale in a ${(bayL / 100).toFixed(2)} by ` +
      `${(bayW / 100).toFixed(2)} metre bay`,
  }) as SVGSVGElement;

  const bx = padL;
  const by = padT;
  const bw = alongCm * scale;
  const bh = acrossCm * scale;

  // The kerb, drawn as the solid edge. A driver reads the picture from the kerb inward.
  svg.appendChild(el("line", { x1: bx - 14, y1: by + bh, x2: bx + bw + 14, y2: by + bh, class: "fd-kerb" }));

  // The bay: painted lines, so dashed.
  svg.appendChild(el("rect", { x: bx, y: by, width: bw, height: bh, rx: 2, class: "fd-bay" }));

  const carW = carAlongCm * scale;
  const carH = carAcrossCm * scale;
  const cx = bx + (bw - carW) / 2;
  const cy = by + (bh - carH) / 2;

  // Mirrors first, so the body sits over them.
  const mirrorW = mirrorAlongCm * scale;
  const mirrorH = mirrorAcrossCm * scale;
  svg.appendChild(
    el("rect", {
      x: bx + (bw - mirrorW) / 2,
      y: by + (bh - mirrorH) / 2,
      width: mirrorW,
      height: mirrorH,
      rx: 3,
      class: "fd-mirrors",
    }),
  );
  svg.appendChild(el("rect", { x: cx, y: cy, width: carW, height: carH, rx: 6, class: "fd-car" }));

  // Clearances, in real centimetres rather than pixels.
  const endGapCm = (alongCm - carAlongCm) / 2;
  const sideGapCm = (acrossCm - mirrorAcrossCm) / 2;
  const binding = (result.fit.binding_constraint || "").toLowerCase();

  svg.appendChild(
    dimension(
      bx,
      by - 16,
      cx,
      by - 16,
      `${Math.round(endGapCm)}`,
      binding.includes("length") ? "is-binding" : "",
    ),
  );
  svg.appendChild(
    dimension(
      cx + carW,
      by - 16,
      bx + bw,
      by - 16,
      `${Math.round(endGapCm)}`,
      binding.includes("length") ? "is-binding" : "",
    ),
  );
  svg.appendChild(
    dimension(
      bx + bw + 18,
      by,
      bx + bw + 18,
      cy,
      `${Math.round(sideGapCm)}`,
      binding.includes("width") ? "is-binding" : "",
    ),
  );

  const bayLabel = el("text", { x: bx, y: by + bh + 22, class: "fd-caption" });
  bayLabel.textContent = `bay ${(bayL / 100).toFixed(2)} x ${(bayW / 100).toFixed(2)} m`;
  svg.appendChild(bayLabel);

  const carLabel = el("text", {
    x: bx + bw,
    y: by + bh + 22,
    "text-anchor": "end",
    class: "fd-caption",
  });
  carLabel.textContent = `${car.label} ${(car.lengthCm / 100).toFixed(2)} x ${(car.bodyWidthCm / 100).toFixed(2)} m`;
  svg.appendChild(carLabel);

  return svg;
}

/** The headline figure: how much room is left, and where it runs out first. */
export function slackSummary(result: Recommendation): { value: string; unit: string; note: string } {
  const slack = Math.round(result.fit.slack_cm);
  const where = result.fit.binding_constraint
    ? result.fit.binding_constraint.replaceAll("_", " ")
    : "";
  if (result.fit.verdict === "DOES_NOT_FIT") {
    return { value: `${Math.abs(slack)}`, unit: "cm short", note: where || "does not fit" };
  }
  if (result.fit.verdict === "UNVERIFIED") {
    return { value: "?", unit: "", note: "select a vehicle to measure this" };
  }
  return { value: `${slack}`, unit: "cm to spare", note: where ? `tightest at ${where}` : "" };
}
