// Build a .docx report from a content JSON:  node build_report.js content.json out.docx
// content = {title, subtitle, meta, blocks:[{type, ...}]}
//   {type:"h1"|"h2", text}                  headings (in the table of contents)
//   {type:"p", text | runs:[{text,bold,italic,color}]}
//   {type:"bullets"|"numbers", items:[text | {runs}]}
//   {type:"table", columns:[...], widths:[dxa...], rows:[[cell...]], numeric:[colIdx...], highlight:[rowIdx...]}
//   {type:"image", path, width (px), height (px), caption}
//   {type:"note", text}                     shaded call-out paragraph
//   {type:"pagebreak"}
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  AlignmentType, ImageRun, ShadingType, BorderStyle, LevelFormat, PageBreak, TableOfContents,
  Footer, PageNumber, TableLayoutType,
} = require("docx");

const [, , contentPath, outPath] = process.argv;
const C = JSON.parse(fs.readFileSync(contentPath, "utf8"));
const baseDir = path.dirname(contentPath);

const FONT = "Calibri";
const INK = "1F1F1F", INK2 = "52514E", ACCENT = "2A78D6", LINE = "D9D8D4", SHADE = "F1F0EC";
const PAGE_W = 11906, MARGIN = 1134;                  // A4, 2 cm margins
const TEXT_W = PAGE_W - 2 * MARGIN;                   // 9638 dxa

function runs(spec, base = {}) {
  if (typeof spec === "string") return [new TextRun({ text: spec, font: FONT, ...base })];
  return spec.map(r => new TextRun({ text: r.text, bold: r.bold, italics: r.italic, color: r.color,
                                     font: FONT, ...base }));
}
const para = (spec, opts = {}) => new Paragraph({ children: runs(spec, opts.run || {}), spacing: { after: 120, line: 276 }, ...opts.p });

function table(b) {
  const cols = b.columns.length;
  const widths = b.widths || Array(cols).fill(Math.floor(TEXT_W / cols));
  const total = widths.reduce((a, x) => a + x, 0);
  const numeric = new Set(b.numeric || []);
  const hl = new Set(b.highlight || []);
  const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
  const cell = (text, i, header, rowHl) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, color: "auto", fill: "EAF1FB" }
                    : rowHl ? { type: ShadingType.CLEAR, color: "auto", fill: "FDF1EC" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: border, bottom: border, left: border, right: border },
    children: [new Paragraph({
      alignment: numeric.has(i) && !header ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(text ?? ""), { size: 18, bold: header, color: header ? INK : INK }),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: [new TableRow({ tableHeader: true, children: b.columns.map((c, i) => cell(c, i, true)) }),
           ...b.rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, hl.has(ri))) }))],
  });
}

const children = [];
children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: runs(C.title) }));
if (C.subtitle) children.push(para(C.subtitle, { run: { size: 26, color: INK2 } }));
if (C.meta) children.push(para(C.meta, { run: { size: 18, color: INK2 } }));
children.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } },
                              spacing: { after: 200 }, children: [] }));
if (C.toc) {
  children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs("Contents") }));
  children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
  children.push(new Paragraph({ children: [new PageBreak()] }));
}

for (const b of C.blocks) {
  switch (b.type) {
    case "h1": children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(b.text) })); break;
    case "h2": children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(b.text) })); break;
    case "p": children.push(para(b.runs || b.text)); break;
    case "note":
      children.push(new Paragraph({ children: runs(b.runs || b.text, { size: 20 }), spacing: { before: 60, after: 160 },
        shading: { type: ShadingType.CLEAR, color: "auto", fill: SHADE },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } }, indent: { left: 160 } }));
      break;
    case "bullets": case "numbers":
      for (const it of b.items) children.push(new Paragraph({
        numbering: { reference: b.type === "bullets" ? "bul" : `num${b.id || 0}`, level: it.level || 0 },
        spacing: { after: 80, line: 264 }, children: runs(it.runs || it.text || it) }));
      break;
    case "table":
      children.push(table(b));
      children.push(new Paragraph({ spacing: { after: 160 }, children: b.caption ? runs(b.caption, { size: 16, color: INK2, italics: true }) : [] }));
      break;
    case "image": {
      const data = fs.readFileSync(path.resolve(baseDir, b.path));
      const pw = data.readUInt32BE(16), ph = data.readUInt32BE(20);   // PNG IHDR: real pixel size
      const maxW = b.maxWidth || 640, w = Math.min(pw, maxW), h = Math.round(ph * w / pw);
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 40 },
        children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h },
                                  altText: { title: b.caption || "chart", description: b.caption || "chart", name: path.basename(b.path) } })] }));
      if (b.caption) children.push(para(b.caption, { run: { size: 16, color: INK2, italics: true }, p: { alignment: AlignmentType.CENTER } }));
      break;
    }
    case "pagebreak": children.push(new Paragraph({ children: [new PageBreak()] })); break;
    default: throw new Error("unknown block " + b.type);
  }
}

const numIds = [...new Set(C.blocks.filter(b => b.type === "numbers").map(b => b.id || 0))];
const doc = new Document({
  creator: C.author || "", title: C.title, description: C.subtitle || "",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 44, bold: true, color: INK },
        paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 320, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: ACCENT }, paragraph: { spacing: { before: 220, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } } ] },
    ...numIds.map(id => ({ reference: `num${id}`, levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } } ] })),
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ font: FONT, size: 16, color: INK2, children: [C.footer || "", "  ·  page ", PageNumber.CURRENT] })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath, buf.length, "bytes"); });
