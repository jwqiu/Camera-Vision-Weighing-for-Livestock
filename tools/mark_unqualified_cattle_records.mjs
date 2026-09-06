import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const mainCsvPath = `${root}/cattle_3d_extraction_66/long_axis_records.csv`;
const annotationsCsvPath = `${root}/cow_visual_annotations.csv`;
const workbookPath = `${root}/outputs/extended_geometric_core_areas/extended_geometric_core_area_comparison_66.xlsx`;
const previewPath = `${root}/outputs/extended_geometric_core_areas/preview_manual_data_quality.png`;
const unqualified = new Set([19, 28, 50, 98, 147]);
const annotatedOn = "2026-09-05";

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ""; }
    else if (ch === '\n') { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.length > 1 || r[0] !== "");
}

function escapeCsv(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function serializeCsv(headers, records) {
  return "\uFEFF" + [headers, ...records.map((record) => headers.map((header) => record[header] ?? ""))]
    .map((row) => row.map(escapeCsv).join(","))
    .join("\n") + "\n";
}

async function readRecords(path) {
  const rows = parseCsv((await fs.readFile(path, "utf8")).replace(/^\uFEFF/, ""));
  const headers = rows[0];
  const records = rows.slice(1).map((row) => Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ""])));
  return { headers, records };
}

function ensureFields(headers, fields) {
  for (const field of fields) if (!headers.includes(field)) headers.push(field);
}

const fields = [
  "analysis_data_quality_code",
  "analysis_include",
  "analysis_data_quality_label_zh",
  "analysis_data_quality_source",
  "analysis_data_quality_annotated_on",
  "analysis_data_quality_notes",
];

const main = await readRecords(mainCsvPath);
ensureFields(main.headers, fields);
for (const record of main.records) {
  const cowId = Number(record.cow_id);
  const isUnqualified = unqualified.has(cowId);
  Object.assign(record, {
    analysis_data_quality_code: isUnqualified ? "unqualified" : "qualified",
    analysis_include: isUnqualified ? "FALSE" : "TRUE",
    analysis_data_quality_label_zh: isUnqualified ? "不合格" : "合格",
    analysis_data_quality_source: "user_review",
    analysis_data_quality_annotated_on: annotatedOn,
    analysis_data_quality_notes: isUnqualified ? "用户指定为不合格数据，后续分析排除" : "",
  });
}
await fs.writeFile(mainCsvPath, serializeCsv(main.headers, main.records), "utf8");

const annotations = await readRecords(annotationsCsvPath);
ensureFields(annotations.headers, fields);
const selectedIds = new Set(main.records.map((record) => Number(record.cow_id)));
for (const record of annotations.records) {
  const cowId = Number(record.cow_id);
  if (!selectedIds.has(cowId)) {
    Object.assign(record, {
      analysis_data_quality_code: "not_assessed",
      analysis_include: "",
      analysis_data_quality_label_zh: "未评估",
      analysis_data_quality_source: "",
      analysis_data_quality_annotated_on: "",
      analysis_data_quality_notes: "",
    });
  } else {
    const isUnqualified = unqualified.has(cowId);
    Object.assign(record, {
      analysis_data_quality_code: isUnqualified ? "unqualified" : "qualified",
      analysis_include: isUnqualified ? "FALSE" : "TRUE",
      analysis_data_quality_label_zh: isUnqualified ? "不合格" : "合格",
      analysis_data_quality_source: "user_review",
      analysis_data_quality_annotated_on: annotatedOn,
      analysis_data_quality_notes: isUnqualified ? "用户指定为不合格数据，后续分析排除" : "",
    });
  }
}
await fs.writeFile(annotationsCsvPath, serializeCsv(annotations.headers, annotations.records), "utf8");

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.add("人工数据质量");
const headers = ["cow_id", "analysis_data_quality_code", "analysis_include", "数据质量", "标记来源", "标记日期", "说明", "原始质量状态", "原始复核原因"];
const rows = main.records.map((record) => [
  Number(record.cow_id),
  record.analysis_data_quality_code,
  record.analysis_include === "TRUE",
  record.analysis_data_quality_label_zh,
  record.analysis_data_quality_source,
  record.analysis_data_quality_annotated_on,
  record.analysis_data_quality_notes,
  record.quality_status,
  record.review_reason,
]);
sheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
sheet.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows;
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.getUsedRange().format.font = { name: "Arial", size: 10 };
sheet.getUsedRange().format.verticalAlignment = "center";
sheet.getRange("A1:I1").format = { fill: "#334155", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
sheet.getRange("A1:I1").format.rowHeight = 38;
sheet.getRange("A1:A67").format.columnWidth = 11;
sheet.getRange("B1:B67").format.columnWidth = 25;
sheet.getRange("C1:C67").format.columnWidth = 16;
sheet.getRange("D1:F67").format.columnWidth = 17;
sheet.getRange("G1:G67").format.columnWidth = 38;
sheet.getRange("H1:I67").format.columnWidth = 32;
sheet.getRange("G2:I67").format.wrapText = true;
sheet.tables.add("A1:I67", true, "ManualDataQualityRecords").style = "TableStyleMedium2";
sheet.getRange("D2:D67").conditionalFormats.add("containsText", { text: "不合格", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } } });

console.log((await workbook.inspect({ kind: "table", range: "人工数据质量!A1:I67", include: "values,formulas", tableMaxRows: 67, tableMaxCols: 9 })).ndjson);
console.log((await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" })).ndjson);
const preview = await workbook.render({ sheetName: "人工数据质量", range: "A1:I16", scale: 1.2, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(JSON.stringify({ workbook: workbookPath, unqualified: [...unqualified], qualified: 61, notAssessedInMaster: 88 }));
