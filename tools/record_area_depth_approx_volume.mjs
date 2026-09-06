import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const path = process.argv[2];
if (!path) throw new Error("Usage: node record_area_depth_approx_volume.mjs RECORDS.csv");

const workbook = await Workbook.fromCSV(await fs.readFile(path, "utf8"), { sheetName: "Data" });
const sheet = workbook.worksheets.getItem("Data");
const original = sheet.getUsedRange(true).values;
original[0][0] = String(original[0][0]).replace(/^\uFEFF+/, "");
sheet.getCell(0, 0).values = [[original[0][0]]];

const fields = [
  "torso_area_median_depth_approx_volume_m3",
  "torso_area_median_depth_approx_volume_l",
  "torso_area_median_depth_volume_formula",
  "torso_area_median_depth_area_source",
  "torso_area_median_depth_depth_source",
  "torso_area_median_depth_volume_status",
  "torso_area_median_depth_method_version",
  "torso_area_median_depth_recorded_on",
];
let headers = sheet.getUsedRange(true).values[0].map(String);
const missing = fields.filter((field) => !headers.includes(field));
if (missing.length) {
  const start = headers.length;
  const matrix = original.map((_, row) => row === 0 ? missing : missing.map(() => ""));
  sheet.getRangeByIndexes(0, start, matrix.length, missing.length).values = matrix;
  headers = sheet.getUsedRange(true).values[0].map(String);
}

const index = Object.fromEntries(headers.map((header, column) => [header, column]));
const values = sheet.getUsedRange(true).values;
const output = values.map((row, rowIndex) => {
  if (rowIndex === 0) return fields;
  const qualified = row[index.analysis_data_quality_code] === "qualified";
  const areaText = row[index.pca_full_torso_projected_area_m2];
  const depthText = row[index.rightview_median_body_depth_m];
  if (!qualified || areaText === "" || depthText === "") return fields.map(() => "");
  const volume = Number(areaText) * Number(depthText);
  return [
    volume,
    volume * 1000,
    "pca_full_torso_projected_area_m2 * rightview_median_body_depth_m",
    "pca_full_torso_projected_area_m2",
    "rightview_median_body_depth_m",
    "ok",
    "pca45_48_area_times_rightview_median_depth_v1.0",
    "2026-09-06",
  ];
});

for (let field = 0; field < fields.length; field++) {
  const column = index[fields[field]];
  sheet.getRangeByIndexes(0, column, output.length, 1).values = output.map((row) => [row[field]]);
}

const check = await workbook.inspect({
  kind: "table", sheetId: "Data", range: "HE1:HL67", include: "values,formulas",
  tableMaxRows: 5, tableMaxCols: 10, summary: "Area-times-median-depth approximate torso volume",
});
console.log(check.ndjson);

const finalValues = sheet.getUsedRange(true).values;
function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
const csvText = "\ufeff" + finalValues.map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
await fs.writeFile(path, csvText, "utf8");
console.log(`updated ${path}: ${finalValues.length - 1} rows, ${finalValues[0].length} columns`);
