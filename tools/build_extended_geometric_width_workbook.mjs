import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const axisDir = `${root}/cattle_3d_extraction_66/skeleton_axis_comparison`;
const payload = JSON.parse(await fs.readFile(`${axisDir}/extended_geometric_max_width_comparison.json`, "utf8"));
const outputDir = `${root}/outputs/extended_geometric_widths`;
await fs.mkdir(outputDir, { recursive: true });

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
  return rows;
}

const profileCsv = (await fs.readFile(`${axisDir}/extended_geometric_width_profiles.csv`, "utf8")).replace(/^\uFEFF/, "");
const profileRowsRaw = parseCsv(profileCsv);
const profileHeaders = profileRowsRaw[0];
const numericProfileColumns = new Set(["cow_id", "station_index", "station_from_rear_m", "center_x_m", "center_y_m", "tangent_x", "tangent_y", "raw_width_m", "smoothed_width_m", "left_offset_m", "right_offset_m", "cross_section_point_count"]);
const profileRows = profileRowsRaw.slice(1).filter((row) => row.length > 1).map((row) => row.map((value, index) => {
  const header = profileHeaders[index];
  if (value === "") return null;
  if (header === "is_maximum_width_station" || header === "is_in_max_width_search_region") return value.toLowerCase() === "true";
  if (numericProfileColumns.has(header)) return Number(value);
  return value;
}));

const workbook = Workbook.create();
const overview = workbook.worksheets.add("差异概览");
const records = workbook.worksheets.add("每头牛最大体宽");
const profiles = workbook.worksheets.add("截面宽度曲线");

const c = payload.comparison;
overview.getRange("A1:B23").values = [
  ["几何中线最大体宽与PCA最大体宽", ""],
  ["指标", "结果"],
  ["牛数量", c.n],
  ["旧方法", c.old_measure],
  ["新方法", c.new_measure],
  ["旧方法平均宽度（m）", c.mean_old_width_m],
  ["新方法平均宽度（m）", c.mean_new_width_m],
  ["平均差：新−旧（m）", c.mean_signed_difference_m],
  ["平均绝对差（m）", c.mean_absolute_difference_m],
  ["中位绝对差（m）", c.median_absolute_difference_m],
  ["RMSE（m）", c.rmse_difference_m],
  ["平均差：新−旧（%）", c.mean_signed_difference_percent / 100],
  ["平均绝对差（%）", c.mean_absolute_difference_percent / 100],
  ["最大绝对差（m）", c.maximum_absolute_difference_m],
  ["两套宽度 Pearson r", c.pearson_correlation_old_vs_new],
  ["95%一致性下限（m）", c.agreement_lower_95_m],
  ["95%一致性上限（m）", c.agreement_upper_95_m],
  ["新值较大的牛数", c.count_new_larger],
  ["新值较小的牛数", c.count_new_smaller],
  ["绝对差超过5 cm的牛数", c.count_absolute_difference_over_5cm],
  ["绝对百分比差超过10%的牛数", c.count_absolute_percent_over_10],
  ["最大宽度搜索范围", "后躯端向内5 cm至头端向内25 cm"],
  ["判断", "两种最大体宽总体差距很小"],
];

const recordHeaders = Object.keys(payload.records[0]);
records.getRangeByIndexes(0, 0, 1, recordHeaders.length).values = [recordHeaders];
records.getRangeByIndexes(1, 0, payload.records.length, recordHeaders.length).values = payload.records.map((record) => recordHeaders.map((header) => record[header] ?? null));
profiles.getRangeByIndexes(0, 0, 1, profileHeaders.length).values = [profileHeaders];
for (let start = 0; start < profileRows.length; start += 1000) {
  const chunk = profileRows.slice(start, start + 1000);
  profiles.getRangeByIndexes(start + 1, 0, chunk.length, profileHeaders.length).values = chunk;
}

const fontFamily = "Arial";
for (const sheet of [overview, records, profiles]) {
  sheet.showGridLines = false;
  sheet.getUsedRange().format.font = { name: fontFamily, size: 10 };
  sheet.getUsedRange().format.verticalAlignment = "center";
}
overview.getRange("A1:B1").format.font = { name: fontFamily, size: 15, bold: true };
overview.getRange("A2:B2").format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" } };
overview.getRange("A2:B23").format.borders = { insideHorizontal: { style: "thin", color: "#D7DEE8" } };
overview.getRange("A1:A23").format.columnWidth = 34;
overview.getRange("B1:B23").format.columnWidth = 76;
overview.getRange("B3:B23").format.wrapText = true;
overview.getRange("B6:B11").format.numberFormat = "0.000000";
overview.getRange("B12:B13").format.numberFormat = "0.00%";
overview.getRange("B14:B17").format.numberFormat = "0.000000";

for (const [sheet, width] of [[records, recordHeaders.length], [profiles, profileHeaders.length]]) {
  const header = sheet.getRangeByIndexes(0, 0, 1, width);
  header.format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
  header.format.rowHeight = 34;
  sheet.freezePanes.freezeRows(1);
  sheet.getUsedRange().format.columnWidth = 18;
}
records.getRange(`A2:A${payload.records.length + 1}`).format.numberFormat = "0";
records.getRange(`B2:M${payload.records.length + 1}`).format.numberFormat = "0.000000";
records.getRange(`N2:Q${payload.records.length + 1}`).format.columnWidth = 32;
profiles.getRange(`A2:B${profileRows.length + 1}`).format.numberFormat = "0";
profiles.getRange(`C2:L${profileRows.length + 1}`).format.numberFormat = "0.000000";
profiles.getRange(`M2:N${profileRows.length + 1}`).format.horizontalAlignment = "center";
records.tables.add(`A1:Q${payload.records.length + 1}`, true, "GeometricWidthRecords").style = "TableStyleMedium2";
profiles.tables.add(`A1:O${profileRows.length + 1}`, true, "GeometricWidthProfiles").style = "TableStyleMedium2";

console.log((await workbook.inspect({ kind: "table", range: "差异概览!A1:B23", include: "values,formulas", tableMaxRows: 23, tableMaxCols: 2 })).ndjson);
console.log((await workbook.inspect({ kind: "table", range: "每头牛最大体宽!A1:Q8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 17 })).ndjson);
console.log((await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 50 }, summary: "final formula error scan" })).ndjson);

for (const [sheetName, range, fileName] of [["差异概览", "A1:B23", "preview_overview.png"], ["每头牛最大体宽", "A1:Q15", "preview_records.png"], ["截面宽度曲线", "A1:O20", "preview_profiles.png"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}
const outputPath = `${outputDir}/extended_geometric_max_width_66.xlsx`;
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ output: outputPath, cows: payload.records.length, profileRows: profileRows.length }));
