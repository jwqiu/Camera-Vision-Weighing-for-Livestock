import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = `${root}/outputs/extended_geometric_core_areas`;
const workbookPath = `${outputDir}/extended_geometric_core_area_comparison_66.xlsx`;
const payload = JSON.parse(
  await fs.readFile(`${root}/cattle_3d_extraction_66/skeleton_axis_comparison/extended_geometric_core_volume_summary.json`, "utf8"),
);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const overview = workbook.worksheets.add("体积概览");
const records = workbook.worksheets.add("每头牛核心体积");
const parameters = workbook.worksheets.add("体积方法参数");
const s = payload.summary;

overview.getRange("A1:B25").values = [
  ["几何中线核心躯干投影体积", ""],
  ["指标", "结果"],
  ["牛数量", s.n],
  ["PCA方法平均体积（m³）", s.mean_old_pca_volume_m3],
  ["几何中线方法平均体积（m³）", s.mean_new_geometric_volume_m3],
  ["平均差：新−旧（m³）", s.mean_signed_difference_m3],
  ["中位差：新−旧（m³）", s.median_signed_difference_m3],
  ["平均绝对差（m³）", s.mean_absolute_difference_m3],
  ["平均差：新−旧（%）", s.mean_signed_difference_percent / 100],
  ["平均绝对差（%）", s.mean_absolute_difference_percent / 100],
  ["最大绝对差（m³）", s.maximum_absolute_difference_m3],
  ["新体积较大的牛数", s.count_new_larger],
  ["新体积较小的牛数", s.count_new_smaller],
  ["新旧体积 Pearson r", s.old_vs_new_volume.pearson_r],
  ["新旧体积 R²", s.old_vs_new_volume.r_squared],
  ["新体积 vs. 体重 Pearson r", s.new_volume_vs_weight.pearson_r],
  ["新体积 vs. 体重 R²", s.new_volume_vs_weight.r_squared],
  ["新体积 vs. 体重 95%CI下限", s.new_volume_vs_weight.fisher_95_ci_lower],
  ["新体积 vs. 体重 95%CI上限", s.new_volume_vs_weight.fisher_95_ci_upper],
  ["旧体积 vs. 体重 Pearson r", s.old_volume_vs_weight.pearson_r],
  ["旧体积 vs. 体重 R²", s.old_volume_vs_weight.r_squared],
  ["旧体积 vs. 体重 95%CI下限", s.old_volume_vs_weight.fisher_95_ci_lower],
  ["旧体积 vs. 体重 95%CI上限", s.old_volume_vs_weight.fisher_95_ci_upper],
  ["需人工复核的牛", s.review_required_cows.length ? s.review_required_cows.join(", ") : "无"],
  ["说明", "投影体积为核心区域内1 cm网格面积乘以对应离地高度后求和"],
];

const recordHeaders = Object.keys(payload.records[0]);
records.getRangeByIndexes(0, 0, 1, recordHeaders.length).values = [recordHeaders];
records.getRangeByIndexes(1, 0, payload.records.length, recordHeaders.length).values = payload.records.map(
  (record) => recordHeaders.map((header) => record[header] ?? null),
);

const parameterRows = [
  ["参数", "数值", "说明"],
  ["方法版本", s.method_version, "新的几何中线核心区域"],
  ["核心区域", payload.parameters.core_region, "使用已经确定的新头侧和屁股侧边界"],
  ["网格边长（m）", payload.parameters.projection_grid_cell_size_m, "1 cm"],
  ["网格面积（m²）", payload.parameters.projection_grid_cell_area_m2, "0.0001 m²"],
  ["高度定义", payload.parameters.height_definition, "地面平面高度减牛背表面高度坐标"],
  ["单格高度", payload.parameters.cell_height_aggregation, "同一网格内取高度中位数，降低噪声影响"],
  ["体积公式", payload.parameters.volume_formula, "所有小柱体积累加"],
  ["单位", "m³，同时记录L", "1 m³ = 1000 L"],
];
parameters.getRangeByIndexes(0, 0, parameterRows.length, 3).values = parameterRows;

const fontFamily = "Arial";
for (const sheet of [overview, records, parameters]) {
  sheet.showGridLines = false;
  sheet.getUsedRange().format.font = { name: fontFamily, size: 10 };
  sheet.getUsedRange().format.verticalAlignment = "center";
}
overview.getRange("A1:B1").format.font = { name: fontFamily, size: 15, bold: true };
overview.getRange("A2:B2").format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" } };
overview.getRange("A2:B25").format.borders = { insideHorizontal: { style: "thin", color: "#D7DEE8" } };
overview.getRange("A1:A25").format.columnWidth = 36;
overview.getRange("B1:B25").format.columnWidth = 72;
overview.getRange("B3:B25").format.wrapText = true;
overview.getRange("B4:B8").format.numberFormat = "0.000000";
overview.getRange("B9:B10").format.numberFormat = "0.00%";
overview.getRange("B11:B11").format.numberFormat = "0.000000";
overview.getRange("B14:B23").format.numberFormat = "0.000000";

const header = records.getRangeByIndexes(0, 0, 1, recordHeaders.length);
header.format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
header.format.rowHeight = 42;
records.freezePanes.freezeRows(1);
records.getUsedRange().format.columnWidth = 18;
records.getRange(`A2:B${payload.records.length + 1}`).format.numberFormat = "0";
records.getRange(`C2:S${payload.records.length + 1}`).format.numberFormat = "0.000000";
records.getRange(`T2:X${payload.records.length + 1}`).format.columnWidth = 30;
records.tables.add(`A1:X${payload.records.length + 1}`, true, "CoreVolumeRecords").style = "TableStyleMedium2";

parameters.getRange("A1:C1").format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
parameters.getRange("A1:A9").format.columnWidth = 28;
parameters.getRange("B1:B9").format.columnWidth = 58;
parameters.getRange("C1:C9").format.columnWidth = 62;
parameters.getRange("A1:C9").format.wrapText = true;
parameters.getRange("B4:B5").format.numberFormat = "0.000000";
parameters.tables.add("A1:C9", true, "CoreVolumeParameters").style = "TableStyleMedium2";

console.log((await workbook.inspect({ kind: "table", range: "体积概览!A1:B25", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 2 })).ndjson);
console.log((await workbook.inspect({ kind: "table", range: "每头牛核心体积!A1:X8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 24 })).ndjson);
console.log((await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" })).ndjson);

for (const [sheetName, range, fileName] of [
  ["体积概览", "A1:B25", "preview_volume_overview.png"],
  ["每头牛核心体积", "A1:X14", "preview_volume_records.png"],
  ["体积方法参数", "A1:C9", "preview_volume_parameters.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(JSON.stringify({ output: workbookPath, cows: payload.records.length }));
