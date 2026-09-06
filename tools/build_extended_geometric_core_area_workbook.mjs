import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const axisDir = `${root}/cattle_3d_extraction_66/skeleton_axis_comparison`;
const payload = JSON.parse(await fs.readFile(`${axisDir}/extended_geometric_core_area_comparison.json`, "utf8"));
const outputDir = `${root}/outputs/extended_geometric_core_areas`;
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const overview = workbook.worksheets.add("差异概览");
const records = workbook.worksheets.add("每头牛核心面积");
const parameters = workbook.worksheets.add("方法参数");
const c = payload.comparison;

overview.getRange("A1:B25").values = [
  ["几何中线核心躯干面积与PCA中轴结果对比", ""],
  ["指标", "结果"],
  ["牛数量", c.n],
  ["旧方法", c.old_measure],
  ["新方法", c.new_measure],
  ["PCA方法平均面积（m²）", c.mean_old_area_m2],
  ["几何中线方法平均面积（m²）", c.mean_new_area_m2],
  ["平均差：新−旧（m²）", c.mean_signed_difference_m2],
  ["中位差：新−旧（m²）", c.median_signed_difference_m2],
  ["平均绝对差（m²）", c.mean_absolute_difference_m2],
  ["中位绝对差（m²）", c.median_absolute_difference_m2],
  ["RMSE（m²）", c.rmse_difference_m2],
  ["平均差：新−旧（%）", c.mean_signed_difference_percent / 100],
  ["平均绝对差（%）", c.mean_absolute_difference_percent / 100],
  ["最大绝对差（m²）", c.maximum_absolute_difference_m2],
  ["两套面积 Pearson r", c.pearson_correlation_old_vs_new],
  ["差值标准差（m²）", c.difference_standard_deviation_m2],
  ["95%一致性下限（m²）", c.agreement_lower_95_m2],
  ["95%一致性上限（m²）", c.agreement_upper_95_m2],
  ["新面积较大的牛数", c.count_new_larger],
  ["新面积较小的牛数", c.count_new_smaller],
  ["绝对差超过0.05 m²的牛数", c.count_absolute_difference_over_0_05_m2],
  ["绝对百分比差超过10%的牛数", c.count_absolute_percent_over_10],
  ["需人工复核的牛", c.review_required_cows.length ? c.review_required_cows.join(", ") : "无"],
  ["判断", "整体高度一致，但几何中线方法存在约5%的系统性增大"],
];

const recordHeaders = Object.keys(payload.records[0]);
records.getRangeByIndexes(0, 0, 1, recordHeaders.length).values = [recordHeaders];
records.getRangeByIndexes(1, 0, payload.records.length, recordHeaders.length).values = payload.records.map(
  (record) => recordHeaders.map((header) => record[header] ?? null),
);

const parameterRows = [
  ["参数", "数值", "说明"],
  ["方法版本", c.method_version, "几何中线自适应体宽核心面积"],
  ["头侧宽度阈值", payload.parameters.head_width_threshold_ratio, "相对最大体宽比例"],
  ["头侧连续长度（m）", payload.parameters.head_continuous_length_m, "连续低于阈值后停止"],
  ["屁股侧宽度阈值", payload.parameters.butt_width_threshold_ratio, "相对最大体宽比例"],
  ["屁股侧连续长度（m）", payload.parameters.butt_continuous_length_m, "连续低于阈值后停止"],
  ["骤降立即停止阈值", payload.parameters.immediate_stop_ratio, "任一截面低于最大体宽该比例时立即停止"],
  ["中线采样步长（m）", payload.parameters.profile_step_m, "沿几何中线计算"],
  ["投影网格边长（m）", payload.parameters.projection_grid_cell_size_m, "1 cm × 1 cm"],
  ["点分配规则", payload.parameters.point_assignment, "牛点分配到最近的几何中线位置"],
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
overview.getRange("A1:A25").format.columnWidth = 34;
overview.getRange("B1:B25").format.columnWidth = 78;
overview.getRange("B3:B25").format.wrapText = true;
overview.getRange("B6:B12").format.numberFormat = "0.000000";
overview.getRange("B13:B14").format.numberFormat = "0.00%";
overview.getRange("B15:B19").format.numberFormat = "0.000000";

const recordHeader = records.getRangeByIndexes(0, 0, 1, recordHeaders.length);
recordHeader.format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
recordHeader.format.rowHeight = 42;
records.freezePanes.freezeRows(1);
records.getUsedRange().format.columnWidth = 18;
records.getRange(`A2:A${payload.records.length + 1}`).format.numberFormat = "0";
records.getRange(`B2:P${payload.records.length + 1}`).format.numberFormat = "0.000000";
records.getRange(`Q2:X${payload.records.length + 1}`).format.columnWidth = 29;
records.tables.add(`A1:X${payload.records.length + 1}`, true, "CoreAreaComparisonRecords").style = "TableStyleMedium2";

parameters.getRange("A1:C1").format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
parameters.getRange("A1:A10").format.columnWidth = 30;
parameters.getRange("B1:B10").format.columnWidth = 43;
parameters.getRange("C1:C10").format.columnWidth = 58;
parameters.getRange("A1:C10").format.wrapText = true;
parameters.getRange("B3:B9").format.numberFormat = "0.000000";
parameters.tables.add("A1:C10", true, "CoreAreaParameters").style = "TableStyleMedium2";

console.log((await workbook.inspect({ kind: "table", range: "差异概览!A1:B25", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 2 })).ndjson);
console.log((await workbook.inspect({ kind: "table", range: "每头牛核心面积!A1:X8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 24 })).ndjson);
console.log((await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 50 }, summary: "final formula error scan" })).ndjson);

for (const [sheetName, range, fileName] of [
  ["差异概览", "A1:B25", "preview_overview.png"],
  ["每头牛核心面积", "A1:X14", "preview_records.png"],
  ["方法参数", "A1:C10", "preview_parameters.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const outputPath = `${outputDir}/extended_geometric_core_area_comparison_66.xlsx`;
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ output: outputPath, cows: payload.records.length }));
