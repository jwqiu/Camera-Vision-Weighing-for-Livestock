import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const skeletonDir = `${root}/cattle_3d_extraction_66/skeleton_axis_comparison`;
const payload = JSON.parse(
  await fs.readFile(`${skeletonDir}/geometric_centerline_records.json`, "utf8"),
);
const outputDir = `${root}/outputs/geometric_centerlines`;
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const notes = workbook.worksheets.add("说明");
const summary = workbook.worksheets.add("几何中线汇总");
const points = workbook.worksheets.add("几何中线坐标点");

notes.getRange("A1:B8").values = [
  ["66头牛几何中线记录", ""],
  ["项目", "说明"],
  ["几何中线", "从现有骨架中线叠加图恢复并记录的弯曲中线坐标"],
  ["PCA中轴", "仍保留在 long_axis_records.csv；未覆盖、未改名"],
  ["坐标原点", "图片左上角"],
  ["坐标方向", "x 向右增大，y 向下增大"],
  ["点序", "从图片左侧到图片右侧；不代表固定的头尾方向"],
  ["方法版本", payload.record_version],
];

const summaryHeaders = [
  "cow_id",
  "record_type",
  "method_version",
  "coordinate_system",
  "source_image_width_px",
  "source_image_height_px",
  "centerline_point_count",
  "start_x_px",
  "start_y_px",
  "end_x_px",
  "end_y_px",
  "polyline_length_px",
  "direct_distance_px",
  "curvature_ratio",
  "source_skeleton_image",
];
summary.getRangeByIndexes(0, 0, 1, summaryHeaders.length).values = [summaryHeaders];
summary.getRangeByIndexes(1, 0, payload.records.length, summaryHeaders.length).values =
  payload.records.map((record) => summaryHeaders.map((header) => record[header] ?? null));

const pointHeaders = [
  "cow_id",
  "centerline_type",
  "method_version",
  "point_index",
  "x_px",
  "y_px",
  "x_normalized",
  "y_normalized",
  "source_image_width_px",
  "source_image_height_px",
];
const pointRows = [];
for (const record of payload.records) {
  for (const point of record.points) {
    pointRows.push([
      record.cow_id,
      "geometric_skeleton",
      payload.record_version,
      point.point_index,
      point.x_px,
      point.y_px,
      point.x_px / (record.source_image_width_px - 1),
      point.y_px / (record.source_image_height_px - 1),
      record.source_image_width_px,
      record.source_image_height_px,
    ]);
  }
}
points.getRangeByIndexes(0, 0, 1, pointHeaders.length).values = [pointHeaders];
const chunkSize = 1000;
for (let start = 0; start < pointRows.length; start += chunkSize) {
  const chunk = pointRows.slice(start, start + chunkSize);
  points.getRangeByIndexes(start + 1, 0, chunk.length, pointHeaders.length).values = chunk;
}

const fontFamily = "Arial";
for (const sheet of [notes, summary, points]) {
  sheet.showGridLines = false;
  const used = sheet.getUsedRange();
  used.format.font = { name: fontFamily, size: 10 };
  used.format.verticalAlignment = "center";
}

notes.getRange("A1:B1").format.font = { name: fontFamily, size: 15, bold: true };
notes.getRange("A2:B2").format = {
  fill: "#334155",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
};
notes.getRange("A2:B8").format.borders = { preset: "insideHorizontal", style: "thin", color: "#D7DEE8" };
notes.getRange("A1:A8").format.columnWidth = 20;
notes.getRange("B1:B8").format.columnWidth = 75;
notes.getRange("B3:B8").format.wrapText = true;
notes.getRange("A1:B8").format.autofitRows();

for (const [sheet, width] of [[summary, summaryHeaders.length], [points, pointHeaders.length]]) {
  const header = sheet.getRangeByIndexes(0, 0, 1, width);
  header.format = {
    fill: "#334155",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  header.format.rowHeight = 32;
  sheet.freezePanes.freezeRows(1);
}

summary.getRange("A2:A67").format.numberFormat = "0";
summary.getRange("E2:G67").format.numberFormat = "0";
summary.getRange("H2:M67").format.numberFormat = "0.000";
summary.getRange("N2:N67").format.numberFormat = "0.000000";
summary.getRange("A1:N67").format.columnWidth = 16;
summary.getRange("B1:D67").format.columnWidth = 31;
summary.getRange("O1:O67").format.columnWidth = 55;

points.getRange("A2:A7025").format.numberFormat = "0";
points.getRange("D2:D7025").format.numberFormat = "0";
points.getRange("E2:F7025").format.numberFormat = "0.000";
points.getRange("G2:H7025").format.numberFormat = "0.00000000";
points.getRange("I2:J7025").format.numberFormat = "0";
points.getRange("A1:J7025").format.columnWidth = 18;
points.getRange("B1:C7025").format.columnWidth = 34;

summary.tables.add("A1:O67", true, "GeometricCenterlineSummary").style = "TableStyleMedium2";
points.tables.add(`A1:J${pointRows.length + 1}`, true, "GeometricCenterlinePoints").style = "TableStyleMedium2";

const summaryCheck = await workbook.inspect({
  kind: "table",
  range: "几何中线汇总!A1:O8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 15,
});
console.log(summaryCheck.ndjson);
const pointCheck = await workbook.inspect({
  kind: "table",
  range: "几何中线坐标点!A1:J12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 10,
});
console.log(pointCheck.ndjson);
const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});
console.log(errorCheck.ndjson);

for (const [sheetName, range, fileName] of [
  ["说明", "A1:B8", "preview_notes.png"],
  ["几何中线汇总", "A1:O15", "preview_summary.png"],
  ["几何中线坐标点", "A1:J20", "preview_points.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.3, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/geometric_centerline_records_66.xlsx`);
console.log(JSON.stringify({ output: `${outputDir}/geometric_centerline_records_66.xlsx`, rows: pointRows.length }));
