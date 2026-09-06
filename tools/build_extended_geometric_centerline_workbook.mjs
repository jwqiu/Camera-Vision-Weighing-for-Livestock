import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const axisDir = `${root}/cattle_3d_extraction_66/skeleton_axis_comparison`;
const payload = JSON.parse(await fs.readFile(`${axisDir}/extended_geometric_centerline_records.json`, "utf8"));
const outputDir = `${root}/outputs/extended_geometric_centerlines`;
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const notes = workbook.worksheets.add("说明");
const summary = workbook.worksheets.add("延伸中线汇总");
const points = workbook.worksheets.add("延伸中线坐标点");

notes.getRange("A1:B11").values = [
  ["66头牛后躯延伸几何中线记录", ""],
  ["项目", "说明"],
  ["记录版本", payload.record_version],
  ["中线方向", "rear_to_head，逐点顺序为后躯到头部"],
  ["图片坐标", "左上角为原点，x向右、y向下，单位为像素"],
  ["传感器坐标", "sensor_u_px、sensor_v_px对应512×424原始顶视深度图"],
  ["点云坐标", "camera_x_m、camera_y_m、camera_z_m，单位为米"],
  ["长度定义", "polyline_length_xy_m按点云x-y平面的相邻点距离累计"],
  ["PCA中轴", "保留在 long_axis_records.csv 的 pca_axis_* 字段"],
  ["原始骨架中线", "保留在 geometric_centerline_records.* 文件"],
  ["本次记录", "新增 extended_geometric_centerline_* 文件和主表字段，未覆盖旧记录"],
];

const summaryHeaders = ["cow_id", "record_type", "method_version", "orientation", "centerline_point_count", "added_rear_extension_point_count", "polyline_length_xy_m", "direct_distance_xy_m", "curvature_ratio_xy", "rear_x_m", "rear_y_m", "head_x_m", "head_y_m", "maximum_pixel_mapping_distance_px", "mean_pixel_mapping_distance_px", "status", "preview_image"];
summary.getRangeByIndexes(0, 0, 1, summaryHeaders.length).values = [summaryHeaders];
summary.getRangeByIndexes(1, 0, payload.records.length, summaryHeaders.length).values = payload.records.map((record) => summaryHeaders.map((header) => record[header] ?? null));

const pointHeaders = ["cow_id", "centerline_type", "method_version", "orientation", "point_index", "image_x_px", "image_y_px", "sensor_u_px", "sensor_v_px", "pixel_mapping_distance_px", "camera_x_m", "camera_y_m", "camera_z_m", "height_above_ground_m", "cumulative_xy_length_m"];
const pointRows = [];
for (const record of payload.records) {
  for (const point of record.points) pointRows.push(pointHeaders.map((header) => point[header] ?? null));
}
points.getRangeByIndexes(0, 0, 1, pointHeaders.length).values = [pointHeaders];
for (let start = 0; start < pointRows.length; start += 1000) {
  const chunk = pointRows.slice(start, start + 1000);
  points.getRangeByIndexes(start + 1, 0, chunk.length, pointHeaders.length).values = chunk;
}

const fontFamily = "Arial";
for (const sheet of [notes, summary, points]) {
  sheet.showGridLines = false;
  sheet.getUsedRange().format.font = { name: fontFamily, size: 10 };
  sheet.getUsedRange().format.verticalAlignment = "center";
}
notes.getRange("A1:B1").format.font = { name: fontFamily, size: 15, bold: true };
notes.getRange("A2:B2").format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" } };
notes.getRange("A2:B11").format.borders = { insideHorizontal: { style: "thin", color: "#D7DEE8" } };
notes.getRange("A1:A11").format.columnWidth = 22;
notes.getRange("B1:B11").format.columnWidth = 78;
notes.getRange("B3:B11").format.wrapText = true;
notes.getRange("A1:B11").format.autofitRows();

for (const [sheet, width] of [[summary, summaryHeaders.length], [points, pointHeaders.length]]) {
  const header = sheet.getRangeByIndexes(0, 0, 1, width);
  header.format = { fill: "#334155", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center" };
  header.format.rowHeight = 34;
  sheet.freezePanes.freezeRows(1);
}

const summaryLastRow = payload.records.length + 1;
summary.getRange(`A2:A${summaryLastRow}`).format.numberFormat = "0";
summary.getRange(`E2:F${summaryLastRow}`).format.numberFormat = "0";
summary.getRange(`G2:O${summaryLastRow}`).format.numberFormat = "0.000000";
summary.getRange(`A1:P${summaryLastRow}`).format.columnWidth = 18;
summary.getRange(`B1:D${summaryLastRow}`).format.columnWidth = 32;
summary.getRange(`Q1:Q${summaryLastRow}`).format.columnWidth = 58;

const pointsLastRow = pointRows.length + 1;
points.getRange(`A2:A${pointsLastRow}`).format.numberFormat = "0";
points.getRange(`E2:E${pointsLastRow}`).format.numberFormat = "0";
points.getRange(`F2:G${pointsLastRow}`).format.numberFormat = "0.000";
points.getRange(`H2:I${pointsLastRow}`).format.numberFormat = "0";
points.getRange(`J2:J${pointsLastRow}`).format.numberFormat = "0.000";
points.getRange(`K2:O${pointsLastRow}`).format.numberFormat = "0.000000";
points.getRange(`A1:O${pointsLastRow}`).format.columnWidth = 18;
points.getRange(`B1:D${pointsLastRow}`).format.columnWidth = 34;

summary.tables.add(`A1:Q${summaryLastRow}`, true, "ExtendedCenterlineSummary").style = "TableStyleMedium2";
points.tables.add(`A1:O${pointsLastRow}`, true, "ExtendedCenterlinePoints").style = "TableStyleMedium2";

console.log((await workbook.inspect({ kind: "table", range: "延伸中线汇总!A1:Q8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 17 })).ndjson);
console.log((await workbook.inspect({ kind: "table", range: "延伸中线坐标点!A1:O12", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 15 })).ndjson);
console.log((await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 50 }, summary: "final formula error scan" })).ndjson);

for (const [sheetName, range, fileName] of [["说明", "A1:B11", "preview_notes.png"], ["延伸中线汇总", "A1:Q15", "preview_summary.png"], ["延伸中线坐标点", "A1:O20", "preview_points.png"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const outputPath = `${outputDir}/extended_geometric_centerline_records_66.xlsx`;
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ output: outputPath, cows: payload.records.length, points: pointRows.length }));
