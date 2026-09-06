import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const [recordsPath, resultsPath] = process.argv.slice(2);
if (!recordsPath || !resultsPath) {
  throw new Error("Usage: node append_rightview_body_depth.mjs RECORDS.csv RESULTS.csv");
}

const recordsBook = await Workbook.fromCSV(await fs.readFile(recordsPath, "utf8"), { sheetName: "Data" });
const resultsBook = await Workbook.fromCSV(await fs.readFile(resultsPath, "utf8"), { sheetName: "Results" });
const recordsSheet = recordsBook.worksheets.getItem("Data");
const resultsSheet = resultsBook.worksheets.getItem("Results");
const records = recordsSheet.getUsedRange(true).values;
const results = resultsSheet.getUsedRange(true).values;
records[0][0] = String(records[0][0]).replace(/^\uFEFF+/, "");
recordsSheet.getCell(0, 0).values = [[records[0][0]]];

const wantedFields = results[0].slice(1).map(String);
const resultByCow = new Map(results.slice(1).map((row) => [String(row[0]), row]));
const existingHeader = records[0].map(String);
const existingIndexes = wantedFields.map((field) => existingHeader.indexOf(field));
const appendFields = wantedFields.filter((_, index) => existingIndexes[index] < 0);

if (appendFields.length) {
  const startColumn = existingHeader.length;
  const appendMatrix = records.map((record, rowIndex) => {
    if (rowIndex === 0) return appendFields;
    const result = resultByCow.get(String(record[0]));
    return appendFields.map((field) => {
      if (!result) return "";
      return result[results[0].indexOf(field)] ?? "";
    });
  });
  recordsSheet.getRangeByIndexes(0, startColumn, appendMatrix.length, appendFields.length).values = appendMatrix;
}

for (let fieldIndex = 0; fieldIndex < wantedFields.length; fieldIndex++) {
  const field = wantedFields[fieldIndex];
  const columnIndex = recordsSheet.getUsedRange(true).values[0].map(String).indexOf(field);
  const column = records.map((record, rowIndex) => {
    if (rowIndex === 0) return [field];
    const result = resultByCow.get(String(record[0]));
    return [result ? (result[fieldIndex + 1] ?? "") : ""];
  });
  recordsSheet.getRangeByIndexes(0, columnIndex, column.length, 1).values = column;
}

const verification = await recordsBook.inspect({
  kind: "table",
  sheetId: "Data",
  range: "GP1:HD67",
  include: "values,formulas",
  tableMaxRows: 5,
  tableMaxCols: 15,
  summary: "Right-view median body-depth fields",
});
console.log(verification.ndjson);

const finalValues = recordsSheet.getUsedRange(true).values;
function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
const csvText = "\ufeff" + finalValues.map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
await fs.writeFile(recordsPath, csvText, "utf8");
console.log(`updated ${recordsPath}: ${finalValues.length - 1} rows, ${finalValues[0].length} columns`);
