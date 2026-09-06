import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const path = process.argv[2];
const workbook = await Workbook.fromCSV(await fs.readFile(path, "utf8"), { sheetName: "Data" });
for (const range of ["GP1:HD4", "EH5:HD5", "EH9:HD9", "EH38:HD38", "EH63:HD65", "GP66:HD67"]) {
  const check = await workbook.inspect({
    kind: "table",
    sheetId: "Data",
    range,
    include: "values,formulas",
    tableMaxRows: 4,
    tableMaxCols: 25,
    summary: "Right-view body-depth values and excluded/pending rows",
  });
  console.log(check.ndjson);
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 30 },
  summary: "formula error scan",
});
console.log(errors.ndjson);
