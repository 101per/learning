package student;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.DataFormat;
import org.apache.poi.ss.usermodel.Font;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class ExcelExporter {
    private static final int[] MIN_COLUMN_WIDTHS = {
            14, 16, 14, 14, 14, 20, 12, 12, 24
    };
    private static final String[] HEADERS = {
            "学号", "姓名", "英语", "高数", "C语言", "Java程序设计", "总分", "平均分", "预警科目"
    };

    public void export(List<Student> students, Path outputPath) throws IOException {
        Path parent = outputPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (Workbook workbook = new XSSFWorkbook()) {
            Sheet sheet = workbook.createSheet("学生成绩");
            CellStyle headerStyle = createHeaderStyle(workbook);
            CellStyle numberStyle = createNumberStyle(workbook);

            Row headerRow = sheet.createRow(0);
            for (int i = 0; i < HEADERS.length; i++) {
                Cell cell = headerRow.createCell(i);
                cell.setCellValue(HEADERS[i]);
                cell.setCellStyle(headerStyle);
            }

            for (int rowIndex = 0; rowIndex < students.size(); rowIndex++) {
                writeStudentRow(sheet.createRow(rowIndex + 1), students.get(rowIndex), numberStyle);
            }

            for (int i = 0; i < HEADERS.length; i++) {
                sheet.autoSizeColumn(i);
                int minimumWidth = MIN_COLUMN_WIDTHS[i] * 256;
                if (sheet.getColumnWidth(i) < minimumWidth) {
                    sheet.setColumnWidth(i, minimumWidth);
                }
            }

            try (OutputStream outputStream = Files.newOutputStream(outputPath)) {
                workbook.write(outputStream);
            }
        }
    }

    private CellStyle createHeaderStyle(Workbook workbook) {
        Font font = workbook.createFont();
        font.setBold(true);
        CellStyle style = workbook.createCellStyle();
        style.setFont(font);
        return style;
    }

    private CellStyle createNumberStyle(Workbook workbook) {
        DataFormat dataFormat = workbook.createDataFormat();
        CellStyle style = workbook.createCellStyle();
        style.setDataFormat(dataFormat.getFormat("0.00"));
        return style;
    }

    private void writeStudentRow(Row row, Student student, CellStyle numberStyle) {
        row.createCell(0).setCellValue(student.getId());
        row.createCell(1).setCellValue(student.getName());
        writeNumber(row, 2, student.getEnglish(), numberStyle);
        writeNumber(row, 3, student.getAdvancedMath(), numberStyle);
        writeNumber(row, 4, student.getCLanguage(), numberStyle);
        writeNumber(row, 5, student.getJavaProgramming(), numberStyle);
        writeNumber(row, 6, student.getTotalScore(), numberStyle);
        writeNumber(row, 7, student.getAverageScore(), numberStyle);
        row.createCell(8).setCellValue(student.getAlertCourses());
    }

    private void writeNumber(Row row, int column, double value, CellStyle numberStyle) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value);
        cell.setCellStyle(numberStyle);
    }
}
