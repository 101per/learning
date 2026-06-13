package student;

import java.util.ArrayList;
import java.util.List;
import javax.swing.table.AbstractTableModel;

public class StudentTableModel extends AbstractTableModel {
    private final String[] columns = {
            "学号", "姓名", "英语", "高数", "C语言", "Java程序设计", "总分", "平均分", "预警科目"
    };
    private List<Student> students = new ArrayList<>();

    public void setStudents(List<Student> students) {
        this.students = new ArrayList<>(students);
        fireTableDataChanged();
    }

    public Student getStudentAt(int rowIndex) {
        return students.get(rowIndex);
    }

    @Override
    public int getRowCount() {
        return students.size();
    }

    @Override
    public int getColumnCount() {
        return columns.length;
    }

    @Override
    public String getColumnName(int column) {
        return columns[column];
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        Student student = students.get(rowIndex);
        return switch (columnIndex) {
            case 0 -> student.getId();
            case 1 -> student.getName();
            case 2 -> student.getEnglish();
            case 3 -> student.getAdvancedMath();
            case 4 -> student.getCLanguage();
            case 5 -> student.getJavaProgramming();
            case 6 -> student.getTotalScore();
            case 7 -> student.getAverageScore();
            case 8 -> student.isAlert() ? student.getAlertCourses() : "";
            default -> "";
        };
    }
}
