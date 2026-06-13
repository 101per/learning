package student;

import java.util.ArrayList;
import java.util.List;
import javax.swing.table.AbstractTableModel;

public class RankingTableModel extends AbstractTableModel {
    private final String[] columns = {
            "名次", "学号", "姓名", "英语", "高数", "C语言", "Java程序设计", "总分", "平均分"
    };
    private List<Student> students = new ArrayList<>();

    public void setStudents(List<Student> students) {
        this.students = new ArrayList<>(students);
        fireTableDataChanged();
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
            case 0 -> rowIndex + 1;
            case 1 -> student.getId();
            case 2 -> student.getName();
            case 3 -> student.getEnglish();
            case 4 -> student.getAdvancedMath();
            case 5 -> student.getCLanguage();
            case 6 -> student.getJavaProgramming();
            case 7 -> student.getTotalScore();
            case 8 -> student.getAverageScore();
            default -> "";
        };
    }
}
