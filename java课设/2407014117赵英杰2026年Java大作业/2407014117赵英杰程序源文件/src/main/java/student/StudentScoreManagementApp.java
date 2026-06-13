package student;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.SwingUtilities;
import javax.swing.table.DefaultTableCellRenderer;

public class StudentScoreManagementApp extends JFrame {
    private static final DateTimeFormatter FILE_TIME_FORMAT =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss");

    private final StudentManager manager = new StudentManager();
    private final StudentRepository repository = new StudentRepository(Path.of("students.txt"));
    private final StudentTableModel tableModel = new StudentTableModel();
    private final JTable table = new JTable(tableModel);
    private final JTextField idField = new JTextField();
    private final JTextField nameField = new JTextField();
    private final JTextField englishField = new JTextField();
    private final JTextField mathField = new JTextField();
    private final JTextField cField = new JTextField();
    private final JTextField javaField = new JTextField();
    private final JLabel resultLabel = new JLabel("请选择学生后点击计算按钮");
    private final JLabel statusLabel = new JLabel("自动备份已启动：每60秒备份一次");
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private final BackupService backupService = new BackupService(
            Path.of("backups"),
            manager::getAll,
            message -> SwingUtilities.invokeLater(() -> statusLabel.setText(message)));
    private RankingWindow rankingWindow;
    private ChartWindow chartWindow;

    public StudentScoreManagementApp() {
        super("学生成绩管理系统");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1040, 620);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(10, 10));

        add(buildFormPanel(), BorderLayout.NORTH);
        add(new JScrollPane(table), BorderLayout.CENTER);
        add(buildSouthPanel(), BorderLayout.SOUTH);

        configureTable();
        manager.addChangeListener(() -> SwingUtilities.invokeLater(this::refreshTable));
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent event) {
                backupService.close();
            }
        });

        loadOnStartup();
        backupService.start(60, 60, TimeUnit.SECONDS);
    }

    private JPanel buildFormPanel() {
        JPanel panel = new JPanel(new GridLayout(2, 6, 8, 6));
        panel.add(new JLabel("学号"));
        panel.add(new JLabel("姓名"));
        panel.add(new JLabel("英语"));
        panel.add(new JLabel("高数"));
        panel.add(new JLabel("C语言"));
        panel.add(new JLabel("Java程序设计"));
        panel.add(idField);
        panel.add(nameField);
        panel.add(englishField);
        panel.add(mathField);
        panel.add(cField);
        panel.add(javaField);
        return panel;
    }

    private JPanel buildSouthPanel() {
        JPanel southPanel = new JPanel(new BorderLayout());
        southPanel.add(buildButtonPanel(), BorderLayout.CENTER);
        southPanel.add(statusLabel, BorderLayout.SOUTH);
        return southPanel;
    }

    private JPanel buildButtonPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton addButton = new JButton("增加");
        JButton deleteButton = new JButton("删除");
        JButton queryButton = new JButton("查询");
        JButton saveButton = new JButton("保存");
        JButton editButton = new JButton("详情/编辑");
        JButton rankingButton = new JButton("实时排名");
        JButton chartButton = new JButton("一键图表");
        JButton exportButton = new JButton("导出Excel");
        JButton averageButton = new JButton("平均成绩");
        JButton deviationButton = new JButton("标准差");
        JButton clearButton = new JButton("清空");
        JButton alertButton = new JButton("预警统计");

        addButton.addActionListener(event -> addStudent());
        deleteButton.addActionListener(event -> deleteStudent());
        queryButton.addActionListener(event -> queryStudent());
        saveButton.addActionListener(event -> saveStudents());
        editButton.addActionListener(event -> openEditorForSelection());
        rankingButton.addActionListener(event -> openRankingWindow());
        chartButton.addActionListener(event -> openChartWindow());
        exportButton.addActionListener(event -> exportExcel());
        averageButton.addActionListener(event -> calculate(new AverageScoreCalculator()));
        deviationButton.addActionListener(event -> calculate(new StandardDeviationCalculator()));
        clearButton.addActionListener(event -> clearForm());
        alertButton.addActionListener(event -> showAlertStatistics());

        panel.add(addButton);
        panel.add(deleteButton);
        panel.add(queryButton);
        panel.add(saveButton);
        panel.add(editButton);
        panel.add(rankingButton);
        panel.add(chartButton);
        panel.add(exportButton);
        panel.add(averageButton);
        panel.add(deviationButton);
        panel.add(clearButton);
        panel.add(alertButton);
        panel.add(resultLabel);
        return panel;
    }

    private void configureTable() {
        table.setAutoCreateRowSorter(true);
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        table.getSelectionModel().addListSelectionListener(event -> {
            if (!event.getValueIsAdjusting() && table.getSelectedRow() >= 0) {
                fillForm(getSelectedStudent());
            }
        });
        table.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent event) {
                if (event.getClickCount() == 2 && table.getSelectedRow() >= 0) {
                    openEditorForSelection();
                }
            }
        });
        table.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable table, Object value,
                    boolean isSelected, boolean hasFocus, int row, int column) {
                Component component = super.getTableCellRendererComponent(
                        table, value, isSelected, hasFocus, row, column);
                Student student = tableModel.getStudentAt(table.convertRowIndexToModel(row));
                if (student.isAlert() && !isSelected) {
                    component.setForeground(Color.RED);
                } else {
                    component.setForeground(isSelected ? table.getSelectionForeground() : table.getForeground());
                }
                return component;
            }
        });
    }

    private void loadOnStartup() {
        try {
            List<Student> students = repository.load();
            if (students.isEmpty()) {
                Path legacyPath = Path.of("2407014117赵英杰程序源文件", "students.txt");
                if (Files.exists(legacyPath)) {
                    students = new StudentRepository(legacyPath).load();
                }
            }
            manager.setStudents(students);
        } catch (IOException | RuntimeException exception) {
            showMessage("启动读取数据失败: " + exception.getMessage());
            refreshTable();
        }
    }

    private void addStudent() {
        Student student = readForm();
        if (student == null) {
            return;
        }
        if (!manager.addStudent(student)) {
            showMessage("学号已存在，不能重复增加");
            return;
        }
        clearForm();
    }

    private void deleteStudent() {
        String id = idField.getText().trim();
        if (id.isEmpty()) {
            showMessage("请先选择或输入要删除的学号");
            return;
        }
        if (!manager.deleteStudent(id)) {
            showMessage("未找到该学号");
            return;
        }
        clearForm();
    }

    private void updateStudent() {
        Student student = readForm();
        if (student == null) {
            return;
        }
        if (!manager.updateStudent(student)) {
            showMessage("未找到该学号，无法修改");
        }
    }

    private void queryStudent() {
        String id = idField.getText().trim();
        if (id.isEmpty()) {
            refreshTable();
            showMessage("已显示全部学生");
            return;
        }
        manager.findById(id).ifPresentOrElse(student -> {
            tableModel.setStudents(List.of(student));
            fillForm(student);
        }, () -> showMessage("未找到该学号"));
    }

    private void saveStudents() {
        try {
            repository.save(manager.getAll());
            showMessage("数据已保存到 students.txt");
        } catch (IOException exception) {
            showMessage("保存失败: " + exception.getMessage());
        }
    }

    private void openEditorForSelection() {
        Student student = getCurrentStudent();
        if (student == null) {
            showMessage("请先选择或输入一个学生");
            return;
        }
        new StudentEditorDialog(this, manager, student).setVisible(true);
    }

    private void openRankingWindow() {
        if (rankingWindow == null || !rankingWindow.isDisplayable()) {
            rankingWindow = new RankingWindow(manager);
        }
        rankingWindow.setVisible(true);
        rankingWindow.toFront();
    }

    private void openChartWindow() {
        if (chartWindow == null || !chartWindow.isDisplayable()) {
            chartWindow = new ChartWindow(manager);
        }
        chartWindow.setVisible(true);
        chartWindow.toFront();
    }

    private void exportExcel() {
        try {
            Path outputPath = Path.of("exports", "students-"
                    + LocalDateTime.now().format(FILE_TIME_FORMAT) + ".xlsx");
            new ExcelExporter().export(manager.getAll(), outputPath);
            showMessage("Excel已导出: " + outputPath.toAbsolutePath());
        } catch (IOException exception) {
            showMessage("导出Excel失败: " + exception.getMessage());
        }
    }

    private void calculate(ScoreCalculator calculator) {
        Student student = getCurrentStudent();
        if (student == null) {
            showMessage("请先选择或输入一个学生");
            return;
        }
        double value = calculator.calculate(student);
        String result = calculator.getName() + ": " + decimalFormat.format(value);
        resultLabel.setText(result);
        statusLabel.setText(result);
    }

    private Student getCurrentStudent() {
        String id = idField.getText().trim();
        if (!id.isEmpty()) {
            return manager.findById(id).orElseGet(this::readForm);
        }
        if (table.getSelectedRow() >= 0) {
            return getSelectedStudent();
        }
        return null;
    }

    private Student getSelectedStudent() {
        return tableModel.getStudentAt(table.convertRowIndexToModel(table.getSelectedRow()));
    }

    private Student readForm() {
        try {
            return new Student(
                    idField.getText(),
                    nameField.getText(),
                    parseScore(englishField.getText(), "英语"),
                    parseScore(mathField.getText(), "高数"),
                    parseScore(cField.getText(), "C语言"),
                    parseScore(javaField.getText(), "Java程序设计"));
        } catch (IllegalArgumentException exception) {
            showMessage(exception.getMessage());
            return null;
        }
    }

    private double parseScore(String text, String courseName) {
        try {
            return Double.parseDouble(text.trim());
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(courseName + "成绩必须是数字");
        }
    }

    private void refreshTable() {
        tableModel.setStudents(manager.getAll());
    }

    private void fillForm(Student student) {
        idField.setText(student.getId());
        nameField.setText(student.getName());
        englishField.setText(Double.toString(student.getEnglish()));
        mathField.setText(Double.toString(student.getAdvancedMath()));
        cField.setText(Double.toString(student.getCLanguage()));
        javaField.setText(Double.toString(student.getJavaProgramming()));
    }

    private void clearForm() {
        idField.setText("");
        nameField.setText("");
        englishField.setText("");
        mathField.setText("");
        cField.setText("");
        javaField.setText("");
        resultLabel.setText("请选择学生后点击计算按钮");
        table.clearSelection();
    }

    private void showAlertStatistics() {
        List<Student> alertStudents = manager.getAlertStudents();
        if (alertStudents.isEmpty()) {
            showMessage("所有学生成绩均达标");
            return;
        }
        StringBuilder sb = new StringBuilder();
        for (Student student : alertStudents) {
            sb.append(student.getId()).append(" ").append(student.getName())
                    .append(": ").append(student.getAlertCourses()).append(System.lineSeparator());
        }
        showMessage(sb.toString());
    }

    private void showMessage(String message) {
        JOptionPane.showMessageDialog(this, message);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new StudentScoreManagementApp().setVisible(true));
    }
}
