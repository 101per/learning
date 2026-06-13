package student;

import java.awt.BorderLayout;
import java.awt.Frame;
import java.awt.GridLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class StudentEditorDialog extends JDialog {
    private final StudentManager manager;
    private final Student originalStudent;
    private final JTextField idField = new JTextField();
    private final JTextField nameField = new JTextField();
    private final JTextField englishField = new JTextField();
    private final JTextField mathField = new JTextField();
    private final JTextField cField = new JTextField();
    private final JTextField javaField = new JTextField();

    public StudentEditorDialog(Frame owner, StudentManager manager, Student student) {
        super(owner, student == null ? "新增学生" : "学生详情/编辑", true);
        this.manager = manager;
        this.originalStudent = student;
        setSize(420, 260);
        setLocationRelativeTo(owner);
        setLayout(new BorderLayout(8, 8));
        add(buildFormPanel(), BorderLayout.CENTER);
        add(buildButtonPanel(), BorderLayout.SOUTH);
        if (student != null) {
            fillForm(student);
            idField.setEditable(false);
        }
    }

    private JPanel buildFormPanel() {
        JPanel panel = new JPanel(new GridLayout(6, 2, 8, 8));
        panel.add(new JLabel("学号"));
        panel.add(idField);
        panel.add(new JLabel("姓名"));
        panel.add(nameField);
        panel.add(new JLabel("英语"));
        panel.add(englishField);
        panel.add(new JLabel("高数"));
        panel.add(mathField);
        panel.add(new JLabel("C语言"));
        panel.add(cField);
        panel.add(new JLabel("Java程序设计"));
        panel.add(javaField);
        return panel;
    }

    private JPanel buildButtonPanel() {
        JPanel panel = new JPanel();
        JButton saveButton = new JButton("保存");
        JButton cancelButton = new JButton("取消");
        saveButton.addActionListener(event -> saveStudent());
        cancelButton.addActionListener(event -> dispose());
        panel.add(saveButton);
        panel.add(cancelButton);
        return panel;
    }

    private void fillForm(Student student) {
        idField.setText(student.getId());
        nameField.setText(student.getName());
        englishField.setText(Double.toString(student.getEnglish()));
        mathField.setText(Double.toString(student.getAdvancedMath()));
        cField.setText(Double.toString(student.getCLanguage()));
        javaField.setText(Double.toString(student.getJavaProgramming()));
    }

    private void saveStudent() {
        Student student = readForm();
        if (student == null) {
            return;
        }
        boolean success = originalStudent == null
                ? manager.addStudent(student)
                : manager.updateStudent(student);
        if (!success) {
            showMessage(originalStudent == null ? "学号已存在，不能重复新增" : "未找到该学生，无法修改");
            return;
        }
        dispose();
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

    private void showMessage(String message) {
        JOptionPane.showMessageDialog(this, message);
    }
}
