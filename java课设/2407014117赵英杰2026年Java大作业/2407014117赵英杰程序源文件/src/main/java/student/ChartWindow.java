package student;

import java.awt.BorderLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.List;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;

public class ChartWindow extends JFrame implements StudentChangeListener {
    private static final String COURSE_AVERAGE = "各科平均分柱状图";
    private static final String SCORE_LEVEL = "成绩区间分布饼图";
    private static final String TOTAL_RANKING = "学生总分排名柱状图";

    private final StudentManager manager;
    private final JComboBox<String> chartTypeBox = new JComboBox<>(
            new String[] { COURSE_AVERAGE, SCORE_LEVEL, TOTAL_RANKING });
    private final JPanel chartHolder = new JPanel(new BorderLayout());

    public ChartWindow(StudentManager manager) {
        super("成绩可视化图表");
        this.manager = manager;
        setSize(820, 520);
        setLocationByPlatform(true);
        setLayout(new BorderLayout(8, 8));
        add(chartTypeBox, BorderLayout.NORTH);
        add(chartHolder, BorderLayout.CENTER);
        chartTypeBox.addActionListener(event -> refresh());
        manager.addChangeListener(this);
        refresh();
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent event) {
                manager.removeChangeListener(ChartWindow.this);
            }
        });
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
    }

    @Override
    public void studentsChanged() {
        SwingUtilities.invokeLater(this::refresh);
    }

    private void refresh() {
        List<Student> students = manager.getAll();
        chartHolder.removeAll();
        if (students.isEmpty()) {
            chartHolder.add(new JLabel("暂无学生数据", SwingConstants.CENTER), BorderLayout.CENTER);
        } else {
            chartHolder.add(new ChartPanel(createStyledChart(students)), BorderLayout.CENTER);
        }
        chartHolder.revalidate();
        chartHolder.repaint();
    }

    private JFreeChart createStyledChart(List<Student> students) {
        JFreeChart chart = createChart(students);
        ChartFontSupport.applyChineseFont(chart);
        return chart;
    }

    private JFreeChart createChart(List<Student> students) {
        String type = (String) chartTypeBox.getSelectedItem();
        if (SCORE_LEVEL.equals(type)) {
            return ChartFactory.createPieChart("成绩区间分布", createScoreLevelDataset(students), true, true, false);
        }
        if (TOTAL_RANKING.equals(type)) {
            return ChartFactory.createBarChart("学生总分排名", "学生", "总分", createTotalRankingDataset());
        }
        return ChartFactory.createBarChart("各科平均分", "科目", "平均分", createCourseAverageDataset(students));
    }

    private DefaultCategoryDataset createCourseAverageDataset(List<Student> students) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        double[] sums = new double[Student.COURSE_NAMES.length];
        for (Student student : students) {
            double[] scores = student.getScores();
            for (int i = 0; i < scores.length; i++) {
                sums[i] += scores[i];
            }
        }
        for (int i = 0; i < Student.COURSE_NAMES.length; i++) {
            dataset.addValue(sums[i] / students.size(), "平均分", Student.COURSE_NAMES[i]);
        }
        return dataset;
    }

    private DefaultPieDataset<String> createScoreLevelDataset(List<Student> students) {
        int excellent = 0;
        int good = 0;
        int pass = 0;
        int fail = 0;
        for (Student student : students) {
            double average = student.getAverageScore();
            if (average >= 90) {
                excellent++;
            } else if (average >= 80) {
                good++;
            } else if (average >= 60) {
                pass++;
            } else {
                fail++;
            }
        }
        DefaultPieDataset<String> dataset = new DefaultPieDataset<>();
        dataset.setValue("优秀(>=90)", excellent);
        dataset.setValue("良好(80-89)", good);
        dataset.setValue("及格(60-79)", pass);
        dataset.setValue("不及格(<60)", fail);
        return dataset;
    }

    private DefaultCategoryDataset createTotalRankingDataset() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (Student student : manager.getRankedStudents()) {
            dataset.addValue(student.getTotalScore(), "总分", student.getName() + "(" + student.getId() + ")");
        }
        return dataset;
    }
}
